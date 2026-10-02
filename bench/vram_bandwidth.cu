// Achieved GDDR6X bandwidth, as a time series, for plan Z's layer A.
//
//     nvcc -O3 -arch=sm_86 -o vram_bandwidth bench/vram_bandwidth.cu
//     ./vram_bandwidth --seconds 600 --window 10            # ascending phase
//     ./vram_bandwidth --seconds 1800 --window 10 --idle 50  # descending phase
//
// Why a time series and not one number. Layer A reads a SLOPE of bandwidth
// against temperature, and a slope needs the scatter it is fitted through: one
// aggregate over a whole run cannot say whether bandwidth moved, only what it
// averaged while the card heated. Each window is the same work, timed on its own,
// printed as it finishes, so the output joins against a temperature trace from
// `bench/vram_temp.py` on wall clock.
//
// Why `--idle`. The ascending phase runs the load continuously from a cold card
// and so traverses temperature and elapsed time together: anything read from it
// alone cannot separate "hotter" from "running longer". With an idle gap after
// each window the card cools between windows, so the descending phase crosses the
// same temperatures in the other direction at different elapsed times. If
// bandwidth is a function of thermal state the two phases trace one curve; if it
// is a function of time since the load began, they do not. That is the control,
// and it is why this flag exists rather than a second program.
//
// Each window is preceded by an unmeasured warm-up pass, so a window from a
// cooler card is not also a window whose first pass is paying for a cold cache.
//
// The load is a pure stream: two buffers, one copied into the other and back,
// nothing else. It is deliberately not a model: what layer A measures is the
// memory path, and a workload that also loads the SMs would confound the thermal
// state of one with the other.
//
// PROVENANCE, and a limit on what this file may be used to claim. ERRATA A16's
// second addendum publishes a rise to ninety degrees under a load holding
// 828 GB/s. That reading was taken on 2026-10-01 with this file's predecessor,
// which had the same `stream` kernel byte for byte, the same 256 threads, the same
// blocks-per-SM occupancy and the same four gibibyte buffers, and differed only in
// its measurement loop and its output: one aggregate at the end instead of a csv
// row per window. The load is therefore the same load.
//
// This file has NOT been compiled. The bench host has `nvcc` and went offline
// before the rewrite existed, and the box this was written on has no CUDA headers.
// So: the published reading rests on the predecessor, this file is the instrument
// for layer A, and nothing may be claimed from it until it has been built and run
// on the bench host and its first plateau reported. `tests/test_harness_invariants.py`
// holds the kernel and the geometry to what produced that reading, so an edit to
// the load fails a test rather than silently invalidating the provenance above.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cuda_runtime.h>

#define OK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) {                \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));   \
    return 1; } } while (0)

__global__ void stream(float4 *__restrict__ dst, const float4 *__restrict__ src,
                       size_t n)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) dst[i] = src[i];
}

// `GPU-f71a8f68-b0bb-...`, the spelling `nvidia-smi --query-gpu=uuid` writes.
//
// Not decoration. This repository holds measurements from at least three
// physically distinct RTX 3090s and NOTHING in it can tell them apart: the only
// card identifier any committed run records is `name`, which is
// "NVIDIA GeForce RTX 3090" on every one of them. A bandwidth figure is a property
// of one card's memory and its cooling, so a figure that cannot name its card is
// a figure that can be attributed to the wrong one for ever.
static void uuid_str(const cudaUUID_t *u, char *out, size_t n)
{
    const unsigned char *b = (const unsigned char *)u->bytes;
    snprintf(out, n,
             "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-"
             "%02x%02x%02x%02x%02x%02x",
             b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
             b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
}


static double monotonic(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

// Wall clock in the same spelling `bench/gpu_telemetry.sh` writes, so a window
// joins against a temperature trace without a mapping between two clocks. Run Y's
// trace could not be attributed to an arm-run for exactly the want of this.
static void wall_iso(char *out, size_t n)
{
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    struct tm tm;
    localtime_r(&t.tv_sec, &tm);
    size_t k = strftime(out, n, "%Y/%m/%d %H:%M:%S", &tm);
    snprintf(out + k, n - k, ".%03ld", t.tv_nsec / 1000000);
}

int main(int argc, char **argv)
{
    double seconds = 300.0, window = 10.0, idle = 0.0, gib = 4.0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *v = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (!v) { fprintf(stderr, "%s wants a value\n", a); return 2; }
        if      (!strcmp(a, "--seconds")) seconds = atof(v);
        else if (!strcmp(a, "--window"))  window  = atof(v);
        else if (!strcmp(a, "--idle"))    idle    = atof(v);
        else if (!strcmp(a, "--gib"))     gib     = atof(v);
        else { fprintf(stderr, "unknown option %s; the options are --seconds, "
                               "--window, --idle, --gib\n", a); return 2; }
        i++;
    }
    // Refusals, not clamps: a run configured wrong should stop, because a window
    // of zero seconds reports a bandwidth of infinity and a reader cannot tell
    // that from a measurement.
    if (seconds <= 0.0 || window <= 0.0 || idle < 0.0 || gib <= 0.0) {
        fprintf(stderr, "--seconds, --window and --gib must be positive and "
                        "--idle may not be negative\n");
        return 2;
    }
    if (window > seconds) {
        fprintf(stderr, "--window %g is longer than --seconds %g, so not one "
                        "window would finish\n", window, seconds);
        return 2;
    }

    size_t bytes = (size_t)(gib * (1ull << 30));
    size_t n = bytes / sizeof(float4);
    float4 *a = NULL, *b = NULL;
    OK(cudaMalloc(&a, bytes));
    OK(cudaMalloc(&b, bytes));
    OK(cudaMemset(a, 1, bytes));
    int per_sm = 0, threads = 256;
    OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, (void *)stream,
                                                     threads, 0));
    cudaDeviceProp p;
    OK(cudaGetDeviceProperties(&p, 0));
    int blocks = per_sm * p.multiProcessorCount;

    // Header to stderr, so stdout is a csv a reader can join on without stripping
    // anything. The repository has one defect from a header that was published as
    // a data row and one from a sample that was published as a header.
    char uuid[64], bus[32];
    uuid_str(&p.uuid, uuid, sizeof uuid);
    if (cudaDeviceGetPCIBusId(bus, (int)sizeof bus, 0) != cudaSuccess)
        snprintf(bus, sizeof bus, "unknown");
    fprintf(stderr, "%s %s at %s, %d SMs, %d blocks x %d threads, "
                    "%.2f GiB per buffer\n",
            p.name, uuid, bus, p.multiProcessorCount, blocks, threads,
            bytes / 1073741824.0);
    fprintf(stderr, "seconds %.0f, window %.1f, idle %.1f\n",
            seconds, window, idle);
    // The uuid is a column and not only a banner, so every ROW names its card. A
    // banner lives in a log beside the csv and a log can be lost: this repository
    // has one complete trace it cannot attribute to an arm-run, and the lesson is
    // that identity belongs in the data and not next to it. Twenty-four bytes a
    // row against a figure that could otherwise be read off the wrong card.
    printf("wall_iso,elapsed_s,window_s,passes,gbytes_per_s,gpu_uuid\n");
    fflush(stdout);

    double t0 = monotonic();
    char stamp[64];
    while (monotonic() - t0 < seconds) {
        // unmeasured warm-up, so the measured window is not paying for the first
        // touch of a buffer the idle gap let fall out of whatever caches it
        stream<<<blocks, threads>>>(b, a, n);
        stream<<<blocks, threads>>>(a, b, n);
        OK(cudaDeviceSynchronize());

        double w0 = monotonic();
        unsigned long long passes = 0;
        while (monotonic() - w0 < window) {
            for (int k = 0; k < 10; k++) {
                stream<<<blocks, threads>>>(b, a, n);
                stream<<<blocks, threads>>>(a, b, n);
            }
            OK(cudaDeviceSynchronize());
            passes += 20;
        }
        double w = monotonic() - w0;
        wall_iso(stamp, sizeof stamp);
        // read plus write on every pass
        printf("%s,%.3f,%.3f,%llu,%.1f,%s\n", stamp, w0 - t0, w, passes,
               passes * 2.0 * bytes / w / 1e9, uuid);
        fflush(stdout);
        if (idle > 0.0) {
            double i0 = monotonic();
            while (monotonic() - i0 < idle) {
                struct timespec s = {0, 50 * 1000 * 1000};
                nanosleep(&s, NULL);
            }
        }
    }
    OK(cudaFree(a));
    OK(cudaFree(b));
    return 0;
}
