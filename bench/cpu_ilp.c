/* A throughput-bound integer kernel, for deciding which logical processors
 * share a physical core.
 *
 *     cc -O2 -Wall -Wextra -o cpu_ilp bench/cpu_ilp.c
 *     cpu_ilp SECONDS            # prints blocks completed, one number
 *
 * Why the shape of the loop is the whole instrument
 * -------------------------------------------------
 * Two hyperthreads of one physical core share its execution ports. They do not
 * share them equally with every workload, and the direction of the error is not
 * symmetric, so the kernel cannot be chosen for convenience.
 *
 * A LATENCY-bound loop -- one long dependent chain -- leaves most ports idle
 * while it waits, so a second thread fills the gaps and the pair runs at nearly
 * twice the single rate. Measured with that kernel, a pair of siblings looks
 * exactly like a pair of distinct cores, and the instrument reports "distinct"
 * for everything. That is the failure that matters, because it is the silent one.
 *
 * So this is THROUGHPUT bound: eight independent LCG chains, interleaved. Each
 * step is one 64-bit multiply and one add, the chains do not depend on each
 * other, and eight of them at a multiply latency of about three cycles is enough
 * to keep the multiplier issuing every cycle. Two threads then contend for one
 * port and the pair runs at well under twice the single rate.
 *
 * No memory traffic at all: eight accumulators live in registers, there is no
 * array, and nothing is read or written in the loop. A memory-bound kernel would
 * measure the shared L2 or the shared ring instead, which is a different
 * question with a different answer -- processors can share a cache without
 * sharing a core.
 *
 * The accumulators are xored into the return value and printed. Without that,
 * -O2 deletes the loop and the program measures how fast nothing runs.
 *
 * Affinity is NOT set here. The caller pins the process, because the caller is
 * the one that knows which processor this run is about, and a kernel that pinned
 * itself could not be asked about a pair.
 */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHAINS 8
#define INNER 4096

static const unsigned long long MUL = 6364136223846793005ULL;
static const unsigned long long ADD = 1442695040888963407ULL;

static double monotonic(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t) != 0) {
        perror("clock_gettime");
        exit(1);
    }
    return (double) t.tv_sec + (double) t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: cpu_ilp SECONDS\n");
        return 2;
    }
    char *end = NULL;
    double seconds = strtod(argv[1], &end);
    if (end == argv[1] || *end != '\0' || seconds <= 0.0 || seconds > 600.0) {
        fprintf(stderr, "SECONDS must be a number in (0, 600]\n");
        return 2;
    }

    unsigned long long a[CHAINS];
    for (int i = 0; i < CHAINS; i++) {
        a[i] = 0x9E3779B97F4A7C15ULL ^ (unsigned long long) i;
    }

    /* One block is INNER iterations of CHAINS multiply-adds. The clock is read
     * once per block, not once per iteration: at INNER 4096 that is one read per
     * roughly thirty thousand multiplies, so the measurement does not pay for
     * its own timing. */
    unsigned long long blocks = 0;
    double t0 = monotonic();
    double elapsed;
    do {
        for (int k = 0; k < INNER; k++) {
            a[0] = a[0] * MUL + ADD;
            a[1] = a[1] * MUL + ADD;
            a[2] = a[2] * MUL + ADD;
            a[3] = a[3] * MUL + ADD;
            a[4] = a[4] * MUL + ADD;
            a[5] = a[5] * MUL + ADD;
            a[6] = a[6] * MUL + ADD;
            a[7] = a[7] * MUL + ADD;
        }
        blocks++;
        elapsed = monotonic() - t0;
    } while (elapsed < seconds);

    unsigned long long sink = 0;
    for (int i = 0; i < CHAINS; i++) {
        sink ^= a[i];
    }

    /* blocks, the elapsed seconds it actually took, and the sink that stops the
     * optimiser from removing the work. The caller divides rather than trusting
     * the requested window: a process that was descheduled took longer than it
     * asked for, and dividing by the request would credit it with the gap. */
    printf("%llu %.6f %llu\n", blocks, elapsed, sink);
    return 0;
}
