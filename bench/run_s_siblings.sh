#!/usr/bin/env bash
# Plan S: what shares a physical core with llama.cpp's main thread.
#
#     bash bench/run_s_siblings.sh     # ~0.5 h, one RTX 3090, exclusive
#
# Pre-registered in v4_audit_2026_08_25/PROSPECTIVE_PLAN_SIBLINGS.md, committed
# before this ran. Read that first: it fixes the estimator, the bands, what each
# outcome licenses, and why the masks are the ones below and not the obvious ones.
#
# The three conditions, all inside ONE invocation because A16's step is BETWEEN
# invocations and an arm measured in its own invocation is confounded with the
# variable this plan exists to isolate:
#
#   distinct   cpu 0,2,4,6,8,10,12,14   one thread from each of the eight P cores
#   packed     cpu 0,1,2,3,4,5,8,9      both threads of four of them
#   packed, and OMP_WAIT_POLICY=passive so the idle OpenMP workers sleep at
#                                    once instead of spinning first
#
# Eight threads every time, `-t`/`-tb` explicit because a cpuset otherwise moves
# the thread count with it. The third condition is an entry in the runner's
# ARM_ENV table, per ARM: an environment variable exported here would reach all
# three arms. It was `--poll 0` until 2026-11-13, and the server here is an
# OpenMP build, where ggml never reads the poll level.
#
# WHY NOT 0,1,2,3,4,5,6,7 for the packed set, which is the obvious choice. The
# bench host's processors 8 to 11 run about a twentieth faster than the other
# twelve P-core threads -- Turbo Boost Max favours their two cores, cpufreq says
# 5800 MHz against 5500, and bench/cpu_siblings.py measured the same ratio in
# throughput. `0,2,4,6,8,10,12,14` puts two of its eight threads on favoured
# processors and `0,1,2,3,4,5,6,7` puts none there, so those two masks differ in
# hyperthread sharing AND in clock, and the expected difference from the clock
# alone is larger than the band the plan reserves for deciding the mechanism is
# absent. `0,1,2,3,4,5,8,9` is four cores with both threads of each and the same
# two favoured processors as the distinct set. One thing differs: how many
# physical cores the eight threads span.
#
# What this refuses, and every one of them is a condition the plan's reading
# depends on rather than a tidiness check:
#
#   1. a runner that is not bench/retest_runner.py at HEAD (outside a git
#      checkout there is no HEAD, and the driver says it cannot check)
#   2. a server binary that cannot start, with the missing libraries named. The
#      published binary needs libcudart.so.12 and libcublas.so.12, this host has
#      no CUDA toolkit, and its RUNPATH names only its own build directory, so
#      the library path is something the caller has to supply and this records it.
#   3. a set that is not eight processors, or that contains an efficiency core
#   4. unequal exposure to the favoured processors, which is the confound above
#   5. a distinct set where any two processors share a physical core, or a packed
#      set where any processor does not -- read from /sys at run time, because
#      the masks are a claim about this host and a host can be reinstalled
#   6. the two sets spanning the same number of physical cores, which would make
#      them the same condition written twice
#   7. telemetry that died at startup, or a run with no samples
#   8. fewer than thirty six arm-runs, or an arm whose kernel mask is not the one
#      it asked for, or whose server was not given exactly its arm's environment
#   9. a server that does not load libgomp, because the third arm's lever is
#      libgomp's and a build without OpenMP would take it without effect
#  10. a measurement already running here: a lock another driver holds, or a
#      llama-server or llama-bench process, which is what bench/host_guard.py
#      looks for. The lock is created atomically and removed only by its owner.
#
# Note what is NOT refused: the two sets OVERLAP, sharing processors 0, 2, 4 and
# 8. Run Y's driver refuses that, correctly for run Y, where the two sets were
# different kinds of core. Here the sets are drawn from the same eight cores on
# purpose and an overlap check would refuse the design.
set -euo pipefail

# `--check` runs every refusal above and stops before the telemetry and the
# runner, so the preconditions can be read on the bench host without touching the
# card. The checks are about the host's processors and the server binary, and
# finding out that one of them fails thirty seconds into the first arm-run costs
# the arm-run.
CHECK_ONLY=0
if [ "${1:-}" = "--check" ]; then
    CHECK_ONLY=1
    shift
fi
if [ $# -ne 0 ]; then
    echo "usage: bash bench/run_s_siblings.sh [--check]" >&2
    exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${BENCH_ROOT:-$HOME/bench}"

# Asked first, before anything is read or launched, because "is a measurement
# already running on this card" does not depend on any of it and because
# `--check` should be able to answer it too. The lock is CREATED further down,
# after `--check` would have returned: a read-only check must not announce a
# measurement that is not happening.
#
# Asked of bench/host_guard.py rather than of one path: the guard searches every
# lock a measurement here may hold AND the process table, so a layer A run, which
# runs no llama-server, and a server some other driver left running are both
# seen. One reader of the lock, the same one every verification suite uses.
GPU_LOCK="${BENCH_GPU_LOCK:-$HOME/.gpu-in-use.lock}"
BENCH_HERE="$HERE" python3 - <<'GUARD' || exit 1
import os
import sys

sys.path.insert(0, os.environ["BENCH_HERE"])
import host_guard                                            # noqa: E402

lock = host_guard.lock_held()
# The process half takes the override the guard's own `protect` takes, for a
# host where something matching the pattern is known not to be measuring; the
# lock half takes none.
running = ([] if os.environ.get("BENCH_ALLOW_CONTENDED") == "1"
           else host_guard.measuring_processes())
if lock or running:
    print("FAIL: a measurement is already running here:", file=sys.stderr)
    if lock:
        print(f"      {lock} exists:", file=sys.stderr)
        try:
            for line in lock.read_text(encoding="utf-8").splitlines()[:3]:
                print(f"        {line}", file=sys.stderr)
        except OSError:
            pass
    for r in running[:3]:
        print(f"      running: {r}", file=sys.stderr)
    print("      Two measurements on one card measure each other.", file=sys.stderr)
    sys.exit(1)
GUARD

RUNNER="${BENCH_RUNNER:-}"
for cand in "$RUNNER" "$HERE/retest_runner.py" "$BENCH/retest_runner.py"; do
    [ -n "$cand" ] && [ -f "$cand" ] && { RUNNER="$cand"; break; }
done
TELE_SH="${BENCH_TELEMETRY:-}"
for cand in "$TELE_SH" "$HERE/gpu_telemetry.sh" "$BENCH/gpu_telemetry.sh"; do
    [ -n "$cand" ] && [ -f "$cand" ] && { TELE_SH="$cand"; break; }
done
[ -f "$RUNNER" ] || { echo "no retest_runner.py: set BENCH_RUNNER" >&2; exit 1; }
[ -f "$TELE_SH" ] || { echo "no gpu_telemetry.sh: set BENCH_TELEMETRY" >&2; exit 1; }

if command -v git >/dev/null && git -C "$HERE/.." rev-parse --git-dir >/dev/null 2>&1; then
    want=$(git -C "$HERE/.." show HEAD:bench/retest_runner.py | sha256sum | cut -d" " -f1)
    got=$(sha256sum < "$RUNNER" | cut -d" " -f1)
    [ "$want" = "$got" ] || {
        echo "FAIL: $RUNNER is not bench/retest_runner.py at HEAD." >&2
        echo "      ${got:0:16} against ${want:0:16}. A measurement taken with a" >&2
        echo "      runner the repository does not hold cannot be checked." >&2
        exit 1
    }
    echo "runner sha ${got:0:16}, which is HEAD's"
else
    echo "NOTE: not a git checkout, so $RUNNER cannot be compared with HEAD;" >&2
    echo "      the run manifest records its sha256 and that is all." >&2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
TELE_SCHEMA="${BENCH_TELEMETRY_SCHEMA:-full}"
TELE_INTERVAL="${BENCH_TELEMETRY_INTERVAL:-1}"
echo "runner    $RUNNER"
echo "telemetry $TELE_SH $TELE_SCHEMA $TELE_INTERVAL S"

export MODEL_TARGET="${MODEL_TARGET:-$HOME/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
export MODEL_DRAFT="${MODEL_DRAFT:-$HOME/models/Qwen3.5-0.8B-Q4_K_M.gguf}"
export MODEL_DFLASH="${MODEL_DFLASH:-$HOME/models/qwen36-dflash-master.gguf}"
export MODEL_MTP="${MODEL_MTP:-$HOME/models/qwen36-mtp-q8_0.gguf}"
export LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$BENCH/llama-retest/build/bin/llama-server}"
[ -x "$LLAMA_SERVER_BIN" ] || { echo "not executable: $LLAMA_SERVER_BIN" >&2; exit 1; }

# The binary has to be able to START, and the usual reason it cannot is the one
# prerequisite 6 of the plan is about: it wants libcudart.so.12 and
# libcublas.so.12, this host has no CUDA toolkit, and its RUNPATH names only its
# own build directory. Checked here rather than discovered thirty seconds into
# the first arm-run, and the diagnosis names the libraries.
if ! "$LLAMA_SERVER_BIN" --help >/dev/null 2>&1; then
    echo "FAIL: $LLAMA_SERVER_BIN cannot start." >&2
    missing=$(ldd "$LLAMA_SERVER_BIN" 2>/dev/null | awk '/not found/{print "        "$1}')
    if [ -n "$missing" ]; then
        echo "      shared libraries it cannot resolve:" >&2
        echo "$missing" >&2
        echo "      There is no CUDA toolkit on this host; the only copies of the" >&2
        echo "      CUDA runtime are pip wheels inside unrelated virtual" >&2
        echo "      environments. Set LD_LIBRARY_PATH to the one you mean and it" >&2
        echo "      will be recorded per arm-run under \`server_mapped\`." >&2
    fi
    exit 1
fi
echo "server    ${LLAMA_SERVER_BIN}  (starts; LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset})"

# The third arm's lever is libgomp's. ggml reads `--poll` only when it is built
# WITHOUT OpenMP, and with it the workers are libgomp's and wait as
# OMP_WAIT_POLICY says. Run Y's server log printed `OPENMP = 1` for this build;
# this checks the binary that is about to run rather than that memory. ldd lists
# what the dynamic linker would load, including libggml-cpu's own NEEDED.
LIBS=$(ldd "$LLAMA_SERVER_BIN" 2>/dev/null || true)
if ! printf '%s\n' "$LIBS" | grep 'libgomp\.so' > /dev/null; then
    echo "FAIL: $LLAMA_SERVER_BIN does not load libgomp, so it is not an OpenMP" >&2
    echo "      build, and OMP_WAIT_POLICY would change nothing in the third arm." >&2
    echo "      Plan S was written for the OpenMP build its runs record." >&2
    exit 1
fi
echo "openmp    libgomp is among the server's libraries"

export BENCH_EXPECT_COMMIT="${BENCH_EXPECT_COMMIT:-3737e41370da1830a44c663f9929a0f27591ffa6}"
export BENCH_PIN_SUFFIX="-packed"
export BENCH_PIN_CPUS="${BENCH_PIN_CPUS:-0,2,4,6,8,10,12,14}"
export BENCH_PIN_ALT_CPUS="${BENCH_PIN_ALT_CPUS:-0,1,2,3,4,5,8,9}"
export BENCH_THREADS="${BENCH_THREADS:-8}"
export BENCH_ARMS="spec-dflash-n2,spec-dflash-n2-packed,spec-dflash-n2-passive-packed"
export BENCH_REPEATS=12
# `latin`, not `mirrored`. This driver was written from run Y's, which has FOUR
# arms, and reversing an even-length order gives each arm two positions. Plan S
# has three: reversing leaves the middle one fixed, and the first attempt at this
# run was stopped after fifteen of its thirty six arm-runs with
# `spec-dflash-n2-packed` at position 2 in every block, read from the arm-runs
# rather than inferred. `latin` over
# twelve repeats puts each of the three arms in each of the three positions four
# times, and the runner validates that before it spends any GPU time.
export BENCH_ORDER=latin
export BENCH_MAX_TOKENS=300
export BENCH_THINK=on
export BENCH_CTX=8192
export BENCH_FIT=on
export BENCH_FIT_TARGET=3072
export BENCH_CONCURRENCY=1
export BENCH_FLAVOR=master
# The threadpool flags reach nothing in an OpenMP build, so none is passed and
# rederive_run_s.py refuses a run whose argv carries one. The OpenMP runtime's
# own variables are unset, every one of them: the third arm's comes from the
# runner's ARM_ENV, and one inherited from this shell would reach all three arms.
unset BENCH_POLL || true
while IFS= read -r v; do unset "$v"; done < <(compgen -e | grep -E '^(OMP_|GOMP_|KMP_)' || true)
unset BENCH_CPU_STRICT || true
unset BENCH_IGNORE_EOS || true
unset BENCH_HARDCAP_SUFFIX || true

# The masks are a claim about THIS host: which processors are performance cores,
# which share a physical core, and which two cores the turbo favours. Read and
# checked here, so a host that is not the one the plan was written for stops the
# run instead of producing a contrast that means something else.
BENCH_HERE="$HERE" python3 - <<'PY'
import os
import pathlib
import sys

# BENCH_HERE is this script's own directory, exported below. A heredoc has no
# `__file__`, and guessing "bench" relative to the working directory would import
# whatever happened to be there.
sys.path.insert(0, os.environ["BENCH_HERE"])
from cpu_siblings import validate_masks                      # noqa: E402


def parse(spec):
    out = set()
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            out |= set(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def read(cpu, leaf):
    p = pathlib.Path(f"/sys/devices/system/cpu/cpu{cpu}/{leaf}")
    if not p.exists():
        sys.exit(f"cpu{cpu} has no {leaf}, so this host cannot be asked about "
                 f"its own processors and the masks are a claim nothing can check")
    return p.read_text().strip()


# The WHOLE host, because the clock tiers are a property of the part and not of
# the selection. The logic itself lives in bench/cpu_siblings.py, where the suite
# can reach it: a check that exists only in this script is a check CI never runs,
# and this host has three clock tiers while a runner has none.
host = sorted(int(q.name[3:]) for q in
              pathlib.Path("/sys/devices/system/cpu").glob("cpu[0-9]*"))
ceiling = {c: int(read(c, "cpufreq/cpuinfo_max_freq")) // 1000 for c in host}
sibling = {c: parse(read(c, "topology/thread_siblings_list")) for c in host}

distinct = parse(os.environ["BENCH_PIN_CPUS"])
packed = parse(os.environ["BENCH_PIN_ALT_CPUS"])
threads = int(os.environ["BENCH_THREADS"])

bad, info = validate_masks(distinct, packed, threads, ceiling, sibling)
print(f"  host tiers (MHz): {info['tiers']}")
if bad:
    sys.exit("\n".join(bad))
print(f"  favoured {info['favoured']} at {info['tiers'][-1]} MHz; "
      f"{info['favoured_in_distinct']} in the distinct set and "
      f"{info['favoured_in_packed']} in the packed one")
print(f"  distinct {sorted(distinct)} spans {info['distinct_cores']} physical cores")
print(f"  packed   {sorted(packed)} spans {info['packed_cores']}")
PY

if [ "$CHECK_ONLY" -eq 1 ]; then
    echo "check only: every precondition passed, nothing was launched"
    exit 0
fi

# The lock, which is the thing that tells any verification work on this host to
# stay off the processors for the next half hour. `bench/host_guard.py` has
# searched for it since it was written and no driver here has ever created it, so
# for every measurement in this repository that half of the guard had nothing to
# detect. Taken at the machine level, `~/.gpu-in-use.lock`, which is the first
# path the guard looks at, and removed on every exit path by the trap.
#
# Created with noclobber, so between the check above and this line a second
# driver cannot also take it: `>` would overwrite another's lock, and the first
# of the two to finish would then delete the lock the other still needs. The
# EXIT trap removes it only if this process created it, and is set BEFORE
# anything below can fail, so no exit path leaves the host looking busy.
LOCK_OURS=0
TELE_PID=""
RUNNER_PID=""
# shellcheck disable=SC2317  # called from a trap, which shellcheck cannot see
cleanup() {
    [ -n "$TELE_PID" ] && kill "$TELE_PID" 2>/dev/null
    [ "$LOCK_OURS" = 1 ] && rm -f "$GPU_LOCK"
    return 0
}
# A signal stops the RUNNER first, which stops its server: the runner treats
# TERM as an interrupt and unwinds through the `finally` that tears the server
# down. Without this the lock and the telemetry went at once while the runner
# went on measuring on a host that no longer said it was busy.
# shellcheck disable=SC2317  # called from a trap, which shellcheck cannot see
on_signal() {
    trap - INT TERM
    if [ -n "$RUNNER_PID" ]; then
        kill -TERM "$RUNNER_PID" 2>/dev/null
        wait "$RUNNER_PID" 2>/dev/null
    fi
    exit 143
}
trap cleanup EXIT
trap on_signal INT TERM
if ! ( set -C; printf 'plan S siblings, pid %s, started %s\nhost %s\narms %s\n' \
        "$$" "$(date -Is)" "$(hostname)" "$BENCH_ARMS" > "$GPU_LOCK" ) 2>/dev/null; then
    echo "FAIL: $GPU_LOCK appeared after the check, so another measurement" >&2
    echo "      started here in the meantime." >&2
    exit 1
fi
LOCK_OURS=1
echo "lock      $GPU_LOCK"

TELE_CSV="$BENCH/gpu_telemetry_S_$STAMP.csv"
BENCH_TELEMETRY_OUT="$TELE_CSV" bash "$TELE_SH" "$TELE_SCHEMA" "$TELE_INTERVAL" "S" &
TELE_PID=$!
sleep 2
kill -0 "$TELE_PID" 2>/dev/null || { echo "FAIL: telemetry died at startup" >&2; exit 1; }

out="$BENCH/matrix_S_siblings_$STAMP"
echo "=== plan S -> $(basename "$out")  $(date -Is) ==="
rc=0
# In the background and waited for, so a signal reaches `on_signal` at once
# rather than after the half hour a foreground child would hold it for.
BENCH_OUT="$out" python3 "$RUNNER" &
RUNNER_PID=$!
if wait "$RUNNER_PID"; then :; else
    echo "!!! the runner exited non-zero" >&2
    rc=1
fi
RUNNER_PID=""
echo "=== S done $(date -Is) ==="

[ -f "$out/RUN_COMPLETE.json" ] || { echo "FAIL: no RUN_COMPLETE.json" >&2; rc=1; }
n=$(find "$out" -maxdepth 1 -name '*__rep*.json' -printf . | wc -c)
echo "arm-runs:$n expected:36"
[ "$n" -eq 36 ] || { echo "FAIL: $n arm-runs, expected 36" >&2; rc=1; }

# The run is checked by the function that will READ it: every condition the plan
# registers -- the masks the kernel granted, one server and one set of libraries,
# each arm's environment read back from /proc, libgomp mapped, the per-thread
# processor time present, the order, the token cap, the build, balanced
# positions. An inline check here held six of those and drifted from the
# reading's; one function cannot.
python3 "$HERE/../analysis/rederive_run_s.py" --conditions "$out" || rc=1

tele_rows=0
[ -s "$TELE_CSV" ] && tele_rows=$(( $(wc -l < "$TELE_CSV") - 1 ))
echo "telemetry rows:$tele_rows"
[ "$tele_rows" -ge 1 ] || { echo "FAIL: $TELE_CSV holds no samples" >&2; rc=1; }
exit "$rc"
