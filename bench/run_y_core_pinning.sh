#!/usr/bin/env bash
# Run Y: what a quarter less processor clock costs the arm ERRATA A16 is about.
#
#     bash bench/run_y_core_pinning.sh     # ~0.7 h, one RTX 3090, exclusive
#
# Pre-registered in v4_audit_2026_08_25/PROSPECTIVE_PLAN_Y_CORE_PINNING.md, which
# was committed before this ran. Read that first: it fixes the estimator, the
# thresholds and what each outcome licenses, and it says why twelve blocks and
# not the six the run X plan named.
#
# The bench host is an i9-13900K. Sixteen of its thirty-two logical processors are
# efficiency cores at 4300 MHz; the rest are performance cores at 5500 with four
# at 5800. Nothing in this repository pinned any process to any processor until
# the commit that carries this script, and no run records which kind of core
# anything ran on -- so core placement has been an uncontrolled variable about a
# quarter of the clock wide, on a scheduler's timescale, producing byte-identical
# output and invisible to `nvidia-smi`. That is A16's signature.
#
# This does not test whether placement caused A16's steps; those runs have no
# placement column and cannot be revisited. It BOUNDS how much placement could
# ever have mattered, by forcing the whole server onto the slow cores.
#
# The two conditions:
#
#   fast   cpu 0,2,4,6,8,10,12,14   one thread of each of the eight P cores
#   slow   cpu 16..23               eight E cores, which have no second thread
#
# Eight threads and eight distinct physical cores either way, `-t`/`-tb` passed
# explicitly because a cpuset otherwise changes the thread count with it. BOTH
# sides are pinned: pinned against unpinned changes two things at once, and the
# runner refuses that configuration outright.
#
# `BENCH_ORDER=mirrored` puts the slow condition first in six of the twelve
# blocks and second in the other six, with each arm's two conditions adjacent in
# every block. That is the balance the run X plan argues for, and it is asserted
# in `tests/test_harness_invariants.py` rather than assumed here.
#
# The build is stock master, NOT the split-timer build run T4 used. The
# variability figures the pre-registration's power arithmetic rests on come from
# T4, so they are an estimate for this run rather than a measurement of it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${BENCH_ROOT:-$HOME/bench}"
# The REPO copy first, then $BENCH. The older launchers here search $BENCH
# first, and on this bench host that directory holds a runner 181 lines away
# from HEAD which matches none of the thirty-seven versions this repository has
# ever committed. The first attempt at this run used it and died on an unknown
# arm, which was luck: a stale copy that happened to accept the flags would have
# measured with code the repository does not have and recorded a
# `runner_sha256` resolving to nothing.
RUNNER="${BENCH_RUNNER:-}"
for cand in "$RUNNER" "$HERE/retest_runner.py" "$BENCH/retest_runner.py"; do
    [ -n "$cand" ] && [ -f "$cand" ] && { RUNNER="$cand"; break; }
done
TELE_SH="${BENCH_TELEMETRY:-}"
for cand in "$TELE_SH" "$HERE/gpu_telemetry.sh" "$BENCH/gpu_telemetry.sh"; do
    [ -n "$cand" ] && [ -f "$cand" ] && { TELE_SH="$cand"; break; }
done
[ -f "$RUNNER" ]  || { echo "no retest_runner.py: set BENCH_RUNNER" >&2; exit 1; }
# Order is not enough: BENCH_RUNNER can still point anywhere, and a working copy
# can be edited. The runner this run measures with has to be a version the
# repository holds, because `runner_sha256` in the manifest is resolved against
# exactly that set and a run whose runner is not there is a run nobody can check.
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
fi
[ -f "$TELE_SH" ] || { echo "no gpu_telemetry.sh: set BENCH_TELEMETRY" >&2; exit 1; }
STAMP="$(date +%Y%m%d_%H%M%S)"
TELE_SCHEMA="${BENCH_TELEMETRY_SCHEMA:-raw}"
TELE_INTERVAL="${BENCH_TELEMETRY_INTERVAL:-1}"
echo "runner    $RUNNER"
echo "telemetry $TELE_SH $TELE_SCHEMA $TELE_INTERVAL Y"

export MODEL_TARGET="${MODEL_TARGET:-$HOME/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
export MODEL_DRAFT="${MODEL_DRAFT:-$HOME/models/Qwen3.5-0.8B-Q4_K_M.gguf}"
export MODEL_DFLASH="${MODEL_DFLASH:-$HOME/models/qwen36-dflash-master.gguf}"
export MODEL_MTP="${MODEL_MTP:-$HOME/models/qwen36-mtp-q8_0.gguf}"
export LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$BENCH/llama-retest/build/bin/llama-server}"
[ -x "$LLAMA_SERVER_BIN" ] || { echo "not executable: $LLAMA_SERVER_BIN" >&2; exit 1; }
export BENCH_EXPECT_COMMIT="${BENCH_EXPECT_COMMIT:-3737e41370da1830a44c663f9929a0f27591ffa6}"

# The arm A16 is about, and the arm with the least host-side work per token, each
# in both conditions. `baseline` is here to say whether a processor-speed effect
# reaches plain decoding at all, not as the denominator of a ratio.
export BENCH_PIN_SUFFIX="-ecore"
export BENCH_PIN_CPUS="${BENCH_PIN_CPUS:-0,2,4,6,8,10,12,14}"
export BENCH_PIN_ALT_CPUS="${BENCH_PIN_ALT_CPUS:-16,17,18,19,20,21,22,23}"
export BENCH_THREADS="${BENCH_THREADS:-8}"
export BENCH_ARMS="spec-dflash-n2,spec-dflash-n2-ecore,baseline,baseline-ecore"
export BENCH_REPEATS=12
# `mirrored`, not `latin`: what has to balance is the ORDER WITHIN A PAIR, and a
# Latin square over four arms would separate a pair by two other arm-runs.
export BENCH_ORDER=mirrored
export BENCH_MAX_TOKENS=300
# run T4's shape, because T4 is where the variability this run was sized against
# was measured
export BENCH_THINK=on
export BENCH_CTX=8192
export BENCH_FIT=on
export BENCH_FIT_TARGET=3072
export BENCH_CONCURRENCY=1
export BENCH_FLAVOR=master
unset BENCH_IGNORE_EOS || true
unset BENCH_HARDCAP_SUFFIX || true

# Refuse before the card is taken rather than after: every processor named has to
# exist and the two sets must be disjoint, or the contrast is between nothing.
python3 - <<'PY'
import os, pathlib, sys
def parse(spec):
    out = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            out |= set(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out
fast = parse(os.environ["BENCH_PIN_CPUS"])
slow = parse(os.environ["BENCH_PIN_ALT_CPUS"])
if fast & slow:
    sys.exit(f"the two conditions share cpus {sorted(fast & slow)}")
if len(fast) != len(slow):
    sys.exit(f"{len(fast)} fast cpus against {len(slow)} slow: the contrast would "
             f"change the core count as well as the kind")
freq = {}
for c in sorted(fast | slow):
    p = pathlib.Path(f"/sys/devices/system/cpu/cpu{c}/cpufreq/cpuinfo_max_freq")
    if not p.exists():
        sys.exit(f"cpu{c} has no cpufreq entry, so this host cannot be asked "
                 f"what kind of core it is")
    freq[c] = int(p.read_text()) // 1000
f_max, s_max = max(freq[c] for c in fast), max(freq[c] for c in slow)
if s_max >= f_max:
    sys.exit(f"the slow set tops out at {s_max} MHz and the fast set at {f_max}: "
             f"they are not the two kinds of core this run is about")
print(f"  fast set {sorted(fast)} up to {f_max} MHz")
print(f"  slow set {sorted(slow)} up to {s_max} MHz, "
      f"{100 * (1 - s_max / f_max):.0f} % below")
PY

bash "$TELE_SH" "$TELE_SCHEMA" "$TELE_INTERVAL" "Y" &
TELE_PID=$!
trap 'kill "$TELE_PID" 2>/dev/null || true' EXIT

out="$BENCH/matrix_Y_pinning_$STAMP"
echo "=== run Y -> $(basename "$out")  $(date -Is) ==="
rc=0
if BENCH_OUT="$out" python3 "$RUNNER"; then :; else
    echo "!!! the runner exited non-zero" >&2
    rc=1
fi
echo "=== Y done $(date -Is) ==="

# Fail closed: forty-eight arm-runs, validated, or this is not a run.
[ -f "$out/RUN_COMPLETE.json" ] || { echo "FAIL: no RUN_COMPLETE.json" >&2; rc=1; }
n=$(find "$out" -maxdepth 1 -name '*__rep*.json' -printf . | wc -c)
echo "arm-runs:$n expected:48"
[ "$n" -eq 48 ] || { echo "FAIL: $n arm-runs, expected 48" >&2; rc=1; }
# and the treatment has to have been applied, not merely asked for
python3 - "$out" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
def cpus(spec):
    s = set()
    for p in (spec or "").split(","):
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-", 1)
            s |= set(range(int(a), int(b) + 1))
        else:
            s.add(int(p))
    return s
bad, seen = [], {}
for f in sorted(out.glob("*__rep*.json")):
    j = json.loads(f.read_text(encoding="utf-8"))
    req, got = j.get("cpus_requested"), j.get("cpus_allowed")
    if not req or not got or cpus(req) != cpus(got):
        bad.append(f"{f.name}: asked {req!r}, kernel {got!r}")
    seen.setdefault(j["arm"], set()).update(cpus(got))
if bad:
    sys.exit("FAIL: the mask was not what was asked for:\n  " + "\n  ".join(bad))
if len({frozenset(v) for v in seen.values()}) < 2:
    sys.exit("FAIL: every arm ran on the same processors")
for arm, v in sorted(seen.items()):
    print(f"  {arm:24s} ran on {sorted(v)}")
PY
exit "$rc"
