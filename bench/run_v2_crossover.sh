#!/usr/bin/env bash
# Run V, again, with the mode order controlled.
#
# The first run V measured `ignore_eos` by running the whole freerun matrix and
# then the whole hard-cap matrix - 22:31:46 and 22:48:08. Each half was
# position-balanced internally and the MODE was not randomised at all, so the
# treatment is confounded with elapsed time and with whatever differs between
# two invocations of the driver. ERRATA A16 makes that fatal rather than
# untidy: it finds an unexplained DFlash-specific invocation effect spanning
# 9.4 pp on the same drafter, and the shift run V reported for
# `spec-dflash-n2` is 9.26 pp.
#
# This is an AB/BA crossover over eight sessions, in the order
#
#     AB BA BA AB BA AB AB BA
#
# where A is freerun and B is the hard cap. Each mode runs first four times and
# second four times, and the mean time position of the AB sessions equals that
# of the BA sessions (4.5 each), so the order contrast is not confounded with
# drift across the night either. The session is the unit to resample over - not
# the difference of two whole-run point estimates.
#
# Eight rather than the four the review asked for, because four sessions give
# three degrees of freedom and a t critical value of 3.18; eight give seven and
# 2.36, which is the difference between bounding the effect and measuring it.
# Analyse it with the session as the block:
#
#     d(arm, session) = log( r_arm,hard / r_base,hard )
#                     - log( r_arm,free / r_base,free )
#
# Everything else is run V's configuration verbatim, so the halves stay
# comparable with what is already published.
#
#   bash bench/run_v2_crossover.sh          # ~5 h, one RTX 3090, exclusive
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${BENCH_ROOT:-$HOME/bench}"
# The bench host keeps its own copy of the runner beside the run directories,
# and does NOT have gpu_telemetry.sh - that lives only in this repository. Both
# are resolved rather than assumed, and a missing one stops the script before
# any GPU time is spent.
RUNNER="${BENCH_RUNNER:-}"
for cand in "$RUNNER" "$BENCH/retest_runner.py" "$HERE/retest_runner.py"; do
    [ -n "$cand" ] && [ -f "$cand" ] && { RUNNER="$cand"; break; }
done
TELE_SH="${BENCH_TELEMETRY:-}"
for cand in "$TELE_SH" "$HERE/gpu_telemetry.sh" "$BENCH/gpu_telemetry.sh"; do
    [ -n "$cand" ] && [ -f "$cand" ] && { TELE_SH="$cand"; break; }
done
[ -f "$RUNNER" ]  || { echo "no retest_runner.py: set BENCH_RUNNER" >&2; exit 1; }
[ -f "$TELE_SH" ] || { echo "no gpu_telemetry.sh: set BENCH_TELEMETRY" >&2; exit 1; }
STAMP="$(date +%Y%m%d_%H%M%S)"
TELE_SCHEMA="${BENCH_TELEMETRY_SCHEMA:-compact}"
TELE_INTERVAL="${BENCH_TELEMETRY_INTERVAL:-5}"
echo "runner    $RUNNER"

export MODEL_TARGET="${MODEL_TARGET:-$HOME/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
export MODEL_DRAFT="${MODEL_DRAFT:-$HOME/models/Qwen3.5-0.8B-Q4_K_M.gguf}"
export MODEL_DFLASH="${MODEL_DFLASH:-$HOME/models/qwen36-dflash-master.gguf}"
export MODEL_MTP="${MODEL_MTP:-$HOME/models/qwen36-mtp-q8_0.gguf}"
# `retest_runner.py` reads LLAMA_SERVER_BIN and only that. This script
# exported BENCH_SERVER, a name nothing consumes, so a clean shell failed
# and a shell that happened to carry someone else's LLAMA_SERVER_BIN ran
# a binary this script never chose. Both are fixed by using the real name
# and asserting what the run must have used.
export LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$BENCH/llama-retest/build/bin/llama-server}"
[ -x "$LLAMA_SERVER_BIN" ] || { echo "not executable: $LLAMA_SERVER_BIN" >&2; exit 1; }
export BENCH_EXPECT_COMMIT="${BENCH_EXPECT_COMMIT:-3737e41370da1830a44c663f9929a0f27591ffa6}"

# run V verbatim
export BENCH_ARMS="baseline,spec-dflash-n2,spec-dflash-n4,spec-mtp-n2,spec-draft-n8"
export BENCH_REPEATS=5
export BENCH_MAX_TOKENS=300
export BENCH_THINK=off
export BENCH_CTX=8192
export BENCH_FIT=on
export BENCH_FIT_TARGET=3072
export BENCH_CONCURRENCY=1
export BENCH_ORDER=latin
export BENCH_FLAVOR=master

echo "telemetry $TELE_SH $TELE_SCHEMA $TELE_INTERVAL V2"
bash "$TELE_SH" "$TELE_SCHEMA" "$TELE_INTERVAL" "V2" &
TELE_PID=$!
trap 'kill "$TELE_PID" 2>/dev/null || true' EXIT

DONE=""
FAILED=""

half() {   # half <session> <mode>
    local session="$1" mode="$2"
    local out="$BENCH/matrix_V2_s${session}_${mode}_$STAMP"
    echo "=== session $session, $mode -> $(basename "$out")  $(date -Is) ==="
    if [ "$mode" = hardcap ]; then export BENCH_IGNORE_EOS=on; else export BENCH_IGNORE_EOS=off; fi
    # A half that fails must not take the night down with it: the runner refuses
    # to write RUN_COMPLETE.json for a run that did not validate, so a missing
    # marker is the whole check, and the analysis uses only sessions where BOTH
    # halves completed. Losing one session is better than losing seven.
    if BENCH_OUT="$out" python3 "$RUNNER"; then :; else
        echo "!!! session $session $mode exited non-zero" >&2
    fi
    if [ -f "$out/RUN_COMPLETE.json" ]; then
        DONE="$DONE s${session}/${mode}"
    else
        FAILED="$FAILED s${session}/${mode}"
        echo "!!! session $session $mode did not complete" >&2
    fi
}

# AB BA BA AB BA AB AB BA - each mode first four times and second four times,
# with the two orders balanced in mean time position
half 1 freerun ; half 1 hardcap
half 2 hardcap ; half 2 freerun
half 3 hardcap ; half 3 freerun
half 4 freerun ; half 4 hardcap
half 5 hardcap ; half 5 freerun
half 6 freerun ; half 6 hardcap
half 7 freerun ; half 7 hardcap
half 8 hardcap ; half 8 freerun

echo "=== V2 done $(date -Is) ==="
echo "completed:$DONE"
echo "failed:${FAILED:- none}"

# This used to end on `find | wc`, which succeeds, so the script exited 0 with
# every session failed. A driver that reports success on no data is worse than
# one that crashes: the operator sees output directories and a summary and
# believes the matrix ran.
halves=$(find "$BENCH" -maxdepth 1 -type d -name "matrix_V2_*_$STAMP" -printf . | wc -c)
complete=$(find "$BENCH" -maxdepth 1 -type d -name "matrix_V2_*_$STAMP" \
             -exec test -f '{}/RUN_COMPLETE.json' \; -printf . | wc -c)
echo "half directories:$halves  validated:$complete  expected:16"
rc=0
[ -z "$FAILED" ] || { echo "FAIL: sessions failed:$FAILED" >&2; rc=1; }
[ "$halves"   -eq 16 ] || { echo "FAIL: $halves half directories, expected 16" >&2; rc=1; }
[ "$complete" -eq 16 ] || { echo "FAIL: $complete validated, expected 16" >&2; rc=1; }
exit "$rc"
