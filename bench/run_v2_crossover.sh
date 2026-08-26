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
# This is an AB/BA crossover over four sessions:
#
#     session 1   freerun -> hardcap
#     session 2   hardcap -> freerun
#     session 3   hardcap -> freerun
#     session 4   freerun -> hardcap
#
# so each mode appears first twice and second twice, and the session is the
# unit to resample over - not the difference of two whole-run point estimates.
# Analyse it with the session as the block:
#
#     d(arm, session) = log( r_arm,hard / r_base,hard )
#                     - log( r_arm,free / r_base,free )
#
# Everything else is run V's configuration verbatim, so the halves stay
# comparable with what is already published.
#
#   bash bench/run_v2_crossover.sh          # ~3 h, one RTX 3090, exclusive
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
TELE="$BENCH/gpu_telemetry_V2_$STAMP.csv"
echo "runner    $RUNNER"
echo "telemetry $TELE_SH"

export MODEL_TARGET="${MODEL_TARGET:-$HOME/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
export MODEL_DRAFT="${MODEL_DRAFT:-$HOME/models/Qwen3.5-0.8B-Q4_K_M.gguf}"
export MODEL_DFLASH="${MODEL_DFLASH:-$HOME/models/qwen36-dflash-master.gguf}"
export MODEL_MTP="${MODEL_MTP:-$HOME/models/qwen36-mtp-q8_0.gguf}"
export BENCH_SERVER="${BENCH_SERVER:-$BENCH/llama-retest/build/bin/llama-server}"

# run V verbatim
export BENCH_ARMS="baseline,spec-dflash-n2,spec-dflash-n4,spec-mtp-n2,spec-draft-n8"
export BENCH_REPEATS=5
export BENCH_MAX_TOKENS=300
export BENCH_THINK=off
export BENCH_CTX=8192
export BENCH_FIT=on
export BENCH_CONCURRENCY=1
export BENCH_ORDER=latin
export BENCH_FLAVOR=master

echo "telemetry -> $TELE"
bash "$TELE_SH" "$TELE" &
TELE_PID=$!
trap 'kill "$TELE_PID" 2>/dev/null || true' EXIT

half() {   # half <session> <mode>
    local session="$1" mode="$2"
    local out="$BENCH/matrix_V2_s${session}_${mode}_$STAMP"
    echo "=== session $session, $mode -> $out  $(date -Is) ==="
    if [ "$mode" = hardcap ]; then export BENCH_IGNORE_EOS=on; else export BENCH_IGNORE_EOS=off; fi
    BENCH_OUT="$out" python3 "$RUNNER"
    # the runner refuses to write RUN_COMPLETE.json for a run that did not
    # validate, so this is the whole check
    test -f "$out/RUN_COMPLETE.json" || { echo "session $session $mode did not complete" >&2; exit 1; }
}

# AB / BA / BA / AB
half 1 freerun ; half 1 hardcap
half 2 hardcap ; half 2 freerun
half 3 hardcap ; half 3 freerun
half 4 freerun ; half 4 hardcap

echo "=== V2 done $(date -Is) ==="
ls -d "$BENCH"/matrix_V2_*_"$STAMP"
