#!/usr/bin/env bash
# The length-matching contrast, measured inside one invocation.
#
# Run V compared `ignore_eos` on and off as two runs sixteen minutes apart, and
# ERRATA A16 finds an unexplained DFlash-specific invocation effect of the same
# size as the shift it reported - runs U3 and U5 are six minutes apart and
# differ by 8.30 pp on this drafter with nothing changed. `run_v2_crossover.sh`
# answers that by balancing the order over eight sessions and letting the
# between-session spread show how big the noise is. This answers it a better
# way: put both modes in the SAME balanced square.
#
# `BENCH_HARDCAP_SUFFIX=-cap` makes `<arm>-cap` run `<arm>`'s server flags and
# send `ignore_eos` on its own requests. So the ten arms
#
#     baseline          spec-dflash-n2      spec-dflash-n4      spec-mtp-n2      spec-draft-n8
#     baseline-cap      spec-dflash-n2-cap  spec-dflash-n4-cap  spec-mtp-n2-cap  spec-draft-n8-cap
#
# are one 10x10 Latin square: every arm visits every position exactly once, the
# two modes of a configuration are minutes apart rather than sixteen, and
# whatever state the drafter is in during that invocation applies to both. The
# contrast is
#
#     d(a) = log( r_{a-cap} / r_{baseline-cap} ) - log( r_a / r_baseline )
#
# and it is computed inside one invocation, not across two.
#
# Two sessions, so the contrast is replicated rather than asserted.
#
#   bash bench/run_v3_within.sh          # ~3.5 h, one RTX 3090, exclusive
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${BENCH_ROOT:-$HOME/bench}"
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
TELE="$BENCH/gpu_telemetry_V3_$STAMP.csv"
echo "runner    $RUNNER"
echo "telemetry $TELE_SH -> $TELE"

export MODEL_TARGET="${MODEL_TARGET:-$HOME/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
export MODEL_DRAFT="${MODEL_DRAFT:-$HOME/models/Qwen3.5-0.8B-Q4_K_M.gguf}"
export MODEL_DFLASH="${MODEL_DFLASH:-$HOME/models/qwen36-dflash-master.gguf}"
export MODEL_MTP="${MODEL_MTP:-$HOME/models/qwen36-mtp-q8_0.gguf}"
export BENCH_SERVER="${BENCH_SERVER:-$BENCH/llama-retest/build/bin/llama-server}"

# run V's five configurations, each in both modes
export BENCH_HARDCAP_SUFFIX="-cap"
export BENCH_ARMS="baseline,baseline-cap,spec-dflash-n2,spec-dflash-n2-cap,spec-dflash-n4,spec-dflash-n4-cap,spec-mtp-n2,spec-mtp-n2-cap,spec-draft-n8,spec-draft-n8-cap"
# ten arms, so ten repeats: the runner refuses `latin` unless every arm visits
# every position the same number of times, and 10/10 = 1
export BENCH_REPEATS=10
export BENCH_ORDER=latin
export BENCH_MAX_TOKENS=300
export BENCH_THINK=off
export BENCH_CTX=8192
export BENCH_FIT=on
export BENCH_FIT_TARGET=3072
export BENCH_CONCURRENCY=1
export BENCH_FLAVOR=master
# NOT set: this is the per-arm mechanism, and a run-level cap would flatten the
# freerun half into the capped one and measure nothing
unset BENCH_IGNORE_EOS || true

bash "$TELE_SH" "$TELE" &
TELE_PID=$!
trap 'kill "$TELE_PID" 2>/dev/null || true' EXIT

DONE=""; FAILED=""
for session in 1 2; do
    out="$BENCH/matrix_V3_s${session}_$STAMP"
    echo "=== session $session -> $(basename "$out")  $(date -Is) ==="
    if BENCH_OUT="$out" python3 "$RUNNER"; then :; else
        echo "!!! session $session exited non-zero" >&2
    fi
    if [ -f "$out/RUN_COMPLETE.json" ]; then DONE="$DONE s$session"; else
        FAILED="$FAILED s$session"; echo "!!! session $session did not complete" >&2
    fi
done

echo "=== V3 done $(date -Is) ==="
echo "completed:$DONE"
echo "failed:${FAILED:- none}"
