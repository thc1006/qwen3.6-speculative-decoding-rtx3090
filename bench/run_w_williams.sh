#!/usr/bin/env bash
# Run W - the carryover-balanced version of run V3.
#
# Why this exists
# ---------------
# V2 (AB/BA crossover) and V3 (both modes in one square) both balance treatment
# POSITION and neither balances first-order carryover. V3's cyclic rotation put
# every capped arm immediately after its own uncapped twin in 9 of 9 within-row
# adjacencies, so "which mode" and "what ran before" are the same variable and
# no analysis of that run can separate them. The fourth review was right that
# `length_mode.py` should not have called V3 "the design that identifies the
# effect", and this is the design that does.
#
# W is V3 with ONE difference: `BENCH_ORDER=williams` instead of `latin`. Same
# ten arms, same models, same fit target, same prompts, same everything. If W
# and V3 disagree, the schedule is the only thing that can explain it.
#
# What the square gives
# ---------------------
#   * every arm in every position exactly once            (Latin, as V3)
#   * every arm preceded by every other exactly once      (Williams, new)
#   * row order shuffled from a recorded seed, so the nine row-boundary
#     transitions - the only adjacencies an n x n square cannot balance - differ
#     between sessions rather than repeating the same nine pairs every time
#
# It also gives the first dataset that can test A16's other open question. Each
# arm runs ten times per session with a different predecessor each time, so
# fifty (predecessor, rate) points per arm over five sessions say whether
# `spec-dflash-n2` is sensitive to what ran before it. Nothing in this
# repository has been able to ask that.
#
#   bash bench/run_w_williams.sh [sessions]
#
# About 78 minutes a session on an exclusive card; five is the default, which is
# four degrees of freedom for the between-session interval.
set -euo pipefail

SESSIONS="${1:-5}"
BENCH="${BENCH_ROOT:-$HOME/bench}"
RUNNER="${BENCH_RUNNER:-$BENCH/retest_runner.py}"
TELE_SH="${BENCH_TELEMETRY:-$BENCH/gpu_telemetry.sh}"
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
# `retest_runner.py` reads LLAMA_SERVER_BIN and only that; see the note in
# run_v2_crossover.sh about the name this used to export instead.
export LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$BENCH/llama-retest/build/bin/llama-server}"
[ -x "$LLAMA_SERVER_BIN" ] || { echo "not executable: $LLAMA_SERVER_BIN" >&2; exit 1; }
export BENCH_EXPECT_COMMIT="${BENCH_EXPECT_COMMIT:-3737e41370da1830a44c663f9929a0f27591ffa6}"

# run V3 verbatim, except the schedule
export BENCH_HARDCAP_SUFFIX="-cap"
# the twins are listed, not inferred - exactly as run V3 listed them
export BENCH_ARMS="baseline,baseline-cap,spec-dflash-n2,spec-dflash-n2-cap,spec-dflash-n4,spec-dflash-n4-cap,spec-mtp-n2,spec-mtp-n2-cap,spec-draft-n8,spec-draft-n8-cap"
export BENCH_REPEATS=10
export BENCH_MAX_TOKENS=300
export BENCH_THINK=off
export BENCH_CTX=8192
export BENCH_FIT=on
export BENCH_FIT_TARGET=3072
export BENCH_CONCURRENCY=1
export BENCH_ORDER=williams
export BENCH_FLAVOR=master
# the run-level flag would cap the free half too and measure nothing
unset BENCH_IGNORE_EOS || true

echo "telemetry $TELE_SH $TELE_SCHEMA $TELE_INTERVAL W"
bash "$TELE_SH" "$TELE_SCHEMA" "$TELE_INTERVAL" "W" &
TELE_PID=$!
trap 'kill "$TELE_PID" 2>/dev/null || true' EXIT

DONE=""
FAILED=""
for session in $(seq 1 "$SESSIONS"); do
    out="$BENCH/matrix_W_s${session}_$STAMP"
    # a seed per session, deterministic from the stamp so the schedule can be
    # rebuilt from the manifest without keeping anything else
    export BENCH_SCHEDULE_SEED=$(( (10#${STAMP//_/} % 100000) + session ))
    export BENCH_OUT="$out"
    echo "=== session $session/$SESSIONS  seed $BENCH_SCHEDULE_SEED  $(date -Is) ==="
    if ! python3 "$RUNNER"; then
        echo "!!! session $session exited non-zero" >&2
    fi
    if [ -f "$out/RUN_COMPLETE.json" ]; then
        DONE="$DONE s$session"
    else
        FAILED="$FAILED s$session"
        echo "!!! session $session did not complete" >&2
    fi
done

echo "=== W done $(date -Is) ==="
echo "completed:$DONE"
echo "failed:${FAILED:- none}"

# Fail closed: every session, every arm-run, and the balance the run is named
# for. A driver that reports success on partial data is worse than one that
# crashes, because the operator sees directories and believes the matrix ran.
sessions=$(find "$BENCH" -maxdepth 1 -type d -name "matrix_W_s*_$STAMP" -printf . | wc -c)
complete=$(find "$BENCH" -maxdepth 1 -type d -name "matrix_W_s*_$STAMP" \
             -exec test -f '{}/RUN_COMPLETE.json' ';' -printf . | wc -c)
echo "sessions:$sessions  validated:$complete  expected:$SESSIONS"
rc=0
[ -z "$FAILED" ] || { echo "FAIL: sessions failed:$FAILED" >&2; rc=1; }
[ "$sessions" -eq "$SESSIONS" ] || { echo "FAIL: $sessions sessions" >&2; rc=1; }
[ "$complete" -eq "$SESSIONS" ] || { echo "FAIL: $complete validated" >&2; rc=1; }
for d in "$BENCH"/matrix_W_s*_"$STAMP"; do
    n=$(find "$d" -maxdepth 1 -name '*__rep*.json' -printf . | wc -c)
    [ "$n" -eq 100 ] || { echo "FAIL: $d has $n arm-runs, expected 100" >&2; rc=1; }
    python3 - "$d" <<'PY' || rc=1
import json, sys
m = json.load(open(f"{sys.argv[1]}/manifest.json"))
for field in ("schedule_is_position_balanced",
              "schedule_first_order_carryover_balanced"):
    if not m.get(field):
        print(f"FAIL: {sys.argv[1]} manifest says {field} is "
              f"{m.get(field)!r}", file=sys.stderr)
        sys.exit(1)
PY
done
exit "$rc"
