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
# W uses V3's treatment definitions, prompts, models, fit target and server
# build, with `BENCH_ORDER=williams` instead of `latin` and a LATER HARNESS
# REVISION. It is not V3 verbatim, and this comment used to say it was: the two
# manifests record different runner hashes, and the diff between exactly those
# two blobs is archived at `v4_audit_2026_08_25/harness/V3_to_W_runner.diff`
# with a hunk-by-hunk classification beside it. 189 lines, all of it the
# Williams schedule builder, provenance assertions and provenance records; none
# of it reaches the request body, the server argv, timing collection, teardown
# or aggregation. One hunk adds a failure mode V3 did not have -- an arm-run
# with rows but no observed target identity is refused -- which can reject more
# and cannot move a number.
#
# So a difference between W and V3 is not attributable to the schedule alone
# without reading that diff. It is a small second difference, it is enumerated,
# and it is not zero.
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
# The run label. Hardcoded "W" until 2026-08-30, and every analyser globs
# `matrix_W_*` with no invocation qualifier, so a second invocation would
# have been pooled with the first without anyone choosing that -- and the
# thing A16 is about is exactly the difference between invocations. A new
# invocation gets its own label so nothing pools it by accident.
LABEL="${BENCH_RUN_LABEL:-W}"
case "$LABEL" in
    *[!A-Za-z0-9]*|"") echo "BENCH_RUN_LABEL must be alphanumeric" >&2; exit 1;;
esac
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

# V3's treatment definitions and environment; see the diff note above for the
# harness revision that is the other difference
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

echo "telemetry $TELE_SH $TELE_SCHEMA $TELE_INTERVAL $LABEL"
bash "$TELE_SH" "$TELE_SCHEMA" "$TELE_INTERVAL" "$LABEL" &
TELE_PID=$!
trap 'kill "$TELE_PID" 2>/dev/null || true' EXIT
# It was started and never looked at again. A sampler that dies in the first
# second leaves a two-line trace and the run still reports success, so the one
# record of what the card was doing is missing exactly when it is wanted.
sleep 2
kill -0 "$TELE_PID" 2>/dev/null || { echo "FAIL: telemetry died at startup" >&2; exit 1; }
RUN_T0=$(date +%s)
BENCH_ROOT_TELECHECK="$(dirname "$0")/check_telemetry_cover.py"

DONE=""
FAILED=""
for session in $(seq 1 "$SESSIONS"); do
    out="$BENCH/matrix_${LABEL}_s${session}_$STAMP"
    # a seed per session, deterministic from the stamp so the schedule can be
    # rebuilt from the manifest without keeping anything else
    export BENCH_SCHEDULE_SEED=$(( (10#${STAMP//_/} % 100000) + session ))
    export BENCH_OUT="$out"
    echo "=== session $session/$SESSIONS  seed $BENCH_SCHEDULE_SEED  $(date -Is) ==="
    # The runner's exit status IS a verdict, not a warning. This printed one and
    # then decided purely on whether RUN_COMPLETE.json existed -- so a leftover
    # attestation from an earlier run, or a runner that failed after writing it,
    # counted as a completed session.
    runner_rc=0
    python3 "$RUNNER" || runner_rc=$?
    if [ "$runner_rc" -ne 0 ]; then
        echo "!!! session $session exited $runner_rc" >&2
        FAILED="$FAILED s$session(rc=$runner_rc)"
    elif [ -f "$out/RUN_COMPLETE.json" ]; then
        DONE="$DONE s$session"
    else
        FAILED="$FAILED s$session(no RUN_COMPLETE)"
        echo "!!! session $session did not complete" >&2
    fi
done

echo "=== W done $(date -Is) ==="
echo "completed:$DONE"
echo "failed:${FAILED:- none}"

# Fail closed: every session, every arm-run, and the balance the run is named
# for. A driver that reports success on partial data is worse than one that
# crashes, because the operator sees directories and believes the matrix ran.
sessions=$(find "$BENCH" -maxdepth 1 -type d -name "matrix_${LABEL}_s*_$STAMP" -printf . | wc -c)
complete=$(find "$BENCH" -maxdepth 1 -type d -name "matrix_${LABEL}_s*_$STAMP" \
             -exec test -f '{}/RUN_COMPLETE.json' ';' -printf . | wc -c)
echo "sessions:$sessions  validated:$complete  expected:$SESSIONS"
rc=0

# Stop the sampler deliberately and read its exit status, then require the trace
# to cover the run rather than merely to exist.
RUN_T1=$(date +%s)
kill "$TELE_PID" 2>/dev/null || true
tele_rc=0
wait "$TELE_PID" 2>/dev/null || tele_rc=$?
# 143 is SIGTERM, which is how it is meant to end
[ "$tele_rc" -eq 0 ] || [ "$tele_rc" -eq 143 ] || {
    echo "FAIL: telemetry exited $tele_rc" >&2; rc=1; }
TELE_CSV=$(find "$BENCH" -maxdepth 1 -name "gpu_telemetry_W_$STAMP.csv" | head -1)
if [ -z "$TELE_CSV" ]; then
    echo "FAIL: no telemetry trace for this run" >&2; rc=1
else
    python3 "$BENCH_ROOT_TELECHECK" "$TELE_CSV" "$RUN_T0" "$RUN_T1" "$TELE_INTERVAL" || rc=1
fi
[ -z "$FAILED" ] || { echo "FAIL: sessions failed:$FAILED" >&2; rc=1; }
[ "$sessions" -eq "$SESSIONS" ] || { echo "FAIL: $sessions sessions" >&2; rc=1; }
[ "$complete" -eq "$SESSIONS" ] || { echo "FAIL: $complete validated" >&2; rc=1; }
for d in "$BENCH"/matrix_"$LABEL"_s*_"$STAMP"; do
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
