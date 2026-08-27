#!/usr/bin/env bash
# Split A12's checkpoint attribution into the synchronisation wait and the copy.
#
# A12 times `ckpt.update_tgt` / `update_dft` / `load_tgt` / `load_dft`. Those
# call `llama_state_seq_get_data_ext` / `set_data_ext`, and at 3737e4137 both
# begin with `ctx->synchronize()` - verified in the tested tree at
# `llama-context.cpp:4083`. So the published 39.07 s is elapsed INSIDE the
# checkpoint API calls, synchronisation included, and 54.7 % is an attribution
# to that boundary rather than a measurement of state-copy cost. Some of the
# wait would be paid elsewhere on a path with no checkpoints rather than
# disappearing.
#
# `apply_split_timers.py` drains the queue explicitly, immediately before each
# call, and times the drain. The call's own internal synchronize() then finds
# nothing outstanding, so what is left is the state work. Total work is
# unchanged - the wait happened either way, a microsecond earlier - and the
# field names are kept, so `update_tgt` still means the whole call. That gives
# the check that matters: the new total must reproduce A12's 39.07 s. If it
# does not, the split changed the thing being measured.
#
# Six repeats rather than run T's four: three arms over four repeats rotates
# 1,2,3,1 and the runner refuses to call that `latin` now. 6/3 = 2 is balanced.
#
# The llama.cpp tree is restored to stock at the end and the stock library hash
# is checked back. Local instrumentation is permitted there; leaving it is not.
#
#   bash bench/run_t4_split_timers.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${BENCH_ROOT:-$HOME/bench}"
TREE="${LLAMA_TREE:-$BENCH/llama-retest}"
APPLY="${SPLIT_APPLIER:-$BENCH/apply_split_timers.py}"
RUNNER="${BENCH_RUNNER:-$BENCH/retest_runner.py}"
TELE_SH="${BENCH_TELEMETRY:-$HERE/gpu_telemetry.sh}"
[ -f "$TELE_SH" ] || TELE_SH="$BENCH/gpu_telemetry.sh"
STAMP="$(date +%Y%m%d_%H%M%S)"
for f in "$APPLY" "$RUNNER" "$TELE_SH"; do
    [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
done

cd "$TREE"
WANT_COMMIT=3737e4137
[ "$(git rev-parse --short=9 HEAD)" = "$WANT_COMMIT" ] || {
    echo "tree is at $(git rev-parse --short=9 HEAD), want $WANT_COMMIT" >&2; exit 1; }
[ -z "$(git status --porcelain)" ] || {
    echo "tree is dirty; refusing to patch on top of something" >&2; exit 1; }

STOCK_IMPL="$(sha256sum build/bin/libllama-server-impl.so | cut -d' ' -f1)"
echo "stock  libllama-server-impl.so ${STOCK_IMPL:0:16}"

restore() {
    cd "$TREE"
    git checkout -- tools/server/server-context.cpp 2>/dev/null || true
    echo "=== rebuilding stock $(date -Is) ==="
    cmake --build build -j "$(nproc)" --target llama-server > "$BENCH/T4_rebuild_stock_$STAMP.log" 2>&1 || true
    local back
    back="$(sha256sum build/bin/libllama-server-impl.so | cut -d' ' -f1)"
    if [ "$back" = "$STOCK_IMPL" ]; then
        echo "=== tree and binary restored to stock (${back:0:16}) ==="
    else
        echo "!!! stock rebuild gave ${back:0:16}, expected ${STOCK_IMPL:0:16}" >&2
    fi
}
trap restore EXIT

python3 "$APPLY" "$TREE"
git diff > "$BENCH/checkpoint_timers_split_$STAMP.patch"
echo "=== patch: $(git diff --shortstat) ==="

echo "=== building instrumented $(date -Is) ==="
cmake --build build -j "$(nproc)" --target llama-server > "$BENCH/T4_build_$STAMP.log" 2>&1
INSTR_IMPL="$(sha256sum build/bin/libllama-server-impl.so | cut -d' ' -f1)"
echo "instrumented libllama-server-impl.so ${INSTR_IMPL:0:16}"
[ "$INSTR_IMPL" != "$STOCK_IMPL" ] || { echo "the build did not change; aborting" >&2; exit 1; }

cd "$BENCH"
export MODEL_TARGET="${MODEL_TARGET:-$HOME/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
export MODEL_DRAFT="${MODEL_DRAFT:-$HOME/models/Qwen3.5-0.8B-Q4_K_M.gguf}"
export MODEL_DFLASH="${MODEL_DFLASH:-$HOME/models/qwen36-dflash-master.gguf}"
export BENCH_SERVER="$TREE/build/bin/llama-server"
export LLAMA_SERVER_BIN="$BENCH_SERVER"
export BENCH_EXPECT_LIB_SHA256="$INSTR_IMPL"
# run T's arms and configuration, six repeats so the schedule balances
export BENCH_ARMS="baseline,spec-draft-n8,spec-dflash-n2"
export BENCH_REPEATS=6
export BENCH_ORDER=latin
export BENCH_MAX_TOKENS=300
export BENCH_THINK=on
export BENCH_CTX=8192
export BENCH_FIT=on
export BENCH_FIT_TARGET=3072
export BENCH_CONCURRENCY=1
export BENCH_FLAVOR=master
unset BENCH_IGNORE_EOS BENCH_HARDCAP_SUFFIX || true

# gpu_telemetry.sh takes [schema] [interval] [label] and names its own file
TELE_SCHEMA="${BENCH_TELEMETRY_SCHEMA:-compact}"
TELE_INTERVAL="${BENCH_TELEMETRY_INTERVAL:-5}"
bash "$TELE_SH" "$TELE_SCHEMA" "$TELE_INTERVAL" "T4" &
TELE_PID=$!
trap 'kill "$TELE_PID" 2>/dev/null || true; restore' EXIT

OUT="$BENCH/matrix_T4_split_$STAMP"
echo "=== measuring -> $(basename "$OUT")  $(date -Is) ==="
BENCH_OUT="$OUT" python3 "$RUNNER"
test -f "$OUT/RUN_COMPLETE.json" || { echo "T4 did not complete" >&2; exit 1; }
echo "=== T4 done $(date -Is) ==="
