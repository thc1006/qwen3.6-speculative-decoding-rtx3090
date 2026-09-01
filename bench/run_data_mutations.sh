#!/usr/bin/env bash
# Run the data perturbation suite as N shards under one host lock.
#
#     bench/run_data_mutations.sh SHARDS [OUT_DIR]
#     bench/run_data_mutations.sh --check-only SHARDS [OUT_DIR]
#
# `tests/data_mutate.py` spawns one claim-checker per perturbation and runs
# eighty-four of them one at a time: fifty-six minutes on a thirty-two processor
# host, using one processor. The work is embarrassingly parallel and what made
# it sequential was sharing one mirror between perturbations, which the restore
# loop then had to undo. A shard gets a mirror of its own, so nothing it
# perturbs can reach another shard at all, and each checks its own mirror clean
# at the end.
#
# What it refuses, each for the reason the probe's launcher gives:
#
#   1. more shards than the host has processors. The work is CPU bound and
#      oversubscribing finishes later, not sooner. A run asked for
#      twenty-eight on an eight-core host and took thirty minutes to notice.
#   2. a shard that failed. `wait` alone discards every exit code, so a crashed
#      shard and a clean one print the same closing line.
#   3. a shard that wrote nothing. An empty log is what a killed process leaves.
#
# The whole-host verification lock is taken HERE, once, around the fan-out.
# `host_guard.serialise` is an exclusive flock and a fan-out cannot take it per
# shard: the first would hold it and the rest would exit. Holding it here keeps
# what the lock is for, which is that two verification pipelines never overlap
# on a host that may be measuring.
set -euo pipefail

CHECK_ONLY=0
if [ "${1:-}" = "--check-only" ]; then
    CHECK_ONLY=1
    shift
fi
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    sed -n '2,5p' "$0" >&2
    exit 2
fi

SHARDS=$1
OUT_DIR=${2:-$HOME/data_mutations_out}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LOCK="${TMPDIR:-/tmp}/.qwen36-verify.lock"

case $SHARDS in ''|*[!0-9]*) echo "FAIL: SHARDS must be a number" >&2; exit 2;; esac
[ "$SHARDS" -ge 1 ] || { echo "FAIL: SHARDS must be at least 1" >&2; exit 2; }

CPUS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc)
if [ "$SHARDS" -gt "$CPUS" ]; then
    echo "FAIL: $SHARDS shards on $CPUS processors. The work is CPU" >&2
    echo "      bound: this does not finish sooner, it finishes later, and" >&2
    echo "      nothing else reports it. Run it where the processors are." >&2
    exit 2
fi

cd "$ROOT"
N=$(python3 - <<'PY'
import ast, pathlib, sys
src = pathlib.Path("tests/data_mutate.py").read_text(encoding="utf-8")
for n in ast.walk(ast.parse(src)):
    if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "MUTATIONS"
                                         for t in n.targets):
        print(len(n.value.elts))
        sys.exit(0)
sys.exit("cannot count MUTATIONS")
PY
)
if [ "$SHARDS" -gt "$N" ]; then
    echo "FAIL: $SHARDS shards for $N perturbations leaves a shard empty," >&2
    echo "      and an empty shard proves nothing while exiting 0." >&2
    exit 2
fi

# Scratch, for the reason the probe's launcher gives about its clones: each
# shard holds TWO mirrors of the tracked tree, a pristine one and the one it
# perturbs, in `TMPDIR`. Eight shards is 2.6 GB and this box has 3.5 GB free;
# thirty-two is 10.4 GB. A run whose scratch fills leaves half a suite and a
# traceback that names neither the disk nor the shard.
TMP_DIR=${TMPDIR:-/tmp}
TREE_KB=$(git ls-files -z | du -sck --files0-from=- 2>/dev/null | tail -1 | awk '{print $1}')
NEED_KB=$((TREE_KB * SHARDS * 2 * 12 / 10))
FREE_KB=$(df -Pk "$TMP_DIR" | tail -1 | awk '{print $4}')
if [ "$FREE_KB" -lt "$NEED_KB" ]; then
    echo "FAIL: $TMP_DIR has $((FREE_KB / 1024)) MB free and $SHARDS shards need" >&2
    echo "      about $((NEED_KB / 1024)) MB, two mirrors of this tree each." >&2
    exit 2
fi

echo "$SHARDS shards on $CPUS processors, $N perturbations, lock $LOCK"
echo "  scratch $TMP_DIR: $((FREE_KB / 1024)) MB free, about $((NEED_KB / 1024)) MB wanted"
[ "$CHECK_ONLY" -eq 1 ] && { echo "check only, nothing launched"; exit 0; }

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/shard_*.log

# One lock for the fan-out. `-n` so a second pipeline fails immediately with a
# message rather than waiting for an hour and looking hung.
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "FAIL: another verification pipeline holds $LOCK. Running two at" >&2
    echo "      once is what produced the CPU bursts a benchmark on this" >&2
    echo "      host recorded as contention. Wait for it." >&2
    exit 2
fi

pids=""
for i in $(seq 0 $((SHARDS - 1))); do
    python3 tests/data_mutate.py "--shard=$i/$SHARDS" \
        > "$OUT_DIR/shard_$i.log" 2>&1 &
    pids="$pids $!"
done

bad=0
empty=0
for p in $pids; do
    if ! wait "$p"; then bad=$((bad + 1)); fi
done
for i in $(seq 0 $((SHARDS - 1))); do
    if [ ! -s "$OUT_DIR/shard_$i.log" ]; then
        empty=$((empty + 1))
        echo "shard $i wrote nothing" >&2
    fi
done

# `|| true` on both: `grep` exits 1 when it matches nothing, `pipefail` makes
# the assignment fail, and `set -e` then kills the script BEFORE it reports
# anything -- so the one case these lines exist for, every shard having failed,
# was the case that produced no diagnosis at all. A crash is worse than a
# failure because of what does not run after it.
caught=$( { grep -ho 'all [0-9]* perturbations detected' "$OUT_DIR"/shard_*.log \
            || true; } | awk '{s += $2} END {print s + 0}')
survived=$( { grep -ho 'SURVIVED' "$OUT_DIR"/shard_*.log || true; } \
            | awk 'END {print NR + 0}')

echo
if [ "$bad" -eq 0 ] && [ "$empty" -eq 0 ] && [ "$caught" -eq "$N" ]; then
    echo "all $N perturbations detected across $SHARDS shards"
    exit 0
fi
echo "FAILED: $bad shard(s) exited non-zero, $empty wrote nothing," >&2
echo "        $caught of $N perturbations reported caught, $survived survived" >&2
grep -h 'SURVIVED\|COULD NOT APPLY' "$OUT_DIR"/shard_*.log >&2 || true
exit 1
