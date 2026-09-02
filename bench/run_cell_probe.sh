#!/usr/bin/env bash
# Launch the per-number coverage probe as N shards.
#
#     bench/run_cell_probe.sh SHARDS EXPECTED_HEAD OUT_DIR [TMP_DIR]
#     bench/run_cell_probe.sh --check-only SHARDS EXPECTED_HEAD OUT_DIR [TMP_DIR]
#
# This exists because the probe was launched from a hand-written script in a
# home directory, where nothing in this repository could see it. On 2026-09-01
# that script asked for twenty-eight shards on an eight-core host, which is not
# an error anywhere: it simply runs at a quarter of the speed with every
# process at thirty per cent of a core, and it took thirty minutes to notice.
# Every gate this repository runs was green at the time and none of them
# covered the file. It is in `bench/` now, so `shellcheck` reaches it.
#
# The four things it refuses, each of which has cost a run here:
#
#   1. more shards than the host has processors. The work is CPU bound and
#      oversubscribing does not make it finish sooner.
#   2. a HEAD that is not the commit the caller means to attest. The shards
#      each clone HEAD and record its SHA, so a wrong checkout produces a
#      perfectly self-consistent attestation for the wrong tree.
#   3. a temporary directory without room for one clone per shard. The clones
#      are hardlinked for the object store and a full checkout otherwise; a
#      run whose scratch filled is how a set of attestations was lost.
#   4. a shard that failed. `wait` alone discards every exit code, so a crashed
#      shard and a finished one print the same closing line.
set -euo pipefail

CHECK_ONLY=0
BOOTSTRAP=0
# --bootstrap is for the one case the probe cannot start without help: its own
# six assertions read the attestations, so a tree whose attestations are stale,
# or whose shard count has just changed, fails them, and the only way to make
# them pass is to run the probe. The flag lets THOSE failures stand and nothing
# else, each shard records the list, and the aggregator refuses a set whose
# shards declared different ones. It is spelled out on every run that uses it so
# it cannot become the way the probe is normally launched.
while [ $# -gt 0 ]; do
    case ${1:-} in
        --check-only) CHECK_ONLY=1; shift ;;
        --bootstrap)  BOOTSTRAP=1; shift ;;
        *) break ;;
    esac
done
if [ $# -lt 3 ] || [ $# -gt 4 ]; then
    sed -n '2,8p' "$0" >&2
    exit 2
fi
SHARDS=$1
WANT_HEAD=$2
OUT_DIR=$3
ALLOW=""
if [ "$BOOTSTRAP" = 1 ]; then
    ALLOW="--allow-stale-probe-evidence"
    echo "BOOTSTRAP: the probe's own six assertions are allowed to be failing" >&2
    echo "  at the start of each shard, and nothing else is. Each attestation" >&2
    echo "  records the list and the aggregator checks the shards agree on it." >&2
fi
TMP_DIR=${4:-${TMPDIR:-/tmp}}

case $SHARDS in
    ''|*[!0-9]*) echo "FAIL: SHARDS must be a positive integer, got '$SHARDS'" >&2; exit 2 ;;
esac
[ "$SHARDS" -ge 1 ] || { echo "FAIL: SHARDS must be at least 1" >&2; exit 2; }

# How the work is DIVIDED and how much of it runs AT ONCE are two different
# numbers, and this script had one. It refused thirty-two shards on an eight
# processor host, which is right about the concurrency and wrong about the
# division: `analysis/verify_claims.py` asserts that exactly thirty-two
# attestations are committed, and that file is one of the six the release binding
# freezes, so the shard count is not a free parameter. Refusing it left no way to
# produce a usable set anywhere but a thirty-two processor host. The shards are
# independent, so eight at a time in four waves produces the same thirty-two
# attestations, and only the concurrency has to fit the host.
CPUS=$(nproc)
JOBS=${BENCH_PROBE_JOBS:-$CPUS}
case $JOBS in
    ''|*[!0-9]*) echo "FAIL: BENCH_PROBE_JOBS must be a positive integer" >&2; exit 2 ;;
esac
[ "$JOBS" -ge 1 ] || { echo "FAIL: BENCH_PROBE_JOBS must be at least 1" >&2; exit 2; }
if [ "$JOBS" -gt "$CPUS" ]; then
    echo "FAIL: $JOBS at once on a $CPUS processor host. The probe is CPU" >&2
    echo "      bound: this does not finish sooner, it finishes later, and" >&2
    echo "      nothing else reports it. Run it where the processors are." >&2
    exit 1
fi
[ "$JOBS" -gt "$SHARDS" ] && JOBS=$SHARDS

HEAD_SHA=$(git rev-parse HEAD)
case $HEAD_SHA in
    "$WANT_HEAD"*) ;;
    *) echo "FAIL: HEAD is $HEAD_SHA, and the run is meant to attest $WANT_HEAD" >&2
       exit 1 ;;
esac

# one clone per RUNNING shard, each a full checkout of the tracked tree. Not
# one per shard: a wave's clones are gone before the next wave starts, so what
# has to fit is the concurrency.
TREE_KB=$(git ls-files -z | du -sc --files0-from=- 2>/dev/null | tail -1 | cut -f1)
NEED_KB=$((TREE_KB * JOBS * 12 / 10))
mkdir -p "$TMP_DIR" "$OUT_DIR"
FREE_KB=$(df -Pk "$TMP_DIR" | tail -1 | awk '{print $4}')
if [ "$FREE_KB" -lt "$NEED_KB" ]; then
        echo "FAIL: $TMP_DIR has $((FREE_KB / 1024)) MB free and $JOBS clones of" >&2
    echo "      this tree need about $((NEED_KB / 1024)) MB" >&2
    exit 1
fi

echo "$SHARDS shards, $JOBS at a time on $CPUS processors, head $HEAD_SHA"
echo "  scratch $TMP_DIR: $((FREE_KB / 1024)) MB free, about $((NEED_KB / 1024)) MB wanted"
[ "$CHECK_ONLY" -eq 1 ] && { echo "check only, nothing launched"; exit 0; }

export TMPDIR="$TMP_DIR"
# In waves of $JOBS. Every exit code is collected, not just the last: `wait` with
# no argument returns 0 whatever the children did, which is why a crashed shard
# used to read as a finished one.
failed=0
i=0
while [ "$i" -lt "$SHARDS" ]; do
    pids=""
    j=0
    while [ "$j" -lt "$JOBS" ] && [ "$i" -lt "$SHARDS" ]; do
        python3 analysis/table_coverage.py --every-cell --covered --json $ALLOW \
            "--shard=$i/$SHARDS" > "$OUT_DIR/shard_$i.json" 2> "$OUT_DIR/shard_$i.err" &
        pids="$pids $!"
        i=$((i + 1))
        j=$((j + 1))
    done
    for p in $pids; do
        if ! wait "$p"; then
            failed=$((failed + 1))
        fi
    done
    echo "  $i of $SHARDS shards started and finished, $failed failed so far"
done

empty=0
for i in $(seq 0 $((SHARDS - 1))); do
    if [ ! -s "$OUT_DIR/shard_$i.json" ]; then
        empty=$((empty + 1))
        echo "  shard $i wrote no attestation:" >&2
        tail -3 "$OUT_DIR/shard_$i.err" >&2 || true
    fi
done

if [ "$failed" -ne 0 ] || [ "$empty" -ne 0 ]; then
    echo "FAILED: $failed shard(s) exited non-zero, $empty wrote nothing" >&2
    exit 1
fi
echo "$SHARDS shards finished, all wrote an attestation"
