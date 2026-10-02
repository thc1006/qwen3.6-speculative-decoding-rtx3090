#!/usr/bin/env bash
# Plan Z layer A: achieved memory bandwidth against thermal state, at LOCKED clocks.
#
#     bash bench/run_z_layer_a.sh [LABEL]
#
# Three phases, in one invocation so one set of locks covers all of them:
#
#   calibrate   bandwidth at several locked core clocks, memory clock fixed. The
#               measurement is only about memory if bandwidth is flat across them;
#               a core clock that limits the copy makes everything after it a
#               measurement of the SMs. It also picks the lock: the lowest clock
#               still at plateau has the most power headroom, so the lock is least
#               likely to be broken by the power cap as the card heats.
#   ascending   one continuous load from a cold card to its plateau.
#   descending  the same window with a long idle gap after each, so the card cools
#               between windows and the same temperatures are crossed in the other
#               direction at different elapsed times. If bandwidth is a function of
#               thermal state the two phases trace one curve; if it is a function of
#               time since the load began, they do not. That is the control.
#
# WHAT THIS REFUSES, each for a reason this repository has already paid for:
#
#   1. a GPU that is already being measured. The lock is whole-host.
#   2. clocks that are ALREADY locked when it starts. Something left them that way
#      and a run inheriting them is a run whose treatment it did not set.
#   3. a load binary that is not this repository's source. Run Y's launcher refuses
#      a runner that is not `bench/retest_runner.py` at HEAD, for the same reason.
#   4. a card that is not declared in `tests/test_harness_invariants.py`. A figure
#      from an undeclared card can be attributed to any 3090 here.
#   5. a lock the driver did not actually apply. Read back, every time.
#   6. a clock that did not HOLD. `-lgc` is a request: the card drops below it under
#      the power cap, and a run whose clock moved is not a run at locked clocks.
#      Checked against the sampler's own trace at the end, not assumed.
#   7. a restore that did not happen. The clocks are restored from a trap on every
#      exit path and the restore is read back; a card left locked would silently
#      halve every later measurement on this host.
set -u

LABEL="${1:-pilotA}"
case $LABEL in
    *[!A-Za-z0-9_]*|"") echo "LABEL must be alphanumeric" >&2; exit 2 ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
BENCH="${BENCH_ROOT:-$HOME/bench}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$BENCH/pilot_layer_a_${LABEL}_$STAMP"
GPU_INDEX="${BENCH_GPU_INDEX:-0}"

# the design, overridable so a dry run is cheap
# 1950 is in the list because the hold test found this card capping at 1965
# under the load with 2100 requested: the highest that HOLDS is what the lock
# wants, and a list that jumps 1800 to 2100 cannot find it.
CAL_CLOCKS="${Z_CAL_CLOCKS:-1200 1500 1800 1950 2100}"
CAL_SECONDS="${Z_CAL_SECONDS:-20}"
MEM_CLOCK="${Z_MEM_CLOCK:-9501}"
ASC_SECONDS="${Z_ASC_SECONDS:-900}"
DESC_SECONDS="${Z_DESC_SECONDS:-1800}"
# The ascending phase sweeps the range in the first couple of minutes and then
# sits at the plateau, so its windows are shorter: more points across the sweep,
# which is where the slope lives. The descending phase is one window per duty
# cycle and has as many points as it has cycles.
ASC_WINDOW="${Z_ASC_WINDOW:-5}"
DESC_WINDOW="${Z_DESC_WINDOW:-10}"
# 50 s gave a duty cycle of one in six, an equilibrium near idle, and a phase
# that never reached the ascending range. One in three starts at the plateau and
# decays THROUGH it, which is what the control needs.
IDLE="${Z_IDLE:-20}"
TELE_INTERVAL="${Z_TELE_INTERVAL:-1}"
# 45 C is reachable from the plateau in about two minutes on this card and the
# idle floor is about 37, so it buys most of the available range without waiting
# for the last few degrees, which take as long again.
# 40 C with the clocks restored: the true idle floor on this card is about 37 and
# the last few degrees take as long as the first fifteen. A locked clock cannot
# reach it at all, which is why the cooling happens unlocked.
COOL_TO="${Z_COOL_TO:-40}"
COOL_TIMEOUT="${Z_COOL_TIMEOUT:-600}"
# bandwidth within this fraction of the best is "at plateau"
PLATEAU="${Z_PLATEAU:-0.995}"

die() { echo "FAIL: $*" >&2; exit 1; }
say() { echo "=== $* ==="; }

command -v nvidia-smi > /dev/null || die "no nvidia-smi"
NVCC="${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}"
[ -x "$NVCC" ] || die "no nvcc; set NVCC"
sudo -n true 2>/dev/null || die "clock locking needs passwordless sudo on this host"

# 1. not during a measurement. Same lock every driver here takes.
# `$HERE`, not a relative "bench": this script can be invoked from anywhere and a
# relative import would find the guard only when it happened to be run from the
# repository root -- which is the shape of a guard that is sometimes absent.
python3 - "$HERE" <<'PY' || exit 1
import sys
sys.path.insert(0, sys.argv[1])
import host_guard
host_guard.protect("plan Z layer A")
PY

mkdir -p "$OUT" || die "cannot create $OUT"

# 3. the load is THIS repository's source, compiled now. A binary lying around is
#    a binary nobody can tie to a commit.
SRC="$REPO/bench/vram_bandwidth.cu"
[ -f "$SRC" ] || die "no $SRC"
if git -C "$REPO" rev-parse --git-dir > /dev/null 2>&1; then
    want="$(git -C "$REPO" show HEAD:bench/vram_bandwidth.cu | sha256sum | cut -d' ' -f1)"
    have="$(sha256sum "$SRC" | cut -d' ' -f1)"
    [ "$want" = "$have" ] || die "$SRC differs from HEAD. Commit it or check it out:
      HEAD $want
      tree $have"
fi
LOAD="$OUT/vram_bandwidth"
"$NVCC" -O3 -arch=sm_86 -Xcompiler -Wall,-Wextra -o "$LOAD" "$SRC" \
    > "$OUT/build.log" 2>&1 || { cat "$OUT/build.log" >&2; die "the load did not build"; }

# 4. which card, and is it one we have written down
# On commas. `read` on whitespace put "GeForce" in the driver field, because the
# card's NAME has spaces in it, and a mangled provenance line is one that gets
# quoted later as if it meant something.
IFS=',' read -r CARD_UUID CARD_NAME CARD_DRIVER CARD_PL <<EOF
$(nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid,name,driver_version,power.limit \
    --format=csv,noheader)
EOF
CARD_UUID="${CARD_UUID# }"; CARD_NAME="${CARD_NAME# }"
CARD_DRIVER="${CARD_DRIVER# }"; CARD_PL="${CARD_PL# }"
[ -n "${CARD_UUID:-}" ] || die "could not read the card's uuid"
PREFIX="$(echo "$CARD_UUID" | sed 's/^GPU-//' | cut -c1-8)"
grep -q "\"$PREFIX\":" "$REPO/tests/test_harness_invariants.py" \
    || die "$CARD_UUID is not a declared card. Add it to EveryMeasurementMustNameItsCard.CARDS
      with what it is, or a figure from it can be attributed to any 3090 here."

# 2. nothing may already be locked -- reported, and then cleared regardless.
#
# There is NO field that reports a `-lgc` lock. `Applications Clocks` is a
# different and deprecated mechanism that reads "Requested functionality has been
# deprecated" on this driver, and the first version of this looked for
# `ApplicationsClocksSetting` inside `nvidia-smi -q -d CLOCK`, which does not even
# contain the event reasons: the guard matched nothing and had never fired. It was
# found when an ad-hoc command timed out before its own restore and left this card
# locked at 1800/9501 for twenty minutes, with `clocks.max.sm` still reading its
# default 2130 and so looking restored.
#
# What a lock looks like is an IDLE card whose clocks are above idle: unlocked this
# card sits at 210 MHz and 36 W, locked it sits at 1800 MHz and 118 W.
IDLE_SM_MAX="${Z_IDLE_SM_MAX:-500}"
inherited_lock_note() {
    local util apps sm
    util=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=utilization.gpu \
        --format=csv,noheader,nounits)
    apps=$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid \
        --format=csv,noheader | wc -l)
    sm=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=clocks.sm \
        --format=csv,noheader,nounits)
    if [ "$util" -eq 0 ] && [ "$apps" -eq 0 ] && [ "$sm" -gt "$IDLE_SM_MAX" ]; then
        echo "NOTE: idle at $sm MHz, above the $IDLE_SM_MAX this card idles at, so" >&2
        echo "      something had left its clocks locked. Clearing them: a lock" >&2
        echo "      this run did not set is a treatment it cannot describe." >&2
    fi
}

unlock() {
    sudo -n nvidia-smi -i "$GPU_INDEX" -rgc > /dev/null 2>&1 || true
    sudo -n nvidia-smi -i "$GPU_INDEX" -rmc > /dev/null 2>&1 || true
    sleep 1
}

RESTORED=0
restore() {
    [ "$RESTORED" = 1 ] && return 0
    RESTORED=1
    unlock
    # 7. read the restore back. A card left locked halves every later measurement.
    local sm mem
    sm=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=clocks.max.sm --format=csv,noheader,nounits)
    mem=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=clocks.max.memory --format=csv,noheader,nounits)
    echo "restored: max sm ${sm} MHz, max mem ${mem} MHz" | tee -a "$OUT/clocks.log"
}
trap 'restore; kill "${TELE_PID:-0}" 2>/dev/null || true' EXIT INT TERM

# 5. apply and READ BACK
lock() {  # lock CORE_MHZ MEM_MHZ
    local core="$1" mem="$2" got_core got_mem
    # captured rather than redirected: the redirect belongs to this shell and not
    # to sudo, which reads as a privilege confusion even where it is harmless
    local said
    said=$(sudo -n nvidia-smi -i "$GPU_INDEX" -lgc "$core,$core" 2>&1) \
        || { printf '%s\n' "$said" >> "$OUT/clocks.log"
             die "could not lock the core clock to $core: $said"; }
    printf '%s\n' "$said" >> "$OUT/clocks.log"
    said=$(sudo -n nvidia-smi -i "$GPU_INDEX" -lmc "$mem,$mem" 2>&1) \
        || { printf '%s\n' "$said" >> "$OUT/clocks.log"
             die "could not lock the memory clock to $mem: $said"; }
    printf '%s\n' "$said" >> "$OUT/clocks.log"
    sleep 2
    got_core=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=clocks.sm --format=csv,noheader,nounits)
    got_mem=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=clocks.mem --format=csv,noheader,nounits)
    echo "asked core $core mem $mem; idle readback core $got_core mem $got_mem" \
        >> "$OUT/clocks.log"
    [ "$got_mem" = "$mem" ] || die "memory clock reads $got_mem after locking $mem"
}

cool_to() {  # cool_to CELSIUS WHY
    local want="$1" why="$2" t waited=0
    t=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=temperature.gpu \
        --format=csv,noheader,nounits)
    say "cooling $why: $t C now, waiting for $want C"
    while [ "$t" -gt "$want" ] && [ "$waited" -lt "$COOL_TIMEOUT" ]; do
        sleep 10
        waited=$((waited + 10))
        t=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=temperature.gpu \
            --format=csv,noheader,nounits)
    done
    # recorded either way: a phase that started warm is a phase with less range,
    # and a reader is entitled to know which happened rather than to assume
    echo "cool $why: reached $t C after ${waited}s (wanted $want, timeout $COOL_TIMEOUT)" \
        | tee -a "$OUT/clocks.log"
    [ "$t" -le "$want" ] || echo "  NOTE: did not reach $want C; the range will be" \
        "$((t - want)) C narrower than the design asks for" >&2
}

inherited_lock_note
unlock

say "plan Z layer A -> $(basename "$OUT")"
{
    echo "card    $CARD_UUID  $CARD_NAME  driver $CARD_DRIVER  limit $CARD_PL W"
    echo "source  $(sha256sum "$SRC" | cut -d' ' -f1)"
    echo "nvcc    $("$NVCC" --version | tail -2 | head -1 | tr -s ' ')"
    echo "design  mem $MEM_CLOCK, calibrate [$CAL_CLOCKS], ascending ${ASC_SECONDS}s,"
    echo "        descending ${DESC_SECONDS}s, windows ${ASC_WINDOW}s and"
    echo "        ${DESC_WINDOW}s, idle ${IDLE}s, cool to ${COOL_TO} C"
} | tee "$OUT/manifest.txt"

# --- calibrate -------------------------------------------------------------
say "calibrate"
: > "$OUT/calibrate.csv"
for core in $CAL_CLOCKS; do
    lock "$core" "$MEM_CLOCK"
    "$LOAD" --seconds "$CAL_SECONDS" --window "$((CAL_SECONDS / 2))" \
        > "$OUT/cal_$core.csv" 2> "$OUT/cal_$core.err" \
        || die "the load failed at core $core"
    best=$(tail -n +2 "$OUT/cal_$core.csv" | cut -d, -f5 | sort -g | tail -1)
    echo "$core,$best" >> "$OUT/calibrate.csv"
    echo "  core $core MHz -> $best GB/s"
done
# The HIGHEST at plateau, not the lowest. Bandwidth is flat across the plateau, so
# the core clock does not buy throughput -- it buys POWER, and power is what heats
# the card. Layer A reads a slope against temperature and a slope needs range: the
# first version of this picked the lowest, which held the clock beautifully and
# reached twelve degrees of range. Whether the highest HOLDS is a separate question
# and the hold test below answers it before an hour is spent.
CANDIDATES=$(python3 - "$OUT/calibrate.csv" "$PLATEAU" <<'PY'
import sys
rows = [l.split(",") for l in open(sys.argv[1]).read().split() if l]
vals = {int(a): float(b) for a, b in rows}
top = max(vals.values())
ok = sorted((c for c, v in vals.items() if v >= float(sys.argv[2]) * top),
            reverse=True)
if not ok:
    sys.exit("no clock reached the plateau")
print(" ".join(str(c) for c in ok))
PY
) || die "could not choose a core clock"
echo "  at plateau, highest first: $CANDIDATES"

# 6a. does it HOLD under sustained load, at this clock, before the long phases.
#     `-lgc` is a request and the power cap outranks it; finding that out at the end
#     of a forty-five minute run costs the run.
hold_test() {  # hold_test CORE
    local core="$1" t csv
    lock "$core" "$MEM_CLOCK"
    csv="$OUT/hold_$core.csv"
    BENCH_TELEMETRY_OUT="$csv" bash "$HERE/gpu_telemetry.sh" full 1 "hold$core" &
    t=$!
    sleep 2
    kill -0 "$t" 2>/dev/null || die "telemetry died during the hold test"
    "$LOAD" --seconds "${Z_HOLD_SECONDS:-60}" --window 10 \
        > "$OUT/hold_$core.load.csv" 2> "$OUT/hold_$core.err" \
        || { kill "$t" 2>/dev/null; die "the load failed at core $core"; }
    kill "$t" 2>/dev/null || true
    wait "$t" 2>/dev/null || true
    python3 "$HERE/check_clocks_held.py" "$csv" "$core" "$MEM_CLOCK" \
        > "$OUT/hold_$core.verdict" 2>&1
}

LOCK_CORE=""
for core in $CANDIDATES; do
    echo "  hold test at $core MHz"
    if hold_test "$core"; then
        LOCK_CORE="$core"
        sed -n '2,5p' "$OUT/hold_$core.verdict" | sed 's/^/    /'
        echo "    held; locking there"
        break
    fi
    echo "    did not hold:"
    grep '^FAIL' "$OUT/hold_$core.verdict" | head -2 | sed 's/^/      /'
done
[ -n "$LOCK_CORE" ] || die "no clock at plateau held under load; the power cap
      binds at every one of [$CANDIDATES], so this card cannot run layer A at
      fixed clocks without a lower memory clock or a higher power limit"

# --- the two phases, under one lock ---------------------------------------
# Cool with the clocks RESTORED. A locked clock is held at idle too, so the card
# burns 112 W doing nothing and settles at 49 C: the first version cooled while
# locked and bought nineteen degrees of range where the card can give twenty-eight.
# The lock goes on after the cooling and the sampler starts after the lock, so the
# trace covers locked time only and `check_clocks_held.py` can hold every sample in
# it to the lock.
unlock
cool_to "$COOL_TO" "before the ascending phase, with the clocks restored"
lock "$LOCK_CORE" "$MEM_CLOCK"
TELE_CSV="$OUT/gpu_telemetry_${LABEL}_$STAMP.csv"
# `full`, not `raw`: the raw schema does not record the memory clock, so a run
# that used it cannot show that clock held, and holding it is the premise.
# `full` also carries the throttle-reason columns, which name the cap the card
# was against, and an ISO wall-clock prefix, which is the unambiguous spelling.
BENCH_TELEMETRY_OUT="$TELE_CSV" bash "$HERE/gpu_telemetry.sh" full "$TELE_INTERVAL" "$LABEL" &
TELE_PID=$!
sleep 2
kill -0 "$TELE_PID" 2>/dev/null || die "telemetry died at startup"

say "ascending"
"$LOAD" --seconds "$ASC_SECONDS" --window "$ASC_WINDOW" \
    > "$OUT/ascending.csv" 2> "$OUT/ascending.err" || die "the ascending phase failed"
# NO cooling here. Plan Z says the descending phase comes "after the plateau",
# and the first version of this cooled first: the duty cycle then equilibrated near
# idle and the phase spanned 1.2 C at 52, never crossing the ascending phase's
# 54 to 68. The two phases have to traverse the SAME range in opposite directions
# or the control does not exist, and it did not: no temperature was visited by
# both, so the run produced a tidy number and no attribution.
say "descending"
"$LOAD" --seconds "$DESC_SECONDS" --window "$DESC_WINDOW" --idle "$IDLE" \
    > "$OUT/descending.csv" 2> "$OUT/descending.err" || die "the descending phase failed"

kill "$TELE_PID" 2>/dev/null || true
wait "$TELE_PID" 2>/dev/null || true
restore

# --- fail closed -----------------------------------------------------------
rc=0
for f in ascending descending; do
    n=$(( $(wc -l < "$OUT/$f.csv") - 1 ))
    echo "$f windows:$n"
    [ "$n" -ge 3 ] || { echo "FAIL: $f has $n windows" >&2; rc=1; }
done
tele_rows=0
[ -s "$TELE_CSV" ] && tele_rows=$(( $(wc -l < "$TELE_CSV") - 1 ))
echo "telemetry rows:$tele_rows"
[ "$tele_rows" -ge 60 ] || { echo "FAIL: $TELE_CSV holds $tele_rows samples" >&2; rc=1; }

# 6. the clock HELD. `-lgc` is a request and the power cap outranks it.
python3 "$HERE/check_clocks_held.py" "$TELE_CSV" "$LOCK_CORE" "$MEM_CLOCK" || rc=1
# 8. and the CONTROL ran. Two phases that share no temperature cannot separate
#    "hotter" from "running longer", which is the whole reason the second one
#    exists. A run that did not execute its own control must not read as one that
#    did, so this is a condition and not a note.
python3 "$REPO/analysis/layer_a_slope.py" --check "$OUT" || rc=1
verdict=FAILED
[ "$rc" = 0 ] && verdict=complete
echo "=== layer A $verdict -> $OUT ==="
exit "$rc"
