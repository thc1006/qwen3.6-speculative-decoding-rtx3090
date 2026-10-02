#!/usr/bin/env bash
# Continuous GPU telemetry for the running matrix: clocks, power, temperature,
# and the throttle-reason bitmask. Without this a 3-hour run cannot rule out
# thermal downclocking biasing later arms.
#
#   bash bench/gpu_telemetry.sh [schema] [interval_seconds] [label]
#
# Three schemas exist because three were used, and this file used to carry only
# one of them while the other two lived inline in driver scripts that were never
# committed. Seventeen traces were recorded during the audit and the committed
# script produced exactly one of them. All three are here now so any trace in
# `gpu_telemetry_*.csv` can be reproduced:
#
#   full     18 nvidia-smi fields plus a wall-clock column, renamed headers.
#            One trace: gpu_telemetry_20260825_205707.csv, which is what
#            ERRATA C4b's thermal table is computed from. Default.
#   compact  9 fields, the form used for runs I/J through O2 and run T -
#            twelve traces, including the one behind ERRATA A16's thermal
#            comparison for run T.
#   raw      nvidia-smi's own `--format=csv` header, 10 fields, 1 s. Runs T3,
#            O3 and later; five traces, counting run Y's. The counts here are
#            derived from the committed traces by a test, because this one said
#            four after run Y added the fifth.
#
# `analysis/thermal_report.py` reads whichever it is given. The trace that
# belongs to a run shares its timestamp: gpu_telemetry_<label>_<stamp>.csv
# beside matrix_<label>_<stamp>/.
set -u
SCHEMA="${1:-full}"
INTERVAL="${2:-5}"
LABEL="${3:-}"
SUFFIX="$(date +%Y%m%d_%H%M%S)"
[ -n "$LABEL" ] && SUFFIX="${LABEL}_${SUFFIX}"
# The path, and a caller that can name it. `$HOME/bench` was written here with no
# override, so this could not run on a host that has no such directory and could
# not be tested anywhere. Worse, a driver could not know the file's name: this
# stamps it with its OWN `date`, and `bench/run_w_williams.sh` looks for the
# driver's `$STAMP` instead, which is the same second only if the two calls do not
# straddle one. `BENCH_TELEMETRY_OUT` lets the driver say the name and then check
# that exact file.
OUT_DIR="${BENCH_TELEMETRY_DIR:-$HOME/bench}"
OUT="${BENCH_TELEMETRY_OUT:-$OUT_DIR/gpu_telemetry_${SUFFIX}.csv}"
mkdir -p "$(dirname "$OUT")" 2>/dev/null || true
# Fail, loudly, before anything prints the line a driver reads as confirmation.
# Without this the sampler printed `TELEMETRY=...` and then failed on every tick
# for the length of the run, writing nothing: a three-hour matrix can finish with
# no record of what the card was doing and the only evidence is a stderr stream
# nobody reads. The same OUTCOME has happened twice here by other routes -- a file
# path passed where a schema was expected, and a trace this driver looked for
# under the wrong name -- and both times the run reported success.
if ! : > "$OUT" 2>/dev/null; then
    echo "FAIL: cannot create $OUT." >&2
    echo "      Set BENCH_TELEMETRY_DIR to a directory this user can write, or" >&2
    echo "      BENCH_TELEMETRY_OUT to the file itself. Refusing to sample into" >&2
    echo "      nothing: a run with no telemetry cannot rule out thermal" >&2
    echo "      downclocking biasing its later arms, which is why this exists." >&2
    exit 1
fi

case "$SCHEMA" in
full)
    FIELDS=timestamp,clocks.current.graphics,clocks.max.graphics,clocks.current.memory,clocks.current.sm,power.draw,power.limit,power.default_limit,temperature.gpu,utilization.gpu,memory.used,pstate,clocks_throttle_reasons.active,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.hw_power_brake_slowdown,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.gpu_idle
    echo 'wall_iso,ts,gfx_mhz,gfx_max_mhz,mem_mhz,sm_mhz,power_w,power_limit_w,power_default_w,temp_c,util_pct,mem_used_mib,pstate,throttle_active,thr_sw_thermal,thr_hw_thermal,thr_hw_power_brake,thr_sw_power_cap,thr_gpu_idle' > "$OUT"
    PREFIX_WALL=1
    ;;
compact)
    FIELDS=timestamp,utilization.gpu,memory.used,temperature.gpu,pstate,clocks.current.sm,clocks.current.memory,power.draw,clocks_throttle_reasons.active
    echo 'ts,util,mem_used,temp,pstate,clk_sm,clk_mem,pwr,throttle' > "$OUT"
    PREFIX_WALL=0
    ;;
raw)
    # nvidia-smi writes its own header; no renaming, units left in place
    nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,clocks.current.graphics,clocks.current.sm,power.draw,temperature.gpu,pstate,clocks_event_reasons.active \
        --format=csv -l "$INTERVAL" > "$OUT" 2>/dev/null &
    SMI=$!
    echo "TELEMETRY=$OUT  schema=raw  interval=${INTERVAL}s  pid=$SMI"
    # `-l` is nvidia-smi's OWN loop, so this shell has a CHILD that outlives
    # it. Every driver here stops its sampler by killing this shell, which for
    # the other two schemas is enough because they loop in bash -- and for this
    # one was not: run Y on 2026-09-26 left two of these sampling at one hertz
    # for four days, reparented to init, and neither a GPU-lock check nor a load
    # average would have named them. Without this trap the only schema that
    # records per-second state is the only one that cannot be turned off.
    trap 'kill "$SMI" 2>/dev/null || true' EXIT INT TERM
    wait "$SMI"
    exit 0
    ;;
*)
    echo "unknown schema '$SCHEMA'; use full, compact or raw" >&2
    exit 1
    ;;
esac

echo "TELEMETRY=$OUT  schema=$SCHEMA  interval=${INTERVAL}s"
while true; do
    [ "$PREFIX_WALL" -eq 1 ] && printf '%s,' "$(date -Iseconds)" >> "$OUT"
    nvidia-smi --query-gpu="$FIELDS" --format=csv,noheader >> "$OUT" 2>&1
    sleep "$INTERVAL"
done
