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
#            O3 and later; four traces.
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
OUT="$HOME/bench/gpu_telemetry_${SUFFIX}.csv"

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
    echo "TELEMETRY=$OUT  schema=raw  interval=${INTERVAL}s  pid=$!"
    wait
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
