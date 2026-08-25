#!/usr/bin/env bash
# Continuous GPU telemetry for the running matrix: clocks, power, temperature,
# and the throttle-reason bitmask. Without this a 3-hour run cannot rule out
# thermal downclocking biasing later arms.
OUT="$HOME/bench/gpu_telemetry_$(date +%Y%m%d_%H%M%S).csv"
echo "wall_iso,$(nvidia-smi --query-gpu=timestamp,clocks.current.graphics,clocks.max.graphics,clocks.current.memory,clocks.current.sm,power.draw,power.limit,power.default_limit,temperature.gpu,utilization.gpu,memory.used,pstate,clocks_throttle_reasons.active,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.hw_power_brake_slowdown,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.gpu_idle --format=csv,noheader | head -0; echo 'ts,gfx_mhz,gfx_max_mhz,mem_mhz,sm_mhz,power_w,power_limit_w,power_default_w,temp_c,util_pct,mem_used_mib,pstate,throttle_active,thr_sw_thermal,thr_hw_thermal,thr_hw_power_brake,thr_sw_power_cap,thr_gpu_idle')" > "$OUT"
echo "TELEMETRY=$OUT"
while true; do
  printf '%s,' "$(date -Iseconds)" >> "$OUT"
  nvidia-smi --query-gpu=timestamp,clocks.current.graphics,clocks.max.graphics,clocks.current.memory,clocks.current.sm,power.draw,power.limit,power.default_limit,temperature.gpu,utilization.gpu,memory.used,pstate,clocks_throttle_reasons.active,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.hw_power_brake_slowdown,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.gpu_idle \
    --format=csv,noheader >> "$OUT" 2>&1
  sleep 5
done
