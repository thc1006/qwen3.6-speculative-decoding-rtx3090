#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# HISTORICAL SCRIPT - kept as evidence, corrected only for portability.
#
# Audited 2026-08-25. See ../ERRATA.md (or ERRATA.md at the repo root).
# The measurement flags below are UNCHANGED so this file still documents what
# was actually executed. Two of them did not do what the comments claim:
#
#   -no-cnv     REJECTED by llama-cli on these builds. The committed logs show
#               "--no-conversation is not supported by llama-cli / please use
#               llama-completion instead". (ERRATA D1)
#   /no_think   Did NOT disable thinking. The same logs contain "[Start
#               thinking]" and a full reasoning trace. The working switches on
#               these builds are `-rea off` and `--reasoning-budget 0`.
#               (ERRATA D2)
#
# Do not use this script for new measurements. Use bench/retest_runner.py.
# Host-specific paths are now environment variables with the original values
# as defaults.
# ---------------------------------------------------------------------------
# Experiment 2 — N=3 code/JSON-only prompts (low-entropy / structured workload)
# Tests workload-dependency boundary: does spec-decode flip positive on consumer
# Ampere when prompts are structured (high expert-overlap) vs the diverse
# natural-language prompts in n3_<ts>/?
set -euo pipefail

MAIN="${MODEL_TARGET:-$HOME/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
DRAFT="${MODEL_DRAFT:-$HOME/models/Qwen3.5-0.8B-Q4_K_M.gguf}"
CLI="${LLAMA_CLI:-$HOME/bench/llama.cpp/build/bin/llama-cli}"
COMMON=(-ngl 999 -c 16384 -fa on -ctk q8_0 -ctv q8_0 -n 200 --temp 0.5 --seed 42 -no-cnv -st)

PROMPTS=(
  "Write a Python class ThreadSafeLRUCache implementing get(key) and put(key, value) with type hints, using threading.RLock. Maximum 100 lines. /no_think"
  "Output a JSON object describing a REST API for user authentication. Top-level keys: name, version, endpoints (array). Each endpoint has: path, method, request_schema, response_schema. Include /login, /logout, /refresh. /no_think"
  "Write a Rust function fn merge_sort<T: Ord + Clone>(arr: &mut [T]) that implements merge sort in place. Include doc comments. /no_think"
  "Generate a JSON config object for an Nginx reverse proxy with 3 upstream servers, SSL termination, and rate limiting. Use realistic field names. /no_think"
  "Write a PostgreSQL query that finds the top 10 customers by total order value in the last 30 days, joined with their email and country. Include comments. /no_think"
)

OUTROOT="${BENCH_OUT:-$HOME/bench/n3_codejson_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTROOT"
exec >> "$OUTROOT/master.log" 2>&1

echo "=== Experiment 2 · N=3 · code/JSON prompts only · standalone 3090 ==="
echo "Start: $(date -u +%FT%TZ)"
echo "Build: $($CLI --version 2>&1 | grep version | head -1)"
echo "Output: $OUTROOT"
nvidia-smi --query-gpu=name,memory.used,memory.total,power.limit --format=csv,noheader

run_cfg() {
  local trial=$1 cfg=$2; shift 2
  local -a extra=("$@")
  local dir="$OUTROOT/trial_$trial/$cfg"
  mkdir -p "$dir"
  echo ""
  echo "--- trial=$trial cfg=$cfg n_extra=${#extra[@]} ---"
  for i in 1 2 3 4 5; do
    local p="${PROMPTS[$((i-1))]}"
    local log="$dir/p$i.log"
    echo "  [$cfg p=$i] $(date -u +%T)"
    if [ ${#extra[@]} -eq 0 ]; then
      "$CLI" -m "$MAIN" "${COMMON[@]}" -p "$p" < /dev/null > "$log" 2>&1 || echo "    [WARN] non-zero p=$i"
    else
      "$CLI" -m "$MAIN" "${COMMON[@]}" -p "$p" "${extra[@]}" < /dev/null > "$log" 2>&1 || echo "    [WARN] non-zero p=$i"
    fi
    grep -E '\[ Prompt:|accept' "$log" | tail -2
  done
}

for trial in 1 2 3; do
  echo ""
  echo "================== Trial $trial / 3 =================="
  echo "$(date -u +%FT%TZ)"
  run_cfg "$trial" "01_baseline"
  run_cfg "$trial" "02_oleg_draft_2_32"      -md "$DRAFT" --draft-min 2 --draft-max 32
  run_cfg "$trial" "03_srogmann_draft_48_64" -md "$DRAFT" --draft-min 48 --draft-max 64
done

echo ""
echo "=== ALL DONE $(date -u +%FT%TZ) ==="
echo "Out: $OUTROOT"
