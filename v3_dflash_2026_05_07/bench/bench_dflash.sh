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
# DFlash bench (5 prompts x 3 draft-max settings)
# Run on RTX 3090 with llama.cpp PR #22105 + z-lab/Qwen3.6-35B-A3B-DFlash drafter
# IMPORTANT: do NOT use 'set -euo pipefail' + grep|tail combo — empty grep
# match (e.g. when a config errors and log has no "Prompt:" string) makes
# pipefail fail the whole script. We omit set -e for that reason.

MAIN="${MODEL_TARGET:-$HOME/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
DRAFT="${MODEL_DFLASH:-$HOME/models/qwen36-dflash.gguf}"
CLI="${LLAMA_CLI:-$HOME/bench/llama.cpp/build/bin/llama-cli}"
OUTDIR="${BENCH_OUT:-$HOME/bench/out_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"/{05_dflash_max16,06_dflash_max8,07_dflash_max4}

COMMON=(-ngl 999 -c 4096 -fa on -ctk q8_0 -ctv q8_0 -n 200 --temp 0.5 --seed 42 -no-cnv -st)
PROMPTS=(
  "Why does the sky look blue? Answer in two sentences. /no_think"
  "Write a Python function fib(n) returning the first n Fibonacci numbers as a list. /no_think"
  "Explain TCP vs UDP in 3 concise bullet points. /no_think"
  "Give 5 numbered steps to cook firm tofu at home. /no_think"
  "Write a short haiku about debugging a memory leak at 2am. /no_think"
)

run_cfg() {
  local tag="$1"; shift
  local -a extra=("$@")
  local dir="$OUTDIR/$tag"
  echo ""
  echo "=== config: $tag === args: ${extra[*]}"
  for i in 1 2 3 4 5; do
    local p="${PROMPTS[$((i-1))]}"
    local log="$dir/p${i}.log"
    echo "  -- [$tag] prompt $i --"
    "$CLI" -m "$MAIN" "${COMMON[@]}" -md "$DRAFT" --dflash "${extra[@]}" -p "$p" < /dev/null > "$log" 2>&1
    grep -E "Prompt:|Generation:" "$log" 2>/dev/null | tail -2
  done
}

run_cfg "05_dflash_max16" --draft-max 16
run_cfg "06_dflash_max8"  --draft-max 8
run_cfg "07_dflash_max4"  --draft-max 4

echo ""
echo "=== bench DFlash complete $(date -u +%FT%TZ) ==="
