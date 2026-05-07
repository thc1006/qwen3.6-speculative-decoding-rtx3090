#!/usr/bin/env bash
# DFlash bench (5 prompts x 3 draft-max settings)
# Run on RTX 3090 with llama.cpp PR #22105 + z-lab/Qwen3.6-35B-A3B-DFlash drafter
# IMPORTANT: do NOT use 'set -euo pipefail' + grep|tail combo — empty grep
# match (e.g. when a config errors and log has no "Prompt:" string) makes
# pipefail fail the whole script. We omit set -e for that reason.

MAIN="$HOME/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
DRAFT="$HOME/models/qwen36-dflash.gguf"
CLI="$HOME/bench/llama.cpp/build/bin/llama-cli"
OUTDIR="$HOME/bench/out_$(date +%Y%m%d_%H%M%S)"
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
