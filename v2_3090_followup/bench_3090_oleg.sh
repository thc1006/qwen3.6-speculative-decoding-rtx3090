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
# Re-bench Qwen3.6-35B-A3B on 3090 host in response to Oleg-dM HF comment.
# Runs 4 configs × 5 prompts + 1 verbose single-run.
set -euo pipefail

MAIN="${MODEL_TARGET:-$HOME/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
DRAFT="${MODEL_DRAFT:-$HOME/models/Qwen3.5-0.8B-Q4_K_M.gguf}"
BIN="${LLAMA_BIN_DIR:-$HOME/bench/llama.cpp/build/bin}"
CLI="$BIN/llama-cli"

OUTDIR="${BENCH_OUT:-$HOME/bench/out_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"

{
  echo "== bench start $(date -u +%FT%TZ) =="
  echo "== host: $(hostname) =="
  echo "== llama.cpp: $("$CLI" --version 2>&1 | grep '^version' | head -1) =="
  echo "== GPU state =="
  nvidia-smi --query-gpu=name,clocks.current.graphics,clocks.current.memory,clocks.max.graphics,clocks.max.memory,power.limit,power.default_limit --format=csv
  echo "== model files =="
  ls -la "$MAIN" "$DRAFT"
} | tee "$OUTDIR/manifest.txt"

COMMON=(-ngl 999 -c 16384 -fa on -ctk q8_0 -ctv q8_0 -n 200 --temp 0.5 --seed 42 -no-cnv -st)

# 5 diverse prompts.  `/no_think` suppresses Qwen3 reasoning for fair
# tok/s measurement against the original post-PR-19493 bench.
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
  mkdir -p "$dir"
  echo ""
  echo "=== config: $tag  ===  args: ${extra[*]}"
  local idx=0
  for p in "${PROMPTS[@]}"; do
    idx=$((idx+1))
    local log="$dir/p${idx}.log"
    echo "  -- [$tag] prompt $idx -- $(date -u +%T) --"
    "$CLI" -m "$MAIN" "${COMMON[@]}" -p "$p" "${extra[@]}" < /dev/null > "$log" 2>&1 || {
      echo "    [WARN] cli exit non-zero; see $log"
    }
    grep -E 'Prompt:|Generation:|accept|draft' "$log" | tail -4
  done
}

# config 1 — baseline
run_cfg "01_baseline"

# config 2 — srogmann ngram-mod recipe (reproduce original)
run_cfg "02_srogmann_ngmod_n24" \
    --spec-type ngram-mod \
    --spec-ngram-size-n 24 \
    --draft-min 48 --draft-max 64

# config 3 — Oleg draft-model with adaptive min=2 max=32
run_cfg "03_oleg_draft_2_32" \
    -md "$DRAFT" \
    --draft-min 2 --draft-max 32

# config 4 — Oleg draft-model with tighter max=16
run_cfg "04_oleg_draft_2_16" \
    -md "$DRAFT" \
    --draft-min 2 --draft-max 16

# config 5 — verbose single-run to inspect per-token acceptance
echo ""
echo "=== verbose per-token dump (config oleg_2_32, prompt 1) ==="
"$CLI" -m "$MAIN" "${COMMON[@]}" \
    -md "$DRAFT" --draft-min 2 --draft-max 32 \
    -v -p "${PROMPTS[0]}" < /dev/null > "$OUTDIR/verbose.log" 2>&1 || true
tail -50 "$OUTDIR/verbose.log"

{
  echo ""
  echo "== bench end $(date -u +%FT%TZ) =="
  echo "Output in $OUTDIR"
  ls -la "$OUTDIR"
} | tee -a "$OUTDIR/manifest.txt"
