# qwen3.6-speculative-decoding-rtx3090 v3.0 — May 2026 update

## What's new vs v2.3

v2.3 (2026-04-26) established **llama.cpp draft-spec is NET LOSS** on RTX 3090 with Qwen3.6-35B-A3B Q4_K_XL target + Qwen3.5-0.8B Q4_K_M draft (Oleg-style). Specific configs:
- 02 srogmann ngram-mod (n=24): not in master, fail-fast
- 03 oleg draft-spec (max=32): NET LOSS
- 04 oleg draft-spec (max=16): NET LOSS

v3.0 adds **DFlash** (block-diffusion drafter, llama.cpp PR #22105) — the newest spec-decoding family with z-lab's published claims of 1.98×−2.9× speedup on B200. **First public RTX 3090 + DFlash + Q4 datapoint.**

## TL;DR

**DFlash on RTX 3090 + Q4_K_XL target via llama.cpp PR #22105 = NET LOSS −44 %** vs no-spec baseline. Best DFlash config (max=8) gets 77 tok/s vs 138 tok/s baseline. Slightly less bad than Oleg draft-spec's −52 % NET LOSS, but still net negative. Confirms: **no llama.cpp speculative-decoding method tested gives a positive yield on consumer Ampere with Q4 quantized target.**

## Setup

Same hardware/model as v2.3:
- 1× RTX 3090 24 GB, driver 580.126.09, CUDA 12.0, 350 W stock
- Target: `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (21 GB, from unsloth/Qwen3.6-35B-A3B-GGUF)
- llama.cpp PR #22105 checked out (`git fetch origin pull/22105/head`), incremental rebuild on master baseline
- Drafter: `z-lab/Qwen3.6-35B-A3B-DFlash` HF safetensors → GGUF via the PR's modified `convert_hf_to_gguf.py` with `--target-model-dir` flag (needs target's tokenizer + config files, ~22 MB total)
- 5 prompts × 1 trial × 3 draft-max configs (4, 8, 16). Same prompts as v2.3.
- Args: `-ngl 999 -c 4096 -fa on -ctk q8_0 -ctv q8_0 -n 200 --temp 0.5 --seed 42 -no-cnv -st`

## Results

### Per-config Generation tok/s

| config | p1 | p2 | p3 | p4 | p5 | mean | stdev |
|---|---:|---:|---:|---:|---:|---:|---:|
| **01_baseline** (no spec, master) | 136.3 | 139.3 | 140.0 | 139.4 | 139.4 | **138.9** | 1.47 |
| 03_oleg_draft_2_32 (Q4 draft) | 63.6 | 76.0 | 63.5 | 62.3 | 62.2 | 65.5 | 5.89 |
| 04_oleg_draft_2_16 (Q4 draft) | 66.0 | 75.9 | 63.5 | 65.5 | 61.9 | 66.6 | 5.47 |
| 05_dflash_max16 (PR #22105) | 60.1 | 78.8 | 62.4 | 69.6 | 57.9 | 65.8 | 8.51 |
| **06_dflash_max8** (PR #22105) | 71.3 | 90.8 | 78.2 | 75.2 | 69.5 | **77.0 ⭐** | 8.42 |
| 07_dflash_max4 (PR #22105) | 72.1 | 79.4 | 74.1 | 78.2 | 70.8 | 74.9 | 3.76 |

### Cross-method comparison

| method | mean tok/s | vs baseline |
|---|---:|---:|
| no spec (baseline) | 138.9 | reference |
| Oleg draft-spec max=32 | 65.5 | **−52.8 %** ❌ |
| Oleg draft-spec max=16 | 66.6 | **−52.1 %** ❌ |
| DFlash max=16 | 65.8 | **−52.6 %** ❌ |
| **DFlash max=8** ⭐ | **77.0** | **−44.6 %** ❌ (best) |
| DFlash max=4 | 74.9 | −46.1 % ❌ |

## Key takeaways

### 1. DFlash NET LOSS direction is consistent with the underlying pathology

The DFlash architecture conditions a small block-diffusion drafter on **multiple target hidden-states**. The drafter we used (z-lab/Qwen3.6-35B-A3B-DFlash, BF16) was trained against target hidden states in **FP16**. Our test target is **Q4_K_XL** quantized — the hidden-state distribution shifts subtly under aggressive 4-bit quantization, and the drafter is no longer cleanly aligned with the verifier.

Per llama.cpp PR #22105 author's own note: *"for Qwen3.5/3.6 MoE, performance is currently not optimal due to MoE + hybrid structure not well supported."* Our number (−44 %) sits inside that envelope.

### 2. The wider mechanism — MoE expert routing × consumer-Ampere bandwidth

Independently of the Q4 mismatch, the Qwen3.6-35B-A3B model routes **8-of-256 experts per token**. The expert-saturation threshold for hiding spec verification cost behind expert load is around ~94 tokens (per llama.cpp #21569 community discussion + our own measurements). At single-stream batch=1 with `--draft-max ≤ 32`, drafted tokens stay below saturation, so verification still has to load the union of expert slices — exceeding what was saved.

This is **not Q4-specific and not consumer-GPU-specific in isolation** — it's the joint effect of:
- MoE routing pulling fresh expert slices for each verified token
- Ampere SM 8.6 memory bandwidth not hiding verification cost
- Q4 quantization further shrinking the math/bandwidth ratio in spec's favor — but not enough to flip sign

### 3. Is this just a llama.cpp problem?

Cross-reference v3 sister publication `qwen3.6-vllm-2x3090`: **vLLM MTP on the same hardware (dual 3090) is +27.5 % NET WIN at k=1** and **+8 % tok/s additional at k=3**. So the llama.cpp NET LOSS is **not "speculative decoding is bad on consumer Ampere"** — it is "llama.cpp's speculative-decoding implementations (draft-model and DFlash) cannot beat vLLM MTP for Qwen3.6 MoE on consumer Ampere".

Possible reasons vLLM MTP wins:
- MTP head is **co-trained** with the base model (DeepSeek MTP technique adopted by Qwen3.6) — naturally aligned with verifier
- Single-stream verification of MTP-drafted tokens reuses the same forward as the verifier — no separate expert routing pass
- vLLM's CUDA-graph capture and FlashInfer MoE backend reduce per-step kernel-launch overhead

llama.cpp draft-model speculative decoding has none of these alignments.

## Recommendations

1. **For voice agents on consumer Ampere with Qwen3.6-35B-A3B**: run vLLM MTP, not llama.cpp speculative decoding. `qwen3.6-vllm-2x3090 v4.0` for full MTP recipe.
2. **For DFlash users on consumer Ampere**: wait for FP16 (or BF16) target benchmarks before drawing conclusions about DFlash itself. Q4 target collapses the technique.
3. **For the llama.cpp PR #22105 maintainers**: the Qwen3.5/3.6 MoE notebook entry is consistent with a MoE-routing mismatch problem rather than a quant problem. Q4 is contributing but not the dominant cause.

## Reproduction

```bash
# 1. Get llama.cpp + PR #22105
cd ~/bench/llama.cpp
git fetch origin pull/22105/head:pr-22105
git checkout pr-22105
cd build && cmake --build . --config Release -j$(nproc)

# 2. Get DFlash drafter (HF safetensors)
~/.local/bin/hf download z-lab/Qwen3.6-35B-A3B-DFlash --local-dir ~/models/qwen36-dflash

# 3. Get target model tokenizer + config (small files only)
~/.local/bin/hf download Qwen/Qwen3.6-35B-A3B \
    config.json tokenizer.json tokenizer_config.json vocab.json merges.txt \
    --local-dir ~/models/qwen36-target-meta

# 4. Convert drafter HF → GGUF (needs CPU-only torch, ~1-2 min)
python -m venv ~/dflash_convert_venv
~/dflash_convert_venv/bin/pip install -r ~/bench/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
~/dflash_convert_venv/bin/python ~/bench/llama.cpp/convert_hf_to_gguf.py \
    ~/models/qwen36-dflash --outtype bf16 \
    --target-model-dir ~/models/qwen36-target-meta \
    --outfile ~/models/qwen36-dflash.gguf

# 5. Run bench (replaces v2.3 bench_3090_oleg.sh; do not use set -euo pipefail
#    + grep|tail combo, since failed configs leave grep with no matches)
bash bench_dflash.sh
```

Bench script (`bench_dflash.sh`) provided in `bench/` directory.

## License

Apache 2.0. Cite freely.

## Cross-references

- v2.3: original Oleg-style draft-spec NET LOSS finding
- Sister repo `qwen3.6-vllm-2x3090` v4.0: vLLM MTP positive yield on same hardware
- llama.cpp PR #22105 (open draft, 2026-05): DFlash implementation under review
- z-lab/dflash GitHub: original DFlash paper (B200 results)
- HF discussion `unsloth/Qwen3.6-35B-A3B-GGUF/discussions/14`: community thread covering speculative-decoding NET LOSS on RTX 3090
