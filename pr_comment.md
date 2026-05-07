# Draft comment for llama.cpp PR #19493 / Issue #20039

_Target URL: https://github.com/ggml-org/llama.cpp/pull/19493 (or as a fresh comment on Issue #20039 linking to repo)_

> **UPDATE 2026-05-07 — v3 DFlash via llama.cpp PR #22105.** First public
> RTX 3090 + DFlash + Q4 datapoint added to the repo under
> `v3_dflash_2026_05_07/`. Setup: same `3090` host + same target Q4_K_XL
> as v2; drafter `z-lab/Qwen3.6-35B-A3B-DFlash` converted to GGUF using
> PR #22105's modified `convert_hf_to_gguf.py` with `--target-model-dir`.
> Result: best DFlash config (`--draft-max=8`) is **77.0 tok/s vs 138.9
> baseline (NET LOSS −44.6 %)**. Slightly less bad than v2 Oleg draft-spec
> (−52 %) but still net negative. The MoE-expert-routing × consumer-Ampere
> bandwidth pathology from v2 generalises: at single-stream batch=1 with
> draft-max ≤ 16 the verify pass loads the union of expert sets at
> K ≪ T_thres ≈ 94, exceeding savings. Q4-target hidden-state mismatch
> with the BF16-trained drafter likely contributes; recovery would
> require an FP16/BF16 target which doesn't fit on a single 3090 (35 GB
> in BF16, 17.5 GB in AWQ Q4). Cross-engine status unchanged: vLLM MTP
> on 2× RTX 3090 PCIe remains the only positive-yield speculative
> decoding path on this hardware (sister repo
> [`thc1006/qwen3.6-vllm-2x3090`](https://github.com/thc1006/qwen3.6-vllm-2x3090)
> v3.0/v4.0). DFlash data + reproducer in
> [`v3_dflash_2026_05_07/`](v3_dflash_2026_05_07/).

> **UPDATE 2026-04-22** — follow-up v2 bench was added to the public
> repo under `v2_3090_followup/` after an HF discussion comment
> questioned `--draft-min 48` aggressiveness and the 100 % acceptance
> number. v2 covers `--draft-min 2 --draft-max 32` and several control
> configs, with a cross-check against current master `bcb5eeb64`
> (post PR #22227 speculative-simple checkpoint) — same results.
> Conclusion holds. See `v2_3090_followup/SUMMARY.md` in the repo.
>
> **UPDATE 2026-04-25 — cross-engine first attempt (SUPERSEDED 2026-04-26).** I originally tested the same
> model on vLLM 0.19.1 with `--speculative-config method=mtp num_speculative_tokens=1`
> (qwen3.6's built-in MTP heads) on 2× RTX 3090 TP=2 and reported mean **111 tok/s vs no-MTP 126** (−12 %),
> framing it as cross-engine corroboration that the regression is a model + hardware property. **That
> framing was wrong.**
>
> **UPDATE 2026-04-26 — v3 clean A/B retest reverses the vLLM sign.** The −12 % had two confounders that
> I controlled in v3: (a) flag mismatch (MTP run used `--gpu-memory-utilization 0.80 --max-num-seqs 2`
> while no-MTP baseline used `0.90 / 8`), and (b) `--enable-prefix-caching` ON in both runs, where MTP
> has a known interaction that drops cache hit rate ~92 % → ~71 % per [vllm #38182](https://github.com/vllm-project/vllm/issues/38182).
> With matched flags `0.90/8` AND `--no-enable-prefix-caching`, vLLM MTP k=1 on the same 2× RTX 3090
> hardware is **+27.5 % faster decode rate** (decode TPOT 7.620 ms → 5.976 ms, N=5 trials × 5 prompts).
> So **the regression in this PR's data is engine + spec-method specific to llama.cpp draft-spec, NOT engine-independent**.
> The MoE expert-saturation argument still explains why llama.cpp's K=5–64 draft-then-verify path loses
> on consumer Ampere (verify pass loads union of K positions' expert sets at K ≪ T_thres ≈ 94), but does
> not generalize to vLLM MTP k=1 which uses structurally smaller K and a lighter-weight verify path that
> reuses target hidden states. Full v3 data + reproducer:
> [thc1006/qwen3.6-vllm-2x3090 v3.0](https://github.com/thc1006/qwen3.6-vllm-2x3090/releases/tag/v3.0).

---

Posting a 19-config spec-decode matrix on a single RTX 3090 (SM 8.6, 24 GB, driver 580.126, CUDA 12.6) for `Qwen3.6-35B-A3B-UD-Q4_K_XL` at commit `9789512` (post-#19493 merge, pre-#20075 land). Happy to get feedback / be pointed at a config I missed.

**Summary:** none of the spec-decode modes — `ngram-cache`, `ngram-mod` (including srogmann's recommended `n=24 --draft-min 48 --draft-max 64`), and classic `--model-draft` with the **correct-vocab** `Qwen3.5-0.8B` (vocab 248320 matching target) — achieve a net speedup over the non-speculative baseline of **135.7 tok/s**. Every variant lands at **116–133 tok/s** mean and hits a **bimodal tail of 59–67 tok/s** on reasoning / code prompts despite **100 % draft acceptance**.

```
config                   mean   min    std    draft_accept
baseline                 135.7  135.3  0.3    —
ngmod-n32                133.7  133.5  0.1    0/0 (never triggered)
ngmod-n8/12/16/20/24     129–131 120–130 2-5  100%
ngcache-kv-fp16          121.3  67.3   27.6   100% (88/88)   ← fp16 KV does not rescue
draft-q35-08b-max{8,16,32} 120–121 59–65 ~30  100% (~270/270)
draft-q35-08b-1000tok    120.2  64.8   28.3   100% (937/937)
ngram-cache              119.1  65.3   27.8   100% (96/96)
ngcache-1000tok          115.9  60.0   28.7   100% (317/317)
```

Controls:

- `baseline-rerun` gives 135.5 / `ngcache-rerun` gives 118.8 — the regression is reproducible, not jitter (std within a config ≤ 0.4 for baseline).
- Switching **`-ctk q8_0 -ctv q8_0` → fp16 KV** leaves `ngram-cache` at 121 tok/s mean — KV quant is not the cause.
- Stretching output from **300 → 1000 tokens** leaves every ratio unchanged — output length is not the cause.
- Classic draft was initially tried with `qwen3:0.6b` (vocab 151936) which produced `failed to create draft context` in the server log — flagging this as a gotcha for others trying to reproduce, since the Qwen3 → Qwen3.5 tokenizer changed vocab.

Interpretation (open to pushback): the pattern matches [MoESD, arXiv 2505.19645](https://arxiv.org/html/2505.19645) and [Utility-Driven SD, arXiv 2506.20675](https://arxiv.org/pdf/2506.20675) — with A3B (3 B active, 8-of-256 routed), the expert-saturation threshold `T_thres ≈ 94` is well above any realistic draft `K`, so each drafted token brings in a fresh expert slice through the memory hierarchy. Verification then pays for the union of those slices. On a bandwidth-bound consumer GPU this eats the savings that would come from skipping per-token forward passes. srogmann's own benchmark on Qwen3.5-**122B-A10B** (10 B active) in PR #20075 already shows a positive +15–45 % speedup, which is consistent with that model being above the threshold — so the fix is working as intended on A10B+, not regressing on A3B.

Reproduction, raw JSON per request, three plots, and the exact `run_*_matrix.sh` are at [github.com/thc1006/qwen3.6-speculative-decoding-rtx3090](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090). I can also run additional configs if useful (e.g. once #20075 lands, re-run to see whether the bimodal tail disappears on Ampere).

Hardware / env snapshot is in `BENCHMARK_ENV.md` in the repo so anyone should be able to reproduce exactly.
