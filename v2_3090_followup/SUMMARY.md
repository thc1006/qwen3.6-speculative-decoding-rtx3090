# Qwen3.6-35B-A3B · 3090 spec-decode follow-up bench (v2)

> [!WARNING]
> **Rescoped by the 2026-08-25 audit.** The measurements below are archived and
> every aggregate was re-derived from the raw logs and reproduced exactly. Three
> of the conclusions did not survive:
>
> 1. **"100 % draft acceptance is genuine" is wrong.** The ratio is 1.0 by
>    construction. On this hybrid Gated-DeltaNet target the context reports
>    `COMMON_CONTEXT_SEQ_RM_TYPE_FULL`, so a partially accepted round takes an
>    early `continue` in `server-context.cpp` that skips both acceptance
>    counters and re-verifies the truncated prefix. Only fully accepted rounds
>    reach the counter. The drafter's own counters, printed one line below the
>    quoted `1.00000 (115 accepted / 115 generated)` in the same
>    `verbose.log`, say **115 of 214 generated draft tokens were accepted —
>    53.7 %**, and 33 of 81 drafts — 40.7 %. See [`../ERRATA.md`](../ERRATA.md) A1.
> 2. **The draft model was not vocabulary-matched.** The same log shows
>    `common_speculative_are_compatible` failing and llama.cpp falling back to
>    token translation. Cause: `Qwen/Qwen3.5-0.8B` has no
>    `generation_config.json` upstream, so its GGUF carries no
>    `tokenizer.ggml.bos_token_id` and llama.cpp substitutes the hard-coded
>    GPT-2 legacy default `11` against the target's `248044`. Both models set
>    `add_bos_token = false`, so the field that gates speculation is one
>    neither model uses. See [`../ERRATA.md`](../ERRATA.md) A2.
> 3. **`/no_think` did not disable thinking and `-no-cnv` was rejected.**
>    61 of 62 logs in this directory contain
>    `--no-conversation is not supported by llama-cli` followed by
>    `[Start thinking]` and a full reasoning trace. The measured workload is
>    long chain-of-thought output. The real switches on this build are
>    `-rea off` / `--reasoning-budget 0`. See [`../ERRATA.md`](../ERRATA.md) D1/D2.
>
> A further provenance gap: the committed `bench_3090_oleg.sh` writes config
> directories named `02_srogmann_ngmod_n24` / `03_oleg_draft_2_32` /
> `04_oleg_draft_2_16`, while the committed data is `02_oleg_draft_2_32` /
> `03_oleg_draft_2_16` / `04_draft_2_64`. The script is therefore not the one
> that produced the logs, no script is committed for `v2_controls/` or
> `v2_master_cross_check/`, and no log records its own argv. See
> [`../ERRATA.md`](../ERRATA.md) D5.



In response to Oleg-dM's comment on HF discussion #14.

## Setup

- **Tested at two llama.cpp commits** (to rule out stale-commit artefact):
  - `97895129e` — original bench's commit (same as short-hash `9789512`)
  - `bcb5eeb64` — current master at time of bench, includes PR #22227
    `speculative-simple : add checkpoint support`
- RTX 3090 24 GB, single GPU, driver 580.126, CUDA 12.0
- GPU at **stock clocks** (graphics 1965 MHz current / 2100 max;
  memory 9751 MHz; power limit 350 W default — no overclocking)
- gcc 13.3.0, Ubuntu 24.04
- common flags: `-ngl 999 -c 16384 -fa on -ctk q8_0 -ctv q8_0 -n 200
  --temp 0.5 --seed 42 -no-cnv -st`
- 5 prompts spanning reasoning / code / factual / procedural / creative,
  with `/no_think` appended to disable Qwen3 reasoning for
  apples-to-apples tok/s measurement
- Draft model: `unsloth/Qwen3.5-0.8B-Q4_K_M.gguf` (508 MB,
  vocab **size** matched to Qwen3.6-35B-A3B - but llama.cpp's
  `common_speculative_are_compatible` rejects the pair and falls back to
  token translation, see [`../ERRATA.md`](../ERRATA.md) A2)

## Results on `97895129e` (tok/s, N=5 per config)

| Config | mean | min | max |
|---|---:|---:|---:|
| baseline (no spec-decode) | 139.9 | 139.7 | 140.0 |
| `-md --draft-max 8` (default `--draft-min=5`) | 56.5 | 51.5 | 63.0 |
| `-md --draft-max 16` (default `--draft-min=5`) | 55.7 | 53.3 | 62.7 |
| `-md --draft-max 32` (default `--draft-min=5`) | 55.3 | 52.9 | 63.1 |
| `-md` (full defaults) | 55.5 | 52.8 | 62.3 |
| **Oleg: `--draft-min 2 --draft-max 32`** | **65.0** | 61.0 | 75.8 |
| `-md --draft-min 2 --draft-max 16` | 66.3 | 60.6 | 76.6 |
| `-md --draft-min 2 --draft-max 64` | 64.7 | 60.6 | 75.3 |
| srogmann-style: `--draft-min 48 --draft-max 64` | **85.6** | 81.3 | 88.0 |

## Cross-validation on current master `bcb5eeb64`

Same config, same prompts, same hardware, same session:

| Config | `97895129e` | `bcb5eeb64` master | Δ |
|---|---:|---:|---:|
| baseline | 139.9 | 139.5 | −0.3 % (noise) |
| Oleg `--draft-min 2 --draft-max 32` | 65.0 | 65.2 | +0.3 % |
| srogmann `--draft-min 48 --draft-max 64` | 85.6 | 85.6 | 0 % |

**Master gives the same results** for the `llama-cli` path — PR #22227 does
not change *these* numbers. Note this is a `llama-cli` cross-check only: an
audit retest on 2026-08-25 found that `llama-server` + a draft model at
`bcb5eeb64` **aborts** with `CUDA error: an unsupported value or parameter`
in `ggml_cuda_op_mul_mat_cublas`, immediately after a partial-accept
checkpoint restore. See [`../ERRATA.md`](../ERRATA.md) A6.

## Conclusions

1. ~~**100 % `n_acc_tokens / n_gen_tokens` is genuine**~~ — **RETRACTED.**
   The source read landed on the right increment but the wrong question: that
   counter is only ever reached on fully accepted rounds. True acceptance in
   that same run is **115 / 214 = 53.7 %** of generated draft tokens and
   **33 / 81 = 40.7 %** of drafts. Reconstruct with
   `python analysis/verbose_accounting.py`.
2. **No draft-model spec-decode configuration beats baseline on this
   box.** Losses range from −39 % (srogmann recipe) to −60 % (default
   `--draft-min=5` with small `--draft-max`).
3. Oleg's `--draft-min 2 --draft-max 32` suggestion beats the default
   by ~10 tok/s (65 vs 55) but is still −54 % vs baseline 139.9.
4. **Aggressive draft windows (`--draft-min 48 --draft-max 64`) are
   the least bad** — contrary to the "wasted compute" intuition,
   larger draft windows amortise the verify / KV-management
   overhead enough to partially hide the cost.
5. The original bench's "mean 120 / bimodal tail 59" is a
   mixture of two regimes — v2's consistent 55–85 is the
   "spec-decode always active" regime isolated.

## Files

- `v2_oleg_suggestions/` — 4 configs × 5 prompts + `verbose.log`
- `v2_controls/` — 5 control configs × 5 prompts (A-E)
- `v2_master_cross_check/` — 3 configs × 5 prompts on master
  `bcb5eeb64`
- `bench_3090_oleg.sh` — reproducible script
