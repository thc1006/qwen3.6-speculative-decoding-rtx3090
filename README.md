# Historical benchmark: llama.cpp speculative decoding for Qwen3.6-35B-A3B UD-Q4_K_XL on one RTX 3090

[![DOI](https://zenodo.org/badge/1216484498.svg)](https://doi.org/10.5281/zenodo.19776558)

> [!IMPORTANT]
> **Audited 2026-08-25.** This repository preserves benchmark runs collected
> between 2026-04-21 and 2026-05-07. The primary result is a historical,
> single-request decode microbenchmark for the exact model files, llama.cpp
> commits, hardware, prompts, and flags listed below. It is not a benchmark of
> current llama.cpp master, of all RTX 3090 systems, of all Qwen3.6
> quantisations, of all speculative-decoding methods, or of end-to-end
> voice-agent latency.
>
> The audit **retracted this repository's headline mechanism.** Earlier
> versions reported "100 % draft acceptance yet slower, therefore MoE
> expert-loading overhead". That 100 % is an artefact of how llama.cpp counts
> acceptance on this model class, not a measurement — see
> [The "100 % acceptance" retraction](#the-100--acceptance-retraction). Three
> further defects turned up that no earlier version noticed: the draft model
> was never actually vocabulary-compatible, three quarters of the v1 requests
> returned truncated thinking rather than answers, and `llama-server` plus a
> draft model aborts on this model at `bcb5eeb64`. Every corrected item, with
> the evidence that settles it, is in [`ERRATA.md`](ERRATA.md); the work queue
> that closes what is still open is [`RETEST_TODO.md`](RETEST_TODO.md).
>
> The narrow negative observation survives all of it — and, after the
> vocabulary defect was fixed and re-measured, it is now on firmer ground than
> when it was published.

## Result in one sentence

On one RTX 3090, at llama.cpp commit
`97895129e5f2bde94d13dc01ca41ee79e9b629f2`, with
`Qwen3.6-35B-A3B-UD-Q4_K_XL`, greedy decoding, and the ten committed prompts,
every tested condition that recorded speculative activity had lower
request-mean **and** lower pooled decode throughput than its matched
no-speculation reference.

The direction holds *for the conditions v1 tested*. The *explanation* published
alongside it does not. Re-run on a binary where llama.cpp counts acceptance
correctly, real acceptance and decode rate correlate at **r = +0.998** across
the ten prompts — the slowdown tracks low acceptance and draft-path cost, which
is ordinary speculative-decoding economics. The "100 % acceptance yet slower,
therefore an MoE pathology" anomaly this repository was built around does not
exist.

**And the direction is not universal.** On 2026-08-26, on the same card and
prompt set with post-merge master, DFlash self-speculation at a short draft
window beats a matched no-speculation control, winning on all ten prompts.
Measured six times across three runs and two memory configurations, the gain is
**+16 % to +19 % on aggregate throughput and +21 % to +24 % pooled**. The
single best-powered measurement — five repeats rather than three — is +21.1 %
pooled at `--spec-draft-n-max 2`; the largest, and the one an unfriendly reader
should attack first, is +18.7 % aggregate at `n_max 4` in run J, which has three
repeats and sits at the top of the range. v1 never tested that method, and the archived v3 attempt at it compared
two different binaries. The sign flips with the draft window — +18.7 % at 4,
−14.8 % at 8, −47.4 % at 16 — so "speculative decoding loses here" was a
statement about draft-window regimes that this repository had not yet separated.

One qualification travels with that number, and with every speculative
measurement here: **speculation is not output-preserving on this build.** The
engine is deterministic — every arm reproduces itself byte-for-byte across
repeats, and the no-speculation baseline reproduces across separate runs — and
against that control, turning speculation on changes the generated text in 27 to
30 of 30 request-pairs. All arms still emit exactly 300 tokens and the baseline's
decode rate varies only 0.8 % across ten very different prompts, so the
throughput comparison stands; but this is a faster computation landing on
slightly different text, not a lossless speedup of the same one
([ERRATA A11](ERRATA.md#a11-speculative-decoding-is-not-output-preserving-on-this-build-and-the-engine-is-deterministic-enough-to-prove-it)).
See [`v4_audit_2026_08_25/README.md`](v4_audit_2026_08_25/README.md#run-j--the-first-configuration-that-is-actually-faster).

The one lever upstream names as the fix — batching — was also tested, and does
not help: no speculation gains +64 % at concurrency 8 while the matched-vocabulary
drafter moves −8 %, so the gap widens rather than closing
([run I](v4_audit_2026_08_25/README.md#run-i--batching-the-lever-upstream-names)).
It does not rescue the winner either. A sweep down to `n_max 1` puts DFlash on a
plateau — +17.1 %, +17.6 %, +17.3 % at 2, 3 and 4, separated by less than the
baseline's own run-to-run SD — and a cliff between 4 and 6; batching then erases
the plateau at four concurrent requests (+0.4 %) and collapses it at eight
(−74.1 %), with draft volume and acceptance barely moving, so it is the draft
*cost* that fails to amortise
([run K](v4_audit_2026_08_25/README.md#run-k--where-the-optimum-is-and-what-batching-does-to-it)).
On this card, speculation pays for one stream at a time or not at all.

And the win belongs to the workload rather than to the method. Repeated with
thinking verifiably off on all 250 requests, `n_max 2` falls from +21.1 % to
+7.6 % and `n_max 4` goes **negative** at −2.7 %, tracking draft acceptance down
with it (72.8 % → 58.5 %, 55.6 % → 40.3 %). Per prompt, step-by-step arithmetic
and Python keep their full gain — their output is constrained and stays ~85–90 %
accepted — while Traditional Chinese free prose goes from +15 % to −25 % as
acceptance falls from 66 % to 29 %. Reasoning text is enumerated, repetitive
planning prose, which is exactly what a drafter predicts well
([run L](v4_audit_2026_08_25/README.md#run-l--the-win-is-a-property-of-the-workload-not-of-the-method)).

Across run L's 60 points acceptance and speed-up correlate at **r = +0.946** and
the line crosses zero near **48 % acceptance**. Pushed out of sample at runs J
and K it calls the **sign 10 times out of 10** and the magnitude badly wrong —
worst error +52.2 pp — so the defensible form is the conservative one: a
configuration here is worth running when it clears roughly 48 % draft
acceptance, and how much it is worth also depends on draft volume, which
acceptance alone does not carry.

![v1 300-token matrix: request-mean vs pooled throughput](analysis/plot_mean_by_config.png)

---

## Experiment registry

These experiments differ in host, tool, commit, sampling, and prompt set. They
are not a cumulative body of evidence for one hypothesis and must not be pooled.

| ID | Date | Host / runner | Design | Evidence level |
|---|---|---|---|---|
| **v1 primary matrix** | 2026-04-21 | one GPU of the dual-3090 `s1` host; `llama-server`; commit `9789512` | 19 run labels (14 recorded draft rounds, 5 did not); 10 prompts; one measured request per prompt/config; `temperature=0`; 300-token cap unless noted | Primary descriptive result. No repeated trials per cell. |
| **v2 follow-up** | 2026-04-22 | different single-3090 host; `llama-cli`; commits `9789512` and `bcb5eeb64` | 5 prompts; `temperature=0.5`; 200-token cap; different runner and host | Directional check, not a controlled replication of v1 absolute rates. Thinking control did not work ([D1/D2](ERRATA.md#d1--d2--no-cnv-was-rejected-and-no_think-did-not-disable-thinking)). |
| **Exp 2 code/JSON** | 2026-04-25/26 | v2 host; `llama-cli` at `bcb5eeb64` | 5 prompts × 3 trials × 3 configs | Exploratory only. Intended workload unverified and per-request outputs not committed ([D3](ERRATA.md#d3-exp-2-cannot-be-audited-so-it-cannot-refute-anything)). |
| **v3 DFlash** | 2026-05-07 | v2 host; `llama-cli` | 5 prompts × 1 run × 3 draft-max settings | Exploratory only. Baseline and treatment used **different binaries** ([D4](ERRATA.md#d4-v3-dflash-compares-two-different-binaries)). |

| **v4 audit** | 2026-08-25/26 | one RTX 3090 (`3090` host); `llama-server` at `bcb5eeb64` and `3737e4137` | runs A–L; ABBA arm order, 3–5 repeats per arm, per-request JSON with full text and token ids, continuous GPU telemetry, pre-registered predictions | The controlled tier. Each run carries its own matched no-speculation baseline. |

The v4 runs, and what each one is for:

| run | question | design |
|---|---|---|
| A / B | does the archive reproduce, and does the abort persist? | `bcb5eeb64` vs post-merge `3737e4137`, 30 requests each |
| C / D | thirteen arms, thinking on and verifiably off | 13 arms × 10 prompts × 3 repeats, twice |
| E | is there anything past MoESD's 95-token coverage threshold? | `n_max` 64 / 96 / 128 |
| H | is `p_min` the lever, not draft length? | `p_min` 0 / 0.50 / 0.75 / 0.90 sweep |
| I | does batching rescue speculation, as upstream says it should? | concurrency 1 / 4 / 8, batch width verified from timestamps |
| J | DFlash off vs on, one binary — the A/B v3 never had | 5 arms × 3 repeats, `-fit on` on every arm |
| K | where is the draft-length optimum, and does it survive batching? | `n_max` 1–8 sweep, then the winner at concurrency 4 / 8 |
| L | does the win survive the workload changing? | same 5 arms twice, thinking on and off, 5 repeats |

The v2 / Exp 2 / v3 files remain valuable archival evidence. Their absolute
rates and their causal interpretations must be read inside those limits.

---

## v1 hardware, software, and artefacts

- **GPU used by the benchmark process** — RTX 3090 24 GiB, `CUDA_VISIBLE_DEVICES=1`, SM 8.6.
- **Physical host** — two RTX 3090s, Intel Core i7-11700, 62 GiB RAM, Ubuntu 24.04.4, kernel 6.17. GPU 0 was deliberately left to an Ollama instance; the benchmark process had one card to itself, the host did not, and no continuous utilisation trace was captured ([C4](ERRATA.md#c4-gpu-0-was-running-another-workload)).
- **Driver** — NVIDIA 580.126.09. `nvidia-smi` reports driver support for CUDA 13.0; llama.cpp was built with the **CUDA 12.6** toolkit. These are different things.
- **llama.cpp** — `97895129e5f2bde94d13dc01ca41ee79e9b629f2` (short `9789512`), authored 2026-04-20, post PR #19493.
- **Target** — `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (~21 GiB), from [`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF).
- **Classic draft model** — `Qwen3.5-0.8B-Q4_K_M.gguf` (~508 MiB), from [`unsloth/Qwen3.5-0.8B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF).
- **Fixed server flags** — `-ngl 999 -c 16384 --jinja -fa on -ctk q8_0 -ctv q8_0 --no-webui`.
- **Sampling** — greedy, `temperature=0`.
- **Warm-up** — one 8-token completion before each config's prompt sequence. This is not a full-shape warm-up.
- **Config execution** — server restarted between configs, so KV and prompt-cache state does not bleed across configs.
- Full snapshot: [`BENCHMARK_ENV.md`](BENCHMARK_ENV.md).

Expected SHA-256 of the v1 model files:

```
707a55a8a4397ecde44de0c499d3e68c1ad1d240d1da65826b4949d1043f4450  Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517  Qwen3.5-0.8B-Q4_K_M.gguf
ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a  Qwen3-0.6B-Q4_K_M.gguf
```

> [!WARNING]
> **Equal vocabulary size is not vocabulary compatibility.** Both
> Qwen3.6-35B-A3B and Qwen3.5-0.8B declare `vocab_size = 248320`, and this
> repository previously called the pair "vocab-matched". llama.cpp disagreed:
> `common_speculative_are_compatible` fails its **special-token** gate and
> enables the token-translation fallback, so every classic-draft round
> detokenised the context to a string and re-tokenised it for the draft model.
> All classic-draft rows below were measured on that path.
>
> The cause is one missing GGUF key. `Qwen/Qwen3.5-0.8B` has no
> `generation_config.json` upstream (HTTP 404), so the converter wrote no
> `tokenizer.ggml.bos_token_id`, and llama.cpp substitutes the hard-coded
> GPT-2 legacy default `11` against the target's `248044`. Both models declare
> `add_bos_token = false`, so the gate compares a field neither model uses.
>
> **It was fixed and re-measured on 2026-08-25, and it does not explain the
> slowdown:** `--override-kv tokenizer.ggml.bos_token_id=int:248044` flips the
> gate, and `long_explain` moves from 48.4 tok/s to 50–51 tok/s against a
> ~126 tok/s baseline. See
> [A2](ERRATA.md#a2-the-draft-model-was-not-vocabulary-compatible-the-run-used-the-token-translation-fallback).

---

## Metric definitions

llama-server reports, per request,
`predicted_per_second = 1000 × predicted_n / predicted_ms`. `predicted_ms`
covers target decoding plus the speculative proposal, verification, and
bookkeeping. It excludes prompt prefill, queueing, network transport, speech
I/O, and application latency.

Three summaries are reported for every config:

- **Request-mean decode rate** — arithmetic mean of each request's
  `predicted_per_second`. Every prompt weighs the same.
- **Pooled decode throughput** — `1000 × Σ predicted_n / Σ predicted_ms`. Every
  generated token weighs the same. For equal-length outputs this is the
  harmonic mean of the per-request rates.
- **Min–max across prompts** — workload heterogeneity. One measurement per
  prompt/config, so it is **not** repeated-run uncertainty, a standard error,
  or a confidence interval.

`draft_n` and `draft_n_accepted` need their own definition, because they do not
mean what their names suggest — see the next section.

---

## The "100 % acceptance" retraction

**Earlier versions of this README said:** every tested configuration returned
100 % draft acceptance, so "high acceptance → high speedup" fails, and "this is
not a measurement artifact; it is MoE expert-loading overhead on every drafted
token."

**It is a measurement artefact.** On this model the ratio can only ever be 1.0.

![What the 100 % acceptance number actually counts](analysis/plot_acceptance_accounting.png)

Qwen3.6-35B-A3B is a hybrid Gated-DeltaNet / MoE model, so
`common_context_can_seq_rm()` returns `COMMON_CONTEXT_SEQ_RM_TYPE_FULL` — the
context cannot roll back part of a sequence. In `server-context.cpp` at the
tested commit, a partially accepted draft therefore takes an early `continue`
that restores a state checkpoint and skips **both** acceptance counters, then
re-verifies the truncated, already-known-accepted prefix on the next pass. Only
rounds accepted in full ever reach the counters.

The drafter's own counters are printed on the next line of the same log and say
something different:

| counter | value | what it counts |
|---|---:|---|
| server `draft_n_accepted / draft_n` — the published number | **115 / 115 = 100.0 %** | tokens re-verified after truncation to the accepted prefix |
| drafter `#acc tokens / #gen tokens` | **115 / 214 = 53.7 %** | true token-level acceptance |
| drafter `#acc drafts / #gen drafts` | **33 / 81 = 40.7 %** | true draft-sequence acceptance |

Over that run: 53 verification attempts, 33 fully accepted, **20 partially
accepted and thrown away**. Rebuild the table yourself with
`python analysis/verbose_accounting.py`.

**Consequences for reading this repository's data.** In
[`analysis/summary.csv`](analysis/summary.csv), `draft_n` means "draft tokens in
verification rounds that were accepted in full", a quantity guaranteed to equal
`draft_n_accepted`. `draft_n = 0` means "no fully accepted round was recorded",
**not** "speculation did not run". The retracted
`analysis/plot_accept_vs_speed.png` — every one of whose 140 points sat at
exactly 100 % — has been deleted.

**And the original intuition survives, on honest evidence at last.** The
published claim was "100 % acceptance yet slower". That 100 % was the artefact
above. But sweeping `--spec-draft-p-min`, which truncates a draft once the
drafter's confidence drops, produces genuine high-acceptance configurations:

| configuration | real acceptance | pooled tok/s | vs baseline |
|---|---:|---:|---:|
| no speculation | — | 123.8 | — |
| `n_max` 8, `p_min` 0.75 | 80.2 % | **42.8** | **−65.5 %** |
| `n_max` 8, `p_min` 0.90 | **88.2 %** | 42.5 | −65.6 % |
| `n_max` 8, `p_min` 0 | 29.7 % | 32.7 | −73.6 % |

**Nine drafted tokens in ten accepted, correctly counted, and still nearly three
times slower than not speculating.** The intuition was right; the evidence for
it was not, and the MoE conclusion drawn from it was an overreach. `p_min` also
turns out to be the knob that matters rather than `n_max`: at `p_min` 0.75,
`n_max` 32 and `n_max` 128 are byte-identical — 6159 drafted tokens, 70.9 %
acceptance, 42.0 tok/s each. See
[A10](ERRATA.md#a10-the-single-regressor-law-is-falsified-out-of-sample-and-p_min-is-the-lever-that-matters).

**What the same log does support.** It contains a direct cost decomposition
that nobody in the earlier write-ups used. For 200 tokens generated at
63.2 tok/s (≈ 3165 ms):

| term | measured |
|---|---:|
| drafter `generate()` time | **999.6 ms ≈ 31.6 % of the generation wall-clock** |
| speculative checkpoints created | 33 × 62.8 MiB = 2.02 GiB written |
| checkpoints restored after a partial accept | 20 = 1.23 GiB read back |
| verification rounds discarded and redone | 20 of 53 (37.7 %) |

About a third of generation time is the draft model alone, and about 38 % of
verification rounds are paid for twice. That accounts for a slowdown of this
size without any appeal to expert-union loading.

---

## v1 representative results

Ten 300-token requests per config, one measurement each. Deltas are against the
matched no-speculation reference. Descriptive only.

| condition | request-mean | pooled | median | min | requests with a counted draft round |
|---|---:|---:|---:|---:|---:|
| baseline | 135.7 | 135.7 | 135.6 | 135.3 | 0 / 10 |
| baseline-rerun | 135.5 (−0.1 %) | 135.5 (−0.1 %) | 135.4 | 135.1 | 0 / 10 |
| draft-qwen3-0.6b *(vocab 151936, draft never attached)* | 135.3 (−0.3 %) | 135.3 (−0.3 %) | 135.3 | 135.0 | 0 / 10 |
| ngmod-n32 | 133.7 (−1.5 %) | 133.7 (−1.5 %) | 133.6 | 133.5 | 0 / 10 |
| ngram-mod-n24 | 131.1 (−3.4 %) | 131.1 (−3.4 %) | 130.0 | 129.6 | 8 / 10 |
| ngmod-n20 / n16 / n8 / n12 | 129.6 – 130.1 (−4.2 to −4.5 %) | 129.5 – 130.0 (−4.2 to −4.6 %) | 129.1 – 132.0 | 119.8 – 128.8 | 7–9 / 10 |
| ngcache-kv-fp16 *(one-sided control, see below)* | 121.3 (−10.6 %) | **113.7 (−16.2 %)** | 137.8 | 67.3 | 3 / 10 |
| draft-q35-08b-max8 | 121.1 (−10.8 %) | **109.9 (−19.0 %)** | 135.6 | 59.2 | 2 / 10 |
| draft-q35-08b-max16 | 121.0 (−10.9 %) | **110.3 (−18.7 %)** | 135.2 | 59.6 | 2 / 10 |
| draft-q35-08b-max32 | 120.3 (−11.4 %) | **110.0 (−19.0 %)** | 134.1 | 59.5 | 2 / 10 |
| ngram-cache | 119.1 (−12.2 %) | **111.3 (−18.0 %)** | 135.6 | 65.3 | 3 / 10 |
| ngcache-rerun | 118.8 (−12.4 %) | **111.1 (−18.1 %)** | 135.0 | 65.6 | 3 / 10 |

Long-output variants use their own reference, `baseline-1000tok` (request-mean
133.2, pooled 133.1):

| condition | request-mean | pooled | note |
|---|---:|---:|---|
| ngmod-n24-1000tok | 131.1 (−1.6 %) | 131.1 (−1.5 %) | |
| draft-q35-08b-1000tok | 120.2 (−9.7 %) | **106.1 (−20.3 %)** | |
| ngcache-1000tok | 115.9 (−13.0 %) | **98.9 (−25.7 %)** | worst pooled result in the matrix |

Full per-request values: [`analysis/summary.csv`](analysis/summary.csv).
Per-config aggregate with all four summaries and both references:
[`analysis/summary_by_config.csv`](analysis/summary_by_config.csv).

### Per-prompt structure

![per-prompt decode rate, normalised to the matched baseline](analysis/plot_per_prompt.png)

Black outlines mark requests that recorded at least one fully accepted draft
round. The picture is not the "chat prompts never trigger, structured prompts
collapse" taxonomy this README used to assert:

- **ngram-mod** records draft rounds on the *chat* prompts — `short_q`,
  `medium_chat`, `medium_rec`, `multi_turn_1`, `multi_turn_2`, `zh_hant` — and
  **not** on `code_small` at n = 24. Its cost is a flat ~4 % everywhere.
- **classic draft** records rounds only on `long_explain` and `code_small`, and
  leaves `reasoning` at full baseline speed.
- **ngram-cache** records rounds on `reasoning`, `long_explain`, `code_small`.

Three configurations, three different prompt partitions. Ten hand-written
prompts cannot establish a prompt taxonomy, and the earlier "entirely bimodal
by prompt class" sentence was wrong ([B5](ERRATA.md#b5-the-regression-is-entirely-bimodal-by-prompt-class-is-false-for-the-ngram-mod-family)).

The heatmap also exposes why `ngcache-kv-fp16` is a **one-sided control**: on
the seven prompts with no draft round it runs at 101–102 % of the q8_0
baseline, because fp16 KV is simply faster when speculation is idle. There is
no no-speculation fp16-KV row in the matrix, so that condition cannot separate
a speculation effect from a KV-precision effect, and the old reading — "fp16 KV
does not rescue, so KV quant is not the cause" — does not follow.

---

## What the audit measured on 2026-08-25

Three new measurements were taken on the original v2/v3 bench host. They are
recorded here because two of them bear directly on how the archived data should
be read.

Full data and method: [`v4_audit_2026_08_25/`](v4_audit_2026_08_25/).

**With acceptance measured properly, the anomaly disappears — and the contrast
has to be named.** Upstream has since made the partial-accept path reachable,
so the counter reports real ratios. Two comparisons are available and they
point opposite ways; reporting either without saying which would repeat exactly
the mistake this audit corrects.

*Within one configuration, across prompts*, prompts the drafter predicts well
run faster, almost exactly in proportion. On post-merge master `3737e4137`,
5 repeats of a 13-arm matrix, **six distinct draft lengths each reproduce this
at r ≥ +0.996**, spanning acceptance from 5 % to 83 %. A control makes it hard
to explain away: with no speculation the prompt barely matters, `baseline`
spanning only 1.4 % across the same ten prompts.

| `--spec-draft-n-max` | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| Pearson r | +0.998 | +0.999 | +0.996 | +0.999 | +0.999 | +0.999 |
| pooled tok/s | 31.1 | 34.2 | **35.6** | 32.1 | 23.7 | 17.3 |
| acceptance | 68.7 % | 60.3 % | 45.4 % | 29.7 % | 15.0 % | 8.0 % |

*Across configurations* the sign flips to r = −0.544, because a configuration
that drafts harder achieves higher acceptance per attempt while paying for far
more drafted tokens. An external 0.8 B drafter proposing 0.50 tokens per
generated token runs at 31.1 tok/s while ngram-cache proposing 0.42 runs at
74.0, so volume is not the whole story either: the per-round cost of the
drafting method is a third, independent term.

**The slowdown is the per-round cost of drafting, times how much is drafted,
divided by how much is accepted.** That is ordinary speculative-decoding
economics. There is no anomaly left, and **no MoE-specific pathology is needed**
— this repository never had evidence for one, and the sweep was extended past
the threshold to check rather than assert. Across `n_max` 1 → 128, spanning
**3.1 % to 98.3 % expected routed-expert coverage**, a single regressor accounts
for the cost:

```
ms per generated token = 27.00 + 4.040 × (draft tokens per generated token)
R² = 0.99303
```

**The step in the residuals at the 95.3 % coverage point is −0.39 percentage
points.** No knee, no break, nothing for a coverage threshold to explain.
Throughput peaks at `n_max` 4 and declines monotonically straight through.
Full detail:
[A7](ERRATA.md#a7-with-acceptance-measured-properly-there-is-no-anomaly-left-to-explain).

**The vocabulary defect is real but is not the cause.** Same binary, same draft
file, same flags, only the BOS override differing:

| binary | arm | request-mean | drafted / accepted |
|---|---|---:|---|
| `bcb5eeb64` | translation fallback, as published | 113.9 | 194 / 194 |
| `bcb5eeb64` | matched via `--override-kv` | 113.5 | 194 / 194 |
| master `3737e4137` | translation fallback | 33.6 | 16590 / 4926 |
| master `3737e4137` | matched via `--override-kv` | 33.7 | 16590 / 4926 |

The drafted and accepted totals are byte-identical across arms on both
binaries, so the translation path was not changing what got drafted. The
negative finding survives, now on a matched path.

**`llama-server` plus a draft model aborts on this model at `bcb5eeb64`.**
Reproducibly, 3 / 3 on the `code_small` prompt, in both arms, immediately after
a partial-accept checkpoint restore:

```
CUDA error: an unsupported value or parameter was passed to the function
  in function ggml_cuda_op_mul_mat_cublas
  #14 server_context_impl::update_slots()
```

The no-speculation arm completes all ten prompts every time. This means v2's
"cross-checked on master `bcb5eeb64`, identical results" is a `llama-cli`
cross-check only — the `llama-server` path that produced every v1 number does
not survive on that commit with a draft attached. It is **fixed on post-merge
master**: all thirty requests complete on `3737e4137`. See
[A6](ERRATA.md#a6-llama-server-plus-a-draft-model-aborts-on-this-model-at-bcb5eeb64).

**Workload shape matters, and it was never controlled.** The audit ran the
comparison Exp 2 was trying to run — the same arms with thinking verifiably on
and verifiably off, 5 repeats each, `thinking_suppressed` recorded per request:

| method | thinking on | thinking off | draft tokens per generated token |
|---|---:|---:|---|
| `ngram-mod` n=24 | −6.8 % | **−0.7 %** | 0.21 → **0.00** |
| `ngram-cache` | −40.0 % | −32.6 % | 0.42 → 0.36 |
| draft model, n_max 8 | −74.0 % | −76.4 % | 1.85 → **2.14** |

With thinking off `ngram-mod` stops drafting entirely and its cost nearly
vanishes; a chain-of-thought trace is the repetitive text an n-gram lookup
feeds on, a direct answer is not. For the draft model the effect reverses —
acceptance falls from 29.7 % to 23.0 %, so reasoning traces are *easier* for a
0.8 B drafter than real answers.

What that implies splits by family, so it cannot be said in one sentence.
Draft-model speculation was measured on its *favourable* workload — thinking
traces are easier to predict and give a better net result — and still lost, so
that finding is more robust than when it was published. ngram methods were
measured on their *unfavourable* one: thinking gives an n-gram lookup much more
to fire on, and firing costs more than it returns, so the historical ngram
figures **overstate** the cost a real-answer workload would produce. Neither
becomes a net win. See
[D3b](ERRATA.md#d3b-workload-shape-does-matter--and-exp-2-pointed-the-wrong-way).

**Three quarters of v1's requests returned no answer.** `message.content` is
empty for 144 of 190 v1 requests, and 19/19 for both `reasoning` and
`code_small` — the 300-token cap was reached inside the thinking block, and
`reasoning_content` was never captured. v1 measured decode throughput on
truncated chain-of-thought, not on answers. See
[A5](ERRATA.md#a5-three-quarters-of-v1s-requests-returned-no-answer-at-all--only-truncated-thinking).

---

## What the v1 data support

- No tested condition that recorded speculative activity exceeded its matched
  no-speculation reference in aggregate, on this setup.
- The published acceptance ratio cannot be used to predict speedup, because it
  is not an acceptance measurement.
- Draft-side cost is large and directly measured: the drafter alone consumed
  ~32 % of generation wall-clock in the one run with a verbose trace, and ~38 %
  of verification rounds in that run were discarded and repeated.
- Deep slow tails are real but config-specific and prompt-specific: 59–67 tok/s
  for particular ngram-cache and classic-draft requests, while the whole
  ngram-mod family stays within 12 % of baseline at its worst.

## What the v1 data do not support

- Any universal statement about Qwen3.6, A3B models, RTX 3090s, consumer
  Ampere, or speculative decoding in general.
- That every run label performed speculation. Five recorded no draft round at
  all, including three baselines and an incompatible-draft control.
- That expert loading, memory bandwidth, quantisation, or any single kernel is
  the root cause. No expert-routing trace, HBM counter, kernel profile, or
  dense / FP16 / cross-GPU factorial control was collected.
- A clean classic-draft effect estimate, because those rows ran on the
  cross-vocabulary translation fallback. Re-measuring with the gate fixed
  recovers only 3–5 %, so the direction holds, but the archived absolute
  numbers are not a matched-path measurement
  ([A2](ERRATA.md#a2-the-draft-model-was-not-vocabulary-compatible-the-run-used-the-token-translation-fallback)).
- A claim about the workload, because 76 % of the requests returned truncated
  thinking rather than an answer, and no version of this benchmark ever
  verified what was being generated
  ([A5](ERRATA.md#a5-three-quarters-of-v1s-requests-returned-no-answer-at-all--only-truncated-thinking)).
- A statement about a *working* speculative path for this model class, because
  the fix for the hybrid-SSM partial-acceptance failure the runs actually hit —
  llama.cpp PR #20075 — was closed without merge
  ([A3](ERRATA.md#a3-the-tested-build-had-a-known-broken-speculative-path-for-this-model-class-and-the-fix-was-never-merged)).
- A production voice-agent recommendation. This measures decode
  microperformance, not streaming TTFT, audio latency, multi-turn cache reuse,
  concurrency, or output quality.
- Behaviour under non-greedy sampling, other seeds, current llama.cpp, or
  untested speculation methods.

---

## Follow-up experiment caveats

### v2 and Exp 2

Different host, `llama-cli` instead of `llama-server`, `temperature=0.5`,
200-token cap, and a different prompt set. Directional at best.

More importantly, the scripts pass `-no-cnv` and append `/no_think`, and the
committed logs show both are inert:

```
--no-conversation is not supported by llama-cli
please use llama-completion instead
```

That line appears in 61 of 62 v2 logs and 30 of 33 v3 logs, and those same logs
then contain `[Start thinking]` and a full reasoning trace. The measured
workload is long chain-of-thought output, not the intended direct answer.

Exp 2 committed only timing summaries — the per-request generated text, token
IDs, and stop reasons were never saved — so its intended "structured,
low-entropy, thinking-off code/JSON" distribution cannot be checked. **Exp 2
therefore does not refute the workload-shape hypothesis.** It shows that the
configurations, as actually executed, were slower for whatever the command
really generated. A valid retest needs `llama-completion` or a verified
chat-template thinking toggle, plus committed outputs.

One thing Exp 2 does establish cleanly: the command is highly repeatable. The
three trial means for the Oleg config are 66.54 / 66.54 / 66.64 tok/s,
SD 0.06 — the published `± 7.57` is spread *between prompts*, not run-to-run
noise.

### v3 DFlash

The historical v3 result was 77.0 tok/s for the best DFlash setting against
138.9 tok/s for its recorded baseline. Treat it as an exploratory datapoint,
not a DFlash effect estimate:

- baseline and Oleg logs report build `b8889-bcb5eeb64`; DFlash logs report
  `b8942-67cb0d507`. **Different binaries.**
- one run per prompt/config;
- the thinking control did not work;
- outputs were not token-identical between conditions;
- no target-precision, draft-precision, dense-model, profiler, or second-GPU
  control.

DFlash PR #22105 was merged upstream on 2026-06-28. That A/B was run on
2026-08-26, and it reverses the sign at short draft windows: on one binary with
a control matching to −0.01 %, `--spec-draft-n-max 4` is **+18.7 %** against no
speculation, while 8 and 16 are −14.8 % and −47.4 %. The archived v3 figure is
what the method looks like at the long windows, measured across a binary change.
Details and controls in
[`v4_audit_2026_08_25/README.md`](v4_audit_2026_08_25/README.md#run-j--the-first-configuration-that-is-actually-faster).

### The vLLM sibling result

[`thc1006/qwen3.6-vllm-2x3090`](https://github.com/thc1006/qwen3.6-vllm-2x3090)
reports a positive vLLM MTP result on the same physical hardware. It is a
useful counterexample to any engine-independent claim, and it is **not** a
matched cross-engine control: two GPUs, tensor parallelism, a different engine,
a different quantisation stack, built-in MTP heads, different flags, and a
different protocol. Cite it as related evidence; do not use it to decompose the
cause of the llama.cpp result.

---

## Candidate mechanism — largely settled, and it is not MoE-specific

The audit's re-measurement ([A7](ERRATA.md#a7-with-acceptance-measured-properly-there-is-no-anomaly-left-to-explain))
removes most of the mystery. Once acceptance is measured rather than assumed,
decode rate tracks acceptance at r = +0.998 across the prompt set. What remains
is ordinary: the drafter proposes tokens, most are rejected, and the run pays
for all of them.

The two candidate mechanisms below are kept because they quantify *how much*
each term costs — not because the outcome still needs an exotic explanation.

**1. Draft-path and state-management overhead.** This one is partly measured
(see the retraction section above): drafter time is ~32 % of generation
wall-clock, ~38 % of verification rounds are discarded and redone, and each
discarded round pays a 62.8 MiB state checkpoint plus its restore. On this
model that path is taken because the context cannot partially roll back a
sequence, which is precisely the defect llama.cpp PR #20075 described and which
was never merged. Nothing here isolates how much of the slowdown each term
contributes.

**2. MoE expert-union cost during multi-token verification — no longer needed,
and never evidenced here.** Kept only so the arithmetic this repository used to
publish can be checked. The official
Qwen3.6-35B-A3B config has 256 routed experts, 8 routed experts per token, plus
a shared expert. Under the i.i.d. uniform-routing approximation in
[MoESD](https://arxiv.org/html/2505.19645):

```
rho  = k_e / n_experts = 8 / 256 = 0.03125
T_95 = ceil( log(1 - 0.95) / log(1 - rho) ) = ceil(94.36) = 95 tokens
```

This is an expected-coverage heuristic under stated assumptions, not a
performance threshold, and it does not by itself explain anything measured
here. Use `γ` for draft length and `k_e = 8` for routed experts per token;
the earlier text used `K` for both.

Note that `Qwen3.5-122B-A10B` has the **same** 256 experts and top-8 routing,
so this formula gives it the **same** threshold. The earlier claim that A10B
has "a correspondingly lower `T_thres`" because of its larger active parameter
count is wrong — those are different quantities. Positive A10B measurements are
a genuine counterexample to any universal A3B-derived rule, but this repository
cannot say which factor explains the sign difference, because model, hardware,
backend, quantisation, draft configuration, and implementation all differ at
once.

**What would still be worth measuring** is the split *within* mechanism 1:
target verify time, draft time, accepted length per verification step, and
discarded-round count, across draft lengths. That is P3-1 in
[`RETEST_TODO.md`](RETEST_TODO.md). Testing mechanism 2 would need expert-routing
instrumentation that nobody here has built, and after A7 there is no result
demanding it.

---

## Reproduction

The historical scripts and raw files are all here, and every host-specific path
is now an environment variable with a documented default. They remain audit
artefacts rather than a one-command reproducer: the v1 warm-up is short, there
is one run per cell, and [D5](ERRATA.md#d5-the-committed-v2-script-does-not-produce-the-committed-v2-directories)
records a provenance gap in v2.

Build the **exact** v1 revision — not current master:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout 97895129e5f2bde94d13dc01ca41ee79e9b629f2
git submodule update --init --recursive

CUDACXX=/usr/local/cuda-12.6/bin/nvcc cmake -S . -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DLLAMA_CURL=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel --target llama-server llama-bench
```

Fetch the artefacts and verify them before benchmarking:

```bash
hf download unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --local-dir models
hf download unsloth/Qwen3.5-0.8B-GGUF --include '*Q4_K_M*' --local-dir models
sha256sum -c SHA256SUMS
```

Run the matrix and the analysis:

```bash
export LLAMA_SERVER_BIN=$PWD/llama.cpp/build/bin/llama-server
export MODEL_TARGET=$PWD/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
export MODEL_DRAFT=$PWD/models/Qwen3.5-0.8B-Q4_K_M.gguf
export BENCH_GPU=1                      # CUDA_VISIBLE_DEVICES for the server

bash run_matrix.sh          # baseline + ngram-cache + ngram-mod n=24 + 0.6B control
bash run_p0_matrix.sh       # classic-draft sweep + 1000-token variants + N sweep + kv-fp16

pip install -r requirements.txt
python analysis/plot.py
python analysis/verbose_accounting.py
```

A corrected harness for *new*, controlled runs — one pinned binary for every
arm, ABBA ordering, N repeats, a manifest that hashes the binary and both
models, and per-request capture of the generated text, the reasoning channel,
the stop reason, the full `timings` block, and token IDs via `logprobs` — is
[`bench/retest_runner.py`](bench/retest_runner.py). It is what produced the
audit measurements above. Note that llama.cpp renamed the speculative arguments
after `bcb5eeb64` (`--draft-max` → `--spec-draft-n-max`) and that `--spec-type`
now defaults to `none`, so `-md` alone loads a draft model and never
speculates; the runner's `BENCH_FLAVOR` switch handles both spellings.

---

## Data map

| Path | Contents |
|---|---|
| [`ERRATA.md`](ERRATA.md) | every corrected claim, with evidence |
| [`results/`](results/), [`results/verify/`](results/verify/) | v1 raw per-request JSON, 19 run labels |
| [`analysis/summary.csv`](analysis/summary.csv) | v1 flat per-request table |
| [`analysis/summary_by_config.csv`](analysis/summary_by_config.csv) | v1 aggregate: request-mean, pooled, median, min–max, activation |
| [`analysis/plot.py`](analysis/plot.py) | aggregation and charts |
| [`analysis/verbose_accounting.py`](analysis/verbose_accounting.py) | reconstructs the acceptance-counter artefact from a `-v` log |
| [`v2_3090_followup/SUMMARY.md`](v2_3090_followup/SUMMARY.md) | v2 methodology and tables |
| [`v2_3090_followup/v2_*/`](v2_3090_followup/) | 62 v2 raw `llama-cli` logs + one `--verbose` trace |
| [`v2_3090_followup/exp2_codejson_n3/`](v2_3090_followup/exp2_codejson_n3/) | Exp 2 aggregates and script |
| [`v3_dflash_2026_05_07/`](v3_dflash_2026_05_07/) | DFlash logs, tables, script |
| [`BENCHMARK_ENV.md`](BENCHMARK_ENV.md) | hardware, software, commits, hashes for v1/v2/v3 |
| [`bench/retest_runner.py`](bench/retest_runner.py) | corrected protocol, never executed |

---

## Upstream status at the 2026-08-25 audit

Checked against the GitHub API on 2026-08-25.

| PR | Status |
|---|---|
| [#19493](https://github.com/ggml-org/llama.cpp/pull/19493) server: speculative checkpointing | merged 2026-04-19 |
| [#22227](https://github.com/ggml-org/llama.cpp/pull/22227) speculative-simple: checkpoint support | merged 2026-04-22 |
| [#20075](https://github.com/ggml-org/llama.cpp/pull/20075) fix: speculative decoding broken on hybrid SSM/MoE | **closed without merge 2026-04-25** |
| [#22105](https://github.com/ggml-org/llama.cpp/pull/22105) DFlash support | merged 2026-06-28 |

`bcb5eeb64` was master on 2026-04-22 and is described here as a dated snapshot,
not as "current master". Future edits should keep using exact tested SHAs.

---

## Related reading

- [MoESD: Unveil Speculative Decoding's Potential for Accelerating Sparse MoE (arXiv 2505.19645)](https://arxiv.org/html/2505.19645)
- [Utility-Driven Speculative Decoding for Mixture-of-Experts (arXiv 2506.20675)](https://arxiv.org/pdf/2506.20675)
- [MoE-SpeQ (arXiv 2511.14102)](https://arxiv.org/html/2511.14102v1)
- [llama.cpp docs/speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
- [llama.cpp Issue #20039 — original feature request](https://github.com/ggml-org/llama.cpp/issues/20039)
- [HF discussion #14 on `unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/discussions/14) — the thread that prompted v2
- [vLLM Issue #38182 — Qwen3.5-35B-A3B MTP × prefix-cache interaction](https://github.com/vllm-project/vllm/issues/38182)
- [llama.cpp PR #18039, comment 3755925892](https://github.com/ggml-org/llama.cpp/pull/18039#issuecomment-3755925892) — the maintainer's **SGLang** cross-check of gpt-oss-120b + EAGLE3 on DGX Spark: 0.46–0.71× baseline at batch 1. Different engine, hardware, model and method; same direction. The strongest independent corroboration of this repository's negative finding, and it predates the audit. Its author also names batching, not draft length, as the lever.
- [llama.cpp PR #22105 (DFlash)](https://github.com/ggml-org/llama.cpp/pull/22105) — states the expert-activation effect for MoE targets and the extra target forward per rejected step on hybrid targets, with target-side deferred commit proposed to remove replay

### Open upstream issues in the same territory

Found during the audit. The pre-audit "validation timeline" cited papers and
unrelated issues; these are the same-class implementation reports, and several
concern this exact model family.

| Issue | Why it matters here |
|---|---|
| [#24055](https://github.com/ggml-org/llama.cpp/issues/24055) — context checkpoints always invalidated on hybrid/recurrent models | The checkpoint machinery this audit measured: 1639 checkpoints of 101.3 MiB for a single 300-token request |
| [#25004](https://github.com/ggml-org/llama.cpp/issues/25004) — recurrent: support equal splits for recurrent-state rollback | The rollback path behind [A1](ERRATA.md#a1-100--draft-acceptance-is-a-counter-artefact-not-a-measurement) and [A6](ERRATA.md#a6-llama-server-plus-a-draft-model-aborts-on-this-model-at-bcb5eeb64) |
| [#24670](https://github.com/ggml-org/llama.cpp/issues/24670) — draft-mtp not activating on Turing with a hybrid SSM+attention **Qwen3.6-35B-A3B** | This repository's exact target model |
| [#25117](https://github.com/ggml-org/llama.cpp/issues/25117) — DFlash regression on AMD APU with a **quantized MoE target**, ~2× slower than baseline | An independent report of v3's direction, on different hardware |
| [#27572](https://github.com/ggml-org/llama.cpp/issues/27572) — draft-mtp acceptance collapses to 0.0 under `-np N` | A known concurrency failure mode; any batching measurement must check acceptance did not collapse rather than assume it |
| [#27569](https://github.com/ggml-org/llama.cpp/issues/27569) — cap the draft context batch instead of inheriting the target's | Bears on long-draft configurations such as the `n_max` 128 arm |
- [`thc1006/qwen3.6-vllm-2x3090`](https://github.com/thc1006/qwen3.6-vllm-2x3090) — sibling repository, different engine and hardware topology

---

## Licence

- **Code and documentation** — MIT, see [`LICENSE`](LICENSE).
- **Benchmark data** — CC0-1.0, scoped by [`DATA_LICENSE`](DATA_LICENSE), full
  text in [`LICENSES/CC0-1.0.txt`](LICENSES/CC0-1.0.txt).

## Citation

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff). Cite the DOI
above for the archived release, and state which version you are citing —
pre-audit releases (v1.0 – v3.0) contain the claims retracted in
[`ERRATA.md`](ERRATA.md).

## Author

Hsiu-Chi Tsai (`thc1006`) · `hctsai1006@cs.nctu.edu.tw` ·
[github.com/thc1006](https://github.com/thc1006)
