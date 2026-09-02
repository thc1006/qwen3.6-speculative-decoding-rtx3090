# v4 — audit measurements, 2026-08-25

New measurements taken during the audit, on the same physical host that
produced v2 and v3, and at the end the sections moved here from the root
README. They settle three questions the archived data could not:

1. Was the draft model's vocabulary incompatibility responsible for the
   slowdown? (**No.**)
2. Is the published "100 % draft acceptance" a real measurement? (**No** — and
   upstream has since fixed the counter, which proves it.)
3. Does the `llama-server` speculative path still work on this model?
   (**Not at `bcb5eeb64`. Yes on post-merge master.**)

Harness: [`../bench/retest_runner.py`](../bench/retest_runner.py), one pinned
binary per run, ABBA arm ordering, N repeats, and a manifest recording the
binary's sha256, both model sha256s, the complete argv, and GPU telemetry
before and after every arm. Per request it keeps the generated text, the
reasoning channel, the stop reason, the whole `timings` block, and token IDs
via `logprobs` (near-complete: `probs_output` drops trailing stop-word tokens,
`server-context.cpp:2036-2039`, so the list can run a few short of
`predicted_n`, which stays the authority for token counts).

> The runs on this page were taken **before** two harness defects were fixed,
> and their JSON shows it: `tokens` is empty in every row, because the first
> version asked for `return_tokens`, which the OAI chat serialiser silently
> drops. Their `content` is empty and `reasoning_content` is full in every
> baseline row; first-hand confirmation of [ERRATA A5](../ERRATA.md), and a
> reminder that **these measurements are of thinking generation, not answers.**
> Later runs use the verified `enable_thinking: false` control.

Common to both runs: `Qwen3.6-35B-A3B-UD-Q4_K_XL` target,
`Qwen3.5-0.8B-Q4_K_M` draft, `-ngl 999 -c 16384 --jinja -fa on -ctk q8_0
-ctv q8_0`, greedy, 300-token cap, the ten v1 prompts, draft window 4–8.

The three arms differ in one thing only:

| arm | what it is |
|---|---|
| `baseline` | no draft model |
| `draft-max8-translate` | draft attached; llama.cpp's compatibility gate fails and it uses the token-translation fallback — this is what every published number in this repository was measured on |
| `draft-max8-matched` | identical, plus `--override-kv tokenizer.ggml.bos_token_id=int:248044`, which makes the gate pass |

---

<!-- A contents block, because these documents are linked into by section name from each other and a reader arriving cold had no way to orient but to scroll. Generated from the headings; `analysis/check_links.py` validates every anchor here, so a heading renamed without this list fails the static job rather than rotting quietly. -->

## Contents

- [Run B — post-merge master `3737e4137`, with speculation actually enabled](#run-b--post-merge-master-3737e4137-with-speculation-actually-enabled)
- [Answer 1 — the vocabulary defect is real, and it is not the cause](#answer-1--the-vocabulary-defect-is-real-and-it-is-not-the-cause)
- [Answer 2 — with acceptance measured properly, there is no anomaly](#answer-2--with-acceptance-measured-properly-there-is-no-anomaly)
- [Runs C and D — the thirteen-arm matrix, and the workload the archive never controlled](#runs-c-and-d--the-thirteen-arm-matrix-and-the-workload-the-archive-never-controlled)
  - [C — thinking on, thirteen arms](#c--thinking-on-thirteen-arms)
  - [D — the same arms with thinking verifiably off](#d--the-same-arms-with-thinking-verifiably-off)
  - [Thermals and drift](#thermals-and-drift)
- [Run I — batching, the lever upstream names](#run-i--batching-the-lever-upstream-names)
  - [The failure mode that did not happen](#the-failure-mode-that-did-not-happen)
- [Run J — the first configuration that is actually faster](#run-j--the-first-configuration-that-is-actually-faster)
  - [The configuration that produced it does not start reliably](#the-configuration-that-produced-it-does-not-start-reliably)
  - [Every measurement of the same quantity, with its power](#every-measurement-of-the-same-quantity-with-its-power)
- [Run K — where the optimum is, and what batching does to it](#run-k--where-the-optimum-is-and-what-batching-does-to-it)
  - [Batching destroys it](#batching-destroys-it)
- [Run L — the win is a property of the workload, not of the method](#run-l--the-win-is-a-property-of-the-workload-not-of-the-method)
  - [Which prompts lose it, and why](#which-prompts-lose-it-and-why)
  - [Acceptance sets the sign — and only the sign](#acceptance-sets-the-sign--and-only-the-sign)
- [Thermals across the 2026-08-26 runs](#thermals-across-the-2026-08-26-runs)
- [Run M — the method the vLLM sibling uses, now measured here too](#run-m--the-method-the-vllm-sibling-uses-now-measured-here-too)
  - [The drafter-precision objection, tested](#the-drafter-precision-objection-tested)
  - [Thinking off, and batching](#thinking-off-and-batching)
- [Run N — the two methods nobody had run, and they do nothing](#run-n--the-two-methods-nobody-had-run-and-they-do-nothing)
- [Run O2 — the same matrix as a balanced Latin square](#run-o2--the-same-matrix-as-a-balanced-latin-square)
- [Run O — the same matrix at three repeats, superseded by O2](#run-o--the-same-matrix-at-three-repeats-superseded-by-o2)
- [Run O — every method, one baseline, one policy](#run-o--every-method-one-baseline-one-policy)
- [Runs P and R — is the win a property of those ten prompts?](#runs-p-and-r--is-the-win-a-property-of-those-ten-prompts)
  - [The metric has to change, and that is not a detail](#the-metric-has-to-change-and-that-is-not-a-detail)
- [Run Q — the anomaly this repository could not explain, resolved](#run-q--the-anomaly-this-repository-could-not-explain-resolved)
- [What is settled, and what is not](#what-is-settled-and-what-is-not)
- [Files](#files)
- [Appendix: the evidence sections moved out of the root README](#appendix-the-evidence-sections-moved-out-of-the-root-readme)
  - [What supports that result, and what limits it](#what-supports-that-result-and-what-limits-it)
  - [Metric definitions](#metric-definitions)
  - [What the v1 data support](#what-the-v1-data-support)
  - [What the v1 data do not support](#what-the-v1-data-do-not-support)
  - [The "100 % acceptance" retraction](#the-100--acceptance-retraction)
  - [Where the time goes, measured, and not MoE-specific](#where-the-time-goes-measured-and-not-moe-specific)
  - [What the audit measured on 2026-08-25 and 2026-08-26](#what-the-audit-measured-on-2026-08-25-and-2026-08-26)
  - [v1 representative results](#v1-representative-results)
  - [Experiment registry](#experiment-registry)
  - [v1 hardware, software, and artefacts](#v1-hardware-software-and-artefacts)
  - [Follow-up experiment caveats](#follow-up-experiment-caveats)

## Run A — `bcb5eeb64`, the binary v2 used

`data/A_bcb5eeb64_legacy/`, 2 repeats, binary sha256 `32c16754e053da2f…`

> **`request-mean` is llama.cpp's own `predicted_per_second`, averaged.** That field divides `n − 1` generated tokens by the time for `n`, in 30 300 of 30 344 committed request rows, so every request-mean here is low by `(n − 1) / n`: 0.33 % at 300 tokens and more at shorter lengths. It is uniform across arms on a run where every request hits the same cap, and it is NOT uniform where the arms stop at different lengths, so it must not carry a cross-arm comparison in the thinking-off runs. Every headline figure and every published delta is a **pooled** rate computed from `predicted_n` and `predicted_ms` directly and contains none of this. See [B8](../ERRATA.md#b8-every-request-mean-here-counts-one-token-fewer-than-it-timed).

| arm | request-mean | pooled | min | accepted / drafted | completed |
|---|---:|---:|---:|---:|---|
| baseline | 123.0 | 122.9 | 116.2 | — | 2 / 2 |
| draft, translation fallback | 113.9 | 100.3 | 48.4 | 194 / 194 = **100.0 %** | **0 / 2** |
| draft, matched vocabulary | 113.5 | 101.0 | 50.0 | 194 / 194 = **100.0 %** | **0 / 2** |

Two things to read off this table.

**The 100 % reproduces exactly**, in both arms, on demand. It is the counter
artefact described in [`../ERRATA.md`](../ERRATA.md) A1, not a property of the
draft model.

**Neither draft arm finished.** Both aborted on the `code_small` prompt, 2 / 2,
with the same CUDA error, immediately after a partial-accept checkpoint
restore: see `data/abort_evidence_bcb5eeb64.txt`:

```
slot update_slots: n_draft=6, accepted=6           <- 6 < 6+1, partial
slot update_slots: restoring speculative checkpoint (size = 65864420)
srv  update_slots: decoding batch, n_tokens = 7
ggml-cuda.cu:97: CUDA error: an unsupported value or parameter was passed
  cublasSgemm_v2(..., CUBLAS_OP_T, CUBLAS_OP_N, row_diff, src1_ncols, ne10, ...)
  #14 server_context_impl::update_slots()
```

The baseline arm completes every time. This means v2's "cross-checked on master
`bcb5eeb64`, identical results" is a `llama-cli` cross-check only; the
`llama-server` path that produced every v1 number does not survive on that
commit with a draft attached.

The `baseline` numbers in this run are depressed (123 rather than ~133) because
its first repeat overlapped a compile on the same host. The arm contrast is
unaffected: both draft arms ran after the compile finished, and ABBA ordering
puts them in the same conditions.

---

## Run B — post-merge master `3737e4137`, with speculation actually enabled

`data/B_master_3737e4137/`, 3 repeats, binary sha256 `b6a5c490bb932ffa…`

llama.cpp renamed the speculative arguments after `bcb5eeb64`
(`--draft-max` → `--spec-draft-n-max`) and `--spec-type` now defaults to
`none`. A first attempt with `-md` alone produced a suspiciously clean "no
crash, no slowdown" result; the server log showed **zero `generate_draft`
calls and `draft_n = 0` on all thirty requests**. On master, `-md` loads a
draft model and never speculates unless `--spec-type` is given. That run is
discarded; this one passes `--spec-type draft-simple`.

| arm | request-mean | pooled | min | accepted / drafted | completed |
|---|---:|---:|---:|---:|---|
| baseline | 132.9 | 133.3 | 131.7 | — | 3 / 3 |
| draft, translation fallback | 33.6 | 32.6 | 26.9 | 4926 / 16590 = **29.7 %** | 3 / 3 |
| draft, matched vocabulary | 33.7 | 32.6 | 26.3 | 4926 / 16590 = **29.7 %** | 3 / 3 |

**The abort is gone.** All thirty requests complete in both draft arms.

**The acceptance counter is fixed.** It now reports 29.7 %, not 100 %. This is
independent confirmation of ERRATA A1: the historical 1.0 was the counter being
unreachable on a `COMMON_CONTEXT_SEQ_RM_TYPE_FULL` context, and once upstream
made partial rounds reachable, real ratios appear.

**Speculation now runs on every prompt.** In v1 only 2 of 10 requests recorded
a draft round; here all 10 do.

---

## Answer 1 — the vocabulary defect is real, and it is not the cause

The two arms are byte-identical in what they draft: `16590` drafted and `4926`
accepted, in **both**, and the same per-prompt pairs (`154/576`, `140/647`,
`211/404`, …). Throughput differs by +0.2 % pooled over both binaries, and
from −2.2 % to +3.7 % across the sixteen (binary, prompt) cells: noise.

So: llama.cpp genuinely was running the token-translation fallback, this
repository's "vocab-matched" claim was genuinely false, and fixing it changes
nothing material. The negative finding survives, now measured on a matched
path. See [`../ERRATA.md`](../ERRATA.md) A2 for the root cause; the draft
model's upstream repo has no `generation_config.json`, so its GGUF carries no
`tokenizer.ggml.bos_token_id`, and llama.cpp substitutes the hard-coded GPT-2
legacy default `11` against the target's `248044`, over a field that neither
model uses because both set `add_bos_token = false`.

## Answer 2 — with acceptance measured properly, there is no anomaly

Per prompt on master, matched arm, mean of 3 repeats:

| prompt | baseline | with draft | vs baseline | real acceptance |
|---|---:|---:|---:|---:|
| reasoning | 132.7 | 45.5 | −65.7 % | 52 % |
| code_small | 132.7 | 45.5 | −65.7 % | 50 % |
| zh_hant | 133.0 | 35.3 | −73.5 % | 35 % |
| medium_rec | 133.3 | 36.4 | −72.7 % | 35 % |
| long_explain | 132.8 | 31.3 | −76.4 % | 28 % |
| short_greet | 133.0 | 31.0 | −76.7 % | 27 % |
| multi_turn_2 | 132.5 | 29.5 | −77.7 % | 24 % |
| short_q | 133.3 | 27.8 | −79.1 % | 22 % |
| medium_chat | 133.1 | 27.3 | −79.5 % | 20 % |
| multi_turn_1 | 132.6 | 27.1 | −79.6 % | 20 % |

**Pearson r between real acceptance rate and decode rate = +0.998 across the
ten prompts.**

That is the headline correction. This repository's central claim was an
*anomaly*: 100 % acceptance yet slower, therefore something MoE-specific must
be destroying the speedup. Once acceptance is measured rather than assumed,
acceptance and speed are almost perfectly correlated, exactly as ordinary
speculative-decoding economics predicts. There is no anomaly to explain. The
slowdown on this hardware is low acceptance plus draft-path cost, and it needs
no MoE-specific pathology.

`analysis/plot_accept_vs_speed.png` (every one of whose 140 points sat at
exactly 100 %, making this relationship invisible) has been deleted.

---

## Runs C and D — the thirteen-arm matrix, and the workload the archive never controlled

`data/C_master_matrix_think_on/` and `data/D_master_matrix_think_off/`, both on
post-merge master `3737e4137`, 5 repeats, ABBA-ordered, one pinned binary.
Continuous GPU telemetry for the whole 110 minutes is in
`data/gpu_telemetry_20260825.csv`.

Reproduce the tables with:

```bash
python analysis/matrix_report.py v4_audit_2026_08_25/data/C_master_matrix_think_on
python analysis/matrix_report.py v4_audit_2026_08_25/data/D_master_matrix_think_off
python analysis/thermal_report.py v4_audit_2026_08_25/data/gpu_telemetry_20260825.csv
```

### C — thinking on, thirteen arms

| arm | pooled tok/s | vs baseline | acceptance | draft tokens per generated token | run-to-run SD |
|---|---:|---:|---:|---:|---:|
| `baseline-kvfp16` | 125.7 | **+1.9 %** | — | 0 | 0.37 |
| `baseline` | 123.4 | — | — | 0 | 2.08 |
| `ngram-simple` | 118.1 | −4.2 % | 4.3 % | 0.06 | 0.88 |
| `ngram-mod` n=24 | 115.0 | −6.8 % | 4.8 % | 0.21 | 0.59 |
| `ngram-cache` | 74.0 | −40.0 % | 1.8 % | 0.42 | 2.48 |
| `ngram-cache-kvfp16` | 70.9 | −42.5 % | 2.0 % | 0.44 | 1.08 |
| draft model n_max 4 | **35.6** | −71.1 % | 45.4 % | 1.12 | 0.08 |
| draft model n_max 2 | 34.2 | −72.2 % | 60.3 % | 0.74 | 0.06 |
| draft model, v1's config | 32.3 | −73.8 % | 29.7 % | 1.84 | 0.56 |
| draft model n_max 8 | 32.1 | −74.0 % | 29.7 % | 1.85 | 0.54 |
| draft model n_max 1 | 31.1 | −74.8 % | 68.7 % | 0.50 | 0.07 |
| draft model n_max 16 | 23.7 | −80.8 % | 15.0 % | 3.58 | 0.86 |
| draft model n_max 32 | 17.3 | −86.0 % | 8.0 % | 6.84 | 0.04 |

The run-to-run SD column is the one honest `±` in this repository: the spread
of five whole-prompt-set repeats, 0.04–2.48 tok/s. The historical `±27–31` was
spread *between prompts*.

It is the SD of the five repeats' **request means**, llama.cpp's
`predicted_per_second` averaged over the ten prompts, and not of the pooled
rate in the column beside it. The two are different estimators and give
different answers: recomputing this column from the per-repeat pooled rate
reproduces 3 of the 13 published values, the request mean reproduces 13 of 13.
Like every request mean in this repository it is low by `(n−1)/n`; see
[`../ERRATA.md` B8](../ERRATA.md#b8-every-request-mean-here-counts-one-token-fewer-than-it-timed).

Two entries in that column need reading carefully. `baseline`'s 2.08 is almost
entirely its cold-start first repeat; excluding it the SD is 0.47. But
`ngram-cache`'s is not: 77.9, 74.7, 75.8, 72.3, 72.0 tok/s, still 1.86 after
dropping rep 0, and drifting downward. It is the least reproducible arm in the
matrix, and this audit does not have an explanation for it. The other eleven
arms span 0.03 to 1.01 once the cold start is removed, the top of that range
being `ngram-cache-kvfp16`; so `ngram-cache` leads by a factor of 1.8, not by
the wide margin an earlier version of this paragraph claimed. See
[`../ERRATA.md` A18](../ERRATA.md#a18-run-cs-spread-claim-excluded-five-arms-without-saying-so).

Three things fall out.

**The best draft length is 4, and it is still 71 % below baseline.** The sweep
is not monotone; cost rises and acceptance falls as the window widens, and
n_max 1 is worse than n_max 2 because a full drafter forward pass buys at most
one token. The peak is real but shallow: per-repeat SD is 0.055 and 0.076 for
n_max 2 and 4, against a 1.4 tok/s gap.

**Drafting volume is not the whole cost.** An external drafter proposing 0.50
tokens per generated token runs at 31.1 tok/s; `ngram-cache` proposing 0.42
runs at 74.0. The per-round cost of the *method* is a third, independent term:
a 0.8 B forward pass every round versus a table lookup.

**The fp16-KV control the v1 matrix lacked.** fp16 KV is 1.9 % faster than
q8_0 with no speculation running, and 4.2 % worse with `ngram-cache` on. See
[`../ERRATA.md`](../ERRATA.md) B7.

### D — the same arms with thinking verifiably off

`thinking_suppressed` is recorded per request. Per arm it is 50 of 50 in D
and 0 of 50 in C, which is 250 of 250 over D against 0 of 650 over C. Output
lengths in D run 22–300 tokens because completions now finish naturally.

| method | thinking on (C) | thinking off (D) | draft tokens per generated token |
|---|---:|---:|---|
| `ngram-mod` n=24 | −6.8 % | **−0.7 %** | 0.21 → **0.00** |
| `ngram-cache` | −40.0 % | −32.6 % | 0.42 → 0.36 |
| draft model n_max 8 | −74.0 % | −76.4 % | 1.85 → **2.14** |

With thinking off `ngram-mod` never drafts at all, zero draft tokens across
fifty requests, and its cost nearly disappears. A chain-of-thought trace is
long and formulaic, which is what an n-gram lookup feeds on; a direct answer is
not.

**The draft model's row does not say what it looks like it says, and this
paragraph used to.** It read: "acceptance falls from 29.7 % to 23.1 %, so
reasoning traces are *easier* for a 0.8 B drafter than real answers". The two
halves are not the same amount of work
([A17](../ERRATA.md#a17-the-thinking-off-comparisons-are-not-comparisons-of-the-same-amount-of-work)):
with thinking off the completions finish naturally, at 22 to 300 tokens, and
acceptance varies along the sequence, so a short generation is scored on its
early tokens alone. On the five prompts where every arm generated the same 300
tokens, acceptance with thinking off is **30.3 %** against 29.7 % with thinking
on, and the cost is smaller with thinking off rather than larger. The fall was
the length, not the workload.

This is the comparison Exp 2 believed it had made, and it could not have made
it: the two halves it compared differ in how much they generated. What survives
the length-matched reading is `ngram-mod`, whose drafting stops entirely, and
that is a statement about an n-gram lookup and not about the workload being
harder.
And it means **every historical number in this repository was measured on the
workload that favours speculation, and speculation still lost.**

### Thermals and drift

1317 telemetry samples over 110 minutes, 1272 under load:

- `power.limit` = `power.default_limit` = `power.max_limit` = 350 W:
  **not overclocked**
- 58–75 °C, mean 64.7, against a ~83 °C throttle point
- 1800–1965 MHz of a 2100 MHz maximum, mean 1937
- `hw_power_brake` never active; `sw_thermal` fires on **2** samples of 1272 and
  `hw_thermal` on **1**, at 64–65 °C and with the clock at 1950, 1950 and 1935
  MHz against a run maximum of 1965, so none of the three carried a meaningful
  downclock. (An earlier version of this line said "the one `sw_thermal` sample
  … at its run maximum". Both halves were wrong; see [ERRATA C4b](../ERRATA.md#c4b-stock-clocks-was-measured-once-before-the-load).)

The baseline is repeated five times across the run, so drift is testable from
the measurement itself: 126.6, 122.2, 122.6, 121.8, 121.6 tok/s. That is not
progressive decline; it is a cold-start first repeat, +3.7 % against a tail
that then varies by 0.89 %. Three other arms measured across the same repeats
show no such step (+0.35 % to +0.47 %), because only the first arm of the first
repeat starts on an idle, cool card.

---

## Run I — batching, the lever upstream names

Upstream's standing answer to "speculative decoding loses on a MoE target" is
that the regime is wrong: the win is supposed to appear when the GPU is not
already saturated by a single stream, i.e. under batching (ERRATA A9). Run I
tests that directly on this host, with the matched-vocabulary drafter at
`--spec-draft-n-max 8`; the arm closest to what v1 ran.

**First, the harness had to be fixed.** The first attempt at this run measured
nothing. `BENCH_CONCURRENCY` passed `--parallel N -cb` to the server, which
allocated N slots, and then the client issued the ten prompts one at a time, so
N−1 slots sat idle. The tell was in the data before any code was read: the c=4
arm-runs took 44 s and 118 s against c=1's 44 s and 116 s. Four times the
nominal concurrency at identical wall-clock is a client batch size of one. That
attempt is discarded, not reported.

The runner now dispatches through a thread pool and records how many **client**
requests were outstanding at once, derived from the request timestamps. That is
**not** the server's decode batch width and is not offered as one: a server
that processed every request serially would still show all of their HTTP
windows overlapping while the later ones sat in its queue. What it does
establish is the negative case this run was re-done for; a value of 1 would
mean the client never had more than one request outstanding. It equals the
configured level in all eighteen arm-runs:

| level | requested | observed client requests in flight |
|---|---|---|
| c=1 | 1 | 1, 1, 1, 1, 1, 1 |
| c=4 | 4 | 4, 4, 4, 4, 4, 4 |
| c=8 | 8 | 8, 8, 8, 8, 8, 8 |

Measuring what the server actually batched needs instrumentation of active
sequences and batch/ubatch token counts per decode, which is queued in
[`../RETEST_TODO.md`](../RETEST_TODO.md) and not done here.

Aggregate throughput: 3000 generated tokens divided by wall-clock over the
ten-prompt set, mean of three repeats:

| concurrency | no speculation | `spec-draft-n8` | speculation ÷ baseline |
|---|---|---|---|
| 1 | 109.7 ± 0.57 | 30.6 ± 0.14 | 0.28× |
| 4 | 154.3 ± 0.27 | 27.0 ± 0.73 | 0.18× |
| 8 | 180.0 ± 15.21 | 28.1 ± 0.66 | 0.16× |

![Aggregate throughput at one, four and eight concurrent client requests. The no-speculation arm rises with concurrency and the external-drafter arm stays flat, so the ratio between them falls across the three levels](../analysis/plot_batching.png)

**Batching helps the target and does nothing for the drafter.** No speculation
gains +40.6 % at c=4 and +64.0 % at c=8. Speculation moves −11.7 % and −8.4 %
over the same range. The gap therefore widens with batching rather than
closing: 0.28× → 0.18× → 0.16×.

On this host, at this model and draft window, batching is not the missing
regime. That is a negative result about one arm on one card, not a refutation
of the upstream argument in general, but it is the specific configuration this
repository has been reporting, measured in the regime it was told to measure it
in.

Two caveats, both against the strength of the result:

- ± is the run-to-run SD of three repeats. At c=8 it is 15.21 because one
  baseline repeat came in at 197.5 against 171.2 twice. Ten prompts over eight
  slots is one full wave plus a wave of two, so wall-clock there is sensitive
  to which prompts land in the short tail; c=4, at 2.5 waves, is the cleaner
  measurement. The conclusion survives either way; the *slowest* c=8 baseline
  repeat is still +56 % over c=1, and speculation never moves at all.
- The prompt set is fixed at ten. A batching benchmark would normally hold the
  arrival rate, not the request count, constant.

### The failure mode that did not happen

llama.cpp issue #27572 reports draft acceptance silently collapsing to 0 under
`-np N`. It did not occur here, and that was checked rather than assumed:

| level | drafted | accepted | counted ratio | requests with `draft_n = 0` |
|---|---|---|---|---|
| c=1 | 5547 / rep | 1646 | 29.7 % | 0 / 10 |
| c=4 | 5572–5691 | 1608–1641 | 28.3–29.5 % | 0 / 10 |
| c=8 | 5656–5687 | 1607–1625 | 28.3–28.7 % | 0 / 10 |

Acceptance is flat across concurrency. Speculation is losing under batching
because the target's own batched decode gets much cheaper while the draft cost
does not, not because the drafter stopped working.

---

## Run J — the first configuration that is actually faster

This is the DFlash A/B that ERRATA D4 says v3 never had: one binary
(`b6a5c490…`), one placement policy, one drafter, three repeats per arm, DFlash
off and on.

Two things had to be true first. The archived v3 drafter GGUF is rejected by
post-merge master for lacking `target_layers`; the re-converted file carries it
(`dflash.target_layers = [2, 11, 20, 29, 38]`). And the BF16 drafter only loads
with `-ngl` unset, because pinning it makes `common_fit_params` abort instead
of adjusting the parameters the caller left unset, so `BENCH_FIT=on` applies to
**every** arm in the run, the baseline included, and placement policy stays
constant across the comparison.

That last decision is a confound if `-fit on` quietly handicaps the control, so
it is checked directly:

| control | aggregate |
|---|---|
| baseline, `-ngl 999` pinned (run I, c=1) | 109.72 ± 0.57 |
| baseline, `-fit on` (run J) | 109.70 ± 0.18 |
| difference | **−0.01 %** |

Both load `41/41` target layers to GPU; the DFlash arm additionally loads `9/9`
drafter layers. The control is not being handicapped.

| arm | pooled | aggregate | vs no speculation | drafted | acceptance |
|---|---|---|---|---|---|
| **`spec-dflash-n4`** | **151.6** | **130.2 ± 1.21** | **+18.7 %** | 11 070 | 55.8 % |
| no speculation | 122.3 | 109.7 ± 0.18 | — | 0 | — |
| `spec-dflash-n8` | 105.2 | 93.5 ± 0.56 | −14.8 % | 18 114 | 36.8 % |
| `spec-dflash-n16` | 62.8 | 57.7 ± 0.19 | −47.4 % | 31 728 | 21.4 % |
| `spec-draft-n8` (matched vocab) | 31.4 | 30.5 ± 0.18 | −72.2 % | 16 641 | 29.7 % |

![Change in aggregate throughput against maximum draft length for two runs, with acceptance in a panel below. A plateau up to draft length four, then a fall to well below the baseline, and the two runs land together at the two draft lengths they share](../analysis/plot_dflash_sweep.png)

**DFlash at `n_max 4` is +18.7 % on aggregate throughput and +24.0 % pooled.**
It is the first configuration in this repository that beats not speculating,
and it is not an average that hides losers: it wins on all ten prompts
individually:

| prompt | no speculation | `dflash-n4` | `dflash-n8` | `dflash-n16` |
|---|---|---|---|---|
| `short_greet` | 122.4 | 156.2 | 108.5 | 60.6 |
| `short_q` | 122.3 | 150.9 | 87.2 | 52.4 |
| `medium_chat` | 121.8 | 141.1 | 93.2 | 53.9 |
| `medium_rec` | 121.8 | 160.9 | 125.7 | 83.7 |
| `reasoning` | 121.8 | 178.4 | 128.9 | 94.7 |
| `long_explain` | 122.0 | 140.8 | 95.5 | 52.7 |
| `code_small` | 121.8 | 189.4 | 170.8 | 96.7 |
| `multi_turn_1` | 121.7 | 141.3 | 96.2 | 58.7 |
| `multi_turn_2` | 121.9 | 132.0 | 87.5 | 53.9 |
| `zh_hant` | 121.4 | 138.4 | 99.5 | 55.2 |

(per-request decode rate, mean of three repeats)

`n_max 8`'s aggregate loss, on the other hand, *is* an average that hides
winners. It is −14.8 % over the ten prompts and it beats no speculation on
three of them — `code_small`, `reasoning` and `medium_rec` — and acceptance
separates the two groups without an overlap: the three winners sit at 71.8 %,
48.6 % and 47.3 %, the seven losers between 27.1 % and 37.8 %, a 9.5 pp gap
with nothing in it. So the best `n_max` is a property of the prompt and not
only of the build, and a single aggregate number chooses it for the average
request rather than for the request in front of you. This repository does not
have enough prompts to say where the boundary is; it has ten, and they fall on
two sides of one.

The sign flips with draft length, and it flips fast: +18.7 % at 4, −14.8 % at
8, −47.4 % at 16, with acceptance falling 55.8 % → 36.8 % → 21.4 % as the
window grows. The archived v3 result, DFlash slower, is what this looks like at
`n_max 8` and 16. v3 measured n_max 4 as well, but across a binary change, and
read the difference as a DFlash effect.

### The configuration that produced it does not start reliably

Run J's fifteen arm-runs all completed: no crashes, no retries. But the same
`-c 16384` DFlash configuration, on the same binary and models forty minutes
later, aborted at its first decode:

```
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 120.28 MiB on device 0:
    cudaMalloc failed: out of memory
srv  update_slots: decode() failed: failed to allocate compute pp buffers
```

Run J's telemetry peaks at 23946 MiB of 24576 with the drafter resident (630
MiB of headroom, 2.6 % of the card) and a 120 MiB allocation still failed, so
the true transient peak is above what five-second sampling can see. With `-fit
on` the fitter sizes the target against whatever it reads as free at startup,
and whether the drafter then fits is decided in that margin.

Run J's numbers are what they are: fifteen clean arm-runs with a matched
control. But **the configuration is marginal on a 24 GiB card**, and anyone
reproducing it should expect to lower the context. Run K does exactly that, `-c
8192` for every arm including its own baseline, which is why K's absolute rates
are not comparable with J's and K carries its own control.

The harness now fails fast on this: `wait_health` watches the process instead
of polling for its full 300-second timeout, so a dead arm costs ~20 seconds to
discover rather than five minutes per repeat.

### Every measurement of the same quantity, with its power

The number this repository headlines should not be the one with the fewest
repeats behind it. All six measurements of the DFlash plateau:

| run | arm | repeats | Δ aggregate | Δ pooled | configuration |
|---|---|---|---|---|---|
| J | `n_max 4` | 3 | **+18.7 %** | +23.9 % | `-c 16384`, fitter default margin |
| K1 | `n_max 2` | 3 | +17.1 % | +20.9 % | `-c 8192`, `--fit-target 2048` |
| K1 | `n_max 3` | 3 | +17.6 % | +21.6 % | `-c 8192`, `--fit-target 2048` |
| K1 | `n_max 4` | 3 | +17.3 % | +21.5 % | `-c 8192`, `--fit-target 2048` |
| L, thinking on | `n_max 2` | **5** | +16.7 % | **+21.1 %** | `-c 8192`, `--fit-target 2048` |
| L, thinking on | `n_max 4` | **5** | +16.1 % | +20.9 % | `-c 8192`, `--fit-target 2048` |

Aggregate spans +16.1 % to +18.7 %; pooled spans +20.9 % to +23.9 %. The two
metrics differ by about 4 pp throughout because aggregate divides by wall-clock,
which includes prompt processing and HTTP overhead that speculation does not
touch; pooled divides by decode time and does not.

Run J's +18.7 % is the top of the aggregate range and has the fewest repeats
behind it. It is reported because it is where the effect was first isolated
under a matched control, not because it is the best estimate. The best estimate
is the five-repeat row.

**What this does and does not establish.** It establishes that on this host,
this target, this drafter and this prompt set, a self-speculative method at a
short draft window beats no speculation by roughly a fifth, with a matched
control and three repeats. It does not establish that the number transfers: the
window is one draft length on one card, thinking is on throughout, and ten
prompts is ten prompts. The draft-length optimum is bracketed from below in run
K, because three points that straddle a peak cannot say where the peak is.

---

## Run K — where the optimum is, and what batching does to it

Run J put `n_max 4` at +18.7 % and `n_max 8` at −14.8 %, which locates the peak
at or below 4 and says nothing about where. Run K brackets it from 1, and then
asks of the winning arm the question run I asked of the matched-vocabulary one.

Everything here is at `-c 8192` with `--fit-target 2048`, applied to every arm
including the baseline, for the reason in [The configuration that produced it
does not start reliably](#the-configuration-that-produced-it-does-not-start-reliably): at the fitter's
default margin the DFlash configuration does not start reliably. Absolute rates
are therefore not comparable with run J's; the deltas against each run's own
baseline are.

| `n_max` | aggregate | run-to-run SD | pooled | vs no speculation | acceptance |
|---|---|---|---|---|---|
| — (no speculation) | 110.6 | 2.10 | 123.3 | — | — |
| 1 | 120.2 | 1.97 | 135.2 | +8.7 % | 82.0 % |
| 2 | 129.5 | 0.13 | 149.0 | +17.1 % | 72.8 % |
| **3** | **130.0** | 1.49 | 149.9 | **+17.6 %** | 63.6 % |
| 4 | 129.8 | 0.51 | 149.8 | +17.3 % | 55.6 % |
| 6 | 100.8 | 0.63 | 114.5 | −8.9 % | 43.0 % |
| 8 | 93.2 | 0.63 | 104.6 | −15.8 % | 37.2 % |

**There is no peak, there is a plateau and then a cliff.** `n_max` 2, 3 and 4
land at 129.5, 130.0 and 129.8; separated by less than the run-to-run SD of the
baseline, so calling any one of them "the optimum" would be reading noise. What
the sweep does establish is the shape: one token of draft is not enough (+8.7
%), two to four are worth about +17 %, and by six the arm is already losing.
The sign change sits between 4 and 6, not between 4 and 8 as run J's coarser
grid suggested.

Run K also **replicates run J** at the two draft lengths they share, from a
different context and a different fitter margin:

| `n_max` | run J | run K |
|---|---|---|
| 4 | +18.7 % | +17.3 % |
| 8 | −14.8 % | −15.8 % |

### Batching destroys it

| | no speculation | `spec-dflash-n4` | vs baseline |
|---|---|---|---|
| 1 request in flight | 110.6 | 129.8 | **+17.3 %** |
| 4 in flight | 154.1 ± 0.48 | 154.7 ± 3.35 | +0.4 % |
| 8 in flight | 153.2 ± 0.61 | 39.6 ± 1.47 | **−74.1 %** |

The advantage is gone at four concurrent requests and the arm collapses at
eight, reproducibly: SD 1.47 on 39.6 across three repeats.

**It is not that the drafts get worse.** Draft volume per generated token and
acceptance barely move across the three levels; only the clock does:

| level | drafted per generated token | acceptance | aggregate |
|---|---|---|---|
| 1 in flight | 1.234 | 55.6 % | 129.8 |
| 4 in flight | 1.243 | 55.0 % | 154.7 |
| 8 in flight | 1.305 | 51.2 % | 39.6 |

The draft work is not being shared across the batch, so its cost scales with the
batch while the target's per-token cost falls. That is the same shape run I
found for the matched-vocabulary drafter, reached by a different method.

Two things checked rather than assumed before reading that −74.1 %:

- **Not context exhaustion.** `-c 8192` across eight slots is 1024 tokens each,
  and the requests report `n_tokens = 328` at release. The speculative
  checkpoint machinery never fired: zero checkpoints created, zero restored. The
  only log warnings are `truncating draft to N tokens`, which is ordinary `p_min`
  truncation, and the `discard` count is identical in both arms.
- **The baseline itself is constrained at this context**, and that is stated
  rather than hidden: run K's c=8 baseline plateaus at 153.2 where run I's, at
  `-c 16384`, reached 180.0. That is a cross-run difference in absolute rate. The
  −74.1 % is a within-run contrast against the 153.2 measured beside it.

---

## Run L — the win is a property of the workload, not of the method

Workload shape has moved a result in this repository before: `ngram-mod` went
from −6.8 % to −0.7 % between the think-on and think-off matrices and drafted
zero tokens in the latter (ERRATA D3b). A headline measured with thinking on has
to be shown with it off.

Run L runs the same five arms twice, five repeats each, at the same context and
fitter margin, differing only in `enable_thinking`. **Pooled** throughput is
the metric here: with thinking off the outputs are shorter *and differ in
length by arm*, median 96 tokens for the baseline against 83 for `dflash-n4`,
because speculation changes the generated text (A11), and aggregate throughput
would mix decode rate with output length.

> [!WARNING]
> **This section used to continue "Pooled is tokens over decode time and does
> not", and that is wrong.** Pooled removes the *wall-clock* dependence on
> output length; it does not make two arms comparable when they generated
> different numbers of tokens, because decode rate falls as the KV cache grows.
> Restricting run L's thinking-off comparison to the five prompts where every
> arm generated exactly 300 tokens moves `spec-dflash-n4` from **−2.7 %** to
> **+14.1 %**, the sign below flips, and moves every other model-drafting arm
> in the four thinking-off runs by +2.5 pp to +16.8 pp. [ERRATA A17](../ERRATA.md#a17-the-thinking-off-comparisons-are-not-comparisons-of-the-same-amount-of-work)
> has the full table and `analysis/length_matching.py` recomputes it. The
> thinking-**on** columns are unaffected: every request there generated exactly
> 300 tokens, and the same restriction moves them by 0.00 pp. The harness now
> has `BENCH_IGNORE_EOS` to force the hard cap; the numbers below predate it
> and are left as measured.

| arm | thinking ON | | thinking OFF | |
|---|---|---|---|---|
| | pooled | vs base | pooled | vs base |
| no speculation | 122.9 | — | 124.1 | — |
| `spec-dflash-n2` | 148.8 | **+21.1 %** | 133.5 | **+7.6 %** |
| `spec-dflash-n4` | 148.5 | +20.9 % | 120.7 | **−2.7 %** |
| `spec-dflash-n6` | 114.8 | −6.6 % | 93.4 | −24.7 % |
| `spec-draft-n8` | 30.6 | −75.1 % | 27.5 | −77.8 % |

Thinking was verifiably suppressed on 250/250 requests in the left half and on
0/250 in the right, measured from the reasoning channel rather than assumed from
the flag.

**The win survives, and shrinks by two thirds.** At `n_max 2` it goes from
+21.1 % to +7.6 %. At `n_max 4`, the configuration run J headlined, it goes
**negative**, −2.7 %. Acceptance falls with it: 72.8 % → 58.5 % at `n_max 2`,
55.6 % → 40.3 % at `n_max 4`.

### Which prompts lose it, and why

`spec-dflash-n2`, per prompt, both halves:

| prompt | ON: acc / Δ | OFF: acc / Δ |
|---|---|---|
| `reasoning` | 82.3 % / +30 % | 85.1 % / **+33 %** |
| `code_small` | 92.4 % / +40 % | 88.6 % / **+33 %** |
| `short_greet` | 72.8 % / +20 % | 68.0 % / +17 % |
| `long_explain` | 70.6 % / +19 % | 59.5 % / +9 % |
| `short_q` | 69.6 % / +18 % | 52.5 % / +4 % |
| `multi_turn_2` | 60.7 % / +11 % | 53.1 % / +3 % |
| `medium_rec` | 80.7 % / +28 % | 55.0 % / +2 % |
| `medium_chat` | 64.6 % / +13 % | 44.8 % / −5 % |
| `multi_turn_1` | 74.6 % / +23 % | 39.6 % / **−12 %** |
| `zh_hant` | 66.1 % / +15 % | 28.6 % / **−25 %** |

Ten of ten prompts win with thinking on; seven of ten with it off. The two that
keep their full gain are the two whose output is most constrained, step-by-step
arithmetic and Python, and their acceptance barely moves. The one that loses
most is Traditional Chinese free prose, where acceptance falls from 66 % to 29
%.

Reasoning text is planning prose: enumerated, repetitive, formulaic. A drafter
predicts it well. Direct answers are shorter and less templated, and Chinese
free prose least templated of all. **The speed-up is not a property of DFlash;
it is a property of how predictable the text being generated is**, and the
thinking channel happens to be very predictable text.

### Acceptance sets the sign — and only the sign

Across all 60 points in run L (ten prompts, three draft lengths, two workloads)
acceptance and speed-up correlate at **r = +0.946**, and the least-squares line
crosses zero at **48.2 % acceptance**. Below it 24 of 25 points are slower;
above it 35 of 35 are faster.

![Change in decode rate against the share of draft tokens accepted, for sixty fitted points and ten out-of-sample ones, with the least-squares line and the band spanned by the three crossings the fit gives on three subsets of the same points](../analysis/plot_acceptance_threshold.png)

ERRATA A10 is what stops that being written down as a law: a single-regressor
fit that looked excellent in sample was falsified out of it. So this one was
pushed at runs J and K, which it never saw and which used a different context
and a different fitter margin:

Scored over **every** arm-run for which both an acceptance figure and a matched
baseline exist, 90 of them, with one exclusion stated up front: four
`ngram-map` arm-runs drafted at most **45 tokens in total**, and a percentage
computed over ten tokens is not a rate. There is a clean gap in the data at
that point; every other (run, arm) drafted at least 132. The remaining 86:

| family | sign predicted correctly |
|---|---|
| self-speculative (DFlash and MTP) | **57 / 59** |
| drafter-free n-gram | 8 / 11 |
| **external 0.8 B drafter** | **13 / 16** |
| all | **78 / 86** |

**And that scorecard barely depends on which acceptance counter you read.** A13
shows the server counter under-reports on any path that takes a speculative
checkpoint. Rescoring with the speculator's own counter gives **79 / 86**
against 78 / 86: one arm-run changes side. Without the minimum-sample exclusion
it does not: the score moves from 82/90 to 79/90 and `ngram-map` verdicts flip,
because on ten drafted tokens the two counters read 0.0 % and 70.0 % for the
same arm-run. Excluding them is not tidying; it is refusing to score a rate
that has no denominator.

The two misses are the informative ones:

| arm | acceptance | measured | why it is interesting |
|---|---|---|---|
| `spec-mtp-n4`, thinking off | 49.5 % | −8.2 % | sits **1.3 pp above** the boundary; a threshold that never missed near its own boundary would be suspicious |
| `spec-draft-n1` | 69.7 % server, **100.0 %** drafter | **−75.1 %** | the external drafter, which is the entire point |

Mean magnitude error +8.1 pp, worst **+52.2 pp**.

**The two failures are the informative ones**, and they were only found because
the first version of this test drew its out-of-sample set from runs J and K,
which contain the external drafter only at 29.7 % acceptance: below the
threshold, where it agreed. Run C swept that drafter down to `n_max 1`:

| arm | acceptance | measured | threshold says |
|---|---|---|---|
| `spec-draft-n1` | **68.7 %** | **−74.8 %** | faster — **wrong** |
| `spec-draft-n2` | 60.3 % | −72.2 % | faster — **wrong** |
| `spec-draft-n4` | 45.4 % | −71.1 % | slower — ok |

**The threshold transfers; the slope does not.** At `n_max 1` the line predicts
+40.5 % from 82 % acceptance and the arm delivers +9.7 %, because one drafted
token per round cannot buy much however often it lands. The worst miss is
`spec-draft-n8` at +52.2 pp; a separate 0.8 B draft model pays a full forward
pass per drafted token where DFlash reuses the target's own layers, so its cost
per unit of acceptance is not the same quantity at all. Even there the sign was
right.

So the usable statement is narrower than it first looked: **within a
self-speculative family on this target, a configuration is worth running when
it clears roughly 48 % draft acceptance.** It is not a statement about
acceptance in general. An external 0.8 B drafter is 75 % slower at 68.7 %
acceptance, because it pays a fixed per-round cost that no acceptance rate can
amortise: a full checkpoint the server reports at 82.079 MiB, saved and
restored, plus a dense forward pass against a target that activates only ~3 B
parameters. That cost is measured in [ERRATA A12](../ERRATA.md#a12-what-the-checkpoint-path-costs-measured-with-timers-in-the-source),
and it is why the threshold holds inside one family and not across them.

---

## Thermals across the 2026-08-26 runs

Every run carried its own continuous `nvidia-smi` trace, and each was checked
rather than assumed, because a five-hour session on one card is exactly where a
throttling artefact would hide.

| run | loaded samples | temperature | SM clock | first half → second half |
|---|---|---|---|---|
| I + J | 539 | 55–73 °C (mean 63.4) | 1695–1965 MHz (mean 1939) | — |
| K | 317 | 53–74 °C (mean 64.0) | 1695–1980 MHz (mean 1937) | −0.05 % |
| L | 421 | 50–74 °C (mean 64.4) | 1695–1965 MHz (mean 1942) | −0.24 % |

The card is never overclocked (350 W stock limit throughout) and never near its
~83 °C throttle point. The dominant flag under load is `0x4`, `SwPowerCap`, the
350 W board limit doing its job, in roughly half the loaded samples,
identically across arms within a run. **One** sample in the I+J trace
additionally carries `0x20`, `SwThermalSlowdown`; that is a software flag and a
single sample of 539, the same transient behaviour ERRATA C4b recorded on
2026-08-25. No hardware throttle bit — `HwSlowdown`, `HwThermal`,
`HwPowerBrake` — is ever set in any of the three traces. Clock drift between
the first and second half of each run is within 0.25 %, so none of the deltas
above can be a decline in the card.

(The first version of this paragraph said `0x4` was the only flag that ever
appears. The regression checker caught it, after its own throttle mask was
corrected: `0xE0` counted `SwThermal` as hardware and missed `HwSlowdown`
entirely. Hardware is `0x8 | 0x40 | 0x80`.)

---

## Run M — the method the vLLM sibling uses, now measured here too

The vLLM result on this same physical hardware is an **MTP** result. Until MTP
ran under llama.cpp on this card, "llama.cpp loses where vLLM wins" confounded
the engine with the speculation method. Nothing was blocking it: the target's
own multi-token-prediction head is in the checkpoint (785 plain BF16 `mtp.*`
tensors, `text_config.mtp_num_hidden_layers = 1`), master's converter exports it
with `--mtp`, and `LLM_ARCH_QWEN35MOE` already declares the seventeen `NEXTN`
tensors. It had simply never been run. The head is 3.7 GB at BF16 and cannot sit
beside a 21 GB target on a 24 GB card, so it is quantised.

Both families in one matrix, one policy, three repeats: aggregate throughput:

| arm | aggregate | vs no speculation | acceptance |
|---|---|---|---|
| **`spec-dflash-n2`** | **127.3** | **+23.2 %** | 72.3 % |
| `spec-mtp-n2` | 122.5 | **+18.6 %** | 78.4 % |
| `spec-dflash-n4` | 119.9 | +16.1 % | 55.2 % |
| `spec-mtp-n1` | 118.4 | +14.6 % | 89.0 % |
| `spec-mtp-n4` | 111.5 | +8.0 % ‡ | 61.4 % |
| no speculation | 103.3 | — | — |
| `spec-mtp-n8` | 71.4 | −30.9 % | 41.4 % |

‡ **This row did not replicate and should not be used.** Measured again at five
repeats (run Q) the same arm reads +2.0 % on pooled decode rate against this
run's +10.5 %, and a third measurement on the extended prompt set reads +2.7 %.
Everything that could differ between the two runs was checked and is identical,
including the recorded argv, the drafter's sha256, the memory fitter's choices,
the per-prompt draft counts, the acceptance and the card's temperature and
clocks. The row is left in place because deleting a measurement that was made
is not the same as correcting it; see [A14](../ERRATA.md#a14-within-run-repeats-are-not-an-error-bar).

**MTP works, and DFlash is better.** Both peak at `n_max 2`; MTP has the higher
acceptance at every length and still loses on throughput, because its head is a
full MoE layer plus `lm_head` while DFlash's is smaller.

### The drafter-precision objection, tested

"You quantised the drafter" is the first thing to say about any MTP number here,
since DFlash ran at BF16. So the same arms were run with a **Q4_K_M** head
instead of Q8_0:

| arm | Q8_0 drafter | Q4_K_M drafter |
|---|---|---|
| `spec-mtp-n2` | +18.6 % (acc 78.4 %) | **+22.3 % (acc 79.4 %)** |
| `spec-mtp-n4` | +8.0 % (acc 61.4 %) | +1.2 % (acc 60.6 %) |

At `n_max 2` the **more** aggressively quantised drafter is **faster**, at
essentially unchanged acceptance; the head is cheaper to run and the drafts are
just as good. So quantisation is not hiding an MTP advantage; if anything it is
one.

The `n_max 4` row appeared to move the other way by 6.8 pp at unchanged
acceptance, and this section originally reported that as measured and
unexplained. **Five repeats of each drafter dissolved it**: the Q8_0 arm was the
one that did not replicate, and with it re-measured the Q4_K_M head is ahead at
*both* draft lengths. See run Q below.

### Thinking off, and batching

| | thinking on (run M1, aggregate) | thinking off (run M3, 5 repeats, pooled) |
|---|---|---|
| `spec-mtp-n2` | +18.6 % | **+11.4 %** (acc 67.5 %) |
| `spec-dflash-n2` | +23.2 % | +8.5 % (acc 58.4 %) |
| `spec-mtp-n4` | +8.0 % | −8.2 % (acc 49.5 %) |

The two columns are different metrics, which the header now says: the left is
run M1's aggregate, quoted from its own table above, and the right is run M3's
pooled decode rate. The flip is not an artefact of that. On pooled rates both
ways it is the same: +26.7 % against +22.1 % with thinking on, +8.5 % against
+11.4 % with it off.

The ranking **flips**: with thinking on DFlash leads, with it off MTP does. MTP
keeps more of its acceptance when the text stops being planning prose, which is
what a head trained on the target's own hidden states should do.

Under batching MTP degrades far more gracefully than DFlash:

| requests in flight | DFlash `n4` | MTP `n2` |
|---|---|---|
| 1 | +17.3 % | +18.6 % |
| 4 | +0.4 % | **+3.4 %** |
| 8 | **−74.1 %** | **−7.6 %** |

DFlash collapses at eight concurrent requests; MTP loses eight per cent. Both
lose their advantage, but only one of them falls off a cliff.

---

## Run N — the two methods nobody had run, and they do nothing

`ngram-map-k` and `ngram-map-k4v` need no draft model, so they were the cheapest
available test of whether "short drafts win" is about draft volume or about
DFlash in particular. Six arms, at the upstream default `size_m 48` and at 8
and 4:

| arm | aggregate | vs baseline | draft tokens over 30 requests | acceptance (server / drafter) |
|---|---|---|---|---|
| no speculation | 109.4 | — | 0 | — |
| `ngram-map-k` (default `m48`) | 107.6 | −1.6 % | 144 | 0.0 % / 14.5 % |
| `ngram-map-k-m8` | 107.6 | −1.6 % | 24 | 0.0 % / 53.3 % |
| `ngram-map-k-m4` | 107.5 | −1.7 % | 12 | 0.0 % / 70.0 % |
| `ngram-map-k4v` (default) | 107.3 | −1.9 % | 144 | 0.0 % / 14.5 % |
| `ngram-map-k4v-m8` | 108.6 | −0.7 % | 24 | 0.0 % / 53.3 % |
| `ngram-map-k4v-m4` | 108.0 | −1.2 % | 12 | 0.0 % / 70.0 % |

**They almost never engage**, and that is measured on the one quantity here
that neither counter can distort. Per repeat, ten requests, the speculator's
`generate()` is called **3271 times** and returns a draft **twice**; across all
thirty requests that is **9 813 calls and six drafts**. The counters are
recorded per repeat, and the three repeats are identical to the token, which is
why one set of figures covers all of them.

The draft-token column is the server's count over all thirty requests, and
`144/48`, `24/8` and `12/4` all give exactly three: by that counter each arm
produced one full-length lookup hit per repeat, whatever `size_m` was set to.
The drafter's own count is both larger and differently shaped (55, 15 and 10
tokens per repeat against the server's 48, 8 and 4, in 2, 2 and 3 drafts) so
the two counters disagree about how much was drafted here and not only about
how much was accepted.

The acceptance column carries two numbers because the two counters disagree,
and this arm is the worst case in the repository: the server reports **0.0 %**
and the drafter's own counter reports up to **70.0 %**. That divergence is not
specific to ngram-map; it appears on every path that takes a speculative
checkpoint and on none that does not ([A13](../ERRATA.md#a13-there-are-two-acceptance-counters-they-disagree-and-the-disagreement-is-exactly-the-checkpoint-path)).
An earlier version of this section reported the 0.0 % alone and called it
"acceptance"; that was wrong.

The draft-length question cannot be asked of a method that drafts twice in 3271
opportunities, so run N answers a different question than it was designed to: on
this workload the ngram-map families are a no-op costing one to two per cent.
That is a clean negative for two of the eleven `--spec-type` values and it closes
the coverage question, but it contributes nothing to the volume argument.

---

## Run O2 — the same matrix as a balanced Latin square

Run O below was three repeats with the arm list reversed on odd repeats. That is
forward/reverse/forward: the first arm sits at position 1 on two of the three, so
**arm position stays confounded with time**, and the run-to-run SD printed beside
each figure came from repeats that shared a position. An external review was
right that this cannot be called an estimator precision, and
[A14](../ERRATA.md#a14-within-run-repeats-are-not-an-error-bar) had already
measured one arm whose between-run spread was 8.6 pp against a within-run SD of
0.53.

O2 is the same nine arms over **nine blocks**, rotated by block index so every
arm occupies every position exactly once. That was checked before the run from
the schedule and after it **from the execution log**, which records the order
each block actually ran in:

```
block 0: baseline        spec-draft-n1 … ngram-map-k4v-m8
block 1: spec-draft-n1   spec-draft-n8 … baseline
…
block 8: ngram-map-k4v-m8 baseline     … ngram-mod-n24
```

Each figure is paired against the baseline measured **inside the same block**,
and the interval is a bootstrap and a t interval over blocks, the unit of
randomisation:

| arm | pooled tok/s | change | 95 % CI | acceptance † | draft tokens |
|---|---:|---:|---:|---:|---:|
| **`spec-dflash-n2`** | **146.2** | **+26.3 %** | [+25.5 %, +27.1 %] | 72.3 % | 21 969 |
| `spec-mtp-n2` | 141.9 | +22.7 % | [+22.1 %, +23.3 %] | 78.4 % | 20 916 |
| `spec-dflash-n4` | 137.9 | +19.2 % | [+18.5 %, +19.9 %] | 55.2 % | 33 489 |
| **no speculation** | **115.7** | — | — | — | 0 |
| `ngram-map-k4v-m8` | 115.4 | −0.3 % | [−0.6 %, +0.0 %] | 50.0 % | 216 |
| `ngram-mod-n24` | 103.1 | −10.9 % | [−11.4 %, −10.5 %] | 5.0 % | 5 184 |
| `ngram-cache` | 93.7 | −19.0 % | [−19.4 %, −18.6 %] | 5.2 % | 4 698 |
| `spec-draft-n8` | 30.9 | −73.3 % | [−73.5 %, −73.2 %] | 29.5 % | 50 112 |
| `spec-draft-n1` | 29.2 | **−74.8 %** | [−74.9 %, −74.7 %] | **69.7 %** | 13 410 |

**Balancing moved every estimate.** Against run O: +1.7 pp for
`spec-dflash-n2`, +0.9 for `spec-mtp-n2`, +1.2 for `spec-dflash-n4`, and
smaller shifts elsewhere. For `spec-mtp-n2` run O's +21.8 % falls **outside**
the interval above. The ordering of the arms is unchanged, and the conclusions
that rest on the ordering are unaffected, but the numbers themselves were
carrying a position effect.

**This is also the first run produced by the repaired harness**, and it is the
first directory the integrity checker calls *attested* rather than *legacy*:
the port was verified free before the server was spawned, liveness is checked
before a health response is accepted, the manifest records the ordering mode
and the exact prompt tag set, `RUN_COMPLETE.json` was written last, and every
one of the 81 arm-runs records the PID and the build read back out of that
server's own startup log, a single identity, `build 10622 (3737e4137)`, across
all of them. `matrix_report --strict` accepts it.

† Server-side acceptance counter; see the note under run O.

---

## Run O — the same matrix at three repeats, superseded by O2

Kept because it was published and because the difference between the two is the
point. Its design and its numbers are below; where they disagree with O2, O2 is
the one to cite.

## Run O — every method, one baseline, one policy

Runs C through N each answer one question well and sit at different contexts and
fitter margins, so every cross-method statement so far has had to be assembled
from deltas. This is the within-run comparison: nine arms, one baseline measured
beside all of them, `-c 8192`, `--fit-target 3072`, ABBA ordering, three
repeats, thinking on.

| arm | pooled tok/s | Δ pooled | aggregate tok/s | Δ aggregate | acceptance † | draft tokens |
|---|---:|---:|---:|---:|---:|---:|
| **`spec-dflash-n2`** — self-speculative | **145.8** | **+24.6 %** | 126.6 | +21.1 % | 72.3 % | 7 323 |
| `spec-mtp-n2` — the target's own MTP head | 142.5 | +21.8 % | 122.8 | +17.5 % | 78.4 % | 6 972 |
| `spec-dflash-n4` | 138.1 | +18.0 % | 119.6 | +14.5 % | 55.2 % | 11 163 |
| **no speculation** | **117.0** | — | **104.5** | — | — | 0 |
| `ngram-map-k4v-m8` | 116.1 | −0.8 % | 103.8 | −0.7 % | 50.0 % | 72 |
| `ngram-mod-n24` | 103.4 | −11.7 % | 93.5 | −10.5 % | 5.0 % | 1 728 |
| `ngram-cache` | 93.9 | −19.7 % | 85.9 | −17.8 % | 5.2 % | 1 566 |
| `spec-draft-n8` — external 0.8 B drafter | 30.8 | −73.7 % | 29.8 | −71.5 % | 29.5 % | 16 704 |
| `spec-draft-n1` — same drafter, one token | 29.1 | **−75.1 %** | 28.2 | −73.0 % | **69.7 %** | 4 470 |

† Server-side counter. It agrees with the speculator's own counter to within
0.5 pp on the three self-speculative rows, which take no speculative
checkpoints, and under-reports on the four rows that do: `spec-draft-n1` reads
69.7 % here and 100.0 % from the drafter, `ngram-map-k4v-m8` 50.0 % against
77.3 %. See [A13](../ERRATA.md#a13-there-are-two-acceptance-counters-they-disagree-and-the-disagreement-is-exactly-the-checkpoint-path).
No throughput figure in this table depends on either counter.

Three things this table settles that no earlier table could.

**The purpose-built draft paths win and the general-purpose external drafter
loses, on the same card in the same hour.** The spread is 145.8 down to 29.1
pooled, a factor of five, and it is not explained by acceptance, by draft length
or by the n-gram/model distinction.

It is also **not** explained by "whether the drafter is a second model", which an
earlier version of this section claimed. All three families load a separate
drafter GGUF through `-md`, and upstream describes MTP as a distinct model with
its own context and KV cache even when it is exported from the target's own
checkpoint. These arms differ at once in architecture, quantisation, parameters
activated per proposed token, reuse of the target's hidden states, rollback
behaviour, full-checkpoint policy and acceptance profile. Run O varies none of
those one at a time, so it establishes an **observational ranking for this
setup** and does not isolate a cause.

**The acceptance threshold is falsified inside a single matrix.**
`spec-draft-n1` sits at **69.7 % acceptance**, higher than every winning arm
except `spec-mtp-n2`, and is **75 % slower** on pooled decode rate, 73 % on aggregate. No reading of acceptance alone
survives that row. What separates it from `spec-dflash-n2` at 72.3 % is the
772 checkpoint saves and 709 restores it pays and DFlash does not
([A12](../ERRATA.md#a12-what-the-checkpoint-path-costs-measured-with-timers-in-the-source)).

**v1's methods are the three worst rows.** `ngram-cache`, `ngram-mod-n24` and
the external drafter are what this repository originally benchmarked, and they
occupy the bottom of the table. The original negative finding was correct about
what it measured; it measured the losing third of the available methods.

---

## Runs P and R — is the win a property of those ten prompts?

Every number above rests on the same ten prompts. 226 arm-runs of them. An
artefact of that mix would be invisible across all of them because they all
share it, and no number of repeats or arms fixes that. Runs P and R put the
same arms on a second, deliberately different set of **twenty** prompts: a long
input with a short required output, JSON and SQL with structure to hit, a regex
to explain, code in Rust, Python and shell, arithmetic with a checkable answer,
a logic puzzle, four languages, open-ended prose, three instruction-following
edge cases, and **two genuinely multi-turn exchanges** — which the v1 set never
had (C3). The gate on the run checked that the multi-turn history actually
reached the model, by requiring it to recall a name and a GPU stated four turns
earlier.

### The metric has to change, and that is not a detail

The extended prompts are longer: **4323 prompt tokens against 1035**, so prompt
processing is **12.7 %** of wall-clock instead of **6.7 %**. Aggregate
throughput divides by wall-clock and therefore carries that term. Speculation
does nothing for prompt processing, so a set with more of it mechanically
compresses the aggregate delta; for reasons that have nothing to do with the
method.

Read on aggregate, the result looks like a collapse. Read on **pooled** decode
rate, which is tokens over decode time and is the only quantity comparable
across prompt sets, it is close to flat:

| arm | v1 ten, pooled | extended twenty, pooled | shift | *(aggregate, not comparable)* |
|---|---|---|---|---|
| `spec-dflash-n2` | +24.6 % | **+20.3 %** | −4.3 pp | *+21.1 % → +10.0 %* |
| `spec-dflash-n4` | +18.0 % | **+21.3 %** | **+3.3 pp** | *+14.5 % → +6.2 %* |
| `spec-mtp-n2` | +21.8 % | **+21.0 %** | −0.8 pp | *+17.5 % → +3.7 %* |
| `spec-draft-n8` | −73.7 % | −73.1 % | +0.6 pp | *−71.5 % → −69.9 %* |

**The result generalises.** Across two prompt sets sharing no prompt, the decode
speed-up moves by at most 4.3 pp, and for `spec-dflash-n4` it moves *upward*.
(Two of the four shifts are positive; the other is `spec-draft-n8` becoming
slightly less bad. An earlier version of this sentence said one.) Had this been
written up on aggregate it would have read "the win halves on a different prompt
set", which is false and would have been a metric artefact of exactly the kind
this repository exists to catch.

Three things were checked before trusting that comparison, because "use the
other metric" is the sort of move that can be made to say anything:

- **`predicted_ms` contains the draft cost**, so pooled does not quietly exclude
  the thing being measured. Decode plus prompt accounts for 96–99 % of
  wall-clock in every arm, and `spec-draft-n8` spends **292.1 s** of decode
  against the baseline's 76.9 s, if drafting sat outside `predicted_ms` that
  arm would look faster, not four times slower.
- **Every request in both sets reaches the 300-token cap** (`finish_reason:
  length`, 30/30 and 60/60), so the two sets generate identical token counts and
  differ only in how much prompt precedes them.
- **The new prompts do not inflate acceptance.** `spec-dflash-n2` accepts 72.3 %
  on the v1 ten and 72.8 % on the extended twenty; `spec-mtp-n2`, 78.4 % and
  77.3 %. A set written by the same hand that wrote the analysis could have been
  tuned to be predictable; measurably it is not.

The workload control repeats on the new set too. Thinking off, pooled:

| arm | v1 ten (run M3) | extended twenty (run R) |
|---|---|---|
| `spec-dflash-n2` | +8.5 % | **+8.6 %** |
| `spec-mtp-n2` | +11.4 % | **+13.2 %** |
| `spec-draft-n8` † | −77.8 % | −74.6 % |

† Run M3 does not carry the external drafter, so its v1-ten thinking-off figure
is run L's rather than M3's, and that row compares two runs on two days instead
of one run's two prompt sets. It read −75.1 % until 2026-08-29, which is run L's
**thinking-on** figure for the same arm: the wrong run and the wrong workload.

So the prompt *mix* moves these arms by 0.1, 1.8 and 3.2 pp while the *workload*
takes two thirds of the advantage away, on both sets. The two figures that come
from one run and differ only in the prompt set are the first two.

---

## Run Q — the anomaly this repository could not explain, resolved

Run M reported a Q4_K_M MTP head at `n_max 4` moving 6.8 pp against Q8_0 at
unchanged acceptance, with run-to-run SDs that did not cover it, and said so
rather than explaining it. Five repeats of each drafter dissolve it:

| arm | drafter | 3 repeats (M1 / M4) | **5 repeats (run Q)** | difference |
|---|---|---|---|---|
| `spec-mtp-n2` | Q8_0 | +22.1 % | **+21.6 %** | 0.5 pp |
| `spec-mtp-n2` | Q4_K_M | +26.6 % | **+27.0 %** | 0.4 pp |
| `spec-mtp-n4` | Q4_K_M | +3.6 % | **+3.6 %** | 0.0 pp |
| `spec-mtp-n4` | Q8_0 | **+10.5 %** | **+2.0 %** | **8.6 pp** |

The difference column is the difference of the measurements, not of the two
rounded cells beside it, which is why the last row reads 8.6 where subtracting
what is printed gives 8.5. It said 8.5 until 2026-08-29; the other three rows
are the same either way, which is why nothing showed.

Three of the four reproduce to within 0.5 pp. The fourth does not, and **the
non-reproducing measurement is the one that created the anomaly.** A third,
independent measurement of that same arm on the extended prompt set (run P)
reads **+2.7 %**, so +2.0 %, +2.7 % and +3.6 % cluster and run M1's +10.5 % is
the outlier.

With it removed the picture is simple and consistent: **the Q4_K_M head is
better than the Q8_0 head at both draft lengths** — +27.0 % against +21.6 % at
`n_max 2`, +3.6 % against +2.0 % at `n_max 4`. The smaller head drafts about as
well and costs less to run. There is no anomaly to explain; there was one
measurement that did not replicate, which is a different and more useful thing
to know, and it is what
[A14](../ERRATA.md#a14-within-run-repeats-are-not-an-error-bar) is about.

---

## What is settled, and what is not

Settled by runs A–O, each against a matched no-speculation baseline measured
beside it:

- the acceptance artefact and its upstream fix (A1), the vocabulary defect (A2),
  the abort (A6), the workload the archive never controlled (A5, D3b)
- **nine of master's eleven `--spec-type` values measured**; the other two,
  `draft-eagle3` and `draft-dspark`, are blocked for reasons read out of the
  source, not assumed
- self-speculation wins on this card and external speculation loses, by a factor
  of five, in one matrix (run O)
- the mechanism, measured rather than argued: state checkpointing that only the
  external-drafter path pays (A12)
- batching does not rescue speculation and widens the gap (run I); it removes
  DFlash's advantage entirely and MTP's partly (runs K, M)
- the win is a property of the workload as much as the method: it halves with
  thinking off, and the DFlash/MTP ranking flips (runs L, M)
- speculation is not output-preserving on this build, against a determinism
  control that holds in every run (A11)

Not settled, and honestly out of reach here:

| gap | why it is still open |
|---|---|
| three repeats on most arms | five on runs L and M3, **nine on run O2**, the balanced matrix the headline uses |
| one host, one card, one quantisation | nothing here separates the model, the quantisation and the GPU |
| ~~`n_max 4` under a Q4_K_M MTP head~~ | **closed by run Q.** It was one Q8_0 measurement that did not replicate, not a drafter-precision effect ([A14](../ERRATA.md#a14-within-run-repeats-are-not-an-error-bar)) |
| ~~ten prompts~~ | **closed by runs P and R.** Twenty different prompts, sharing none with the v1 set, move the decode speed-up by at most 4.3 pp |
| ~~`multi_turn_1` / `multi_turn_2`~~ | **closed for new runs.** The extended set carries two genuinely multi-turn exchanges, gated on the model recalling four-turn-old context. The v1 tags keep their names and their behaviour so archived joins still work |
| between-run reproducibility | median 0.55 pp over twelve independently repeated groups, and one at 8.57 pp that resisted every check ([A14](../ERRATA.md#a14-within-run-repeats-are-not-an-error-bar)) |
| ~~the wall-clock cost of checkpointing~~ | **closed by runs T and T3.** The timers upstream left commented out at `server-context.cpp:2963` and `:2967` were uncommented and three more added: **39.07 s of a 71.4 s excess, 54.7 %**, replicated at 54.6 % in a second balanced run ([A12](../ERRATA.md#a12-what-the-checkpoint-path-costs-measured-with-timers-in-the-source)) |
| the unattributed 21 % of the external drafter's excess decode time | 54.7 % is checkpoint work and 24.2 % is the drafter's own `generate()`; the remainder is verification of discarded tokens and scheduling, in unknown proportion. The figure was "the other 76 %" before the timers existed |
| why two runs of the same configuration differ by 3.4 % on one arm | identical binary, identical models, identical fit, identical clocks, byte-identical output ([A16](../ERRATA.md#a16-two-runs-identical-in-every-recorded-respect-and-byte-identical-in-output-differ-by-34--on-one-arm)). Nothing recorded distinguishes them |
| every thinking-off comparison here | the arms generated different numbers of tokens, and controlling for it moves each model-drafting arm by +2.5 to +16.8 pp and flips one published sign ([A17](../ERRATA.md#a17-the-thinking-off-comparisons-are-not-comparisons-of-the-same-amount-of-work)). `BENCH_IGNORE_EOS` measures it properly; no archived run used it |
| expert routing | never instrumented, and after A7 and A12 nothing demands it |

Absolute rates must not be compared across runs that differ in `-ngl`, `-c` or
`--fit-target`; the table in [`../BENCHMARK_ENV.md`](../BENCHMARK_ENV.md) says
which is which. The no-speculation baseline is also ~8 % faster on master than
on `bcb5eeb64` on the same host.

## Files

This table listed two of the run directories and the harness. There are
77, of which 74 are runs and three are start-up checks.

| Path | Contents |
|---|---|
| `data/<run>/` | one directory per run — `manifest.json`, one `<arm>__rep<n>.json` per arm-run, and `RUN_COMPLETE.json` on the runs the harness validated. 77 directories, 3005 arm-runs. |
| `data/matrix_O2_latin_*/`, `data/matrix_O3_latin_*/` | the balanced nine-arm matrix the README leads with, and its five-hours-later replication on the same stock binary — 810 of 810 request-pairs byte-identical. Each carries its own `paired_blocks.json` |
| `data/matrix_T_timers_*/`, `data/matrix_T3_timers_*/` | the two source-timed checkpoint runs behind [ERRATA A12](../ERRATA.md#a12-what-the-checkpoint-path-costs-measured-with-timers-in-the-source) and [A16](../ERRATA.md#a16-two-runs-identical-in-every-recorded-respect-and-byte-identical-in-output-differ-by-34--on-one-arm), with `checkpoint_timers.json` and the SHA-256 of every log they were extracted from |
| `data/matrix_U*_dflashvar_*/` | six independent invocations of one configuration, the designed test behind [A16](../ERRATA.md#a16-two-runs-identical-in-every-recorded-respect-and-byte-identical-in-output-differ-by-34--on-one-arm) |
| `data/matrix_V_freerun_*/`, `data/matrix_V_hardcap_*/` | the same five arms with and without `ignore_eos`, which measures the length confound of [A17](../ERRATA.md#a17-the-thinking-off-comparisons-are-not-comparisons-of-the-same-amount-of-work) instead of subsetting it |
| `data/checkpoint_timers_20260826.json` | run T's twelve timer records, four repeats per arm |
| `data/gpu_telemetry_*.csv` | the continuous nvidia-smi traces the thermal claims are computed from, one per session. `analysis/thermal_report.py` reads all three schemas the audit recorded |
| `data/acceptance_counter_comparison.json` | the two acceptance counters, side by side ([A13](../ERRATA.md#a13-there-are-two-acceptance-counters-they-disagree-and-the-disagreement-is-exactly-the-checkpoint-path)) |
| `data/A_bcb5eeb64_legacy/` | manifests and per-request JSON, run A |
| `data/B_master_3737e4137/` | manifests and per-request JSON, run B |
| `data/abort_evidence_bcb5eeb64.txt` | the CUDA abort in context, both arms |
| `patches/checkpoint_timers.patch` | the only non-stock binary in this repository, with the reasoning beside it |
| `../bench/retest_runner.py` | the harness |
| `../bench/collect_evidence.sh` | builds the SHA-256 manifest and archive for the ~3 GB of server logs, which are too large to commit |
| `../analysis/` | the checkers: `verify_claims.py`, `check_data_integrity.py`, `paired_blocks.py`, `matrix_report.py`, `plot_v4_runs.py` |

The server logs themselves are **not** committed: **4071 MB across 702 files**.
[`EVIDENCE_MANIFEST.sha256`](EVIDENCE_MANIFEST.sha256) holds the SHA-256 of
every one of them and of the 19 telemetry traces, and each attested run
additionally records `server_log_sha256` per arm-run, so a single log can be
checked without the manifest. `bash bench/collect_evidence.sh ~/bench`
regenerates both and builds `raw_logs.tar.zst`, **271 028 599 bytes**, sha256
`29c2401f100390268bbd52e43b5c2da9a61440bad3dabe502ca1684478771fd6`, with
`telemetry.tar.zst` at **162 320 bytes**, sha256 `8a29cc875e30bc66c6e83913b1bf40b075295218a37eef37897317805b47d03c`.

**That archive is published**, with the 19 telemetry traces beside it, as an
asset of the release `raw-evidence-2026-08-27`. A second tranche sits beside
it: `raw_logs_20260827.tar.zst`, **187 358 414 bytes**, sha256
`d56a7f88a099550bdab229ccb2bd36840f167550cea7689f575fd6d0f11da8ff`, and
`telemetry_20260827.tar.zst` at **46 603 bytes**, sha256
`db833395470aeaf842225a33fb544e933f1c0a3047cac161524aca1ac1aef061`, carrying
the **618 logs and 2 traces** of the runs the third review asked for: the
eight-session crossover V2, the within-invocation V3 and the split-timer T4.
Two archives rather than one rebuilt archive, so the first one's digest keeps
meaning what it meant when it was published; unpack both into one bench root
and this repository's whole manifest verifies, and all **620** of the second
tranche's entries did before it was published. A third tranche followed on
2026-08-28 for run W: `raw_logs_20260828.tar.zst`, **221 242 327 bytes**,
sha256 `5af671bf3cf47a20fa2ca78504c089642b9bb2ea249b8997577b4852caa7a5c2`, 500
logs, with `telemetry_20260828.tar.zst` at **58 905 bytes**, sha256
`72e331bf0cbfaaee73619acf320523598cab4a46b7a7daa51fabbd6e2c455bf3`, and all
**501** of its entries were verified against the unpacked archive before
publishing, which is the check that was missing when the manifest named them
one commit early. A fourth followed on 2026-08-31 for run W2:
`raw_logs_20260831.tar.zst`, **365 852 615 bytes**, sha256
`524ce5db75d494028b7b596f45e98b2384d8234726a924135775cce4f85b4cda`, 1200
logs, with `telemetry_20260831.tar.zst` at **114 587 bytes**, sha256
`4abdb6d701171bf0178ae36dad86f3190801cc047e88220d16f5442b7189949b`, and all
**1201** of its entries were verified against the unpacked archive before
publishing. It was hashed into `EVIDENCE_MANIFEST.sha256` a commit before it
was packaged, and for that one commit the manifest carried a marker naming
exactly what was not yet published rather than leaving the gap to be inferred
from a count. All four tranches are prepared as assets of
`v4.2`, cut at the commit that carries the dataset and the verifier together,
which is this branch's head and not the merge: a merge does not change the
tree, and waiting for one would only delay the check that the tag publishes
what it names. When it exists, one release identity verifies the whole
manifest. Until then the manifest is verified against the three published
tranches and the fourth out of band, which is what the evidence workflow does.

Committed hashes tie the derived JSON
to files nobody else could see, which is a weaker claim than it sounds; the
point of publishing is that the extraction can be re-run rather than trusted.
`analysis/rederive_from_logs.py <bench-root>` does exactly that, and this is
what it reproduces from the archive alone:

| derived file | records | identical | not reproducible |
|---|---:|---:|---|
| `data/spec_accounting_20260826.json` | 12 | **12** | — |
| `data/checkpoint_timers_20260826.json` | 12 | **12** | — |
| `data/checkpoint_timers_20260827_split.json` | 18 | **18** | — |
| `data/acceptance_counter_comparison.json` | 535 | **526** | 9 |

Zero records differ. The nine that are not reproducible belong to runs **G**,
**I** and **J**, three exploratory runs whose logs are in the archive but whose
arm-run JSON is not committed; the extractor needs both, and those three runs
never completed their cell set, so committing them would attest to runs that
are not whole. `analysis/check_data_integrity.py` refuses them for exactly that
reason. The runs are `matrix_G_dflash_20260826_000124`,
`matrix_I_conc1_20260826_012917` and `matrix_J_dflash_fit_20260826_014308`, and
they contribute nine of the 535 rows behind [A13](../ERRATA.md#a13-there-are-two-acceptance-counters-they-disagree-and-the-disagreement-is-exactly-the-checkpoint-path);
the claim there survives on the other 526.

> [!IMPORTANT]
> **That table was produced by running the script. CI has now reproduced it.**
> [`.github/workflows/evidence.yml`](../.github/workflows/evidence.yml) fetched
> the archive, checked it against the manifest, unpacked it, re-derived the
> committed JSON from the raw logs and ran the claim checker over the result.
> Its **first run was on 2026-08-28**, and it did all of that: it failed on the
> last step alone, the chart comparison, and passed in full thirty minutes
> later. This note used to end **It has failed since 2026-08-31**, and it had,
> for two reasons in turn. The tag it names became `v4.2`, which had not been
> cut, so it stopped at the download and skipped every later step. Once the
> release existed it got past that and stopped at the manifest check instead,
> on a `FileNotFoundError` for `/tmp/manifest`: the manifest is split into
> three slices and that call kept the name of the file that used to hold all
> of it. Both are fixed; the last failing run was on 2026-09-01. This note
> also said that `workflow_dispatch`, `release: published` and the weekly
> cron all read the workflow from the
> **default branch**, and this file lives only on `audit-2026-08-25` until it
> merges. The dispatch and the cron did; a `release` reads the tag's own ref
> and fired twice before the merge. What fired here was `push`, which reads
> the workflow from the ref being pushed and lists what the checker imports.
>
> **All three are live since the merge on 2026-09-01, and the first
> `workflow_dispatch` has run.** It fetched the archives, checked every file
> against the manifest, unpacked them, re-derived the four log-derived files,
> required the result to be what is committed, and ran the claim checker over
> it: `success`. The binding step skipped in that run and was right to, because
> a manual dispatch runs from a branch and that step is for a tag. The figures
> above are also what `python analysis/rederive_from_logs.py <bench-root>`
> prints after unpacking both published tranches into one directory, which is a
> command anyone with the release can run.

Four run directories are archived under their bench-host names rather than the
descriptive ones used here: `C_master_matrix_think_on`,
`D_master_matrix_think_off`, `E_past_threshold` and `H_pmin_sweep` are
`matrix_C_20260825_204529`, `matrix_D_20260825_204529`,
`matrix_E_threshold_20260825_224802` and `matrix_H_pmin_20260826_005716` in the
archive. Thirteen of the nineteen telemetry traces are committed here byte for
byte; six (the aborted IJ, K, MN, N, T2 and T3 traces) are in the release only.

---

## Appendix: the evidence sections moved out of the root README

These eleven sections were in [`README.md`](../README.md) until
2026-09-01. They are the argument for the headline and the tables behind
it, and a root README that carries them is 7 569 words before its first
reproduction step. They are moved rather than summarised: every number
that was in them is here, which was checked by taking each one out of the
old file and requiring it to still exist in the tree.

### What supports that result, and what limits it

The headline in [the root README](../README.md#the-result) is one invocation
of one matrix. These four things are
what makes it believable and what stops it being a general claim: a
replication five hours later, twelve measurements of the same arm in one
day that span nine percentage points, a prompt set that is only ten prompts,
and a designed follow-up that removes the one confound the first three
leave open. This section used to sit inside [The result](../README.md#the-result), which was
then called "Result in one sentence" and had grown to three thousand words.

**It was run twice.** Run O3 is the same nine arms, nine balanced blocks, same
stock binary and same models, five hours later, with the harness asserting the
library hash on **every arm-run** rather than once. All **810 request-pairs are
byte-identical** to O2 (same token ids, same text), and acceptance matches to a
tenth of a point on every arm. What moves is the time:

| arm | O2 | O3 | shift |
|---|---:|---:|---:|
| `spec-dflash-n2` | +26.3 % | **+23.4 %** | **−2.9 pp** |
| `spec-mtp-n2` | +22.7 % | +21.7 % | −1.0 pp |
| `spec-dflash-n4` | +19.2 % | +18.3 % | −0.9 pp |
| `ngram-cache` | −19.0 % | −19.7 % | −0.7 pp |
| `ngram-mod-n24` | −10.9 % | −11.5 % | −0.5 pp |
| `ngram-map-k4v-m8` | −0.3 % | −0.6 % | −0.3 pp |
| `spec-draft-n8` | −73.3 % | −73.5 % | −0.2 pp |
| `spec-draft-n1` | −74.8 % | −75.0 % | −0.2 pp |
| *no speculation, absolute* | *115.7* | *116.5* | *+0.7 %* |

Eight arms move by 0.2 to 1.0 pp. **`spec-dflash-n2` moves by 2.9**, and it did
the same thing between runs T and T3, by 5.2 pp, also on byte-identical output.
Whatever this is, it is specific to that arm and it is reproducible, and nothing
recorded distinguishes the runs:
[ERRATA A16](../ERRATA.md#a16-two-runs-identical-in-every-recorded-respect-and-byte-identical-in-output-differ-by-34--on-one-arm).

The design matters and the numbers show it: run O measured the same arms at
three repeats with the arm list merely reversed on odd repeats, which leaves
position confounded with time. Its point estimates were 0.3–1.7 pp away from
these, and for `spec-mtp-n2` its +21.8 % falls **outside** the interval above.

† This table is run O2, and **the intervals in it describe run O2, not the
configuration.** `spec-dflash-n2` was measured **twelve times** on 2026-08-26
under this memory policy, on the same target, drafter and prompts, thinking on,
one request at a time:

| | | | | | |
|---|---|---|---|---|---|
| M1 **+26.7 %** 07:59 | O **+24.6 %** 09:00 | O2 **+26.3 %** 15:37 | T **+25.9 %** 18:26 | T3 **+20.7 %** 20:32 | O3 **+23.4 %** 20:44 |
| U1 **+22.3 %** 22:12 | U2 **+24.2 %** 22:15 | U3 **+17.3 %** 22:18 | U4 **+19.9 %** 22:21 | U5 **+25.6 %** 22:24 | U6 **+24.3 %** 22:27 |

**Range 9.4 pp, SD 2.9.** The no-speculation baseline over the same twelve runs
holds 115.72–117.25 tok/s, a CV of **0.42 %**: the reference is steady and the
arm under test is not. Every one of the twelve produced byte-identical output,
and `draft_n` is 2441 with acceptance 72.3 % in all 43 of their blocks: the
speculative work is the same to the token and only the time differs.

Pooling those 43 blocks gives **+17.0 % to +27.8 %**, and the values cluster by
run rather than scattering inside one: split at +23 % and **eleven of the twelve
runs fall wholly on one side**, averaging +25.7 % above and +20.3 % below. Run
O3 is the one that crosses, at block 4, and **in those blocks only this arm
moves**, including `spec-dflash-n4`, the same drafter at twice the draft length,
which never leaves ±1.01 % of its own first block. Whatever it is survives the
server restart between arm-runs, so a single measurement lands wherever it
happens to be.

The paired-block interval above is 1.6 pp wide and it is measuring the wrong
variance component. Read the configuration as **+17 % to +27 %**, and the
interval as within-invocation precision only:
[ERRATA A16](../ERRATA.md#a16-two-runs-identical-in-every-recorded-respect-and-byte-identical-in-output-differ-by-34--on-one-arm).

O2 is quoted because the documents are built on it and because run O3 replicates
it arm for arm; it is not the lowest. Run U3 is, at +17.3 %. Runs K1 and L read about +21 % for the same arm and are
excluded from the comparison above: they ran at `--fit-target 2048`, a different
memory policy, which [`BENCHMARK_ENV.md`](../BENCHMARK_ENV.md) records as a variable
across runs. Until 2026-08-26 this footnote quoted run O's +24.6 % as though it
were this table's own figure. It was written when run O *was* the headline table
and was not updated when O2 replaced it. What a three-repeat delta is actually
worth is measured in
[ERRATA A14](../ERRATA.md#a14-within-run-repeats-are-not-an-error-bar).

† Server-side acceptance counter. It agrees with llama.cpp's other counter to
0.5 pp on the self-speculative rows and under-reports on the rest; the divergence
tracks the speculative-checkpoint path exactly
([ERRATA A13](../ERRATA.md#a13-there-are-two-acceptance-counters-they-disagree-and-the-disagreement-is-exactly-the-checkpoint-path)).
No throughput figure depends on either counter.

A factor of five separates the top from the bottom, and it is **not** explained
by acceptance, by draft length, or by model-versus-n-gram. What it is explained
by, this matrix cannot say: the purpose-built DFlash and MTP draft paths and the
general-purpose 0.8 B drafter differ simultaneously in architecture,
quantisation, parameters activated per proposed token, reuse of the target's
hidden states, rollback behaviour, full-checkpoint policy and acceptance
profile, and nothing here varies them one at a time.

All three are separately loaded draft models: the harness passes `-md <GGUF>`
for every one of them, and upstream describes MTP as a distinct model with its
own context and KV cache even when it comes from the same file. An earlier
version of this section said the divide was "whether the drafter is a second
model"; that is simply false of these arms. `spec-draft-n1` accepts 69.7 % of
its drafts, third-highest in the table, above one of the three winning arms and
below the other two, and is 75 % slower, because a separate
draft context makes this hybrid target save and restore a full checkpoint on
every partially accepted round. The server reports 82.079 MiB per checkpoint,
772 creates and 709 restores in one arm-run, which DFlash logs zero times at
draft lengths 1 to 16 and MTP zero times at 1 to 8
([ERRATA A12](../ERRATA.md#a12-what-the-checkpoint-path-costs-measured-with-timers-in-the-source)).

**Every arm here that represents a method v1 benchmarked is below baseline.**
That is four rows, not three: v1 tested an external draft model, `ngram-cache`
and `ngram-mod`, and the external drafter appears twice: `spec-draft-n8` and
`spec-draft-n1` are two configurations of one method, with `ngram-cache` and
`ngram-mod-n24` above them. The original negative finding was right about what
it measured. What it did not measure is DFlash and MTP, which is where the wins
are.

**And it is not a property of those ten prompts.** Every number above and every
number this repository has ever published rests on the same ten. Repeated on a
second set of twenty sharing none of them (long inputs, JSON and SQL, four
languages, arithmetic, two genuinely multi-turn exchanges), the decode speed-up
moves by at most 4.3 pp, and for `spec-dflash-n4` it moves *upward*
(two of the four shifts are positive; the other is `spec-draft-n8` becoming
slightly less bad)
([runs P and R](../v4_audit_2026_08_25/README.md#runs-p-and-r--is-the-win-a-property-of-those-ten-prompts)).
That comparison has to be made on pooled decode rate: the longer prompts put
12.7 % of wall-clock into prompt processing against 6.7 % for the v1 ten, and
aggregate throughput divides by wall-clock, so reading it there would have shown
a collapse that is an artefact of prompt length and not of the method. v1 never tested that method, and the archived v3 attempt at it compared
two different binaries. The sign flips with the draft window: +18.7 % at 4,
−14.8 % at 8, −47.4 % at 16, so "speculative decoding loses here" was a
statement about draft-window regimes that this repository had not yet separated.

One qualification travels with that number, and with every speculative
measurement here: **speculation is not output-preserving on this build.** The
engine is deterministic. Every arm reproduces itself byte-for-byte across
repeats, and the no-speculation baseline reproduces across separate runs.
Against that control, turning speculation on changes the generated text in 27 to
30 of 30 request-pairs. All arms still emit exactly 300 tokens and the baseline's
decode rate varies only 0.8 % across ten very different prompts, so the
throughput comparison stands; but this is a faster computation landing on
slightly different text, not a lossless speedup of the same one
([ERRATA A11](../ERRATA.md#a11-speculative-decoding-is-not-output-preserving-on-this-build-and-the-engine-is-deterministic-enough-to-prove-it)).
See [`v4_audit_2026_08_25/README.md`](../v4_audit_2026_08_25/README.md#run-j--the-first-configuration-that-is-actually-faster).

The one lever upstream names as the fix, batching, was also tested, and does
not help: no speculation gains +64 % at concurrency 8 while the matched-vocabulary
drafter moves −8 %, so the gap widens rather than closing
([run I](../v4_audit_2026_08_25/README.md#run-i--batching-the-lever-upstream-names)).
It does not rescue the winner either. A sweep down to `n_max 1` puts DFlash on a
plateau (+17.1 %, +17.6 %, +17.3 % at 2, 3 and 4, separated by less than the
baseline's own run-to-run SD) and a cliff between 4 and 6; batching then erases
the plateau at four concurrent requests (+0.4 %) and collapses it at eight
(−74.1 %), with draft volume and acceptance barely moving, so it is the draft
*cost* that fails to amortise
([run K](../v4_audit_2026_08_25/README.md#run-k--where-the-optimum-is-and-what-batching-does-to-it)).
On this card, speculation pays for one stream at a time or not at all.

And the win belongs to the workload rather than to the method. Repeated with
thinking verifiably off on all 250 requests, `n_max 2` falls from +21.1 % to
+7.6 % and `n_max 4` goes **negative** at −2.7 %, tracking draft acceptance down
with it (72.8 % → 58.5 %, 55.6 % → 40.3 %). Per prompt, step-by-step arithmetic
and Python keep their full gain, their output being constrained and their
acceptance staying between 82 % and 92 %, while Traditional Chinese free prose
goes from +15 % to −25 % as
acceptance falls from 66 % to 29 %. Reasoning text is enumerated, repetitive
planning prose, which is exactly what a drafter predicts well
([run L](../v4_audit_2026_08_25/README.md#run-l--the-win-is-a-property-of-the-workload-not-of-the-method)).

Across run L's 60 points acceptance and speed-up correlate at **r = +0.946** and
the line crosses zero at **48.2 % acceptance**. Half those points come from run
L's thinking-off half, where the arms generated different numbers of tokens
([ERRATA A17](../ERRATA.md#a17-the-thinking-off-comparisons-are-not-comparisons-of-the-same-amount-of-work));
refitting without the confound moves the crossing to **46.5 %** on the
length-matched prompts and **45.4 %** on the thinking-on half alone, while the
slope moves nearly three times as far in relative terms. The threshold is the
stable quantity and the slope is not, which is the same thing A10 found out of
sample. Read it as **45–48 %**, not as 48. Scored across every (run, arm) with a matched baseline and enough drafts to
define a rate (86 of 90, the four excluded having drafted at most 45 tokens in
total against at least 132 for the rest), it calls the sign **57 / 59 inside the
self-speculative families**, **13 / 16 on the external drafter** and **8 / 11 on
the drafter-free n-gram arms**: **78 / 86 overall**, or 79 / 86 read through
llama.cpp's other acceptance counter.

The eight it misses are the informative part, and they are two kinds. Three are
`spec-draft-n1`, which reaches **69.7 % acceptance** (100.0 % by the drafter's
own counter) and is **75 % slower**: the same arm in runs O, O2 and O3, and the
structural failure this section is about. The other five sit **within 2 pp of
the boundary**: `ngram-map-k4v-m8` at 50.0 % three times while moving −0.3 to
−0.8 %, `spec-mtp-n4` at 49.5 % and −8.2 %, and run V's `spec-dflash-n4` at
47.9 % and **+10.6 %**. A threshold is uninformative at its own boundary; that
is what those five say.

> Published as **35 / 37** until 2026-08-27. That scorecard came from
> `analysis/compare_acceptance_counters.py`, which read only `*__rep0.log` and
> predated runs O2, O3, T, T3, U and V. Over every repeat of every run the base
> grows from 37 to 86 and the hit rate falls from 94.6 % to 90.7 %.

So the threshold is not a law about acceptance; it tracks the drafter. What is
measured is that a separate draft context makes this hybrid target log **772
full-checkpoint creates and 709 restores in one ten-prompt arm-run**, at a
reported 82.079 MiB each, a nominal **118.7 GiB** by event count × logged size,
which is an estimate and not measured memory traffic. DFlash logs none of these
events at draft lengths 1 to 16 and MTP none at 1 to 8. What that costs in wall
clock **is** measured, by rebuilding llama.cpp with timers around the four
calls: **39.07 s of a 71.4 s excess, 54.7 %**, replicated to 54.6 % in a second
balanced run. That is elapsed time *inside the checkpoint API calls*, which at
this commit begin with `ctx->synchronize()`, so it includes waiting for queued
backend work as well as the copying, which makes it an attribution to the API
boundary rather than an isolated measurement of checkpoint memory traffic
([A12](../ERRATA.md#a12-what-the-checkpoint-path-costs-measured-with-timers-in-the-source)).
This sentence said "not established here" until 2026-08-26, which was true when
it was written and stopped being true when run T was measured. And the
external drafter is 0.8 B *dense* against a target that activates only ~3 B
parameters per token, so drafting costs a quarter of a target step before any
state management: 17.24 s in `generate()` in run J2 against 1.89 s for MTP at
`n_max` 1 and 3.43 s for DFlash at 2, the two winning configurations. Across
every self-speculative arm measured the span is 1.89 s to 6.27 s, still a
fraction of the dense drafter's
([ERRATA A12](../ERRATA.md#a12-what-the-checkpoint-path-costs-measured-with-timers-in-the-source)).

![v1 300-token matrix: request-mean vs pooled throughput](../analysis/plot_mean_by_config.png)

---

### Metric definitions

llama-server reports, per request,
`predicted_per_second = 1000 × predicted_n / predicted_ms`. `predicted_ms`
covers target decoding plus the speculative proposal, verification, and
bookkeeping. It excludes prompt prefill, queueing, network transport, speech
I/O, and application latency.

Three summaries are reported for every config:

- **Request-mean decode rate**: arithmetic mean of each request's
  `predicted_per_second`. Every prompt weighs the same.
- **Pooled decode throughput**: `1000 × Σ predicted_n / Σ predicted_ms`. Every
  generated token weighs the same. For equal-length outputs this is the
  harmonic mean of the per-request rates.
- **Min–max across prompts**: workload heterogeneity. One measurement per
  prompt/config, so it is **not** repeated-run uncertainty, a standard error,
  or a confidence interval.

`draft_n` and `draft_n_accepted` need their own definition, because they do not
mean what their names suggest; see
[The "100 % acceptance" retraction](#the-100--acceptance-retraction).

---

### What the v1 data support

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

### What the v1 data do not support

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
  ([A2](../ERRATA.md#a2-the-draft-model-was-not-vocabulary-compatible-the-run-used-the-token-translation-fallback)).
- A claim about the workload, because 76 % of the requests returned truncated
  thinking rather than an answer, and no version of this benchmark ever
  verified what was being generated
  ([A5](../ERRATA.md#a5-three-quarters-of-v1s-requests-returned-no-answer-at-all--only-truncated-thinking)).
- A statement about a *working* speculative path for this model class, because
  the fix for the hybrid-SSM partial-acceptance failure the runs actually hit,
  llama.cpp PR #20075, was closed without merge
  ([A3](../ERRATA.md#a3-the-tested-build-had-a-known-broken-speculative-path-for-this-model-class-and-the-fix-was-never-merged)).
- A production voice-agent recommendation. This measures decode
  microperformance, not streaming TTFT, audio latency, multi-turn cache reuse,
  concurrency, or output quality.
- Behaviour under non-greedy sampling, other seeds, current llama.cpp, or
  untested speculation methods.

---

### The "100 % acceptance" retraction

**Earlier versions of the root README said:** every tested configuration returned
100 % draft acceptance, so "high acceptance → high speedup" fails, and "this is
not a measurement artifact; it is MoE expert-loading overhead on every drafted
token."

**It is a measurement artefact.** On this model the ratio can only ever be 1.0.

![What the 100 % acceptance number actually counts](../analysis/plot_acceptance_accounting.png)

Qwen3.6-35B-A3B is a hybrid Gated-DeltaNet / MoE model, so
`common_context_can_seq_rm()` returns `COMMON_CONTEXT_SEQ_RM_TYPE_FULL`: the
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
[`analysis/summary.csv`](../analysis/summary.csv), `draft_n` means "draft tokens in
verification rounds that were accepted in full", a quantity guaranteed to equal
`draft_n_accepted`. `draft_n = 0` means "no fully accepted round was recorded",
**not** "speculation did not run". The retracted
`analysis/plot_accept_vs_speed.png`, every one of whose 140 points sat at
exactly 100 %, has been deleted.

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
`n_max` 32 and `n_max` 128 are byte-identical, at 6159 drafted tokens, 70.9 %
acceptance and 42.0 tok/s each. See
[A10](../ERRATA.md#a10-the-single-regressor-law-is-falsified-out-of-sample-and-p_min-is-the-lever-that-matters).

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

### Where the time goes, measured, and not MoE-specific

The audit's re-measurement ([A7](../ERRATA.md#a7-with-acceptance-measured-properly-there-is-no-anomaly-left-to-explain))
removed the mystery: decode rate tracks acceptance at r = +0.998 across the
prompt set. What the 2026-08-26 runs add is where the extra time actually goes,
timed in the source rather than inferred from log intervals.

An external 0.8 B drafter spends **71.4 s more in decode** than no speculation
does, over one ten-prompt arm-run of 3000 tokens:

| | seconds | share |
|---|---|---|
| speculative checkpoint save (785) | 17.34 | 24.3 % |
| speculative checkpoint restore (728) | 21.74 | 30.4 % |
| drafter `generate()` | 17.27 | 24.2 % |
| unattributed | 15.05 | 21.1 % |

**More than half of the excess is spent inside the checkpoint calls** —
39.07 s, reproducible to two hundredths of a second across four arm-runs, at a
median of 21.9 ms per save and 22.4 ms per restore. Inside, not on: the state
APIs synchronise first, so this is the API boundary. Run T4 times that wait
separately and it is **0.002 s of 39.09 s**, so what the boundary measures is
post-drain state-save/restore API work rather than waiting
([A12](../ERRATA.md#a12-what-the-checkpoint-path-costs-measured-with-timers-in-the-source)). `spec-dflash-n2` on the same prompts performs **zero** of
these operations, spends 3.41 s drafting, and finishes **5.3 s faster than not
speculating at all**.

Both sides of that comparison are measured at the same depth: all twelve logs
of the run are extracted, four repeats per arm. `baseline` and `spec-dflash-n2`
emit zero checkpoint records in every one of their four arm-runs, and
`spec-draft-n8` emits 785 saves and 728 restores in every one of its four.

The measurement required rebuilding llama.cpp with timers around the four
checkpoint calls, so that run alone is not on a stock binary. It was used as its
own control first: its throughput reproduces the stock build to within 0.54 %,
and to −0.00 % on the arm being attributed. Patch, reasoning and scope in
[ERRATA A12](../ERRATA.md#a12-what-the-checkpoint-path-costs-measured-with-timers-in-the-source).

The second term is arithmetic and needs no instrumentation: this is a 35 B model
with roughly 3 B active per token, so a 0.8 B **dense** drafter is not the 1–2 %
of target cost that speculative decoding usually assumes. It is nearer a quarter.

### What the audit measured on 2026-08-25 and 2026-08-26

Four things were measured on the original v2/v3 bench host and are recorded
here: what acceptance does once it is counted correctly, whether the vocabulary
defect explains the slowdown, whether `llama-server` survives a draft model on
the archived binary, and what the workload shape was doing. The middle two bear
directly on how the archived data should be read.

Full data and method: [`v4_audit_2026_08_25/`](../v4_audit_2026_08_25/).

**With acceptance measured properly, the anomaly disappears, and the contrast
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
economics. There is no anomaly left, and **no MoE-specific pathology is
needed**. This repository never had evidence for one, and the sweep was extended
past the threshold to check rather than assert. Across `n_max` 1 → 128, spanning
**3.1 % to 98.3 % expected routed-expert coverage**, a single regressor accounts
for the cost:

```
ms per generated token = 27.00 + 4.040 × (draft tokens per generated token)
R² = 0.99303
```

**The step in the residuals at the 95.3 % coverage point is −0.39 percentage
points**, against a residual arc of about ±11 %. No knee, no break, nothing for
a coverage threshold to explain.
Throughput peaks at `n_max` 4 and declines monotonically straight through.
Full detail:
[A7](../ERRATA.md#a7-with-acceptance-measured-properly-there-is-no-anomaly-left-to-explain).

**The vocabulary defect is real but is not the cause.** Same binary, same draft
file, same flags, only the BOS override differing:

> **`request-mean` is llama.cpp's own `predicted_per_second`, averaged.** That field divides `n − 1` generated tokens by the time for `n`, in 30 300 of 30 344 committed request rows, so every request-mean here is low by `(n − 1) / n` — 0.33 % at 300 tokens and more at shorter lengths. It is uniform across arms on a run where every request hits the same cap, and it is NOT uniform where the arms stop at different lengths, so it must not carry a cross-arm comparison in the thinking-off runs. Every headline figure and every published delta is a **pooled** rate computed from `predicted_n` and `predicted_ms` directly and contains none of this. See [B8](../ERRATA.md#b8-every-request-mean-here-counts-one-token-fewer-than-it-timed).

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
cross-check only: the `llama-server` path that produced every v1 number does
not survive on that commit with a draft attached. It is **fixed on post-merge
master**: all thirty requests complete on `3737e4137`. See
[A6](../ERRATA.md#a6-llama-server-plus-a-draft-model-aborts-on-this-model-at-bcb5eeb64).

**Workload shape matters, and it was never controlled.** The audit ran the
comparison Exp 2 was trying to run, the same arms with thinking verifiably on
and verifiably off, 5 repeats each, `thinking_suppressed` recorded per request:

| method | thinking on | thinking off | draft tokens per generated token |
|---|---:|---:|---|
| `ngram-mod` n=24 | −6.8 % | **−0.7 %** | 0.21 → **0.00** |
| `ngram-cache` | −40.0 % | −32.6 % | 0.42 → 0.36 |
| draft model, n_max 8 | −74.0 % | −76.4 % | 1.85 → **2.14** |

With thinking off `ngram-mod` stops drafting entirely and its cost nearly
vanishes; a chain-of-thought trace is the repetitive text an n-gram lookup
feeds on, a direct answer is not. That one is length-independent (zero draft
tokens is zero however long the output) and it stands.

> [!WARNING]
> **The rest of this table is confounded by output length, and one reading of
> it does not survive.** With thinking off the arms stop in different places
> (ERRATA A11), so this compares arms that generated different numbers of
> tokens; with thinking on every request here ran to exactly 300. On the five
> prompts where every arm generated the same 300 tokens:
>
> | | thinking on | thinking off, as above | thinking off, length-matched |
> |---|---:|---:|---:|
> | draft model `n_max 8` | −74.0 % | −76.4 % | **−72.7 %** |
> | its acceptance | 29.7 % | 23.1 % | **30.3 %** |
> | `ngram-cache` | −40.0 % | −32.6 % | **−39.0 %** |
> | its acceptance | 1.8 % | 1.4 % | **1.8 %** |
>
> This paragraph used to read "for the draft model the effect reverses —
> acceptance falls from 29.7 % to 23.1 %, so reasoning traces are *easier* for a
> 0.8 B drafter than real answers". On the length-matched half, acceptance is
> **30.3 %** against 29.7 % with thinking on, and the throughput cost is
> *smaller* with thinking off, not larger. The fall was the short outputs, not
> the workload: acceptance varies along the sequence and a short generation is
> all early tokens.
>
> The matched half is five prompts of ten and not a random five: they are the
> ones long enough that every arm hit the cap, so this is not a corrected value
> either. What it establishes is that the original reading was not supported.
> [ERRATA A17](../ERRATA.md#a17-the-thinking-off-comparisons-are-not-comparisons-of-the-same-amount-of-work),
> `analysis/length_matching.py`, and `BENCH_IGNORE_EOS` in the harness for
> measuring it properly next time.

What that implies splits by family, so it cannot be said in one sentence.
Draft-model speculation was measured on its *favourable* workload, since
thinking traces are easier to predict and give a better net result, and still
lost, so
that finding is more robust than when it was published. ngram methods were
measured on their *unfavourable* one: thinking gives an n-gram lookup much more
to fire on, and firing costs more than it returns, so the historical ngram
figures **overstate** the cost a real-answer workload would produce. Neither
becomes a net win. See
[D3b](../ERRATA.md#d3b-workload-shape-does-matter--and-exp-2-pointed-the-wrong-way).

**Three quarters of v1's requests returned no answer.** `message.content` is
empty for 144 of 190 v1 requests, and 19/19 for both `reasoning` and
`code_small`: the 300-token cap was reached inside the thinking block, and
`reasoning_content` was never captured. v1 measured decode throughput on
truncated chain-of-thought, not on answers. See
[A5](../ERRATA.md#a5-three-quarters-of-v1s-requests-returned-no-answer-at-all--only-truncated-thinking).

---

### v1 representative results

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

Full per-request values: [`analysis/summary.csv`](../analysis/summary.csv).
Per-config aggregate with all four summaries and both references:
[`analysis/summary_by_config.csv`](../analysis/summary_by_config.csv).

#### Per-prompt structure

![per-prompt decode rate, normalised to the matched baseline](../analysis/plot_per_prompt.png)

Black outlines mark requests that recorded at least one fully accepted draft
round. The picture is not the "chat prompts never trigger, structured prompts
collapse" taxonomy the root README used to assert:

- **ngram-mod** records draft rounds on eight of the ten prompts at n = 24 —
  `short_q`, `medium_chat`, `medium_rec`, `reasoning`, `long_explain`,
  `multi_turn_1`, `multi_turn_2`, `zh_cn` — and **not** on `short_greet` or
  `code_small`. An earlier version of this bullet listed six of the eight and
  called them the chat prompts, which made the three arms below look like a
  partition of the prompt set when they overlap; it also wrote `zh_hant`, a tag
  from the v4 prompt set, for the v1 tag `zh_cn` this figure actually plots.
  Its cost is a flat ~4 % on the eight it fires on.
- **classic draft** records rounds only on `long_explain` and `code_small`, and
  leaves `reasoning` at full baseline speed.
- **ngram-cache** records rounds on `reasoning`, `long_explain`, `code_small`.

Three configurations, three different prompt partitions. Ten hand-written
prompts cannot establish a prompt taxonomy, and the earlier "entirely bimodal
by prompt class" sentence was wrong ([B5](../ERRATA.md#b5-the-regression-is-entirely-bimodal-by-prompt-class-is-false-for-the-ngram-mod-family)).

The heatmap also exposes why `ngcache-kv-fp16` is a **one-sided control**: on
the seven prompts with no draft round it runs at 101–102 % of the q8_0
baseline, because fp16 KV is simply faster when speculation is idle. There is
no no-speculation fp16-KV row in the matrix, so that condition cannot separate
a speculation effect from a KV-precision effect, and the old reading, "fp16 KV
does not rescue, so KV quant is not the cause", does not follow.

---

### Experiment registry

These experiments differ in host, tool, commit, sampling, and prompt set. They
are not a cumulative body of evidence for one hypothesis and must not be pooled.

| ID | Date | Host / runner | Design | Evidence level |
|---|---|---|---|---|
| **v1 primary matrix** | 2026-04-21 | one GPU of the dual-3090 `s1` host; `llama-server`; commit `9789512` | 19 run labels (14 recorded draft rounds, 5 did not); 10 prompts; one measured request per prompt/config; `temperature=0`; 300-token cap unless noted | Primary descriptive result. No repeated trials per cell. |
| **v2 follow-up** | 2026-04-22 | different single-3090 host; `llama-cli`; commits `9789512` and `bcb5eeb64` | 5 prompts; `temperature=0.5`; 200-token cap; different runner and host | Directional check, not a controlled replication of v1 absolute rates. Thinking control did not work ([D1/D2](../ERRATA.md#d1--d2--no-cnv-was-rejected-and-no_think-did-not-disable-thinking)). |
| **Exp 2 code/JSON** | 2026-04-25/26 | v2 host; `llama-cli` at `bcb5eeb64` | 5 prompts × 3 trials × 3 configs | Exploratory only. Intended workload unverified and per-request outputs not committed ([D3](../ERRATA.md#d3-exp-2-cannot-be-audited-so-it-cannot-refute-anything)). |
| **v3 DFlash** | 2026-05-07 | v2 host; `llama-cli` | 5 prompts × 1 run × 3 draft-max settings | Exploratory only. Baseline and treatment used **different binaries** ([D4](../ERRATA.md#d4-v3-dflash-compares-two-different-binaries)). |
| **v4 audit** | 2026-08-25 to 2026-08-31 | one RTX 3090 (`3090` host); `llama-server` at `bcb5eeb64` and `3737e4137` | runs A to W2, 3002 arm-runs in 74 committed directories, beside three one-request start-up checks; 2 to 10 repeats per arm; arm order ABBA, then a Latin square from run O2, then a Williams square in run W; per-request JSON with full text and token ids, continuous GPU telemetry, pre-registered predictions | The controlled tier. Each run carries its own matched no-speculation baseline. |

The v4 runs, and what each one is for:

| run | question | design |
|---|---|---|
| A / B | does the archive reproduce, and does the abort persist? | 3 arms × 10 prompts; A at `bcb5eeb64`, 2 repeats, where both speculative arms abort part way; B at post-merge `3737e4137`, 3 repeats, 30 requests an arm |
| C / D | thirteen arms, thinking on and verifiably off | C: 13 arms × 10 prompts × 5 repeats; D: 5 of those arms again with thinking off, 5 repeats |
| E | is there anything past MoESD's 95-token coverage threshold? | `n_max` 32 / 64 / 96 / 128, spanning the threshold, 3 repeats |
| H | is `p_min` the lever, not draft length? | `p_min` 0 / 0.50 / 0.75 / 0.90 at `n_max` 8, plus `n_max` 32 and 128 at 0.75, 3 repeats |
| I | does concurrency rescue speculation, as upstream says it should? | 1 / 4 / 8 concurrent client requests, verified from timestamps; server-side batch width not instrumented |
| J | DFlash off vs on, one binary — the A/B v3 never had | 5 arms × 3 repeats, `-fit on` on every arm |
| K | where is the draft-length optimum, and does it survive batching? | `n_max` 1, 2, 3, 4, 6, 8 at 3 repeats, then the winner at concurrency 4 / 8 |
| L | does the win survive the workload changing? | same 5 arms twice, thinking on and off, 5 repeats |
| M | the MTP head the vLLM sibling uses, measured here | 7 arms at 3 repeats, then batching, thinking off, and a Q4_K_M head |
| N | do the two ngram-map methods nobody had run do anything? | 7 arms × 3 repeats |
| O | every method against one baseline under one policy | 9 arms × 3 repeats, ABBA order |
| O2 / O3 | the same nine arms as a Latin square, and again five hours later | 9 arms × 9 blocks, twice; 810 of 810 request-pairs byte-identical |
| P / R | is the win a property of those ten prompts? | a second set of twenty sharing none of them, thinking on and off |
| Q | does run M1's outlier reproduce at five repeats? | 3 arms × 5 repeats, two drafter quantisations |
| T / T3 / T4 | where does the extra time actually go? | llama.cpp rebuilt with timers around the four checkpoint calls; T4 splits the wait from the state work |
| U | how far does one configuration move between invocations? | the same two arms, six invocations back to back |
| V / V2 / V3 | what does forcing every request to the same length change? | free-running against a hard cap; V2 is eight sessions, V3 two within-invocation squares |
| W | is the mode contrast an artefact of what ran before it? | five sessions of a 10 × 10 Williams square, 500 arm-runs, balanced for position **and** for first-order carryover |
| W2 | the same question at the power to answer it | twelve sessions of the same 10 × 10 square, 1200 arm-runs, analysis plan committed before the driver was invoked |

The `smoke` directories are start-up checks and carry one arm-run each.

The v2 / Exp 2 / v3 files remain valuable archival evidence. Their absolute
rates and their causal interpretations must be read inside those limits.

---

### v1 hardware, software, and artefacts

One RTX 3090 24 GiB (`CUDA_VISIBLE_DEVICES=1`, SM 8.6) in a two-card host whose
other card was deliberately left to an Ollama instance, so the benchmark process
had a card to itself and the host did not
([C4](../ERRATA.md#c4-gpu-0-was-running-another-workload)). Driver 580.126.09,
llama.cpp `97895129e5f2bde94d13dc01ca41ee79e9b629f2` built with the **CUDA 12.6**
toolkit; `nvidia-smi` reports driver support for CUDA 13.0, which is a different
thing. Target `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`, classic drafter
`Qwen3.5-0.8B-Q4_K_M.gguf`, both from unsloth. Server flags
`-ngl 999 -c 16384 --jinja -fa on -ctk q8_0 -ctv q8_0 --no-webui`, greedy
decoding, the server restarted between configurations so no cache state crosses
them, and one 8-token warm-up per configuration which is not a full-shape
warm-up.

> [!NOTE]
> **Equal vocabulary size is not vocabulary compatibility.** Both
> Qwen3.6-35B-A3B and Qwen3.5-0.8B declare `vocab_size = 248320`, and this
> repository previously called the pair "vocab-matched". llama.cpp disagreed:
> [A2](../ERRATA.md#a2-the-draft-model-was-not-vocabulary-compatible-the-run-used-the-token-translation-fallback).

Every one of those, with the three model digests, the RAM, the kernel and the
build flags, is in [`BENCHMARK_ENV.md`](../BENCHMARK_ENV.md). This section used to repeat that file in
full while linking to it in its own last line; what stays is what something
here reads.

### Follow-up experiment caveats

The v2, Exp 2 and v3 runs are archival and each has a caveat that changes how it
may be read: v2 and Exp 2 used a different host, `llama-cli` rather than
`llama-server`, `temperature=0.5`, a 200-token cap and a different prompt set,
and their scripts pass two flags the committed logs show to be inert; v3 measured
DFlash on a PR branch before the merge. The vLLM sibling repository measures a
different engine on two cards and is not comparable to anything here.

One thing Exp 2 does establish cleanly, and this is the only place it is
recorded: the command is highly repeatable. The three trial means for the Oleg
configuration are 66.54 / 66.54 / 66.64 tok/s, SD 0.06; the published
`± 7.57` is spread *between prompts*, not run-to-run noise.

The full statement of each caveat, with the log lines that prove the inert
flags, is in [`BENCHMARK_ENV.md`](../BENCHMARK_ENV.md) and in the dated entries of
[`CHANGELOG.md`](../CHANGELOG.md); this section used to restate both. What is kept
here is what exists nowhere else, which was checked by diffing the numbers out
of the old section against the whole repository rather than by assuming.
