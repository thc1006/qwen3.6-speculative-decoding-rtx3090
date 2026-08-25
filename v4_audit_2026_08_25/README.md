# v4 — audit measurements, 2026-08-25

New measurements taken during the audit, on the same physical host that
produced v2 and v3. They exist to settle three questions the archived data
could not:

1. Was the draft model's vocabulary incompatibility responsible for the
   slowdown? (**No.**)
2. Is the published "100 % draft acceptance" a real measurement? (**No** — and
   upstream has since fixed the counter, which proves it.)
3. Does the `llama-server` speculative path still work on this model?
   (**Not at `bcb5eeb64`. Yes on post-merge master.**)

Harness: [`../bench/retest_runner.py`](../bench/retest_runner.py) — one pinned
binary per run, ABBA arm ordering, N repeats, and a manifest recording the
binary's sha256, both model sha256s, the complete argv, and GPU telemetry
before and after every arm. Per request it keeps the generated text, the
reasoning channel, the stop reason, the whole `timings` block, and token IDs via `logprobs` (near-complete: `probs_output` drops trailing stop-word tokens, `server-context.cpp:2036-2039`, so the list can run a few short of `predicted_n`, which stays the authority for token counts).

> The runs on this page were taken **before** two harness defects were fixed,
> and their JSON shows it: `tokens` is empty in every row, because the first
> version asked for `return_tokens`, which the OAI chat serialiser silently
> drops. Their `content` is empty and `reasoning_content` is full in every
> baseline row — first-hand confirmation of [ERRATA A5](../ERRATA.md), and a
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

## Run A — `bcb5eeb64`, the binary v2 used

`data/A_bcb5eeb64_legacy/`, 2 repeats, binary sha256 `32c16754e053da2f…`

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
restore — see `data/abort_evidence_bcb5eeb64.txt`:

```
slot update_slots: n_draft=6, accepted=6           <- 6 < 6+1, partial
slot update_slots: restoring speculative checkpoint (size = 65864420)
srv  update_slots: decoding batch, n_tokens = 7
ggml-cuda.cu:97: CUDA error: an unsupported value or parameter was passed
  cublasSgemm_v2(..., CUBLAS_OP_T, CUBLAS_OP_N, row_diff, src1_ncols, ne10, ...)
  #14 server_context_impl::update_slots()
```

The baseline arm completes every time. This means v2's "cross-checked on master
`bcb5eeb64`, identical results" is a `llama-cli` cross-check only — the
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
`211/404`, …). The throughput difference is +0.3 % overall, and per prompt it
ranges from −1.2 % to +3.7 % — noise.

So: llama.cpp genuinely was running the token-translation fallback, this
repository's "vocab-matched" claim was genuinely false, and fixing it changes
nothing material. The negative finding survives, now measured on a matched
path. See [`../ERRATA.md`](../ERRATA.md) A2 for the root cause — the draft
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

`analysis/plot_accept_vs_speed.png` — every one of whose 140 points sat at
exactly 100 %, making this relationship invisible — has been deleted.

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

Two entries in that column need reading carefully. `baseline`'s 2.08 is almost
entirely its cold-start first repeat — excluding it the SD is 0.47. But
`ngram-cache`'s is not: 77.9, 74.7, 75.8, 72.3, 72.0 tok/s, still 1.86 after
dropping rep 0, and drifting downward. It is the least reproducible arm in the
matrix by a wide margin, and this audit does not have an explanation for it.
Every other arm sits between 0.03 and 0.48 once the cold start is removed.

Three things fall out.

**The best draft length is 4, and it is still 71 % below baseline.** The sweep
is not monotone — cost rises and acceptance falls as the window widens, and
n_max 1 is worse than n_max 2 because a full drafter forward pass buys at most
one token. The peak is real but shallow: per-repeat SD is 0.055 and 0.076 for
n_max 2 and 4, against a 1.4 tok/s gap.

**Drafting volume is not the whole cost.** An external drafter proposing 0.50
tokens per generated token runs at 31.1 tok/s; `ngram-cache` proposing 0.42
runs at 74.0. The per-round cost of the *method* is a third, independent term —
a 0.8 B forward pass every round versus a table lookup.

**The fp16-KV control the v1 matrix lacked.** fp16 KV is 1.9 % faster than
q8_0 with no speculation running, and 4.2 % worse with `ngram-cache` on. See
[`../ERRATA.md`](../ERRATA.md) B7.

### D — the same arms with thinking verifiably off

`thinking_suppressed` is recorded per request: 50/50 in D, 0/50 in C. Output
lengths in D run 22–300 tokens because completions now finish naturally.

| method | thinking on (C) | thinking off (D) | draft tokens per generated token |
|---|---:|---:|---|
| `ngram-mod` n=24 | −6.8 % | **−0.7 %** | 0.21 → **0.00** |
| `ngram-cache` | −40.0 % | −32.6 % | 0.42 → 0.36 |
| draft model n_max 8 | −74.0 % | −76.4 % | 1.85 → **2.14** |

With thinking off `ngram-mod` never drafts at all — zero draft tokens across
fifty requests — and its cost nearly disappears. A chain-of-thought trace is
long and formulaic, which is what an n-gram lookup feeds on; a direct answer is
not. For the draft model the effect runs the other way: acceptance falls from
29.7 % to 23.0 %, so reasoning traces are *easier* for a 0.8 B drafter than
real answers.

This is the comparison Exp 2 believed it had made. Its conclusion was not
merely unverifiable, it was backwards for the methods where shape matters most.
And it means **every historical number in this repository was measured on the
workload that favours speculation, and speculation still lost.**

### Thermals and drift

1317 telemetry samples over 110 minutes, 1272 under load:

- `power.limit` = `power.default_limit` = `power.max_limit` = 350 W — **not overclocked**
- 58–75 °C, mean 64.7, against a ~83 °C throttle point
- 1800–1965 MHz of a 2100 MHz maximum, mean 1937
- `hw_thermal_slowdown` and `hw_power_brake` essentially never active; the one
  `sw_thermal` sample fires at 64 °C with the clock at its run maximum

The baseline is repeated five times across the run, so drift is testable from
the measurement itself: 126.6, 122.2, 122.6, 121.8, 121.6 tok/s. That is not
progressive decline — it is a cold-start first repeat, +3.7 % against a tail
that then varies by 0.89 %. Three other arms measured across the same repeats
show no such step (+0.35 % to +0.47 %), because only the first arm of the first
repeat starts on an idle, cool card.

---

## Run I — batching, the lever upstream names

Upstream's standing answer to "speculative decoding loses on a MoE target" is
that the regime is wrong: the win is supposed to appear when the GPU is not
already saturated by a single stream, i.e. under batching (ERRATA A9). Run I
tests that directly on this host, with the matched-vocabulary drafter at
`--spec-draft-n-max 8` — the arm closest to what v1 ran.

**First, the harness had to be fixed.** The first attempt at this run measured
nothing. `BENCH_CONCURRENCY` passed `--parallel N -cb` to the server, which
allocated N slots, and then the client issued the ten prompts one at a time, so
N−1 slots sat idle. The tell was in the data before any code was read: the c=4
arm-runs took 44 s and 118 s against c=1's 44 s and 116 s. Four times the
nominal batch width at identical wall-clock is a batch size of one. That
attempt is discarded, not reported.

The runner now dispatches through a thread pool **and reads the achieved width
back out of the request timestamps**, because `--parallel` on the server and N
worker threads in the client are each necessary and neither is sufficient. Every
arm-run below records `max_in_flight`, and it equals the configured level in all
eighteen:

| level | requested | observed `max_in_flight` |
|---|---|---|
| c=1 | 1 | 1, 1, 1, 1, 1, 1 |
| c=4 | 4 | 4, 4, 4, 4, 4, 4 |
| c=8 | 8 | 8, 8, 8, 8, 8, 8 |

Aggregate throughput — 3000 generated tokens divided by wall-clock over the
ten-prompt set, mean of three repeats:

| concurrency | no speculation | `spec-draft-n8` | speculation ÷ baseline |
|---|---|---|---|
| 1 | 109.7 ± 0.57 | 30.6 ± 0.14 | 0.28× |
| 4 | 154.3 ± 0.27 | 27.0 ± 0.73 | 0.18× |
| 8 | 180.0 ± 15.21 | 28.1 ± 0.66 | 0.16× |

![Run I: aggregate throughput against concurrency](../analysis/plot_batching.png)

**Batching helps the target and does nothing for the drafter.** No speculation
gains +40.6 % at c=4 and +64.0 % at c=8. Speculation moves −11.7 % and −8.4 %
over the same range. The gap therefore widens with batching rather than
closing: 0.28× → 0.18× → 0.16×.

On this host, at this model and draft window, batching is not the missing
regime. That is a negative result about one arm on one card, not a refutation
of the upstream argument in general — but it is the specific configuration this
repository has been reporting, measured in the regime it was told to measure it
in.

Two caveats, both against the strength of the result:

- ± is the run-to-run SD of three repeats. At c=8 it is 15.21 because one
  baseline repeat came in at 197.5 against 171.2 twice. Ten prompts over eight
  slots is one full wave plus a wave of two, so wall-clock there is sensitive to
  which prompts land in the short tail; c=4, at 2.5 waves, is the cleaner
  measurement. The conclusion survives either way — the *slowest* c=8 baseline
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
with `-ngl` unset, because pinning it makes `common_fit_params` abort instead of
adjusting the parameters the caller left unset — so `BENCH_FIT=on` applies to
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

![DFlash draft-length sweep, runs J and K](../analysis/plot_dflash_sweep.png)

**DFlash at `n_max 4` is +18.7 % on aggregate throughput and +24.0 % pooled.**
It is the first configuration in this repository that beats not speculating,
and it is not an average that hides losers — it wins on all ten prompts
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

The sign flips with draft length, and it flips fast: +18.7 % at 4, −14.8 % at 8,
−47.4 % at 16, with acceptance falling 55.8 % → 36.8 % → 21.4 % as the window
grows. The archived v3 result — DFlash slower — is what this looks like at
`n_max 8` and 16. v3 measured n_max 4 as well, but across a binary change, and
read the difference as a DFlash effect.

### The configuration that produced it does not start reliably

Run J's fifteen arm-runs all completed — no crashes, no retries. But the same
`-c 16384` DFlash configuration, on the same binary and models forty minutes
later, aborted at its first decode:

```
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 120.28 MiB on device 0:
    cudaMalloc failed: out of memory
srv  update_slots: decode() failed: failed to allocate compute pp buffers
```

Run J's telemetry peaks at 23946 MiB of 24576 with the drafter resident —
630 MiB of headroom, 2.6 % of the card — and a 120 MiB allocation still failed,
so the true transient peak is above what five-second sampling can see. With
`-fit on` the fitter sizes the target against whatever it reads as free at
startup, and whether the drafter then fits is decided in that margin.

Run J's numbers are what they are: fifteen clean arm-runs with a matched
control. But **the configuration is marginal on a 24 GiB card**, and anyone
reproducing it should expect to lower the context. Run K does exactly that —
`-c 8192` for every arm including its own baseline — which is why K's absolute
rates are not comparable with J's and K carries its own control.

The harness now fails fast on this: `wait_health` watches the process instead
of polling for its full 300-second timeout, so a dead arm costs ~20 seconds to
discover rather than five minutes per repeat.

**What this does and does not establish.** It establishes that on this host,
this target, this drafter and this prompt set, a self-speculative method at a
short draft window beats no speculation by roughly a fifth, with a matched
control and three repeats. It does not establish that the number transfers: the
window is one draft length on one card, thinking is on throughout, and ten
prompts is ten prompts. The draft-length optimum is bracketed from below in run
K, because three points that straddle a peak cannot say where the peak is.

---

## Answer 3 — what still needs doing

These runs settle the three questions above. They do **not** make this a
properly powered benchmark of current llama.cpp: three repeats, ten prompts,
one draft window, one host, and thinking is still on. The remaining queue is in
[`../RETEST_TODO.md`](../RETEST_TODO.md) — in particular a working thinking
control (P0-2), the draft-window sweep with cost instrumentation (P3-1), and
DFlash on one post-merge binary (P2), which is now `--spec-type draft-dflash`.
Post-merge master also exposes `draft-eagle3` and `draft-mtp`, which earlier
versions of this repository listed as "not evaluated here".

Absolute rates must not be compared across the two runs: no-speculation
baseline is ~8 % faster on master than on `bcb5eeb64` on the same host.

## Files

| Path | Contents |
|---|---|
| `data/A_bcb5eeb64_legacy/` | manifests and per-request JSON, run A |
| `data/B_master_3737e4137/` | manifests and per-request JSON, run B |
| `data/abort_evidence_bcb5eeb64.txt` | the CUDA abort in context, both arms |
| `../bench/retest_runner.py` | the harness |
