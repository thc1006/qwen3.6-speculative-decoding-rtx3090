# Changelog

All notable changes to this bench are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning is not strictly semver — each numbered release is a public
publication point with its own data set.

## [Unreleased] — the review pass

Two external reviews of the pull request. Every specific accusation in the
second was verified against the code before anything was changed, and all four
were correct.

### Added

- **Run T3** — run T repeated at three **balanced** blocks on the same
  instrumented build, with the library hash asserted per arm-run rather than
  once per run. The checkpoint attribution replicates: 785 creates and 728
  restores in every arm-run of both runs, 39.159 s against 39.075 s, 54.6 % of
  the excess against 54.7 %.
- **ERRATA A16** — and run T3 also produced **byte-identical output** to run T
  on all three arms and all ten prompts, with identical acceptance, identical
  fit and identical GPU thermal and clock state, while running the DFlash arm
  **3.4 % slower on every prompt**. The cause is not isolated. The consequence
  is that the 1.6 pp paired-block interval on the headline arm describes the
  run, not the configuration: five runs of it span +20.7 % to +26.7 %, and the
  README now says so beside the table.
- `tests/fake_llama_server.py` — enough of the llama.cpp HTTP surface to drive
  the runner end to end in a second without a GPU. The guards that fire after
  the arm loop had no test because reaching them needed a 20 GiB model, and a
  mutation that deleted the completeness check survived the whole suite.
- `bench/collect_evidence.sh`, `requirements-plot.lock`, `analysis/plot_data.json`,
  and the SHA-256 manifest of the twelve run T logs.

### Fixed

- `BENCH_ORDER=latin` generated a cyclic rotation for any (arms, repeats) pair
  and labelled the result balanced. Three arms over four repeats rotates
  0, 1, 2, 0 — which is what run T did, while its manifest recorded `latin`. The
  schedule is now built and validated before the first server starts, the
  unbalanced rotation is named `cyclic`, and the manifest carries the schedule
  and the verified balance. `analysis/paired_blocks.py` re-derives the schedule
  from each arm-run's monotonic timestamps and warns when the design does not
  hold, rather than trusting the label.
- `RUN_COMPLETE.json` was written unconditionally when the arm loop returned, so
  a run with a crashed arm still carried the marker every consumer reads as
  "this is a whole run". Gated on validation; failures write `RUN_FAILED.json`.
- `analysis/extract_checkpoint_timers.py` stripped only the literal
  `__rep0.log`, so repeats 1–3 were filed under separate arm names and the
  controls rested on one log each. Fixing it exposed the same class of defect in
  `analysis/verify_claims.py`, where the timer records were keyed by arm and
  four arm-runs collapsed into one.
- `analysis/check_data_integrity.py` counted arm-runs instead of checking the
  exact (arm, repeat) product, so rep0 twice and no rep3 passed. It now also
  checks each filename against the `arm`/`repeat` inside it, and validates
  `RUN_COMPLETE.json` against the directory it attests to.
- The headline footnote quoted run O's **+24.6 %** as though it were run O2's
  own figure. It was written when run O *was* the headline table and was not
  updated when O2 replaced it. The footnote is now derived from the data and
  asserted against the README text.
- The headline table's "95 % CI" column is the Student-t interval while the
  prose above it described a block bootstrap. Both are computed and both are in
  `paired_blocks.json`; the column is now labelled, and t is the wider of the
  two on every row.
- `stop_server` printed a warning when the driver had not handed the memory back
  and carried on, so the failure landed on the *next* arm. It is a run-level
  failure now.
- `bench/convert_dflash.sh` promoted the converted drafter to its final path
  before the load check, leaving a file the loader refuses where the next run
  would pick it up.
- `DATA_LICENSE` did not name `v4_audit_2026_08_25/data/**`, leaving 601
  committed measurement files without a stated licence, and `CITATION.cff`
  listed MIT alone for a dataset whose measurements are CC0.
- Provenance: `runner_sha256`, `harness_tree_sha`, `prompt_set_sha256` (over the
  prompts, not their label), and per-arm-run `server_lib_sha256`,
  `server_loaded_commit` and `server_log_sha256`. `BENCH_EXPECT_COMMIT` and
  `BENCH_EXPECT_LIB_SHA256` fail the run on the first arm that disagrees.
- CI pins every action to a commit SHA, installs the chart dependencies from a
  hash-pinned lockfile, checks the committed charts against the committed data,
  and aggregates every attested run under `--strict`.

---

## [v4.1] — 2026-08-26 · the controlled tier, and a reversal

The title loses the word "Historical": it was added by the v4.0 audit to stop
readers taking v1 as current, and it stopped being true the moment this
repository grew a controlled tier on post-merge master.

v4.0 audited what this repository had published. v4.1 measures what it had never
run. Nine of master's eleven `--spec-type` methods, on one card, against a
matched no-speculation baseline inside every run. The headline reverses.

### Added

- **Runs I–O**, 110 arm-runs on post-merge master `3737e4137`, each with its own
  baseline, ABBA ordering, 3–5 repeats, per-request text and token ids,
  recorded batch width, and a continuous GPU trace:

  | run | question | answer |
  |---|---|---|
  | I | does batching rescue speculation, as upstream says? | no — no-speculation gains +64 % at eight in flight, the drafter −8 % |
  | J | DFlash off vs on, one binary — the A/B v3 never had | **+18.7 %**, and it reverses v3's sign |
  | K | where is the optimum, and does it survive batching? | a plateau at `n_max` 2–4, a cliff after; batching erases it |
  | L | does the win survive the workload changing? | it halves with thinking off, and `n_max 4` goes negative |
  | M | `draft-mtp` — the method the vLLM sibling uses | **+17.5 to +21.8 %**; nothing was blocking it, it had never been run |
  | N | `ngram-map-k` / `k4v`, never run here | they never engage: 0.0 % acceptance, three lookup hits in thirty requests |
  | O | every method, one baseline, one matrix, one policy | a factor of five, split by drafter architecture |

- **ERRATA A11** — speculation is not output-preserving on this build. Against a
  determinism control that holds in every run (170/170 self-reproducible), token
  streams differ from no-speculation in 27–30 of 30 request-pairs at the
  300-token cap, and the divergence rate tracks output length.
- **ERRATA A12** — the mechanism, measured. An external drafter forces the
  hybrid target to checkpoint and restore 82.079 MiB of target state plus
  19.266 MiB of draft state on every partially accepted round: 772 saves and 709
  restores per arm-run, a nominal ≈118.7 GiB by event count × the size the server
  reports. DFlash logs none at draft lengths 1–16 and MTP none at 1–8. **Both
  figures were corrected after review**: the volume double-counted the draft
  component that `size()` already includes, and a "≈19.5 % of the wall clock"
  claim was withdrawn because the create and restore messages sit on opposite
  sides of the work they name.
- **Runs P, Q and R**, 60 further arm-runs. P and R repeat the key arms on a
  second set of **twenty** prompts sharing none with the v1 ten — long inputs,
  structured output, four languages, arithmetic, and two genuinely multi-turn
  exchanges, gated on the model recalling four-turn-old context. The decode
  speed-up moves by at most 4.3 pp, so it is not a property of the original
  prompt mix. Q settles the one anomaly v4.1 could not explain.
- **ERRATA A14** — within-run repeats are not an error bar. Ten (arm,
  configuration) pairs measured in two or three independent runs have a median
  between-run spread of 0.56 pp, and one pair differs by 8.5 pp with the argv,
  the drafter hash, the fitter's choices, the per-prompt draft counts, the
  acceptance, the temperature and the clocks all identical. A three-repeat delta
  is accurate to about a point, not to its printed SD.
- **ERRATA A13** — llama.cpp keeps *two* acceptance counters and this repository
  had only ever quoted one. Across 73 single-request arm-runs they agree to
  within 0.5 pp on every path that takes no speculative checkpoint (31 runs) and
  disagree by at least 1.0 pp on every path that does (42 runs), with no overlap.
  The worst case is 53.3 pp. This narrows A1's account of the upstream fix — the
  denominator moved, the early return did not — and it discredits the drafter's
  counter too: `spec-draft-n1` reports 1639 of 1639 accepted on an arm running at
  a quarter of baseline.
- `analysis/plot_v4_runs.py`, `analysis/extract_spec_accounting.py`,
  `analysis/compare_acceptance_counters.py`, `bench/stage_mtp_source.py`, and
  four new charts.

### Changed

- **The headline.** "Every tested condition that recorded speculative activity
  was slower" is still true of the archival tier, and the archival tier tested
  the losing third of the methods. Self-speculation wins here by a fifth;
  external speculation loses by three quarters. The README leads with the
  nine-arm table rather than with v1's negative.
- README now describes an archival tier and a controlled tier separately, and
  says which to cite about current llama.cpp. Its IMPORTANT block had claimed
  this repository holds only 2026-04-21…05-07 data and "is not a benchmark of
  current llama.cpp master"; both were false once runs A–O existed.
- `BENCHMARK_ENV.md` gains the v4 artefact hashes and, more usefully, the
  memory-policy table — `-ngl`, `-c` and `--fit-target` are variables across
  these runs, which is exactly why absolute rates do not transfer between them.
- `bench/retest_runner.py`: `BENCH_CONCURRENCY` now actually dispatches
  concurrently and reads the achieved batch width back out of request
  timestamps; `wait_health` notices a dead server instead of burning a
  300-second timeout; `stop_server` waits for the driver to release VRAM;
  `BENCH_CTX` and `BENCH_FIT_TARGET` make the memory policy explicit.

### Retracted

- **"A configuration is worth running when it clears ~48 % draft acceptance."**
  Published in this repository on 2026-08-26 and falsified the same day. It is
  12/12 inside the DFlash family and fails on the external drafter exactly where
  it should matter: `spec-draft-n1` reaches **68.7 % acceptance and is 74.8 %
  slower**. Scored per family it is 21/23, and both failures are at the
  high-acceptance end. The threshold is a property of the drafter, not of
  acceptance — see A12.
- **"A Q4_K_M MTP head moves 6.8 pp against Q8_0 at unchanged acceptance."**
  Reported in this branch as measured-and-unexplained. Five repeats of each
  drafter dissolve it: the Q8_0 arm at `n_max 4` was one measurement that did
  not replicate. With it re-measured the Q4_K_M head is ahead at both draft
  lengths, which is the simpler and correct story.
- **Run N's "0.0 % acceptance".** Written and corrected the same day. The server
  counter reads 0.0 %; the speculator's own counter reads up to 70.0 %. What is
  measurable without either is that `generate()` was called 3271 times and
  returned a draft twice.
- **v3's DFlash direction.** What v3 measured at short draft windows was a
  binary change, not a DFlash penalty.
- **"vLLM MTP remains the only positive-yield speculative decoding path on this
  hardware"** (`pr_comment.md`). MTP now runs under llama.cpp here, and so does
  DFlash, and both are positive.

### Fixed

- The data map described `bench/retest_runner.py` as "never executed"; it
  produced every v4 measurement.
- `RETEST_TODO.md`'s MTP entry was wrong twice — first "no head weights", then
  "a converter gap". The weights are present, and the stock converter and stock
  runtime both support the architecture. A patch attempt refuted the second
  claim in one step with `TypeError: Cannot create a consistent MRO`.
- Two independent waiter loops raced and ran two benchmarks on one card for
  ninety seconds. Everything they touched was discarded and the runs re-driven
  from a single sequential process.

## [v4.0] — 2026-08-25 · audit and retraction

> Never tagged or published on its own. Its changes ship inside **v4.1**, whose
> tag both entries link to. The two entries are kept separate because they
> document different work: v4.0 corrected what had been published, v4.1 measured
> what had not been run.

A full adversarial re-examination of everything this repository has published.
No raw measurement file was edited; `results/`, `v2_3090_followup/v2_*/`,
`exp2_codejson_n3/master.log`, and `v3_dflash_2026_05_07/data/` are
byte-identical to the archived releases, and every historical aggregate was
re-derived from them and reproduced exactly. What changed is the wording, the
statistics, and the causal claims. Full itemised list with evidence:
[`ERRATA.md`](ERRATA.md).

### Retracted

- **"100 % draft acceptance."** The ratio is 1.0 by construction on this model.
  Qwen3.6-35B-A3B is a hybrid Gated-DeltaNet/MoE model, so the context reports
  `COMMON_CONTEXT_SEQ_RM_TYPE_FULL`; a partially accepted round then takes an
  early `continue` in `server-context.cpp` that skips both acceptance counters
  and re-verifies the truncated prefix. The drafter's own counters, printed one
  line below the quoted `1.00000 (115 / 115)` in the same committed
  `verbose.log`, report **115 of 214 generated draft tokens accepted (53.7 %)**
  and **33 of 81 drafts (40.7 %)**. `analysis/plot_accept_vs_speed.png`, whose
  entire x-axis was that artefact, is deleted.
- **"Vocab-matched draft model."** llama.cpp rejected the pair on its
  special-token gate and ran the token-translation fallback for every
  classic-draft measurement ever published here.
- **"The regression is structural / engine-independent / hardware-independent."**
- **"Q4 collapses the technique"**, **"the mechanism generalises to DFlash"**,
  **"co-trained heads are the only positive yield path"** — all removed from v3.
- **"Exp 2 refutes the workload-shape hypothesis."** Downgraded to exploratory:
  `-no-cnv` was rejected, `/no_think` did not disable thinking, and per-request
  outputs were never committed, so the intended treatment is unverifiable.
- **All "first public benchmark / first public datapoint" claims**, from the
  title, the v3 banner, `pr_comment.md`, and this changelog's v3.0 entry. No
  novelty search was ever performed.

### Corrected

- Pooled decode throughput now reported alongside the request-mean. The gap is
  material: classic draft `--draft-max 8` is −10.8 % by request-mean but
  **−19.0 % pooled**; `ngcache-1000tok` is −25.7 % pooled against its matched
  long-output baseline.
- `±` columns relabelled as across-prompt spread. Each v1 cell was measured
  once, so they were never run-to-run uncertainty. Exp 2's `± 7.57` decomposes
  into a trial-mean SD of **0.058** and prompt-to-prompt spread.
- "19-config matrix" → 19 run labels, of which 14 recorded a draft round and 5
  did not.
- "Every configuration hits 59–67 tok/s" → that band belongs to specific
  ngram-cache and classic-draft requests; the ngram-mod family bottoms out at
  119.8–129.6.
- "Entirely bimodal by prompt class" → three configurations produce three
  different prompt partitions; ngram-mod fires on exactly the chat prompts the
  old text said it could not.
- "all completions reach the cap" → true for the 300-token group, false for
  the 1000-token group (`baseline-1000tok` returned 354–1000 tokens).
- Target artefact named consistently as `UD-Q4_K_XL`; `Q4_K_M` is the draft.
- `T_thres ≈ 94` → `ceil(94.36) = 95`, restated as an expected-coverage
  heuristic; `k_e = 8` and `γ` replace the overloaded `K`.
- Qwen3.5-122B-A10B has the **same** 256 experts and top-8 routing, so the
  "larger active footprint gives a lower `T_thres`" sentence is deleted.
- PR #20075 reattributed to **eauchs**; it is an external-draft-model plus SSM
  checkpoint/restore fix, not ngram-mod.
- Upstream statuses re-checked: #20075 **closed unmerged** 2026-04-25 (was
  listed OPEN); #22105 **merged** 2026-06-28 (v3 said open draft).
- Licence conflict resolved: MIT for code and docs, CC0-1.0 for data with
  `DATA_LICENSE` and `LICENSES/CC0-1.0.txt` present. v3's "Apache 2.0" removed.

### Found during the audit

- **v1 measured truncated thinking, not answers.** 144 of 190 v1 requests
  (75.8 %) returned an empty `message.content`; `reasoning` and `code_small`
  are 19/19. The 300-token cap was reached inside the thinking block and
  `reasoning_content` was never captured.
- **Root cause of the vocabulary failure.** `Qwen/Qwen3.5-0.8B` has no
  `generation_config.json` upstream (HTTP 404), so its GGUF carries no
  `tokenizer.ggml.bos_token_id` and llama.cpp substitutes the hard-coded GPT-2
  legacy default `11` against the target's `248044`. Both models declare
  `add_bos_token = false`, so the field that gates speculation is one neither
  model uses. `--override-kv tokenizer.ggml.bos_token_id=int:248044` flips
  `vocab_cmpt` from 0 to 1.
- **The fix does not explain the slowdown.** A same-binary A/B on 2026-08-25
  moved `long_explain` from 48.4 tok/s to 50–51 tok/s, against a baseline of
  ~126. The negative finding survives, now measured on a matched-vocabulary
  path.
- **`llama-server` plus a draft model aborts at `bcb5eeb64`** with
  `CUDA error: an unsupported value or parameter` in
  `ggml_cuda_op_mul_mat_cublas`, reproducibly, immediately after a
  partial-accept checkpoint restore. 3/3 on `code_small`, in both the
  translation and matched arms.
- **The committed v2 script does not match the committed v2 data** — different
  config directory names — and no v2/v3 log records its own argv.
- **`--spec-type` is server-only**, not "missing from master" as v3 claimed.

### Added

- [`ERRATA.md`](ERRATA.md) — every corrected claim with its evidence.
- [`RETEST_TODO.md`](RETEST_TODO.md) — the runnable work queue that closes the
  open items, with the bench host's verified state.
- [`analysis/verbose_accounting.py`](analysis/verbose_accounting.py) —
  reconstructs the acceptance-counter artefact from any `-v` log.
- [`bench/retest_runner.py`](bench/retest_runner.py) — one pinned binary,
  ABBA ordering, N repeats, hashed manifest, and per-request capture of text,
  reasoning channel, stop reason, timings, and token IDs via `logprobs` (near-complete: `probs_output` drops trailing stop-word tokens, `server-context.cpp:2036-2039`, so the list can run a few short of `predicted_n`, which stays the authority for token counts).
  Thinking suppression is verified per request, not assumed.
- `analysis/plot_acceptance_accounting.png`, replacing the retracted scatter.
- `SHA256SUMS`, `CITATION.cff`, `DATA_LICENSE`, `LICENSES/CC0-1.0.txt`.
- Errata headers on every historical script; all host-specific paths are now
  environment variables.

### What survives

Under the exact conditions archived here, no tested condition that recorded
speculative activity beat its matched no-speculation reference in aggregate.
That statement is narrower than what was published, and it is what the data
support.

## Sibling-repo notice — vllm-2x3090 v5.0 (2026-05-17)

A natural follow-up to this repo's MTP findings was published in the sibling [`qwen3.6-vllm-2x3090` v5.0](https://github.com/thc1006/qwen3.6-vllm-2x3090/releases/tag/v5.0): a same-hardware A/B between the production MoE Qwen3.6-35B-A3B-AWQ + MTP k=3 + TP=2 stack and the new dense sibling **Qwen3.6-27B-AWQ** on a voice-agent workload (10 prompts × 3 trials × 2 models = 60 samples).

Result: **MoE+MTP production stack wins decisively** — TTFT 178 ms vs Dense 771 ms (**4.34×**), tok/s 88 vs 16 (**5.42×**), e2e 274 ms vs 1684 ms (**6.13×**). The "Dense 27B fits TP=1, should be cheaper to serve" intuition is falsified on this hardware × workload, and the MTP k=3 production recommendation from this repo's v3.0 is corroborated by the absence of any cheaper Dense-no-spec alternative.

Caveat: see the [vllm-2x3090 v5 README scope section](https://github.com/thc1006/qwen3.6-vllm-2x3090/blob/master/v5_2026_05_17/README.md#scope-and-known-caveats) — N=3, single hardware, plus several vLLM-config and prompt-specification confounds documented in the ERRATA. **Latency findings are compute-bound and robust; tool-accuracy findings need a prompt-matched retest.** This repo (`qwen3.6-speculative-decoding-rtx3090`) does not get a version bump or new release for this event — the work belongs in the sibling repo's lineage.

## [v3.0] — 2026-05-07

### Added

- **DFlash speculative-decoding bench** via llama.cpp PR #22105 on the same
  hardware as v2.x. Full content in [`v3_dflash_2026_05_07/`](v3_dflash_2026_05_07/).
  - Setup: same RTX 3090 (single card) + `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` target.
    Drafter is `z-lab/Qwen3.6-35B-A3B-DFlash` (HF safetensors), converted to
    GGUF via PR #22105's modified `convert_hf_to_gguf.py` with `--target-model-dir`.
  - 5 prompts x 1 trial x 3 draft-max configs (4, 8, 16) = 15 measurements.
  - Result: **NET LOSS -44.6 % at best (DFlash --draft-max=8: 77.0 tok/s vs
    138.9 tok/s no-spec baseline)**. Slightly less bad than v2.x's Oleg
    draft-spec NET LOSS (-52 %), but still net negative.
  - ~~First public RTX 3090 + DFlash + Q4 quantized target datapoint.~~
    **RETRACTED 2026-08-25** — no novelty search was performed (ERRATA F3).
  - Reproduction: [`v3_dflash_2026_05_07/bench/bench_dflash.sh`](v3_dflash_2026_05_07/bench/bench_dflash.sh)
    (note: do NOT use `set -euo pipefail` + `grep | tail` combo; empty grep
    matches kill the whole script via pipefail).

### Cross-method ranking (single 3090, Qwen3.6-35B-A3B Q4_K_XL target)

| method | tok/s (mean) | vs baseline |
|---|---:|---:|
| no spec (baseline) | 138.9 | reference |
| Oleg draft-spec max=32 | 65.5 | -52.8 % |
| Oleg draft-spec max=16 | 66.6 | -52.1 % |
| DFlash --draft-max 16 | 65.8 | -52.6 % |
| **DFlash --draft-max 8** | **77.0** | **-44.6 %** (best DFlash) |
| DFlash --draft-max 4 | 74.9 | -46.1 % |

### Mechanism note

Per llama.cpp PR #22105 author: *"for Qwen3.5/3.6 MoE, performance is
currently not optimal due to MoE + hybrid structure not well supported."*
The wider mechanism is **MoE expert-routing x consumer-Ampere bandwidth**:
Qwen3.6-35B-A3B routes 8-of-256 experts per token, expert-saturation
threshold ~94 tokens. At single-stream batch=1 with `--draft-max ≤ 32`,
drafted tokens stay below saturation, so verification has to load the
union of expert slices, exceeding savings even at 100 % acceptance. This
is **not Q4-specific or consumer-GPU-specific in isolation** — it's the
joint MoE-routing × bandwidth × Q4-coupling effect. Cross-reference: the
[sister repo](https://github.com/thc1006/qwen3.6-vllm-2x3090) v3.0/v4.0
shows vLLM MTP (co-trained head, no separate KV cache) is +27 % NET WIN
on the same hardware.


## [v2.3] — 2026-04-26 (afternoon)

### Added
- [`v2_3090_followup/exp2_codejson_n3/`](v2_3090_followup/exp2_codejson_n3/)
  — workload-shape probe. 5 code/JSON prompts (Python class with RLock,
  REST API spec JSON, Rust merge_sort, Nginx reverse proxy config,
  PostgreSQL top-10 customers query) × 3 trials × 3 configs (baseline,
  Oleg `--draft-min 2 --draft-max 32`, srogmann `--draft-min 48 --draft-max 64`)
  on standalone single 3090. **Result confirms v2 negative direction**:
  baseline 139.22 ± 0.46 tok/s, Oleg 66.57 ± 7.57 tok/s (−52 %), srogmann
  83.84 ± 1.80 tok/s (−40 %). The "structured prompts win" hypothesis is
  refuted on llama.cpp + Q4 + RTX 3090.
- README banner at the top documenting Exp 2 + the cross-engine status
  correction.

### Changed
- **MAJOR scope correction** — earlier wording (v2.2) claimed the spec-decode
  negative finding is "hardware-class-independent" across consumer Ampere +
  datacenter Ampere NVLink + Hopper. A v3 clean A/B retest in the sibling
  repo [`thc1006/qwen3.6-vllm-2x3090`](https://github.com/thc1006/qwen3.6-vllm-2x3090)
  (matched flags + `--no-enable-prefix-caching`) found vLLM MTP k=1 on the
  **same 2× RTX 3090 PCIe hardware** is **+27.5 % faster decode rate**, not
  −12 %. So the negative direction is **engine + spec-method specific**:
  it holds for `llama.cpp` draft-then-verify (this repo's data) but does
  not hold for `vLLM` MTP k=1 with prefix-cache disabled. README "Cross-engine
  confirmation" paragraph and the "Why" paragraph have been corrected.
- README banner updated; `v2_3090_followup` directory expanded.

## [v2.2] — 2026-04-26

### Changed
- README cross-engine paragraph: disclose 3090 vLLM published config
  confound (baseline `--gpu-memory-utilization 0.90 --max-num-seqs 8`
  vs MTP run `0.80 / 2`); cite **2× A100-80GB SXM4 NVLink** Modal clean
  A/B as the canonical cross-hardware MTP datapoint (prompt-4
  decode-only delta **−11.4 %**, TTFT-robust).
- README "Why" paragraph: remove "memory-bandwidth-bound" attribution.
  The mechanism is **bandwidth-independent** — the same negative
  direction is now observed on (a) consumer 3090 GDDR6X 936 GB/s,
  (b) datacenter A100 SXM4 HBM2e 2 TB/s + NVLink, (c) Hopper H20-3e
  per [vllm #38182](https://github.com/vllm-project/vllm/issues/38182).
- Replace "engine-independent on Ampere" with "**hardware-class-
  independent at single-stream batch=1** across consumer Ampere +
  datacenter Ampere (and Hopper per #38182)".

### Added
- README image hot-link to cross-hardware comparison chart hosted in
  the sibling `qwen3.6-vllm-2x3090` repo (`analysis/plot_cross_hardware.png`).
- N=3 trial replication on a fresh standalone 3090 host of the v2 numbers
  (baseline 139.19, Oleg `--draft-min 2 --draft-max 32` 65.24, srogmann
  `--draft-min 48 --draft-max 64` 85.50). Run-to-run stdev <0.11 tok/s,
  matching v2 published 139.9/65.0/85.6 within <0.5 pp. Addresses the
  N=1 caveat in v2 limitations.

## [v2.1] — 2026-04-25

### Added
- README **Validation timeline (post-publication)** section consolidating six
  independent corroborations and one academic theoretical framing that have
  appeared since v1 / v2:
  - [MoE-Spec (arXiv 2602.16052)](https://arxiv.org/abs/2602.16052) names
    the phenomenon ("expert budgeting") and proposes a training-free
    verification-time budget cap.
  - [Alloc-MoE (arXiv 2604.08133)](https://arxiv.org/abs/2604.08133) and
    [XShare (arXiv 2602.07265)](https://arxiv.org/pdf/2602.07265) frame
    the same expert-saturation pressure.
  - [vllm #35387](https://github.com/vllm-project/vllm/issues/35387) —
    H100 + FP8 + Qwen3-Next-80B-A3B with `method=qwen3_next_mtp` reports
    −76.5 % latency regression (different hardware/quant/arch from this
    bench, suspected `mamba_postprocess` CPU sync; same direction).
  - [vllm #38182](https://github.com/vllm-project/vllm/issues/38182) —
    H20-3e + Qwen3.5-35B-A3B + MTP drops prefix-cache hit rate
    ≈92 % → ≈71 %; @Angazenn pinpoints the cause to
    `single_type_kv_cache_manager.py:L457`.
  - [vLLM Qwen3.5/3.6 official Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
    now state up-front that "MTP-1 reduces per-token latency but degrades
    text throughput under high concurrency".
- README cross-engine confirmation note in TL;DR + Related reading pointer
  to sibling repo
  [`thc1006/qwen3.6-vllm-2x3090`](https://github.com/thc1006/qwen3.6-vllm-2x3090).
- README applicability note (iv) — batched multi-user serving caveat —
  and (v) — explicit scope: this bench tests `ngram-cache`, `ngram-mod`,
  classic `--model-draft` in llama.cpp and `mtp k=1` in vLLM. **EAGLE-3**
  with CUDA graphs (vLLM Model Runner V2) is not evaluated here.
- README counter-example block: corrected attribution for the +15–45 %
  Qwen3.5-122B-A10B speedup on PR #20075 — that data is from the PR
  author's M3 Max bench plus @0xSero's AMD Strix Halo follow-up, not
  srogmann's bench. Strix Halo also reports up to **+119 %** with the
  REAP-pruned variant on gfx1151.

### Changed
- README applicability note (ii) — `[PR #20075](https://github.com/ggml-org/llama.cpp/pull/20075)`
  was open at v1 publication; on 2026-04-25 a community comment suggested
  it can be closed because its functionality is superseded elsewhere. The
  note now reflects that the hybrid-SSM/MoE checkpoint situation is fluid.

### Older Unreleased entries (carried over from earlier in 2026-04-22 → 2026-04-25)
- `v2_3090_followup/results_v2.json` — machine-readable summary of all
  v2 runs (per-prompt `llama-cli` stats + per-config mean / min / max /
  std), extracted from the `.log` artefacts.
- `v2_3090_followup/extract_results.py` — the extraction script used to
  produce the above JSON.
- `v2_3090_followup/plot_v2.py` + `plot_v2_configs.png` — horizontal
  bar chart comparing all 9 original-commit configs against baseline,
  with master-commit cross-check annotation.
- `CHANGELOG.md` — this file.

## [v2.0] — 2026-04-22

Follow-up bench addressing [Oleg-dM's comment](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/discussions/14)
on the HF-discussion thread for `unsloth/Qwen3.6-35B-A3B-GGUF`.

### Added
- `v2_3090_followup/` — fresh single-3090 bench on a different
  physical 3090 covering:
  - baseline (no spec-decode)
  - `--draft-min 2 --draft-max 32` (Oleg's suggestion)
  - `--draft-min 2 --draft-max 16` and `--draft-max 64` (bracketing sweep)
  - default `--draft-min=5` variants (`--draft-max 8/16/32` + bare `-md`)
  - srogmann-style `--draft-min 48 --draft-max 64`
  - `--verbose` per-token acceptance dump
- **Cross-check on current master** `bcb5eeb64`
  (post PR #22227 `speculative-simple: add checkpoint support`) — 3 key
  configs re-run on master hardware to rule out stale-commit artefact.
- 45 per-prompt `llama-cli` logs + `verbose.log`.
- `v2_3090_followup/SUMMARY.md` with methodology + full result table.
- `v2_3090_followup/bench_3090_oleg.sh` reproducible script.
- v2 section appended to `BENCHMARK_ENV.md` documenting the v2
  hardware, toolchain, and both commits tested.
- `.gitignore` negation for `v2_3090_followup/**/*.log` so bench
  evidence survives `git clean` and `git add -A` round-trips.

### Changed
- `README.md` — UPDATE banner summarising v2 findings and linking to
  the follow-up artefacts.
- `pr_comment.md` — UPDATE note so the historical llama.cpp PR-comment
  draft stays consistent with the repo state.

### Key findings
- Oleg's `--draft-min 2 --draft-max 32` beats the `--draft-min=5`
  defaults by +18 % (65.0 vs 55.3 tok/s) but is still −54 % vs
  baseline 139.9 tok/s.
- Aggressive `--draft-min 48 --draft-max 64` is the **least bad**
  recipe at 85.6 tok/s (−39 %) — counter-intuitively, the "wasteful"
  large-window config amortises verify + KV-management overhead
  better than tight windows.
- 100 % draft acceptance is genuine: source read of
  `common/speculative.cpp` (`impl->n_acc_tokens += n_accepted;` in
  `common_speculative_accept()`) + `--verbose` run emitting
  `draft acceptance rate = 1.00000 (115 accepted / 115 generated)`.
- Master cross-check gives identical numbers within ±0.3 % noise, so
  the regression is architectural rather than a stale-commit artefact.

### Conclusion of v1 stands
On a consumer RTX 3090 with Qwen3.6-35B-A3B at Q4_K_M, **no
speculative decoding configuration is a net win** — regardless of
commit, regardless of draft-min / draft-max, regardless of which
measurement regime. H100 / H200 or NVLinked pairs may flip the sign.

## [v1.0] — 2026-04-21

Initial public release of the spec-decode benchmark matrix.

### Added
- 19-configuration bench matrix on a single RTX 3090 (of the two on s1)
  via `llama-server` at commit `9789512` (post PR #19493 merge).
- `Qwen3.6-35B-A3B-UD-Q4_K_XL` main + `Qwen3.5-0.8B-Q4_K_M` draft
  (~~vocab-matched~~, 248320 vocab size). **Corrected 2026-08-25:** the vocab
  *sizes* match, but llama.cpp rejects the pair on its special-token gate and
  ran every classic-draft measurement through the token-translation fallback.
  See [`ERRATA.md`](ERRATA.md) A2.
- llama-server-based Python bench runner (`bench_runner.py`) plus
  three shell driver scripts (`run_matrix.sh`, `run_p0_matrix.sh`,
  `run_verify_matrix.sh`).
- Analysis plots: bar chart, per-prompt heatmap, accept-vs-speed
  scatter (in `analysis/`).
- Per-config JSON results in `results/` + `results/verify/`.
- `pr_comment.md` draft for llama.cpp PR #19493 / Issue #20039.
- `BENCHMARK_ENV.md` environment snapshot (s1, 2× RTX 3090,
  i7-11700, Ubuntu 24.04).
- `collect_env.sh` helper to regenerate the env snapshot.

### Key finding
No speculative-decode configuration achieves a net speedup over the
non-speculative baseline of 135.7 tok/s. Mean decode drops 3–12 %
across ngram-cache, ngram-mod, and classic draft-model variants, with
a bimodal tail reaching 59–67 tok/s on reasoning / code prompts
despite 100 % draft acceptance. Interpretation aligned with
MoESD (arXiv 2505.19645) and Utility-Driven SD (arXiv 2506.20675):
for a 3B-active MoE, draft batch K stays below the expert-saturation
threshold (~94 tokens for this sparsity), so each drafted token pulls
new experts through the memory hierarchy and verification pays for
the union.

[v4.1]: https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090/releases/tag/v4.1
[v4.0]: https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090/releases/tag/v4.1
[v3.0]: https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090/releases/tag/v3.0
[v2.3]: https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090/releases/tag/v2.3
[v2.2]: https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090/releases/tag/v2.3
[v2.1]: https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090/releases/tag/v2.1
[v2.0]: https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090/releases/tag/v2.0
[v1.0]: https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090/releases/tag/v1.0
