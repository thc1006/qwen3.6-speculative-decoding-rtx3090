# Archived and re-measured: llama.cpp speculative decoding for Qwen3.6-35B-A3B UD-Q4_K_XL on one RTX 3090

[![DOI](https://zenodo.org/badge/1216484498.svg)](https://doi.org/10.5281/zenodo.19776558)

> [!IMPORTANT]
> **Audited 2026-08-25, extended through 2026-08-28.** This repository now holds two
> tiers, and they must not be read as one body of evidence.
>
> The **archival tier** is the published v1/v2/v3 runs, collected 2026-04-21 to
> 2026-05-07 at llama.cpp `97895129e`, `bcb5eeb64` and PR-branch `67cb0d507`,
> one run per cell. It is a single-request decode microbenchmark for exactly the
> model files, commits, hardware, prompts and flags listed below. It is not a
> benchmark of all RTX 3090 systems, of all Qwen3.6 quantisations, of all
> speculative-decoding methods, or of end-to-end voice-agent latency.
>
> The **controlled tier** is runs A to W2, 3002 arm-runs in 74 directories,
> collected 2026-08-25 to 2026-08-31. Run A is the legacy `bcb5eeb64` binary,
> kept as the comparison; every other run is post-merge master `3737e4137`.
> Each is repeated arm-runs with a matched no-speculation baseline
> inside each run, thinking suppression verified per request rather than
> assumed, concurrent client requests verified from request timestamps, full per-request text
> and token ids, and continuous GPU telemetry. Its findings are the ones to
> cite **for llama.cpp `3737e4137`**, under the model files, binary, hardware
> and workload recorded below, not for current master, which has moved and
> which carries open work on recurrent rollback, output row ordering and
> hybrid checkpoint invalidation that touches these paths directly
> ([the upstream table](#upstream-status-checked-2026-08-25-open-items-re-checked-2026-09-01)). Two limits
> are stated up front rather than
> buried: the same configuration measured **twelve times in one day spans
> 9.4 pp**, clustered by run rather than scattered, on byte-identical output
> ([A16](ERRATA.md#a16-two-runs-identical-in-every-recorded-respect-and-byte-identical-in-output-differ-by-34--on-one-arm)),
> so quote the range and not the interval; and **every thinking-off comparison
> that let the arms stop where they liked is confounded by output length**
> ([A17](ERRATA.md#a17-the-thinking-off-comparisons-are-not-comparisons-of-the-same-amount-of-work)),
> which is enough to change one published sign. Run V's hard-cap half is the
> exception (it forces every request to the same token count), but its two
> halves were not interleaved, so it measures a difference it cannot
> attribute. The thinking-on results, which is everything in the
> table below, are unaffected by the second: 5770 requests recorded
> `thinking_suppressed` false and every one of them ran to the 300-token cap, as
> did the 134 in runs A and B, which predate the per-request field.
>
> The audit **retracted this repository's headline mechanism.** Earlier versions
> reported "100 % draft acceptance yet slower, therefore MoE expert-loading
> overhead". That 100 % is an artefact of how llama.cpp counts acceptance on
> this model class, not a measurement; see
> [The "100 % acceptance" retraction](v4_audit_2026_08_25/README.md#the-100--acceptance-retraction). Three
> further defects turned up that no earlier version noticed: the draft model was
> never actually vocabulary-compatible, three quarters of the v1 requests
> returned truncated thinking rather than answers, and `llama-server` plus a
> draft model aborts on this model at `bcb5eeb64`. Every corrected item, with
> the evidence that settles it, is in [`ERRATA.md`](ERRATA.md); the queue that
> closes what is still open is [`RETEST_TODO.md`](RETEST_TODO.md).
>
> **The negative observation survives for the methods v1 tested, and only for
> those.** With an external draft model, speculation still loses badly here, and
> batching widens the gap rather than closing it. With the target's *own* layers
> as the drafter, DFlash and the model's built-in multi-token-prediction head,
> it wins at short draft windows and one request at a time. How much it wins by
> depends on which invocation you measure, and the width of that band is not
> rounding: the same DFlash configuration was measured **twelve times** on
> 2026-08-26 and spans **+17.3 % to +26.7 %**,
> on byte-identical output and identical draft counts, while the no-speculation
> reference beside it holds to a CV of 0.42 %. The values cluster by run, and
> only this one arm moves between the clusters
> ([ERRATA A16](ERRATA.md#a16-two-runs-identical-in-every-recorded-respect-and-byte-identical-in-output-differ-by-34--on-one-arm)).
> "Speculative decoding loses on this hardware" was a statement about a regime
> this repository had not separated.

<!-- A contents block, because these documents are linked into by section name from each other and a reader arriving cold had no way to orient but to scroll. Generated from the headings; `analysis/check_links.py` validates every anchor here, so a heading renamed without this list fails the static job rather than rotting quietly. -->

## Contents

- [Reproduction](#reproduction)
  - [Reproducing the 2026-08-26 runs](#reproducing-the-2026-08-26-runs)
- [Data map](#data-map)
  - [The 2026-08-25/26 controlled runs](#the-2026-08-2526-controlled-runs)
  - [Tooling](#tooling)
- [The rest of the evidence](#the-rest-of-the-evidence)
- [Upstream status: checked 2026-08-25, open items re-checked 2026-09-01](#upstream-status-checked-2026-08-25-open-items-re-checked-2026-09-01)
- [Related reading](#related-reading)
  - [Open upstream issues in the same territory](#open-upstream-issues-in-the-same-territory)
- [Licence](#licence)
- [Citation](#citation)
- [Author](#author)

## The result

On one RTX 3090, at llama.cpp commit
`97895129e5f2bde94d13dc01ca41ee79e9b629f2`, with
`Qwen3.6-35B-A3B-UD-Q4_K_XL`, greedy decoding, and the ten committed prompts,
every tested condition that recorded speculative activity had lower
request-mean **and** lower pooled decode throughput than its matched
no-speculation reference.

The direction holds *for the conditions v1 tested*. The *explanation* published
alongside it does not. Re-run on a binary where llama.cpp counts acceptance
correctly, real acceptance and decode rate correlate at **r = +0.998** across
the ten prompts: the slowdown tracks low acceptance and draft-path cost, which
is ordinary speculative-decoding economics. The "100 % acceptance yet slower,
therefore an MoE pathology" anomaly this repository was built around does not
exist.

**And the direction is not universal.** On 2026-08-26, eight speculative
configurations and a no-speculation baseline were measured on this card in one
matrix under one memory policy, as a **Latin square balanced for position**:
nine blocks, each arm appearing exactly once per block and visiting every
position exactly once, verified from the execution log rather than from the
design. It is not balanced for carryover, and `analysis/carryover.py` refuses to
report a predecessor contrast for it. Run W is the design that balances both.
Its analysis plan was finalised at 360 of run W's 500 arm-runs and before the
completed dataset was committed
(`v4_audit_2026_08_25/PROSPECTIVE_ANALYSIS_PLAN_W.md`, which says so itself and
declines the word preregistration). Run W2's plan is the one that is an
ancestor of the commit carrying its data. Each change below is
paired against the baseline measured **inside the same block**, and the interval
is over blocks, which is the unit of replication and of resampling.

`analysis/paired_blocks.py` computes two of them: a percentile bootstrap that
resamples whole blocks, and a Student-t interval on the log ratios. **The column
below is the t interval**, which is the wider of the two on every row here. The
bootstrap can only ever resample the nine values it has, so at this block count
it under-covers, and quoting the narrower one would be the wrong direction to
err in. Both are in each run's `paired_blocks.json`.

> [!IMPORTANT]
> **Read that interval as within-invocation, because that is all it covers.**
> The nine blocks are nine sequential blocks of one invocation of the driver,
> in one fixed rotation, in whatever performance regime that invocation was in.
> The headline arm measured **twelve times in one day** spans **+17.3 % to
> +26.7 %**, on byte-identical output and identical draft counts, and
> [A16](ERRATA.md#a16-two-runs-identical-in-every-recorded-respect-and-byte-identical-in-output-differ-by-34--on-one-arm)
> could not find a recorded field that distinguishes the regimes. So the
> honest headline for `spec-dflash-n2` is:
>
> | | |
> |---|---|
> | across repeated invocations | **+17 % to +27 %** |
> | run O2, point estimate | +26.3 % |
> | run O2, t interval over its own nine blocks | [+25.5 %, +27.1 %] |
>
> The twelve runs are not twelve independent measurements of the configuration
> either (they mix the stock and instrumented builds, two-, three- and
> nine-arm matrices, and different neighbouring treatments), so the range is a
> bound on what was observed, not a confidence interval. Pooling their 43
> blocks would not fix that: the blocks are nested inside invocations, and the
> invocation is the level the variation lives at. A design that identifies it
> needs the invocation as the resampling unit, the build stratified, and the
> preceding treatment recorded; the fixed rotation used here balances position
> but not first-order carryover, since each arm's predecessor is nearly always
> the same arm.

| arm | pooled tok/s | change | 95 % CI (t, blocks within this invocation) | draft/gen ‡ | acceptance † |
|---|---:|---:|---:|---:|---:|
| **`spec-dflash-n2`** | **146.2** | **+26.3 %** | [+25.5 %, +27.1 %] | 0.81 | 72.3 % |
| `spec-mtp-n2` | 141.9 | +22.7 % | [+22.1 %, +23.3 %] | 0.77 | 78.4 % |
| `spec-dflash-n4` | 137.9 | +19.2 % | [+18.5 %, +19.9 %] | 1.24 | 55.2 % |
| **no speculation** | **115.7** | — | — | 0.00 | — |
| `ngram-map-k4v-m8` | 115.4 | −0.3 % | [−0.6 %, +0.0 %] | **0.01** | 50.0 % |
| `ngram-mod-n24` | 103.1 | −10.9 % | [−11.4 %, −10.5 %] | 0.19 | 5.0 % |
| `ngram-cache` | 93.7 | −19.0 % | [−19.4 %, −18.6 %] | 0.17 | 5.2 % |
| `spec-draft-n8` | 30.9 | −73.3 % | [−73.5 %, −73.2 %] | 1.86 | 29.5 % |
| `spec-draft-n1` | 29.2 | **−74.8 %** | [−74.9 %, −74.7 %] | 0.50 | **69.7 %** |

‡ Draft tokens proposed per token generated. It is the column that makes the
acceptance column readable, and it is why `ngram-map-k4v-m8` is not the
half-accepted success its 50.0 % suggests: it drafted **216 tokens across 27 000
generated**, one per 125, so its acceptance rate is 108 of 216 and it neither
helps nor hurts because it almost never fires. `ngram-mod-n24` and `ngram-cache`
are the opposite: they draft on a sixth to a fifth of tokens, 0.17 and 0.19 per
token generated, and have almost all of it rejected, which is what a 10–19 %
loss is made of.

![Eight speculative configurations, one baseline, one matrix](analysis/plot_head_to_head.png)

## Reproduction

The historical scripts and raw files are all here, and every host-specific path
is now an environment variable with a documented default. They remain audit
artefacts rather than a one-command reproducer: the v1 warm-up is short, there
is one run per cell, and [D5](ERRATA.md#d5-the-committed-v2-script-does-not-produce-the-committed-v2-directories)
records a provenance gap in v2.

Build the **exact** v1 revision, not current master:

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

A corrected harness for *new*, controlled runs, with one pinned binary for every
arm, ABBA ordering, N repeats, a manifest that hashes the binary and both
models, and per-request capture of the generated text, the reasoning channel,
the stop reason, the full `timings` block, and token IDs via `logprobs` — is
[`bench/retest_runner.py`](bench/retest_runner.py). It is what produced the
audit measurements above. Note that llama.cpp renamed the speculative arguments
after `bcb5eeb64` (`--draft-max` → `--spec-draft-n-max`) and that `--spec-type`
now defaults to `none`, so `-md` alone loads a draft model and never
speculates; the runner's `BENCH_FLAVOR` switch handles both spellings.

### Reproducing the 2026-08-26 runs

Every setting below is recorded in the corresponding `manifest.json`, so this
recipe is checkable against the committed data rather than trusted.

```bash
git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp
git checkout 3737e41370da1830a44c663f9929a0f27591ffa6      # the audit binary
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build -j

export LLAMA_SERVER_BIN=$PWD/build/bin/llama-server
export MODEL_TARGET=.../Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
export MODEL_DRAFT=.../Qwen3.5-0.8B-Q4_K_M.gguf
export BENCH_FLAVOR=master BENCH_GPU=0 BENCH_MAX_TOKENS=300

# run I - concurrency. Client requests in flight are read back from timestamps;
# --parallel alone allocates slots that an unmodified client never uses.
for C in 1 4 8; do
  BENCH_CONCURRENCY=$C BENCH_REPEATS=3 BENCH_THINK=on \
  BENCH_ARMS=baseline,spec-draft-n8 BENCH_OUT=out/I_conc$C python bench/retest_runner.py
done

# runs J / K / L / M - speculation methods. -fit on is REQUIRED: the BF16
# drafters only load with -ngl unset, and it is applied to every arm including
# the baseline so placement policy never differs between an arm and its control.
BENCH_FIT=on BENCH_CTX=8192 BENCH_FIT_TARGET=2048 BENCH_REPEATS=3 \
MODEL_DFLASH=.../qwen36-dflash-master.gguf \
BENCH_ARMS=baseline,spec-dflash-n1,spec-dflash-n2,spec-dflash-n4,spec-dflash-n8 \
BENCH_OUT=out/K1 python bench/retest_runner.py
```

The two drafters that are not off-the-shelf downloads:

```bash
# DFlash - the archived v3 GGUF lacks `target_layers` and master rejects it
bash bench/convert_dflash.sh

# MTP - export the target's own multi-token-prediction head as a drafter.
# Stock llama.cpp: `supports_mtp_export` is already True for this architecture
# and LLM_ARCH_QWEN35MOE already declares the NEXTN tensors. The staging step
# exists only because conversion/base.py's AWQ guard dispatches on config.json's
# quant_method rather than on the tensors being exported, and every tensor in
# the --mtp export set is unquantised. The script verifies that and refuses if
# it is ever untrue.
python bench/stage_mtp_source.py
python convert_hf_to_gguf.py ~/models/qwen36-mtp-src --mtp --outtype bf16 \
       --outfile qwen36-mtp-bf16.gguf
./build/bin/llama-quantize qwen36-mtp-bf16.gguf qwen36-mtp-q8_0.gguf Q8_0
```

Then check the numbers against the documents:

```bash
python analysis/verify_claims.py     # re-derives every quoted figure, exits non-zero on drift
python analysis/check_links.py       # relative links and heading anchors
python analysis/matrix_report.py v4_audit_2026_08_25/data/matrix_*
python analysis/thermal_report.py v4_audit_2026_08_25/data/gpu_telemetry_*.csv
python analysis/plot_v4_runs.py
```

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
| [`v2_3090_followup/v2_*/`](v2_3090_followup/) | 60 v2 raw `llama-cli` logs + one `--verbose` trace |
| [`v2_3090_followup/exp2_codejson_n3/`](v2_3090_followup/exp2_codejson_n3/) | Exp 2 aggregates and script |
| [`v3_dflash_2026_05_07/`](v3_dflash_2026_05_07/) | DFlash logs, tables, script |
| [`BENCHMARK_ENV.md`](BENCHMARK_ENV.md) | hardware, software, commits, hashes for v1/v2/v3, and the v4 memory-policy table |

### The 2026-08-25/26 controlled runs

| Path | Contents |
|---|---|
| [`v4_audit_2026_08_25/README.md`](v4_audit_2026_08_25/README.md) | what each run asked, what it measured, and every control |
| [`v4_audit_2026_08_25/PREREGISTERED_PREDICTION.md`](v4_audit_2026_08_25/PREREGISTERED_PREDICTION.md) | predictions committed to git before the data existed |
| `v4_audit_2026_08_25/data/A_*`, `B_*` | `bcb5eeb64` against post-merge master; 30 requests an arm on B, and 12 on A's two speculative arms, which abort |
| `v4_audit_2026_08_25/data/C_*`, `D_*` | the thirteen-arm matrix, thinking on and verifiably off |
| `v4_audit_2026_08_25/data/E_*`, `H_*` | past the MoESD coverage threshold; the `p_min` sweep |
| `v4_audit_2026_08_25/data/matrix_I2_conc{1,4,8}_*` | concurrency, with the client requests in flight recorded |
| `v4_audit_2026_08_25/data/matrix_J2_*` | DFlash off vs on, one binary |
| `v4_audit_2026_08_25/data/matrix_K*` | the draft-length sweep and the winner under batching |
| `v4_audit_2026_08_25/data/matrix_L_think{on,off}_*` | the same arms under both workloads |
| `v4_audit_2026_08_25/data/gpu_telemetry_*.csv` | continuous 5 s GPU traces covering every run |
| `v4_audit_2026_08_25/data/smoke_*` | the gate runs that decide a matrix is safe to start |

Each run directory holds one `manifest.json` (hashing the binary and every
model, and recording the full `BENCH_*` configuration) and one
`<arm>__rep<N>.json` per arm-run with full per-request capture. The harness
also writes an `all_results.json` concatenating the same content. It is not
committed, as of 2026-09-01. This sentence used to call it "the same content"
and nothing checked that: 119 MB across 53 directories, read by no analyser,
assertion or workflow, and in `matrix_O2_latin_20260826_153711`, the run the
headline table comes from, all 81 of its records disagreed with the arm-run
files, which had been backfilled with the server's build and commit while the
concatenation had not. A derived copy nothing reads is a second place for the
truth to be, and it was already wrong in the one that mattered.

### Tooling

| Path | Contents |
|---|---|
| [`bench/retest_runner.py`](bench/retest_runner.py) | the controlled harness; it produced every v4 measurement |
| [`bench/convert_dflash.sh`](bench/convert_dflash.sh) | re-converts the DFlash drafter with post-merge master |
| [`bench/stage_mtp_source.py`](bench/stage_mtp_source.py) | stages the checkpoint so `--mtp` can export the MTP head, and verifies the export set is unquantised before doing so |
| [`analysis/verify_claims.py`](analysis/verify_claims.py) | re-derives every quoted figure from committed data **and** greps the documents for it; exits non-zero on any drift |
| [`analysis/check_links.py`](analysis/check_links.py) | relative links and heading anchors |
| [`analysis/matrix_report.py`](analysis/matrix_report.py) | per-arm request-mean, pooled, repeat SD, acceptance, drift, activation |
| [`analysis/thermal_report.py`](analysis/thermal_report.py) | throttle flags and clock drift from a telemetry trace |
| [`analysis/plot_v4_runs.py`](analysis/plot_v4_runs.py) | the batching, draft-length, acceptance-threshold, head-to-head and two-level charts, and `plot_data.json`, which `--check` compares against the data |

---

## The rest of the evidence

The argument for the headline, and every table behind it, is in
[`v4_audit_2026_08_25/README.md`](v4_audit_2026_08_25/README.md#appendix-the-evidence-sections-moved-out-of-the-root-readme).
It was in this file until 2026-09-01 and was moved rather than shortened: the
sections are unchanged and nothing was dropped.

| Section | What is in it |
|---|---|
| [What supports that result, and what limits it](v4_audit_2026_08_25/README.md#what-supports-that-result-and-what-limits-it) | the replication, the twelve measurements, the ten prompts, the designed follow-up |
| [Metric definitions](v4_audit_2026_08_25/README.md#metric-definitions) | what each column means, and what its interval is over |
| [What the v1 data support](v4_audit_2026_08_25/README.md#what-the-v1-data-support) | the three findings v1 can carry |
| [What the v1 data do not support](v4_audit_2026_08_25/README.md#what-the-v1-data-do-not-support) | and the four it cannot |
| [The "100 % acceptance" retraction](v4_audit_2026_08_25/README.md#the-100--acceptance-retraction) | the counter artefact, and what the counters actually say |
| [Where the time goes, measured, and not MoE-specific](v4_audit_2026_08_25/README.md#where-the-time-goes-measured-and-not-moe-specific) | the profile, and why the shape is not this architecture's |
| [What the audit measured on 2026-08-25 and 2026-08-26](v4_audit_2026_08_25/README.md#what-the-audit-measured-on-2026-08-25-and-2026-08-26) | the controlled runs, arm by arm |
| [v1 representative results](v4_audit_2026_08_25/README.md#v1-representative-results) | the 2026-04-21 matrix |
| [Experiment registry](v4_audit_2026_08_25/README.md#experiment-registry) | every run, what it asked and what it answered |
| [v1 hardware, software, and artefacts](v4_audit_2026_08_25/README.md#v1-hardware-software-and-artefacts) | the v1 environment |
| [Follow-up experiment caveats](v4_audit_2026_08_25/README.md#follow-up-experiment-caveats) | what the follow-ups do and do not establish |

---

## Upstream status: checked 2026-08-25, open items re-checked 2026-09-01

Checked against the GitHub API on 2026-08-25.

| PR | Status |
|---|---|
| [#19493](https://github.com/ggml-org/llama.cpp/pull/19493) server: speculative checkpointing | merged 2026-04-19 |
| [#22227](https://github.com/ggml-org/llama.cpp/pull/22227) speculative-simple: checkpoint support | merged 2026-04-22 |
| [#20075](https://github.com/ggml-org/llama.cpp/pull/20075) fix: speculative decoding broken on hybrid SSM/MoE | **closed without merge 2026-04-25** |
| [#22105](https://github.com/ggml-org/llama.cpp/pull/22105) DFlash support | merged 2026-06-28 |

`bcb5eeb64` was master on 2026-04-22 and is described here as a dated snapshot,
not as "current master". Future edits should keep using exact tested SHAs.

**And `3737e4137` is one too.** Master has moved, and four open items touch the
paths measured here. None is a code dependency of this repository; every one is
a reason its findings are bounded to the tested snapshot. Statuses re-checked
against the GitHub API on 2026-09-01.

| Open upstream | Status | What it would change here |
|---|---|---|
| [#25004](https://github.com/ggml-org/llama.cpp/pull/25004) recurrent: equal splits for recurrent-state rollback | open, last touched 2026-08-08 | Changes concurrent recurrent-rollback batching. The discussion also reports a single-stream regression and proposes a smaller one-slot design, so the upstream answer has not converged. The batching results in run I/I2/M2 are the ones at risk |
| [#27705](https://github.com/ggml-org/llama.cpp/pull/27705) fix: output reorder index space | open, last touched 2026-08-28 | Token-row versus output-row permutation and a post-decode layout mode flip, with regression tests. It does not claim to fix #27572. If it lands, [A11](ERRATA.md#a11-speculative-decoding-is-not-output-preserving-on-this-build-and-the-engine-is-deterministic-enough-to-prove-it) and the DFlash and MTP output and throughput figures all need a minimal replication |
| [#27572](https://github.com/ggml-org/llama.cpp/issues/27572) `draft-mtp` acceptance collapses to 0.0 under `-np N` | open | Backend- and workload-dependent, on HIP with long prompts. The discussion has retracted its own first two explanations, so the issue title is not a settled mechanism. The CUDA control on this RTX 3090 does not reproduce the zero acceptance |
| [#24055](https://github.com/ggml-org/llama.cpp/issues/24055) context checkpoints always invalidated on hybrid/recurrent models | open | The checkpoint machinery [A12](ERRATA.md#a12-what-the-checkpoint-path-costs-measured-with-timers-in-the-source) times |

If any of them lands, the minimum re-measurement is the O3 headline subset, run
V under a controlled mode order, the run T checkpoint subset, and a long-prompt
concurrent MTP smoke test.

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
concern this exact model family. The left column quotes each report's own
title, so the figures in it are theirs; every figure in the right column is
this repository's and is re-derived by `analysis/verify_claims.py`.

| Issue | Why it matters here |
|---|---|
| [#24055](https://github.com/ggml-org/llama.cpp/issues/24055) — context checkpoints always invalidated on hybrid/recurrent models | The checkpoint machinery this audit measured: 1639 checkpoints of **82.079 MiB** in one `n_max` 1 arm-run of ten 300-token requests, 163.9 per request. This row read 101.3 MiB until 2026-08-26, which is 82.079 + 19.266 — the draft component added a second time to a total that already contains it ([A12](ERRATA.md#a12-what-the-checkpoint-path-costs-measured-with-timers-in-the-source)) |
| [#25004](https://github.com/ggml-org/llama.cpp/issues/25004) — recurrent: support equal splits for recurrent-state rollback | The rollback path behind [A1](ERRATA.md#a1-100--draft-acceptance-is-a-counter-artefact-not-a-measurement) and [A6](ERRATA.md#a6-llama-server-plus-a-draft-model-aborts-on-this-model-at-bcb5eeb64) |
| [#24670](https://github.com/ggml-org/llama.cpp/issues/24670) — draft-mtp not activating on Turing with a hybrid SSM+attention **Qwen3.6-35B-A3B** | This repository's exact target model |
| [#25117](https://github.com/ggml-org/llama.cpp/issues/25117) — DFlash regression on AMD APU with a **quantized MoE target**, ~2× slower than baseline | An independent report of v3's direction, on different hardware |
| [#27572](https://github.com/ggml-org/llama.cpp/issues/27572) — draft-mtp acceptance collapses to 0.0 under `-np N` | A known concurrency failure mode; any batching measurement must check acceptance did not collapse rather than assume it |
| [#27569](https://github.com/ggml-org/llama.cpp/issues/27569) — cap the draft context batch instead of inheriting the target's | Bears on long-draft configurations such as the `n_max` 128 arm |
- [`thc1006/qwen3.6-vllm-2x3090`](https://github.com/thc1006/qwen3.6-vllm-2x3090) — sibling repository, different engine and hardware topology

---

## Licence

- **Code and documentation**: MIT, see [`LICENSE`](LICENSE).
- **Benchmark data**: CC0-1.0, scoped by [`DATA_LICENSE`](DATA_LICENSE), full
  text in [`LICENSES/CC0-1.0.txt`](LICENSES/CC0-1.0.txt).

## Citation

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff). Cite the DOI
above for the archived release, and state which version you are citing:
pre-audit releases (v1.0 – v3.0) contain the claims retracted in
[`ERRATA.md`](ERRATA.md).

## Author

Hsiu-Chi Tsai (`thc1006`) · `hctsai1006@cs.nctu.edu.tw` ·
[github.com/thc1006](https://github.com/thc1006)
