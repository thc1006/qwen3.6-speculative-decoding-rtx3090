# v3 — DFlash exploratory run, 2026-05-07

> [!WARNING]
> **Retracted and rescoped by the 2026-08-25 audit.** This directory's numbers
> stand as archived measurements. Its original conclusions do not.
>
> The comparison here **changed the binary and the speculation method at the
> same time**: the baseline and Oleg arms report build `b8889-bcb5eeb64`, the
> DFlash arms report `b8942-67cb0d507`. With one run per prompt/config and a
> thinking control that did not work, this is an exploratory historical
> datapoint, not a DFlash effect estimate. The mechanism claims below have been
> removed. See [`../ERRATA.md`](../ERRATA.md) items D4, D6, F1, F3, F4.

## What this directory is

Five prompts run against `Qwen3.6-35B-A3B-UD-Q4_K_XL` on one RTX 3090 with the
DFlash drafter from llama.cpp PR #22105, on 2026-05-07, one run per
prompt/config. It is archival evidence. It is not a controlled experiment.

## Confounds, stated up front

| # | Confound | Evidence |
|---|---|---|
| 1 | **Baseline and treatment are different binaries.** `01_baseline`, `03_oleg_draft_2_32`, `04_oleg_draft_2_16` report `build : b8889-bcb5eeb64`; `05/06/07_dflash_*` report `build : b8942-67cb0d507`. | the `build :` banner in every `data/out_*/*/p*.log` |
| 2 | **N = 1 per prompt/config.** No repeats, so no run-to-run uncertainty can be estimated. | 5 logs per config |
| 3 | **The thinking control did not work.** Every script passes `-no-cnv` and appends `/no_think`. `llama-cli` prints `--no-conversation is not supported by llama-cli / please use llama-completion instead`, and the logs then contain `[Start thinking]` and a full reasoning trace. 30 of 33 v3 logs show both. | `data/out_*/*/p*.log` |
| 4 | **Outputs were not token-identical** between conditions, so the arms did not generate the same work. | no output comparison was performed |
| 5 | **No isolating control.** No target-precision, draft-precision, dense-model, profiler, second-GPU, or expert-routing measurement was collected. | — |
| 6 | `--temp 0.5` with a fixed seed, so the arms are not greedy-deterministic. | `bench/bench_dflash.sh` |

## Archived results

Generation tok/s as reported by `llama-cli`, one run per cell. Re-derived from
the raw logs during the audit and reproduced exactly.

| config | build | p1 | p2 | p3 | p4 | p5 | mean | sd |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `01_baseline` (no spec) | `b8889-bcb5eeb64` | 136.3 | 139.3 | 140.0 | 139.4 | 139.4 | **138.9** | 1.47 |
| `03_oleg_draft_2_32` | `b8889-bcb5eeb64` | 63.6 | 76.0 | 63.5 | 62.3 | 62.2 | 65.5 | 5.89 |
| `04_oleg_draft_2_16` | `b8889-bcb5eeb64` | 66.0 | 75.9 | 63.5 | 65.5 | 61.9 | 66.6 | 5.47 |
| `05_dflash_max16` | `b8942-67cb0d507` | 60.1 | 78.8 | 62.4 | 69.6 | 57.9 | 65.8 | 8.51 |
| `06_dflash_max8` | `b8942-67cb0d507` | 71.3 | 90.8 | 78.2 | 75.2 | 69.5 | **77.0** | 8.42 |
| `07_dflash_max4` | `b8942-67cb0d507` | 72.1 | 79.4 | 74.1 | 78.2 | 70.8 | 74.9 | 3.76 |

The `sd` column is spread across five different prompts, not repeated-run
uncertainty — each cell was measured once.

`02_srogmann_ngmod_n24` produced no timing data. Its log contains only:

```
error: invalid argument: --spec-type
```

The original note said `--spec-type` is "not in master". That is wrong. The
argument exists at both `97895129e` and `bcb5eeb64`, registered in
`common/arg.cpp` as `.set_examples({LLAMA_EXAMPLE_SERVER})` — it is accepted by
`llama-server` and rejected by `llama-cli`. v1 used `llama-server` and
exercised it successfully. See [`../ERRATA.md`](../ERRATA.md) item D6.

## What this run supports

Under the commands actually executed on 2026-05-07, every DFlash and
draft-model configuration produced a lower generation rate than the run
recorded as its baseline.

## What this run does not support

- **A DFlash effect estimate.** Confound 1 alone prevents it.
- **That Q4 target quantisation "collapses the technique".** No FP16 or BF16
  target was ever run. The earlier recommendation to that effect has been
  removed; it was a hypothesis presented as a finding.
- **That MoE expert routing or Ampere bandwidth explains the result.** Nothing
  in this directory measures expert routing, HBM traffic, or kernel time.
- **That the mechanism "generalises to DFlash".** Two configurations being slow
  is not evidence that they are slow for the same reason.
- **That co-trained speculative heads are "the only positive yield path on this
  hardware".** Only a handful of methods were tested, and the vLLM comparison
  cited for that claim uses a different engine, two GPUs, tensor parallelism, a
  different quantisation stack, and a different protocol. The defensible
  statement is that a separately tested vLLM MTP configuration was positive on
  different hardware topology.
- **Any "first public datapoint" claim.** No novelty search was performed, and
  none is recorded. Removed.

## Setup as recorded

- 1 × RTX 3090 24 GiB, driver 580.126.09, stock 350 W.
- Target `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`, sha256 `707a55a8…f4450`.
- Draft (Oleg arms) `Qwen3.5-0.8B-Q4_K_M.gguf`, sha256 `bd258782…dc517`.
- DFlash drafter: `z-lab/Qwen3.6-35B-A3B-DFlash` converted to BF16 GGUF with
  PR #22105's `convert_hf_to_gguf.py --target-model-dir`.
- Args: `-ngl 999 -c 4096 -fa on -ctk q8_0 -ctv q8_0 -n 200 --temp 0.5 --seed 42 -no-cnv -st`
  (note `-no-cnv` was rejected, see confound 3).
- `BENCHMARK_ENV.md` records `llama-cli --version` as
  `8889 (bcb5eeb64) -- inherited from master at fork point`, while the run logs
  report `b8942-67cb0d507`. **The logs are authoritative**; the `--version`
  string was captured before the DFlash rebuild. This is why a run manifest
  must hash the binary rather than trust `--version`.

## Upstream status, checked 2026-08-25

llama.cpp PR #22105 (DFlash) was **merged on 2026-06-28**. Earlier text in this
directory described it as an open draft. The archived measurements come from
the **pre-merge** branch `67cb0d507` and are not measurements of the
implementation now in master.

## What a valid DFlash result would need

One pinned post-merge binary, DFlash disabled and enabled, ABBA-ordered, at
least five repeats per cell, a thinking control verified in the output, full
per-request capture, and the binary's sha256 in the manifest. That work is
queued as P2 in [`../RETEST_TODO.md`](../RETEST_TODO.md).

## Files

| Path | Contents |
|---|---|
| `data/out_20260507_183341/` | raw per-prompt logs, seven configs |
| `data/out_20260507_183341/manifest.txt` | GPU state and model file listing at run start |
| `data/run_dflash_2026_05_07.log`, `data/run_03_04_2026_05_07.log` | driver logs |
| `bench/bench_dflash.sh` | the script, kept as-is with an errata header |

## Licence

Same as the repository root: code and documentation MIT
([`../LICENSE`](../LICENSE)), benchmark data CC0-1.0
([`../DATA_LICENSE`](../DATA_LICENSE)). An earlier version of this file
claimed Apache 2.0; that was never correct for this repository and has been
removed.
