# Retest TODO

Work queue that closes the open items in [`ERRATA.md`](ERRATA.md) with new
measurements. Every task below is runnable **today** on the original v2/v3
bench host.

## Environment confirmed 2026-08-25

Probed read-only over Tailscale, then used for the runs recorded below.

| | `3090` (100.112.135.98) — the v2/v3 bench host | `thc1006-debian13` (this box) |
|---|---|---|
| GPU | 1 × RTX 3090, **82 MiB used, 0 % util — idle** | 1 × RTX 3090, **20.2 GiB used** by a qwen3.8 `llama-server` |
| driver | 580.173.02 (was 580.126.09 at bench time) | 610.43.02 |
| disk free | **262 GiB** | 29 GiB — too small for the 22 GiB target |
| target model | `~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` ✅ | absent |
| draft model | `~/models/Qwen3.5-0.8B-Q4_K_M.gguf` ✅ | absent |
| DFlash drafter | `~/models/qwen36-dflash.gguf` ✅ | absent |
| llama.cpp | `~/bench/llama.cpp` @ `bcb5eeb64`, branch `pr-22105` present, libs built for `8863`/`8889`/`8942` | `llamacpp-master` @ `c060ca9`, `llamacpp-dflash2` @ `d1a522f` |
| toolchain | `nvcc`, `gcc`, `cmake`, `ninja` present; `git fetch` reaches upstream (master now `c1d0e7a00`) | CUDA 13.3, no nvcc on PATH |
| `llama-completion` | present ✅ | not built |
| gguf tooling | `gguf_set_metadata.py` / `gguf_new_metadata.py` present, but system python has **no numpy/tqdm** | — |

**All retests run on `3090`.** Per the fleet rule, binaries do not travel
between hosts; only within-host deltas are comparable. This box is the wrong
place: no models, not enough disk, and its GPU is busy with other work.

Two flags confirmed live on `bcb5eeb64`, both of which the historical scripts
should have used and did not:

- `--spec-type` — present in `llama-server --help`, **absent** from
  `llama-completion --help`. Confirms [D6](ERRATA.md#d6---spec-type-is-not-missing-from-master-it-is-server-only) by execution, not just by source.
- `-rea off` / `--reasoning-budget 0` / `--reasoning-format` — the real
  thinking switches. `/no_think` in the prompt text was never one.

---

## Results so far — runs executed 2026-08-25

All on the `3090` host. Every run's manifest records the binary's sha256, both
model sha256s, the full argv, and GPU telemetry before and after each arm.

### P0-1 · DONE — the BOS defect is real, and it is **not** the cause

The fix turned out not to need a file edit at all. `--override-kv
tokenizer.ggml.bos_token_id=int:248044` propagates to the draft model
(`server-context.cpp` builds the draft params as `auto params_dft =
params_base;`, so `kv_overrides` carries over) and flips `vocab_cmpt` from `0`
to `1`. That also proves the two token arrays are byte-identical, since the
gate's per-token text comparison from id 5 to 248320 only runs after the
special-token check passes.

`gguf_set_metadata.py` would **not** have worked: it exits with
`! Field ... not found` for a key that is absent, because it memory-maps the
file and overwrites a scalar in place. `gguf_new_metadata.py
--special-token-by-id bos 248044` is the file-rewrite alternative, and it needs
numpy and tqdm, which the host's system python does not have.

Same binary (`llama-server` @ `bcb5eeb64`), same draft file, same flags,
ABBA-ordered, 2 repeats:

| arm | `long_explain` | counted draft tokens | `code_small` |
|---|---:|---|---|
| translation fallback | 48.4 tok/s | 97 / 97 | **abort** |
| matched vocabulary | 51.1, 50.0 tok/s | 97 / 97 | **abort** |
| baseline (quiet host) | ~125–129 tok/s | — | fine |

**Verdict: the gate defect costs about 3–5 %, not 60 %.** The counted
draft-token totals are identical across arms, so the translation path was not
changing what got drafted. The negative finding survives and is now measured on
a matched path. Written up as ERRATA A2.

### A6 · NEW — `llama-server` + draft aborts at `bcb5eeb64`

Reproduced 3 / 3 on `code_small`, in both arms, immediately after a
partial-accept checkpoint restore:
`CUDA error: an unsupported value or parameter` in
`ggml_cuda_op_mul_mat_cublas`, from `server_context_impl::update_slots()`.
The no-speculation arm completes every time. Written up as ERRATA A6.

### P2-1 · DONE — post-merge master built

Pinned at `3737e41370da1830a44c663f9929a0f27591ffa6` (build 10622), CUDA arch
86, in a separate worktree at `~/bench/llama-retest`. Two things changed
upstream since `bcb5eeb64`:

- `--draft-max` / `--draft-min` were **removed**; they are now
  `--spec-draft-n-max` / `--spec-draft-n-min`.
- `--spec-type` gained `draft-simple`, `draft-eagle3`, `draft-mtp`,
  `draft-dflash`, `draft-dspark` alongside the ngram family. **DFlash is now
  `--spec-type draft-dflash`**, and EAGLE-3 and MTP — which this repository's
  older text listed as "not evaluated here" — are available on one binary.

**Trap worth recording:** on master, `--spec-type` defaults to `none`, so
passing `-md` alone loads the draft model and then never speculates. A first
master run looked like a clean "no crash, no slowdown" result until the server
log showed zero `generate_draft` calls and `draft_n = 0` on all thirty
requests. Any master comparison must pass `--spec-type` explicitly.

### Still running / next

- P0-1 on master with `--spec-type draft-simple`: does the abort persist, and
  what is the slowdown on a post-merge binary?
- No-speculation baseline is **~6 % faster on master** than on `bcb5eeb64`
  on the same host (133–137 vs 125–129 tok/s), so absolute rates must not be
  compared across those two binaries.

---

## P0 — decisive and cheap. Do these first.

### P0-1 · The draft GGUF is missing one metadata key ⭐ — **DONE, see Results above**

Kept for the record. The conclusion is in "Results so far": the defect is real,
the fix works, and it accounts for 3–5 % rather than the 60 % that needs
explaining.

The two tokenizers are *identical*: same `tokenizer.ggml.model = gpt2`, same
`pre = qwen35`, same 248320 tokens, same 247587 merges, same
`eos_token_id = 248046`, same `padding_token_id = 248055`. Exactly one key
differs:

| | target `Q4_K_XL` | draft `0.8B-Q4_K_M` |
|---|---|---|
| `tokenizer.ggml.bos_token_id` | `248044` | **key absent** |
| resolved BOS | `248044 '<\|endoftext\|>'` | **`11 ','`** (fallback) |

`common_speculative_are_compatible()` tests
`llama_vocab_bos(tgt) != llama_vocab_bos(dft)` → `248044 ≠ 11` → incompatible →
llama.cpp silently switches to the token-translation path
([A2](ERRATA.md#a2-the-draft-model-was-not-vocabulary-compatible-the-run-used-the-token-translation-fallback)).
Every classic-draft number this repository has ever published was measured on
that path.

The working recipe — no file edit needed, and a no-op for the target:

```bash
./build/bin/llama-server -m ~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    -md ~/models/Qwen3.5-0.8B-Q4_K_M.gguf -ngl 999 -c 4096 -v \
    --override-kv tokenizer.ggml.bos_token_id=int:248044 2>&1 \
  | grep -E "Using metadata override|vocab_cmpt|not compatible"
# vocab_cmpt = 1, and the "not compatible" line is gone
```

Then run the A/B — same target, same draft weights, same binary, same flags,
only the BOS key differs:

- arm A: `-md Qwen3.5-0.8B-Q4_K_M.gguf` (translation path, reproduces history)
- arm B: `-md Qwen3.5-0.8B-Q4_K_M-bosfix.gguf` (matched path)
- plus the no-spec baseline, ABBA-ordered, 10 v1 prompts × 5 repeats.

**Cost** ~20 min GPU. **Closes** A2, and re-opens or confirms the headline.
**Interpretation:** if arm B recovers most of the gap, the published negative
result was a draft-GGUF metadata bug; if it does not, the negative result
becomes far stronger and is finally measured on a clean path.

> Also check whether `add_bos_token` needs setting to `false` to match the
> target, and whether upstream unsloth has since republished the draft GGUF
> with the key present.

### P0-2 · Thinking control that actually works

Replace `llama-cli … -no-cnv … "prompt /no_think"` with `llama-completion` plus
a real switch, and **prove** it in the output.

```bash
# on 3090, ~5 min GPU
./build/bin/llama-completion -m ~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    -ngl 999 -c 4096 -fa on -ctk q8_0 -ctv q8_0 -n 200 --temp 0 --seed 42 \
    --jinja -rea off --reasoning-budget 0 \
    -p "Explain TCP vs UDP in 3 concise bullet points." 2>&1 | tee /tmp/think_off.log
grep -c "Start thinking\|<think>" /tmp/think_off.log   # must be 0
```

Compare `-rea off`, `--reasoning-budget 0`, and both together; pick whichever
provably suppresses thinking and record which one was used. **Cost** ~5 min.
**Closes** D1, D2, and unblocks any Exp 2 replacement.

### P0-3 · Capture the *true* acceptance rate for every config

Every historical run except one used no `-v`, so only one prompt in the whole
repository has real acceptance data. Re-run the v2/v1 config set with `-v` and
harvest the drafter's own `statistics draft:` line per request.

```bash
# on 3090, ~30 min GPU
# for each config: run with -v, then
grep -E "statistics draft:|draft acceptance rate" run.log
python analysis/verbose_accounting.py run.log   # already handles this format
```

**Cost** ~30 min. **Closes** A1 quantitatively across the matrix instead of
n = 1, and gives the real acceptance-vs-speedup scatter that
`plot_accept_vs_speed.png` was supposed to show.

---

## P1 — the properly powered v1-style retest

### P1-1 · One binary, ABBA order, N ≥ 5, full capture

Everything the v1 matrix lacked. Skeleton is in
[`bench/retest_runner.py`](bench/retest_runner.py) — **never executed**;
review before trusting it.

Requirements:

- one pinned binary for **all** arms; record its `sha256sum` and
  `git rev-parse HEAD` in the manifest, never `--version` alone
  ([D4](ERRATA.md#d4-v3-dflash-compares-two-different-binaries))
- ABBA / randomised config ordering, not baseline-then-treatments
- ≥ 5 repeats per prompt × config, so `±` finally means run-to-run uncertainty
  ([B2](ERRATA.md#b2-the--column-was-across-prompt-spread-not-repeated-run-uncertainty))
- full-shape warm-up, not one 8-token completion
- persist per request: generated text, the **reasoning channel**, stop reason,
  `timings`, `draft_n` / `draft_n_accepted`, the `-v` drafter statistics, and
  token IDs via `logprobs` (near-complete: `probs_output` drops trailing stop-word tokens, `server-context.cpp:2036-2039`, so the list can run a few short of `predicted_n`, which stays the authority for token counts)
- persist per run: argv, binary sha256, model sha256s, `nvidia-smi` telemetry
  before/after ([D5](ERRATA.md#d5-the-committed-v2-script-does-not-produce-the-committed-v2-directories))
- separate a deterministic `temperature=0` timing study from any realistic
  `temperature>0` study; do not mix them in one table

**Cost** ~2.5–3 h GPU for 19 labels × 5 repeats.

### P1-2 · The missing fp16-KV no-speculation baseline

`ngcache-kv-fp16` has no matched control, so it cannot separate a speculation
effect from a KV-precision effect. The new heatmap shows fp16 KV is 1–2 %
*faster* than q8_0 when speculation is idle. Add `baseline-kv-fp16`.
**Cost** ~10 min. **Closes** [B7](ERRATA.md#b7-the-fp16-kv-row-is-a-one-sided-control).

### P1-3 · Length-matched long-output comparison

The 1000-token runs stop at different lengths per prompt and per config, so
`predicted_n` is not constant ([B3](ERRATA.md#b3-all-completions-reach-the-cap-is-false-for-the-1000-token-variants)).
Either force `ignore_eos` + a hard cap so all arms generate identical counts,
or report pooled throughput only and state the differing lengths.
**Cost** ~30 min.

### P1-4 · Repair the prompt set

- `zh_cn` → `zh_hant`; the prompt is Traditional Chinese ([C2](ERRATA.md#c2-the-zh_cn-prompt-is-traditional-chinese)).
- `multi_turn_1` / `multi_turn_2` are two independent single-turn requests, and
  `medium_rec` refers to a turn that never happened ([C3](ERRATA.md#c3-multi_turn_1--multi_turn_2-are-not-multi-turn)).
  Either build a real multi-turn arm that carries conversation state and reuses
  the prefix cache, or rename the tags to stop implying one.
- Ten hand-written prompts cannot support a prompt taxonomy. If prompt-class
  effects are to be claimed, sample a larger set with a pre-registered
  partition ([B5](ERRATA.md#b5-the-regression-is-entirely-bimodal-by-prompt-class-is-false-for-the-ngram-mod-family)).

### P1-5 · Host isolation, clocks and thermals

GPU 0 on the v1 host ran Ollama during the v1 matrix
([C4](ERRATA.md#c4-gpu-0-was-running-another-workload)). The `3090` host is a
single-GPU box, so that part is already better.

Two further factors no historical run controlled, now instrumented
([C4b](ERRATA.md#c4b-stock-clocks-was-measured-once-before-the-load)):

- **Overclocking.** `power.limit` vs `power.default_limit` vs `power.max_limit`
  is the fingerprint. Measured 350 / 350 / 350 W — stock.
- **Thermal and power-cap downclocking across a multi-hour run.**
  [`bench/gpu_telemetry.sh`](bench/gpu_telemetry.sh) samples clocks, power,
  temperature, pstate and `clocks_throttle_reasons` every 5 s for the whole
  run; `bench/retest_runner.py` also snapshots before and after each arm.
  Measured under sustained load: 65–68 °C against a ~83 °C throttle point,
  1815–1935 MHz of 2100, 246 W peak of 350, throttle bitmask constantly `0x4`
  (SW power cap) and **never** a thermal bit. Note `temperature.memory` reads
  `N/A` on this card, so GDDR6X junction temperature is not observable here.

Beyond capturing it, the matrix repeats its no-speculation baseline five times
spread across the run, so drift is testable from the measurement itself: if
rep 0 and rep 4 of `baseline` agree, no drift large enough to matter occurred.
**Cost** free.

---

## P2 — DFlash, done properly

### P2-1 · Build one post-merge binary

PR #22105 merged upstream on **2026-06-28**, so current master carries DFlash.
The archived v3 comparison used `b8889-bcb5eeb64` for baseline and
`b8942-67cb0d507` (the pre-merge PR branch) for DFlash — two different binaries
([D4](ERRATA.md#d4-v3-dflash-compares-two-different-binaries)).

```bash
# on 3090; upstream master is c1d0e7a00 as of this probe
cd ~/bench/llama.cpp && git fetch origin && git checkout <pinned-post-merge-sha>
cmake -B build-retest -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DCMAKE_BUILD_TYPE=Release -DGGML_CCACHE=ON -GNinja
cmake --build build-retest -j
sha256sum build-retest/bin/llama-server   # goes in the manifest
```

Pin an explicit SHA — do not track a moving master. **Cost** ~30–45 min CPU
with ccache warm.

### P2-2 · DFlash off vs on, one binary

Same binary, same target, same prompts, DFlash disabled vs enabled, ≥ 5
repeats, ABBA. Also re-check whether the DFlash drafter GGUF has the same
missing-BOS problem as P0-1, and whether it still needs re-conversion against
the merged converter. **Cost** ~1.5 h GPU. **Closes** D4 and lets the v3
conclusion be restated or retracted.

---

## P3 — mechanism: separate draft cost from MoE cost

The repository currently cannot say which term dominates
([A4](ERRATA.md#a4-a-measured-cost-decomposition-was-in-the-repository-the-whole-time-and-was-never-used)).

### P3-1 · Draft-length sweep with cost instrumentation

Sweep `γ ∈ {1, 2, 4, 8, 16, 32, 64}` and record per run: drafter
`generate()` ms, target verify ms, accepted length per step, discarded-round
count, checkpoint bytes written and restored. All of these are already in the
`-v` output; `analysis/verbose_accounting.py` parses them.

If the slowdown tracks discarded rounds and drafter time, mechanism 1 wins and
the MoE story is unnecessary. **Cost** ~1.5 h.

### P3-2 · Does the partial-acceptance fallback still exist?

The `COMMON_CONTEXT_SEQ_RM_TYPE_FULL` → `continue` path is what makes the
acceptance counter a tautology and what forces a 62.8 MiB checkpoint restore on
every partial accept ([A1](ERRATA.md#a1-100--draft-acceptance-is-a-counter-artefact-not-a-measurement),
[A3](ERRATA.md#a3-the-tested-build-had-a-known-broken-speculative-path-for-this-model-class-and-the-fix-was-never-merged)).
PR #20075, which fixed the underlying hybrid-SSM rollback, was **closed without
merge**. Check current master's `server-context.cpp` and re-run
`common_context_can_seq_rm` against this target on the post-merge binary. If it
still returns `FULL`, that is a reportable upstream finding in its own right —
and a defensible reason to open an issue with this repository's data attached.
**Cost** ~20 min.

### P3-3 · Expert-routing instrumentation (optional, expensive)

Nothing here has ever measured expert routing. Testing the MoESD story needs a
patched build counting unique experts activated per layer per verification
step, plus HBM traffic and kernel time. Only worth doing after P3-1 shows the
draft-path terms do **not** account for the slowdown. **Cost** days.

### P3-4 · Controls this repository still lacks

Dense-model control, FP16/BF16 target control, and a second-GPU control are all
absent. The 3090's 24 GiB cannot hold a BF16 35B target, so the FP16 control
needs different hardware or a smaller target — note it as out of scope rather
than leaving it as an unstated gap.

---

## P4 — repository hygiene, no GPU needed

| Item | Status |
|---|---|
| `ERRATA.md` with every corrected claim + evidence | ✅ written |
| README rewritten around scope, metrics, retraction | ✅ written |
| `analysis/plot.py`: pooled throughput, median, activation, honest error bars | ✅ done |
| `analysis/verbose_accounting.py`: reconstructs the counter artefact | ✅ done |
| `plot_accept_vs_speed.png` retracted, replaced by `plot_acceptance_accounting.png` | ✅ done |
| Exp 2 `results.json` interpretation neutralised | ✅ done |
| v2 `SUMMARY.md` / `README.md` errata banners | ✅ done |
| v3 `README.md`: binary confound, PR status, licence, causal claims | ✅ done |
| `BENCHMARK_ENV.md`: `-no-cnv` description, CUDA versions, `--version` vs build | ✅ done |
| `pr_comment.md` errata banner | ✅ done |
| `CHANGELOG.md` audit entry | ✅ done |
| Historical scripts: errata headers + env-var paths | ✅ done |
| `bench/retest_runner.py` (corrected, **executed**) | ✅ done |
| `SHA256SUMS`, `CITATION.cff`, `DATA_LICENSE`, `LICENSES/CC0-1.0.txt` | ✅ done |

---

## Suggested order

1. **P0-1** — one metadata key decides whether the headline survives. ~20 min.
2. **P0-2**, **P0-3** — cheap, and every later run depends on them. ~35 min.
3. **P1-1**, **P1-2** — the first properly powered dataset. ~3 h.
4. **P2** — restate or retract v3. ~2 h.
5. **P3-1**, **P3-2** — mechanism, and a possible upstream report. ~2 h.

Total to a defensible v4: roughly **8 hours of `3090` GPU time**, none of it
blocked on hardware, downloads, or upstream.
