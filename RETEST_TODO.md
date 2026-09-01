# Retest TODO

Work queue that closes the open items in [`ERRATA.md`](ERRATA.md) with new
measurements. Every task below is runnable **today** on the original v2/v3
bench host.

## Environment confirmed 2026-08-25

Probed read-only over Tailscale, then used for the runs recorded below.
† marks a figure that was read off a host on that date and is not
reproducible from anything in this archive. Every figure in the
`thc1006-debian13` column is one, because nothing here was measured on
that machine, and so are the bench host's free disk and driver. The
`3090` column's GPU reading is not: every run manifest in
[`v4_audit_2026_08_25/data/`](v4_audit_2026_08_25/data/) records the same
`82 MiB, 0 %` from its own `nvidia-smi`.

| | `3090` (100.112.135.98) — the v2/v3 bench host | `thc1006-debian13` (this box) |
|---|---|---|
| GPU | 1 × RTX 3090, **82 MiB used, 0 % util — idle** | 1 × RTX 3090 †, **20.2 GiB used** † by a qwen3.8 `llama-server` |
| driver | 580.173.02 † (was 580.126.09 at bench time) | 610.43.02 † |
| disk free | **262 GiB** † | 29 GiB †, against 21 GiB for the target and 508 MiB and 905 MiB for the two drafters |
| target model | `~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` ✅ | absent |
| draft model | `~/models/Qwen3.5-0.8B-Q4_K_M.gguf` ✅ | absent |
| DFlash drafter | `~/models/qwen36-dflash.gguf` ✅ | absent |
| llama.cpp | `~/bench/llama.cpp` @ `bcb5eeb64`, branch `pr-22105` present, libs built for `8863`/`8889`/`8942` | `llamacpp-master` @ `c060ca9`, `llamacpp-dflash2` @ `d1a522f` |
| toolchain | `nvcc`, `gcc`, `cmake`, `ninja` present; `git fetch` reaches upstream (master now `c1d0e7a00`) | CUDA 13.3 †, no nvcc on PATH |
| `llama-completion` | present ✅ | not built |
| gguf tooling | `gguf_set_metadata.py` / `gguf_new_metadata.py` present, but system python has **no numpy/tqdm** | — |

**All retests run on `3090`.** Per the fleet rule, binaries do not travel
between hosts; only within-host deltas are comparable. This box is the wrong
place: no models, not enough disk, and its GPU is busy with other work.

Two flags confirmed live on `bcb5eeb64`, both of which the historical scripts
should have used and did not:

- `--spec-type`: present in `llama-server --help`, **absent** from
  `llama-completion --help`. Confirms [D6](ERRATA.md#d6---spec-type-is-not-missing-from-master-it-is-server-only) by execution, not just by source.
- `-rea off` / `--reasoning-budget 0` / `--reasoning-format`: the real
  thinking switches. `/no_think` in the prompt text was never one.

---

## Status board

| # | task | state |
|---|---|---|
| P0-1 | draft-GGUF BOS key, matched vs translation A/B | **done** — real defect, worth +0.2 %, not the cause |
| P0-2 | a thinking control that actually works | **done** — `chat_template_kwargs {"enable_thinking": false}`, verified per request, 50 of 50 per arm |
| P0-3 | true acceptance across configurations | **done** — every arm of the matrix carries honest counters on post-merge master |
| P1-1 | one binary, ABBA, N ≥ 5, full capture | **done** — 13 arms × 5 repeats, 900 requests, hashed manifest |
| P1-2 | the missing fp16-KV no-speculation control | **done** — closes ERRATA B7 |
| P1-3 | length-matched long-output comparison | **partly — run V, and it needs a rerun.** Run V measured it but did not identify it: every freerun block ran before every hard-cap block (22:31:46 against 22:48:08), so the mode is confounded with time and with the invocation effect A16 found on the same drafter, which is the same size as the shift reported. What is still needed is an AB/BA crossover, or randomised mode order, with the session as the resampling unit. This turned out to matter more than the entry suggested: with thinking off the *arms* generate different token counts, not just the prompts, and controlling for it moves every model-drafting arm by +2.5 to +16.8 pp and flips one published sign ([A17](ERRATA.md#a17-the-thinking-off-comparisons-are-not-comparisons-of-the-same-amount-of-work)). `BENCH_IGNORE_EOS=on` forces the hard cap, and run V measures it: the same five arms twice in one session, once as the archive did it and once with the cap, moves every arm by 6.31 to 11.90 pp and flips `spec-dflash-n4` from −1.35 % to +10.55 % |
| P1-4 | repair the prompt set | **partial** — `zh_hant` relabelled; the extended set carries two genuinely multi-turn exchanges, and the v1 `multi_turn_*` tags keep their names and behaviour so archived joins still work |
| P1-5 | host isolation, clocks, thermals | **done** — 1317-sample trace, no OC, no meaningful throttling, drift diagnosed as cold-start |
| P2-1 | build one pinned post-merge binary | **done** — `3737e4137` |
| P2-2 | DFlash off vs on, one binary | **done** — run J. The archived drafter GGUF lacks `target_layers`; `bench/convert_dflash.sh` re-converts it. **The sign reverses**: +18.7 % at `n_max 4`. |
| P2-3 | `draft-mtp`, the method the vLLM sibling uses | **done** — run M. Nothing was blocking it; stock converter and stock runtime both support it. |
| P4-1 | does batching rescue speculation, as upstream says? | **done** — run I. No: no-speculation gains +64 % at concurrency 8, the drafter moves −8 %. |
| P4-2 | where is the draft-length optimum, and does it survive batching? | **done** — run K. A plateau at `n_max` 2–4, a cliff after it, and batching erases it (+0.4 % at 4 in flight, −74.1 % at 8). |
| P4-3 | does the win survive the workload changing? | **done, with the thinking-off figure superseded** — run L. It halves with thinking off. `n_max 4` reads negative there, but that comparison is length-confounded ([A17](ERRATA.md#a17-the-thinking-off-comparisons-are-not-comparisons-of-the-same-amount-of-work)): on the prompts where every arm generated the same number of tokens it is +14.1 %, and under a forced cap it is +10.55 %. Acceptance falls with thinking off either way. |
| P3-1 | draft-length sweep with cost instrumentation | **done** — 1…32 in run C, 64/96/128 in run E |
| P3-2 | does the partial-accept fallback still exist upstream | **done** — the abort is gone and the counter is fixed on post-merge master |
| P3-3 | expert-routing instrumentation | **not done**, and after A7 nothing demands it |
| P3-4 | dense / FP16 / second-GPU controls | **not done** — out of scope on 24 GiB |

New work the audit generated that was not on the original list:

- the workload-shape comparison Exp 2 could not make (ERRATA D3b)
- the pre-registered past-threshold prediction and its test
  ([`v4_audit_2026_08_25/PREREGISTERED_PREDICTION.md`](v4_audit_2026_08_25/PREREGISTERED_PREDICTION.md))
- `analysis/verify_claims.py`, `check_links.py`, `matrix_report.py`,
  `thermal_report.py`, `plot_v4_runs.py`, `past_threshold_fit.py` as standing
  regression checks, and `rederive_from_logs.py`, which rebuilds the derived
  JSON from the published raw logs rather than checking it against itself
- the batching arm the harness could not actually run: `BENCH_CONCURRENCY`
  documented concurrent dispatch and issued the prompts one at a time, so
  `--parallel N` allocated slots that stayed idle. How many client requests
  were **outstanding at once** is now read back out of the request timestamps
  and asserted. That is not the server's decode batch width (how many sequences
  shared a decode graph is not something the harness observes) and nothing here
  should be read as measuring it.
- `bench/stage_mtp_source.py`, which makes the target's own multi-token
  prediction head exportable as a drafter without modifying llama.cpp
- ERRATA A11: speculation is not output-preserving on this build, established
  against a determinism control that holds in every run
- the acceptance threshold: **45–48 %**, quoted as a range because half its
  fitted points come from a thinking-off run whose arms generated different
  token counts; and, more importantly, the out-of-sample test showing that the
  threshold transfers and the slope does not
- the checkpoint path, timed in the source rather than inferred from log
  intervals: 39.07 s of a 71.4 s excess, replicated in a second balanced run
- two runs of one configuration that are identical in every recorded respect,
  produce byte-identical output, and differ by 3.4 % on one arm
  ([A16](ERRATA.md#a16-two-runs-identical-in-every-recorded-respect-and-byte-identical-in-output-differ-by-34--on-one-arm))
- an end-to-end harness test with no GPU (`tests/fake_llama_server.py`), which is
  what made the completeness and attestation guards testable at all


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
| translation fallback | 48.4, 49.6 tok/s | 97 / 97 | **abort** |
| matched vocabulary | 51.1, 50.0 tok/s | 97 / 97 | **abort** |
| baseline, measured beside them | 123.3, 126.0 tok/s | — | fine |

The baseline row read `~125–129 tok/s (quiet host)` until 2026-08-29. Run A's
own baseline on that prompt is 123.3 and 126.0, and the translation row showed
one of its two repeats where the matched row showed both.

**Verdict: the gate defect costs 3.2 % on this prompt and 0.2 % overall, not
60 %.** The counted
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
  `--spec-type draft-dflash`**, and EAGLE-3 and MTP, which this repository's
  older text listed as "not evaluated here", are available on one binary.

**Trap worth recording:** on master, `--spec-type` defaults to `none`, so
passing `-md` alone loads the draft model and then never speculates. A first
master run looked like a clean "no crash, no slowdown" result until the server
log showed zero `generate_draft` calls and `draft_n = 0` on all thirty
requests. Any master comparison must pass `--spec-type` explicitly.

### Known failure modes to check for, not assume away

- **Acceptance collapsing to zero under `-np N`.** llama.cpp
  [#27572](https://github.com/ggml-org/llama.cpp/issues/27572) reports draft
  acceptance going to exactly 0.00000 under parallel slots on a hybrid Qwen3.x
  target, with generation falling below no-speculation speed and completions
  coming back with empty `content`. That report is HIP + `draft-mtp` and this
  work is CUDA + `draft-simple`, so it probably does not apply, but the
  batching run must **verify** acceptance is non-zero rather than presume it,
  because the symptom is silent.
- **Checkpoint invalidation on hybrid targets.**
  [#24055](https://github.com/ggml-org/llama.cpp/issues/24055). The audit
  measured 1639 checkpoints at a server-reported 82.079 MiB in one arm-run of ten 300-token requests; whether
  that is the same bug is untested here.

### Still open

> [!NOTE]
> Runs K, L and M were listed here as "still running / next" long after they
> were done and written up. They are in the controlled tier and their results
> are in [`v4_audit_2026_08_25/README.md`](v4_audit_2026_08_25/README.md).
> This section is what is genuinely still open.

- ~~**Run V, rerun with the mode order controlled.**~~ **Done 2026-08-27.**
  `bench/run_v2_crossover.sh` ran eight sessions in `AB BA BA AB BA AB AB BA`
  order, 400 of 400 arm-runs, with the session as the resampling unit;
  `bench/run_v3_within.sh` added a within-invocation square, 200 of 200. Run V
  overstated `spec-dflash-n2` by about 3.3 pp and the other three of its four
  numbers land inside the eight-session intervals
  ([A17](ERRATA.md#a17-the-thinking-off-comparisons-are-not-comparisons-of-the-same-amount-of-work)).
- ~~**A carryover-balanced version of that design.**~~ **Done 2026-08-28.**
  Run W: five sessions of a 10 × 10 Williams square, row order shuffled from a
  per-session seed, 500 of 500 arm-runs, V3's treatment definitions under a
  later harness revision rather than V3 verbatim, differing only in the
  schedule, and run W2: the same square again for twelve sessions and 1200
  arm-runs. Every arm preceded by every other exactly once within a repeat,
  verified from the data in every session of both. The mode effect survives:
  `spec-dflash-n4`'s sign flip holds at +12.03, +12.17, +12.10 and +12.13 pp,
  and `spec-dflash-n2` reads +8.29 pp [+7.97, +8.60] in W and
  +8.50 pp [+8.08, +8.91] in W2, both overlapping V3 and neither overlapping V2
  ([A17](ERRATA.md#a17-the-thinking-off-comparisons-are-not-comparisons-of-the-same-amount-of-work)).
- ~~**Does the predecessor explain `spec-dflash-n2`?**~~ **Answered 2026-08-31
  by run W2, and the answer is no.** W's balanced schedule gave the contrast
  between running after a capped neighbour and after a free-running one:
  −1.20 % [−2.61, +0.22] for that arm, the largest of any by six times and
  pointing the way A17 guessed, with **no arm's interval excluding zero**. This
  entry said that answered it on 2026-08-28. It did not: −2.4 is inside
  [−2.61, +0.22], so W bounded the effect without excluding the size that
  mattered, and the entry was one of three copies of a sentence the pull
  request body had already retracted. W2's twelve sessions close it. The
  pre-registered matched contrast is **−0.14 % [−0.68, +0.41]**, and on the
  quantity the 2.4 pp is actually quoted in, A17's `shift_pp`, the predecessor
  moves it by **+0.49 pp [−0.80, +1.77]** where W's five sessions gave
  +1.52 pp [−0.98, +4.02]. W's interval contains 2.4; W2's does not. That
  removes the one candidate this repository had been able to name for the
  V2/V3 gap.
- **What the gap actually is.** W and V3 agree and the crossover does not, so
  the difference is between measuring the two modes inside one invocation and
  across two. That is [A16](ERRATA.md#a16-two-runs-identical-in-every-recorded-respect-and-byte-identical-in-output-differ-by-34--on-one-arm),
  and it is still unexplained. W reproduces it (1.69 % within-session CV on
  that arm against 0.31 % for no speculation, on work identical to the token)
  on the largest dataset this repository has for it. What would move it next is
  instrumentation, not another schedule: host CPU load (recordable now, in no
  run yet) and the GDDR6X memory-junction temperature, which NVML does not
  expose on Linux at all.
- **Recompute every `request-mean` from `predicted_n` and `predicted_ms`.**
  [B8](ERRATA.md#b8-every-request-mean-here-counts-one-token-fewer-than-it-timed):
  llama.cpp's `predicted_per_second` is a rate over `n − 1` tokens divided by
  the time for `n`, and every request-mean here is the mean of it. On the
  fixed-300-token runs that is a uniform 0.33 %, so no delta and no conclusion
  moves; on the thinking-off runs, where arms stop at different lengths, the
  bias differs per arm. Several dozen published figures would move by 0.33 %
  and each needs re-verifying, which is why it is here rather than done.
- ~~**The checkpoint timers, split.**~~ **Done 2026-08-27.**
  `bench/apply_split_timers.py` drains the queue explicitly before each call
  and times the drain: **0.002 s of 39.09 s**, 0.003 % of the excess, so what
  the 54.7 % measures is post-drain checkpoint state-save/restore API work
  rather than waiting on target work queued earlier
  ([A12](ERRATA.md#a12-what-the-checkpoint-path-costs-measured-with-timers-in-the-source)).
  The residual is not raw copy cost: it still contains serialisation,
  allocation and resize, backend transfer, state traversal and the API's own
  bookkeeping, none of which this run separates.
- **A design that identifies the A16 invocation effect**: randomised order with
  the session as the resampling unit, the stock and instrumented builds
  stratified, and the preceding treatment recorded. The fixed rotation used so
  far balances position but not first-order carryover.
- ~~**The raw evidence archive.**~~ Published 2026-08-27 as
  `raw-evidence-2026-08-27`, with a workflow that re-runs the extractors over
  it. 526 of the 535 counter rows, and all 24 timer and accounting records,
  come back identical; the nine that do not belong to three exploratory runs
  whose arm-run JSON is not committed because they never completed.
- No-speculation baseline is **~6 % faster on master** than on `bcb5eeb64`
  on the same host (133–137 vs 125–129 tok/s), so absolute rates must not be
  compared across those two binaries.
- **`draft-mtp`; nothing was blocking it. It had simply never been tried.**
  This line has now been wrong twice. It first said both `draft-eagle3` and
  `draft-mtp` "need head weights this repository does not have"; that was
  written without checking, and `~/models/qwen36-awq` holds **785 plain BF16
  `mtp.*` tensors** for this exact target with
  `text_config.mtp_num_hidden_layers = 1`. It was then rewritten to say the
  blocker was a converter gap; that `_QwenMtpMixin` (`conversion/qwen.py:277`)
  is inherited by `Qwen3NextModel` (`:372`) but not by the class registered for
  this checkpoint's architecture (`:636`). **That was also wrong**, and it was
  wrong in a way that a patch attempt exposed immediately: adding the mixin
  raised `TypeError: Cannot create a consistent MRO`, because the class already
  has it. The real chain is `Qwen3_5MoeTextModel → _Qwen35MRopeMixin →
  _LinearAttentionVReorderBase → Qwen3NextModel → _QwenMtpMixin → Qwen2MoeModel
  → TextModel`, and `supports_mtp_export` reads `True` on the **stock**
  converter. The runtime side was never in doubt either: `LLM_ARCH_QWEN35MOE`
  declares 17 `NEXTN` tensor entries in `src/llama-arch.cpp`, the same count as
  `LLM_ARCH_QWEN3NEXT`.

  So `--mtp` works with an unmodified llama.cpp, on both the converter and the
  runtime, and the only reason this arm was missing is that nobody ran it. The
  llama.cpp working tree was restored to stock after the failed patch; no
  modification of that repository is involved in any measurement here.

- **Coverage of `--spec-type` is now complete for what this model can run.**
  Master exposes eleven values. Nine are measured here: `none`, `draft-simple`,
  `draft-dflash`, `draft-mtp`, `ngram-simple`, `ngram-mod`, `ngram-cache`,
  `ngram-map-k`, `ngram-map-k4v`. The other two are blocked for reasons that
  were checked rather than assumed:

  | type | why it cannot run here |
  |---|---|
  | `draft-eagle3` | the loader rejects any drafter that is not an EAGLE3 model — "expected 3 extract layers", `common/speculative.cpp:471` — and no EAGLE3 head for this target exists on any of these hosts |
  | `draft-dspark` | `convert_hf_to_gguf.py` refuses: "`--dspark` is only supported for `DeepseekV4ForCausalLM`". The DFlash checkpoint here is `DFlashDraftModel` and carries no DSpark config or tensors. This is a model-family limit, not a missing download. |
- **MTP matters more than the other open items** because the vLLM sibling result
  on this same physical hardware is an MTP result. Until it runs here, "llama.cpp
  loses where vLLM wins" confounds the engine with the speculation method.

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

The working recipe: no file edit needed, and a no-op for the target:

```bash
./build/bin/llama-server -m ~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    -md ~/models/Qwen3.5-0.8B-Q4_K_M.gguf -ngl 999 -c 4096 -v \
    --override-kv tokenizer.ggml.bos_token_id=int:248044 2>&1 \
  | grep -E "Using metadata override|vocab_cmpt|not compatible"
# vocab_cmpt = 1, and the "not compatible" line is gone
```

Then run the A/B: same target, same draft weights, same binary, same flags,
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

### P0-2 · Thinking control that actually works — **DONE**

The switch that works on this stack is the request-level
`chat_template_kwargs: {"enable_thinking": false}` on `/v1/chat/completions`,
and `bench/retest_runner.py` records `thinking_suppressed` per request from the
length of the reasoning channel rather than assuming it. Measured, per arm:
50 of 50 requests suppressed with it and 0 of 50 without, which is 250 of 250
over run D against 0 of 650 over run C, and completions then stop naturally at
22–300 tokens instead of every one hitting the cap. Original text below.


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

### P0-3 · Capture the *true* acceptance rate for every config — **DONE**

Superseded by the matrix: post-merge master fixed the counter, so every arm
reports honest ratios directly, from 1.4 % to 68.7 % depending on method and
draft length. Original text below.


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
[`bench/retest_runner.py`](bench/retest_runner.py): **never executed**; review
before trusting it.

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

### P1-2 · The missing fp16-KV no-speculation baseline — **DONE**

Run in the 13-arm matrix on post-merge master, 5 repeats: `baseline-kvfp16`
reaches 125.7 pooled tok/s against `baseline`'s 123.4 (**+1.9 %**), while
`ngram-cache-kvfp16` reaches 70.9 against `ngram-cache`'s 74.0 (**−4.2 %**).
fp16 KV is faster than q8_0 with no speculation running and does not help when
speculation is on. **Closes** [B7](ERRATA.md#b7-the-fp16-kv-row-is-a-one-sided-control--now-closed-by-measurement).

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
  is the fingerprint. Measured 350 / 350 / 350 W: stock.
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
`b8942-67cb0d507` (the pre-merge PR branch) for DFlash: two different binaries ([D4](ERRATA.md#d4-v3-dflash-compares-two-different-binaries)).

```bash
# on 3090; upstream master is c1d0e7a00 as of this probe
cd ~/bench/llama.cpp && git fetch origin && git checkout <pinned-post-merge-sha>
cmake -B build-retest -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DCMAKE_BUILD_TYPE=Release -DGGML_CCACHE=ON -GNinja
cmake --build build-retest -j
sha256sum build-retest/bin/llama-server   # goes in the manifest
```

Pin an explicit SHA; do not track a moving master. **Cost** ~30–45 min CPU with
ccache warm.

### P2-2 · DFlash off vs on, one binary — **DONE 2026-08-26, and it reverses the sign**

Run J: one binary, one placement policy, three repeats per arm. The archived
drafter did need re-conversion: post-merge master rejects it for lacking
`target_layers`. No BOS problem: DFlash reuses the target's vocabulary, so the
special-token gate that broke P0-1 is not in play.

| arm | aggregate | vs no speculation |
|---|---|---|
| `spec-dflash-n4` | 130.2 | **+18.7 %** |
| no speculation | 109.7 | — |
| `spec-dflash-n8` | 93.5 | −14.8 % |
| `spec-dflash-n16` | 57.7 | −47.4 % |

Three repeats rather than five, and the control is the part that took the work:
`-fit on` is required for the BF16 drafter to load, so it was applied to every
arm including the baseline, and that baseline lands within −0.01 % of the
pinned one at identical 41/41 placement. D4 is closed and v3's conclusion is
retracted; what v3 measured was a binary change, and the method it blamed is
the fastest thing in this repository at a short draft window.

Still open from this: the configuration is marginal on a 24 GiB card (peaks at
23946 MiB of 24576), five repeats were not run, and the draft-length optimum is
being bracketed in run K.

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
every partial accept ([A1](ERRATA.md#a1-100--draft-acceptance-is-a-counter-artefact-not-a-measurement), [A3](ERRATA.md#a3-the-tested-build-had-a-known-broken-speculative-path-for-this-model-class-and-the-fix-was-never-merged)).
PR #20075, which fixed the underlying hybrid-SSM rollback, was **closed without
merge**. Check current master's `server-context.cpp` and re-run
`common_context_can_seq_rm` against this target on the post-merge binary. If it
still returns `FULL`, that is a reportable upstream finding in its own right,
and a defensible reason to open an issue with this repository's data attached.
**Cost** ~20 min.

**DONE 2026-08-26; the path still exists; the counter no longer depends on
it.** At `3737e4137` the partial-accept branch still returns early
(`server-context.cpp:3835`) and the checkpoint restore still happens: a hybrid
Gated-DeltaNet/MoE target still cannot roll back part of a sequence. What moved
is the denominator, which is what made the ratio a tautology. `n_draft_tokens
+= draft.size()` now runs at `:2939`, when the draft is *produced*, and a
replaying slot never reaches it because `drafting.push_back(&slot)` at `:2921`
sits inside the `else` of `if (!slot.spec_draft.empty())` at `:2893`. The
numerator subtracts one on a replay (`:3851`). So master reports honest ratios
(29.7 %, 55.8 %) where `97895129e` could only report 1.00000. Full derivation
in ERRATA A1.

### P3-3 · Expert-routing instrumentation (optional, expensive)

Nothing here has ever measured expert routing. Testing the MoESD story needs a
patched build counting unique experts activated per layer per verification
step, plus HBM traffic and kernel time. Only worth doing after P3-1 shows the
draft-path terms do **not** account for the slowdown. **Cost** days.

### P3-4 · Controls this repository still lacks

Dense-model control, FP16/BF16 target control, and a second-GPU control are all
absent. The 3090's 24 GiB cannot hold a BF16 35B target, so the FP16 control
needs different hardware or a smaller target: note it as out of scope rather
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
