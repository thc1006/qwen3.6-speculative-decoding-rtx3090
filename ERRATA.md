# ERRATA — audit of 2026-08-25

This file lists every claim in this repository that the 2026-08-25 audit found
to be wrong, unsupported, or materially misleading, together with the evidence
that settles it. Nothing in `results/`, `v2_3090_followup/`, or
`v3_dflash_2026_05_07/data/` was edited: the raw measurements stand, and every
item below is derived from files that were already committed. What changed is
the wording, the statistics, and the causal claims built on top of them.

Reproduce the numeric items with:

```bash
python analysis/plot.py                 # request-mean, pooled, median, activation
python analysis/verbose_accounting.py   # the acceptance-counter reconstruction
```

Severity: **A** changes the headline conclusion · **B** changes the reported
statistics · **C** is a naming or scope defect · **D** invalidates a follow-up
experiment's stated treatment · **E** is a theory error · **F** is metadata.

---

## A — findings that change the headline conclusion

### A1. "100 % draft acceptance" is a counter artefact, not a measurement

**Claimed.** v1 README: "every tested config returning **100 % acceptance** …
the classical intuition 'high acceptance → high speedup' fails here. This is
not a measurement artifact; it is MoE expert-loading overhead on every drafted
token." v2 SUMMARY: "100 % `n_acc_tokens / n_gen_tokens` is genuine. Verified
by source read of `common/speculative.cpp` … and a `--verbose` run emitting
`draft acceptance rate = 1.00000 (115 accepted / 115 generated)`."

**Actual.** The ratio is 1.0 by construction for this model on this build. In
`tools/server/server-context.cpp` at commit `97895129e`, the per-request
counters are updated *after* an early `continue`:

```cpp
// check for partial draft acceptance
if (accepted.size() < slot.spec_draft.size() + 1) {
    if (slot.ctx_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
        slot.spec_draft = std::move(accepted);      // truncate to the accepted prefix
        ... llama_state_seq_set_data_ext(...) ...   // restore the checkpoint
        continue;                                   // <-- counters never reached
    }
}
common_speculative_accept(slot.spec.get(), accepted.size() - 1);
...
slot.n_draft_accepted += ids.size() - 1;            // only full accepts land here
slot.n_draft_total    += n_draft;
```

Qwen3.6-35B-A3B is a hybrid Gated-DeltaNet / MoE model, so
`common_context_can_seq_rm()` returns `COMMON_CONTEXT_SEQ_RM_TYPE_FULL`. The
committed log records exactly that:

```
common_context_can_seq_rm: the target context does not support partial sequence removal
```
`v2_3090_followup/v2_oleg_suggestions/verbose.log:2758`

Every partially accepted round therefore takes the `continue`, is dropped from
both numerator and denominator, and is re-verified next pass against the
truncated — already known-accepted — prefix. Only rounds accepted in full ever
reach the counter, so `draft_n_accepted / draft_n` can only be 1.0.

**The same log contradicts the headline on the very next line.** The drafter
keeps its own counters:

```
draft acceptance rate = 1.00000 (  115 accepted /   115 generated)
statistics draft: #calls(b,g,a) = 1 82 33, #gen drafts = 81, #acc drafts = 33,
                  #gen tokens = 214, #acc tokens = 115, dur(b,g,a) = 0.001, 999.634, 0.005 ms
```
`verbose.log:8841-8842`

| counter | value | meaning |
|---|---:|---|
| server `draft_n_accepted / draft_n` | **115 / 115 = 100.0 %** | tokens re-verified after truncation to the accepted prefix |
| drafter `#acc tokens / #gen tokens` | **115 / 214 = 53.7 %** | true token-level acceptance |
| drafter `#acc drafts / #gen drafts` | **33 / 81 = 40.7 %** | true draft-sequence acceptance |

Round-level accounting over the same run, every quantity cross-checked against
an independent path in the log:

| quantity | value | independent check |
|---|---:|---|
| drafts generated | 81 (214 tokens) | `called impl` log lines = `#gen drafts`; sum of `gen=` = `#gen tokens` |
| dropped by `--draft-min 2` before any verification | 49 drafts (48 tokens) | 1 empty + 48 of size 1; 82 generate calls - 49 = 33 |
| reached verification as a fresh draft | 33 drafts (166 tokens) | 214 - 48 = 166 |
| verification attempts | 53 | 33 fresh + 20 re-verifies |
| partially accepted -> discarded and redone | 20 | equals the checkpoint-restore count |
| longest verify chain | 2 | no draft needed more than one re-verification |
| server counter | 115 / 115 | sum of `n_draft` over the 33 fully accepted rounds |

Three separate derivations close on the same numbers, so the reconstruction is
not an inference: 130 (partial proposals) + 36 (first-try-full) = 166 tokens
reaching verification, and 79 (re-verified) + 36 = 115 reaching the counter.

**Consequence.** `draft_n` in `analysis/summary.csv` is not "draft tokens
generated". It is "draft tokens in verification rounds that were accepted in
full", a quantity guaranteed to equal `draft_n_accepted`. A row with
`draft_n = 0` means "no fully accepted round was recorded", **not**
"speculation did not run". Every "100 %" in the v1/v2/v3 write-ups has been
removed or relabelled, and `analysis/plot_accept_vs_speed.png` — whose entire
x-axis was that artefact, with all 140 points at exactly 100 % — has been
deleted and replaced by `analysis/plot_acceptance_accounting.png`.

**Upstream fixed this, and the audit's post-merge acceptance figures rest on
the fix.** Every acceptance percentage this audit reports from master — 29.7 %
for `spec-draft-n8`, 55.8 % for `spec-dflash-n4` — would be meaningless if the
same tautology were still in place, so the counter path was read at the tested
commit `3737e4137` rather than assumed to have changed:

| | `97895129e` (v1) | `3737e4137` (audit) |
|---|---|---|
| denominator | after the early `continue` | `server-context.cpp:2939`, at draft **generation** |
| numerator | after the early `continue` | `server-context.cpp:3859`, with a replay correction |
| replayed tokens | re-entered both counters | excluded from the denominator |

The partial-accept branch still returns early (`:3835`) — a hybrid target still
cannot roll back part of a sequence — but the denominator no longer lives behind
it. `slot.stats.n_draft_tokens += draft.size()` runs when the draft is
*produced*, and a slot replaying a truncated draft never reaches that line,
because `drafting.push_back(&slot)` at `:2921` sits inside the `else` of
`if (!slot.spec_draft.empty())` at `:2893`. The numerator then subtracts one on
a replay (`:3851`) so the token carried over from the truncated draft is not
counted twice.

Worked through: a round drafts 8 tokens and 3 are accepted. The denominator
takes all 8 at generation; the partial branch truncates and returns without
touching the numerator; the replay round re-verifies the 4-token remainder,
reaches the numerator with `n_accepted = 4`, decrements to 3 for the replay, and
the ratio is 3/8. That is the honest quantity, and it is why master reports
values like 29.7 % where `97895129e` could only ever report 1.00000.

---

### A2. The draft model was **not** vocabulary-compatible; the run used the token-translation fallback

**Claimed.** Everywhere: "classic draft with the **vocab-matched**
`Qwen3.5-0.8B` (vocab 248320)", "correct-vocab classic SD".

**Actual.** The vocabulary *sizes* match — `Qwen/Qwen3.6-35B-A3B`
`text_config.vocab_size = 248320` and `Qwen/Qwen3.5-0.8B`
`text_config.vocab_size = 248320`, both verified against the model cards. But
llama.cpp's compatibility test is stricter, and it fails:

```
common_speculative_are_compatible: vocab_type tgt: 1
common_speculative_are_compatible: vocab_type dft: 1
common_speculative_are_compatible: draft model special tokens must match target model to use speculation
vocab_cmpt = 0
the target and draft vocabs are not compatible - tokens will be translated between the two
```
`v2_3090_followup/v2_oleg_suggestions/verbose.log:2759-2763`

The check that fails is the special-token gate in
`common/speculative.cpp:69-77` (`add_bos` / `add_eos` / `bos` / `eos`), which
runs *before* the size and per-token-text checks. llama.cpp then enables the
translation path, visible per drafting round as

```
draft: main->draft detokenized string: '<|im_start|>user …'
```

so every speculation round detokenises the context to a string and re-tokenises
it for the draft model.

**Root cause, established 2026-08-25.** The two tokenizers are identical: same
`tokenizer.ggml.model = gpt2`, same `pre = qwen35`, same 248320 tokens, same
247587 merges, same `eos_token_id = 248046`, same `padding_token_id = 248055`.
Exactly one key differs, and it is one the draft GGUF does not have at all:

| | target `UD-Q4_K_XL` | draft `0.8B-Q4_K_M` |
|---|---|---|
| `tokenizer.ggml.bos_token_id` | `248044` | **absent** |
| `tokenizer.ggml.add_bos_token` | `false` | absent (default `false`) |
| BOS as resolved by llama.cpp | `248044 '<\|endoftext\|>'` | **`11 ','`** |

`Qwen/Qwen3.5-0.8B` has **no `generation_config.json` upstream** — HTTP 404 on
both the Qwen and unsloth repos — so `convert_hf_to_gguf.py` had no
`bos_token_id` to write. `src/llama-vocab.cpp:1838` then supplies the
hard-coded GPT-2 legacy default `special_bos_id = 11`, and the KV loop at line
2313 never overrides it. Meanwhile **both** models declare `bos_token = None`
and `add_bos_token = false` in their upstream `tokenizer_config.json`, so
neither ever prepends a BOS token. The field that gates speculation is one that
neither model uses when tokenising.

Of the gate's four conditions, exactly one fails:

| condition | target | draft | |
|---|---|---|---|
| `llama_vocab_get_add_bos` | false | false (default) | pass |
| `llama_vocab_get_add_eos` | false (default) | false (default) | pass |
| `llama_vocab_bos` | 248044 | **11** | **fail** |
| `llama_vocab_eos` | 248046 | 248046 | pass |

Adding `--override-kv tokenizer.ggml.bos_token_id=int:248044` at load time — no
file edit, and a no-op for the target, which already carries that value — flips
`vocab_cmpt` from `0` to `1` and removes the warning. That also proves the two
token arrays are byte-identical, because the gate's per-token text comparison
from id 5 to 248320 runs only once the special-token check passes.

**Does it explain the slowdown? Tested, and no.** A same-binary, same-file A/B
on 2026-08-25 (`llama-server` @ `bcb5eeb64`, `--draft-max 8 --draft-min 4`,
greedy, the ten v1 prompts, ABBA-ordered, three arms interleaved):

| binary | arm | request-mean | drafted / accepted |
|---|---|---:|---|
| `bcb5eeb64` | translation fallback, as published | 113.9 | 194 / 194 |
| `bcb5eeb64` | matched via `--override-kv` | 113.5 | 194 / 194 |
| master `3737e4137` | translation fallback | 33.6 | 16590 / 4926 |
| master `3737e4137` | matched via `--override-kv` | 33.7 | 16590 / 4926 |

The drafted and accepted totals are **byte-identical** across arms on both
binaries, per prompt as well as in aggregate (`154/576`, `140/647`,
`211/404`, …). Throughput differs by +0.3 % overall and by −1.2 % to +3.7 %
per prompt — noise. The translation path was not changing what got drafted.
Full data: [`v4_audit_2026_08_25/`](v4_audit_2026_08_25/).

**Consequence.** The repository's "vocab-matched" and "correct-vocab classic
SD" claims were false and have been removed — vocabulary *size* equality was
treated as proof of compatibility, and it is not. But the negative finding
survives the fix, and is now measured on a genuinely matched path, which makes
it stronger rather than weaker.

---

### A3. The tested build had a known-broken speculative path for this model class, and the fix was never merged

The `COMMON_CONTEXT_SEQ_RM_TYPE_FULL` fallback in A1 is not incidental. The
same string — "the target context does not support partial sequence removal" —
is the symptom llama.cpp PR #20075 was opened to fix. From its body:

> Speculative decoding on hybrid SSM/MoE models is broken right now. With a
> draft model you either crash immediately ("the target context does not
> support partial sequence removal") or end up with garbage loops. … On top of
> that, `llama_memory_recurrent` has no rollback mechanism at all for the SSM
> state when draft tokens get rejected, which is what causes the state drift.

PR #20075 was **closed without merge on 2026-04-25**. The README described it
as a peripheral "may change these numbers" note and, at v3, still labelled it
`OPEN`. It is in fact the most relevant upstream item to this benchmark: the
measured configuration is the pre-fix state of a path its author considered
broken for exactly this model family.

---

### A4. A measured cost decomposition was in the repository the whole time and was never used

`verbose.log` carries a per-run breakdown that bounds the draft-side cost
directly, without any appeal to MoE expert-union theory. For the
`--draft-min 2 --draft-max 32` run on prompt 1, 200 generated tokens at the
reported 63.2 tok/s (≈ 3165 ms of generation):

| term | value | source |
|---|---:|---|
| drafter `generate()` time | **999.6 ms ≈ 31.6 % of generation wall-clock** | `dur(b,g,a)` field |
| speculative checkpoints created | 33 × 62.8 MiB = **2.02 GiB written** | `created speculative checkpoint` |
| checkpoints restored after a partial accept | 20 = **1.23 GiB read back** | `restoring speculative checkpoint` |
| verification rounds thrown away and redone | **20 of 53 (37.7 %)** | round accounting |

Roughly a third of the generation wall-clock is the draft model alone, and
about 38 % of verification rounds are paid for twice. These terms account for
the observed slowdown without invoking expert-union loading. They do not rule
that mechanism out — nothing here measures expert routing — but the repository
previously presented the expert-union story as the explanation while this
decomposition sat unread in its own committed log.

---

### A5. Three quarters of v1's requests returned no answer at all — only truncated thinking

`bench_runner.py` stored `content[:120]` and nothing else. Re-reading that field
during the audit shows it is **empty for 144 of the 190 v1 requests (75.8 %)**:

| prompt | requests with empty `message.content` |
|---|---|
| `reasoning` | **19 / 19** |
| `code_small` | **19 / 19** |
| `short_q`, `medium_chat`, `medium_rec`, `long_explain`, `multi_turn_1`, `multi_turn_2`, `zh_cn` | 15 / 19 each |
| `short_greet` | 1 / 19 |

With `--jinja` and a reasoning model, llama-server routes `<think>` content to
`message.reasoning_content` and the answer to `message.content`. An empty
`content` with `predicted_n == 300` means the cap was reached **inside the
thinking block**. v1 never captured `reasoning_content`, so this was invisible.

The two prompts carrying the entire published slow tail — `code_small` and
`reasoning` — are 19/19 thinking-only. v1 therefore measured decode throughput
on truncated chain-of-thought traces, not on answers, and the "structured
prompts collapse" narrative is really "the drafter matched inside a long
reasoning trace". v1 never attempted to disable thinking; v2, v3 and Exp 2
attempted it and failed (D1/D2). The defect is common to all four experiments.

### A7. With acceptance measured properly, there is no anomaly left to explain

This repository's central claim was an *anomaly*: 100 % acceptance yet slower,
therefore something MoE-specific must be destroying the speedup. Upstream has
since made the partial-accept path reachable, so the counter now reports real
ratios, and the anomaly disappears.

**The contrast matters, so name it.** Two different comparisons are available
and they point in opposite directions. Reporting either without saying which
one it is would repeat the mistake this audit exists to correct.

**Contrast 1 — within one configuration, across prompts.** Prompts the drafter
predicts well run faster, almost exactly in proportion. On post-merge master
`3737e4137`, five repeats of a thirteen-arm matrix, every draft-model
configuration reproduces this independently:

| configuration | Pearson r | acceptance range across prompts |
|---|---:|---|
| `--spec-draft-n-max 1` | **+0.998** | 55.7 – 83.2 % |
| `--spec-draft-n-max 2` | **+0.999** | 49.8 – 77.9 % |
| `--spec-draft-n-max 4` | **+0.996** | 34.4 – 65.8 % |
| `--spec-draft-n-max 8` | **+0.999** | 20.4 – 52.2 % |
| `--spec-draft-n-max 16` | **+0.999** | 9.8 – 32.3 % |
| `--spec-draft-n-max 32` | **+0.999** | 5.2 – 15.7 % |
| v1's configuration (max 8, min 4) | **+0.999** | 20.4 – 52.2 % |

**Six distinct draft lengths**, spanning acceptance from 5 % to 83 %, all at
r ≥ +0.996. The seventh row is not an independent configuration and should not
be counted as one: v1's setting differs from `n_max 8` only in `n_min`, and at
this draft length that changes almost nothing — 32.27 against 32.10 pooled
tok/s, 29.69 % against 29.67 % acceptance, and three of ten prompts drafting a
byte-identical number of tokens. What it does establish is that `n_min` is not
the knob that matters here.

A control makes the correlation harder to explain away: with no speculation the
prompt barely affects speed at all. `baseline` spans **122.1 – 123.8 tok/s
across the ten prompts, a 1.4 % total spread**, so the large per-prompt
variation inside every speculative arm is driven by speculation rather than by
some prompts being intrinsically faster to decode.

**Contrast 2 — across configurations.** Here the sign flips:
r = **−0.544** over the eleven speculative arms. That is not a contradiction,
it is a different question. A configuration that drafts aggressively achieves
*higher* acceptance per attempt while paying for far more drafted tokens, so
across methods the total cost dominates. Draft volume against speed gives
r = −0.603, and the ordering makes the mechanism plain:

| arm | draft tokens per generated token | pooled tok/s | acceptance |
|---|---:|---:|---:|
| ngram-simple | 0.06 | 118.1 | 4.3 % |
| ngram-mod n=24 | 0.21 | 115.0 | 4.8 % |
| ngram-cache | 0.42 | 74.0 | 1.8 % |
| draft model, n_max 1 | 0.50 | 31.1 | 68.7 % |
| draft model, n_max 8 | 1.85 | 32.1 | 29.7 % |
| draft model, n_max 32 | 6.84 | 17.3 | 8.0 % |

Note the fourth row. An external draft model proposing **0.50** tokens per
generated token runs at 31.1 tok/s, while ngram-cache proposing **0.42** runs
at 74.0. Volume alone does not explain that, so there is a third term: the
**per-round cost of the method itself**. An external 0.8 B drafter needs a
forward pass every round; an ngram lookup is nearly free.

**The honest model therefore has three terms**, not one: the fixed per-round
cost of the drafting method, the volume of tokens drafted, and the fraction of
that work accepted. Within a fixed method and draft length, acceptance predicts
speed almost perfectly. Across methods it does not, because the other two terms
change.

**The draft-length sweep is not monotone.** Sweeping `--spec-draft-n-max` with
`n_min` pinned to 1 and matched vocabulary:

| n_max | pooled tok/s | vs baseline | drafted | acceptance |
|---:|---:|---:|---:|---:|
| 1 | 31.1 | −74.8 % | 7 450 | 68.7 % |
| 2 | 34.2 | −72.2 % | 11 070 | 60.3 % |
| **4** | **35.6** | **−71.1 %** | 16 855 | 45.4 % |
| 8 | 32.1 | −74.0 % | 27 735 | 29.7 % |
| 16 | 23.7 | −80.8 % | 53 740 | 15.0 % |
| 32 | 17.3 | −86.0 % | 102 575 | 8.0 % |

There is an optimum, at n_max = 4 — and it is still 71 % below the
no-speculation baseline. The peak is real but shallow: per-repeat SD is 0.055
for n_max 2 and 0.076 for n_max 4, against a 1.4 tok/s gap between them.

**The sweep was extended past the threshold, so this is now answered by
measurement.** Run E adds `n_max` 64, 96 and 128 on the same pinned binary,
three repeats each, run-to-run SD 0.03–0.17 tok/s. The sweep now spans
**3.1 % to 98.3 % expected routed-expert coverage** and crosses the 95-token
point MoESD's argument is about.

| `n_max` | coverage | acceptance | draft/gen | pooled tok/s |
|---:|---:|---:|---:|---:|
| 32 | 63.8 % | 8.0 % | 6.84 | 17.3 |
| 64 | 86.9 % | 4.3 % | 12.74 | 12.4 |
| **96** | **95.3 %** | 3.1 % | 17.80 | 10.0 |
| 128 | 98.3 % | 2.5 % | 22.04 | 8.9 |

Throughput declines monotonically straight through the threshold. More usefully,
one regressor accounts for the whole sweep:

```
ms per generated token = 27.00 + 4.040 × (draft tokens per generated token)
R² = 0.99303,  n_max = 1 … 128
```

No expert term, 99.3 % of the variance, and the slope reads sensibly — 4.04 ms
per speculated position against a measured 7.87 ms no-speculation decode step,
about half a target step per drafted token, which is what an autoregressive
0.8 B drafter plus its share of the verify pass should cost.

> [!IMPORTANT]
> **Read this law inside its scope.** Every arm of the sweep ran at
> `p_min = 0`, post-merge master's default, which means the drafter emitted the
> full `n_max` every round regardless of confidence. Every archived number in
> this repository ran at `p_min = 0.75` instead ([A8](#a8-the-audits-own-matrix-has-an-uncontrolled-difference-from-the-archive-p_min)).
> The law is fitted, and holds, only in the first regime. Tested out of sample
> in the second it fails — see [A10](#a10-the-single-regressor-law-is-falsified-out-of-sample-and-p_min-is-the-lever-that-matters).

**The step in the residuals at the 95.3 % coverage point is −0.39 percentage
points**: −0.27 % mean below it, −0.67 % at or above. There is no knee, no
break, and nothing left for a coverage threshold to explain on this hardware.

Two tempting wrong answers were eliminated getting here, and both are recorded
in [`v4_audit_2026_08_25/PREREGISTERED_PREDICTION.md`](v4_audit_2026_08_25/PREREGISTERED_PREDICTION.md):
that per-drafted-token cost amortises past the threshold (an artefact of fitting
only the sub-threshold points, which are dominated by `n_max` 1–4), and that
speculative state checkpointing dominates (refuted by its own test — checkpoint
traffic per generated token *falls* as cost *rises*, correlation −0.52).

The model's predictions for 64/96/128 were registered in git **before** those
measurements existed. They held to −7.5 %, −5.7 % and 0.0 % — though the
agreement is partly two errors cancelling, which that file states rather than
claims as a clean win.

**Conclusion. No MoE-specific pathology is needed to explain any of this**, and
this repository never had evidence for one. The slowdown is the per-round cost
of drafting, multiplied by how much is drafted, divided by how much is
accepted.

Aggregate for the original three-arm run, 3 repeats: baseline request-mean
132.9 / pooled 133.3; with a draft model 33.7 / 32.6 and **4926 of 16590 draft
tokens accepted (29.7 %)**. The counter reporting 29.7 % rather than 1.00000 is
independent upstream confirmation of A1. Data:
[`v4_audit_2026_08_25/`](v4_audit_2026_08_25/).

### A8. The audit's own matrix has an uncontrolled difference from the archive: `p_min`

Found by reading upstream rather than by reasoning, which is the point of
checking.

`--spec-draft-p-min` truncates a draft as soon as the drafter's confidence for
the next position falls below it. **The default changed between the binaries
this repository has used:**

| build | `p_min` default | what ran on it |
|---|---:|---|
| `9789512` | 0.75 | the entire v1 matrix |
| `bcb5eeb64` | 0.75 | v2, Exp 2, v3, and audit run A |
| master `3737e4137` | **0.00** | **the whole audit matrix: runs B, C, D, E** |

So every archived number was measured with draft truncation **on**, and every
number the audit's own matrix produced was measured with it **off**, by default,
without either being stated. That is exactly the class of silent
configuration difference this file exists to catch, and it is in my own work.

It matters most at long draft lengths. With `p_min = 0`, the drafter emits the
full `n_max` every round regardless of confidence, which is why draft volume
reaches 22 tokens per generated token at `n_max` 128 and acceptance falls to
2.5 %. With `p_min = 0.75` the draft stops early when the drafter is unsure, so
volume and acceptance both behave differently.

Consequences:

- **Cross-binary absolute comparisons are confounded** by this on top of the
  binary change already noted. Within-matrix arm contrasts are unaffected —
  every arm in a run shares the same `p_min`.
- **The draft-length sweep in A7 measures "always draft `n_max`", not "draft
  until unsure".** The single-regressor law and the absence of a coverage-
  threshold step both stand, because they are within-matrix, but the sweep
  should not be read as the best speculative decoding can do.
- A `p_min` sweep is queued to measure the difference rather than argue about
  it: `n_max` 8 at `p_min` ∈ {0, 0.5, 0.75, 0.9}, plus `n_max` 32 and 128 at
  0.75.

### A10. The single-regressor law is falsified out of sample, and `p_min` is the lever that matters

A7's law was fitted entirely at `p_min = 0`. A8 flagged that as a confound. The
sweep that measures it is now done — 7 arms, 3 repeats, run-to-run SD
0.13–0.39 tok/s — and it does two things at once: it **falsifies the law** and
it produces the cleanest version of this repository's original claim.

| arm | draft/gen | real acceptance | pooled tok/s | vs baseline | law residual |
|---|---:|---:|---:|---:|---:|
| baseline | — | — | 123.8 | — | — |
| `n_max` 8, **`p_min` 0.75** | 0.61 | **80.2 %** | **42.8** | **−65.5 %** | −20.7 % |
| `n_max` 8, `p_min` 0.90 | 0.46 | **88.2 %** | 42.5 | −65.6 % | −18.5 % |
| `n_max` 128, `p_min` 0.75 | 0.68 | 70.9 % | 42.0 | −66.1 % | −20.0 % |
| `n_max` 32, `p_min` 0.75 | 0.68 | 70.9 % | 42.0 | −66.1 % | −19.9 % |
| `n_max` 8, `p_min` 0.50 | 0.94 | 58.8 % | 39.6 | −68.0 % | −17.9 % |
| `n_max` 8, `p_min` 0 (the whole audit matrix) | 1.85 | 29.7 % | 32.7 | −73.6 % | −11.3 % |

**The law fails.** `ms/tok = 27.00 + 4.040 × draft/gen`, fitted at `p_min = 0`,
over-predicts cost by a mean of **19.4 %** on every arm with `p_min > 0`, all in
the same direction. A7's law holds in the regime it was fitted in and nowhere
else, which is now stated there.

The cleanest demonstration needs no regression at all. Two configurations draft
almost exactly the same number of tokens per generated token and cost wildly
different amounts:

| configuration | draft/gen | rounds/gen | ms/token |
|---|---:|---:|---:|
| `n_max` 1, `p_min` 0 | 0.50 | 0.55 | 32.09 |
| `n_max` 8, `p_min` 0.90 | 0.46 | **0.19** | **23.62** |

Volume differs by 9 %, cost by **36 %**, and rounds by **186 %**. Whatever is
driving the cost, it is not the number of drafted tokens. `n_max` 1 pays a
drafter forward pass for every single token it proposes; `p_min` 0.90 proposes
several per round and stops early when unsure.

**And this partly rehabilitates the term A7 discarded.** The two-term model —
rounds and volume — was demoted because rounds earned only 0.096 percentage
points of R² on the `p_min = 0` sweep. That sweep could not separate them:
rounds per generated token sat at 0.25–0.26 across almost its whole range. The
`p_min` sweep varies rounds independently, and refitting across both families,
14 configurations spanning `p_min` 0–0.90 and `n_max` 1–128:

| model | bias on `p_min > 0` arms | bias on `p_min = 0` arms | separation |
|---|---:|---:|---:|
| volume only | −10.1 % | +4.6 % | **14.7 pp** |
| rounds + volume | −4.5 % | +1.2 % | **5.7 pp** |

R² is useless for this — volume alone still reaches 0.988 because `ms/token`
spans 23 to 112 and a 36 % error at the low end barely registers. The family
separation is the metric that matters: a model that captured the physics would
show none. Adding rounds cuts it by 61 % and does not remove it, so **neither
model is right**, and the coefficients of either should be read as descriptive
rather than physical. Both fits use the rep-0 server logs, which are where the
checkpoint counts that stand in for round counts live.

**`p_min` is the dominant knob, not `n_max`.** Going from 0 to 0.75 at fixed
`n_max` 8 halves draft volume, raises real acceptance from 29.7 % to 80.2 %, and
lifts throughput by **31 %**. Meanwhile `n_max` 32 and `n_max` 128 at
`p_min = 0.75` are not merely similar — they are **identical**, 6159 drafted
tokens each, 70.9 % acceptance each, 42.0 tok/s each. Above the confidence
threshold, `n_max` is inert.

That last point retires a question rather than answering it. On the default and
historical configuration the MoESD coverage threshold is not just un-crossed,
it is **unreachable**: the drafter stops on confidence long before a draft
approaches 95 tokens. The threshold-crossing sweep in A7 could only exist
because `p_min = 0` disabled the mechanism that would otherwise prevent it.

**And this is where the repository's original claim finally gets honest
evidence.** The published version was "100 % acceptance yet slower, therefore an
MoE pathology". The 100 % was a counter artefact (A1). But at `p_min = 0.90`
the acceptance rate is a genuine, correctly counted **88.2 %** — and throughput
is still **−65.6 %**. So the *intuition* was right all along: high acceptance
does not imply speedup here. It was the evidence that was wrong, and the
conclusion that was overreached. Nine drafted tokens in ten are accepted and
the configuration is still nearly three times slower than not speculating.

Best speculative configuration measured anywhere in this audit: `n_max` 8 with
`p_min` 0.75, at 42.8 tok/s against a 123.8 baseline. **−65.5 %.**

### A9. Upstream already documents two of the mechanisms this audit "found"

Credit where it belongs. Reading llama.cpp PR #22105's description turns up
both of the mechanisms this audit arrived at independently, stated by the
implementer:

> For MoE targets, DFlash speedup is generally smaller than for dense attention
> targets because **more experts get activated during the parallel verification
> step than during single-token autoregressive decoding** (same observation as
> in #18039 for gpt-oss EAGLE3).

That is the expert-union effect this repository's original write-up asserted,
and the implementer cross-references a second PR observing it for EAGLE3.

**The cross-check behind that cross-reference is the strongest external evidence
this repository has ever had, and it was never cited.** In
[PR #18039](https://github.com/ggml-org/llama.cpp/pull/18039#issuecomment-3755925892)
the same maintainer reran the comparison on **SGLang** — a different engine —
on **DGX Spark** with **gpt-oss-120b + EAGLE3**, to check whether llama.cpp was
at fault:

| prompt | baseline | EAGLE3, draft 8 | speedup | EAGLE3, draft 3 | speedup |
|---|---:|---:|---:|---:|---:|
| quicksort in Python | 52.50 t/s | 36.4 | **0.69×** | 37.36 | 0.71× |
| explain the Pythagorean theorem | 52.64 t/s | 24.4 | **0.46×** | 27.96 | 0.53× |
| plan a one-day trip to DC | 52.69 t/s | 24.7 | **0.47×** | 26.53 | 0.50× |

Different engine, different hardware, different model, different speculative
method — and speculative decoding is still a **29–54 % net loss** on a MoE
target at batch 1. That is this repository's finding, reproduced independently
by someone who had every incentive to find the opposite.

He also names the lever, and it is not draft length:

> A larger batch size means more experts are already activated during native
> decoding, so activating additional experts in the target model may not become
> a bottleneck during the Eagle3 verification stage.

and notes that lmsys's published gpt-oss speedups used tensor parallelism across
four H200s, so the positive results live at multi-GPU, multi-batch scale.

This sharpens rather than softens the audit's position. The narrow findings
stand: no *coverage-threshold* signature appears in the sweep, and a single
regressor in draft volume accounts for 99.3 % of the cost across `n_max`
1 → 128. But "no MoE effect exists" was never the claim, upstream evidence for
a qualitative one predates this repository, and **the untested dimension is
batching** — which the original README listed as caveat (iv) and never
pursued. Single-stream is exactly the regime where the effect is expected to be
worst.

> Speedup is intrinsically limited on hybrid target models … each rejected step
> may require one extra target forward … A more fundamental future improvement
> would be target-side deferred commit: verify would compute temporary
> recurrent states, and only the accepted-prefix state would be committed.
> **That would remove replay from the hybrid path.**

That is the replay path behind A1's counter artefact and A6's abort. It is a
known upstream limitation with a proposed fix, not a discovery of this audit.
What this audit adds is the measurement: 1639 state checkpoints of 101.3 MiB
for a single 300-token request, the counter being unreachable because of it,
and the abort it caused at `bcb5eeb64`.

### A6. `llama-server` plus a draft model aborts on this model at `bcb5eeb64`

An audit retest on 2026-08-25, on the v2/v3 host with the same binary v2 used,
aborted reproducibly:

```
ggml/src/ggml-cuda/ggml-cuda.cu:97: CUDA error
CUDA error: an unsupported value or parameter was passed to the function
  in function ggml_cuda_op_mul_mat_cublas at ggml-cuda.cu:1618
  cublasSgemm_v2(..., CUBLAS_OP_T, CUBLAS_OP_N, row_diff, src1_ncols, ne10, ...)
  #14 server_context_impl::update_slots()
```

It fires immediately after a partial-accept checkpoint restore:

```
slot update_slots: n_draft=6, accepted=6          <- 6 < 6+1, so partial
slot update_slots: restoring speculative checkpoint (size = 65864420)
slot update_batch:  generate_draft: #tokens=60, #draft=6
srv  update_slots: decoding batch, n_tokens = 7
                                                  <- cublasSgemm gets a bad dimension
```

Reproduced 3 / 3 on the `code_small` prompt, in **both** the translation and the
matched-vocabulary arm, so the vocabulary fallback does not cause it. The
no-speculation arm completes all ten prompts every time.

This bears on two published claims. First, v2's "cross-checked on master
`bcb5eeb64`, identical results" is a `llama-cli` cross-check only — the
`llama-server` path that produced every v1 number does not survive on that
commit with a draft model attached. Second, the abort sits in the checkpoint
machinery introduced by PR #22227, in exactly the hybrid-SSM rollback area
PR #20075 was opened to fix and which was closed unmerged (A3). Whether it
persists on post-merge master was then tested: it does **not**. All thirty
requests complete on `3737e4137`. The abort is specific to the tested historical
commit. Evidence: [`v4_audit_2026_08_25/data/abort_evidence_bcb5eeb64.txt`](v4_audit_2026_08_25/data/abort_evidence_bcb5eeb64.txt).

---

### A11. Speculative decoding is not output-preserving on this build, and the engine is deterministic enough to prove it

Speculative decoding is normally described as exact: at temperature 0 the
verified tokens are the tokens the target would have produced alone, so the
method buys speed and changes nothing else. Every throughput comparison in this
repository has quietly assumed that. It is false here.

Run J logs the full text and the token ids for all 150 requests, so the claim is
checkable against committed data. Against the no-speculation arm, request by
request and repeat by repeat:

| arm | token streams identical to no speculation |
|---|---|
| `spec-dflash-n4` | 3 / 30 |
| `spec-dflash-n8` | 0 / 30 |
| `spec-dflash-n16` | 3 / 30 |
| `spec-draft-n8` | 0 / 30 |

**The control that makes this readable.** Divergence only means something if the
engine reproduces itself, and it does, exactly:

- every arm is byte-identical across its own three repeats — 10 / 10 prompts for
  all five arms;
- the no-speculation baseline is byte-identical across two *different runs* with
  different `-fit` settings — 30 / 30.

So the engine is deterministic for a fixed configuration, and turning
speculation on is what changes the output. The divergence is not early noise
either: the first differing token sits at index 40 at the earliest, median
110–126 of 300.

The likely mechanism is ordinary and not a llama.cpp defect: verifying a drafted
block computes the target's logits at a different batch shape than decoding one
token at a time, floating-point addition is not associative, and a near-tie in
the argmax resolves the other way. It is the same batch-invariance problem that
makes batched inference non-reproducible in general.

**That mechanism makes a prediction, and the prediction holds.** If divergence
is per-token near-ties accumulating, its probability should rise with output
length. Across every run of 2026-08-26:

| run | output length | token streams identical to no speculation |
|---|---|---|
| J | 300 (capped) | 6 / 120 |
| K1 | 300 (capped) | 0 / 180 |
| L, thinking on | 300 (capped) | 0 / 200 |
| L, thinking off | median 96 | **85 / 200** |

and within run L's thinking-off half, split at its own median: outputs under 96
tokens diverge 37.5 % of the time, outputs at or above it 70.8 %. The
determinism control holds everywhere — 70/70 self-reproducible prompts in K1 and
50/50 in each half of L, on top of J's original 50/50.

So the effect is not a quirk of one run, and it is not constant: at a 300-token
cap it is effectively certain, and at short answers it is a coin toss weighted by
length.

**What it changes here.** It does not invalidate run J's +18.7 %: every arm
generates exactly 300 tokens, so the throughput comparison counts the same
work, and the no-speculation arm's decode rate varies by only 0.8 % across ten
very different prompts — content-driven variation of that size cannot produce an
18.7 % difference. What it changes is the description. This is not a lossless
speedup of the same computation; it is a faster computation that lands on
slightly different text.

It also explains an observation v3 recorded without a cause: "outputs were not
token-identical between conditions" was listed there as a limitation of that
run (D4). It is a property of speculation on this build, and it reproduces.

---

## B — statistics that were reported incorrectly

### B1. `mean tok/s` was the request-mean only; pooled throughput is materially worse

`analysis/plot.py` averaged `predicted_per_second` across prompts, weighting a
fast 300-token request the same as a slow one. Pooled throughput,
`1000 · Σ predicted_n / Σ predicted_ms`, weights tokens equally. For configs
with a deep slow tail the two diverge sharply:

| config | request-mean | pooled | median | min |
|---|---:|---:|---:|---:|
| baseline | 135.7 (—) | 135.7 (—) | 135.6 | 135.3 |
| ngram-mod-n24 | 131.1 (−3.4 %) | 131.1 (−3.4 %) | 130.0 | 129.6 |
| draft-q35-08b-max8 | 121.1 (−10.8 %) | **109.9 (−19.0 %)** | 135.6 | 59.2 |
| ngram-cache | 119.1 (−12.2 %) | **111.3 (−18.0 %)** | 135.6 | 65.3 |
| ngcache-1000tok | 115.9 (−13.0 % vs 300-tok base) | **98.9 (−25.7 % vs baseline-1000tok)** | 133.1 | 60.0 |

Both are now reported, in `analysis/summary_by_config.csv` and on the charts.

### B2. The `±` column was across-prompt spread, not repeated-run uncertainty

Each v1 cell is one measurement of one prompt. The ten values behind a `std`
are ten *different prompts*, so the column is workload heterogeneity — not a
standard error, confidence interval, or run-to-run noise estimate. The charts
now draw min–max and say so; the CSV column is renamed `across_prompt_sd`.

The same conflation is in Exp 2's `66.57 ± 7.57 tok/s`. Decomposing its
15 cells (3 trials × 5 prompts):

| config | SD over all 15 cells | SD of the 3 trial means | mean within-prompt SD |
|---|---:|---:|---:|
| 01_baseline | 0.455 | 0.111 | 0.318 |
| 02_oleg_draft_2_32 | **7.566** | **0.058** | 0.175 |
| 03_srogmann_draft_48_64 | 1.804 | 0.053 | 0.211 |

The command is highly repeatable; the prompts differ from each other. The
published `± 7.57` invites the opposite reading.

### B3. "all completions reach the cap" is false for the 1000-token variants

README methodology: "Output capped at 300 tokens (and 1000 tokens in the
`-1000tok` variants); all completions reach the cap, so `predicted_n` is
constant across runs within a config."

True for the 300-token group — every one of those 150 requests returned
exactly 300. False for the 1000-token group. `baseline-1000tok` returned
`[354, 514, 801, 427, 1000, 891, 1000, 384, 1000, 484]`: three of ten hit the
cap. The other three 1000-token configs differ from it *and from each other* in
per-request length, so their aggregates must use actual token counts and must
be compared against `baseline-1000tok`, never against the 300-token baseline.

Relatedly, `baseline-1000tok` at −1.8 % versus `baseline` is not a clean
length effect: output length and generated content both change.

### B4. "every configuration hits a bimodal tail reaching as low as 59–67 tok/s" is false

The TL;DR contradicts the table directly below it. Minimum request rate by
family:

| family | minimum |
|---|---:|
| ngram-mod n = 8 / 12 / 16 / 20 / 24 | 120.0 / 119.8 / 123.8 / 128.8 / **129.6** |
| ngram-cache family | 65.3 / 65.6 / 60.0 |
| ngcache-kv-fp16 | 67.3 |
| classic draft max 8 / 16 / 32 | 59.2 / 59.6 / 59.5 |

The 59–67 band belongs to specific ngram-cache and classic-draft rows on
specific prompts. The whole ngram-mod family stays within 12 % of baseline at
its worst.

### B5. "the regression is entirely bimodal by prompt class" is false for the ngram-mod family

README: "chat prompts (`short_greet`, `multi_turn_*`, `zh_cn`) where ngram
cannot find hits stay at ~135 tok/s; structured prompts (`reasoning`,
`code_small`, `long_explain`) where drafts do trigger collapse to 59–95 tok/s."

Per-request `draft_n` says otherwise. For `ngram-mod-n24`, draft rounds were
recorded on `short_q`, `medium_chat`, `medium_rec`, `reasoning`,
`long_explain`, `multi_turn_1`, `multi_turn_2`, `zh_cn` — that is, on the chat
prompts the sentence says cannot find hits — and **not** on `code_small`.
Classic draft is the opposite: rounds only on `long_explain` and `code_small`,
with `reasoning` at full baseline speed. Two different configurations, two
different prompt partitions, neither matching the published taxonomy. Ten
hand-written prompts cannot establish a prompt taxonomy in any case.
`analysis/plot_per_prompt.png` now outlines the cells that actually recorded a
draft round.

### B6. "19-config speculative-decoding matrix" overstates the denominator

19 run labels, of which **14** recorded at least one fully accepted draft round
and **5** recorded none: `baseline`, `baseline-rerun`, `baseline-1000tok`,
`draft-qwen3-0.6b` (vocab 151936 ≠ 248320, draft never attached), and
`ngmod-n32`. Writing "every configuration is net-negative" over a denominator
that includes three baselines and a control inflates the claim.

A further gap: `run_verify_matrix.sh` defines an H1 `ngcache-nofa` condition,
but no `results/verify/ngcache-nofa.json` exists — `-fa off` is incompatible
with `-ctk q8_0`, so that hypothesis was never tested. `run_p0_matrix.sh` says
so in a comment; the README never mentioned it.

### B7. The fp16-KV row is a one-sided control — now closed by measurement

`ngcache-kv-fp16` was read as "fp16 KV does not rescue — KV quant is not the
cause". There is no no-speculation fp16-KV baseline in the v1 matrix, so the
row cannot separate a speculation effect from a KV-precision effect. It is
visible in the heatmap: on the seven prompts with no draft round it runs at
101–102 % of the q8_0 baseline, i.e. fp16 KV is *faster* when speculation is
idle.

The audit matrix adds the control v1 lacked. Post-merge master `3737e4137`,
5 repeats, ABBA-ordered, everything else held:

| arm | pooled tok/s | vs `baseline` |
|---|---:|---:|
| `baseline` (q8_0 KV) | 123.4 | — |
| `baseline-kvfp16` | **125.7** | **+1.9 %** |
| `ngram-cache` (q8_0 KV) | 74.0 | −40.0 % |
| `ngram-cache-kvfp16` | 70.9 | −42.5 % |

So fp16 KV is about 2 % faster than q8_0 KV with no speculation running, and it
does **not** help when speculation is on — the speculative arm is slightly
worse with fp16 KV, not better. The original reading happened to reach a
defensible conclusion, but it could not have known that: it was comparing a
speculative fp16-KV row against a non-speculative q8_0-KV row and attributing
the whole difference to speculation.

---

## C — artefact and scope naming

### C1. The target quantisation is `UD-Q4_K_XL`, not `Q4_K_M`

The v1/v2 conclusion sentence — "no spec-decode configuration on a consumer
3090 is a net win for Qwen3.6-35B-A3B **at Q4_K_M**" — names the wrong
artefact. `Q4_K_M` is the *draft* model's quantisation
(`Qwen3.5-0.8B-Q4_K_M.gguf`) and the Ollama comparison's, not the target's.
The tested target is `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`,
sha256 `707a55a8…f4450`. Quantisation recipe is a treatment dimension; the
generic "Q4" must not stand in for it.

### C2. The `zh_cn` prompt is Traditional Chinese

`bench_runner.py` prompt 10 is `你是桌面機器人。` / `請用一到兩句話介紹你自己。`
— Traditional Chinese, tagged `zh_cn`. The raw JSON keeps the historical tag;
charts and prose now use `zh_hant`.

### C3. `multi_turn_1` / `multi_turn_2` are not multi-turn

Each entry in `PROMPTS` is sent as a fresh two-message conversation. No state
carries between them, so they measure two independent single-turn requests.
`medium_rec` — "Earlier the user said their name is Hctsai. What did they tell
you?" — refers to a turn that never happened. A genuine multi-turn workload
would need conversation state and prefix reuse, which this harness does not do.

### C4b. "Stock clocks" was measured once, before the load

`BENCHMARK_ENV.md` states "**Stock clocks — no overclocking.** GPU is at the
factory-default power limit of 350 W", citing a single `nvidia-smi` snapshot
taken at bench start:

```
clocks.current.graphics : 1965 MHz     power.limit         : 350.00 W
clocks.current.memory   : 9751 MHz     power.default_limit : 350.00 W
```

The *no-overclocking* half is sound: `power.limit == power.default_limit` is a
real OC fingerprint. The rest is an idle-state reading, taken before any model
was loaded, and it says nothing about clocks under sustained load. No v1, v2 or
v3 run captured a clock, power or temperature trace **during** benchmarking, so
none of them can rule out thermal or power-cap drift across a run — and v1's
19 configs, v2's 9 and v3's 7 were each executed as one long sequence.

The audit added a continuous trace ([`bench/gpu_telemetry.sh`](bench/gpu_telemetry.sh),
5 s sampling of clocks, power, temperature, pstate and the
`clocks_throttle_reasons` bitmask) plus before/after snapshots per arm in
`bench/retest_runner.py`. Measured on this host under sustained matrix load:

Measured across the audit's own matrix run
(`python analysis/thermal_report.py <trace.csv>` reproduces this):

| quantity | observed |
|---|---|
| `power.limit` / `power.default_limit` / `power.max_limit` | 350 / 350 / 350 W — **not overclocked** |
| GPU temperature | 59–75 °C, mean 65.5, against a ~83 °C throttle point |
| graphics clock | 1815–1950 MHz of a 2100 MHz maximum, mean 1934 |
| `clocks_throttle_reasons.sw_power_cap` | active on 223 of 343 samples — the normal state for a GeForce card under load |
| `clocks_throttle_reasons.sw_thermal_slowdown` | active on **1 of 343**, at 64 °C |
| `clocks_throttle_reasons.hw_thermal_slowdown` | active on **1 of 343**, at 64 °C |
| `clocks_throttle_reasons.hw_power_brake_slowdown` | never active |
| `temperature.memory` | `N/A` — this card does not expose GDDR6X junction temperature through `nvidia-smi`, so the memory junction is not observable here |

Both thermal flags were raised while the clock sat at **1950 MHz, the run's
maximum**, so neither carried a downclock. And the drift runs the wrong way for
a thermal-bias worry: comparing the first half of the trace with the second,
the clock rose 1929 → 1940 MHz (+0.62 %) while temperature fell 66.9 → 64.0 °C.
The card got slightly cooler and slightly faster as the run went on.

An earlier draft of this section said no thermal bit was ever set. That was
written from the first 18 samples and it is wrong — two transient flags appear
later in the trace. Neither has a performance consequence, so the conclusion is
unchanged, but the claim had to be narrowed as data arrived. That is the whole
argument for keeping a trace instead of a single snapshot.

The matrix additionally repeats its no-speculation baseline across the run, so
drift is testable from the measurement itself and not only from telemetry: on
the three-repeat master run already committed, baseline reads 133.5 / 132.7 /
132.5 tok/s — a first-to-last change of −0.75 % and a full swing of 0.76 % of
the mean.

### C4. GPU 0 was running another workload

`run_matrix.sh` states its own confound: "All configs pinned to GPU 1
(`CUDA_VISIBLE_DEVICES=1`) so GPU 0 stays free for **Ollama**", and
`bench_runner.py --gpu` help says "use 1 to avoid GPU0 shared with Ollama". The
benchmark process had one 3090 to itself, but the host did not. No continuous
utilisation trace was captured, so host isolation is asserted, not shown.

---

## D — follow-up experiments whose stated treatment was not applied

### D1 / D2. `-no-cnv` was rejected and `/no_think` did not disable thinking

Every v2, v3, and Exp 2 script passes `-no-cnv` to `llama-cli` and appends
`/no_think` to each prompt. The committed logs show both fail:

```
--no-conversation is not supported by llama-cli
please use llama-completion instead
```

present in **61 of 62** v2 logs and **30 of 33** v3 logs — and the same logs
then contain `[Start thinking]` followed by a full reasoning trace, in 61/62
v2 and 30/33 v3 logs. The measured workload is long chain-of-thought output,
not the intended direct answer.

`BENCHMARK_ENV.md` described the tool as "`llama-cli -st -no-cnv` (single-turn
non-conversational)". That description is not what ran.

### D3. Exp 2 cannot be audited, so it cannot refute anything

`run_n3_codejson.sh` writes per-prompt logs to
`$HOME/bench/n3_codejson_*/trial_N/cfg/pI.log`; only `master.log` — 125 lines
of timing summaries — and `results.json` were committed. The generated text,
token IDs, and stop reasons are gone. Exp 2 ran on build `8889 (bcb5eeb64)`,
the same binary whose committed v2 cross-check logs prove `-no-cnv` is rejected
and `/no_think` inert, so there is no reason to believe the intended
"structured, low-entropy, thinking-off code/JSON" distribution was produced,
and no way to check.

`results.json` previously ended: "Workload-shape hypothesis (joshua Spark
NVFP4 idea) is **REFUTED** for this hardware/engine." That has been replaced
with a neutral status. Exp 2 shows the executed command was slower; it does not
test the hypothesis it was designed to test.

### D3b. Workload shape does matter — and Exp 2 pointed the wrong way

Exp 2 concluded that the workload-shape hypothesis was "REFUTED". D3 shows it
could not have tested that, because its thinking control never engaged. The
audit ran the test it was trying to run: the same five arms with thinking
verifiably on and verifiably off, 5 repeats each, on post-merge master
`3737e4137`, with `thinking_suppressed` recorded per request (50/50 in the
off run, 0/50 in the on run).

| method | thinking on | thinking off | draft tokens per generated token |
|---|---:|---:|---|
| `ngram-mod` n=24 | −6.8 % | **−0.7 %** | 0.21 → **0.00** |
| `ngram-cache` | −40.0 % | −32.6 % | 0.42 → 0.36 |
| draft model, n_max 8 | −74.0 % | −76.4 % | 1.85 → **2.14** |

**Workload shape changes the ngram result almost completely.** With thinking
off, `ngram-mod` stops drafting altogether — zero draft tokens across all 50
requests — and its deficit collapses from −6.8 % to −0.7 %. A chain-of-thought
trace is long and formulaic, which is exactly the repetitive text an n-gram
lookup feeds on; a direct answer is short and is not.

For the draft model the effect runs the other way. Acceptance falls from 29.7 %
to 23.0 % and drafted tokens per generated token rise from 1.85 to 2.14, so
turning thinking off makes it slightly *worse*. Reasoning traces are easier for
a 0.8 B drafter to predict than real answers.

Two consequences.

First, Exp 2's conclusion is not merely unverifiable, it is backwards for the
family of methods where workload shape matters most.

Second — and this needs stating per family, because a single sentence about it
would be wrong for half the methods. Every historical number here was taken on
the thinking workload: 76 % of v1's requests were truncated reasoning (A5), and
v2, v3 and Exp 2 all believed they had disabled it and had not (D1, D2). What
that means depends on the method:

- **Draft-model speculation was measured on its favourable workload.** Thinking
  traces are easier for a 0.8 B drafter to predict — 29.7 % acceptance against
  23.1 % on real answers — and the net result is better too, −74.0 % against
  −76.4 %. It still lost. For this family the negative direction is more robust
  than when it was published.
- **ngram methods were measured on their *unfavourable* workload.** Thinking
  traces give an n-gram lookup far more to fire on, and firing costs more than
  it returns here: `ngram-mod` goes from −6.8 % to −0.7 % once thinking is off,
  and `ngram-cache` from −40.0 % to −32.6 %. So the historical ngram figures
  **overstate** the cost on a real-answer workload. v1's `ngram-mod` −3.4 % is
  plausibly much closer to zero on the workload a user would actually run.

Neither family becomes a net win, but only one of them was being judged
generously.

### D4. v3 DFlash compares two different binaries

The build banner in the committed v3 logs:

| config | build |
|---|---|
| `01_baseline`, `03_oleg_draft_2_32`, `04_oleg_draft_2_16` | `b8889-bcb5eeb64` |
| `05_dflash_max16`, `06_dflash_max8`, `07_dflash_max4` | `b8942-67cb0d507` |

The 138.9 → 77.0 tok/s comparison therefore changes the binary and the
speculation method together, at one run per prompt/config, with the thinking
control inert. It is a negative observation for that run, not a DFlash effect
estimate. A clean A/B needs one pinned post-merge binary with DFlash off and on.

**Closed by measurement on 2026-08-26, and it reverses the sign.** Run J ran
DFlash off and on against one binary (`b6a5c490…`), one placement policy and
one drafter re-converted by post-merge master, three repeats per arm. At
`--spec-draft-n-max 4`, DFlash is **+18.7 %** against no speculation on
aggregate throughput (130.2 against 109.7 tok/s) and **+24.0 %** pooled (151.6
against 122.3), and it is faster on all ten prompts, not on average across
them. The archived v3 direction — DFlash slower — reappears only at longer
draft windows: −14.8 % at n_max 8 and −47.4 % at n_max 16.

So v3's number was not merely unattributable, it pointed the wrong way about
the method. What v3 measured at n_max 4 was a binary change, and what it read
as a DFlash penalty is, on one binary, the largest speedup in this repository.
See [`v4_audit_2026_08_25/README.md`](v4_audit_2026_08_25/README.md#run-j--the-first-configuration-that-is-actually-faster).

Note also that `BENCHMARK_ENV.md` recorded `llama-cli --version` as
`8889 (bcb5eeb64) -- inherited from master at fork point`, while the run logs
report `b8942-67cb0d507`. The logs are authoritative; `--version` was captured
before the DFlash rebuild.

### D5. The committed v2 script does not produce the committed v2 directories

`v2_3090_followup/bench_3090_oleg.sh` writes configs tagged
`01_baseline`, `02_srogmann_ngmod_n24`, `03_oleg_draft_2_32`,
`04_oleg_draft_2_16`. The committed data directory `v2_oleg_suggestions/`
contains `01_baseline`, `02_oleg_draft_2_32`, `03_oleg_draft_2_16`,
`04_draft_2_64`. The two do not correspond, so the committed script is not the
one that generated the committed logs. No script at all is committed for
`v2_controls/` (A–E) or `v2_master_cross_check/` (M1–M3).

Compounding this, no v2 or v3 log records its own argv. The mapping from a
directory name to the flags that produced it rests entirely on prose. Config
identity for v2/v3 is therefore asserted, not archived. The v2 aggregate
numbers themselves do reproduce exactly from the logs — every value in
`SUMMARY.md` was re-derived during this audit — but which flags produced which
directory cannot be verified from the repository.

### D6. `--spec-type` is not "missing from master"; it is server-only

v3 recorded `02_srogmann_ngmod_n24` as "not in master, fail-fast" after

```
error: invalid argument: --spec-type
```

The argument exists at both tested revisions. At `97895129e` and at
`bcb5eeb64`, `common/arg.cpp` registers it as
`.set_examples({LLAMA_EXAMPLE_SERVER})` — it is accepted by `llama-server` and
rejected by `llama-cli`. v1 used `llama-server` and exercised the flag
successfully; v2/v3 used `llama-cli` and could not. The ngram-mod family is
therefore v1-only in this repository for a tooling reason, not an upstream one.

---

## E — theory errors

### E1. The coverage threshold is 95, and it is a heuristic

With 256 routed experts and top-8 routing, ρ = 8/256 = 0.03125 and

```
log(1 − 0.95) / log(1 − ρ) = 94.36   →   T_95 = 95 tokens
```

(94 tokens gives 94.94 % coverage; 95 gives 95.10 %.) The README's "≈ 94" is
the un-ceilinged value. More important than the off-by-one: this is the
expected number of tokens to touch 95 % of routed experts under an i.i.d.
uniform-routing approximation. It is not a performance threshold, and crossing
it is neither necessary nor sufficient for speculative decoding to win.

Notation: the README used `K` for both routed-experts-per-token and draft
length. Now `k_e = 8` and `γ` respectively.

### E2. Qwen3.5-122B-A10B has the *same* routing, so the same threshold

README: "A10B has a 3.3× larger active footprint and a correspondingly lower
`T_thres`, which is why it gains where A3B loses."

`Qwen/Qwen3.5-122B-A10B` `text_config`: `num_experts = 256`,
`num_experts_per_tok = 8` — identical to Qwen3.6-35B-A3B. Under the formula the
README itself cites, ρ and therefore `T_95` are the same for both models.
Larger active parameter count is a different quantity and does not move this
threshold. The A10B result is a genuine counterexample to any universal
A3B-derived rule, but this repository cannot say which factor explains the sign
difference, because model, hardware, backend, quantisation, draft
configuration, and implementation all differ at once.

### E3. PR #20075 is not "the same ngram-mod machinery", and is not srogmann's

README: "the same `ngram-mod` machinery in PR #20075 shows Qwen3.5-122B-A10B …
gaining roughly +15–45 % on Apple M3 Max (PR author's bench, 0.8 B draft…)".
`pr_comment.md` additionally attributes it to "srogmann's own benchmark".

PR #20075 is authored by **eauchs** and is a `find_slot` correction plus an SSM
checkpoint/restore rolling buffer. Its benchmark is
`Qwen3.5-122B-A10B-UD-Q4_K_XL` with an **external `Qwen3.5-0.8B` draft model**,
baseline ~20.4 t/s → 23.5–29.7 t/s (the source of "+15–45 %"), acceptance
63–89 %. No ngram-mod is involved. The percentage was right; the mechanism and
the attribution were not.

---

## F — metadata

### F1. Upstream statuses, checked 2026-08-25 via the GitHub API

| PR | README said | actual |
|---|---|---|
| #19493 speculative checkpointing | merged 2026-04-19 | merged 2026-04-19 ✓ |
| #22227 speculative-simple checkpoint | merged 2026-04-22 | merged 2026-04-22 ✓ |
| #20075 hybrid SSM/MoE fix | **OPEN** | **closed unmerged 2026-04-25** |
| #22105 DFlash | v3: **open draft** | **merged 2026-06-28** |

### F2. "Cross-validated on current master `bcb5eeb64`"

`bcb5eeb64` was master on 2026-04-22. It has not been master for four months.
The claim is now phrased as a dated snapshot cross-check. Only
`v2_master_cross_check/` was run on it — `v2_oleg_suggestions/` and
`v2_controls/` are `b8863-97895129e`.

### F3. "first public benchmark / first public datapoint"

Absolute novelty claims appeared in the title, the v3 banner, the v3 README,
`pr_comment.md`, and `CHANGELOG.md`. None was accompanied by a search date,
query set, or inclusion criteria, so none is defensible. All have been removed
in favour of an exact scope statement.

### F4. Licence conflict

`LICENSE` is MIT; the README said data is "CC-0" with no CC0 text in the tree;
`v3_dflash_2026_05_07/README.md` said "Apache 2.0". Resolved: MIT for code and
documentation, CC0-1.0 for benchmark data with `DATA_LICENSE` and
`LICENSES/CC0-1.0.txt` present and scoped. The Apache 2.0 line is removed.
`CITATION.cff` added.

### F5. The reproduce section could not reproduce the result

It ran `git clone --depth 1 … && cmake` — i.e. built whatever master happened
to be — for a benchmark pinned to `97895129e`. It also hard-coded
`~/benchmarks/...` paths and a `/home/reachym/dev/reachy-agent/robot/.venv`
interpreter, and conflated the driver-supported CUDA 13.0 reported by
`nvidia-smi` with the CUDA 12.6 toolkit used to build. Fixed: exact
`git checkout`, model SHA-256 verification, and env-var overrides for every
host-specific path.

---

## What the audit did **not** change

- No raw measurement file was edited. `results/`, `results/verify/`,
  `v2_3090_followup/v2_*/`, `exp2_codejson_n3/master.log`, and
  `v3_dflash_2026_05_07/data/` are byte-identical to the published releases.
- Every v1, v2, v3, and Exp 2 aggregate was re-derived from the raw files
  during the audit and reproduced exactly. The arithmetic was never the problem.
- The narrow negative observation survives **inside its own scope**: under the
  exact conditions archived here, no tested condition that recorded speculative
  activity beat its matched no-speculation reference in aggregate. What does not
  survive is the acceptance anomaly, the mechanism, and the generality.
- That scope turned out to matter. Every method v1 tested drives an **external**
  drafter, and every one of them still loses on the controlled tier — batching
  included, which widens the gap rather than closing it. Methods that draft from
  the target's *own* layers were never in the archive, and they win: see
  [D4](#d4-v3-dflash-compares-two-different-binaries) and the run J, K and L
  sections of
  [`v4_audit_2026_08_25/README.md`](v4_audit_2026_08_25/README.md). "Speculative
  decoding loses on this hardware" was a statement about a regime this
  repository had not separated, not about the hardware.
