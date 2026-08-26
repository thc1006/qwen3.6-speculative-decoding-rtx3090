"""Controlled speculative-decoding retest runner (2026-08-25 audit).

Fixes every protocol defect the audit found in the v1/v2/v3 harnesses:

  * one pinned binary for every arm, hashed into the manifest
    (v3 compared bcb5eeb64 against 67cb0d507 - ERRATA D4)
  * ABBA / interleaved arm ordering, not baseline-then-treatments
  * N repeats per arm, so a spread is run-to-run uncertainty (ERRATA B2)
  * full per-request capture: generated text, the reasoning channel, token IDs,
    stop reason, and the whole `timings` block (v1 kept only a 120-char head of
    `content` and discarded `reasoning_content`; Exp 2 kept nothing - ERRATA D3,
    A5). Token IDs come from `logprobs`, not from `return_tokens`: the OAI chat
    serialiser `to_json_oaicompat_chat()` emits only finish_reason / index /
    message / logprobs, so `return_tokens` is silently dropped on that endpoint.
    `probs_vector_to_json` emits `{"id": p.tok, ...}`, which is the id we keep.
    That list is near-complete rather than exact: `probs_output` drops the
    trailing stop-word tokens (`server-context.cpp:2036-2039`), so it can run a
    few short of `predicted_n`. Measured on this build: exact for some requests,
    2-5 short for others. `predicted_n` remains the authority for token counts;
    `tokens` is for inspecting what was actually generated.
  * the server's own `-v` log per arm, so the drafter's honest
    `statistics draft:` counters survive (ERRATA A1)
  * a manifest recording argv, binary sha256, model sha256s, and GPU telemetry
    before and after each arm (ERRATA D5: the committed v2 script does not
    correspond to the committed v2 data)
  * every host-specific path is an environment variable

Everything is configured through the environment:

    LLAMA_SERVER_BIN   path to llama-server                    (required)
    MODEL_TARGET       target .gguf                            (required)
    MODEL_DRAFT        draft .gguf                             (optional)
    MODEL_DFLASH       DFlash drafter .gguf                    (optional)
    BENCH_GPU          CUDA_VISIBLE_DEVICES value              (default 0)
    BENCH_PORT         server port                             (default 18131)
    BENCH_OUT          output directory                        (default ./retest_out)
    BENCH_REPEATS      repeats of the full prompt set per arm  (default 3)
    BENCH_MAX_TOKENS   max_tokens per completion               (default 300)
    BENCH_ARMS         comma-separated arm names to run        (default: all)
    BENCH_FLAVOR       legacy | master  - speculative flag spelling (default legacy)
    BENCH_PROMPTS      v1 | extended - which prompt set. Default v1, the ten
                       prompts every archived and v4 number rests on. `extended`
                       is twenty deliberately different ones, including two real
                       multi-turn exchanges, and exists to test whether a result
                       is a property of the v1 mix.
    BENCH_CTX          N  - context passed as -c, applied to every arm in the
                       run (default 16384). Lower it to buy headroom, but note
                       that with BENCH_FIT=on the fitter will reclaim what you
                       free unless BENCH_FIT_TARGET is raised too.
    BENCH_FIT_TARGET   MiB - passed as --fit-target when BENCH_FIT=on. The
                       margin the fitter leaves free per device; upstream default
                       is 1024, which is not enough room for a BF16 draft model
                       and its compute buffer.
    BENCH_FIT          on | off - drop the -ngl 999 pin so llama.cpp's memory
                       fitter can adjust unset parameters. Applied to every arm
                       in the run so placement policy stays constant.
    BENCH_CONCURRENCY  N  - keep N prompts in flight at once and add
                       --parallel N -cb to the server (default 1). BOTH halves
                       are needed: --parallel only ALLOCATES N slots, so a
                       client that waits for each reply before sending the next
                       leaves N-1 of them idle and measures nothing about
                       batching. Above 1 the honest metric changes: per-request
                       predicted_per_second is no longer system throughput, so
                       the runner records wall_s and aggregate_tok_s over the
                       whole prompt set - at every level, c=1 included, so the
                       two are comparable on one denominator. Upstream names
                       batching, not draft length, as the lever that could make
                       speculative decoding pay on a MoE target (ERRATA A9), and
                       this repository never tested it - the original README
                       listed it as an untested caveat.
    BENCH_THINK        on | off  - request-level thinking control (default on)
                       `off` sends chat_template_kwargs {"enable_thinking": false}.
                       Whether it took effect is VERIFIED per request and recorded
                       as `thinking_suppressed`; never assume it worked - /no_think
                       in the prompt text never did (ERRATA D2).

Run: python bench/retest_runner.py
"""
from __future__ import annotations

import hashlib
import concurrent.futures as cf
import json
import re
import socket
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# --------------------------------------------------------------- config ----

SERVER = os.environ.get("LLAMA_SERVER_BIN", "")
TARGET = os.environ.get("MODEL_TARGET", "")
DRAFT = os.environ.get("MODEL_DRAFT", "")
DFLASH = os.environ.get("MODEL_DFLASH", "")
# Qwen3.5/3.6 ship a multi-token-prediction head. `--mtp` on master's converter
# exports it as a standalone draft GGUF; nothing about that needed patching,
# it had simply never been run here. See RETEST_TODO.
MTP = os.environ.get("MODEL_MTP", "")
GPU = os.environ.get("BENCH_GPU", "0")
PORT = int(os.environ.get("BENCH_PORT", "18131"))
OUT = Path(os.environ.get("BENCH_OUT", "retest_out"))
REPEATS = int(os.environ.get("BENCH_REPEATS", "3"))
MAX_TOKENS = int(os.environ.get("BENCH_MAX_TOKENS", "300"))
# Accept the obvious spellings. A silent mismatch here would put thinking back
# on without saying so, which is exactly the failure mode that made v2, v3 and
# Exp 2 unauditable (ERRATA D2) - so normalise, and record the result in the
# manifest and per request.
IGNORE_EOS = os.environ.get("BENCH_IGNORE_EOS", "off").strip().lower() in (
    "on", "1", "true", "yes")
_THINK_RAW = os.environ.get("BENCH_THINK", "on").strip().lower()
_THINK_OFF = {"off", "0", "false", "no", "think_off", "disabled", "none"}
_THINK_ON = {"on", "1", "true", "yes", "enabled"}
# Fail closed. Mapping every unrecognised string to "on" is exactly the silent
# mismatch this was written to prevent: BENCH_THINK=of or =disabledd would have
# quietly left thinking enabled and produced a plausible-looking matrix.
if _THINK_RAW in _THINK_OFF:
    THINK = "off"
elif _THINK_RAW in _THINK_ON:
    THINK = "on"
else:
    sys.exit(f"BENCH_THINK={_THINK_RAW!r} is not recognised; use one of "
             f"{sorted(_THINK_ON)} or {sorted(_THINK_OFF)}")
CONCURRENCY = max(1, int(os.environ.get("BENCH_CONCURRENCY", "1")))
_ORDER_RAW = os.environ.get("BENCH_ORDER", "latin").strip().lower()
if _ORDER_RAW not in ("latin", "cyclic", "mirrored"):
    sys.exit("BENCH_ORDER must be 'latin', 'cyclic' or 'mirrored'")
ORDER_MODE = _ORDER_RAW
# Pinning -ngl 999 makes llama.cpp's memory fitter abort ("n_gpu_layers already
# set by user to 999, abort") instead of adjusting the parameters the caller
# left unset. That is why the BF16 DFlash drafter appeared not to fit: with -ngl
# unset the same drafter loads at the full 16384 context. BENCH_FIT=on drops the
# pin for EVERY arm in a run, so the placement policy stays constant across the
# comparison instead of differing between the arm that needed it and the rest.
FIT = os.environ.get("BENCH_FIT", "off").strip().lower() in ("on", "1", "true", "yes")

# v1's fixed server flags, kept verbatim so the retest stays comparable.
# Context is a run-level control, not a constant. With the BF16 DFlash drafter
# resident at -c 16384 the card peaks at 23946 MiB of 24576 - 630 MiB spare, and
# allocations of 120 MiB have still failed there. Lowering the context for EVERY
# arm in a run, baseline included, buys headroom without making the arms
# incomparable; it does make absolute rates incomparable ACROSS runs, so a run
# that changes it needs its own baseline, which is why every matrix here carries
# one.
CTX = os.environ.get("BENCH_CTX", "16384").strip()
# --fit-target is the margin the fitter leaves free per device, default 1024 MiB.
# That default is why DFlash arms die at the edge: the fitter sizes the TARGET to
# leave 1024 MiB, and then the drafter has to live inside that margin - a BF16
# DFlash drafter plus its compute buffer does not fit in 1 GiB, so the run peaks
# at 23.9 GiB of 24 GiB and whether an arm survives is decided by whether the
# previous arm's memory came back in time. Lowering -c does not help, because the
# fitter simply takes the freed memory back; raising the margin does.
FIT_TARGET = os.environ.get("BENCH_FIT_TARGET", "").strip()
COMMON_ARGS_PINNED = [
    "-ngl", "999", "-c", CTX, "--jinja",
    "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
    "--no-webui", "-v",
]
COMMON_ARGS = ([a for i, a in enumerate(COMMON_ARGS_PINNED)
                if not (a == "-ngl" or (i > 0 and COMMON_ARGS_PINNED[i - 1] == "-ngl"))]
               if os.environ.get("BENCH_FIT", "off").strip().lower() in ("on", "1", "true", "yes")
               else COMMON_ARGS_PINNED)

# The BOS key the draft GGUF never received: Qwen/Qwen3.5-0.8B has no
# generation_config.json upstream (HTTP 404), so convert_hf_to_gguf wrote no
# tokenizer.ggml.bos_token_id and llama.cpp substitutes the hard-coded GPT-2
# legacy default 11. common_speculative_are_compatible then compares 248044
# against 11 and disables matched-vocabulary speculation - over a field that
# neither model uses, since both set add_bos_token = false. See ERRATA A2.
BOS_OVERRIDE = ["--override-kv", "tokenizer.ggml.bos_token_id=int:248044"]

# v1's ten prompts. `zh_cn` is renamed to its correct script (ERRATA C2);
# the multi_turn tags are kept but are two independent single-turn requests,
# which is exactly what the historical harness measured (ERRATA C3).
PROMPTS: list[tuple[str, str, str]] = [
    ("short_greet",  "You are a friendly desk robot.", "Hey there!"),
    ("short_q",      "You are a friendly desk robot.", "How's your day going?"),
    ("medium_chat",  "You are a friendly desk robot.",
     "What do you think about humans writing code for you?"),
    ("medium_rec",   "You are a friendly desk robot.",
     "Earlier the user said their name is Hctsai. What did they tell you?"),
    ("reasoning",    "You solve problems step by step.",
     "If a train leaves Paris at 2pm going 120km/h and another leaves Berlin at 3pm "
     "going 80km/h on a 1000km track towards each other, where do they meet? "
     "Think briefly step by step."),
    ("long_explain", "You teach patiently.",
     "Explain to a curious 10-year-old what makes a rainbow form, in 4 to 6 sentences."),
    ("code_small",   "You write Python.",
     "Write a short Python function that takes a list of ints and returns only the "
     "primes, with a quick comment."),
    ("multi_turn_1", "You are a friendly desk robot.",
     "Let's play a game. Pick a random number between 1 and 100 and tell me."),
    ("multi_turn_2", "You are a friendly desk robot.",
     "What is your favorite kind of music and why? Give two concrete examples."),
    ("zh_hant",      "你是桌面機器人。", "請用一到兩句話介紹你自己。"),
]

# P0-1: the decisive A/B. Identical binary, identical draft file, identical
# flags - the only difference is whether the compatibility gate sees a matching
# BOS id, i.e. whether speculation runs on the token-translation fallback.
#
# llama.cpp renamed the speculative arguments after bcb5eeb64:
#   --draft-max -> --spec-draft-n-max ,  --draft-min -> --spec-draft-n-min
# BENCH_FLAVOR selects the spelling. `-md` and `-ngld` are accepted by both.
FLAVOR = os.environ.get("BENCH_FLAVOR", "legacy")
_NMAX, _NMIN = (("--draft-max", "--draft-min") if FLAVOR == "legacy"
                else ("--spec-draft-n-max", "--spec-draft-n-min"))
# On post-merge master `--spec-type` defaults to `none`: passing `-md` alone
# loads the draft model and then never speculates (verified 2026-08-25 -
# zero `generate_draft` calls, draft_n = 0 on every request). At bcb5eeb64
# `-md` was sufficient. The explicit type is required on master.
_SPEC_TYPE = [] if FLAVOR == "legacy" else ["--spec-type", "draft-simple"]
_DRAFT_ARGS = ["-md", "{DRAFT}", "-ngld", "99", _NMAX, "8", _NMIN, "4"] + _SPEC_TYPE

# An arm may carry the sentinel "--kv-fp16", which is not a llama.cpp flag: the
# runner strips it and drops `-ctk q8_0 -ctv q8_0` from COMMON_ARGS for that arm.
# It exists because the v1 matrix has an `ngcache-kv-fp16` row with no matched
# no-speculation fp16-KV control, so that row cannot separate a speculation
# effect from a KV-precision effect (ERRATA B7).
# Every number in the v4 tier rests on the ten prompts above - 226 arm-runs of
# them. A result that is an artefact of that particular mix would be invisible
# across all of them, because they all share it. PROMPTS_EXTENDED is a second,
# deliberately different set: longer inputs, structured output, four languages,
# arithmetic, and two genuinely multi-turn exchanges rather than the v1 set's
# two independent single-turn requests. Select with BENCH_PROMPTS=extended;
# the default is unchanged so every historical comparison still joins.
PROMPTS_EXTENDED: list[tuple[str, str, object]] = [
    # --- long input, short output: the shape a summariser has ---------------
    ("summarise_long", "You summarise accurately and briefly.",
     "Summarise the following in exactly three sentences.\n\n"
     "The Gated DeltaNet layer replaces softmax attention with a linear "
     "recurrence whose state is updated by a delta rule, so the cost of "
     "generating a token does not grow with the length of the context. The "
     "trade-off is that the state is a fixed-size summary, so information that "
     "is not written into it is unrecoverable. Hybrid models interleave a small "
     "number of full-attention layers among the recurrent ones to recover exact "
     "recall where it matters. In practice the interleaving ratio is chosen "
     "empirically, and the resulting model behaves like a recurrent network for "
     "throughput purposes and like a transformer for recall on the layers that "
     "keep full attention. Serving such a model complicates any technique that "
     "needs to roll a sequence back, because the recurrent state cannot be "
     "truncated in place the way a key-value cache can."),
    # --- structured, low-entropy output -------------------------------------
    ("json_schema", "You output JSON and nothing else.",
     "Emit a JSON object describing a job queue: keys name, version, and "
     "queues (array). Each queue has name, max_retries, visibility_timeout_s, "
     "and dead_letter (string or null). Include queues named ingest, transcode "
     "and notify."),
    ("sql_report", "You write PostgreSQL.",
     "Write a query returning, per month of 2026, the number of distinct users "
     "who placed at least two orders that month. Tables: orders(id, user_id, "
     "placed_at, total_cents). Comment each clause."),
    ("regex_explain", "You explain precisely.",
     "What does the regular expression ^(?!.*--)[a-z0-9](?:[a-z0-9-]{0,61}"
     "[a-z0-9])?$ match, and what does the negative lookahead add? Answer in "
     "four or five sentences."),
    # --- code, three languages ----------------------------------------------
    ("code_rust", "You write idiomatic Rust.",
     "Write a function that takes &[u8] and returns Result<String, Utf8Error> "
     "containing the input hex-encoded, without allocating per byte. Include a "
     "doc comment."),
    ("code_python", "You write Python with type hints.",
     "Write a context manager `timed(label)` that logs the wall-clock duration "
     "of the block at DEBUG level, and re-raises any exception unchanged after "
     "logging the duration."),
    ("code_bash", "You write portable shell.",
     "Write a POSIX sh function that retries a command up to N times with "
     "exponential backoff, printing each attempt to stderr, and returns the "
     "command's final exit status."),
    # --- arithmetic and multi-step reasoning --------------------------------
    ("arithmetic", "You compute carefully and show your work.",
     "A server writes 82.079 MiB of state 772 times and reads back 82.08 MiB "
     "709 times during a 123.9-second run. How many GiB moved in total, and "
     "what fraction of a 936 GB/s memory bandwidth does that represent if it "
     "were spread evenly? Show the steps."),
    ("logic_puzzle", "You reason step by step.",
     "Five benchmark runs A-E finished in some order. C did not finish first or "
     "last. A finished immediately before E. B finished after D but before C. "
     "What was the order? Explain each deduction."),
    # --- four languages ------------------------------------------------------
    ("zh_hant_long", "你是一位技術寫作者,使用臺灣繁體中文。",
     "請用四到六句話說明:為什麼在混合式遞迴模型上,推測解碼的回滾成本會比在純"
     "注意力模型上高?請避免使用簡體字詞彙。"),
    ("ja_explain", "あなたは簡潔な技術ライターです。",
     "投機的デコーディングにおける「受理率」と「実効速度」が必ずしも一致しない"
     "理由を、三文から四文で説明してください。"),
    ("de_explain", "Du schreibst präzise technische Erklärungen.",
     "Erkläre in vier Sätzen, warum ein separates Entwurfsmodell auf einem "
     "Mixture-of-Experts-Ziel teurer sein kann, als die Intuition nahelegt."),
    ("fr_explain", "Tu écris des explications techniques concises.",
     "Explique en quatre phrases la différence entre l'auto-spéculation et la "
     "spéculation avec un modèle de brouillon distinct."),
    # --- genuinely multi-turn, which the v1 set never was --------------------
    ("multiturn_real_1", "You are a careful assistant with a long memory.",
     [{"role": "user", "content": "I'm benchmarking a 35B MoE model on a single 24 GB card."},
      {"role": "assistant", "content": "Understood. What are you measuring?"},
      {"role": "user", "content": "Decode throughput with and without speculative decoding."},
      {"role": "assistant", "content": "Noted. Anything else I should keep in mind?"},
      {"role": "user", "content": "Yes - the draft model has to fit in what's left. "
                                  "Given all that, what should I watch out for first?"}]),
    ("multiturn_real_2", "You are a careful assistant with a long memory.",
     [{"role": "user", "content": "My name is Hctsai and my card is an RTX 3090."},
      {"role": "assistant", "content": "Got it, Hctsai."},
      {"role": "user", "content": "The target model is 21 GB on disk."},
      {"role": "assistant", "content": "That is most of the card."},
      {"role": "user", "content": "Remind me what I told you about myself and the hardware, "
                                  "then say what headroom is left."}]),
    # --- open-ended prose, high entropy -------------------------------------
    ("creative", "You write vividly and briefly.",
     "Write a six-line poem about a benchmark that disproves its own author. "
     "Do not use the word 'irony'."),
    ("opinion", "You give balanced, concrete answers.",
     "Is it reasonable to publish a benchmark that retracts its own headline? "
     "Give the strongest argument on each side, in about six sentences."),
    # --- instruction-following edge cases -----------------------------------
    ("constrained", "You follow formatting instructions exactly.",
     "List exactly seven differences between a recurrent state and a key-value "
     "cache. Number them 1 to 7. Each item must be one sentence and must not "
     "contain a comma."),
    ("refusal_adjacent", "You are helpful and precise.",
     "Explain how speculative decoding could be used to fingerprint which model "
     "a server is running, and what a server operator could do about it."),
    ("terse", "You answer in as few words as possible.",
     "What is the break-even draft acceptance rate for a drafter that costs one "
     "quarter of a target forward pass and drafts two tokens per round? Show the "
     "expression, then the number."),
]

PROMPT_SETS = {"v1": PROMPTS, "extended": PROMPTS_EXTENDED}
_PS = os.environ.get("BENCH_PROMPTS", "v1").strip().lower()
if _PS not in PROMPT_SETS:
    sys.exit(f"BENCH_PROMPTS must be one of {', '.join(PROMPT_SETS)}")
PROMPT_SET_NAME = _PS
PROMPTS = PROMPT_SETS[_PS]

KV_FP16 = "--kv-fp16"


def _draft(nmax, nmin=1, extra=(), p_min=None):
    args = (["-md", "{DRAFT}", "-ngld", "99", _NMAX, str(nmax), _NMIN, str(nmin)]
            + _SPEC_TYPE + BOS_OVERRIDE + list(extra))
    if p_min is not None:
        args += ["--spec-draft-p-min", str(p_min)]
    return args


def _ngram(kind, extra=()):
    if FLAVOR == "legacy":
        return ["--spec-type", kind] + list(extra)
    return ["--spec-type", kind] + list(extra)


def _mtp(nmax: int, nmin: int = 1) -> list[str]:
    """An MTP arm at draft length nmax, using the target's own MTP head."""
    return ["-md", "{MTP}", "-ngld", "99", _NMAX, str(nmax), _NMIN, str(nmin),
            "--spec-type", "draft-mtp"]


def _dflash(nmax: int, nmin: int = 1) -> list[str]:
    """A DFlash arm at draft length nmax.

    No BOS override: that fix is specific to the Qwen3.5-0.8B matched-vocabulary
    drafter, which shipped without tokenizer.ggml.bos_token_id (ERRATA A2).
    DFlash reuses the target's own vocabulary, so the special-token gate in
    common_speculative_are_compatible() is not in play.
    """
    return ["-md", "{DFLASH}", "-ngld", "99", _NMAX, str(nmax), _NMIN, str(nmin),
            "--spec-type", "draft-dflash"]


ARMS: dict[str, list[str]] = {
    # --- P0-1: the decisive vocabulary A/B, kept for continuity ---------------
    "baseline":             [],
    "draft-max8-translate": list(_DRAFT_ARGS),
    "draft-max8-matched":   list(_DRAFT_ARGS) + BOS_OVERRIDE,

    # --- P3-1: draft-length sweep, matched vocabulary, n_min pinned to 1 so the
    #           only thing varying is how many tokens the drafter may propose ---
    "spec-draft-n1":  _draft(1),
    "spec-draft-n2":  _draft(2),
    "spec-draft-n4":  _draft(4),
    "spec-draft-n8":  _draft(8),
    "spec-draft-n16": _draft(16),
    "spec-draft-n32": _draft(32),
    # Past MoESD's ~95-token expected-coverage threshold. A sweep that stops at
    # 32 cannot test that argument at all, so these three exist to reach the
    # regime it actually describes: if the amortisation MoESD predicts is real,
    # throughput should stop falling somewhere around here.
    "spec-draft-n64": _draft(64),
    "spec-draft-n96": _draft(96),
    "spec-draft-n128": _draft(128),

    # P2-2: DFlash against the SAME binary, which the archived v3 comparison
    # never had - it read a bcb5eeb64-vs-67cb0d507 difference as a DFlash
    # effect (ERRATA D4). Needs a drafter re-converted by post-merge master:
    # the archived GGUF lacks `target_layers` and the merged loader rejects it.
    # No BOS override here - that fix is specific to the Qwen3.5-0.8B drafter.
    # MTP: the target's own multi-token-prediction head, exported as a drafter.
    # This is the method the vLLM sibling result on this same hardware uses, so
    # until it runs here "llama.cpp loses where vLLM wins" confounds engine with
    # method. The drafter reuses the target's vocabulary, so no BOS override.
    "spec-mtp-n1":  _mtp(1),
    "spec-mtp-n2":  _mtp(2),
    "spec-mtp-n3":  _mtp(3),
    "spec-mtp-n4":  _mtp(4),
    "spec-mtp-n6":  _mtp(6),
    "spec-mtp-n8":  _mtp(8),

    # Run J found n4 at +18.7 % aggregate and n8 already negative, so the
    # optimum is at or below 4 and the sweep has to reach down to 1 to bracket
    # it. A three-point sweep that happens to straddle the peak cannot say
    # where the peak is.
    "spec-dflash-n1":  _dflash(1),
    "spec-dflash-n2":  _dflash(2),
    "spec-dflash-n3":  _dflash(3),
    "spec-dflash-n4":  _dflash(4),
    "spec-dflash-n6":  _dflash(6),
    "spec-dflash-n8":  _dflash(8),
    "spec-dflash-n16": _dflash(16),

    # p_min truncates a draft once the drafter's confidence drops below it. The
    # default CHANGED between the binaries this repository has used:
    #   9789512 / bcb5eeb64 -> 0.75   (every archived v1/v2/v3 number)
    #   master  3737e4137   -> 0.00   (the whole audit matrix, by default)
    # So the matrix ran with draft truncation OFF while every historical figure
    # ran with it on. These arms measure that difference instead of assuming it
    # away.
    "spec-draft-n8-pmin50":  _draft(8, p_min=0.50),
    "spec-draft-n8-pmin75":  _draft(8, p_min=0.75),
    "spec-draft-n8-pmin90":  _draft(8, p_min=0.90),
    "spec-draft-n32-pmin75": _draft(32, p_min=0.75),
    "spec-draft-n128-pmin75": _draft(128, p_min=0.75),
    # v1's actual classic-draft configuration, for comparability
    "spec-draft-v1cfg": _draft(8, 4),

    # --- the ngram family, which v1 could run and v2/v3 could not (ERRATA D6) --
    "ngram-cache":  _ngram("ngram-cache"),
    "ngram-simple": _ngram("ngram-simple"),
    # Successor mapping for v1's `--spec-type ngram-mod --spec-ngram-size-n 24
    # --draft-min 48 --draft-max 64`, checked against both trees:
    #   bcb5eeb64  ngram-mod read the GENERIC speculative.n_min / n_max and the
    #              shared speculative.ngram_size_n (whose help text mentioned
    #              only ngram-simple/ngram-map, but the ngram-mod draft loop
    #              used it as the lookup length)
    #   master     those knobs are split per implementation:
    #              ngram_mod.n_min / n_max / n_match
    # so 48/64/24 map to --spec-ngram-mod-n-{min,max,match} respectively.
    "ngram-mod-n24": _ngram("ngram-mod", ["--spec-ngram-mod-n-max", "64",
                                          "--spec-ngram-mod-n-min", "48",
                                          "--spec-ngram-mod-n-match", "24"]),

    # --- P1-2: the fp16-KV control the v1 matrix never had --------------------
    "baseline-kvfp16":    [KV_FP16],
    "ngram-cache-kvfp16": _ngram("ngram-cache") + [KV_FP16],

    # The two ngram-map variants master exposes and this repository had never
    # run. They need no draft model at all, so they are the cheapest available
    # test of whether "short drafts win" is a property of DFlash or a property
    # of draft volume. Their defaults are size_n 12 / size_m 48 / min_hits 1 -
    # a 48-token draft, which every other family here has been punished for.
    "ngram-map-k":        _ngram("ngram-map-k"),
    "ngram-map-k-m8":     _ngram("ngram-map-k", ["--spec-ngram-map-k-size-n", "12",
                                                 "--spec-ngram-map-k-size-m", "8"]),
    "ngram-map-k-m4":     _ngram("ngram-map-k", ["--spec-ngram-map-k-size-n", "12",
                                                 "--spec-ngram-map-k-size-m", "4"]),
    "ngram-map-k4v":      _ngram("ngram-map-k4v"),
    "ngram-map-k4v-m8":   _ngram("ngram-map-k4v", ["--spec-ngram-map-k4v-size-n", "12",
                                                   "--spec-ngram-map-k4v-size-m", "8"]),
    "ngram-map-k4v-m4":   _ngram("ngram-map-k4v", ["--spec-ngram-map-k4v-size-n", "12",
                                                   "--spec-ngram-map-k4v-size-m", "4"]),
}


# ---------------------------------------------------------------- utils ----

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def runner_sha256() -> str:
    """This file's own hash. The manifest records which binary answered and
    which model it loaded; without this it does not record which harness asked.
    Run O2's identity fields were empty because of a regex in this file, and
    nothing in its output said which version of the file that was."""
    try:
        return sha256(__file__)
    except Exception:  # noqa: BLE001
        return ""


def prompt_set_sha256() -> str:
    """Hash of the prompt set as sent, not of its name.

    `prompt_set: v1` in a manifest is a label; two runs can carry the same label
    and different prompts if the list is edited between them, and every
    cross-run comparison in this repository assumes they cannot.
    """
    canon = json.dumps(PROMPTS, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":")).encode("utf-8")
    return sha256_bytes(canon)


def harness_tree_sha() -> dict:
    """Where the harness came from, when it is run from a checkout.

    The benchmark host has no clone of this repository - the runner is copied
    to it - so this is best-effort and `BENCH_HARNESS_SHA` is how the caller
    states it explicitly. Recorded as `declared` in that case, so a reader can
    tell an assertion from an observation.
    """
    declared = os.environ.get("BENCH_HARNESS_SHA", "").strip()
    if declared:
        return {"sha": declared, "source": "declared via BENCH_HARNESS_SHA"}
    try:
        out = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent),
                              "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            dirty = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent),
                                    "status", "--porcelain"],
                                   capture_output=True, text=True, timeout=10).stdout
            return {"sha": out.stdout.strip(),
                    "dirty": bool(dirty.strip()),
                    "source": "git rev-parse in the harness directory"}
    except Exception:  # noqa: BLE001
        pass
    return {"sha": None, "source": "not a checkout and BENCH_HARNESS_SHA unset"}


# If set, every arm-run must report this commit / these library hashes, and the
# run aborts on the first that does not. Without it a binary swapped between
# arms is invisible: `server_lib_sha256` is taken once, at run start.
EXPECT_COMMIT = os.environ.get("BENCH_EXPECT_COMMIT", "").strip()
EXPECT_LIB = os.environ.get("BENCH_EXPECT_LIB_SHA256", "").strip()


def check_identity(arm: str, rep: int, ident: dict, libs: dict) -> list[str]:
    """Compare what answered against what the caller said should answer."""
    bad: list[str] = []
    if EXPECT_COMMIT:
        got = ident.get("commit") or ""
        if not got.startswith(EXPECT_COMMIT) and not EXPECT_COMMIT.startswith(got or "\0"):
            bad.append(f"{arm} rep{rep}: server reports commit {got!r}, "
                       f"BENCH_EXPECT_COMMIT={EXPECT_COMMIT!r}")
    if EXPECT_LIB:
        impl = libs.get("libllama-server-impl.so", "")
        if not impl.startswith(EXPECT_LIB):
            bad.append(f"{arm} rep{rep}: libllama-server-impl.so is {impl[:16]!r}, "
                       f"BENCH_EXPECT_LIB_SHA256={EXPECT_LIB[:16]!r}")
    return bad


def _server_lib_hashes() -> dict:
    """Hash every shared object beside the server binary, de-duplicated by the
    file the symlink resolves to. Both `libfoo.so` and `libfoo.so.0` are present
    for most of them and they are the same file."""
    if not SERVER:
        return {}
    seen: dict[str, str] = {}
    by_target: dict[str, str] = {}
    for p in sorted(Path(SERVER).parent.glob("*.so*")):
        try:
            real = str(p.resolve())
        except OSError:
            continue
        if real in by_target:
            continue
        by_target[real] = p.name
        seen[p.name] = sha256(real)
    return seen


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# A multi-hour matrix can be biased by the GPU downclocking as it heats, and by
# whether the card is overclocked or power-capped at all. Two snapshots per arm
# plus the always-on trace from bench/gpu_telemetry.sh make that testable rather
# than assumed. `clocks_event_reasons.active` is a bitmask: 0x1 GpuIdle,
# 0x2 ApplicationsClocks, 0x4 SwPowerCap, 0x8 HwSlowdown, 0x20 SwThermal,
# 0x40 HwThermal, 0x80 HwPowerBrake. Comparing power.limit against
# power.default_limit is the OC fingerprint.
GPU_FIELDS = (
    "index,name,memory.used,utilization.gpu,"
    "clocks.current.graphics,clocks.max.graphics,"
    "clocks.current.sm,clocks.current.memory,clocks.max.memory,"
    "power.draw,power.limit,power.default_limit,power.max_limit,"
    "temperature.gpu,pstate,clocks_throttle_reasons.active"
)


def nvidia_smi() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader"],
            text=True, timeout=20).strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}"


def port_is_free(port: int) -> bool:
    """True if nothing is listening on the port.

    Checked before every spawn. Without it a stale server left on the configured
    port answers /health, `wait_health` returns before it ever looks at the
    process it just started, and the whole arm-run is measured against the wrong
    binary and the wrong model while the manifest records the intended ones.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sk:
        sk.settimeout(1.0)
        return sk.connect_ex(("127.0.0.1", port)) != 0


def server_identity(log_path: Path) -> dict:
    """Read back who actually answered, from the server's own startup log."""
    out: dict = {}
    try:
        text = log_path.read_text(errors="replace")[:400_000]
    except Exception:  # noqa: BLE001
        return out
    # `common_params_print_info: build 10622 (3737e4137) with GNU ...` - no colon
    # after `build`. The first version of this required one and silently matched
    # nothing, so all 81 arm-runs of run O2 recorded an empty identity.
    if m := re.search(r"\bbuild\s+(\d+)\s+\(([0-9a-f]{7,40})\)", text):
        out["build"] = m.group(1)
        out["commit"] = m.group(2)
    for key, pat in (("model_path", r"llama_model_loader: loaded meta data from ([^\s]+)"),
                     ("arch", r"general\.architecture\s*(?:str|=)\s*=?\s*(\w+)")):
        if m := re.search(pat, text):
            out[key] = m.group(1)
    return out


def wait_health(port: int, timeout: float = 300.0,
                proc: subprocess.Popen | None = None) -> float:
    """Block until /health answers, the server exits, or the timeout expires.

    Watching `proc` is the whole point of the second argument. Without it a
    server that aborts during startup - a compute-buffer OOM, a rejected GGUF -
    burns the full timeout doing nothing, and a matrix with a systematically
    failing arm spends `arms x repeats x timeout` seconds discovering the same
    failure over and over. Run K lost three minutes per dead arm to exactly
    this before the check was added.
    """
    t0 = time.perf_counter()
    url = f"http://127.0.0.1:{port}/health"
    while time.perf_counter() - t0 < timeout:
        # Liveness BEFORE readiness. The other order accepts a 200 from whatever
        # happens to own the port, including a server this run did not start.
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited with code {proc.returncode} after "
                f"{time.perf_counter() - t0:.1f}s without becoming healthy")
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return time.perf_counter() - t0
        except Exception:  # noqa: BLE001
            pass
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited with code {proc.returncode} after "
                f"{time.perf_counter() - t0:.1f}s without becoming healthy")
        time.sleep(0.5)
    raise RuntimeError(f"llama-server not ready after {timeout}s")


def start_server(extra: list[str], log_path: Path,
                 args_of_arm: list[str] | None = None) -> subprocess.Popen:
    args_of_arm = args_of_arm if args_of_arm is not None else extra
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU
    common = list(COMMON_ARGS)
    extra = [a for a in extra if a != KV_FP16]
    if KV_FP16 in list(args_of_arm):
        # fp16 KV: drop the q8_0 cache-type flags entirely
        for f in ("-ctk", "-ctv"):
            i = common.index(f)
            del common[i:i + 2]
    cmd = [SERVER, "-m", TARGET, "--host", "127.0.0.1", "--port", str(PORT)]
    if FIT and FIT_TARGET:
        cmd += ["--fit-target", FIT_TARGET]
    if CONCURRENCY > 1:
        # Total -c is held at 16384 rather than scaled by N, so VRAM and model
        # placement stay identical across concurrency levels; llama.cpp splits
        # it into N slots, and 16384/8 = 2048 per slot is still far more than
        # this workload uses (longest prompt + 300 generated tokens). Holding
        # the memory footprint constant is the control that matters here.
        cmd += ["--parallel", str(CONCURRENCY), "-cb"]
    cmd += common + [a.replace("{DRAFT}", DRAFT).replace("{DFLASH}", DFLASH)
                     .replace("{MTP}", MTP)
                     for a in extra]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not port_is_free(PORT):
        raise RuntimeError(
            f"port {PORT} already has a listener. Refusing to start: a stale "
            f"server there would answer /health and this arm-run would measure "
            f"it while the manifest recorded the binary and models below.")
    proc = subprocess.Popen(cmd, env=env, stdout=log_path.open("w"),
                            stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    proc._cmd = cmd  # type: ignore[attr-defined]
    return proc


def gpu_mem_used_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
             "-i", GPU],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return int(out[0].strip())
    except Exception:  # noqa: BLE001
        return None


def stop_server(proc: subprocess.Popen, settle_mib: int = 2048,
                settle_timeout: float = 60.0) -> dict:
    """Kill the server AND wait for the driver to hand the memory back.

    Reaping the process is not the same as the CUDA context being torn down.
    The next arm's `-fit on` probe reads free device memory to choose its
    parameters, so starting it against a stale reading picks parameters that do
    not fit and the arm dies on its first decode. Run J's telemetry peaks at
    23946 MiB of 24576 with the DFlash drafter resident - 630 MiB of headroom,
    2.6 % of the card - and a 120 MiB allocation still failed in run K, so the
    true transient peak is higher than 5-second sampling can see. At that margin
    a late teardown is the difference between an arm running and an arm
    crashing.
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:  # noqa: BLE001
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=15)
        except Exception:  # noqa: BLE001
            pass
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < settle_timeout:
        used = gpu_mem_used_mib()
        if used is None or used <= settle_mib:
            return {"settled": True, "mib_after": used,
                    "wait_s": round(time.perf_counter() - t0, 2),
                    "readable": used is not None}
        time.sleep(1.0)
    used = gpu_mem_used_mib()
    print(f"    ! GPU still reports {used} MiB in use "
          f"{settle_timeout:.0f}s after the server was killed", flush=True)
    # Printing this and carrying on is what the first version did. The next arm
    # then sizes itself with `-fit on` against a stale free-memory reading, so
    # the failure lands on the *following* arm and looks like that arm's fault.
    # Recorded per arm-run and treated as a run-level failure below.
    return {"settled": False, "mib_after": used,
            "wait_s": round(time.perf_counter() - t0, 2), "readable": True}


def chat(system: str, user) -> dict:
    """`user` is a string for a single turn, or a list of message dicts for a
    real multi-turn exchange. The v1 set only ever used the first form, which is
    why its `multi_turn_1` / `multi_turn_2` tags are two independent single-turn
    requests (ERRATA C3). The extended set uses the second."""
    turns = user if isinstance(user, list) else [{"role": "user", "content": user}]
    body = {
        "model": "retest",
        "messages": [{"role": "system", "content": system}] + turns,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "seed": 42,
        "stream": False,
        # token ids arrive through logprobs on this endpoint - see module docstring
        "logprobs": True,
    }
    if THINK == "off":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if IGNORE_EOS:
        # Force every arm to generate exactly MAX_TOKENS.
        #
        # With thinking on, every request hits the cap anyway and the arms are
        # length-matched for free. With thinking off they are not: speculation
        # is not output-preserving on this build (ERRATA A11), so the arms stop
        # at different points, and 34 of 60 arm/prompt pairs in run R differ in
        # length - one prompt by 38 %, another by 48 % in the other direction.
        # Pooled throughput then compares arms that did different amounts of
        # work. This is RETEST_TODO P1-3's "force ignore_eos + a hard cap".
        body["ignore_eos"] = True
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        data = json.loads(r.read())
    wall_ms = (time.perf_counter() - t0) * 1000
    t = data.get("timings", {}) or {}
    ch = data["choices"][0]
    msg = ch.get("message", {}) or {}
    lp = (ch.get("logprobs") or {}).get("content") or []
    reasoning = msg.get("reasoning_content", "") or ""
    content = msg.get("content", "") or ""
    return {
        "wall_ms": wall_ms,
        # the request's own window on the process clock. Two requests overlap
        # iff their windows do, which is how max_in_flight below is derived
        # instead of trusted.
        "t_start": t0,
        "t_end": t0 + wall_ms / 1000.0,
        "finish_reason": ch.get("finish_reason"),
        "usage": data.get("usage", {}),
        "timings": t,
        "predicted_ms": t.get("predicted_ms", 0),
        "predicted_n": t.get("predicted_n", 0),
        "predicted_per_second": t.get("predicted_per_second", 0),
        "draft_n": t.get("draft_n", 0),
        "draft_n_accepted": t.get("draft_n_accepted", 0),
        # full text, both channels - the audit found Exp 2 unauditable
        # precisely because these were discarded (ERRATA D3)
        "content": content,
        "reasoning_content": reasoning,
        # measured, not assumed: v1/v2/v3 all believed thinking was off when it
        # was not (ERRATA A5, D2). An empty reasoning channel is the evidence.
        "thinking_suppressed": len(reasoning) == 0,
        "n_reasoning_chars": len(reasoning),
        "n_content_chars": len(content),
        "tokens": [x.get("id") for x in lp],
    }


# ----------------------------------------------------------------- main ----

def _report(arm: str, rep: int, r: dict) -> None:
    acc = (f"  counted-draft {r['draft_n_accepted']}/{r['draft_n']}"
           if r["draft_n"] else "")
    think = "" if r["thinking_suppressed"] else "  THINKING"
    print(f"    [{arm} rep{rep} {r['tag']:13s}] {r['predicted_n']:>4d}tok "
          f"@ {r['predicted_per_second']:6.1f} tok/s{acc}{think}", flush=True)


def run_prompt_set(arm: str, rep: int,
                   proc: subprocess.Popen) -> tuple[list[dict], dict | None, float]:
    """Issue the prompt set and return (rows, crashed, wall_s).

    CONCURRENCY is the number of requests in flight, and that is the entire
    point of the concurrency arm. An earlier revision of this file documented
    concurrent dispatch in its env block but issued the prompts one at a time,
    so `--parallel 4` allocated four slots and three of them sat idle. The
    signature was unmistakable once measured: the c=4 arm-runs took 44 s and
    118 s against c=1's 44 s and 116 s. Identical wall-clock at four times the
    nominal batch width is what a batch size of one looks like.

    wall_s is measured at every level, c=1 included, so a batched arm and a
    sequential one can be compared on the same denominator instead of the
    analysis having to reconstruct one from sum(wall_ms).
    """
    rows: list[dict] = []
    crashed: dict | None = None
    t0 = time.perf_counter()

    if CONCURRENCY == 1:
        for tag, sysmsg, usermsg in PROMPTS:
            try:
                r = chat(sysmsg, usermsg)
            except Exception as e:  # noqa: BLE001
                # A server death is a finding, not a harness failure: record
                # where it happened and move on to the next arm.
                crashed = {"tag": tag, "error": f"{type(e).__name__}: {e}",
                           "returncode": proc.poll()}
                print(f"    [{arm} rep{rep} {tag:13s}] SERVER DIED: "
                      f"{type(e).__name__}", flush=True)
                break
            r["tag"] = tag
            rows.append(r)
            _report(arm, rep, r)
        return rows, crashed, time.perf_counter() - t0

    # Every prompt is submitted up front; the pool holds CONCURRENCY of them in
    # flight. Results are re-ordered to the prompt list afterwards so row order
    # matches the sequential path and repeats stay comparable.
    got: dict[str, dict] = {}
    with cf.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        fut = {pool.submit(chat, sysmsg, usermsg): tag
               for tag, sysmsg, usermsg in PROMPTS}
        for f in cf.as_completed(fut):
            tag = fut[f]
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001
                # Keep the FIRST failure: once the server is gone the rest of
                # the batch fails for the same reason and would bury it.
                if crashed is None:
                    crashed = {"tag": tag, "error": f"{type(e).__name__}: {e}",
                               "returncode": proc.poll()}
                print(f"    [{arm} rep{rep} {tag:13s}] REQUEST FAILED: "
                      f"{type(e).__name__}", flush=True)
                continue
            r["tag"] = tag
            got[tag] = r
            _report(arm, rep, r)
    wall_s = time.perf_counter() - t0
    rows = [got[tag] for tag, _, _ in PROMPTS if tag in got]
    return rows, crashed, wall_s


def max_client_requests_in_flight(rows: list[dict]) -> int:
    """Largest number of CLIENT requests outstanding at any instant.

    This is not the server's decode batch width and must never be reported as
    one. It is computed from when the client sent each request and when it
    received the complete response, so a server that processes every request
    strictly serially still shows all of their HTTP windows overlapping while
    the later ones sit in its queue. An earlier version of this file called it
    `max_in_flight` and described it as reading the batch width "back out of
    the timestamps", which it cannot do.

    What it does establish is the negative case: if this is 1 while N were
    requested, the client never had more than one request outstanding and the
    run measures nothing about concurrency. That is the failure it was written
    to catch, and it remains valid for that.

    Measuring the achieved batch width needs server-side instrumentation of
    active sequences and batch/ubatch token counts per decode.
    """
    events = []
    for r in rows:
        if "t_start" in r and "t_end" in r:
            events.append((r["t_start"], 1))
            events.append((r["t_end"], -1))
    if not events:
        return 0
    # ends before starts at an identical timestamp, so a handover is not
    # miscounted as an overlap
    events.sort(key=lambda e: (e[0], e[1]))
    cur = peak = 0
    for _, d in events:
        cur += d
        peak = max(peak, cur)
    return peak


SCHEDULE: list[list[str]] = []
TEARDOWN: dict = {}


def position_counts(schedule: list[list[str]]) -> dict[str, list[int]]:
    pos: dict[str, list[int]] = {a: [] for a in schedule[0]}
    for order in schedule:
        for i, a in enumerate(order):
            pos[a].append(i + 1)
    return pos


def build_schedule(arms: list[str], repeats: int, mode: str) -> list[list[str]]:
    """Build the arm order for every block, and refuse to call it balanced
    unless it is.

    `latin` is reserved for a schedule where every arm visits every position
    exactly once, which a cyclic rotation gives only when `repeats == len(arms)`.
    An earlier version generated the rotation for any pair and labelled the
    result `latin` regardless: three arms over four repeats produces rotations
    0, 1, 2, 0, so one arm sits in the same position twice while the manifest
    claimed balance. `cyclic` is that same rotation, named for what it is.
    """
    n = len(arms)
    if mode == "mirrored":
        sched = [arms if r % 2 == 0 else list(reversed(arms)) for r in range(repeats)]
    else:
        step = max(1, n // max(1, repeats))
        sched = [arms[(r * step) % n:] + arms[:(r * step) % n] for r in range(repeats)]
    if mode == "latin":
        pos = position_counts(sched)
        balanced = all(sorted(v) == list(range(1, n + 1)) for v in pos.values())
        if not balanced:
            sys.exit(
                f"BENCH_ORDER=latin asks for a position-balanced schedule and "
                f"{n} arms over {repeats} repeats cannot give one: "
                + "; ".join(f"{a} at {v}" for a, v in list(pos.items())[:3])
                + f". Use BENCH_REPEATS={n} for a full cyclic Latin square, or "
                f"BENCH_ORDER=cyclic to run the rotation without claiming balance.")
    return sched


def run_arm(arm: str, rep: int) -> dict:
    extra = ARMS[arm]
    log_path = OUT / "server_logs" / f"{arm}__rep{rep}.log"
    # Caveat on this snapshot: the previous arm's server has been waited on, but
    # the driver can still report the outgoing process's utilisation and clocks
    # for a moment, so `gpu_before` occasionally shows ~0 MiB used alongside
    # 100 % utilisation. The continuous trace from bench/gpu_telemetry.sh is the
    # authority for GPU state; these two snapshots are a convenience.
    gpu_before = nvidia_smi()
    proc = start_server(extra, log_path, args_of_arm=extra)
    try:
        try:
            ready_s = wait_health(PORT, proc=proc)
        except Exception as e:  # noqa: BLE001 - bad flags / OOM / model rejected
            return {"arm": arm, "repeat": rep, "ready_s": None,
                    "argv": proc._cmd,  # type: ignore[attr-defined]
                    "gpu_before": gpu_before, "gpu_after": nvidia_smi(),
                    "server_log": str(log_path.relative_to(OUT)),
                    "crashed": {"tag": "__startup__", "error": f"{type(e).__name__}: {e}",
                                "returncode": proc.poll()},
                    "rows": []}
        crashed = None
        # full-shape warm-up: same length as the measured requests, so the
        # first measured prompt is not paying graph/alloc costs (v1 warmed up
        # with a single 8-token completion)
        try:
            # Warm up at the SAME batch width as the measurement: the first
            # batched decode otherwise pays graph and allocation costs that a
            # single warm-up request never triggers, and that cost would land
            # inside the measured window.
            _wu = ("You are concise.", "Warm up with a few sentences about the weather.")
            if CONCURRENCY == 1:
                chat(*_wu)
            else:
                with cf.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
                    for f in [pool.submit(chat, *_wu) for _ in range(CONCURRENCY)]:
                        f.result()
        except Exception as e:  # noqa: BLE001
            return {"arm": arm, "repeat": rep, "ready_s": ready_s,
                    "argv": proc._cmd,  # type: ignore[attr-defined]
                    "gpu_before": gpu_before, "gpu_after": nvidia_smi(),
                    "server_log": str(log_path.relative_to(OUT)),
                    "crashed": {"tag": "__warmup__", "error": f"{type(e).__name__}: {e}",
                                "returncode": proc.poll()},
                    "rows": []}
        rows, crashed, wall_s = run_prompt_set(arm, rep, proc)
        n_tok = sum(r["predicted_n"] for r in rows)
        agg = n_tok / wall_s if wall_s > 0 else float("nan")
        peak = max_client_requests_in_flight(rows)
        print(f"    [{arm} rep{rep} {'AGGREGATE':13s}] {n_tok:>4d}tok in "
              f"{wall_s:6.1f}s = {agg:6.1f} tok/s aggregate  "
              f"(c={CONCURRENCY}, peak client requests in flight {peak})", flush=True)
        if peak < CONCURRENCY:
            print(f"    [{arm} rep{rep}] WARNING: asked for {CONCURRENCY} client "
                  f"requests in flight, only ever observed {peak}. This arm-run "
                  f"does NOT measure concurrency.", flush=True)
        return {
            "arm": arm, "repeat": rep, "ready_s": ready_s,
            # who actually answered, read back from the server's own startup log
            "server_pid": proc.pid,
            "server_identity": server_identity(log_path),
            # Taken per arm-run, not once per run: the run-level hash cannot
            # see a binary replaced between two arms.
            "server_lib_sha256": _server_lib_hashes(),
            "server_loaded_commit": server_identity(log_path).get("commit"),
            # `server_log_sha256` is filled in by the driver after the server is
            # stopped: hashing it here would hash a file still being written.
            # system-level metric, valid at every concurrency level
            "concurrency": CONCURRENCY,
            "wall_s": wall_s,
            "aggregate_tok_s": agg,
            "max_client_requests_in_flight": peak,
            "max_in_flight": peak,  # deprecated alias, runs A-R used this name
            "argv": proc._cmd,  # type: ignore[attr-defined]
            "gpu_before": gpu_before, "gpu_after": nvidia_smi(),
            "server_log": str(log_path.relative_to(OUT)),
            "crashed": crashed,
            "rows": rows,
        }
    finally:
        TEARDOWN[(arm, rep)] = stop_server(proc)


def validate_run(out: Path, arms: list[str], repeats: int,
                 results: list[dict]) -> list[str]:
    """Return every reason this directory is not a complete, self-consistent run.

    Checks the exact (arm, repeat) Cartesian product, not a count: N*R files can
    be reached with one arm run twice and another not at all.
    """
    problems: list[str] = []
    expected = {(a, r) for a in arms for r in range(repeats)}

    seen: dict[tuple[str, int], int] = {}
    for res in results:
        key = (res.get("arm"), res.get("repeat"))
        seen[key] = seen.get(key, 0) + 1
    for key, n in sorted(seen.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        if n > 1:
            problems.append(f"arm-run {key[0]} rep{key[1]} recorded {n} times")
    for a, r in sorted(expected - set(seen)):
        problems.append(f"arm-run {a} rep{r} missing from the results")
    for key in sorted(set(seen) - expected, key=lambda k: (str(k[0]), str(k[1]))):
        problems.append(f"arm-run {key[0]} rep{key[1]} was not scheduled")

    for res in results:
        arm, rep = res.get("arm"), res.get("repeat")
        problems += check_identity(arm, rep, res.get("server_identity") or {},
                                   res.get("server_lib_sha256") or {})
        td = res.get("teardown") or {}
        if td.get("readable") and not td.get("settled"):
            problems.append(f"arm-run {arm} rep{rep}: the GPU still held "
                            f"{td.get('mib_after')} MiB {td.get('wait_s')}s after "
                            f"teardown, so the next arm sized itself against a "
                            f"stale reading")
        if res.get("crashed"):
            c = res["crashed"]
            problems.append(f"arm-run {arm} rep{rep} crashed at "
                            f"{c.get('tag')}: {c.get('error')}")
            continue
        rows = res.get("rows") or []
        if len(rows) != len(PROMPTS):
            problems.append(f"arm-run {arm} rep{rep} has {len(rows)} prompt rows, "
                            f"expected {len(PROMPTS)}")
        got = {row.get("tag") for row in rows}
        want = {t for t, _, _ in PROMPTS}
        if got != want:
            problems.append(f"arm-run {arm} rep{rep} prompt tags differ: "
                            f"missing {sorted(want - got)}, extra {sorted(got - want)}")
        for row in rows:
            if not row.get("predicted_n"):
                problems.append(f"arm-run {arm} rep{rep} prompt {row.get('tag')} "
                                f"produced no tokens")
            elif IGNORE_EOS and row["predicted_n"] != MAX_TOKENS:
                problems.append(f"arm-run {arm} rep{rep} prompt {row.get('tag')} "
                                f"generated {row['predicted_n']} tokens with "
                                f"BENCH_IGNORE_EOS on; the point of that flag is "
                                f"that every arm generates exactly {MAX_TOKENS}")

    # A file whose name disagrees with its contents makes every per-arm glob a
    # lie, and no count-based check can see it.
    for f in sorted(out.glob("*__rep*.json")):
        stem = f.name[:-len(".json")]
        f_arm, _, f_rep = stem.rpartition("__rep")
        try:
            body = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:                       # noqa: BLE001 - report, not raise
            problems.append(f"{f.name} is not readable JSON: {e}")
            continue
        if body.get("arm") != f_arm or str(body.get("repeat")) != f_rep:
            problems.append(f"{f.name} contains arm={body.get('arm')!r} "
                            f"repeat={body.get('repeat')!r}")
    on_disk = {f.name for f in out.glob("*__rep*.json")}
    for a, r in sorted(expected):
        if f"{a}__rep{r}.json" not in on_disk:
            problems.append(f"{a}__rep{r}.json was not written")
    return problems


def main() -> None:
    missing = [n for n, v in (("LLAMA_SERVER_BIN", SERVER), ("MODEL_TARGET", TARGET)) if not v]
    if missing:
        sys.exit(f"set {', '.join(missing)}")
    if ORDER_MODE == "mirrored" and REPEATS % 2 == 1:
        sys.exit(f"BENCH_ORDER=mirrored with BENCH_REPEATS={REPEATS} is not balanced: "
                 f"the arm order runs forward/reverse/forward and the first arm never "
                 f"leaves position 1 on the odd repeats. Use an even repeat count, or "
                 f"BENCH_ORDER=latin.")
    # A directory that already holds results will be globbed by the analysis as
    # if it belonged to this run. Refuse rather than mix two runs together.
    stale = sorted(OUT.glob("*__rep*.json")) if OUT.exists() else []
    if stale:
        sys.exit(f"BENCH_OUT={OUT} already contains {len(stale)} arm-run files "
                 f"(e.g. {stale[0].name}). Refusing to write into it: the analysis "
                 f"cannot tell the two runs apart. Use a fresh directory.")
    wanted = os.environ.get("BENCH_ARMS")
    arms = [a.strip() for a in wanted.split(",")] if wanted else list(ARMS)
    for a in arms:
        if a not in ARMS:
            sys.exit(f"unknown arm {a!r}; known: {', '.join(ARMS)}")
    if any("{DRAFT}" in x for a in arms for x in ARMS[a]) and not DRAFT:
        sys.exit("set MODEL_DRAFT for the draft arms")
    if any("{MTP}" in x for a in arms for x in ARMS[a]) and not MTP:
        sys.exit("set MODEL_MTP for the MTP arms")
    if any("{DFLASH}" in x for a in arms for x in ARMS[a]) and not DFLASH:
        sys.exit("set MODEL_DFLASH for the DFlash arms")

    # Build and validate the block schedule before the first server starts, so a
    # schedule that cannot deliver what BENCH_ORDER promises costs zero GPU time.
    global SCHEDULE
    SCHEDULE = build_schedule(arms, REPEATS, ORDER_MODE)

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": os.uname().nodename,
        "gpu_env": GPU,
        "server_bin": SERVER,
        "server_sha256": sha256(SERVER),
        # `llama-server` is a ~18 kB launcher; the server logic lives in the
        # shared objects beside it, and the launcher is byte-identical between a
        # stock build and one with instrumentation compiled in. Hashing only the
        # launcher records nothing about the code that answered the requests, so
        # every shared object in the same directory is hashed too.
        "server_lib_sha256": _server_lib_hashes(),
        "target": TARGET, "target_sha256": sha256(TARGET),
        "draft": DRAFT or None,
        "draft_sha256": sha256(DRAFT) if DRAFT else None,
        "dflash": DFLASH or None,
        "mtp": MTP or None,
        "dflash_sha256": sha256(DFLASH) if DFLASH else None,
        "mtp_sha256": sha256(MTP) if MTP else None,
        "common_args": COMMON_ARGS,
        "flavor": FLAVOR,
        "arms": {a: ARMS[a] for a in arms},
        "repeats": REPEATS, "max_tokens": MAX_TOKENS,
        "temperature": 0.0, "seed": 42, "think": THINK, "think_env": _THINK_RAW, "ignore_eos": IGNORE_EOS,
        "concurrency": CONCURRENCY, "fit": FIT, "ctx": CTX,
        "runner_sha256": runner_sha256(),
        "harness_tree_sha": harness_tree_sha(),
        "prompt_set_sha256": prompt_set_sha256(),
        "expect_commit": EXPECT_COMMIT or None,
        "expect_lib_sha256": EXPECT_LIB or None,
        "order_mode": ORDER_MODE,
        "schedule": SCHEDULE,
        "schedule_position_counts": position_counts(SCHEDULE) if SCHEDULE else {},
        "schedule_is_position_balanced": bool(SCHEDULE) and all(
            sorted(v) == list(range(1, len(SCHEDULE[0]) + 1))
            for v in position_counts(SCHEDULE).values()),
        "prompt_set": PROMPT_SET_NAME, "n_prompts": len(PROMPTS),
        "prompt_tags": [t for t, _, _ in PROMPTS],
        "fit_target": FIT_TARGET or None,
        "ordering": {
            "latin": "cyclic rotation, validated position-balanced before the run",
            "cyclic": "cyclic rotation; NOT position-balanced and not claimed to be",
            "mirrored": "arm order reversed on odd repeats; balanced only for even repeats",
        }[ORDER_MODE],
        "gpu_fields": GPU_FIELDS,
        "nvidia_smi": nvidia_smi(),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "arms"}, indent=2))

    results = []
    for rep in range(REPEATS):
        # Arm position is confounded with time unless it is balanced. Reversing
        # the list on odd repeats - what this called "ABBA" - only balances when
        # the repeat count is even, and with three repeats it runs
        # forward/reverse/forward, leaving the first arm at positions 1, N, 1.
        #
        #   latin     rotate by the repeat index. Each arm occupies `repeats`
        #             distinct positions, advancing uniformly; with repeats == N
        #             it is a full cyclic Latin square. This is the default.
        #   cyclic    the same rotation when repeats != N, where it cannot be
        #             position-balanced. Named separately so the manifest never
        #             claims a balance the schedule does not have.
        #   mirrored  the old forward/reverse alternation, kept for continuity
        #             with runs A-R. Rejected for odd repeats, where it is not
        #             balanced at all.
        order = SCHEDULE[rep]
        print(f"\n=== repeat {rep}  order: {' -> '.join(order)} ===", flush=True)
        for arm in order:
            res = run_arm(arm, rep)
            res["teardown"] = TEARDOWN.get((arm, rep))
            # after stop_server, so the file is complete and closed
            _lp = OUT / str(res.get("server_log") or "")
            res["server_log_sha256"] = sha256(str(_lp)) if _lp.is_file() else None
            results.append(res)
            (OUT / f"{arm}__rep{rep}.json").write_text(
                json.dumps(res, indent=2, ensure_ascii=False) + "\n")

    (OUT / "all_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"\n=== wrote {OUT}/ ({len(results)} arm-runs) ===")
    # RUN_COMPLETE.json is the marker every downstream consumer trusts to mean
    # "this directory holds a whole run". It used to be written unconditionally
    # once the arm loop returned, so a run in which an arm crashed - run_arm
    # records the failure and continues - still produced the marker, and the
    # integrity checker read it as an attestation of completeness. Validate
    # first; on failure write RUN_FAILED.json instead, so a partial directory
    # announces itself rather than passing silently.
    problems = validate_run(OUT, arms, REPEATS, results)
    stamp = {
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "arms": arms, "repeats": REPEATS, "prompt_set": PROMPT_SET_NAME,
        "n_prompts": len(PROMPTS),
        "expected_arm_runs": len(arms) * REPEATS,
        "observed_arm_runs": len(results),
        "order_mode": ORDER_MODE,
        "schedule_is_position_balanced": manifest["schedule_is_position_balanced"],
    }
    if problems:
        stamp["problems"] = problems
        (OUT / "RUN_FAILED.json").write_text(
            json.dumps(stamp, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\n=== RUN FAILED: {len(problems)} problem(s) ===", flush=True)
        for x in problems:
            print(f"    - {x}", flush=True)
        sys.exit(1)
    (OUT / "RUN_COMPLETE.json").write_text(
        json.dumps(stamp, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
