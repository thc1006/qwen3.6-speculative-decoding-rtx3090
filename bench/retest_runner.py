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
    BENCH_THINK        on | off  - request-level thinking control (default on)
                       `off` sends chat_template_kwargs {"enable_thinking": false}.
                       Whether it took effect is VERIFIED per request and recorded
                       as `thinking_suppressed`; never assume it worked - /no_think
                       in the prompt text never did (ERRATA D2).

Run: python bench/retest_runner.py
"""
from __future__ import annotations

import hashlib
import json
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
GPU = os.environ.get("BENCH_GPU", "0")
PORT = int(os.environ.get("BENCH_PORT", "18131"))
OUT = Path(os.environ.get("BENCH_OUT", "retest_out"))
REPEATS = int(os.environ.get("BENCH_REPEATS", "3"))
MAX_TOKENS = int(os.environ.get("BENCH_MAX_TOKENS", "300"))
# Accept the obvious spellings. A silent mismatch here would put thinking back
# on without saying so, which is exactly the failure mode that made v2, v3 and
# Exp 2 unauditable (ERRATA D2) - so normalise, and record the result in the
# manifest and per request.
_THINK_RAW = os.environ.get("BENCH_THINK", "on").strip().lower()
THINK = "off" if _THINK_RAW in ("off", "0", "false", "no", "think_off",
                                "disabled", "none") else "on"

# v1's fixed server flags, kept verbatim so the retest stays comparable.
COMMON_ARGS = [
    "-ngl", "999", "-c", "16384", "--jinja",
    "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
    "--no-webui", "-v",
]

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
    "spec-dflash-n8": ["-md", "{DFLASH}", "-ngld", "99", _NMAX, "8", _NMIN, "1",
                       "--spec-type", "draft-dflash"],
    "spec-dflash-n4": ["-md", "{DFLASH}", "-ngld", "99", _NMAX, "4", _NMIN, "1",
                       "--spec-type", "draft-dflash"],
    "spec-dflash-n16": ["-md", "{DFLASH}", "-ngld", "99", _NMAX, "16", _NMIN, "1",
                        "--spec-type", "draft-dflash"],

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
}


# ---------------------------------------------------------------- utils ----

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


def wait_health(port: int, timeout: float = 300.0) -> float:
    t0 = time.perf_counter()
    url = f"http://127.0.0.1:{port}/health"
    while time.perf_counter() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return time.perf_counter() - t0
        except Exception:  # noqa: BLE001
            pass
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
    cmd += common + [a.replace("{DRAFT}", DRAFT).replace("{DFLASH}", DFLASH)
                     for a in extra]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd, env=env, stdout=log_path.open("w"),
                            stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    proc._cmd = cmd  # type: ignore[attr-defined]
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:  # noqa: BLE001
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass


def chat(system: str, user: str) -> dict:
    body = {
        "model": "retest",
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "seed": 42,
        "stream": False,
        # token ids arrive through logprobs on this endpoint - see module docstring
        "logprobs": True,
    }
    if THINK == "off":
        body["chat_template_kwargs"] = {"enable_thinking": False}
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
            ready_s = wait_health(PORT)
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
            chat("You are concise.", "Warm up with a few sentences about the weather.")
        except Exception as e:  # noqa: BLE001
            return {"arm": arm, "repeat": rep, "ready_s": ready_s,
                    "argv": proc._cmd,  # type: ignore[attr-defined]
                    "gpu_before": gpu_before, "gpu_after": nvidia_smi(),
                    "server_log": str(log_path.relative_to(OUT)),
                    "crashed": {"tag": "__warmup__", "error": f"{type(e).__name__}: {e}",
                                "returncode": proc.poll()},
                    "rows": []}
        rows = []
        for tag, sysmsg, usermsg in PROMPTS:
            try:
                r = chat(sysmsg, usermsg)
            except Exception as e:  # noqa: BLE001
                # A server death is a finding, not a harness failure: record
                # where it happened and move on to the next arm.
                crashed = {"tag": tag, "error": f"{type(e).__name__}: {e}",
                           "returncode": proc.poll()}
                print(f"    [{arm} rep{rep} {tag:13s}] SERVER DIED: {type(e).__name__}",
                      flush=True)
                break
            r["tag"] = tag
            rows.append(r)
            acc = ""
            if r["draft_n"]:
                acc = f"  counted-draft {r['draft_n_accepted']}/{r['draft_n']}"
            think = "" if r["thinking_suppressed"] else "  THINKING"
            print(f"    [{arm} rep{rep} {tag:13s}] {r['predicted_n']:>4d}tok "
                  f"@ {r['predicted_per_second']:6.1f} tok/s{acc}{think}", flush=True)
        return {
            "arm": arm, "repeat": rep, "ready_s": ready_s,
            "argv": proc._cmd,  # type: ignore[attr-defined]
            "gpu_before": gpu_before, "gpu_after": nvidia_smi(),
            "server_log": str(log_path.relative_to(OUT)),
            "crashed": crashed,
            "rows": rows,
        }
    finally:
        stop_server(proc)


def main() -> None:
    missing = [n for n, v in (("LLAMA_SERVER_BIN", SERVER), ("MODEL_TARGET", TARGET)) if not v]
    if missing:
        sys.exit(f"set {', '.join(missing)}")
    wanted = os.environ.get("BENCH_ARMS")
    arms = [a.strip() for a in wanted.split(",")] if wanted else list(ARMS)
    for a in arms:
        if a not in ARMS:
            sys.exit(f"unknown arm {a!r}; known: {', '.join(ARMS)}")
    if any("{DRAFT}" in x for a in arms for x in ARMS[a]) and not DRAFT:
        sys.exit("set MODEL_DRAFT for the draft arms")
    if any("{DFLASH}" in x for a in arms for x in ARMS[a]) and not DFLASH:
        sys.exit("set MODEL_DFLASH for the DFlash arms")

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": os.uname().nodename,
        "gpu_env": GPU,
        "server_bin": SERVER,
        "server_sha256": sha256(SERVER),
        "target": TARGET, "target_sha256": sha256(TARGET),
        "draft": DRAFT or None,
        "draft_sha256": sha256(DRAFT) if DRAFT else None,
        "dflash": DFLASH or None,
        "dflash_sha256": sha256(DFLASH) if DFLASH else None,
        "common_args": COMMON_ARGS,
        "flavor": FLAVOR,
        "arms": {a: ARMS[a] for a in arms},
        "repeats": REPEATS, "max_tokens": MAX_TOKENS,
        "temperature": 0.0, "seed": 42, "think": THINK, "think_env": _THINK_RAW,
        "ordering": "ABBA: arm order is reversed on odd repeats",
        "gpu_fields": GPU_FIELDS,
        "nvidia_smi": nvidia_smi(),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "arms"}, indent=2))

    results = []
    for rep in range(REPEATS):
        # ABBA: reverse the arm order on odd repeats so any monotone drift
        # (thermal, clock, cache) cannot alias onto the arm contrast.
        order = arms if rep % 2 == 0 else list(reversed(arms))
        print(f"\n=== repeat {rep}  order: {' -> '.join(order)} ===", flush=True)
        for arm in order:
            res = run_arm(arm, rep)
            results.append(res)
            (OUT / f"{arm}__rep{rep}.json").write_text(
                json.dumps(res, indent=2, ensure_ascii=False) + "\n")

    (OUT / "all_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"\n=== wrote {OUT}/ ({len(results)} arm-runs) ===")


if __name__ == "__main__":
    main()
