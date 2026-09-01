"""Spec-decode bench runner for Qwen3.6-35B-A3B on RTX 3090 (v1, HISTORICAL).

HISTORICAL SCRIPT - kept as evidence. Audited 2026-08-25, see ERRATA.md.
Known defects, left in place so this file still documents what v1 ran:

  * `content_head` keeps only the first 120 characters of `message.content`
    and discards `message.reasoning_content` entirely. 144 of the 190 v1
    requests (75.8 %) have an EMPTY content field, meaning the 300-token cap
    was hit inside the thinking block - v1 measured truncated chain-of-thought
    output, not answers. `reasoning`, `code_small`: 19/19. (ERRATA A5)
  * token IDs, stop reason, and the full `timings` block are not stored.
  * `draft_n` / `draft_n_accepted` count only fully accepted verification
    rounds, so their ratio is 1.0 by construction on this model. `draft_n = 0`
    means "no fully accepted round", not "speculation did not run". (ERRATA A1)
  * the warm-up is a single 8-token completion, not a full-shape warm-up.
  * `--gpu 1` was chosen because GPU 0 ran an Ollama instance. (ERRATA C4)
  * the `zh_cn` prompt is Traditional Chinese. (ERRATA C2)
  * `multi_turn_1` / `multi_turn_2` are two independent single-turn requests,
    and `medium_rec` refers to a turn that never happened. (ERRATA C3)

Use bench/retest_runner.py for new measurements.

Spawns a llama-server subprocess with given spec-decode config, sends a fixed
prompt set through OpenAI-compat /v1/chat/completions, records timings.

Usage:
    python bench_runner.py --config baseline --output results/baseline.json
    python bench_runner.py --config ngram-mod-24 --output results/ngram.json

Each config line in configs.json defines:
    {"name": "...", "server_args": ["--spec-type","ngram-mod","--spec-ngram-size-n","24",...]}

Emits one JSON per run with: config, per-prompt timings, server startup time,
llama.cpp commit, hardware, CUDA version.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
LLAMA_SERVER = Path(os.environ.get(
    "LLAMA_SERVER_BIN", Path.home() / "benchmarks/llama.cpp/build/bin/llama-server"))
MODEL_PATH = Path(os.environ.get(
    "MODEL_TARGET",
    Path.home() / "benchmarks/models/qwen3.6-ud-q4kxl/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"))
LLAMA_CPP_REPO = Path(os.environ.get(
    "LLAMA_CPP_REPO", Path.home() / "benchmarks/llama.cpp"))
DEFAULT_PORT = int(os.environ.get("BENCH_PORT", "18123"))


PROMPTS = [
    # (tag, system, user)
    ("short_greet",    "You are a friendly desk robot.", "Hey there!"),
    ("short_q",        "You are a friendly desk robot.", "How's your day going?"),
    ("medium_chat",    "You are a friendly desk robot.", "What do you think about humans writing code for you?"),
    ("medium_rec",     "You are a friendly desk robot.", "Earlier the user said their name is Hctsai. What did they tell you?"),
    ("reasoning",      "You solve problems step by step.", "If a train leaves Paris at 2pm going 120km/h and another leaves Berlin at 3pm going 80km/h on a 1000km track towards each other, where do they meet? Think briefly step by step."),
    ("long_explain",   "You teach patiently.", "Explain to a curious 10-year-old what makes a rainbow form, in 4 to 6 sentences."),
    ("code_small",     "You write Python.", "Write a short Python function that takes a list of ints and returns only the primes, with a quick comment."),
    ("multi_turn_1",   "You are a friendly desk robot.", "Let's play a game. Pick a random number between 1 and 100 and tell me."),
    ("multi_turn_2",   "You are a friendly desk robot.", "What is your favorite kind of music and why? Give two concrete examples."),
    ("zh_cn",          "你是桌面機器人。", "請用一到兩句話介紹你自己。"),
]


def wait_health(port: int, timeout: float = 180.0) -> float:
    t0 = time.perf_counter()
    url = f"http://127.0.0.1:{port}/health"
    while time.perf_counter() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return time.perf_counter() - t0
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"llama-server did not become ready in {timeout}s")


def start_server(extra_args: list[str], port: int, gpu: int | None = None,
                 fa: bool = True, kv_q8: bool = True) -> subprocess.Popen:
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        str(LLAMA_SERVER),
        "-m", str(MODEL_PATH),
        "--host", "127.0.0.1",
        "--port", str(port),
        "-ngl", "999",
        "-c", "16384",
        "--jinja",
        "-fa", "on" if fa else "off",
        "--no-webui",
    ]
    if kv_q8:
        cmd += ["-ctk", "q8_0", "-ctv", "q8_0"]
    cmd += extra_args
    print(f"\n  server cmd: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, env=env,
                             stdout=open(f"/tmp/llama_server_{port}.log", "w"),
                             stderr=subprocess.STDOUT,
                             preexec_fn=os.setsid)
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def chat(port: int, system: str, user: str, max_tokens: int = 300) -> dict:  # default kept for back-compat
    body = {
        "model": "qwen3.6",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,   # greedy — spec-decode 最強情境，對比才乾淨
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read())
    wall_ms = (time.perf_counter() - t0) * 1000
    timings = data.get("timings", {}) or {}
    usage = data.get("usage", {}) or {}
    choice = data["choices"][0]
    content = choice["message"].get("content", "") or ""
    return {
        "wall_ms":          wall_ms,
        "prompt_tokens":    usage.get("prompt_tokens", 0),
        "completion_tokens":usage.get("completion_tokens", 0),
        # llama.cpp-specific fields (OpenAI-compat returns them in "timings")
        "predicted_ms":     timings.get("predicted_ms", 0),
        "predicted_n":      timings.get("predicted_n", 0),
        "predicted_per_second": timings.get("predicted_per_second", 0),
        "prompt_ms":        timings.get("prompt_ms", 0),
        "prompt_per_second":timings.get("prompt_per_second", 0),
        # spec-decode stats (if available)
        "draft_n":          timings.get("draft_n", 0),
        "draft_n_accepted": timings.get("draft_n_accepted", 0),
        "content_head":     content[:120],
    }


def run_config(config_name: str, server_args: list[str], port: int, gpu: int | None,
               warmup: int, fa: bool = True, kv_q8: bool = True,
               max_tokens: int = 300) -> dict:
    proc = start_server(server_args, port, gpu, fa=fa, kv_q8=kv_q8)
    try:
        ready_s = wait_health(port, timeout=240)
        print(f"  server ready in {ready_s:.1f}s", flush=True)
        for i in range(warmup):
            chat(port, "You are concise.", f"hi {i}", max_tokens=8)
        rows = []
        for tag, sys_msg, user_msg in PROMPTS:
            r = chat(port, sys_msg, user_msg, max_tokens=max_tokens)
            r["tag"] = tag
            rows.append(r)
            acc = ""
            if r.get("draft_n", 0) > 0:
                acc = f"  draft_accept {r['draft_n_accepted']}/{r['draft_n']}={100*r['draft_n_accepted']/r['draft_n']:.0f}%"
            print(f"  [{tag:14s}] {r['predicted_n']:>3d}tok @ {r['predicted_per_second']:6.1f} tok/s  wall {r['wall_ms']:>5.0f}ms{acc}", flush=True)
        return {
            "config": config_name,
            "server_args": server_args,
            "ready_s": ready_s,
            "rows": rows,
        }
    finally:
        stop_server(proc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    # Quoted string: --server-args "--spec-type ngram-mod --spec-ngram-size-n 24"
    ap.add_argument("--server-args", type=str, default="",
                    help="Extra llama-server flags as a single quoted string.")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--gpu", type=int, default=int(os.environ.get("BENCH_GPU", "1")),
                    help="CUDA_VISIBLE_DEVICES. v1 used 1 because GPU0 ran Ollama (ERRATA C4)")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=300, help="max_tokens per completion")
    ap.add_argument("--no-fa", action="store_true", help="disable flash-attn (-fa off)")
    ap.add_argument("--kv-fp16", action="store_true", help="fp16 KV cache (no q8_0 quant)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect env metadata for reproducibility
    try:
        commit = subprocess.check_output(["git", "-C", str(LLAMA_CPP_REPO),
                                          "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    try:
        nvsmi = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                          "--format=csv,noheader"], text=True).strip()
    except Exception:
        nvsmi = "unknown"

    print(f"=== bench config: {args.config} ===")
    print(f"llama.cpp commit: {commit}")
    print(f"GPU(s):\n  {nvsmi}")
    import shlex as _shlex
    extra_args = _shlex.split(args.server_args) if args.server_args else []
    result = run_config(
        args.config, extra_args, args.port, args.gpu, args.warmup,
        fa=not args.no_fa, kv_q8=not args.kv_fp16, max_tokens=args.max_tokens,
    )
    result["meta"] = {"fa": not args.no_fa, "kv_q8": not args.kv_fp16, "max_tokens": args.max_tokens}
    result["llama_cpp_commit"] = commit
    result["gpu_info"] = nvsmi
    result["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n=== wrote {out_path} ===")

    # Aggregate print
    toks = [r["predicted_per_second"] for r in result["rows"] if r["predicted_per_second"] > 0]
    if toks:
        print(f"  mean decode  : {sum(toks)/len(toks):.1f} tok/s")
        print(f"  min / max    : {min(toks):.1f} / {max(toks):.1f}")
    accepts = [r for r in result["rows"] if r.get("draft_n", 0) > 0]
    if accepts:
        total_d = sum(r["draft_n"] for r in accepts)
        total_a = sum(r["draft_n_accepted"] for r in accepts)
        print(f"  draft accept : {100*total_a/total_d:.1f}%  ({total_a}/{total_d})")


if __name__ == "__main__":
    main()
