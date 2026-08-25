"""Parse v2 per-prompt llama-cli logs into a single machine-readable JSON.

Run from the v2_3090_followup/ directory:
    python extract_results.py

Produces:
    results_v2.json — all runs + per-config summary (mean / min / max / std)
"""
import json
import pathlib
import re
import statistics

HERE = pathlib.Path(__file__).resolve().parent

PROMPT_RE = re.compile(
    r"\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]"
)

PROMPTS = [
    "Why does the sky look blue? Answer in two sentences. /no_think",
    "Write a Python function fib(n) returning the first n Fibonacci numbers as a list. /no_think",
    "Explain TCP vs UDP in 3 concise bullet points. /no_think",
    "Give 5 numbered steps to cook firm tofu at home. /no_think",
    "Write a short haiku about debugging a memory leak at 2am. /no_think",
]

COMMON_FLAGS = [
    "-ngl", "999", "-c", "16384", "-fa", "on",
    "-ctk", "q8_0", "-ctv", "q8_0",
    "-n", "200", "--temp", "0.5", "--seed", "42",
    "-no-cnv", "-st",
]

CONFIGS_ORIGINAL_COMMIT = {
    "baseline": {
        "dir": "v2_oleg_suggestions/01_baseline",
        "extra_args": [],
        "note": "no spec-decode",
    },
    "oleg_draft_2_32": {
        "dir": "v2_oleg_suggestions/02_oleg_draft_2_32",
        "extra_args": ["-md", "Qwen3.5-0.8B-Q4_K_M.gguf",
                       "--draft-min", "2", "--draft-max", "32"],
        "note": "Oleg-dM's HF-discussion suggestion",
    },
    "oleg_draft_2_16": {
        "dir": "v2_oleg_suggestions/03_oleg_draft_2_16",
        "extra_args": ["-md", "Qwen3.5-0.8B-Q4_K_M.gguf",
                       "--draft-min", "2", "--draft-max", "16"],
        "note": "Oleg-style tighter upper bound",
    },
    "oleg_draft_2_64": {
        "dir": "v2_oleg_suggestions/04_draft_2_64",
        "extra_args": ["-md", "Qwen3.5-0.8B-Q4_K_M.gguf",
                       "--draft-min", "2", "--draft-max", "64"],
        "note": "Oleg-style wider upper bound",
    },
    "default_draft_max8": {
        "dir": "v2_controls/A_default_draft_max8",
        "extra_args": ["-md", "Qwen3.5-0.8B-Q4_K_M.gguf", "--draft-max", "8"],
        "note": "default --draft-min=5",
    },
    "default_draft_max16": {
        "dir": "v2_controls/B_default_draft_max16",
        "extra_args": ["-md", "Qwen3.5-0.8B-Q4_K_M.gguf", "--draft-max", "16"],
        "note": "default --draft-min=5",
    },
    "default_draft_max32": {
        "dir": "v2_controls/C_default_draft_max32",
        "extra_args": ["-md", "Qwen3.5-0.8B-Q4_K_M.gguf", "--draft-max", "32"],
        "note": "default --draft-min=5",
    },
    "srogmann_min48_max64": {
        "dir": "v2_controls/D_srogmann_min48_max64",
        "extra_args": ["-md", "Qwen3.5-0.8B-Q4_K_M.gguf",
                       "--draft-min", "48", "--draft-max", "64"],
        "note": "srogmann-style aggressive",
    },
    "bare_md": {
        "dir": "v2_controls/E_bare_md",
        "extra_args": ["-md", "Qwen3.5-0.8B-Q4_K_M.gguf"],
        "note": "bare -md, all defaults",
    },
}

CONFIGS_MASTER_COMMIT = {
    "baseline": {
        "dir": "v2_master_cross_check/M1_baseline",
        "extra_args": [],
        "note": "master baseline",
    },
    "oleg_draft_2_32": {
        "dir": "v2_master_cross_check/M2_oleg_draft_2_32",
        "extra_args": ["-md", "Qwen3.5-0.8B-Q4_K_M.gguf",
                       "--draft-min", "2", "--draft-max", "32"],
        "note": "master, Oleg's suggestion",
    },
    "srogmann_min48_max64": {
        "dir": "v2_master_cross_check/M3_srogmann_48_64",
        "extra_args": ["-md", "Qwen3.5-0.8B-Q4_K_M.gguf",
                       "--draft-min", "48", "--draft-max", "64"],
        "note": "master, srogmann-style",
    },
}


def parse_config_dir(dirpath: pathlib.Path):
    runs = []
    for i in range(1, 6):
        logf = dirpath / f"p{i}.log"
        if not logf.exists():
            runs.append({"prompt_idx": i, "error": "log missing"})
            continue
        txt = logf.read_text(encoding="utf-8", errors="replace")
        m = list(PROMPT_RE.finditer(txt))
        if not m:
            runs.append({"prompt_idx": i, "error": "no stats line"})
            continue
        last = m[-1]
        runs.append({
            "prompt_idx": i,
            "prompt_t_per_s": float(last.group(1)),
            "gen_t_per_s": float(last.group(2)),
        })
    gens = [r["gen_t_per_s"] for r in runs if "gen_t_per_s" in r]
    summary = {"n_runs": len(gens)}
    if gens:
        summary["mean_gen"] = round(statistics.mean(gens), 2)
        summary["min_gen"] = round(min(gens), 2)
        summary["max_gen"] = round(max(gens), 2)
        summary["std_gen"] = (
            round(statistics.stdev(gens), 2) if len(gens) > 1 else 0.0
        )
    return runs, summary


def build():
    data = {
        "bench_version": "v2",
        "date": "2026-04-22",
        "hardware": {
            "gpu": "NVIDIA GeForce RTX 3090",
            "memory_mib": 24576,
            "driver": "580.126.09",
            "cuda": "12.0.140",
            "os": "Ubuntu 24.04",
            "gcc": "13.3.0",
            "clocks_note": (
                "stock (no OC): graphics 1965 MHz / max 2100, "
                "memory 9751 MHz, power 350 W"
            ),
        },
        "models": {
            "main":  "unsloth/Qwen3.6-35B-A3B-GGUF :: Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
            "draft": "unsloth/Qwen3.5-0.8B-GGUF :: Qwen3.5-0.8B-Q4_K_M.gguf",
        },
        "commits": [
            {
                "name": "original",
                "sha": "97895129e5f2bde94d13dc01ca41ee79e9b629f2",
                "short": "97895129e",
                "context": "post PR #19493 merge (same commit as v1.0's 9789512 short hash)",
            },
            {
                "name": "master",
                "sha": "bcb5eeb64",
                "short": "bcb5eeb64",
                "context": "post PR #22227 'speculative-simple: add checkpoint support'",
            },
        ],
        "prompts": PROMPTS,
        "common_flags": COMMON_FLAGS,
        "runs_original_commit": {},
        "runs_master_commit": {},
    }
    for k, v in CONFIGS_ORIGINAL_COMMIT.items():
        runs, summary = parse_config_dir(HERE / v["dir"])
        data["runs_original_commit"][k] = {
            **v, "runs": runs, "summary": summary,
        }
    for k, v in CONFIGS_MASTER_COMMIT.items():
        runs, summary = parse_config_dir(HERE / v["dir"])
        data["runs_master_commit"][k] = {
            **v, "runs": runs, "summary": summary,
        }
    return data


def main():
    data = build()
    out = HERE / "results_v2.json"
    # Preserve the 2026-08-25 audit block. Re-running this extractor must not
    # silently drop the corrections, the same failure mode extract_exp2.py had.
    try:
        prev = json.loads(out.read_text(encoding="utf-8"))
        for k in ("audit_2026_08_25",):
            if k in prev:
                data[k] = prev[k]
    except (FileNotFoundError, ValueError):
        pass
    out.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out} ({out.stat().st_size} bytes)\n")
    print("=== ORIGINAL COMMIT ({}) ===".format(data["commits"][0]["short"]))
    for k, v in data["runs_original_commit"].items():
        s = v["summary"]
        if s.get("n_runs", 0) == 0:
            print(f"  {k:28s}  NO DATA")
            continue
        print(
            f"  {k:28s}  mean={s['mean_gen']:7.2f}"
            f"  min={s['min_gen']:7.2f}"
            f"  max={s['max_gen']:7.2f}"
            f"  std={s['std_gen']:.2f}"
        )
    print("\n=== MASTER COMMIT ({}) ===".format(data["commits"][1]["short"]))
    for k, v in data["runs_master_commit"].items():
        s = v["summary"]
        if s.get("n_runs", 0) == 0:
            print(f"  {k:28s}  NO DATA")
            continue
        print(
            f"  {k:28s}  mean={s['mean_gen']:7.2f}"
            f"  min={s['min_gen']:7.2f}"
            f"  max={s['max_gen']:7.2f}"
            f"  std={s['std_gen']:.2f}"
        )


if __name__ == "__main__":
    main()
