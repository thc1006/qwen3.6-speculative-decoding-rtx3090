# ---------------------------------------------------------------------------
# Audited 2026-08-25. The interpretation string this script emits used to end
# "Workload-shape hypothesis ... is REFUTED for this hardware/engine". That
# conclusion is withdrawn and the string below is the rescoped replacement, so
# re-running this extractor no longer overwrites results.json with a retracted
# claim. The numbers it computes are unchanged and were reproduced exactly
# during the audit. See ../../ERRATA.md items D1, D2, D3.
# ---------------------------------------------------------------------------
"""Extract Exp 2 (code/JSON workload, N=3, standalone 3090, llama.cpp) bench
results from master.log into a clean per-prompt JSON for repo inclusion."""
import json, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "exp2_codejson_master.log")

PROMPTS_LABEL = {
    1: "ThreadSafeLRUCache (Python class, RLock)",
    2: "REST API JSON spec for user authentication",
    3: "Rust merge_sort<T: Ord + Clone> in-place",
    4: "Nginx reverse proxy JSON config",
    5: "PostgreSQL top-10 customers (last 30 days)",
}

with open(LOG, encoding="utf-8") as f:
    text = f.read()

# Pattern: '--- trial=N cfg=CFG n_extra=K ---' followed by 5×
# `[ Prompt: X t/s | Generation: Y t/s ]` lines (each preceded by `[CFG p=I]`).
trial_re = re.compile(r"--- trial=(\d+) cfg=(\S+) n_extra=\d+ ---")
gen_re = re.compile(r"\[\s*Prompt:\s+([\d.]+)\s+t/s\s+\|\s+Generation:\s+([\d.]+)\s+t/s\s*\]")

results = {}  # cfg -> list of (trial, prompt_idx, prompt_t_s, gen_t_s)
trial_blocks = re.split(r"--- trial=", text)
for block in trial_blocks[1:]:
    m = re.match(r"(\d+)\s+cfg=(\S+)", block)
    if not m:
        continue
    trial = int(m.group(1))
    cfg = m.group(2)
    matches = gen_re.findall(block)
    if cfg not in results:
        results[cfg] = []
    for i, (pt, gt) in enumerate(matches[:5], 1):
        results[cfg].append((trial, i, float(pt), float(gt)))

# Aggregate per config
agg = {}
for cfg, rows in results.items():
    decode_rates = [r[3] for r in rows]
    per_prompt = {p: [r[3] for r in rows if r[1] == p] for p in range(1, 6)}
    agg[cfg] = {
        "n_samples": len(decode_rates),
        "mean_tok_s": statistics.mean(decode_rates),
        "stdev_tok_s": statistics.stdev(decode_rates) if len(decode_rates) > 1 else 0.0,
        "min_tok_s": min(decode_rates),
        "max_tok_s": max(decode_rates),
        "per_prompt_mean": {p: statistics.mean(v) for p, v in per_prompt.items() if v},
        "raw": rows,
    }

baseline_mean = agg["01_baseline"]["mean_tok_s"]
for cfg in agg:
    delta = (agg[cfg]["mean_tok_s"] - baseline_mean) / baseline_mean * 100
    agg[cfg]["delta_vs_baseline_pct"] = round(delta, 2)

OUT = {
    "experiment": "Exp 2 — N=3 code/JSON workload only (workload-dependency probe)",
    "host": "standalone 3090 (tailscale-3090, 100.112.135.98)",
    "gpu": "1x RTX 3090 24GB, no power limit",
    "engine": "llama.cpp build 8889 (bcb5eeb64), post PR #22227 speculative-simple checkpoint support",
    "main_model": "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf (~21 GB)",
    "draft_model": "Qwen3.5-0.8B-Q4_K_M.gguf (~508 MB, vocab-SIZE-matched only; llama.cpp's compatibility gate REJECTS the pair and falls back to token translation - see ERRATA A2)",
    "common_args": "-ngl 999 -c 16384 -fa on -ctk q8_0 -ctv q8_0 -n 200 --temp 0.5 --seed 42 -no-cnv -st",
    "prompts": PROMPTS_LABEL,
    "n_trials": 3,
    "n_prompts": 5,
    "timestamp": "2026-04-26T00:01:41Z",
    "configs": agg,
    "interpretation": (
        "Tests whether structured/low-entropy prompts (code & JSON) flip "
        "spec-decode positive on consumer Ampere — they do not. baseline "
        f"{baseline_mean:.1f} tok/s, oleg --draft-min 2 --draft-max 32 "
        f"{agg['02_oleg_draft_2_32']['mean_tok_s']:.1f} tok/s "
        f"({agg['02_oleg_draft_2_32']['delta_vs_baseline_pct']:+.1f}%), "
        f"srogmann --draft-min 48 --draft-max 64 "
        f"{agg['03_srogmann_draft_48_64']['mean_tok_s']:.1f} tok/s "
        f"({agg['03_srogmann_draft_48_64']['delta_vs_baseline_pct']:+.1f}%). "
        "EXPLORATORY (rescoped 2026-08-25): the intended treatment - direct, "
        "thinking-disabled, low-entropy code/JSON output - was NOT verified. "
        "-no-cnv is rejected by llama-cli on this build, /no_think did not "
        "disable thinking, and per-request outputs were not committed. This "
        "does not refute the low-entropy workload hypothesis; it is a "
        "command-level replication of the negative direction. See "
        "../../ERRATA.md items D1, D2, D3."
    ),
}

OUT_PATH = os.path.join(HERE, "exp2_codejson_results.json")
# Preserve any audit/errata block already present. Re-running the extractor
# must not silently drop the 2026-08-25 rescoping (ERRATA D3).
try:
    with open(OUT_PATH, encoding="utf-8") as f:
        _prev = json.load(f)
    for _k in ("audit_2026_08_25",):
        if _k in _prev:
            OUT[_k] = _prev[_k]
except (FileNotFoundError, ValueError):
    pass

with open(OUT_PATH, "w") as f:
    json.dump(OUT, f, indent=2)
print(f"Written: {OUT_PATH}")
print()
print("=" * 70)
print("Summary:")
print("=" * 70)
for cfg, d in agg.items():
    print(
        f"  [{cfg}] N={d['n_samples']} mean={d['mean_tok_s']:.2f} tok/s "
        f"stdev={d['stdev_tok_s']:.2f} delta={d['delta_vs_baseline_pct']:+.2f}%"
    )
print()
print("Per-prompt mean (oleg config; p=2 outlier visible):")
for p, v in agg["02_oleg_draft_2_32"]["per_prompt_mean"].items():
    print(f"  p{p} ({PROMPTS_LABEL[p][:40]}): {v:.2f} tok/s")
