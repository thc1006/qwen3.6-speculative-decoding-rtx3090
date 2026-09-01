"""Is the comparison between arms a comparison of the same amount of work?

Pooled decode rate is tokens over decode-milliseconds, which is the right metric
when the arms generate the same number of tokens and a confounded one when they
do not. With thinking ON every request in this repository hits the 300-token cap
and the question does not arise. With thinking OFF it does: speculation is not
output-preserving on this build (ERRATA A11), so the arms stop at different
points, and in run R the baseline generates 300 tokens on `code_bash` where the
speculative arms generate 187, 188 and 188, and 203 on `code_rust` where all
three generate 300.

This computes, for every run, each arm's pooled change against the baseline
twice: over all prompts, and over only those prompts where every arm in the run
generated exactly the same number of tokens. Neither is "the" answer - the
restricted set is biased toward prompts long enough that every arm hit the cap -
but the gap between them is the size of the confound, and it was not reported.

Writes analysis/length_matching.json.

Run: python analysis/length_matching.py
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "v4_audit_2026_08_25" / "data"


def analyse(run_dir: Path) -> dict | None:
    man_path = run_dir / "manifest.json"
    if not man_path.is_file():
        return None
    man = json.loads(man_path.read_text(encoding="utf-8"))
    if "baseline" not in (man.get("arms") or {}):
        return None
    rows: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(str(run_dir / "*__rep*.json")):
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        if r.get("crashed"):
            continue
        for x in r.get("rows") or []:
            rows[x["tag"]][r["arm"]].append(
                (x["predicted_n"], x["predicted_ms"], x["draft_n"], x["draft_n_accepted"]))
    arms = [a for a in man["arms"] if a != "baseline"]
    present = [t for t, d in rows.items()
               if "baseline" in d and all(a in d for a in arms)]
    if not present or not arms:
        return None
    matched = [t for t in present
               if len({r[0] for a in rows[t] for r in rows[t][a]}) == 1]

    def pooled(tags: list[str], arm: str) -> float | None:
        n = sum(r[0] for t in tags for r in rows[t][arm])
        ms = sum(r[1] for t in tags for r in rows[t][arm])
        return 1000 * n / ms if ms else None

    def acceptance(tags: list[str], arm: str) -> float | None:
        """Acceptance is a ratio, so it looks length-independent, and is not:
        it varies along the sequence, and a short generation is all early
        tokens. Reported both ways for the same reason throughput is."""
        d = sum(r[2] for t in tags for r in rows[t][arm])
        a = sum(r[3] for t in tags for r in rows[t][arm])
        return round(100 * a / d, 2) if d else None

    out = {
        "run": run_dir.name,
        "think": man.get("think"),
        "ignore_eos": man.get("ignore_eos", False),
        "prompts": len(present),
        "length_matched_prompts": len(matched),
        "arms": {},
    }
    for a in arms:
        pa, pb = pooled(present, a), pooled(present, "baseline")
        if not pa or not pb:
            continue
        rec = {"all_prompts_pct": round(100 * (pa / pb - 1), 2),
               "acceptance_all_prompts": acceptance(present, a)}
        if matched:
            qa, qb = pooled(matched, a), pooled(matched, "baseline")
            if qa and qb:
                rec["length_matched_pct"] = round(100 * (qa / qb - 1), 2)
                rec["shift_pp"] = round(rec["length_matched_pct"]
                                        - rec["all_prompts_pct"], 2)
            rec["acceptance_length_matched"] = acceptance(matched, a)
        out["arms"][a] = rec
    return out


def main() -> None:
    runs = []
    for d in sorted(DATA.iterdir()):
        if not d.is_dir():
            continue
        rec = analyse(d)
        if rec:
            runs.append(rec)
    dest = ROOT / "analysis" / "length_matching.json"
    dest.write_text(json.dumps({"runs": runs}, indent=2) + "\n", encoding="utf-8")

    print(f"  {'run':<40s} {'think':>6s} {'prompts':>8s} {'matched':>8s} {'max |shift| pp':>15s}")
    for r in runs:
        shifts = [abs(v["shift_pp"]) for v in r["arms"].values() if "shift_pp" in v]
        print(f"  {r['run']:<40s} {str(r['think']):>6s} {r['prompts']:>8d} "
              f"{r['length_matched_prompts']:>8d} "
              f"{(max(shifts) if shifts else 0):>15.2f}")
    print(f"\n  wrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
