"""Summarise a bench/retest_runner.py matrix run.

Reports, per arm: request-mean, pooled throughput, median, min-max, the
run-to-run SD of the per-repeat means (which is what a `+/-` may honestly be
called once N > 1), the real acceptance rate, and whether thinking was actually
suppressed.

Also runs three checks the archived experiments could not:

  drift      compares the first and last repeat of the no-speculation baseline.
             A multi-hour matrix can be biased by the GPU downclocking as it
             heats; repeating the baseline across the run makes that testable
             instead of arguable (ERRATA C4b).
  activation how many requests recorded a draft round, and how many drafts were
             generated versus accepted. `draft_n = 0` never meant "speculation
             did not run" on the historical builds (ERRATA A1).
  workload   how many requests actually had thinking suppressed, and the output
             length distribution. v1, v2, v3 and Exp 2 all believed they were
             measuring direct answers and were not (ERRATA A5, D2).

Run: python analysis/matrix_report.py <run-dir> [<run-dir> ...]
"""
from __future__ import annotations

import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else float("nan")


def load(run_dir: Path):
    arms = defaultdict(list)
    for f in sorted(glob.glob(str(run_dir / "*__rep*.json"))):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        arms[d["arm"]].append(d)
    man = {}
    mf = run_dir / "manifest.json"
    if mf.exists():
        man = json.loads(mf.read_text(encoding="utf-8"))
    return arms, man


def arm_stats(runs):
    rates, lens, rep_means = [], [], []
    n = ms = dn = da = 0
    complete = think_off = requests = 0
    for r in runs:
        rr = [x["predicted_per_second"] for x in r["rows"]]
        if rr:
            rep_means.append(st.mean(rr))
        if len(r["rows"]) == 10:
            complete += 1
        for x in r["rows"]:
            requests += 1
            rates.append(x["predicted_per_second"])
            lens.append(x["predicted_n"])
            n += x["predicted_n"]
            ms += x["predicted_ms"]
            dn += x["draft_n"]
            da += x["draft_n_accepted"]
            think_off += 1 if x.get("thinking_suppressed") else 0
    return {
        "reps": len(runs), "requests": requests, "complete": complete,
        "mean": st.mean(rates) if rates else float("nan"),
        "pooled": 1000 * n / ms if ms else float("nan"),
        "median": st.median(rates) if rates else float("nan"),
        "min": min(rates) if rates else float("nan"),
        "max": max(rates) if rates else float("nan"),
        # the only quantity here that is genuinely repeated-run uncertainty
        "rep_sd": st.stdev(rep_means) if len(rep_means) > 1 else 0.0,
        "drafted": dn, "accepted": da,
        "acc_pct": (100 * da / dn) if dn else None,
        "think_off": think_off,
        "len_min": min(lens) if lens else 0, "len_max": max(lens) if lens else 0,
        "crashes": [r["crashed"] for r in runs if r.get("crashed")],
    }


def report(run_dir: Path) -> None:
    arms, man = load(run_dir)
    if not arms:
        print(f"  no arm results in {run_dir}")
        return
    print("=" * 108)
    print(f"{run_dir.name}")
    if man:
        print(f"  binary {man.get('server_sha256','?')[:16]}...  flavor={man.get('flavor','?')}  "
              f"think={man.get('think','?')} (env {man.get('think_env','?')})  "
              f"repeats={man.get('repeats')}  max_tokens={man.get('max_tokens')}")
        print(f"  gpu at start: {man.get('nvidia_smi','?')}")
    print("=" * 108)

    stats = {a: arm_stats(r) for a, r in arms.items()}
    base = stats.get("baseline")
    hdr = (f"{'arm':22s} {'reps':>4s} {'req-mean':>9s} {'pooled':>8s} {'vs base':>8s} "
           f"{'median':>7s} {'min':>7s} {'repSD':>6s} {'drafted':>8s} {'acc%':>6s} "
           f"{'think-off':>9s} {'len':>9s} {'ok':>5s}")
    print(hdr)
    print("-" * len(hdr))
    for a in sorted(stats, key=lambda k: -stats[k]["pooled"]):
        s = stats[a]
        vs = (f"{100 * (s['pooled'] / base['pooled'] - 1):+7.1f}%"
              if base and base["pooled"] == base["pooled"] else "      -")
        acc = f"{s['acc_pct']:5.1f}" if s["acc_pct"] is not None else "    -"
        print(f"{a:22s} {s['reps']:4d} {s['mean']:9.1f} {s['pooled']:8.1f} {vs:>8s} "
              f"{s['median']:7.1f} {s['min']:7.1f} {s['rep_sd']:6.2f} {s['drafted']:8d} "
              f"{acc:>6s} {s['think_off']:4d}/{s['requests']:<4d} "
              f"{s['len_min']:4d}-{s['len_max']:<4d} {s['complete']}/{s['reps']}")
        for c in s["crashes"]:
            print(f"{'':24s}! died on {c['tag']}: {c['error'][:60]}")

    # ---- drift: does the baseline move across the run? ----------------------
    if base and base["reps"] > 1:
        bm = [st.mean([x["predicted_per_second"] for x in r["rows"]])
              for r in sorted(arms["baseline"], key=lambda r: r["repeat"]) if r["rows"]]
        print(f"\n  drift check - baseline per repeat: "
              f"{', '.join(f'{v:.1f}' for v in bm)} tok/s")
        if len(bm) > 1:
            swing = 100 * (max(bm) - min(bm)) / st.mean(bm)
            print(f"    first-to-last {100 * (bm[-1] / bm[0] - 1):+.2f} %, "
                  f"full swing {swing:.2f} % of the mean"
                  f"{'  <- negligible' if swing < 1.5 else '  <- INVESTIGATE'}")

    # ---- acceptance vs speed, the relationship the retracted chart hid ------
    pts = [(s["acc_pct"], s["pooled"]) for a, s in stats.items()
           if s["acc_pct"] is not None]
    if len(pts) >= 3:
        r = pearson([p[0] for p in pts], [p[1] for p in pts])
        print(f"\n  acceptance vs pooled throughput across {len(pts)} speculative arms: "
              f"Pearson r = {r:+.3f}")

    # ---- draft-length sweep -------------------------------------------------
    sweep = sorted(((int(a.rsplit("n", 1)[1]), stats[a]) for a in stats
                    if a.startswith("spec-draft-n") and a.rsplit("n", 1)[1].isdigit()),
                   key=lambda t: t[0])
    if len(sweep) >= 3:
        print("\n  draft-length sweep (n_min pinned to 1, matched vocabulary):")
        print(f"    {'n_max':>6s} {'pooled':>8s} {'vs base':>8s} {'drafted':>8s} {'acc%':>6s}")
        for nmax, s in sweep:
            vs = f"{100 * (s['pooled'] / base['pooled'] - 1):+7.1f}%" if base else "      -"
            print(f"    {nmax:6d} {s['pooled']:8.1f} {vs:>8s} {s['drafted']:8d} "
                  f"{s['acc_pct']:5.1f}" if s["acc_pct"] is not None else "")
    print()


def main() -> None:
    dirs = [Path(a) for a in sys.argv[1:]]
    if not dirs:
        sys.exit("usage: python analysis/matrix_report.py <run-dir> [<run-dir> ...]")
    for d in dirs:
        report(d)


if __name__ == "__main__":
    main()
