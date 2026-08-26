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


def arm_stats(runs, n_prompts):
    rates, lens, rep_means = [], [], []
    n = ms = dn = da = 0
    complete = think_off = think_known = requests = 0
    for r in runs:
        rr = [x["predicted_per_second"] for x in r["rows"]]
        # Only a repeat that ran the whole prompt set contributes to the
        # run-to-run SD: a repeat cut short by a crash has a different prompt
        # mix, so its mean is not comparable with a complete one.
        if len(r["rows"]) == n_prompts:
            complete += 1
            if rr:
                rep_means.append(st.mean(rr))
        for x in r["rows"]:
            requests += 1
            rates.append(x["predicted_per_second"])
            lens.append(x["predicted_n"])
            n += x["predicted_n"]
            ms += x["predicted_ms"]
            dn += x["draft_n"]
            da += x["draft_n_accepted"]
            if "thinking_suppressed" in x:
                think_known += 1
                think_off += 1 if x["thinking_suppressed"] else 0
    return {
        "reps": len(runs), "requests": requests, "complete": complete,
        "tokens": n, "think_known": think_known,
        "mean": st.mean(rates) if rates else float("nan"),
        "pooled": 1000 * n / ms if ms else float("nan"),
        "median": st.median(rates) if rates else float("nan"),
        "min": min(rates) if rates else float("nan"),
        "max": max(rates) if rates else float("nan"),
        # The only quantity here that is genuinely repeated-run uncertainty,
        # and only over COMPLETE repeats. None rather than 0.0 when fewer than
        # two are available: a zero would read as perfect reproducibility when
        # it actually means "not measurable", which is the class of misleading
        # figure this whole audit is about.
        "rep_sd": st.stdev(rep_means) if len(rep_means) > 1 else None,
        "rep_sd_n": len(rep_means),
        "drafted": dn, "accepted": da,
        "acc_pct": (100 * da / dn) if dn else None,
        "think_off": think_off,
        "len_min": min(lens) if lens else 0, "len_max": max(lens) if lens else 0,
        "crashes": [r["crashed"] for r in runs if r.get("crashed")],
    }


STRICT = False
FAILED: list[str] = []


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

    # Completeness must come from the manifest, not from the data. Inferring it
    # as "the largest row count seen here" marks every arm complete when every
    # arm is truncated the same way, which is exactly the case that matters.
    n_prompts = man.get("n_prompts")
    expected_tags = set(man.get("prompt_tags") or [])
    if n_prompts is None:
        n_prompts = max((len(r["rows"]) for runs in arms.values() for r in runs), default=0)
        print(f"  ! manifest carries no n_prompts; falling back to the largest row "
              f"count seen ({n_prompts}). Runs written before that field existed "
              f"cannot be checked for uniform truncation.")
    else:
        print(f"  prompt set size from the manifest: {n_prompts}"
              + (f" ({man.get('prompt_set')})" if man.get("prompt_set") else ""))
    if not (run_dir / "RUN_COMPLETE.json").exists():
        print("  ! no RUN_COMPLETE.json: this directory may hold a run that was "
              "interrupted, or files from more than one run")
    if STRICT:
        bad = []
        for a, runs in arms.items():
            for r in runs:
                tags = [x["tag"] for x in r["rows"]]
                if r.get("crashed"):
                    bad.append(f"{a} rep{r['repeat']}: crashed at {r['crashed']['tag']}")
                elif len(tags) != n_prompts:
                    bad.append(f"{a} rep{r['repeat']}: {len(tags)} rows, expected {n_prompts}")
                elif len(set(tags)) != len(tags):
                    bad.append(f"{a} rep{r['repeat']}: duplicate prompt tags")
                elif expected_tags and set(tags) != expected_tags:
                    bad.append(f"{a} rep{r['repeat']}: tag set differs from the manifest")
        if not (run_dir / "RUN_COMPLETE.json").exists():
            bad.append("no RUN_COMPLETE.json")
        # Which arm-runs are PRESENT, not only whether the present ones are
        # whole. Deleting a whole arm-run passed this check, and so did renaming
        # the arm inside a file: the report aggregated whatever it found.
        declared = set(man.get("arms") or {})
        reps = man.get("repeats")
        if declared and isinstance(reps, int) and reps > 0:
            want = {(a, r) for a in declared for r in range(reps)}
            have: dict = {}
            for a, runs in arms.items():
                for r in runs:
                    have[(a, r["repeat"])] = have.get((a, r["repeat"]), 0) + 1
            for a, r in sorted(want - set(have)):
                bad.append(f"{a} rep{r}: missing")
            for a, r in sorted(set(have) - want):
                bad.append(f"{a} rep{r}: not in the manifest's arm list")
            for k, n in sorted(have.items()):
                if n > 1:
                    bad.append(f"{k[0]} rep{k[1]}: {n} files claim this arm-run")
        # and the name on the file has to be the name inside it
        for f in sorted(run_dir.glob("*__rep*.json")):
            stem = f.name[:-len(".json")]
            f_arm, _, f_rep = stem.rpartition("__rep")
            d = json.loads(f.read_text(encoding="utf-8"))
            if d.get("arm") != f_arm or str(d.get("repeat")) != f_rep:
                bad.append(f"{f.name}: contains arm={d.get('arm')!r} "
                           f"repeat={d.get('repeat')!r}")
        if bad:
            print("\n  STRICT: refusing to aggregate")
            for b in bad[:12]:
                print(f"    {b}")
            FAILED.append(str(run_dir))
            return
    stats = {a: arm_stats(r, n_prompts) for a, r in arms.items()}
    base = stats.get("baseline")
    hdr = (f"{'arm':22s} {'reps':>4s} {'req-mean':>9s} {'pooled':>8s} {'vs base':>8s} "
           f"{'median':>7s} {'min':>7s} {'repSD':>6s} {'drafted':>8s} {'acc%':>6s} "
           f"{'think-off':>9s} {'len':>9s} {'ok':>5s}")
    print(hdr)
    print("-" * len(hdr))
    # repSD is over complete repeats; "n/a" means fewer than two were available
    for a in sorted(stats, key=lambda k: -stats[k]["pooled"]):
        s = stats[a]
        vs = (f"{100 * (s['pooled'] / base['pooled'] - 1):+7.1f}%"
              if base and base["pooled"] == base["pooled"] else "      -")
        acc = f"{s['acc_pct']:5.1f}" if s["acc_pct"] is not None else "    -"
        think = (f"{s['think_off']:4d}/{s['think_known']:<4d}"
                 if s["think_known"] else "   n/a   ")
        rsd = f"{s['rep_sd']:6.2f}" if s["rep_sd"] is not None else "   n/a"
        print(f"{a:22s} {s['reps']:4d} {s['mean']:9.1f} {s['pooled']:8.1f} {vs:>8s} "
              f"{s['median']:7.1f} {s['min']:7.1f} {rsd:>6s} {s['drafted']:8d} "
              f"{acc:>6s} {think:>9s} "
              f"{s['len_min']:4d}-{s['len_max']:<4d} {s['complete']}/{s['reps']}")
        for c in s["crashes"]:
            print(f"{'':24s}! died on {c['tag']}: {c['error'][:60]}")

    exact100 = [a for a, s in stats.items()
                if s["drafted"] and s["acc_pct"] is not None and abs(s["acc_pct"] - 100.0) < 1e-9]
    if exact100:
        print(f"\n  WARNING: {len(exact100)} arm(s) report EXACTLY 100 % acceptance "
              f"({', '.join(exact100)}).")
        print("    On a COMMON_CONTEXT_SEQ_RM_TYPE_FULL context that ratio is 1.0 by "
              "construction,")
        print("    not a measurement - partially accepted rounds skip both counters. "
              "See ERRATA A1.")

    tot = sum(s["tokens"] for s in stats.values())
    print(f"\n  pooled throughput weights tokens equally; {tot} generated tokens in total")

    # Pooled removes the wall-clock dependence on output length. It does not
    # make two arms comparable when they generated different numbers of tokens:
    # decode rate falls as the KV cache grows, so an arm that stopped at 187
    # tokens is being scored on cheaper tokens than one that ran to 300. This is
    # invisible in the table above, where every arm shows the same min-max.
    per_tag: dict = {}
    for arm, runs in arms.items():
        for r in runs:
            for x in r["rows"]:
                per_tag.setdefault(x["tag"], {}).setdefault(arm, set()).add(x["predicted_n"])
    diverging = [(t, d) for t, d in per_tag.items()
                 if len(arms) > 1 and len(d) == len(arms)
                 and len({n for v in d.values() for n in v}) > 1]
    if diverging:
        worst = max(diverging, key=lambda td: max(n for v in td[1].values() for n in v)
                    - min(n for v in td[1].values() for n in v))
        lo = min(n for v in worst[1].values() for n in v)
        hi = max(n for v in worst[1].values() for n in v)
        print(f"  ! {len(diverging)} of {len(per_tag)} prompts have the arms generating "
              f"DIFFERENT numbers of tokens, worst `{worst[0]}` {lo}-{hi} "
              f"({100 * (hi - lo) / lo:.0f} %). The `vs base` column then compares "
              f"arms that did different amounts of work - see ERRATA A17 and "
              f"analysis/length_matching.py. `BENCH_IGNORE_EOS=on` forces the "
              f"hard cap.")

    # ---- drift: does the baseline move across the run? ----------------------
    if base and base["reps"] > 1:
        # complete repeats only, so the comparison is like-for-like
        bm = [st.mean([x["predicted_per_second"] for x in r["rows"]])
              for r in sorted(arms["baseline"], key=lambda r: r["repeat"])
              if len(r["rows"]) == n_prompts]
        print(f"\n  drift check - baseline per repeat: "
              f"{', '.join(f'{v:.1f}' for v in bm)} tok/s")
        if len(bm) > 1:
            swing = 100 * (max(bm) - min(bm)) / st.mean(bm)
            print(f"    first-to-last {100 * (bm[-1] / bm[0] - 1):+.2f} %, "
                  f"full swing {swing:.2f} % of the mean")
            # Distinguish a cold-start first repeat from progressive drift. The
            # first arm of the first repeat runs on an idle, cool GPU; every
            # later arm starts warm. A single high rep0 followed by a flat tail
            # is warm-up, not thermal decline, and the two call for different
            # responses - discard the warm-up, or investigate the cooling.
            if len(bm) > 2:
                tail = bm[1:]
                tail_swing = 100 * (max(tail) - min(tail)) / st.mean(tail)
                rep0 = 100 * (bm[0] / st.mean(tail) - 1)
                print(f"    rep0 vs the rest {rep0:+.2f} %, "
                      f"swing excluding rep0 {tail_swing:.2f} %")
                if abs(rep0) > 1.5 and tail_swing < 1.5:
                    print("    -> cold-start warm-up in rep0, flat afterwards; "
                          "treat rep0 as warm-up rather than drift")
                elif tail_swing >= 1.5:
                    print("    -> the tail itself is moving; INVESTIGATE drift")
                else:
                    print("    -> negligible")
            elif swing >= 1.5:
                print("    -> INVESTIGATE")

    # ---- acceptance vs speed, the relationship the retracted chart hid ------
    # Across ARMS this is a weak statistic and needs enough of them: three
    # monotone points fit a line almost perfectly whatever the mechanism, so
    # reporting r = -1.000 from three arms would be exactly the kind of
    # impressive-looking number this audit exists to remove. The strong version
    # of this relationship is WITHIN one arm across prompts, which is a
    # different contrast - see ERRATA A7.
    MIN_ARMS = 5
    pts = [(s["acc_pct"], s["pooled"]) for a, s in stats.items()
           if s["acc_pct"] is not None]
    if len(pts) >= MIN_ARMS:
        r = pearson([p[0] for p in pts], [p[1] for p in pts])
        print(f"\n  acceptance vs pooled throughput across {len(pts)} speculative arms: "
              f"Pearson r = {r:+.3f}")
        print("    (across-arm contrast; the within-arm, across-prompt one is "
              "the strong relationship - ERRATA A7)")
    elif pts:
        print(f"\n  acceptance vs pooled throughput: only {len(pts)} speculative arm(s), "
              f"fewer than {MIN_ARMS} - not reporting a correlation")

    # ---- draft-length sweep -------------------------------------------------
    sweep = sorted(((int(a.rsplit("n", 1)[1]), stats[a]) for a in stats
                    if a.startswith("spec-draft-n") and a.rsplit("n", 1)[1].isdigit()),
                   key=lambda t: t[0])
    if len(sweep) >= 3:
        print("\n  draft-length sweep (n_min pinned to 1, matched vocabulary):")
        print(f"    {'n_max':>6s} {'pooled':>8s} {'vs base':>8s} {'drafted':>8s} {'acc%':>6s}")
        for nmax, s in sweep:
            vs = f"{100 * (s['pooled'] / base['pooled'] - 1):+7.1f}%" if base else "      -"
            acc = f"{s['acc_pct']:5.1f}" if s["acc_pct"] is not None else "    -"
            print(f"    {nmax:6d} {s['pooled']:8.1f} {vs:>8s} {s['drafted']:8d} {acc:>6s}")
    print()


def main() -> None:
    global STRICT
    args = [a for a in sys.argv[1:] if a != "--strict"]
    STRICT = "--strict" in sys.argv[1:]
    dirs = [Path(a) for a in args]
    if not dirs:
        sys.exit("usage: python analysis/matrix_report.py [--strict] <run-dir> [...]")
    for d in dirs:
        report(d)
    if STRICT and FAILED:
        sys.exit(f"\n{len(FAILED)} run directory/ies failed the strict check: "
                 + ", ".join(Path(f).name for f in FAILED))


if __name__ == "__main__":
    main()
