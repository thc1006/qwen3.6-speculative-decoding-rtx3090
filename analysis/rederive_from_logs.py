#!/usr/bin/env python3
"""Re-derive the committed JSON from the raw server logs and diff it.

The claim checker proves the derived files and the documents agree with each
other. That is not the same as proving the derived files came from the logs
they name, and until 2026-08-27 nothing proved the second: the logs were not
published, so the extraction could be trusted but not re-run.

    python analysis/rederive_from_logs.py <bench-root>

`<bench-root>` is the unpacked archive with the committed arm-run JSON copied
in beside the logs, which is what `.github/workflows/evidence.yml` builds.

Exits non-zero if any record differs. Records that cannot be produced at all -
runs whose arm-run JSON is not committed - are listed and counted against the
number this repository publishes for them, so "fewer records" fails too.
"""

import json
import os
import subprocess
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "v4_audit_2026_08_25", "data")

# Runs whose logs are in the archive and whose arm-run JSON is not committed,
# because the runs never completed their cell set and check_data_integrity.py
# refuses incomplete runs. Their rows in the counter dump cannot be regenerated
# from what is published; the count is asserted rather than waved through.
NOT_REPRODUCIBLE = {"matrix_G_dflash_20260826_000124",
                    "matrix_I_conc1_20260826_012917",
                    "matrix_J_dflash_fit_20260826_014308"}

FAIL = []


def run(script, *args):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "analysis", script), *args],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode:
        sys.exit(f"{script} failed:\n{r.stderr[-2000:]}")
    return json.loads(r.stdout)


def committed(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as fh:
        return json.load(fh)


def compare(label, regenerated, published, key, expected_gap=0, scope=None):
    """`scope` keeps the comparison to the runs the committed dump covers.

    The bench root grows: the 2026-08-27 tranche adds 618 logs from runs that
    postdate `acceptance_counter_comparison.json`. Regenerating over the whole
    root then produces rows the dump cannot contain, which is not a discrepancy
    and must not be reported as one - but the count of them is worth printing,
    because "the extractor produced more than the dump" and "the extractor
    produced something different" look identical if you only count.
    """
    A = {key(r): r for r in regenerated}
    B = {key(r): r for r in published}
    if scope is not None:
        outside = {k for k in A if not scope(k)}
        if outside:
            print(f"  {label}: {len(outside)} record(s) from runs the dump "
                  f"predates, not compared")
            A = {k: v for k, v in A.items() if k not in outside}
    differing = sorted(k for k in set(A) & set(B) if A[k] != B[k])
    missing = sorted(set(B) - set(A))
    extra = sorted(set(A) - set(B))
    ok = len(set(A) & set(B)) - len(differing)
    print(f"  {label}: {ok} of {len(B)} identical, {len(differing)} differing, "
          f"{len(missing)} not regenerated, {len(extra)} unexpected")
    for k in differing[:3]:
        print(f"      differs at {k}: "
              + ", ".join(f"{f} {A[k][f]!r} != {B[k].get(f)!r}"
                          for f in A[k] if A[k][f] != B[k].get(f))[:300])
    for k in missing[:3]:
        print(f"      missing {k}")
    if differing:
        FAIL.append(f"{label}: {len(differing)} record(s) differ")
    if len(missing) != expected_gap:
        FAIL.append(f"{label}: {len(missing)} records could not be regenerated, "
                    f"and {expected_gap} is what this repository documents")
    if extra:
        FAIL.append(f"{label}: {len(extra)} record(s) the dump does not contain")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    bench = os.path.abspath(sys.argv[1])
    print(f"re-deriving from {bench}\n")

    # --- the acceptance counters behind A13 --------------------------------
    published = committed("acceptance_counter_comparison.json")
    gap = sum(1 for r in published if r["run"] in NOT_REPRODUCIBLE)
    _runs_in_dump = {r["run"] for r in published}
    compare("acceptance_counter_comparison.json",
            run("compare_acceptance_counters.py", bench), published,
            lambda r: (r["run"], r["arm"], r.get("repeat")), expected_gap=gap,
            scope=lambda k: k[0] in _runs_in_dump)

    # --- the source timers behind A12 --------------------------------------
    pub = committed("checkpoint_timers_20260826.json")
    pub_rows = pub if isinstance(pub, list) else pub["rows"]
    runs = {r.get("run") for r in pub_rows if r.get("run")} or \
        {"matrix_T_timers_20260826_182639"}
    logs = []
    for r in sorted(runs):
        d = os.path.join(bench, r, "server_logs")
        logs += [os.path.join(d, f) for f in sorted(os.listdir(d))
                 if f.endswith(".log")]
    got = run("extract_checkpoint_timers.py", *logs)
    compare("checkpoint_timers_20260826.json",
            got if isinstance(got, list) else got["rows"], pub_rows,
            lambda r: (r["arm"], r["repeat"]))

    # --- run T4's split timers, which answer A12's boundary question --------
    # Same extractor, a different instrumented build: the drain is timed
    # separately, so `sync_total_s` is the wait A12 was accused of attributing
    # to the copying. It is 0.002 s, and it has to come back from the logs.
    d = os.path.join(bench, "matrix_T4_split_20260827_175051", "server_logs")
    if os.path.isdir(d):
        logs = [os.path.join(d, f) for f in sorted(os.listdir(d))
                if f.endswith(".log")]
        got = run("extract_checkpoint_timers.py", *logs)
        compare("checkpoint_timers_20260827_split.json",
                got if isinstance(got, list) else got["rows"],
                committed("checkpoint_timers_20260827_split.json"),
                lambda r: (r["arm"], r["repeat"]))
    else:
        print("  checkpoint_timers_20260827_split.json: run T4's logs are not "
              "in this archive, not compared")

    # --- the speculative accounting behind A1 and A4 ------------------------
    pub = committed("spec_accounting_20260826.json")
    by_run = defaultdict(list)
    for r in pub:
        by_run[r["run"]].append(r["log"])
    keyed = []
    for r, names in sorted(by_run.items()):
        paths = [os.path.join(bench, r, "server_logs", n) for n in names]
        absent = [p for p in paths if not os.path.exists(p)]
        if absent:
            FAIL.append(f"spec accounting: {len(absent)} log(s) missing from {r}")
            continue
        # The extractor reports a `run` field of its own, taken from the log's
        # parent directory. Overwriting it with the run this call was made for
        # is what makes the key unambiguous: `spec-draft-n8__rep0.log` exists in
        # several runs, and matching regenerated records back by basename alone
        # assigns the first run that happens to contain that name.
        for rec in run("extract_spec_accounting.py", *paths):
            rec["run"] = r
            keyed.append(rec)
    compare("spec_accounting_20260826.json", keyed, pub,
            lambda r: (r["run"], r["log"]))

    print()
    if FAIL:
        sys.exit("  " + "\n  ".join(FAIL))
    print("  every committed record this archive can produce came back identical")


if __name__ == "__main__":
    main()
