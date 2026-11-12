#!/usr/bin/env python3
"""Every prose figure outside the census is derived, or accounted for with a reason.

    python3 analysis/figure_registry.py           # the coverage, and what is bare
    python3 analysis/figure_registry.py --strict  # and exit 1 on anything bare or stale

Why this exists
---------------
A perturbation sweep on 2026-10-03 changed one prose decimal at a time in the
documents outside the decimal census and ran the whole claims job after each:
eight of twenty-six changes were caught and eighteen were not. The derivations
did not catch the figures in the documents they are named for, because each
checks the list of figures IT derives and the documents carry many more than
that. Nothing measured the difference.

The sharpest case is a figure quoted across documents. When this registry was
first run over plan X it found run Y's result quoted there as 15.6 % and 15.9 %:
the mean log ratios printed as percentages, the defect corrected in the two
documents that carry run Y's table and left standing in the third, because
nothing compared a quote with the derivation.

What it does
------------
It asks each derivation what it checks -- `checked_figures()`, the same function
that derivation's own check uses, never a second list -- and enumerates each
document's prose decimals the way the census does: no fenced blocks, no table
rows. A decimal the derivations report for that document is derived. Anything
else has to be in `ACCOUNTED` with a kind and a reason, and the kinds that can
be verified are verified:

  derived    a derivation reports it for ANOTHER document; `why` names the
             module, and the value has to be in that module's set. This is how
             a cross-document quote is held to the derivation: if the data moved,
             the quote stops matching and is bare.
  quote      quoted from a censused document, where `verify_claims.py` and the
             census hold it; `why` names the document, and the value has to
             appear in it.
  simulated  derived by a check too slow to rerun here, which the claims job
             runs; `why` names it.
  name       part of a proper name or a version, not a measurement.
  recorded   observed on a host and written down, with no artefact in this tree.
  underived  answerable from this tree and not derived by anything yet. Debt,
             printed every time this runs.
  archived   a document this repository keeps as it was published.

Every entry is per document and per value, so a figure edited anywhere becomes a
value nothing accounts for, and `--strict` fails on it. An entry for a value the
document no longer carries also fails, and so does one for a value a derivation
already reports there: an exemption that outlives its reason is how a list like
this goes stale.
"""
from __future__ import annotations

import importlib
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

ROOT = pathlib.Path(__file__).resolve().parents[1]
DERIVATIONS = ("plan_z_power", "plan_siblings_power", "rederive_run_y",
               "layer_a_slope", "load_run_power")
DEC = re.compile(r"(?<![\w.])\d+\.\d+(?![\w])")
KINDS = ("derived", "quote", "simulated", "name", "recorded", "underived",
         "archived")

_X = "v4_audit_2026_08_25/PROSPECTIVE_PLAN_X_HOST_LOAD.md"
_Y = "v4_audit_2026_08_25/PROSPECTIVE_PLAN_Y_CORE_PINNING.md"
_Z = "v4_audit_2026_08_25/PROSPECTIVE_PLAN_Z_MEMORY_STATE.md"
_S = "v4_audit_2026_08_25/PROSPECTIVE_PLAN_SIBLINGS.md"
_V5 = "v5_pinning_2026_09_26/README.md"
_V6 = "v6_pilot_layer_a_2026_10_02/README.md"

_RUN_T_T3 = ("ERRATA.md", "run T against run T3, two invocations two hours apart, "
             "from ERRATA's table of the same three arms")
_A13 = ("ERRATA.md", "A13's table: the server's acceptance counter against the "
        "drafter's own")
_HAZARD = ("simulated", "the design's power at the two ends of the hazard "
           "interval; `load_run_power.py --check` simulates them in the claims job, "
           "about three minutes")
_LAYER_A_PLATEAU = ("recorded", "seven invocations at a flat plateau on the pilot "
                    "card, 2026-10-02; written in `layer_a_slope.py`'s comments and "
                    "in prose, and the runs are not committed")
_A16_STEP = ("derived", "plan_z_power")


def _archived(reason: str, values: str) -> dict:
    return {v: ("archived", reason) for v in values.split()}


ACCOUNTED: dict[str, dict[str, tuple[str, str]]] = {
    _S: {"3.0": ("name", "Turbo Boost Max 3.0, the name of Intel's feature")},
    _X: {
        "0.11": ("quote", _RUN_T_T3[0]), "0.79": ("quote", _RUN_T_T3[0]),
        "3.40": ("quote", _RUN_T_T3[0]),
        "3.5": ("quote", "ERRATA.md"),
        "29.7": ("quote", _A13[0]), "41.3": ("quote", _A13[0]),
        "72.8": ("quote", _A13[0]), "73.0": ("quote", _A13[0]),
        "0.64": _HAZARD, "0.82": _HAZARD, "0.85": _HAZARD, "0.92": _HAZARD,
        "14.42": ("derived", "rederive_run_y"), "14.73": ("derived", "rederive_run_y"),
        "15.30": ("derived", "rederive_run_y"), "15.34": ("derived", "rederive_run_y"),
    },
    _Y: {
        "0.27": ("derived", "rederive_run_y"),
        "0.760": ("derived", "load_run_power"),
        "3.93": _A16_STEP,
        "0.355": ("underived", "the power table's fourth column, the simulated "
                  "chance that a two-sided reading holds under a true null; "
                  "`rederive_run_y.plan_power` says it does not derive it, because "
                  "the simulator in `load_run_power.py` is written for a "
                  "between-condition contrast"),
        "6.9": ("underived", "the arm's milliseconds per token, consistent with "
                "run T4's faster level, 1000 / 145.161 = 6.889; the plan does not "
                "say which figure it rounded. ERRATA's 6.9 is a percentage-point "
                "gap and a different quantity, so it is not entered as a quote"),
    },
    _Z: {
        "0.014": ("derived", "layer_a_slope"), "0.022": ("derived", "layer_a_slope"),
        "2.35": ("derived", "layer_a_slope"), "51.6": ("derived", "layer_a_slope"),
        "0.27": ("derived", "rederive_run_y"), "14.42": ("derived", "rederive_run_y"),
        "0.013": _LAYER_A_PLATEAU,
        "0.036": ("recorded", "the first readings of `bench/vram_bandwidth.cu` on "
                  "the pilot card, 2026-10-02, four consecutive windows; not "
                  "committed"),
        "828.4": ("recorded", "the same first readings: sustained bandwidth at "
                  "four gibibyte buffers; not committed"),
        "828.7": ("recorded", "the same first readings; not committed"),
        "13.3": ("name", "the version of nvcc the tool was compiled with"),
    },
    _V5: {"3.93": _A16_STEP},
    _V6: {
        "1.67": ("derived", "plan_z_power"), "3.93": _A16_STEP,
        "0.013": _LAYER_A_PLATEAU,
        "0.006": ("recorded", "five-second against ten-second windows at a flat "
                  "plateau, a balanced test on the pilot card on 2026-10-02; in "
                  "`layer_a_slope.py`'s comments and in prose, not committed"),
        "175.8": ("recorded", "a process on the pilot host, read from its process "
                  "accounting before the pilot; the host's state, not this tree's"),
        "197.4": ("recorded", "the same process's CPU time, read the same way"),
        "197.5": ("recorded", "the same process's elapsed time, read the same way"),
    },
    "v4_audit_2026_08_25/RELEASE_PROCEDURE.md": {
        "1.0": ("recorded", "the size of the release archives, which are release "
                "assets and not files in this tree")},
    # Archived: kept as they were published. Every value is listed, so an edit to
    # any of them is a value nothing accounts for. The direction is the reverse
    # of a quote: ERRATA quotes these documents, not the other way round.
    "v2_3090_followup/README.md": _archived(
        "the v2 follow-up as published; ERRATA quotes it and it is not re-derived",
        "0.3 0.8 1.00000 1.0 31.6 40.7 53.7 62.8 999.6"),
    "v2_3090_followup/SUMMARY.md": _archived(
        "the v2 summary as published; ERRATA quotes it and it is not re-derived",
        "0.5 1.00000 1.0 12.0 13.3 24.04 40.7 53.7 139.9 580.126"),
    "v3_dflash_2026_05_07/README.md": _archived(
        "the v3 sub-README as published; ERRATA quotes it and it is not re-derived",
        "0.5 1.0 2.0 14.8 18.7 47.4 580.126"),
    "pr_comment.md": _archived(
        "a retracted draft, never posted, kept as the record of what was almost "
        "said; ERRATA refutes its claims, and two of its figures, 7.620 and 5.976 "
        "ms, are the upstream reporter's",
        "0.19 0.4 0.80 0.90 1.0 5.976 7.620 8.6 12.6 17.5 18.0 18.7 25.7 27.5 "
        "44.6 53.7 77.0 94.36 118.8 119.8 129.6 135.5 135.7 138.9 580.126 "
        "2505.19645 2506.20675"),
}


def prose_decimals(doc: str) -> list[str]:
    """The same reading the decimal census takes: no fenced blocks, no tables."""
    body = (ROOT / doc).read_text(encoding="utf-8")
    body = re.sub(r"```.*?```", "", body, flags=re.S)
    lines = [ln for ln in body.splitlines() if not ln.lstrip().startswith("|")]
    return DEC.findall("\n".join(lines))


def scope() -> list[str]:
    """The documents outside the census, read from the census itself.

    `table_coverage.EXCLUDED` is the list of documents the census does not
    count, each with its reason, and nothing may be missing from it and DOCS
    together. A second list here would drift from it.
    """
    import table_coverage                                           # noqa: E402
    return sorted(table_coverage.EXCLUDED)


def reported() -> dict[str, dict[str, set]]:
    """module -> document -> the decimals that module reports for it."""
    out: dict[str, dict[str, set]] = {}
    for name in DERIVATIONS:
        mod = importlib.import_module(name)
        fn = getattr(mod, "checked_figures", None)
        if fn is None:
            raise SystemExit(
                f"analysis/{name}.py has no checked_figures(), so this cannot ask "
                f"it what it covers and would have to keep a second list")
        out[name] = {}
        for doc, figs in fn().items():
            s = out[name].setdefault(doc, set())
            for v in figs.values():
                s |= set(DEC.findall(str(v)))
    return out


def audit(rep: dict | None = None, accounted: dict | None = None,
          docs: list | None = None) -> dict:
    """Classify every prose decimal of every document in scope.

    Returns {doc: {"derived": set, kind: set, ..., "bare": set}} and a list of
    problems with the entries themselves.
    """
    rep = rep if rep is not None else reported()
    acc = accounted if accounted is not None else ACCOUNTED
    docs = docs if docs is not None else scope()
    anywhere = {m: set().union(*d.values()) if d else set() for m, d in rep.items()}
    problems: list[str] = []
    out: dict = {}
    for doc in docs:
        decs = set(prose_decimals(doc))
        here = set().union(*(d.get(doc, set()) for d in rep.values()))
        row: dict = {"derived": decs & here, "bare": set()}
        row.update({k: set() for k in KINDS if k != "derived"})
        row["derived elsewhere"] = set()
        mine = acc.get(doc, {})
        for v in sorted(decs - here):
            if v not in mine:
                row["bare"].add(v)
                continue
            kind, why = mine[v]
            if kind not in KINDS or not why.strip():
                problems.append(f"{doc}: {v} has kind {kind!r} and reason {why!r}")
                continue
            if kind == "derived":
                if why not in rep:
                    problems.append(f"{doc}: {v} is said to be derived by {why!r}, "
                                    f"which is not one of {DERIVATIONS}")
                elif v not in anywhere[why]:
                    problems.append(f"{doc}: {v} is said to be derived by {why}, "
                                    f"which does not report it for any document")
                else:
                    row["derived elsewhere"].add(v)
                continue
            if kind == "quote":
                src = ROOT / why
                if not src.is_file():
                    problems.append(f"{doc}: {v} is quoted from {why}, which "
                                    f"does not exist")
                elif v not in DEC.findall(src.read_text(encoding="utf-8")):
                    problems.append(f"{doc}: {v} is quoted from {why}, which does "
                                    f"not carry it")
                else:
                    row["quote"].add(v)
                continue
            row[kind].add(v)
        for v, (kind, _why) in sorted(mine.items()):
            if v not in decs:
                problems.append(f"{doc}: an entry for {v} ({kind}), which the "
                                f"document no longer carries in its prose")
            elif v in here:
                problems.append(f"{doc}: an entry for {v} ({kind}), which a "
                                f"derivation already reports for this document")
        out[doc] = row
    for doc in acc:
        if doc not in docs:
            problems.append(f"{doc} has entries and is not a document outside the "
                            f"census")
    return {"rows": out, "problems": problems}


def main() -> int:
    args = sys.argv[1:]
    for a in args:
        if a != "--strict":
            print(f"unknown argument {a!r}; the only one is --strict",
                  file=sys.stderr)
            return 2
    strict = "--strict" in args
    res = audit()
    total = {k: 0 for k in ("derived", "derived elsewhere", *KINDS[1:], "bare")}
    for doc, row in res["rows"].items():
        n = sum(len(v) for v in row.values())
        if not n:
            continue
        print(doc)
        parts = [f"{len(row[k])} {k}" for k in total if row.get(k)]
        print(f"  {n} distinct prose decimals: {', '.join(parts)}")
        for k in ("underived", "bare"):
            if row.get(k):
                print(f"  {k}: {', '.join(sorted(row[k], key=float))}")
        for k in total:
            total[k] += len(row.get(k, ()))
    print("\n" + ", ".join(f"{v} {k}" for k, v in total.items()))
    for p in res["problems"]:
        print(f"  PROBLEM {p}")
    failed = total["bare"] + len(res["problems"])
    if failed:
        print(f"\n{total['bare']} prose figure(s) accounted for by nothing, and "
              f"{len(res['problems'])} entry problem(s)")
    return 1 if (strict and failed) else 0


if __name__ == "__main__":
    sys.exit(main())
