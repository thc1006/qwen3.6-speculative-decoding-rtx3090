#!/usr/bin/env python3
"""Re-derive every number run Y's result table publishes, and check the documents.

    python analysis/rederive_run_y.py          # derive, check both copies, exit 1 on a mismatch
    python analysis/rederive_run_y.py --show   # derive and print, check nothing

Why this file exists. Run Y's result table was computed by hand, published in
two documents, and every inferential column in it was the MEAN LOG RATIO printed
as a percentage: a mean log ratio of -0.15573 reached both documents as
"-15.57 %" when the change it describes is -14.42 %. `analysis/paired_blocks.py`
has done this correctly since run O2 -- `point = exp(mean(log ratio))` -- and it
was never run on this round, because it pairs every arm against one baseline ARM
and the contrast here is each arm against its own unpinned self.

Nothing read the table. `analysis/verify_claims.py` holds every other published
number and is one of the six files `bench/check_release_binding.py` compares
between the `v4.2` tag and HEAD, so it cannot grow an assertion for a round
published after that tag. The exclusion keeping run Y's README out of the census
said its tables were guarded by `check_data_integrity.py` instead; that walks
directory structure and reads no published value. This is what that sentence
should have named.

What it covers: the seven columns of the result table, for both arms, in both
documents, compared as whole rows -- a table in two documents with one copy
wired to an assertion is ERRATA A51 -- and the generated, drafted and accepted
token totals the mechanism paragraph quotes.

What it does NOT cover, named here rather than left for a reader to find: the
four telemetry figures in that paragraph (SM clock, power, utilisation and
temperature per condition). The arm-runs timestamp with CLOCK_MONOTONIC and the
sampler writes wall clock, so segmenting the trace by condition needs a mapping
this file does not have.
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paired_blocks import (                                      # noqa: E402
    load_blocks, t_critical_95_one_sided, t_critical_975)

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN = ROOT / "v5_pinning_2026_09_26/data/matrix_Y_pinning_20260926_131110"
# fast (performance cores) against slow (efficiency cores), within the block
CONTRASTS = (("spec-dflash-n2", "spec-dflash-n2-ecore"),
             ("baseline", "baseline-ecore"))
DOCS = ("v5_pinning_2026_09_26/README.md",
        "v4_audit_2026_08_25/PROSPECTIVE_PLAN_Y_CORE_PINNING.md")
BLOCKS = 12
# the pre-registered equivalence margin: the arm holds if the interval is inside it
MARGIN_PCT = 1.0
MINUS = "−"          # U+2212, which is what the documents carry


def _pct(v: float) -> str:
    """`-14.42 %` the way the tables spell it, with a real minus sign."""
    return f"{MINUS if v < 0 else ''}{abs(v):.2f} %"


def derive() -> list[dict]:
    blocks = load_blocks(RUN)
    rows = []
    for fast, slow in CONTRASTS:
        have = sorted(b for b, v in blocks.items() if fast in v and slow in v)
        if len(have) != BLOCKS:
            sys.exit(f"{fast}: {len(have)} complete blocks, not {BLOCKS}. "
                     f"The interval's degrees of freedom are published, so a "
                     f"different count is a different number, not a near miss.")
        f = [blocks[b][fast] for b in have]
        s = [blocks[b][slow] for b in have]
        lr = [math.log(s[k] / f[k]) for k in range(len(have))]
        m = st.fmean(lr)
        se = st.stdev(lr) / math.sqrt(len(lr))
        df = len(lr) - 1
        half = t_critical_975(df) * se
        # back-transformed, which is the whole point of this file
        change = (math.exp(m) - 1) * 100
        lo = (math.exp(m - half) - 1) * 100
        hi = (math.exp(m + half) - 1) * 100
        bound = (1 - math.exp(m - t_critical_95_one_sided(df) * se)) * 100
        rows.append({
            "arm": fast, "fast": st.fmean(f), "slow": st.fmean(s),
            "change_pct": change, "ci_lo_pct": lo, "ci_hi_pct": hi,
            "one_sided_bound_pct": bound, "blocks": len(lr), "df": df,
            "mean_log_ratio": m,
            # the equivalence reading: holds only if the whole interval is inside
            # the margin, so a wide interval that straddles it does not pass
            "verdict": ("holds" if -MARGIN_PCT <= lo and hi <= MARGIN_PCT
                        else "moves"),
        })
    return rows


def cells(r: dict) -> dict:
    """What each column of the result table should hold, by its header.

    Keyed by header rather than position, because the two documents carry the
    same table with different column SETS: the README adds the equivalence
    verdict and the plan does not. A checker that compared one canonical row
    string would have had to skip the plan's copy, and skipping a copy is what
    ERRATA A51 is about.
    """
    return {
        "arm": f"`{r['arm']}`",
        "fast": f"{r['fast']:.3f}",
        "slow": f"{r['slow']:.3f}",
        "change": _pct(r["change_pct"]),
        "95 % interval": f"[{_pct(r['ci_lo_pct'])}, {_pct(r['ci_hi_pct'])}]",
        "one-sided upper limit": _pct(r["one_sided_bound_pct"]),
        "plus or minus one per cent": r["verdict"],
    }


def _split(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def check_doc(doc: str, rows: list[dict]) -> list[str]:
    """Every cell of every arm row, against the header the document declares."""
    lines = (ROOT / doc).read_text(encoding="utf-8").splitlines()
    head = [k for k, ln in enumerate(lines)
            if ln.startswith("| arm |") and "one-sided upper limit" in ln]
    if len(head) != 1:
        return [f"{doc}: {len(head)} result tables found, expected exactly one"]
    cols = _split(lines[head[0]])
    want_all = cells(rows[0])
    unknown = [c for c in cols if c not in want_all]
    if unknown:
        # a column added to the table and not wired here is A51 exactly: it
        # would sit in a published table with nothing deriving it
        return [f"{doc}: the table has column(s) {unknown} that nothing derives"]
    bad = []
    for r in rows:
        want = cells(r)
        got = [ln for ln in lines[head[0]:] if _split(ln)[:1] == [f"`{r['arm']}`"]]
        if len(got) != 1:
            bad.append(f"{doc}: {len(got)} rows for {r['arm']}, expected one")
            continue
        for col, cell in zip(cols, _split(got[0])):
            if cell != want[col]:
                bad.append(f"{doc}: {r['arm']} column {col!r} is {cell!r} "
                           f"and the data gives {want[col]!r}")
    return bad


def token_totals() -> dict:
    out = {}
    for p in sorted(RUN.glob("*__rep*.json")):
        j = json.loads(p.read_text(encoding="utf-8"))
        g, d, a = out.setdefault(j["arm"], [0, 0, 0])
        out[j["arm"]] = [g + sum(r["predicted_n"] for r in j["rows"]),
                         d + sum(r["draft_n"] for r in j["rows"]),
                         a + sum(r["draft_n_accepted"] for r in j["rows"])]
    return out


def main() -> None:
    show_only = "--show" in sys.argv[1:]
    for a in sys.argv[1:]:
        if a != "--show":
            sys.exit(f"unknown argument {a!r}; the only one is --show")
    rows = derive()
    print(f"{RUN.name}: {BLOCKS} blocks, {rows[0]['df']} degrees of freedom")
    for r in rows:
        print(f"  mean log ratio {r['mean_log_ratio']:+.5f}  ->  "
              f"change {r['change_pct']:+.2f} %   "
              f"(printing the log ratio as a percentage gives "
              f"{r['mean_log_ratio'] * 100:+.2f} %, which is the defect)")
        c = cells(r)
        print("  | " + " | ".join(c[k] for k in (
            "arm", "fast", "slow", "change", "95 % interval",
            "one-sided upper limit", "plus or minus one per cent")) + " |")
    tot = token_totals()
    for fast, slow in CONTRASTS:
        if tot[fast] != tot[slow]:
            sys.exit(f"{fast}: generated, drafted and accepted totals are "
                     f"{tot[fast]} fast and {tot[slow]} slow. The documents say "
                     f"the output is identical to the token, and it is not.")
        print(f"  {fast}: generated {tot[fast][0]}, drafted {tot[fast][1]}, "
              f"accepted {tot[fast][2]}, identical in both conditions")
    if show_only:
        return
    bad = []
    for doc in DOCS:
        bad += check_doc(doc, rows)
        text = (ROOT / doc).read_text(encoding="utf-8")
        for fast, _slow in CONTRASTS:
            for n in token_totals()[fast]:
                # published grouped with a thin space, as this repository spells
                # a thousand, and asserted so that "identical to the token" is a
                # claim about numbers a reader can find rather than a sentence
                if n and f"{n:,}".replace(",", " ") not in text:
                    bad.append(f"{doc}: does not carry the token total "
                               f"{n:,}".replace(",", " "))
    if bad:
        print("\nFAILED")
        for b in bad:
            print("  " + b)
        sys.exit(1)
    print(f"\nboth copies of the table match the data, in {len(DOCS)} documents")


if __name__ == "__main__":
    main()
