#!/usr/bin/env python3
"""How many of the published numbers would notice if they were wrong.

    python analysis/table_coverage.py                     # the table census
    python analysis/table_coverage.py --json
    python analysis/table_coverage.py --probe             # ground truth, slow
    python analysis/table_coverage.py --probe --covered   # the converse check
    python analysis/table_coverage.py --prose [--probe]   # the other half

Why this exists
---------------
Six times a published figure has been computed from the data, compared against
a literal, and printed into a table that nothing read. Planting a wrong number
in the table passed every check: run M's aggregates, three tables the second
review's pass found, the merged checkpoint cost table in two documents, and
A16's own O2-against-O3 table. Each was found by accident. This counts them
instead.

Three measurements, in increasing order of how much they are worth:

`census()` is structural. A table is COVERED if its header is a literal fed to
one of `verify_claims.py`'s table readers, which is what makes it parsed cell
by cell. It is a proxy in both directions: a cell can be guarded by a literal
grep without its table being parsed, and a parser can match a header and assert
nothing about the column you changed.

`--probe` is the ground truth for the tables that are NOT parsed. It writes a
wrong number into one cell of each in turn, runs the claim checker, and asks
whether any assertion that was passing now fails. About twenty seconds a table,
so it is not run in CI; `analysis/verify_claims.py` holds the census to an
exact table count and a floor on the parsed ones instead.

`--probe --covered` is the converse, and the one that keeps the census honest:
it perturbs the tables the census calls parsed and requires each to be caught.

`--prose` counts the decimal numbers that are not in any table, which is the
larger half. Its census is weaker still - "does not appear as a string literal
anywhere in the checker" is an upper bound on the gap - so `--prose --probe`
perturbs a fixed sample of them and reports the measured rate.

A table with no derivable numbers, one listing paths or upstream links, does
not need parsing and is counted separately rather than held against the total.

The verdict is never the checker's exit status. A first version used it and
every table came back caught, because the checker had one unrelated failing
assertion at the time and so exited non-zero whatever was done to the
documents. A probe whose control and treatment agree measures nothing.
"""
from __future__ import annotations

import ast
import contextlib
import json
import pathlib
import random
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ["README.md", "ERRATA.md", "CHANGELOG.md", "RETEST_TODO.md",
        "BENCHMARK_ENV.md", "PULL_REQUEST.md",
        "v4_audit_2026_08_25/README.md",
        "v4_audit_2026_08_25/PREREGISTERED_PREDICTION.md",
        "v4_audit_2026_08_25/PREREGISTERED_W.md"]
# Not censused, and why. Nothing may be missing from DOCS + EXCLUDED: a new
# document that is in neither would escape the count silently, which is the
# failure this whole file exists to stop.
EXCLUDED = {
    "v2_3090_followup/README.md": "archived artefact; ERRATA quotes it, it is not re-derived",
    "v2_3090_followup/SUMMARY.md": "archived artefact; ERRATA quotes it, it is not re-derived",
    "v3_dflash_2026_05_07/README.md": "archived artefact; ERRATA quotes it, it is not re-derived",
    "v4_audit_2026_08_25/harness/README.md": "provenance note; hashes, not measurements",
    "v4_audit_2026_08_25/patches/README.md": "provenance note; hashes, not measurements",
    # marked DO NOT POST at the top and never sent: it is a record of claims
    # this repository retracted, and every number in it is quoted from ERRATA
    # beside the finding that refutes it
    "pr_comment.md": "a retracted draft, never posted; its numbers are ERRATA's",
}
# Each reader is bound to one document, so a header literal only covers a table
# in THAT document. Matching on the header alone counted a duplicated table in
# another file as parsed - the re-derivation table appears in three documents,
# two of them read - and `--probe --covered` is what caught it.
READER_DOC = {"_md_table": "ERRATA.md",
              "_v4_table": "v4_audit_2026_08_25/README.md",
              "_pr_table": "PULL_REQUEST.md",
              "_root_table": "README.md",
              "_ch_table": "CHANGELOG.md"}
DASH = {"−": "-", "–": "-", "—": "-"}
# a cell that is a measurement rather than a path, hash, date, link or issue
NOT_A_VALUE = re.compile(r"https?://|\.py|\.json|\.md|\.log|\.sh|\.cff"
                         r"|`[0-9a-f]{8}|\d{4}-\d{2}-\d{2}|#\d")
NUMERIC = re.compile(r"\d+\.\d+|\d{2,}")


def norm(s: str) -> str:
    for a, b in DASH.items():
        s = s.replace(a, b)
    return s


# Two tables are read by a bespoke loop in `verify_claims.py` rather than by one
# of the readers above, so the AST scan cannot see them. Declared here, and the
# declaration is a claim rather than a courtesy: `--probe --covered` perturbs
# them like any other and both come back caught.
HAND_PARSED = {
    ("README.md", "| arm | pooled tok/s | change |"),
    ("README.md", "| arm | O2 | O3 | shift |"),
}


def parsed_headers() -> set[tuple[str, str]]:
    """(document, header) for every table the checker feeds to a reader."""
    src = (ROOT / "analysis" / "verify_claims.py").read_text(encoding="utf-8")
    out = set()
    for n in ast.walk(ast.parse(src)):
        if (isinstance(n, ast.Call) and getattr(n.func, "id", "") in READER_DOC
                and n.args and isinstance(n.args[0], ast.Constant)):
            out.add((READER_DOC[n.func.id], n.args[0].value))
    return out | HAND_PARSED


def tables(rel: str) -> list[dict]:
    """Every markdown table in one document, blockquoted ones included."""
    lines = [l.rstrip() for l in (ROOT / rel).read_text(encoding="utf-8").splitlines()]
    out = []
    for i, line in enumerate(lines[:-1]):
        head = line.strip().lstrip("> ").strip()
        rule = lines[i + 1].strip().lstrip("> ").strip()
        bare = rule.replace("|", "").replace(" ", "")
        if not (head.startswith("|") and bare and set(bare) <= set("-:")):
            continue
        body = []
        for m in lines[i + 2:]:
            s = m.strip().lstrip("> ").strip()
            if not s.startswith("|"):
                break
            body.append(s)
        cells = [c.strip() for r in body for c in r.strip().strip("|").split("|")]
        values = [c for c in cells if NUMERIC.search(c) and not NOT_A_VALUE.search(c)]
        out.append({"doc": rel, "line": i + 1, "header": head,
                    "rows": len(body), "value_cells": len(values)})
    return out


# Numbers live outside tables too, and there are more of them. The prose census
# is structural and weaker than the table one: a number can be guarded without
# appearing as a literal here, so "not a literal in verify_claims.py" is an
# upper bound on the gap, not the gap. `--prose --probe` measures the real rate
# on a fixed sample, because perturbing all of them one at a time is hours.
PROSE_SAMPLE = 40
PROSE_SEED = 20260828


def _checker_literals(src: str | None = None) -> str:
    """Every string literal in the checker EXCEPT the labels of its assertions.

    A label is prose about a check, not a check: `chk("acceptance falls from
    66 % to 29 %", ...)` would otherwise make 66 and 29 look guarded wherever
    they appear in a document, including in sentences nothing reads. Leaving
    labels in also made the count move whenever an assertion was reworded,
    which is a measurement that reports the measurer.
    """
    if src is None:
        src = (ROOT / "analysis" / "verify_claims.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    labels = {id(n.args[0]) for n in ast.walk(tree)
              if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "chk"
              and n.args and isinstance(n.args[0], ast.Constant)}
    return " ".join(n.value for n in ast.walk(tree)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                    and id(n) not in labels)


def prose_numbers() -> list[dict]:
    """Every decimal number outside a table, a fenced block and an indented block."""
    out = []
    dec = re.compile(r"(?<![\w.])\d+\.\d+(?![\w])")
    for rel in DOCS:
        fenced = False
        for i, raw in enumerate((ROOT / rel).read_text(encoding="utf-8").splitlines()):
            line = raw.strip().lstrip("> ").strip()
            if line.startswith("```"):
                fenced = not fenced
                continue
            if fenced or line.startswith("|") or raw.startswith(("    ", "\t")):
                continue
            for v in dec.findall(line):
                out.append({"doc": rel, "line": i + 1, "value": v})
    return out


def prose_census() -> dict:
    blob = _checker_literals()
    nums = prose_numbers()
    absent = [n for n in nums if n["value"] not in blob]
    return {"prose_numbers": len(nums), "not_a_literal": len(absent),
            "absent": absent}


def prose_probe(absent: list[dict]) -> list[dict]:
    """Perturb a fixed sample and count what the checker notices.

    Fixed seed, so the sample is the same on every machine and the number can
    be republished without drifting. The sample is of the numbers that are NOT
    literals in the checker, because the ones that are need no measuring.
    """
    rng = random.Random(PROSE_SEED)
    pick = rng.sample(absent, min(PROSE_SAMPLE, len(absent)))
    out = []
    with contextlib.ExitStack() as stack:
        wt = _worktree(stack)
        base_fails, base_crash = _run(wt)
        print(f"baseline: {len(base_fails)} failing assertion(s)", file=sys.stderr)
        for n, item in enumerate(sorted(pick, key=lambda x: (x["doc"], x["line"])),
                                 start=1):
            p = wt / item["doc"]
            orig = p.read_text(encoding="utf-8")
            lines = orig.splitlines()
            row = lines[item["line"] - 1]
            was = item["value"]
            head, dot, tail = was.partition(".")
            now = f"{int(head) + 7}.{tail}"
            if was not in row:
                out.append({**item, "probe": "moved"})
                continue
            lines[item["line"] - 1] = row.replace(was, now, 1)
            p.write_text("\n".join(lines) + "\n", encoding="utf-8")
            try:
                fails, crashed = _run(wt)
            finally:
                p.write_text(orig, encoding="utf-8")
            caught = bool(fails - base_fails) or (crashed and not base_crash)
            out.append({**item, "probe": "caught" if caught else "SURVIVED",
                        "perturbation": f"{was} -> {now}"})
            print(f"  [{n}/{len(pick)}] {item['doc']}:{item['line']} {was} -> {now}  "
                  f"{'caught' if caught else 'SURVIVED'}", file=sys.stderr, flush=True)
    return out


def census() -> dict:
    known = parsed_headers()
    covered, uncovered, no_values = [], [], []
    for rel in DOCS:
        for t in tables(rel):
            hn = norm(t["header"])
            if any(rel == doc and hn.startswith(norm(p)) for doc, p in known):
                covered.append(t)
            elif t["value_cells"] < 3:
                no_values.append(t)
            else:
                uncovered.append(t)
    numeric = len(covered) + len(uncovered)
    return {"documents": len(DOCS), "documents_excluded": len(EXCLUDED),
            "tables": numeric + len(no_values),
            "carrying_values": numeric,
            "parsed": len(covered),
            "not_parsed": len(uncovered),
            "no_values": len(no_values),
            "coverage_pct": round(100.0 * len(covered) / numeric, 1) if numeric else 0.0,
            "uncovered": sorted(uncovered, key=lambda t: -t["value_cells"]),
            "covered": sorted(covered, key=lambda t: -t["value_cells"])}


def _worktree(stack: contextlib.ExitStack) -> pathlib.Path:
    """A throwaway copy of the working tree, so probing never dirties the real one.

    The probe writes a wrong number into a published document and restores it
    afterwards. A crash, a timeout or a Ctrl-C between those two writes would
    leave the document wrong, so neither write touches the repository the user
    is working in. `git worktree` shares `.git`, which the claim checker needs
    for the provenance assertions, and uncommitted edits are copied across so
    the probe measures what is on disk rather than what is committed.
    """
    dest = pathlib.Path(tempfile.mkdtemp(prefix="table-probe-"))/ "wt"
    stack.callback(shutil.rmtree, dest.parent, ignore_errors=True)
    subprocess.run(["git", "worktree", "add", "--detach", "-q", str(dest), "HEAD"],
                   cwd=ROOT, check=True, capture_output=True)
    stack.callback(lambda: subprocess.run(
        ["git", "worktree", "remove", "--force", str(dest)],
        cwd=ROOT, capture_output=True))
    # tracked-and-modified plus untracked: this file itself is usually the
    # second of those, and a worktree without it measures the wrong tree
    dirty = subprocess.run(["git", "diff", "--name-only", "HEAD"], cwd=ROOT,
                           check=True, capture_output=True, text=True).stdout.split()
    dirty += subprocess.run(["git", "ls-files", "--others", "--exclude-standard"],
                            cwd=ROOT, check=True, capture_output=True,
                            text=True).stdout.split()
    for rel in dirty:
        src = ROOT / rel
        if src.is_file():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / rel)
    return dest


_FAILED = re.compile(r"^  FAIL  (.*?)\s+got=", re.M)


def _run(wt: pathlib.Path) -> tuple[set[str], bool]:
    """The checker's failure set, and whether it fell over instead of reporting."""
    r = subprocess.run([sys.executable, "analysis/verify_claims.py"],
                       cwd=wt, capture_output=True, text=True, timeout=1800)
    crashed = "Traceback (most recent call last)" in r.stderr
    return set(_FAILED.findall(r.stdout)), crashed


def probe(candidates: list[dict]) -> list[dict]:
    """Ground truth: change one cell, run the checker, see whether it notices.

    The structural census is a proxy and it undercounts, because a cell can
    also be guarded by a literal grep somewhere in the checker without its
    table ever being parsed. Only this distinguishes the two.

    The verdict is NOT the exit status. A first version of this used it, and
    every table came back "caught" - because the checker had one unrelated
    failing assertion at the time, so it exited non-zero whatever was done to
    the documents. A probe whose control and treatment give the same answer
    measures nothing. So the baseline failure set is taken first and a
    perturbation counts as caught only if it produces a failure that is not in
    it, or makes the checker crash when the baseline did not.
    """
    # Integers count. A first version only matched `d+.d+` and looked eight
    # lines past the header, so 15 of the 95 tables were reported as probed
    # when nothing in them had been touched - a silent no-op that read as
    # coverage. The whole table body is searched now, decimals first because a
    # decimal is the least ambiguous thing to perturb.
    dec = re.compile(r"(?<![\w.])(\d+)\.(\d+)(?![\w.])")
    integer = re.compile(r"(?<![\w.])(\d+)(?![\w.])")
    out = []
    with contextlib.ExitStack() as stack:
        wt = _worktree(stack)
        base_fails, base_crash = _run(wt)
        print(f"baseline: {len(base_fails)} failing assertion(s)"
              f"{', CRASHED' if base_crash else ''}", file=sys.stderr, flush=True)
        for name in sorted(base_fails):
            print(f"  standing failure (ignored): {name}", file=sys.stderr)
        for n, t in enumerate(candidates, start=1):
            p = wt / t["doc"]
            orig = p.read_text(encoding="utf-8")
            lines = orig.splitlines()
            body = []
            for j in range(t["line"] + 1, len(lines)):
                if not lines[j].strip().lstrip("> ").strip().startswith("|"):
                    break
                body.append(j)
            row = next((j for j in body if dec.search(lines[j])), None)
            num = dec
            if row is None:
                row = next((j for j in body if integer.search(lines[j])), None)
                num = integer
            if row is None:
                out.append({**t, "probe": "no numeric cell"})
                continue
            m = num.search(lines[row])
            was = m.group(0)
            now = (f"{int(m.group(1)) + 7}.{m.group(2)}" if num is dec
                   else str(int(m.group(1)) + 7))
            lines[row] = lines[row][:m.start()] + now + lines[row][m.end():]
            p.write_text("\n".join(lines) + "\n", encoding="utf-8")
            try:
                fails, crashed = _run(wt)
            finally:
                p.write_text(orig, encoding="utf-8")
            new_fails = sorted(fails - base_fails)
            caught = bool(new_fails) or (crashed and not base_crash)
            out.append({**t, "probe": "caught" if caught else "SURVIVED",
                        "perturbation": f"{was} -> {now}",
                        "noticed_by": new_fails[:2]})
            print(f"  [{n}/{len(candidates)}] {t['doc']}:{t['line']} "
                  f"{was} -> {now}  {'caught' if caught else 'SURVIVED'}",
                  file=sys.stderr, flush=True)
    return out


def main() -> None:
    unknown = [a for a in sys.argv[1:]
               if a not in ("--json", "--probe", "--covered", "--prose")]
    if unknown:
        sys.exit(f"unrecognised option(s): {' '.join(unknown)}")
    if "--prose" in sys.argv:
        pc = prose_census()
        if "--probe" in sys.argv:
            pc["sample"] = prose_probe(pc["absent"])
        pc.pop("absent")
        if "--json" in sys.argv:
            print(json.dumps(pc, indent=2))
            return
        print(f"decimal numbers in prose:    {pc['prose_numbers']}")
        print(f"  not a literal in the checker: {pc['not_a_literal']}")
        if "sample" in pc:
            got = [x for x in pc["sample"] if x.get("probe") in ("caught", "SURVIVED")]
            surv = sum(1 for x in got if x["probe"] == "SURVIVED")
            print(f"  probed sample:              {len(got)}")
            print(f"  accepted a wrong number:    {surv}")
        return
    c = census()
    # --covered is the converse check, and the one that keeps the census
    # honest: parsing a table is only coverage if perturbing a cell actually
    # fails the checker. A parser can match a header, return rows, and assert
    # nothing about the column you changed.
    _key = "covered" if "--covered" in sys.argv else "uncovered"
    if "--probe" in sys.argv:
        c[_key] = probe(c[_key])
    if "--json" in sys.argv:
        print(json.dumps(c, indent=2))
        return
    print(f"documents censused:          {c['documents']}"
          f"  ({c['documents_excluded']} excluded, see EXCLUDED)")
    print(f"published tables:            {c['tables']}")
    print(f"  carrying measurements:     {c['carrying_values']}")
    print(f"  parsed cell by cell:       {c['parsed']}  ({c['coverage_pct']} %)")
    print(f"  NOT parsed:                {c['not_parsed']}")
    print(f"  paths, links or prose:     {c['no_values']}")
    print(f"\n{'parsed' if _key == 'covered' else 'not parsed'}, most numbers first:")
    for t in c[_key][:20]:
        mark = f"  {t.get('probe', '')}" if "probe" in t else ""
        print(f"  {t['doc']:32s}:{t['line']:<5} cells={t['value_cells']:<4} "
              f"{t['header'][:52]}{mark}")


if __name__ == "__main__":
    main()
