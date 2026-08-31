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
import hashlib
import contextlib
import json
import pathlib
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ["README.md", "ERRATA.md", "CHANGELOG.md", "RETEST_TODO.md",
        "BENCHMARK_ENV.md", "PULL_REQUEST.md",
        "v4_audit_2026_08_25/README.md",
        "v4_audit_2026_08_25/PREREGISTERED_PREDICTION.md",
        "v4_audit_2026_08_25/PROSPECTIVE_ANALYSIS_PLAN_W.md",
        "v4_audit_2026_08_25/PROSPECTIVE_ANALYSIS_PLAN_W2.md",
        "RELEASE_NOTES_v4.2.md"]
# Not censused, and why. Nothing may be missing from DOCS + EXCLUDED: a new
# document that is in neither would escape the count silently, which is the
# failure this whole file exists to stop.
EXCLUDED = {
    "v2_3090_followup/README.md": "archived artefact; ERRATA quotes it, it is not re-derived",
    "v2_3090_followup/SUMMARY.md": "archived artefact; ERRATA quotes it, it is not re-derived",
    "v3_dflash_2026_05_07/README.md": "archived artefact; ERRATA quotes it, it is not re-derived",
    "v4_audit_2026_08_25/harness/README.md": "provenance note; hashes, not measurements",
    # Its tables are two runner hashes and a hunk-by-hunk classification of a
    # diff. Not measurements -- but the diff's own line counts ARE checkable, so
    # verify_claims asserts them against the archived file rather than leaving
    # them to a table parser that would have nothing else to read.
    # A procedure, not a result: its only numbers are a commit hash and shell
    # in a fenced block, and the counts it tells the operator to publish are
    # asserted where they are computed rather than quoted here.
    "v4_audit_2026_08_25/RELEASE_PROCEDURE.md":
        "the release procedure; a commit hash and commands, no measurement",
    "v4_audit_2026_08_25/harness/V3_TO_W_DIFF.md":
        "provenance note; runner hashes and a diff classification, and its counts are checked directly",
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
              "_ch_table": "CHANGELOG.md",
              # found by the header-literal scan below, which is why that scan
              # exists: a whole reader was missing from this map and both
              # tables it reads were being counted as unparsed
              "_pre_table":
                  "v4_audit_2026_08_25/PREREGISTERED_PREDICTION.md",
              "_rn_table": "RELEASE_NOTES_v4.2.md"}
# The generic readers take the document's LINES as their first argument and the
# header as their second, so the document has to come from the variable name.
# Adding one of these without adding it here counts its tables as unparsed,
# which is the mistake `_pre_table` made above in the other direction.
LINES_DOC = {"_ER_LINES": "ERRATA.md",
             "_RM_LINES": "README.md",
             "_ROOT_LINES": "README.md",
             "_V4R_LINES": "v4_audit_2026_08_25/README.md",
             "_PR_LINES": "PULL_REQUEST.md",
             "_CH_LINES": "CHANGELOG.md",
             "_BE_LINES": "BENCHMARK_ENV.md",
             "_RT_LINES": "RETEST_TODO.md"}
GENERIC_READERS = {"_num_rows", "_num_rows_seq"}
DASH = {"−": "-", "–": "-", "—": "-"}
# a cell that is a measurement rather than a path, hash, date, link or issue
NOT_A_VALUE = re.compile(r"https?://|\.py|\.json|\.md|\.log|\.sh|\.cff"
                         r"|`[0-9a-f]{7,}|\d{4}-\d{2}-\d{2}|#\d|[Ss][Hh][Aa]-?\d")
NUMERIC = re.compile(r"\d+\.\d+|\d{2,}")


def norm(s: str) -> str:
    for a, b in DASH.items():
        s = s.replace(a, b)
    return s


# Read by a bespoke loop in `verify_claims.py` rather than by one of the readers
# above, so the AST scan cannot see them. Declared here, and the declaration is
# a claim rather than a courtesy: `--probe --covered` perturbs them like any
# other and they come back caught. Four of these were being counted as unparsed
# and were about to be parsed a second time.
HAND_PARSED = {
    ("README.md", "| arm | pooled tok/s | change |"),
    ("README.md", "| arm | O2 | O3 | shift |"),
    ("README.md", "| configuration | real acceptance | pooled tok/s | vs baseline |"),
    ("README.md", "| | seconds | share |"),
    ("ERRATA.md", "| arm | freerun, as the archive did it | hard cap | shift |"),
    # one entry, two tables: runs A and B share a header and are located by
    # index, so `_v4_table` cannot tell them apart and neither can the scan
    ("v4_audit_2026_08_25/README.md", "| arm | request-mean | pooled | min |"),
    # one table in three documents, read by index and required to agree
    ("README.md", "| method | thinking on"),
    ("ERRATA.md", "| method | thinking on"),
    ("v4_audit_2026_08_25/README.md", "| method | thinking on"),
}


def header_literals() -> set[str]:
    """Every string in the checker that looks like a table header.

    A bespoke reader - `next(l for l in LINES if l.startswith("| arm | ..."))` -
    is invisible to the scan above, and four tables were counted as unparsed
    while being read that way. This finds the literal wherever it sits; the test
    in `tests/` requires each one to be either a reader argument or an entry in
    HAND_PARSED, so a new bespoke reader cannot go unrecorded.
    """
    src = (ROOT / "analysis" / "verify_claims.py").read_text(encoding="utf-8")
    return {n.value for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and n.value.startswith("| ") and n.value.count("|") >= 2}


def parsed_headers() -> set[tuple[str, str]]:
    """(document, header) for every table the checker feeds to a reader."""
    src = (ROOT / "analysis" / "verify_claims.py").read_text(encoding="utf-8")
    out = set()
    for n in ast.walk(ast.parse(src)):
        if not isinstance(n, ast.Call):
            continue
        fn = getattr(n.func, "id", "")
        if (fn in READER_DOC and n.args
                and isinstance(n.args[0], ast.Constant)):
            out.add((READER_DOC[fn], n.args[0].value))
        elif (fn in GENERIC_READERS and len(n.args) >= 2
                and isinstance(n.args[0], ast.Name)
                and isinstance(n.args[1], ast.Constant)
                and n.args[0].id in LINES_DOC):
            out.add((LINES_DOC[n.args[0].id], n.args[1].value))
    return out | HAND_PARSED


def _cells_with_numbers(row: str) -> list[str]:
    """The cells of one row that hold at least one probeable number."""
    return [row[a:b] for a, b in _pipe_spans(row)
            if any(True for _ in _numbers_in(row, (a, b)))]


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
        # a cell carries a value if the probe would perturb something in it,
        # so the census and the probe cannot disagree about what a value is
        values = [c for r in body
                  for c in _cells_with_numbers(r)]
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
    """Every decimal number in prose: outside tables, fenced blocks and CODE.

    An indented line used to be skipped unconditionally, as an indented code
    block. Inside a list item four spaces is not a code block, it is how a
    paragraph continues the item, and a code block there needs eight. So the
    continuation text of every numbered item in ERRATA and CHANGELOG was outside
    this population: 57 decimals on 2026-09-01, 4 % of it, and not marginal
    ones. They include "-2.4 is inside [-2.61, +0.22]", the corrected within-run
    spread of "0.90 % to 1.75 %", the withdrawn "101.3 MiB per checkpoint" and
    the end-to-end "30.2 ... 8.8 tok/s". A19's own list of corrections, which is
    the longest run of prose in this repository, was the largest block of it.
    Understating the denominator of a coverage measurement flatters the
    coverage, which is the one direction an audit must not be wrong in.
    """
    out = []
    dec = re.compile(r"(?<![\w.])\d+\.\d+(?![\w])")
    marker = re.compile(r"^(?:\d+[.)]|[-*+])\s")
    for rel in DOCS:
        fenced = False
        in_item = False
        for i, raw in enumerate((ROOT / rel).read_text(encoding="utf-8").splitlines()):
            line = raw.strip().lstrip("> ").strip()
            if line.startswith("```"):
                fenced = not fenced
                continue
            if fenced or line.startswith("|"):
                continue
            indent = 8 if raw.startswith("\t") else len(raw) - len(raw.lstrip(" "))
            if line and indent == 0:
                in_item = bool(marker.match(line))
            if raw.startswith(("    ", "\t")) and not (in_item and indent < 8):
                continue
            # The SPAN, not just the value. Two identical decimals on one line
            # produced two census records, and `prose_probe` perturbed the first
            # occurrence for both -- so "40 numbers probed" could be fewer than
            # 40 distinct locations, and one of them was never touched. The
            # offsets are into `raw`, which is what the probe rewrites.
            off = raw.index(line) if line and line in raw else 0
            for m in dec.finditer(line):
                out.append({"doc": rel, "line": i + 1, "value": m.group(0),
                            "start": off + m.start(), "end": off + m.end()})
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
        base_fails, base_n, base_rc = _run(wt)
        _CTRL["before"] = bool(base_fails) or base_rc != 0
        print(f"baseline: {base_n} assertions, {len(base_fails)} failing, "
              f"exit {base_rc}", file=sys.stderr)
        # same rule as `cell_probe`: a control that does not pass makes every
        # "caught" reading suspect, and suspect in the reassuring direction
        if base_fails or base_rc != 0:
            raise SystemExit(
                f"the checker does not pass on an unperturbed worktree "
                f"({len(base_fails)} failing, exit {base_rc}), so nothing this "
                f"probe reports would mean anything.\n  "
                + "\n  ".join(sorted(base_fails)[:5]))
        for n, item in enumerate(sorted(pick, key=lambda x: (x["doc"], x["line"])),
                                 start=1):
            p = wt / item["doc"]
            orig = p.read_text(encoding="utf-8")
            lines = orig.splitlines()
            row = lines[item["line"] - 1]
            was = item["value"]
            head, dot, tail = was.partition(".")
            now = f"{int(head) + 7}.{tail}"
            a, b = item.get("start"), item.get("end")
            # Perturb THAT occurrence, by offset. `row.replace(was, now, 1)`
            # always hit the first one, so a line carrying the same decimal
            # twice had one of its two census records probing the other's
            # position and the second number was never perturbed at all.
            if a is None or b is None or row[a:b] != was:
                out.append({**item, "probe": "moved"})
                continue
            lines[item["line"] - 1] = row[:a] + now + row[b:]
            p.write_text("\n".join(lines) + "\n", encoding="utf-8")
            try:
                fails, ran, rc = _run(wt)
            finally:
                p.write_text(orig, encoding="utf-8")
            caught = (bool(fails - base_fails) or ran < base_n or rc != base_rc)
            out.append({**item, "probe": "caught" if caught else "SURVIVED",
                        "perturbation": f"{was} -> {now}"})
            print(f"  [{n}/{len(pick)}] {item['doc']}:{item['line']} {was} -> {now}  "
                  f"{'caught' if caught else 'SURVIVED'}", file=sys.stderr, flush=True)
        # the control again, for the reason `cell_probe` gives at the same spot
        end_fails, end_n, end_rc = _run(wt)
        _CTRL["after"] = bool(end_fails) or end_rc != 0 or end_n != base_n
        if end_fails or end_rc != 0 or end_n != base_n:
            raise SystemExit(
                f"the control no longer passes after the run "
                f"({len(end_fails)} failing, exit {end_rc}, {end_n} of "
                f"{base_n} assertions), so this sample means nothing.\n  "
                + "\n  ".join(sorted(end_fails)[:5]))
    return out


# Tables that carry digits but no measurement, each named and reasoned rather
# than caught by a threshold. The census used to file anything with fewer than
# three numeric cells as "paths, links or prose", which is a different claim
# from "no derivable numbers" and let a one- or two-value RESULT table leave the
# coverage population without anyone deciding that it should. Zero is the
# threshold now, and these five are listed because a reviewer looked at each.
#
# Keyed on (document, first line of the header row) so it survives the table
# moving within its file. A stale entry -- one that matches no table -- fails.
EXCLUDED_TABLES = {
    ("README.md", "| Open upstream | Status | What it would change here |"):
        "upstream tracker; the digits are PR and issue numbers and dates",
    ("ERRATA.md", "| | `97895129e` (v1) | `3737e4137` (audit) |"):
        "commit hashes and source line numbers, not measurements",
    ("RETEST_TODO.md", "| type | why it cannot run here |"):
        "drafter types and the loader errors that reject them; the digit is a source line",
    ("RETEST_TODO.md", "| Item | Status |"):
        "a written/not-written checklist; no quantity in it",
    ("v4_audit_2026_08_25/README.md", "| arm | what it is |"):
        "arm definitions; the digit is a tokenizer id passed with --override-kv",
    ("RELEASE_NOTES_v4.2.md", "| asset | bytes | sha256 |"):
        "release asset digests; a sha256 is not a measurement and parsing one "
        "as numbers yields garbage. The bytes and the digests are checked "
        "against evidence.yml and the audit README by name in verify_claims.py",
}

def census() -> dict:
    known = parsed_headers()
    covered, uncovered, no_values, excluded, seen_ex = [], [], [], [], set()
    for rel in DOCS:
        for t in tables(rel):
            hn = norm(t["header"])
            # Order matters, and it was wrong. Asking "is it parsed?" first put
            # `README.md`'s `| Path | Contents |` -- zero numeric cells, read by
            # a parser that wants its paths -- into `carrying_values`, so the
            # published "125 tables carry measurements" counted one that
            # carries none. Whether a parser reads a table says nothing about
            # whether the table has a measurement in it, so the population is
            # decided first and coverage second.
            if (rel, t["header"]) in EXCLUDED_TABLES:
                seen_ex.add((rel, t["header"]))
                excluded.append(dict(t, reason=EXCLUDED_TABLES[(rel, t["header"])]))
            elif t["value_cells"] == 0:
                # ZERO, not "fewer than three". The threshold used to be < 3, so a
                # table carrying one or two measurements was filed as "paths, links
                # or prose" and left the coverage population entirely -- and a
                # two-cell before/after table can be the most important result in a
                # document. "Few numbers" and "no derivable numbers" are different
                # claims, and only the second one licenses skipping the table.
                no_values.append(t)
            elif any(rel == doc and hn.startswith(norm(p)) for doc, p in known):
                covered.append(t)
            else:
                uncovered.append(t)
    numeric = len(covered) + len(uncovered)
    # A registry entry that matches nothing is a claim about a table that is no
    # longer there, and it would silently shrink the population it was written
    # to keep honest.
    stale = sorted(k for k in EXCLUDED_TABLES if k not in seen_ex)
    return {"documents": len(DOCS), "documents_excluded": len(EXCLUDED),
            "tables": numeric + len(no_values) + len(excluded),
            "carrying_values": numeric,
            "parsed": len(covered),
            "not_parsed": len(uncovered),
            "no_values": len(no_values),
            "excluded_tables": len(excluded),
            "excluded_stale": stale,
            "excluded_detail": excluded,
            "coverage_pct": round(100.0 * len(covered) / numeric, 1) if numeric else 0.0,
            "uncovered": sorted(uncovered, key=lambda t: -t["value_cells"]),
            "covered": sorted(covered, key=lambda t: -t["value_cells"])}


def _unwind_on_signal(stack: contextlib.ExitStack) -> None:
    """Make SIGTERM and SIGINT unwind the ExitStack instead of skipping it.

    Python's default SIGTERM handling ends the process without running
    `finally` blocks, `atexit` or an ExitStack's callbacks, so a probe stopped
    with `pkill` leaves its worktree behind: 163 MB of checkout, plus an entry
    in the repository's worktree registry. Seven of them, from runs stopped on
    2026-08-29 and 2026-08-30, had `/tmp` at 176 MB free on a 16 GB tmpfs.

    Raising SystemExit from the handler is what unwinds the stack. SIGKILL
    cannot be caught; since the copy became a clone rather than a `git
    worktree` there is no registry entry to leave behind, only a directory
    under the system temporary directory, which is what `mkdtemp` is for.
    """
    def _bail(signum, _frame):
        raise SystemExit(f"stopped by signal {signum}")

    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_sig, _bail)
        except ValueError:
            pass            # not the main thread; the ExitStack still runs


def _worktree(stack: contextlib.ExitStack) -> pathlib.Path:
    """A throwaway copy of the tree, so probing never dirties the real one.

    The probe writes a wrong number into a published document and restores it
    afterwards. A crash, a timeout or a Ctrl-C between those two writes would
    leave the document wrong, so neither write touches the repository the
    operator is working in.

    A CLONE, not a `git worktree`. A worktree shares one `.git`, and the claim
    checker's provenance assertions run `git` on every invocation, so sixteen
    shards against one object store made three of those assertions flake on
    2026-08-29 and the run had to be held to eight. `git clone --local`
    hardlinks the object store, which costs nothing on disk and takes a second,
    and gives each shard its own refs, index and locks. Twenty-eight shards
    then fit on a thirty-two core host at 185 MB each, and the run that took
    110 minutes takes about thirty.

    Uncommitted edits are copied across, so the probe measures what is on disk
    rather than what is committed.
    """
    _unwind_on_signal(stack)
    dest = pathlib.Path(tempfile.mkdtemp(prefix="table-probe-")) / "wt"
    stack.callback(shutil.rmtree, dest.parent, ignore_errors=True)
    head = _head_sha() or "HEAD"
    subprocess.run(["git", "clone", "--local", "--quiet", str(ROOT), str(dest)],
                   check=True, capture_output=True)
    # detached at exactly the commit the launch tree is on, so `head_sha` in the
    # attestation is the commit that was measured and not whatever branch the
    # clone happened to check out
    subprocess.run(["git", "-C", str(dest), "checkout", "--detach", "--quiet", head],
                   check=True, capture_output=True)
    # tracked-and-modified plus untracked: this file itself is usually the
    # second of those, and a copy without it measures the wrong tree
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


_RAN = re.compile(r"^  (?:PASS|FAIL)  ", re.M)


def _run(wt: pathlib.Path) -> tuple[set[str], int, int]:
    """The checker's failure set, how many assertions it ran, and its status.

    Three signals because two were not enough, twice. Reading only the FAIL
    lines missed a perturbation that made the checker abort with
    `SystemExit("unrecognised run C row label")`, which prints no traceback and
    adds no failure. Adding the exit status missed the same thing whenever the
    baseline was already exiting 1 for an unrelated reason: an abort also exits
    1, so the comparison saw no change and called it unnoticed. The count of
    assertions that ran is the signal that does not care - an abort always runs
    fewer - and it is the one that would have caught both.
    """
    r = subprocess.run([sys.executable, "analysis/verify_claims.py"],
                       cwd=wt, capture_output=True, text=True, timeout=1800)
    return (set(_FAILED.findall(r.stdout)), len(_RAN.findall(r.stdout)),
            r.returncode)


def _pipe_spans(raw: str) -> list[tuple[int, int]]:
    """(start, end) of each cell's text in a raw markdown row, pipes excluded.

    Offsets into the raw line, so a blockquote marker or any spacing survives
    the edit. Splitting and rejoining would reformat the row, and a probe that
    also reformats cannot say which change the checker noticed.
    """
    bars = [k for k, ch in enumerate(raw) if ch == "|"]
    return [(bars[k] + 1, bars[k + 1]) for k in range(len(bars) - 1)]


def _token_around(text: str, at: int) -> str:
    """The whitespace-delimited token containing offset `at`.

    `NOT_A_VALUE` used to be applied to the whole cell, so one commit hash in
    a cell hid every real number beside it: the run registry's `3 arms x 10
    prompts; A at \u0060bcb5eeb64\u0060, 2 repeats` offered nothing to probe at
    all. Excluding the token instead keeps the hash, the date and the issue
    number out while leaving the measurements in.
    """
    a = text.rfind(" ", 0, at) + 1
    b = text.find(" ", at)
    return text[a:] if b < 0 else text[a:b]


def _numbers_in(raw: str, span: tuple[int, int]):
    """Every number inside one cell, as (absolute_start, absolute_end, text).

    Every number, not the first: an interval cell holds two, and perturbing
    only the low bound leaves the high one untested. The W table passed a
    one-number-per-table probe with two of its three columns unread; this is
    the resolution that would have caught it.

    Decimals and integers both, and this had the same bug it exists to find.
    The first version fell back to integers only when the cell held no
    decimal, so `p_min 0 / 0.50 / 0.75 / 0.90 at n_max 8` was probed on its
    four decimals and never on the 8. Both lookarounds already exclude the
    halves of a decimal, so there is nothing for the fallback to protect
    against and it only hid numbers.
    """
    dec = re.compile(r"(?<![\w.])\d+\.\d+(?![\w.])")
    integer = re.compile(r"(?<![\w.])\d+(?![\w.])")
    text = raw[span[0]:span[1]]
    seen = []
    for pat in (dec, integer):
        for m in pat.finditer(text):
            if NOT_A_VALUE.search(_token_around(text, m.start())):
                continue
            seen.append((span[0] + m.start(), span[0] + m.end(), m.group(0)))
    yield from sorted(seen)


def _bump(was: str) -> str:
    """Add seven to the integer part, keeping the shape of the number."""
    head, dot, tail = was.partition(".")
    return f"{int(head) + 7}.{tail}" if dot else str(int(head) + 7)


def cell_population(tables_: list[dict]) -> int:
    """How many numbers `cell_probe` would perturb over these tables.

    A function rather than a loop inside `--count`, because the published
    figure for it is a number in a document like any other and the claim
    checker has to be able to derive it. Two copies of one derivation is the
    defect A12 was written about: the second copy is the one nobody reads.
    """
    n = 0
    for t in tables_:
        lines = (ROOT / t["doc"]).read_text(encoding="utf-8").splitlines()
        for j in range(t["line"] + 1, len(lines)):
            if not lines[j].strip().lstrip("> ").strip().startswith("|"):
                break
            n += sum(1 for span in _pipe_spans(lines[j])
                     for _ in _numbers_in(lines[j], span))
    return n


# `checker_sha` and `head` are recorded from the WORKTREE the shard measured
# in, not from the repository the shard was launched from. `_checker_sha()`
# below reads this file's own parent, which is the launch tree, and a launch
# tree that changes mid-run gives shards that finish at different times
# different hashes for a checker none of them used. The aggregator refused a
# real run for that on 2026-09-01 and its diagnosis was right for the wrong
# reason: the shards had measured with one checker and reported two.
_CTRL = {"before": None, "after": None, "checker_sha": None, "head": None,
         "population_sha": None}


def _head_sha(root: pathlib.Path | None = None) -> str:
    """The commit these shards were taken on. Eight attestations from different
    heads are eight measurements of different documents.

    `root` is the worktree when a shard has one, because that is the tree whose
    documents were perturbed. Defaulting to the launch tree keeps every other
    caller working.
    """
    try:
        return subprocess.run(
            ["git", "-C", str(root or ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                              # noqa: BLE001
        return ""


def _sha_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _checker_sha() -> str:
    """The checker the shard ran. A perturbation is caught BY something, and if
    two shards ran different checkers their results are not one population."""
    f = ROOT / "analysis" / "verify_claims.py"
    return hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else ""


def _population_sha(tables_: list[dict]) -> str:
    """A digest of the whole census population, not of this shard's slice.

    Every shard computes it over the same input, so a mismatch says the
    documents moved under one of them.
    """
    c = census()
    key = json.dumps([[t["doc"], t["line"], t["header"], t["value_cells"]]
                      for t in sorted(c["covered"] + c["uncovered"],
                                      key=lambda t: (t["doc"], t["line"]))],
                     sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()


def cell_probe(tables_: list[dict], shard: tuple[int, int] = (0, 1)) -> list[dict]:
    """Perturb EVERY numeric cell of every table given, one at a time.

    The one-cell-per-table probe is too weak to say a table is guarded: the W
    three-design table passed it while two of its three columns were unread,
    because the cell it happened to pick was in the third. This is the version
    that would have caught that, and it is why it exists.

    `shard` is (index, count); tables are dealt out round robin so several
    processes can run at once, each in its own worktree.
    """
    out = []
    picked = [t for k, t in enumerate(tables_) if k % shard[1] == shard[0]]
    with contextlib.ExitStack() as stack:
        wt = _worktree(stack)
        base_fails, base_n, base_rc = _run(wt)
        # RECORDED, not assumed. `_CTRL` was written by `prose_probe` alone, so
        # a cell-probe attestation carried `control_before: pass` whichever way
        # the control had gone: `"pass" if not _CTRL["before"]` reads `None` as
        # passing. The refusal below is what actually protected the run, and a
        # field that is true whatever happens is the shape this repository
        # spends its time removing.
        _CTRL["before"] = bool(base_fails) or base_rc != 0
        _CTRL["checker_sha"] = _sha_file(wt / "analysis" / "verify_claims.py")
        _CTRL["head"] = _head_sha(wt)
        # Computed IN the worktree, by the worktree's own copy of this module,
        # for the reason the two above are: `census()` reads `DOCS` under this
        # module's `ROOT`, which is the tree the shard was launched from. On
        # 2026-09-01 that tree was edited while eight shards ran and they
        # reported two population digests for one set of documents none of them
        # had seen change. `head_sha` and `checker_sha256` had been moved to the
        # worktree an hour earlier and this one was missed, which is why the
        # aggregator refused a second consecutive run.
        _pop = subprocess.run(
            [sys.executable, "analysis/table_coverage.py", "--population-sha"],
            cwd=wt, capture_output=True, text=True, timeout=600)
        _CTRL["population_sha"] = _pop.stdout.strip() if _pop.returncode == 0 else ""
        print(f"shard {shard[0]}/{shard[1]}: {len(picked)} tables, baseline "
              f"{base_n} assertions, {len(base_fails)} failing, exit {base_rc}",
              file=sys.stderr, flush=True)
        # A failing control biases the result in the reassuring direction: a
        # perturbation is "caught" when the failure set grows, so any failure
        # the baseline did not have counts, including one that has nothing to
        # do with the number that was changed. Sixteen shards sharing one
        # `.git` produced three such failures on 2026-08-29 and one shard alone
        # produced none, so the reading was contention and the result would
        # have been a clean run that measured nothing. This file has now had to
        # record a broken control three times; it refuses instead.
        if base_fails or base_rc != 0:
            raise SystemExit(
                f"shard {shard[0]}/{shard[1]}: the checker does not pass on an "
                f"unperturbed worktree ({len(base_fails)} failing, exit "
                f"{base_rc}), so nothing this shard reports would mean "
                f"anything. Fix the tree, or run fewer shards at once: several "
                f"checkers against one `.git` make the git-gated assertions "
                f"flake.\n  " + "\n  ".join(sorted(base_fails)[:5]))
        for t in picked:
            p = wt / t["doc"]
            orig = p.read_text(encoding="utf-8")
            lines = orig.splitlines()
            body = []
            for j in range(t["line"] + 1, len(lines)):
                if not lines[j].strip().lstrip("> ").strip().startswith("|"):
                    break
                body.append(j)
            for j in body:
                raw = orig.splitlines()[j]
                targets = [(col, a, b, txt)
                           for col, span in enumerate(_pipe_spans(raw))
                           for a, b, txt in _numbers_in(raw, span)]
                for col, a, b, was in targets:
                    now = _bump(was)
                    lines[j] = raw[:a] + now + raw[b:]
                    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    try:
                        fails, ran, rc = _run(wt)
                    finally:
                        lines[j] = raw
                        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    new_fails = sorted(fails - base_fails)
                    # An abort is a detection and it shows as a short run --
                    # but a short run is also what a transient failure looks
                    # like, and eight worktrees sharing one `.git` produced
                    # three of those on 2026-08-29. A NAMED new failure is
                    # attributable to the perturbation; a bare assertion-count
                    # drop or exit-code change is not, so it is confirmed by
                    # repeating the same mutation once before being believed.
                    if new_fails:
                        caught, caught_by = True, "assertion"
                    elif ran < base_n or rc != base_rc:
                        lines[j] = raw[:a] + now + raw[b:]
                        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
                        try:
                            f2, ran2, rc2 = _run(wt)
                        finally:
                            lines[j] = raw
                            p.write_text("\n".join(lines) + "\n", encoding="utf-8")
                        again = sorted(f2 - base_fails) or ran2 < base_n or rc2 != base_rc
                        caught = bool(again)
                        caught_by = "abort, repeated" if caught else "abort, not repeatable"
                        if not caught:
                            print(f"  FLAKE {t['doc']}:{j + 1} col{col} {was} -> {now}: "
                                  f"the first reading aborted and the repeat did not",
                                  file=sys.stderr, flush=True)
                    else:
                        caught, caught_by = False, "nothing"
                    rec = {"doc": t["doc"], "table_line": t["line"],
                           "row": j + 1, "col": col,
                           # the exact character span, so a cell holding two
                           # numbers yields two DIFFERENT location keys. Without
                           # it the shard attestations below could not tell one
                           # perturbation from the other and a union check would
                           # silently accept a missing one.
                           "start": a, "end": b,
                           "perturbation": f"{was} -> {now}",
                           "probe": "caught" if caught else "SURVIVED",
                           "caught_by": caught_by,
                           "noticed_by": new_fails[:1]
                           or ([f"aborted after {ran} of {base_n}"]
                               if ran < base_n else [f"exit {rc}"])}
                    out.append(rec)
                    if not caught:
                        print(f"  SURVIVED {t['doc']}:{j + 1} col{col} "
                              f"{was} -> {now}", file=sys.stderr, flush=True)
            p.write_text(orig, encoding="utf-8")
            _n = sum(1 for r in out if r["table_line"] == t["line"]
                     and r["doc"] == t["doc"])
            _s = sum(1 for r in out if r["table_line"] == t["line"]
                     and r["doc"] == t["doc"] and r["probe"] == "SURVIVED")
            print(f"  {t['doc']}:{t['line']}  {_n} cells, {_s} survived",
                  file=sys.stderr, flush=True)
        # The control again, after the work. Passing once at the start does
        # not bound half an hour of running beside seven other shards, and a
        # "caught" reading is only as good as the control at the moment it was
        # taken. It also catches the other way a long run goes wrong: a
        # perturbation that was written and not restored leaves the worktree
        # dirty, and every reading after it was against a different document.
        end_fails, end_n, end_rc = _run(wt)
        _CTRL["after"] = bool(end_fails) or end_rc != 0 or end_n != base_n
        if end_fails or end_rc != 0 or end_n != base_n:
            raise SystemExit(
                f"shard {shard[0]}/{shard[1]}: the control no longer passes "
                f"after the run ({len(end_fails)} failing, exit {end_rc}, "
                f"{end_n} of {base_n} assertions), so the catches it reported "
                f"cannot be trusted and neither can the survivors.\n  "
                + "\n  ".join(sorted(end_fails)[:5]))
    return out


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
        base_fails, base_n, base_rc = _run(wt)
        print(f"baseline: {base_n} assertions, {len(base_fails)} failing, "
              f"exit {base_rc}", file=sys.stderr, flush=True)
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
                fails, ran, rc = _run(wt)
            finally:
                p.write_text(orig, encoding="utf-8")
            new_fails = sorted(fails - base_fails)
            caught = bool(new_fails) or ran < base_n or rc != base_rc
            out.append({**t, "probe": "caught" if caught else "SURVIVED",
                        "perturbation": f"{was} -> {now}",
                        "noticed_by": new_fails[:2]})
            print(f"  [{n}/{len(candidates)}] {t['doc']}:{t['line']} "
                  f"{was} -> {now}  {'caught' if caught else 'SURVIVED'}",
                  file=sys.stderr, flush=True)
    return out


def aggregate(paths: list[str]) -> int:
    """Do these shard attestations cover the population exactly once each?

    The claim was "2373 numbers across 125 tables, all caught in eight shards".
    Nothing could check it. Not that all eight ran -- `--shard=8/8` selected no
    tables and exited 0. Not that their locations were disjoint. Not that their
    union was the whole population. Not that they were taken on one head with
    one checker. Not that each one's control passed at both ends.

    Every one of those is a way the sentence could be false while every
    individual shard output looked fine, so every one of them is checked here.
    """
    shards = []
    for f in paths:
        with open(f, encoding="utf-8") as fh:
            shards.append((f, json.load(fh)))
    bad = []
    if not shards:
        return _fail(["no shard attestations given"])

    counts = {tuple(d["shard"])[1] for _, d in shards}
    if len(counts) != 1:
        bad.append(f"the shards disagree on how many there are: {sorted(counts)}")
    n = sorted(counts)[0]
    seen = {}
    for f, d in shards:
        i, c = d["shard"]
        if i in seen:
            bad.append(f"shard {i}/{c} appears twice: {seen[i]} and {f}")
        seen[i] = f
        for k in ("control_before", "control_after"):
            if d.get(k) != "pass":
                bad.append(f"{f}: {k} is {d.get(k)!r}, so nothing it reports means anything")
    missing = sorted(set(range(n)) - set(seen))
    if missing:
        bad.append(f"shard(s) {missing} of {n} have no attestation")

    for field in ("head_sha", "checker_sha256", "population_sha256"):
        vals = {d.get(field) for _, d in shards}
        if len(vals) != 1:
            bad.append(f"the shards do not share one {field}: {sorted(vals)}")
        elif not sorted(vals)[0]:
            bad.append(f"{field} is empty in every shard, so nothing pins them together")

    union, dupes = set(), []
    for f, d in shards:
        for loc in d.get("locations", []):
            k = tuple(loc)
            if k in union:
                dupes.append((f, loc))
            union.add(k)
    if dupes:
        bad.append(f"{len(dupes)} location(s) are perturbed by more than one "
                   f"shard, so the shards are not a partition: {dupes[:3]}")

    want = {d.get("population_size") for _, d in shards}
    if len(want) == 1 and sorted(want)[0] is not None:
        w = sorted(want)[0]
        if len(union) != w:
            bad.append(f"the union covers {len(union)} locations and the "
                       f"population is {w}: {w - len(union)} never probed")
    surv = [r for _, d in shards for r in d.get("survived", [])]
    if surv:
        bad.append(f"{len(surv)} perturbation(s) survived")
    if bad:
        return _fail(bad)
    print(f"{len(shards)} shards, one head {shards[0][1]['head_sha'][:12]}, "
          f"one checker {shards[0][1]['checker_sha256'][:12]}, "
          f"{len(union)} locations covered exactly once, 0 survived")
    return 0


def _fail(msgs: list[str]) -> int:
    print("shard aggregation FAILED:", file=sys.stderr)
    for m in msgs:
        print(f"  {m}", file=sys.stderr)
    return 1


def main() -> None:
    shard = (0, 1)
    argv = []
    for a in sys.argv[1:]:
        if a.startswith("--shard="):
            spec = a.split("=", 1)[1]
            try:
                i_s, n_s = spec.split("/")
                i, n = int(i_s), int(n_s)
            except ValueError:
                sys.exit(f"--shard wants INDEX/COUNT, both integers; got {spec!r}")
            # Range-checked. Without this `--shard=8/8` selects nothing, because
            # `k % 8 == 8` is never true, and exits 0 reporting "0 numbers
            # perturbed, 0 survived" -- a shard that did no work and said it
            # succeeded. `--shard=-1/8` is the same, and `--shard=0/0` divides by
            # zero halfway through instead of at the boundary.
            if n < 1:
                sys.exit(f"--shard count must be at least 1; got {n}")
            if not 0 <= i < n:
                sys.exit(f"--shard index must be in [0, {n}); got {i}")
            shard = (i, n)
        else:
            argv.append(a)
    if "--population-sha" in argv:
        # one line, so a shard can ask the worktree what its own documents hash
        # to without importing this module across trees
        print(_population_sha(census()["covered"]))
        return
    if argv and argv[0] == "--aggregate":
        sys.exit(aggregate(argv[1:]))
    unknown = [a for a in argv
               if a not in ("--json", "--probe", "--covered", "--prose",
                            "--every-cell", "--count", "--aggregate",
                            "--population-sha")]
    if unknown:
        sys.exit(f"unrecognised option(s): {' '.join(unknown)}")
    if "--every-cell" in argv:
        c = census()
        picked = c["covered" if "--covered" in argv else "uncovered"]
        if "--count" in argv:
            print(f"{len(picked)} tables, {cell_population(picked)} numbers")
            return
        res = cell_probe(picked, shard)
        surv = [r for r in res if r["probe"] == "SURVIVED"]
        if "--json" in argv:
            # An ATTESTATION, not a summary. Eight shards were reported as
            # covering 2373 numbers across 125 tables and nothing in the
            # repository could show that: not that all eight ran, not that their
            # locations were disjoint, not that their union was the whole
            # population, not that they were the same head and the same checker.
            # `--aggregate` below verifies exactly that against these files.
            print(json.dumps({
                # from the worktree that did the measuring, falling back to
                # the launch tree only when no shard recorded one
                "head_sha": _CTRL["head"] or _head_sha(),
                "checker_sha256": _CTRL["checker_sha"] or _checker_sha(),
                "population_sha256": (_CTRL["population_sha"]
                                      or _population_sha(picked)),
                "population_size": cell_population(picked),
                "shard": list(shard),
                "control_before": "pass" if not _CTRL["before"] else "FAIL",
                "control_after": "pass" if not _CTRL["after"] else "FAIL",
                "probed": len(res),
                "locations": sorted([r["doc"], r["table_line"], r["row"],
                                     r["col"], r["start"], r["end"]]
                                    for r in res),
                "survived": surv}, indent=2))
            return
        print(f"shard {shard[0]}/{shard[1]}: {len(res)} numbers perturbed, "
              f"{len(surv)} survived")
        for r in surv:
            print(f"  {r['doc']}:{r['row']} col{r['col']}  {r['perturbation']}")
        return
    if "--prose" in argv:
        pc = prose_census()
        if "--probe" in argv:
            pc["sample"] = prose_probe(pc["absent"])
        pc.pop("absent")
        if "--json" in argv:
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
    _key = "covered" if "--covered" in argv else "uncovered"
    if "--probe" in argv:
        c[_key] = probe(c[_key])
    if "--json" in argv:
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
