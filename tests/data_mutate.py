"""Mutation-test the claim checker against the DATA, not against the code.

`analysis/verify_claims.py` refuses an assertion that compares two literals, and
`tests/mutate.py` breaks each published fix and requires its guard to fail.
Neither shows that changing a *measurement* changes the verdict. An assertion can
name a figure, load the file it lives in, and still not depend on it.

So: mirror the tree, perturb one committed measurement, and require the checker
to exit non-zero. A perturbation that survives means nothing reads that quantity.

The first run of this found one: setting a control arm's checkpoint *count* to
three passed, because the controls were asserted through `checkpoint_total_s`
alone and zero seconds is not zero events.

No GPU, no network. Copies ~90 MB into a temporary directory.

Run: python tests/data_mutate.py
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Everything verify_claims.py reads. It grew: the historical-script check added
# four files at the repository root, and the mirror silently lacked them until
# the checker crashed on the unperturbed copy.
COPY = ("analysis", "bench", "tests", "v4_audit_2026_08_25", "results",
        "v2_3090_followup", "v3_dflash_2026_05_07", "README.md", "ERRATA.md",
        "CHANGELOG.md", "RETEST_TODO.md", "BENCHMARK_ENV.md",
        "CITATION.cff",
        # the CI-install guard reads both locks, so a mirror
        # without them turns that test into a FileNotFoundError
        "requirements-lint.lock", "requirements-plot.lock",
        # the v1 archive's own request payload carries `temperature: 0.0`, which
        # the tier registry's row is checked against, and `pr_comment.md` is the
        # third document quoting the 0.6B drafter's vocabulary
        "bench_runner.py", "pr_comment.md",
        "PULL_REQUEST.md",
        # the coverage census asserts that every markdown file in the tree is
        # either censused or excluded with a reason, so a mirror missing one
        # fails on the unperturbed copy
        "pr_comment.md",
        "run_matrix.sh", "run_p0_matrix.sh", "run_verify_matrix.sh",
        "collect_env.sh")


def mirror(into: Path) -> Path:
    into.mkdir(parents=True, exist_ok=True)
    for p in COPY:
        src = ROOT / p
        if not src.exists():
            continue
        dst = into / p
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    return into


def _rows(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj), encoding="utf-8")


def scale_row(rel: str, field: str, factor: float):
    def go(m: Path):
        p = m / rel
        d = _rows(p)
        d["rows"][0][field] = d["rows"][0][field] * factor
        _write(p, d)
    go.touches = (rel,)
    return go


def scale_timing(rel: str, field: str, factor: float):
    """Perturb the copy inside `timings` rather than the top-level one."""
    def go(m: Path):
        p = m / rel
        d = _rows(p)
        d["rows"][0]["timings"][field] = d["rows"][0]["timings"][field] * factor
        _write(p, d)
    go.touches = (rel,)
    return go


def set_row(rel: str, field: str, value):
    def go(m: Path):
        p = m / rel
        d = _rows(p)
        d["rows"][0][field] = value
        _write(p, d)
    go.touches = (rel,)
    return go


def timers(arm: str, field: str, fn):
    def go(m: Path):
        p = m / "v4_audit_2026_08_25/data/checkpoint_timers_20260826.json"
        d = _rows(p)
        for r in d:
            if r["arm"] == arm:
                r[field] = fn(r[field])
        _write(p, d)
    go.touches = ("v4_audit_2026_08_25/data/checkpoint_timers_20260826.json",)
    return go


def drop(rel: str):
    def go(m: Path):
        (m / rel).unlink()
    go.touches = (rel,)
    return go


def zero_length_shifts(m: Path):
    p = m / "analysis/length_matching.json"
    d = _rows(p)
    for r in d["runs"]:
        if r["run"].startswith("matrix_L_thinkoff"):
            for v in r["arms"].values():
                if "shift_pp" in v:
                    v["shift_pp"] = 0.0
    _write(p, d)


zero_length_shifts.touches = ("analysis/length_matching.json",)


# This had no `touches` until 2026-08-27, so the restore loop skipped it and the
# mirror kept a telemetry trace 5 C warmer for the rest of the run. Every
# perturbation after it then ran against a tree the checker was ALREADY failing
# on, so `returncode != 0` proved nothing about the perturbation itself.
_WARM_TELEMETRY_FILE = "v4_audit_2026_08_25/data/gpu_telemetry_T_20260826_182639.csv"


def warm_telemetry(m: Path):
    p = m / _WARM_TELEMETRY_FILE
    out = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        if i == 0:
            out.append(line); continue
        cols = line.split(",")
        if len(cols) > 3 and cols[3].strip().isdigit():
            cols[3] = str(int(cols[3]) + 5)          # temp
        out.append(",".join(cols))
    p.write_text("\n".join(out) + "\n", encoding="utf-8")


warm_telemetry.touches = (_WARM_TELEMETRY_FILE,)


def edit_doc(rel: str, old: str, new: str):
    """Perturb a published FIGURE rather than a measurement.

    Greping a document for a computed value proves the string exists somewhere,
    not that the row that should carry it does. Changing the headline table's
    "+26.3 %" to "+26.9 %" passed every check here, because "+26.3 %" still
    appeared in the replication table below it. These require the tables to be
    parsed rather than searched.
    """
    def go(m: Path):
        p = m / rel
        t = p.read_text(encoding="utf-8")
        n = t.count(old)
        if n != 1:
            raise AssertionError(
                f"anchor appears {n} times in {rel}, must be exactly one: "
                f"{old[:48]!r}")
        p.write_text(t.replace(old, new, 1), encoding="utf-8")
    go.touches = (rel,)
    go.anchor = (rel, old)
    return go


def edit_doc_re(rel: str, pattern: str, repl):
    """`edit_doc` for a figure whose current value this suite must not pin.

    The assertion count in `PULL_REQUEST.md` is one: hard-coding it here would
    make this file a second place that has to be updated whenever an assertion
    is added, and that is how a published figure goes stale in the first place.
    """
    def go(m: Path):
        p = m / rel
        t = p.read_text(encoding="utf-8")
        out, n = re.subn(pattern, repl, t, count=1)
        if not n:
            raise AssertionError(f"pattern not found in {rel}: {pattern!r}")
        p.write_text(out, encoding="utf-8")
    go.touches = (rel,)
    return go


def edit_json(rel: str, mutate):
    """Perturb a committed derived JSON in place in the mirror."""
    def go(m: Path):
        f = m / rel
        d = json.loads(f.read_text(encoding="utf-8"))
        mutate(d)
        f.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    go.touches = (rel,)
    return go


E = "v4_audit_2026_08_25/data/E_past_threshold"
H = "v4_audit_2026_08_25/data/H_pmin_sweep"
C = "v4_audit_2026_08_25/data/C_master_matrix_think_on"
PRE = "v4_audit_2026_08_25/PREREGISTERED_PREDICTION.md"
O2 = "v4_audit_2026_08_25/data/matrix_O2_latin_20260826_153711"
O3 = "v4_audit_2026_08_25/data/matrix_O3_latin_20260826_203251"
T3 = "v4_audit_2026_08_25/data/matrix_T3_timers_20260826_203251"

MUTATIONS = [
    ("O2 decode time on one request +5 %",
     scale_row(f"{O2}/spec-dflash-n2__rep0.json", "predicted_ms", 1.05)),
    ("O2 draft acceptance on one request halved",
     scale_row(f"{O2}/spec-dflash-n2__rep0.json", "draft_n_accepted", 0.5)),
    ("O2 draft volume on one request doubled",
     scale_row(f"{O2}/ngram-map-k4v-m8__rep0.json", "draft_n", 2)),
    ("O2 generated token count on one request off by one",
     scale_row(f"{O2}/baseline__rep0.json", "predicted_n", 299 / 300)),
    ("O3 output text changed on one request",
     set_row(f"{O3}/baseline__rep0.json", "content", "different")),
    ("run T checkpoint create count off by one",
     timers("spec-draft-n8", "creates", lambda v: v + 1)),
    ("run T checkpoint total seconds +1 %",
     timers("spec-draft-n8", "checkpoint_total_s", lambda v: v * 1.01)),
    ("a control's checkpoint count becomes non-zero",
     timers("baseline", "creates", lambda _v: 3)),
    ("a control's restore count becomes non-zero",
     timers("spec-dflash-n2", "restores", lambda _v: 2)),
    ("the length-matching shift on run L is zeroed", zero_length_shifts),
    ("run T's telemetry runs 5 C warmer", warm_telemetry),
    ("T3 loses one arm-run", drop(f"{T3}/baseline__rep2.json")),

    # published figures, not measurements
    ("headline table: the DFlash change becomes +26.9 %",
     edit_doc("README.md", "| **+26.3 %** | [+25.5 %, +27.1 %] | 0.81 |",
              "| **+26.9 %** | [+25.5 %, +27.1 %] | 0.81 |")),
    ("headline table: an acceptance cell becomes 82.3 %",
     edit_doc("README.md", "| 0.81 | 72.3 % |", "| 0.81 | 82.3 % |")),
    ("headline table: a draft/gen cell becomes 1.96",
     edit_doc("README.md", "| 1.86 | 29.5 % |", "| 1.96 | 29.5 % |")),
    ("headline table: the baseline's pooled rate becomes 116.7",
     edit_doc("README.md", "| **no speculation** | **115.7**",
              "| **no speculation** | **116.7**")),
    ("replication table: O3's DFlash figure becomes +24.4 %",
     edit_doc("README.md", "| +26.3 % | **+23.4 %** | **\u22122.9 pp** |",
              "| +26.3 % | **+24.4 %** | **\u22122.9 pp** |")),
    ("replication table: a shift becomes -1.0 pp",
     edit_doc("README.md", "| +22.7 % | +21.7 % | \u22121.0 pp |",
              "| +22.7 % | +21.7 % | \u22122.0 pp |")),
    ("footnote: run M1's figure becomes +27.7 %",
     edit_doc("README.md", "M1 **+26.7 %** 07:59", "M1 **+27.7 %** 07:59")),
    ("footnote: a run's clock time is wrong by a minute",
     edit_doc("README.md", "O3 **+23.4 %** 20:44", "O3 **+23.4 %** 20:45")),
    ("footnote: run U3's figure becomes +19.3 %",
     edit_doc("README.md", "U3 **+17.3 %** 22:18", "U3 **+19.3 %** 22:18")),
    ("a withdrawn figure reappears in the headline table",
     edit_doc("README.md", "| `ngram-cache` | 93.7 |",
              "| `ngram-cache` | 93.7 (101.3 MiB) |")),
    ("A12: the checkpoint total becomes 39.80 s",
     edit_doc("ERRATA.md", "| **speculative checkpoint, total** | **39.07** |",
              "| **speculative checkpoint, total** | **39.80** |")),
    ("A12: run T's nominal volume becomes 131.27 GiB",
     edit_doc("ERRATA.md", "| **121.27** |", "| **131.27** |")),
    ("A12 table: a component's seconds become 18.34",
     edit_doc("ERRATA.md", "| `update_tgt` \u2014 785 checkpoint creates | 17.34 |",
              "| `update_tgt` \u2014 785 checkpoint creates | 18.34 |")),
    ("A12 table: a component's share becomes 25.3 %",
     edit_doc("ERRATA.md", "| 17.34 | 24.3 % |", "| 17.34 | 25.3 % |")),
    ("A12 table: the excess becomes 74.4 s",
     edit_doc("ERRATA.md", "| **excess to account for** | **71.4** |",
              "| **excess to account for** | **74.4** |")),
    ("A13 table: a server counter becomes 39.7 %",
     edit_doc("ERRATA.md", "| `spec-draft-n8` | 29.7 % | 41.3 % |",
              "| `spec-draft-n8` | 39.7 % | 41.3 % |")),
    ("A13 table: a checkpoint count becomes 872",
     edit_doc("ERRATA.md", "| 11.6 pp | 772 |", "| 11.6 pp | 872 |")),
    ("v4 README: run L's DFlash throughput becomes 150.8",
     edit_doc("v4_audit_2026_08_25/README.md",
              "| `spec-dflash-n2` | 148.8 |", "| `spec-dflash-n2` | 150.8 |")),
    ("v4 README: run M's aggregate becomes 130.3",
     edit_doc("v4_audit_2026_08_25/README.md", "| **127.3** |", "| **130.3** |")),
    ("v4 README: an n-gram throughput becomes 110.6",
     edit_doc("v4_audit_2026_08_25/README.md",
              "| `ngram-map-k-m8` | 107.6 |", "| `ngram-map-k-m8` | 110.6 |")),
    ("A17: run V's hard-cap figure becomes +22.60 %",
     edit_doc("ERRATA.md", "| `spec-dflash-n2` | +11.35 % | **+20.60 %** |",
              "| `spec-dflash-n2` | +11.35 % | **+22.60 %** |")),
    ("A17: the sign-flipping arm stops flipping",
     edit_doc("ERRATA.md", "| `spec-dflash-n4` | **\u22121.35 %** | **+10.55 %** |",
              "| `spec-dflash-n4` | **+1.35 %** | **+10.55 %** |")),
    ("A16: a run U figure becomes +19.3 %",
     edit_doc("README.md", "U3 **+17.3 %** 22:18", "U3 **+19.3 %** 22:18")),
    # --- the pull request body, which had no code path until 2026-08-27 -----
    # It is a published document with four numeric tables in it and nothing was
    # reading them, which is the defect this audit has now closed five times.
    ("PR body: the headline DFlash rate becomes 149.2",
     edit_doc("PULL_REQUEST.md", "| **146.2** |", "| **149.2** |")),
    ("PR body: the no-speculation baseline becomes 112.7",
     edit_doc("PULL_REQUEST.md", "| **115.7** |", "| **112.7** |")),
    ("PR body: the synchronisation wait becomes 2.002 s",
     edit_doc("PULL_REQUEST.md", "| **0.002** |", "| **2.002** |")),
    # --- run W, the carryover-balanced design ------------------------------
    ("W decode time on one arm-run +5 %",
     scale_row("v4_audit_2026_08_25/data/matrix_W_s1_20260828_104222/spec-dflash-n2__rep0.json", "predicted_ms", 1.05)),
    ("W draft acceptance on one request halved",
     scale_row("v4_audit_2026_08_25/data/matrix_W_s1_20260828_104222/spec-dflash-n2-cap__rep0.json", "draft_n_accepted", 0.5)),
    ("W output text changed on one request",
     set_row("v4_audit_2026_08_25/data/matrix_W_s1_20260828_104222/baseline__rep0.json", "content", "different")),
    ("W a manifest stops claiming carryover balance",
     edit_json("v4_audit_2026_08_25/data/matrix_W_s1_20260828_104222/manifest.json",
               lambda d: d.__setitem__("schedule_first_order_carryover_balanced", False))),
    ("W a manifest claims the cyclic schedule instead",
     edit_json("v4_audit_2026_08_25/data/matrix_W_s1_20260828_104222/manifest.json", lambda d: d.__setitem__("order_mode", "latin"))),
    ("A17: W's disagreeing arm is made to agree with the crossover",
     edit_doc("ERRATA.md", "| **+8.29** [+7.97, +8.60] |",
              "| **+5.92** [+4.86, +6.99] |")),
    ("A17: the predecessor null is made to exclude zero",
     edit_doc("ERRATA.md", "| [\u22122.61 %, +0.22 %] |", "| [\u22122.61 %, \u22120.22 %] |")),
    ("A17: W's per-repeat CV for the unstable arm becomes 0.69",
     edit_doc("ERRATA.md", "| `spec-dflash-n2` | **1.69 %** |",
              "| `spec-dflash-n2` | **0.69 %** |")),
    ("v4 README: the re-derivation record count for the split dump becomes 20",
     edit_doc("v4_audit_2026_08_25/README.md",
              "| `data/checkpoint_timers_20260827_split.json` | 18 | **18** | \u2014 |",
              "| `data/checkpoint_timers_20260827_split.json` | 20 | **20** | \u2014 |")),
    ("v4 README: it goes back to claiming CI produced the table",
     edit_doc("v4_audit_2026_08_25/README.md",
              "produced by running the script, not by CI",
              "produced by CI, from the archive alone")),
    ("PR body: the re-derivation count for the split dump becomes 20",
     edit_doc("PULL_REQUEST.md",
              "| `data/checkpoint_timers_20260827_split.json` | 18 | **18** | 0 |",
              "| `data/checkpoint_timers_20260827_split.json` | 20 | **20** | 0 |")),
    ("README cost table: the restore share goes back to 30.5 %",
     edit_doc("README.md", "| 21.74 | 30.4 % |", "| 21.74 | 30.5 % |")),
    ("README cost table: the drafter's seconds become 19.27",
     edit_doc("README.md", "| drafter `generate()` | 17.27 |",
              "| drafter `generate()` | 19.27 |")),
    ("PR body cost table: the restore share goes back to 30.5 %",
     edit_doc("PULL_REQUEST.md", "| 21.74 | 30.4 % |", "| 21.74 | 30.5 % |")),
    ("PR body: the checkpoint share becomes 64.7 %",
     edit_doc("PULL_REQUEST.md",
              "| inside the checkpoint calls | **39.09** | **54.7 %** |",
              "| inside the checkpoint calls | **39.09** | **64.7 %** |")),
    ("PR body: V2's sign-flipping arm stops flipping",
     edit_doc("PULL_REQUEST.md", "| **\u22121.66 %** [\u22121.98, \u22121.35] |",
              "| **+1.66 %** [\u22121.98, \u22121.35] |")),
    ("PR body: V3's disagreeing arm is made to agree",
     edit_doc("PULL_REQUEST.md", "| **+8.65 pp** |", "| **+5.92 pp** |")),
    ("PR body: the assertion count goes stale by one",
     edit_doc_re("PULL_REQUEST.md", r"# (\d+) assertions",
                 lambda m: f"# {int(m.group(1)) + 1} assertions")),
    ("C4b: the clock mean becomes 1947",
     edit_doc("ERRATA.md", "1800\u20131965 MHz of a 2100 MHz maximum, mean 1937",
              "1800\u20131965 MHz of a 2100 MHz maximum, mean 1947")),
    # --- the pre-registration, which had no code path until 2026-08-27 -------
    # scales the TOP-LEVEL copy only, which is the point: `timings` carries the
    # same number and different analyses read different copies
    ("E past-threshold decode time on one request +5 %",
     scale_row(f"{E}/spec-draft-n96__rep0.json", "predicted_ms", 1.05)),
    ("the nested copy of a decode time is changed instead",
     scale_timing(f"{E}/spec-draft-n96__rep1.json", "predicted_ms", 1.05)),
    ("E past-threshold draft volume on one request doubled",
     scale_row(f"{E}/spec-draft-n128__rep0.json", "draft_n", 2)),
    ("E past-threshold acceptance on one request halved",
     scale_row(f"{E}/spec-draft-n64__rep0.json", "draft_n_accepted", 0.5)),
    ("the run C baseline the model is scaled against drifts 1 %",
     scale_row(f"{C}/baseline__rep2.json", "predicted_ms", 1.01)),
    ("the fitted per-round coefficient moves one hundredth",
     edit_doc(PRE, "27.56 * (rounds per generated token)",
              "27.57 * (rounds per generated token)")),
    ("a registered prediction row is nudged after the fact",
     edit_doc(PRE, "| 96 | 4.9 % | 13.86 | 94.42 | **10.6** |",
              "| 96 | 4.9 % | 13.86 | 94.42 | **10.9** |")),
    ("the outcome table is rounded back into an exact hit",
     edit_doc(PRE, "| 128 | 8.9 | 8.85 | **\u22120.6 %** |",
              "| 128 | 8.9 | 8.9 | **0.0 %** |")),
    ("a measured outcome rate is improved",
     edit_doc(PRE, "| 64 | 13.4 | 12.38 |", "| 64 | 13.4 | 12.98 |")),
    ("the residual step at the coverage point grows a knee",
     edit_doc(PRE, "coverage point: \u22120.39 percentage points",
              "coverage point: \u22124.39 percentage points")),
    ("the no-speculation step reverts to the repeat-0 value",
     edit_doc(PRE, "measured 8.11 ms no-speculation", "measured 7.87 ms no-speculation")),
    ("checkpoint traffic reverts to the withdrawn checkpoint size",
     edit_doc(PRE, "*falls* from 44.8 MiB at `n_max` 1 to 20.2 at 128",
              "*falls* from 55.4 MiB at `n_max` 1 to 24.9 at 128")),
    ("the decode series is relabelled as end to end",
     edit_doc(PRE, "Pooled decode rate: 31.1", "End-to-end throughput: 31.1")),
    ("the wall-clock baseline is replaced by the decode rate",
     edit_doc(PRE, "against a **110.8** baseline", "against a **123.4** baseline")),
    ("1639 checkpoints go back to being one request",
     edit_doc(PRE, "in one\n`n_max` 1 arm-run", "for a single 300-token request in an\n`n_max` 1 arm-run")),

    # --- A7's sweep and A10's falsification, parsed since 2026-08-27 --------
    ("run H p_min decode time on one request +5 %",
     scale_row(f"{H}/spec-draft-n8-pmin75__rep1.json", "predicted_ms", 1.05)),
    ("run H p_min draft volume on one request doubled",
     scale_row(f"{H}/spec-draft-n8-pmin90__rep0.json", "draft_n", 2)),
    ("the A7 sweep drafted total is nudged",
     edit_doc("ERRATA.md", "| 8 | 32.1 | \u221274.0 % | 27 735 | 29.7 % |",
              "| 8 | 32.1 | \u221274.0 % | 27 335 | 29.7 % |")),
    ("the A7 coverage table gains a point of coverage",
     edit_doc("ERRATA.md", "| **96** | **95.3 %** | 3.1 % | 17.80 | 10.0 |",
              "| **96** | **96.3 %** | 3.1 % | 17.80 | 10.0 |")),
    ("an A10 law residual is softened",
     edit_doc("ERRATA.md", "| `n_max` 8, `p_min` 0.50 | 0.94 | 58.8 % | 39.6 | \u221268.0 % | \u221217.9 % |",
              "| `n_max` 8, `p_min` 0.50 | 0.94 | 58.8 % | 39.6 | \u221268.0 % | \u22127.9 % |")),
    ("the A10 family separation is exaggerated",
     edit_doc("ERRATA.md", "| rounds + volume | \u22125.2 % | +1.6 % | **6.9 pp** |",
              "| rounds + volume | \u22125.2 % | +1.6 % | **2.9 pp** |")),
    ("the two-configuration cost gap reverts to the rep-0 value",
     edit_doc("ERRATA.md", "| `n_max` 8, `p_min` 0.90 | 0.46 | **0.19** | **23.51** |",
              "| `n_max` 8, `p_min` 0.90 | 0.46 | **0.19** | **23.62** |")),
    ("the README and A10 p_min tables disagree",
     edit_doc("README.md", "| `n_max` 8, `p_min` 0.90 | **88.2 %** | 42.5 | \u221265.6 % |",
              "| `n_max` 8, `p_min` 0.90 | **88.2 %** | 43.5 | \u221265.6 % |")),

    ("an interval file stops recording its schedule balance",
     edit_json(f"{O2}/paired_blocks.json",
               lambda d: d.pop("schedule_position_balanced", None))),
    ("an unbalanced schedule is recorded as balanced",
     edit_json("v4_audit_2026_08_25/data/matrix_T_timers_20260826_182639/"
               "paired_blocks.json",
               lambda d: d.__setitem__("schedule_position_balanced", True))),
    ("an interval file stops saying what it covers",
     edit_json(f"{O2}/paired_blocks.json",
               lambda d: d.__setitem__("interval_scope", "everything"))),
    ("an interval file records the wrong t critical value",
     edit_json(f"{O2}/paired_blocks.json",
               lambda d: d.__setitem__("t_critical_975", 1.96))),

]


def main() -> None:
    # This spawns one checker per perturbation for several minutes. On a host
    # that is measuring, that burst invalidates the arm-pass it lands on - it
    # did, on 2026-08-27, and the measurement had to be re-run. Refuse rather
    # than be polite about it. See bench/host_guard.py for what actually caused
    # the six cores, which was not this suite being parallel; it is sequential.
    sys.path.insert(0, str(ROOT / "bench"))
    import host_guard
    host_guard.protect("the data perturbation suite")
    host_guard.serialise("verify")

    # Every perturbation must say which files it touched, because the restore
    # loop only puts back what is declared. One of them did not, and the mirror
    # kept a 5 C warmer telemetry trace for the rest of the run - so every
    # perturbation after it ran against a tree the checker was already failing
    # on and was reported "caught" without its own guard ever firing.
    undeclared = [n for n, fn in MUTATIONS if not getattr(fn, "touches", None)]
    if undeclared:
        sys.exit("  perturbation(s) with no restore declaration, which would "
                 "leak into every later one: " + "; ".join(undeclared))
    print(f"  {'perturbation':52s} verdict")
    survived = []
    with tempfile.TemporaryDirectory() as tmp:
        pristine = mirror(Path(tmp) / "pristine")
        work = Path(tmp) / "work"
        base = subprocess.run([sys.executable, "analysis/verify_claims.py"],
                              cwd=mirror(work), capture_output=True, text=True,
                              timeout=900)
        if base.returncode != 0:
            sys.exit("the checker fails on an unperturbed mirror; fix that first\n"
                     + base.stdout[-2000:])
        for name, fn in MUTATIONS:
            # restore only what the last mutation touched. Re-copying the whole
            # mirror each time cost about six minutes of the seven this took.
            touched = getattr(fn, "touches", None)
            if touched:
                for rel in touched:
                    dst = work / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(pristine / rel, dst)
            else:
                shutil.rmtree(work)
                shutil.copytree(pristine, work)
            try:
                fn(work)
            except Exception as e:  # noqa: BLE001
                print(f"  {name:52s} COULD NOT APPLY: {e}")
                survived.append(f"{name} (could not apply)")
                continue
            r = subprocess.run([sys.executable, "analysis/verify_claims.py"],
                               cwd=work, capture_output=True, text=True, timeout=900)
            caught = r.returncode != 0
            print(f"  {name:52s} {'caught' if caught else '*** SURVIVED ***'}")
            if not caught:
                survived.append(name)
            for rel in (getattr(fn, "touches", None) or ()):
                dst = work / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(pristine / rel, dst)
        # The mirror must be pristine again. If it is not, some perturbation
        # leaked and every result after it is worthless - which is exactly what
        # happened while one of them declared no `touches`.
        r = subprocess.run([sys.executable, "analysis/verify_claims.py"],
                           cwd=work, capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            sys.exit("  the mirror does not pass after the last restore, so a "
                     "perturbation leaked and the results above cannot be "
                     "trusted:\n" + r.stdout[-1500:])
        print("  mirror verified clean after the last restore")

    print()
    if survived:
        sys.exit(f"  {len(survived)} perturbation(s) survived: " + "; ".join(survived))
    print(f"  all {len(MUTATIONS)} perturbations detected")


if __name__ == "__main__":
    main()
