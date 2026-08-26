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


def warm_telemetry(m: Path):
    p = m / "v4_audit_2026_08_25/data/gpu_telemetry_T_20260826_182639.csv"
    out = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        if i == 0:
            out.append(line); continue
        cols = line.split(",")
        if len(cols) > 3 and cols[3].strip().isdigit():
            cols[3] = str(int(cols[3]) + 5)          # temp
        out.append(",".join(cols))
    p.write_text("\n".join(out) + "\n", encoding="utf-8")


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
        if old not in t:
            raise AssertionError(f"anchor not found in {rel}: {old[:40]!r}")
        p.write_text(t.replace(old, new, 1), encoding="utf-8")
    go.touches = (rel,)
    return go


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
    ("C4b: the clock mean becomes 1947",
     edit_doc("ERRATA.md", "1800\u20131965 MHz of a 2100 MHz maximum, mean 1937",
              "1800\u20131965 MHz of a 2100 MHz maximum, mean 1947")),
]


def main() -> None:
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
    print()
    if survived:
        sys.exit(f"  {len(survived)} perturbation(s) survived: " + "; ".join(survived))
    print(f"  all {len(MUTATIONS)} perturbations detected")


if __name__ == "__main__":
    main()
