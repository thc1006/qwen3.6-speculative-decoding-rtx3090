"""Structural checks on every committed run directory.

Separate from `verify_claims.py`, which asks whether the documents match the
data. This asks whether the data is well-formed at all: does every file parse,
does every arm-run carry the prompts its manifest says it should, did anything
crash, is anything left over from a different run, and is every scheduled
(arm, repeat) cell present exactly once.

That last one is why counts are not enough. An earlier version compared only
per-arm occurrence counts against `repeats`, so a directory holding rep0 twice
and no rep3 had the right number of files for the right arm and passed. The
check below builds the exact Cartesian product `arms x range(repeats)` and
requires a bijection with what is on disk, and separately requires each file's
name to agree with the `arm`/`repeat` inside it - a mismatch makes every
per-arm glob downstream silently wrong while every count stays correct.

Two tiers, because the fields the runner writes changed during the audit:

  attested  the directory carries RUN_COMPLETE.json, written by the runner after
            the last arm-run. Completeness is checked against the manifest's
            `n_prompts` and exact `prompt_tags`.
  legacy    written before those fields existed. Everything checkable is still
            checked - parseable JSON, no crash flag, a manifest, uniform row
            counts across arms - and the directory is reported as legacy rather
            than silently treated as attested.

Exits non-zero on any structural failure. Legacy status is not a failure; a
legacy directory that is internally inconsistent is.

Run: python analysis/check_data_integrity.py [<data-root>]
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ROW_FIELDS = ("tag", "predicted_n", "predicted_ms", "draft_n", "draft_n_accepted")
REQUIRED_MANIFEST = ("arms", "repeats", "max_tokens", "server_sha256", "target_sha256")


def check_dir(d: Path) -> tuple[list[str], str]:
    bad: list[str] = []
    man_path = d / "manifest.json"
    if not man_path.exists():
        return [f"{d.name}: no manifest.json"], "broken"
    try:
        man = json.loads(man_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return [f"{d.name}: manifest does not parse: {e}"], "broken"
    for k in REQUIRED_MANIFEST:
        if k not in man:
            bad.append(f"{d.name}: manifest missing {k!r}")

    # A crash is a structural failure unless the directory says, in writing,
    # that the crash is what it was recording. Run A exists to capture an abort.
    expected_crashes: set[str] = set()
    exp_path = d / "EXPECTED.json"
    if exp_path.exists():
        try:
            exp = json.loads(exp_path.read_text(encoding="utf-8"))
            expected_crashes = set(exp.get("crashes") or [])
            if not exp.get("why"):
                bad.append(f"{d.name}: EXPECTED.json has no `why`")
        except Exception as e:  # noqa: BLE001
            bad.append(f"{d.name}: EXPECTED.json does not parse: {e}")

    attested = (d / "RUN_COMPLETE.json").exists()
    n_expected = man.get("n_prompts")
    tags_expected = set(man.get("prompt_tags") or [])

    files = sorted(glob.glob(str(d / "*__rep*.json")))
    if not files:
        return [f"{d.name}: no arm-run files"], "broken"

    row_counts = Counter()
    seen_arms = Counter()
    cells: Counter = Counter()
    seen_expected: set[str] = set()
    for f in files:
        name = Path(f).name
        try:
            r = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            bad.append(f"{name}: does not parse: {e}"); continue
        for k in ("arm", "repeat", "rows"):
            if k not in r:
                bad.append(f"{name}: missing {k!r}"); break
        else:
            # the name is what every downstream glob selects on; the contents are
            # what every downstream analysis reads. They must be the same run.
            stem = name[:-len(".json")]
            f_arm, _, f_rep = stem.rpartition("__rep")
            if r["arm"] != f_arm or str(r["repeat"]) != f_rep:
                bad.append(f"{name}: filename says arm={f_arm!r} rep={f_rep!r}, "
                           f"contents say arm={r['arm']!r} rep={r['repeat']!r}")
            cells[(r["arm"], r["repeat"])] += 1
            if r.get("crashed"):
                if name in expected_crashes:
                    seen_expected.add(name)
                else:
                    bad.append(f"{name}: crashed at {r['crashed'].get('tag')}")
            tags = [x.get("tag") for x in r["rows"]]
            if len(set(tags)) != len(tags):
                bad.append(f"{name}: duplicate prompt tags")
            for i, row in enumerate(r["rows"]):
                missing = [k for k in REQUIRED_ROW_FIELDS if k not in row]
                if missing:
                    bad.append(f"{name}: row {i} missing {missing}"); break
            # an allowlisted crash legitimately stops early, so it cannot be
            # held against the uniform-row-count check
            if name not in expected_crashes:
                row_counts[len(tags)] += 1
            seen_arms[r["arm"]] += 1
            if name not in expected_crashes:
                if n_expected is not None and len(tags) != n_expected:
                    bad.append(f"{name}: {len(tags)} rows, manifest says {n_expected}")
                if tags_expected and set(tags) != tags_expected:
                    bad.append(f"{name}: tag set differs from the manifest")

    # uniform truncation is invisible to a per-file count check when the manifest
    # predates n_prompts, so compare the arms against each other too
    if n_expected is None and len(row_counts) > 1:
        bad.append(f"{d.name}: arm-runs disagree on row count {dict(row_counts)}")

    declared = set(man.get("arms") or {})
    if declared and set(seen_arms) - declared:
        bad.append(f"{d.name}: results for arms not in the manifest: "
                   f"{sorted(set(seen_arms) - declared)}")

    # the exact (arm, repeat) set, not a count
    reps = man.get("repeats")
    if declared and isinstance(reps, int) and reps > 0:
        expected_cells = {(a, r) for a in declared for r in range(reps)}
        dupes = sorted(k for k, v in cells.items() if v > 1)
        for k in dupes:
            bad.append(f"{d.name}: cell {k[0]} rep{k[1]} appears {cells[k]} times")
        for a, r in sorted(expected_cells - set(cells)):
            bad.append(f"{d.name}: cell {a} rep{r} is missing")
        for a, r in sorted(set(cells) - expected_cells):
            bad.append(f"{d.name}: cell {a} rep{r} was never scheduled")

    # RUN_COMPLETE.json is an attestation; check it against the directory it
    # attests to rather than treating its presence as proof.
    rc_path = d / "RUN_COMPLETE.json"
    if rc_path.exists():
        try:
            rc = json.loads(rc_path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            bad.append(f"{d.name}: RUN_COMPLETE.json does not parse: {e}")
        else:
            if set(rc.get("arms") or []) != declared:
                bad.append(f"{d.name}: RUN_COMPLETE arms differ from the manifest")
            if rc.get("repeats") != reps:
                bad.append(f"{d.name}: RUN_COMPLETE repeats={rc.get('repeats')} "
                           f"but manifest says {reps}")
            exp_runs = rc.get("expected_arm_runs")
            if exp_runs is not None and exp_runs != len(files):
                bad.append(f"{d.name}: RUN_COMPLETE claims {exp_runs} arm-runs, "
                           f"{len(files)} on disk")
            if rc.get("n_prompts") is not None and n_expected is not None \
                    and rc["n_prompts"] != n_expected:
                bad.append(f"{d.name}: RUN_COMPLETE n_prompts disagrees with the manifest")
    if (d / "RUN_FAILED.json").exists():
        bad.append(f"{d.name}: RUN_FAILED.json present - the runner rejected this run")

    # the allowlist must not outlive the crashes it excuses
    stale = expected_crashes - seen_expected
    if stale:
        bad.append(f"{d.name}: EXPECTED.json lists crashes that did not happen: "
                   f"{sorted(stale)}")

    return bad, ("attested" if attested else "legacy")


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "v4_audit_2026_08_25" / "data"
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not dirs:
        sys.exit(f"no run directories under {root}")
    all_bad: list[str] = []
    tiers = Counter()
    for d in dirs:
        bad, tier = check_dir(d)
        tiers[tier] += 1
        mark = "ok  " if not bad else "FAIL"
        print(f"  {mark} {tier:8s} {d.name}")
        for b in bad:
            print(f"         {b}")
        all_bad += bad
    print(f"\n  {len(dirs)} directories: " +
          ", ".join(f"{v} {k}" for k, v in sorted(tiers.items())))
    if all_bad:
        sys.exit(f"\n{len(all_bad)} structural problem(s)")
    print("  no structural problems")


if __name__ == "__main__":
    main()
