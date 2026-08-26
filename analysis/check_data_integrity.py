"""Structural checks on every committed run directory.

Separate from `verify_claims.py`, which asks whether the documents match the
data. This asks whether the data is well-formed at all: does every file parse,
does every arm-run carry the prompts its manifest says it should, did anything
crash, and is anything left over from a different run.

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
    if man.get("repeats") and any(v > man["repeats"] for v in seen_arms.values()):
        bad.append(f"{d.name}: more repeats on disk than the manifest declares")

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
