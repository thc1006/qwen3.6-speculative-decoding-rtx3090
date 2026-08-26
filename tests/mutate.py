"""Prove the test suite has teeth.

A green suite on correct code says nothing about the suite. This breaks each fix
in turn — restoring the exact defect that was published — and requires the test
guarding it to fail. If a mutation survives, the guard is decorative and this
exits non-zero.

One mutation already escaped when this was first run: reverting the completeness
check was masked by an independent tag-set check, so the count check itself was
untested. The isolating fixture in `test_uniform_truncation_fails_on_the_row_count_alone`
exists because of that.

Run: python tests/mutate.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (description, file, correct fragment, defect to restore, test that must fail)
MUTATIONS = [
    ("extractor double-counts the draft component",
     "analysis/extract_spec_accounting.py",
     "sum(float(a) for a, _ in created) / 1024, 2)",
     "sum(float(a) + (float(b) if b else 0.0) for a, b in created) / 1024, 2)",
     "tests.test_harness_invariants.CheckpointSizeMustNotDoubleCount"),
    ("wait_health accepts a 200 before checking liveness",
     "bench/retest_runner.py",
     """        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited with code {proc.returncode} after "
                f"{time.perf_counter() - t0:.1f}s without becoming healthy")
        try:
            with urllib.request.urlopen(url, timeout=3) as r:""",
     """        try:
            with urllib.request.urlopen(url, timeout=3) as r:""",
     "tests.test_harness_invariants.PortMustBeFreeBeforeSpawn"),
    ("BENCH_THINK maps unrecognised values to on",
     "bench/retest_runner.py",
     '''    sys.exit(f"BENCH_THINK={_THINK_RAW!r} is not recognised; use one of "
             f"{sorted(_THINK_ON)} or {sorted(_THINK_OFF)}")''',
     '    THINK = "on"',
     "tests.test_harness_invariants.ConfigMustFailClosed"),
    ("mirrored ordering stops rejecting odd repeats",
     "bench/retest_runner.py",
     '    if ORDER_MODE == "mirrored" and REPEATS % 2 == 1:',
     "    if False:",
     "tests.test_harness_invariants.OrderingMustBeBalanced"),
    ("completeness falls back to the largest row count seen",
     "analysis/matrix_report.py",
     '    n_prompts = man.get("n_prompts")',
     "    n_prompts = None",
     "tests.test_harness_invariants.StrictAggregationMustRefuseBadRuns"),
    ("staging drops its self-destruction guard",
     "bench/stage_mtp_source.py",
     "    if stage_r == src_r or src_r in stage_r.parents or stage_r in src_r.parents:",
     "    if False:",
     "tests.test_harness_invariants.StagingMustNotDestroyItsSource"),
    ("the runner stops refusing a non-empty output directory",
     "bench/retest_runner.py",
     "    if stale:",
     "    if False:",
     "tests.test_harness_invariants.OutputDirectoryMustBeFresh"),
]


def main() -> None:
    print(f"  {'mutation':52s} guarding test")
    escaped = []
    for name, path, correct, defect, test in MUTATIONS:
        p = ROOT / path
        backup = p.read_text(encoding="utf-8")
        if correct not in backup:
            print(f"  {name:52s} ANCHOR MOVED - mutation not applied")
            escaped.append(f"{name} (anchor moved)")
            continue
        try:
            p.write_text(backup.replace(correct, defect, 1), encoding="utf-8")
            r = subprocess.run([sys.executable, "-m", "unittest", test],
                               cwd=ROOT, capture_output=True, text=True, timeout=600)
            caught = r.returncode != 0
            print(f"  {name:52s} {'caught' if caught else '*** SURVIVED ***'}")
            if not caught:
                escaped.append(name)
        finally:
            p.write_text(backup, encoding="utf-8")

    print()
    if escaped:
        sys.exit(f"  {len(escaped)} mutation(s) survived: " + "; ".join(escaped))
    print(f"  all {len(MUTATIONS)} mutations detected")


if __name__ == "__main__":
    main()
