"""Prove the test suite has teeth.

Runs in a MIRROR of the tree, never in place. The first version edited the real
source files and restored them in a `finally`, which does not run when the
process is killed - `timeout`, a `pkill`, a cancelled CI job - and which also
clobbers any edit made to the same file while it holds its backup. Both
happened: `bench/retest_runner.py` was committed with
`body.pop("ignore_eos", None)` where it should read `body["ignore_eos"] = True`,
and `analysis/extract_checkpoint_timers.py` silently lost its `with open(...)`.
`tests/test_harness_invariants.EveryPublishedFixIsStillHere` exists because of
the first and caught it.

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

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIDECAR = ROOT / ".mutate-in-progress"


def _restore_from_sidecar() -> bool:
    """Put back a file a previous run was killed in the middle of mutating.

    `finally` does not run when the process is killed, and this script edits
    source files in place. A run interrupted by SIGKILL - `timeout`, a `pkill`,
    a CI cancellation - leaves the tree MUTATED, and the mutation is then
    committed by whoever commits next. That happened: `body["ignore_eos"] = True`
    was committed as `body.pop("ignore_eos", None)`, and only the
    fix-presence test noticed.

    So the original is written to a sidecar before each mutation and removed
    after. If the sidecar exists at startup, a previous run died: restore and
    say so.
    """
    if not SIDECAR.exists():
        return False
    rel, _, body = SIDECAR.read_text(encoding="utf-8").partition("\n")
    (ROOT / rel).write_text(body, encoding="utf-8")
    SIDECAR.unlink()
    print(f"  recovered {rel} from an interrupted run")
    return True

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
    ('latin stops validating that its schedule is balanced',
     'bench/retest_runner.py',
     '        if not is_position_balanced(pos):',
     '        if False:',
     'tests.test_harness_invariants.LatinSquareMustBeBalancedOrNotCalledLatin'),
    ('timer extraction strips only rep0 from the arm name',
     'analysis/extract_checkpoint_timers.py',
     '"arm": re.sub(r"__rep\\d+\\.log$", "", os.path.basename(path)).replace(".log", ""),',
     '"arm": os.path.basename(path).replace("__rep0.log", "").replace(".log", ""),',
     'tests.test_harness_invariants.TimerExtractionMustBeRepeatIndependent'),
    ('integrity check falls back to counting arm-runs',
     'analysis/check_data_integrity.py',
     '        for a, r in sorted(expected_cells - set(cells)):',
     '        for a, r in sorted(()):',
     'tests.test_harness_invariants.RunMustCoverTheExactCell'),
    ("a file's name is no longer checked against its contents",
     'analysis/check_data_integrity.py',
     '            if r["arm"] != f_arm or str(r["repeat"]) != f_rep:',
     '            if False:',
     'tests.test_harness_invariants.RunMustCoverTheExactCell'),
    ('the completion marker is trusted rather than checked',
     'analysis/check_data_integrity.py',
     '            if set(rc.get("arms") or []) != declared:',
     '            if False:',
     'tests.test_harness_invariants.RunMustCoverTheExactCell'),
    ('RUN_COMPLETE is written whether or not the run validated',
     'bench/retest_runner.py',
     '    problems = validate_run(OUT, arms, REPEATS, results)',
     '    problems = []',
     'tests.test_harness_invariants.RunnerMustNotAttestAFailedRun'),
    ('identity assertions stop being applied',
     'bench/retest_runner.py',
     '        problems += check_identity(arm, rep, res.get("server_identity") or {},',
     '        problems += [] or ([] and check_identity(arm, rep, res.get("server_identity") or {},',
     'tests.test_harness_invariants.ProvenanceMustIdentifyWhatRan'),
    ('the prompt set is identified by its name rather than its contents',
     'bench/retest_runner.py',
     '    canon = json.dumps(PROMPTS, sort_keys=True, ensure_ascii=False,',
     '    canon = json.dumps(PROMPT_SET_NAME, sort_keys=True, ensure_ascii=False,',
     'tests.test_harness_invariants.ProvenanceMustIdentifyWhatRan'),
    ('an unsettled teardown is only printed, not recorded',
     'bench/retest_runner.py',
     '        if td.get("readable") and not td.get("settled"):',
     '        if False:',
     'tests.test_harness_invariants.TeardownMustNotContaminateTheNextArm'),
    ('ignore_eos is recorded but never sent',
     'bench/retest_runner.py',
     '        body["ignore_eos"] = True',
     '        body.pop("ignore_eos", None)',
     'tests.test_harness_invariants.ArmsMustGenerateTheSameAmountOfWork'),
    ('a short generation under a hard cap stops being a failure',
     'bench/retest_runner.py',
     '            elif IGNORE_EOS and row["predicted_n"] != MAX_TOKENS:',
     '            elif False:',
     'tests.test_harness_invariants.ArmsMustGenerateTheSameAmountOfWork'),
    ('balance goes back to requiring exactly one visit per position',
     'bench/retest_runner.py',
     '    want = sorted(list(range(1, n + 1)) * int(per))',
     '    want = list(range(1, n + 1))',
     'tests.test_harness_invariants.LatinSquareMustBeBalancedOrNotCalledLatin'),
    ('a run with no arm-runs stops being refused',
     'bench/retest_runner.py',
     '    if REPEATS < 1:',
     '    if False:',
     'tests.test_harness_invariants.AnEmptyRunIsNotACompleteRun'),
    ('a repeated arm name stops being refused',
     'bench/retest_runner.py',
     '    if dupes:',
     '    if False:',
     'tests.test_harness_invariants.AnEmptyRunIsNotACompleteRun'),
    ('a request ending as the next begins counts as an overlap',
     'bench/retest_runner.py',
     '    events.sort(key=lambda e: (e[0], e[1]))',
     '    events.sort(key=lambda e: (e[0], -e[1]))',
     'tests.test_harness_invariants.InFlightCountMustBeASweepLine'),
    ('the teardown ceiling goes back to an absolute threshold',
     'bench/retest_runner.py',
     '    ceiling = headroom_mib if baseline_mib is None else baseline_mib + headroom_mib',
     '    ceiling = headroom_mib',
     'tests.test_harness_invariants.TeardownMustNotContaminateTheNextArm'),
    ("the runner stops refusing a non-empty output directory",
     "bench/retest_runner.py",
     "    if stale:",
     "    if False:",
     "tests.test_harness_invariants.OutputDirectoryMustBeFresh"),
]


COPY = ("analysis", "bench", "tests", "v4_audit_2026_08_25", "results",
        "v2_3090_followup", "v3_dflash_2026_05_07", "README.md", "ERRATA.md",
        "CHANGELOG.md", "RETEST_TODO.md", "BENCHMARK_ENV.md",
        "run_matrix.sh", "run_p0_matrix.sh", "run_verify_matrix.sh",
        "collect_env.sh")


def mirror(into: Path) -> Path:
    into.mkdir(parents=True, exist_ok=True)
    for rel in COPY:
        src = ROOT / rel
        if not src.exists():
            continue
        dst = into / rel
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    return into


def main() -> None:
    _restore_from_sidecar()          # clean up after any older in-place run
    print(f"  {'mutation':52s} guarding test")
    escaped = []
    with tempfile.TemporaryDirectory() as tmp:
        work = mirror(Path(tmp) / "work")
        for name, path, correct, defect, test in MUTATIONS:
            p = work / path
            original = p.read_text(encoding="utf-8")
            if correct not in original:
                print(f"  {name:52s} ANCHOR MOVED - mutation not applied")
                escaped.append(f"{name} (anchor moved)")
                continue
            try:
                p.write_text(original.replace(correct, defect, 1), encoding="utf-8")
                r = subprocess.run([sys.executable, "-m", "unittest", test],
                                   cwd=work, capture_output=True, text=True,
                                   timeout=600)
                caught = r.returncode != 0
                print(f"  {name:52s} {'caught' if caught else '*** SURVIVED ***'}")
                if not caught:
                    escaped.append(name)
            finally:
                p.write_text(original, encoding="utf-8")

    print()
    if escaped:
        sys.exit(f"  {len(escaped)} mutation(s) survived: " + "; ".join(escaped))
    print(f"  all {len(MUTATIONS)} mutations detected")


if __name__ == "__main__":
    main()
