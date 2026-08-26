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
# There is no sidecar any more. Mutations run in a mirror under a temporary
# directory, so an interrupted run leaves the real checkout untouched and there
# is nothing to recover. The recovery path that used to live here read a
# relative path out of `.mutate-in-progress` and wrote it under ROOT without
# resolving it first, which a stale or crafted `../..` entry could have used to
# write outside the repository - a fail-open recovery for a failure mode the
# mirror removed.

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
    ("a treatment boolean maps unrecognised values to the default",
     "bench/retest_runner.py",
     '''    sys.exit(f"{name}={raw!r} is not a recognised boolean; use one of "
             f"{sorted(_TRUE)} or {sorted(_FALSE)}")''',
     "    return default",
     "tests.test_harness_invariants.ConfigMustFailClosed"),
    ("a treatment choice maps unrecognised values to the default",
     "bench/retest_runner.py",
     '        sys.exit(f"{name}={v!r} is not one of {sorted(allowed)}")',
     "        return default",
     "tests.test_harness_invariants.ConfigMustFailClosed"),
    ("an out-of-range integer is clamped instead of refused",
     "bench/retest_runner.py",
     '''        sys.exit(f"{name}={v} is out of range "
                 f"[{low}, {\'inf\' if high is None else high}]")''',
     "        v = low",
     "tests.test_harness_invariants.ConfigMustFailClosed"),
    ("a prefix is accepted as the expected library digest",
     "bench/retest_runner.py",
     'if EXPECT_LIB and not re.fullmatch(r"[0-9a-f]{64}", EXPECT_LIB):',
     "if False:",
     "tests.test_harness_invariants.ConfigMustFailClosed"),
    ("only the server impl library is compared between arms",
     "bench/retest_runner.py",
     "    elif libs != _LIB_BASELINE:",
     "    elif False:",
     "tests.test_harness_invariants.TheWholeLibraryMapIsPinned"),
    ("a card that stops answering mid-run counts as settled",
     "bench/retest_runner.py",
     "            unverified = baseline_mib is not None",
     "            unverified = False",
     "tests.test_harness_invariants.TeardownMustNotContaminateTheNextArm"),
    ("a host with no GPU at all is treated as an unverified teardown",
     "bench/retest_runner.py",
     "            unverified = baseline_mib is not None",
     "            unverified = True",
     "tests.test_harness_invariants.TeardownMustNotContaminateTheNextArm"),
    ("one low reading settles the teardown",
     "bench/retest_runner.py",
     "            if ok >= consecutive:",
     "            if True:",
     "tests.test_harness_invariants.TeardownMustNotContaminateTheNextArm"),
    ("the analysis goes back to its own definition of balanced",
     "analysis/paired_blocks.py",
     "    want = sorted(list(range(1, n + 1)) * int(per))",
     "    want = list(range(1, n + 1))",
     "tests.test_harness_invariants.PairedBlocksAgreesWithTheRunner"),
    ("the t critical value falls back to the normal one",
     "analysis/paired_blocks.py",
     "        tail = _betainc(df / 2.0, 0.5, df / (df + mid * mid))",
     "        tail = 0.05 if mid >= 1.959964 else 1.0",
     "tests.test_harness_invariants.PairedBlocksAgreesWithTheRunner"),
    ("only the --opt=value spelling is parsed",
     "analysis/paired_blocks.py",
     "            elif i + 1 < len(argv) and not argv[i + 1].startswith(\"--\"):",
     "            elif False:",
     "tests.test_harness_invariants.PairedBlocksAgreesWithTheRunner"),
    ("staging accepts more than one safetensors index",
     "bench/stage_mtp_source.py",
     "    if len(idx_files) > 1:",
     "    if False:",
     "tests.test_harness_invariants.StagingMustNotDestroyItsSource"),
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
     '        if not td.get("settled"):',
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
    ('--strict stops checking which arm-runs are present',
     'analysis/matrix_report.py',
     '            for a, r in sorted(want - set(have)):',
     '            for a, r in sorted(()):',
     'tests.test_harness_invariants.StrictAggregationMustRefuseBadRuns'),
    ("the runner stops refusing a non-empty output directory",
     "bench/retest_runner.py",
     "    if stale:",
     "    if False:",
     "tests.test_harness_invariants.OutputDirectoryMustBeFresh"),
    ("the re-derivation tolerates records that changed",
     "analysis/rederive_from_logs.py",
     "    if differing:",
     "    if False:",
     "tests.test_harness_invariants.RederivationMustNotForgive"),
    ("the re-derivation tolerates records that vanished",
     "analysis/rederive_from_logs.py",
     "    if len(missing) != expected_gap:",
     "    if False:",
     "tests.test_harness_invariants.RederivationMustNotForgive"),

    ("the model the server loaded stops being compared",
     "bench/retest_runner.py",
     "        if got and TARGET and os.path.realpath(got) != os.path.realpath(TARGET):",
     "        if False:",
     "tests.test_harness_invariants.WhatAnsweredMustHaveLoadedTheRightModel"),
    ("the model path pattern goes back to one that never matched",
     "bench/retest_runner.py",
     'r"llama_model_loader: loaded meta data with .*? tensors "',
     'r"llama_model_loader: loaded meta data "',
     "tests.test_harness_invariants.WhatAnsweredMustHaveLoadedTheRightModel"),

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
