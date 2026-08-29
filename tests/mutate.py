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
    ("a test replaces time.sleep for the whole process and never restores it",
     "tests/test_harness_invariants.py",
     'self.patch(rr.time, "sleep", lambda *_a: None)',
     "rr.time.sleep = lambda *_a: None",
     "tests.test_harness_invariants.NoTestMayLeaveTheStandardLibraryPatched"),
    ("the stub server stops exiting when the process that started it dies",
     "tests/fake_llama_server.py",
     "threading.Thread(target=_exit_when_orphaned, daemon=True).start()",
     "pass  # watchdog removed",
     "tests.test_harness_invariants.TheStubServerMustNotOutliveItsParent"),
    # --- the fourth review's findings, each broken here ---------------------
    ("the perturbation suite stops refusing to run during a measurement",
     "tests/data_mutate.py",
     'host_guard.protect("the data perturbation suite")',
     'pass  # guard removed',
     "tests.test_harness_invariants.TheVerificationSuitesMustRefuseAMeasuringHost"),
    ("benchmark detection goes back to matching anywhere in the command line",
     "bench/host_guard.py",
     "    if exe in _BENCH_EXE:",
     "    if any(n in ' '.join(argv) for n in _BENCH_EXE):",
     "tests.test_harness_invariants.TheVerificationSuitesMustRefuseAMeasuringHost"),
    ("the rerun script goes back to a variable the runner never reads",
     "bench/run_v2_crossover.sh",
     'export LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-',
     'export BENCH_SERVER="${BENCH_SERVER:-',
     "tests.test_harness_invariants.TheRerunScriptsMustBeSelfContainedAndFailClosed"),
    ("the rerun script stops failing when a session fails",
     "bench/run_v3_within.sh",
     '[ -z "$FAILED" ] || { echo "FAIL: sessions failed:$FAILED" >&2; rc=1; }',
     'true',
     "tests.test_harness_invariants.TheRerunScriptsMustBeSelfContainedAndFailClosed"),
    ("re-derivation indexes with a dict comprehension again",
     "analysis/rederive_from_logs.py",
     '    A = index_unique(regenerated, key, f"{label}: regenerated")',
     "    A = {key(r): r for r in regenerated}",
     "tests.test_harness_invariants.RederivationMustNotCollapseOrMiscount"),
    ("re-derivation goes back to counting the gap instead of naming it",
     "analysis/rederive_from_logs.py",
     "if set(missing) != set(expected_missing):",
     "if len(missing) != len(expected_missing):",
     "tests.test_harness_invariants.RederivationMustNotCollapseOrMiscount"),
    ("a completed arm-run may again have no observed target identity",
     "bench/retest_runner.py",
     "    if healthy and not seen_identity:",
     "    if False and not seen_identity:",
     "tests.test_harness_invariants.AHealthyArmRunMustSayWhatItLoaded"),
    ("the workflows lose their named shell, and with it pipefail",
     ".github/workflows/audit.yml",
     "defaults:\n  run:\n    shell: bash\n",
     "",
     "tests.test_harness_invariants.TheWorkflowsMustNameTheirShell"),
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
     '                  and row["predicted_n"] != MAX_TOKENS):',
     '                  and False):',
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
     "elif len(missing) != expected_gap:",
     "elif False:",
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

    ("a hard-cap arm stops sending the cap",
     "bench/retest_runner.py",
     "    if IGNORE_EOS or hardcap:",
     "    if IGNORE_EOS:",
     "tests.test_harness_invariants.BothModesMustFitInOneInvocation"),
    ("the suffix stops being recognised at all",
     "bench/retest_runner.py",
     "    return arm[:-len(HARDCAP_SUFFIX)] in ARMS",
     "    return False",
     "tests.test_harness_invariants.BothModesMustFitInOneInvocation"),
    ("a hard-cap arm gets its own flags rather than the base arm's",
     "bench/retest_runner.py",
     "    return arm[:-len(HARDCAP_SUFFIX)] if arm_is_hardcap(arm) else arm",
     "    return arm",
     "tests.test_harness_invariants.BothModesMustFitInOneInvocation"),
    ("the cap requirement goes back to being run-level",
     "bench/retest_runner.py",
     "            elif ((IGNORE_EOS or arm_is_hardcap(str(arm)))",
     "            elif ((IGNORE_EOS and arm_is_hardcap(str(arm)))",
     "tests.test_harness_invariants.BothModesMustFitInOneInvocation"),
    ("a real arm loses to the suffix rule",
     "bench/retest_runner.py",
     "    if not HARDCAP_SUFFIX or arm in ARMS or not arm.endswith(HARDCAP_SUFFIX):",
     "    if not HARDCAP_SUFFIX or not arm.endswith(HARDCAP_SUFFIX):",
     "tests.test_harness_invariants.BothModesMustFitInOneInvocation"),

    ("timer extraction goes back to matching fields by position",
     "analysis/extract_checkpoint_timers.py",
     'RE_AUDIT = re.compile(r"AUDIT_US ((?:\\w+=\\d+ ?)+)")',
     'RE_AUDIT = re.compile(r"AUDIT_US (update_tgt=\\d+|load_tgt=\\d+ load_dft=\\d+)")',
     "tests.test_harness_invariants.TimerExtractionMustNotDependOnFieldOrder"),

    ("a run script stops pinning the fit target run V used",
     "bench/run_v2_crossover.sh",
     "export BENCH_FIT_TARGET=3072",
     "export BENCH_FIT_TARGET_UNSET=1024",
     "tests.test_harness_invariants.ARunScriptMustSetEveryFieldItClaimsToReproduce"),
    ("the within-invocation script takes a run-level cap after all",
     "bench/run_v3_within.sh",
     'export BENCH_HARDCAP_SUFFIX="-cap"',
     'export BENCH_IGNORE_EOS=on',
     "tests.test_harness_invariants.ARunScriptMustSetEveryFieldItClaimsToReproduce"),

    ("the mode contrast stops dividing out its own baseline",
     "analysis/length_mode.py",
     '        cap_pct = 100.0 * (cap_rates[a] / cap_rates[base] - 1.0)',
     '        cap_pct = 100.0 * (cap_rates[a] / free_rates[a] - 1.0)',
     "tests.test_harness_invariants.TheLengthModeAnalysisMustReadBothDesigns"),
    ("the log form of the contrast drifts from the percentage form",
     "analysis/length_mode.py",
     """                  "log_delta": (math.log(cap_rates[a] / cap_rates[base])
                                - math.log(free_rates[a] / free_rates[base]))}""",
     """                  "log_delta": math.log(cap_rates[a] / free_rates[a])}""",
     "tests.test_harness_invariants.TheLengthModeAnalysisMustReadBothDesigns"),
    ("a session with one half missing is used anyway",
     "analysis/length_mode.py",
     "        if set(halves_) == {\"freerun\", \"hardcap\"} and all(",
     "        if True or set(halves_) == {\"freerun\", \"hardcap\"} and all(",
     "tests.test_harness_invariants.TheLengthModeAnalysisMustReadBothDesigns"),

    ("a run script starts the telemetry sampler with a path again",
     "bench/run_v2_crossover.sh",
     'bash "$TELE_SH" "$TELE_SCHEMA" "$TELE_INTERVAL" "V2" &',
     'bash "$TELE_SH" "$BENCH/gpu_telemetry_V2.csv" &',
     "tests.test_harness_invariants.ARunScriptMustSetEveryFieldItClaimsToReproduce"),

]


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
        "run_matrix.sh", "run_p0_matrix.sh", "run_verify_matrix.sh",
        "collect_env.sh", "PULL_REQUEST.md", "tools",
        # the workflows are mutated too now, and a mirror without them turns a
        # mutation into a FileNotFoundError rather than a verdict
        ".github")


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
    # One `unittest` subprocess per mutation, for minutes. Same reason as
    # tests/data_mutate.py: a burst on a measuring host costs an arm-pass.
    sys.path.insert(0, str(ROOT / "bench"))
    import host_guard
    host_guard.protect("the code mutation suite")
    host_guard.serialise("verify")

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
