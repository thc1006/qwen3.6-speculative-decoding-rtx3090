"""Regression tests for defects this repository actually shipped.

Every case below is a bug that was published and then found — six of them by an
external review of the pull request, the rest by the audit itself. A test that
only demonstrates the fixed code working proves nothing, so each one asserts
that the *broken* input is rejected: an occupied port, an unbalanced repeat
count, a uniformly truncated matrix, an unrecognised flag value, a staging
directory pointed at its own source.

No GPU, no network, no third-party packages. `python -m unittest discover tests`.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "bench" / "retest_runner.py"
sys.path.insert(0, str(ROOT / "analysis"))


def run_runner(env_extra: dict, out: Path) -> subprocess.CompletedProcess:
    """Invoke the runner far enough to hit its argument validation."""
    env = dict(os.environ)
    env.update({
        "LLAMA_SERVER_BIN": "/bin/true",
        "MODEL_TARGET": "/dev/null",
        "BENCH_ARMS": "baseline",
        "BENCH_OUT": str(out),
        "BENCH_REPEATS": "2",
    })
    env.update(env_extra)
    return subprocess.run([sys.executable, str(RUNNER)], env=env,
                          capture_output=True, text=True, timeout=120)


class ConfigMustFailClosed(unittest.TestCase):
    """An unrecognised value must stop the run, not pick a default."""

    def test_unrecognised_think_value_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            r = run_runner({"BENCH_THINK": "of"}, Path(d) / "out")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not recognised", r.stdout + r.stderr)

    def test_recognised_think_values_are_accepted(self):
        for v in ("off", "on", "0", "true"):
            with tempfile.TemporaryDirectory() as d:
                r = run_runner({"BENCH_THINK": v}, Path(d) / "out")
            self.assertNotIn("not recognised", r.stdout + r.stderr,
                             f"{v!r} should be accepted")

    def test_unknown_order_mode_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            r = run_runner({"BENCH_ORDER": "abba"}, Path(d) / "out")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("BENCH_ORDER", r.stdout + r.stderr)


class OrderingMustBeBalanced(unittest.TestCase):
    """Reversing the arm list on odd repeats leaves position confounded."""

    def test_mirrored_with_odd_repeats_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            r = run_runner({"BENCH_ORDER": "mirrored", "BENCH_REPEATS": "3"},
                           Path(d) / "out")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not balanced", r.stdout + r.stderr)

    def test_mirrored_with_even_repeats_is_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            r = run_runner({"BENCH_ORDER": "mirrored", "BENCH_REPEATS": "2"},
                           Path(d) / "out")
        self.assertNotIn("not balanced", r.stdout + r.stderr)

    def test_latin_rotation_spreads_each_arm_across_positions(self):
        arms = list("ABCDEFGHI")
        repeats = 3
        step = max(1, len(arms) // max(1, repeats))
        seen = []
        for rep in range(repeats):
            k = (rep * step) % len(arms)
            seen.append((arms[k:] + arms[:k]).index("A") + 1)
        self.assertEqual(len(set(seen)), repeats, f"positions repeated: {seen}")
        # and it must beat what mirrored gives at the same repeat count
        mirrored = [(arms if r % 2 == 0 else arms[::-1]).index("A") + 1
                    for r in range(repeats)]
        self.assertLess(len(set(mirrored)), len(set(seen)))


class PortMustBeFreeBeforeSpawn(unittest.TestCase):
    """A stale server answering /health would be measured in place of this one."""

    def test_occupied_port_is_detected(self):
        spec = _load_runner()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sk:
            sk.bind(("127.0.0.1", 0))
            sk.listen(1)
            port = sk.getsockname()[1]
            self.assertFalse(spec.port_is_free(port))
        self.assertTrue(spec.port_is_free(port))

    def test_liveness_is_checked_before_readiness(self):
        src = RUNNER.read_text(encoding="utf-8")
        body = src[src.index("def wait_health"):]
        body = body[:body.index("\ndef ", 1)]
        self.assertLess(body.index("proc.poll()"),
                        body.index("return time.perf_counter"),
                        "a 200 from a stale server would be accepted first")


class CheckpointSizeMustNotDoubleCount(unittest.TestCase):
    """`common_prompt_checkpoint::size()` already includes the draft component."""

    LOG = (
        "0.00.001.000 I build : 10622 (deadbeef) with GNU\n"
        "0.00.002.000 D slot operator(): created speculative checkpoint "
        "(pos_min = 1, pos_max = 1, n_tokens = 2, size = 1000.000 MiB, draft = 400.000 MiB)\n"
        "0.00.003.000 D slot operator(): created speculative checkpoint "
        "(pos_min = 2, pos_max = 2, n_tokens = 3, size = 1000.000 MiB, draft = 400.000 MiB)\n"
        "0.00.004.000 D slot operator(): restoring speculative checkpoint "
        "(pos_min = 1, pos_max = 1, size = 104857600)\n"
    )

    def test_written_volume_uses_the_logged_total_only(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "arm__rep0.log"
            p.write_text(self.LOG, encoding="utf-8")
            out = json.loads(subprocess.run(
                [sys.executable, str(ROOT / "analysis" / "extract_spec_accounting.py"), str(p)],
                capture_output=True, text=True, check=True).stdout)[0]
        self.assertEqual(out["checkpoint_total_mib"], 1000.0)
        self.assertEqual(out["checkpoint_draft_component_mib"], 400.0)
        # Compare against both candidate formulas at the extractor's own
        # precision, rather than against a number retyped here: 2 creates of the
        # LOGGED TOTAL, not 2 creates of total-plus-draft-again.
        single = round(2 * 1000.0 / 1024, 2)
        doubled = round(2 * (1000.0 + 400.0) / 1024, 2)
        self.assertNotEqual(single, doubled, "fixture must distinguish the two")
        self.assertEqual(out["nominal_state_written_gib"], single)
        self.assertNotEqual(out["nominal_state_written_gib"], doubled)

    def test_the_withdrawn_wall_clock_share_is_not_emitted(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "arm__rep0.log"
            p.write_text(self.LOG, encoding="utf-8")
            out = json.loads(subprocess.run(
                [sys.executable, str(ROOT / "analysis" / "extract_spec_accounting.py"), str(p)],
                capture_output=True, text=True, check=True).stdout)[0]
        self.assertNotIn("checkpoint_share_pct", out)
        self.assertNotIn("checkpoint_excess_s", out)


def _fixture_run(d: Path, *, rows: int = 10, arms=("baseline", "arm-x"),
                 repeats: int = 2, crashed: bool = False,
                 duplicate_tag: bool = False, complete: bool = True,
                 with_tags: bool = True) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    tags = [f"p{i}" for i in range(10)]
    (d / "manifest.json").write_text(json.dumps({
        "arms": {a: [] for a in arms}, "repeats": repeats, "max_tokens": 300,
        "think": "on", "server_sha256": "x" * 64, "target_sha256": "y" * 64,
        "n_prompts": 10, "prompt_set": "fixture",
        **({"prompt_tags": tags} if with_tags else {}),
    }), encoding="utf-8")
    for a in arms:
        for rep in range(repeats):
            use = tags[:rows]
            if duplicate_tag and len(use) > 1:
                use = use[:-1] + [use[0]]
            (d / f"{a}__rep{rep}.json").write_text(json.dumps({
                "arm": a, "repeat": rep,
                "crashed": ({"tag": use[0], "error": "boom"} if crashed and rep == 0 else None),
                "rows": [{"tag": t, "predicted_n": 300, "predicted_ms": 3000.0,
                          "predicted_per_second": 100.0, "draft_n": 0,
                          "draft_n_accepted": 0, "timings": {}} for t in use],
            }), encoding="utf-8")
    if complete:
        (d / "RUN_COMPLETE.json").write_text(json.dumps({
            "expected_arm_runs": len(arms) * repeats, "arms": list(arms),
            "repeats": repeats, "n_prompts": 10}), encoding="utf-8")
    return d


class StrictAggregationMustRefuseBadRuns(unittest.TestCase):
    """Completeness read from the data marks uniform truncation as complete."""

    def _strict(self, d: Path) -> int:
        return subprocess.run(
            [sys.executable, str(ROOT / "analysis" / "matrix_report.py"), "--strict", str(d)],
            capture_output=True, text=True).returncode

    def test_a_complete_run_passes(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(self._strict(_fixture_run(Path(t) / "ok")), 0)

    def test_uniform_truncation_still_fails(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertNotEqual(self._strict(_fixture_run(Path(t) / "short", rows=9)), 0)

    def test_a_crashed_arm_run_fails(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertNotEqual(self._strict(_fixture_run(Path(t) / "crash", crashed=True)), 0)

    def test_duplicate_tags_fail(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertNotEqual(self._strict(_fixture_run(Path(t) / "dup", duplicate_tag=True)), 0)

    def test_uniform_truncation_fails_on_the_row_count_alone(self):
        """Isolates the count check: a manifest without `prompt_tags`, as runs
        written before that field existed have, leaves the row count as the only
        guard. A mutation test showed the tag-set check was masking it."""
        with tempfile.TemporaryDirectory() as t:
            d = _fixture_run(Path(t) / "short_no_tags", rows=9, with_tags=False)
            self.assertNotEqual(self._strict(d), 0)
        with tempfile.TemporaryDirectory() as t:
            d = _fixture_run(Path(t) / "full_no_tags", rows=10, with_tags=False)
            self.assertEqual(self._strict(d), 0, "a complete run must still pass")

    def test_missing_completion_marker_fails(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertNotEqual(self._strict(_fixture_run(Path(t) / "partial", complete=False)), 0)


class RunMustCoverTheExactCell(unittest.TestCase):
    """Counts are not coverage.

    Every check here defeats a count-based one: the directory holds the right
    number of files, for the right arms, with the right rows, and is still not
    the run it claims to be.
    """

    def _integrity(self, d: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(ROOT / "analysis" / "check_data_integrity.py"), str(d.parent)],
            capture_output=True, text=True)

    def _one(self, t: str, name: str, **kw) -> Path:
        root = Path(t) / "root"; root.mkdir()
        return _fixture_run(root / name, **kw)

    def test_a_well_formed_run_passes(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(self._integrity(self._one(t, "ok")).returncode, 0)

    def test_a_missing_arm_fails(self):
        with tempfile.TemporaryDirectory() as t:
            d = self._one(t, "no_arm")
            for f in d.glob("arm-x__rep*.json"):
                f.unlink()
            r = self._integrity(d)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("arm-x rep0 is missing", r.stdout + r.stderr)

    def test_a_missing_repeat_fails(self):
        with tempfile.TemporaryDirectory() as t:
            d = self._one(t, "no_rep")
            (d / "arm-x__rep1.json").unlink()
            r = self._integrity(d)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("arm-x rep1 is missing", r.stdout + r.stderr)

    def test_a_duplicated_cell_with_the_right_file_count_fails(self):
        """rep0 twice, rep1 never: four files, two arms, all rows present."""
        with tempfile.TemporaryDirectory() as t:
            d = self._one(t, "dupe_cell")
            body = json.loads((d / "arm-x__rep0.json").read_text())
            (d / "arm-x__rep1.json").write_text(json.dumps(body), encoding="utf-8")
            self.assertEqual(len(list(d.glob("*__rep*.json"))), 4)
            r = self._integrity(d)
            self.assertNotEqual(r.returncode, 0)
            out = r.stdout + r.stderr
            self.assertIn("appears 2 times", out)
            self.assertIn("rep1 is missing", out)

    def test_a_filename_that_disagrees_with_its_contents_fails(self):
        with tempfile.TemporaryDirectory() as t:
            d = self._one(t, "mislabelled")
            body = json.loads((d / "arm-x__rep1.json").read_text())
            body["arm"] = "baseline"
            (d / "arm-x__rep1.json").write_text(json.dumps(body), encoding="utf-8")
            r = self._integrity(d)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("filename says", r.stdout + r.stderr)

    def test_a_forged_completion_marker_fails(self):
        """The marker is an attestation, not proof: check it against the data."""
        with tempfile.TemporaryDirectory() as t:
            d = self._one(t, "forged")
            (d / "RUN_COMPLETE.json").write_text(json.dumps({
                "expected_arm_runs": 99, "arms": ["baseline", "arm-x", "ghost"],
                "repeats": 7, "n_prompts": 10}), encoding="utf-8")
            r = self._integrity(d)
            self.assertNotEqual(r.returncode, 0)
            out = r.stdout + r.stderr
            self.assertIn("RUN_COMPLETE arms differ", out)
            self.assertIn("RUN_COMPLETE repeats=7", out)
            self.assertIn("claims 99 arm-runs", out)

    def test_a_run_the_runner_rejected_is_not_read_as_data(self):
        with tempfile.TemporaryDirectory() as t:
            d = self._one(t, "rejected")
            (d / "RUN_FAILED.json").write_text('{"problems": ["arm-x rep1 crashed"]}',
                                               encoding="utf-8")
            r = self._integrity(d)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("RUN_FAILED", r.stdout + r.stderr)


class LatinSquareMustBeBalancedOrNotCalledLatin(unittest.TestCase):
    """`latin` was generated for any (arms, repeats) pair and labelled balanced.

    Three arms over four repeats rotates 0, 1, 2, 0, so one arm sits in the same
    position twice - and run T's manifest recorded `order_mode: latin` for
    exactly that shape.
    """

    def _build(self, arms, repeats, mode):
        import importlib.util
        os.environ.update(LLAMA_SERVER_BIN="/bin/true", MODEL_TARGET="/dev/null",
                          BENCH_OUT="/tmp/_lsq_probe")
        spec = importlib.util.spec_from_file_location("_rr_lsq", RUNNER)
        rr = importlib.util.module_from_spec(spec); spec.loader.exec_module(rr)
        return rr, rr.build_schedule(list(arms), repeats, mode)

    def test_three_arms_over_four_repeats_is_refused_as_latin(self):
        with self.assertRaises(SystemExit) as cm:
            self._build("abc", 4, "latin")
        self.assertIn("does not give one", str(cm.exception))

    def test_the_same_shape_runs_as_cyclic(self):
        rr, sched = self._build("abc", 4, "cyclic")
        self.assertEqual(len(sched), 4)
        pos = rr.position_counts(sched)
        self.assertNotEqual(sorted(pos["a"]), [1, 2, 3, 4])

    def test_balance_means_equal_visits_not_exactly_one(self):
        """Two arms over four repeats puts each arm in each position twice.
        That is balanced, and the first version of this refused it, which would
        have pushed a caller wanting four blocks onto an unbalanced `cyclic`
        run instead."""
        rr, sched = self._build("ab", 4, "latin")
        pos = rr.position_counts(sched)
        self.assertEqual(sorted(pos["a"]), [1, 1, 2, 2])
        self.assertTrue(rr.is_position_balanced(pos))
        rr2, sched2 = self._build("abc", 6, "latin")
        self.assertTrue(rr2.is_position_balanced(rr2.position_counts(sched2)))

    def test_a_repeat_count_that_is_not_a_multiple_is_still_refused(self):
        with self.assertRaises(SystemExit) as cm:
            self._build("ab", 3, "latin")
        self.assertIn("does not give one", str(cm.exception))

    def test_a_square_schedule_is_balanced(self):
        rr, sched = self._build("abc", 3, "latin")
        for arm, positions in rr.position_counts(sched).items():
            self.assertEqual(sorted(positions), [1, 2, 3], arm)

    def test_the_manifest_records_whether_it_is_balanced(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out"
            r = run_runner({"BENCH_ORDER": "cyclic", "BENCH_ARMS": "baseline",
                            "BENCH_REPEATS": "2"}, out)
            self.assertIn("schedule_is_position_balanced", r.stdout + r.stderr)


class TimerExtractionMustBeRepeatIndependent(unittest.TestCase):
    """`.replace("__rep0.log", "")` left rep1..N carrying the suffix, so a
    per-arm assertion covered rep0 alone while looking like it covered them all.
    """

    def test_every_repeat_normalises_to_the_same_arm(self):
        sys.path.insert(0, str(ROOT / "analysis"))
        import extract_checkpoint_timers as ect
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            for rep in range(4):
                (d / f"spec-draft-n8__rep{rep}.log").write_text(
                    "AUDIT_US update_tgt=1000\nAUDIT_US load_tgt=1 load_dft=0\n",
                    encoding="utf-8")
            arms = {ect.analyse(str(f))["arm"] for f in sorted(d.glob("*.log"))}
        self.assertEqual(arms, {"spec-draft-n8"},
                         f"repeats did not collapse to one arm: {sorted(arms)}")

    def test_the_repeat_index_is_preserved(self):
        sys.path.insert(0, str(ROOT / "analysis"))
        import extract_checkpoint_timers as ect
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "baseline__rep2.log"
            f.write_text("AUDIT_US update_tgt=5\n", encoding="utf-8")
            rec = ect.analyse(str(f))
        self.assertEqual((rec["arm"], rec["repeat"]), ("baseline", 2))


class RunnerMustNotAttestAFailedRun(unittest.TestCase):
    """End-to-end, against `tests/fake_llama_server.py`.

    `RUN_COMPLETE.json` used to be written unconditionally once the arm loop
    returned. `run_arm` records a crash and carries on, so a run in which an arm
    died still produced the marker that every downstream consumer reads as
    "this directory holds a whole run" - and no test could see it, because
    reaching that line needed a GPU. The stub server reaches it in a second.
    """

    FAKE = ROOT / "tests" / "fake_llama_server.py"
    PORT = "18921"

    def _run(self, out: Path, extra: dict | None = None) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env.update({
            "LLAMA_SERVER_BIN": str(self.FAKE), "MODEL_TARGET": "/dev/null",
            "BENCH_ARMS": "baseline", "BENCH_REPEATS": "2",
            "BENCH_ORDER": "cyclic", "BENCH_OUT": str(out),
            "BENCH_PORT": self.PORT, "BENCH_MAX_TOKENS": "8", "BENCH_FIT": "off",
        })
        env.update(extra or {})
        return subprocess.run([sys.executable, str(RUNNER)], env=env,
                              capture_output=True, text=True, timeout=600)

    def test_a_clean_run_is_attested(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "ok"
            r = self._run(out)
            self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
            self.assertTrue((out / "RUN_COMPLETE.json").exists())
            self.assertFalse((out / "RUN_FAILED.json").exists())
            stamp = json.loads((out / "RUN_COMPLETE.json").read_text())
            self.assertEqual(stamp["observed_arm_runs"], stamp["expected_arm_runs"])
            self.assertEqual(len(list(out.glob("*__rep*.json"))), 2)

    def test_a_crashed_arm_run_is_not_attested(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "crash"
            r = self._run(out, {"FAKE_FAIL_ON_TAG": "5"})
            self.assertNotEqual(r.returncode, 0)
            self.assertFalse((out / "RUN_COMPLETE.json").exists(),
                             "a run with a crashed arm was marked complete")
            failed = json.loads((out / "RUN_FAILED.json").read_text())
            self.assertTrue(any("crashed" in x for x in failed["problems"]))

    def test_an_arm_run_that_produced_no_tokens_is_not_attested(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "empty"
            r = self._run(out, {"FAKE_PREDICTED_N": "0"})
            self.assertNotEqual(r.returncode, 0)
            self.assertFalse((out / "RUN_COMPLETE.json").exists())
            failed = json.loads((out / "RUN_FAILED.json").read_text())
            self.assertTrue(any("produced no tokens" in x for x in failed["problems"]))

    def test_a_server_that_never_becomes_healthy_is_not_attested(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "dead"
            r = self._run(out, {"FAKE_EXIT_BEFORE_HEALTH": "1"})
            self.assertNotEqual(r.returncode, 0)
            self.assertFalse((out / "RUN_COMPLETE.json").exists())

    def test_the_identity_is_read_back_from_the_server_own_log(self):
        """`server_identity()` parses the banner rather than trusting the flags;
        its regex once required a colon after `build` and silently recorded
        nothing for all 81 arm-runs of run O2."""
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "ident"
            self._run(out, {"FAKE_BUILD": "12345", "FAKE_COMMIT": "abc1234"})
            body = json.loads((out / "baseline__rep0.json").read_text())
            self.assertEqual(body["server_identity"],
                             {"build": "12345", "commit": "abc1234"})


class ProvenanceMustIdentifyWhatRan(unittest.TestCase):
    """A manifest that names the binary and the model still does not say which
    harness asked, which prompts it sent, or whether the binary stayed the same
    between arms. Run O2 recorded an empty `server_identity` for all 81 arm-runs
    and nothing in its output said which version of the runner produced it."""

    FAKE = ROOT / "tests" / "fake_llama_server.py"

    def _run(self, out: Path, port: str, extra: dict | None = None):
        env = dict(os.environ)
        env.update({
            "LLAMA_SERVER_BIN": str(self.FAKE), "MODEL_TARGET": "/dev/null",
            "BENCH_ARMS": "baseline", "BENCH_REPEATS": "1",
            "BENCH_ORDER": "cyclic", "BENCH_OUT": str(out),
            "BENCH_PORT": port, "BENCH_MAX_TOKENS": "8", "BENCH_FIT": "off",
        })
        env.update(extra or {})
        return subprocess.run([sys.executable, str(RUNNER)], env=env,
                              capture_output=True, text=True, timeout=600)

    def test_the_manifest_identifies_the_harness_and_the_prompts(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "prov"
            self._run(out, "18931", {"BENCH_HARNESS_SHA": "cafebabe"})
            man = json.loads((out / "manifest.json").read_text())
            self.assertRegex(man["runner_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(man["prompt_set_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(man["harness_tree_sha"]["sha"], "cafebabe")

    def test_the_prompt_hash_moves_when_the_prompts_do(self):
        with tempfile.TemporaryDirectory() as t:
            a = Path(t) / "v1"; b = Path(t) / "ext"
            self._run(a, "18932", {"BENCH_PROMPTS": "v1"})
            self._run(b, "18933", {"BENCH_PROMPTS": "extended"})
            ha = json.loads((a / "manifest.json").read_text())["prompt_set_sha256"]
            hb = json.loads((b / "manifest.json").read_text())["prompt_set_sha256"]
            self.assertNotEqual(ha, hb, "two different prompt sets hashed the same")

    def test_editing_a_prompt_changes_the_hash_under_an_unchanged_name(self):
        """The failure this guards against is a prompt edited in place while the
        label stays `v1`, which no name-based identifier can see and which every
        cross-run comparison in this repository assumes cannot happen."""
        import importlib.util
        os.environ.update(LLAMA_SERVER_BIN="/bin/true", MODEL_TARGET="/dev/null",
                          BENCH_OUT="/tmp/_prompt_hash_probe")
        spec = importlib.util.spec_from_file_location("_rr_ph", RUNNER)
        rr = importlib.util.module_from_spec(spec); spec.loader.exec_module(rr)
        before = rr.prompt_set_sha256()
        name_before = rr.PROMPT_SET_NAME
        tag, sysmsg, usermsg = rr.PROMPTS[0]
        rr.PROMPTS = [(tag, sysmsg, str(usermsg) + " (edited)")] + list(rr.PROMPTS[1:])
        after = rr.prompt_set_sha256()
        self.assertEqual(rr.PROMPT_SET_NAME, name_before, "the label must not move")
        self.assertNotEqual(before, after,
                            "an edited prompt hashed the same under the same label")

    def test_each_arm_run_records_the_library_that_answered_it(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "libs"
            self._run(out, "18934")
            r = json.loads((out / "baseline__rep0.json").read_text())
            self.assertIn("server_lib_sha256", r)
            self.assertRegex(r["server_log_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(r["server_loaded_commit"], r["server_identity"]["commit"])

    def test_a_commit_that_is_not_the_expected_one_fails_the_run(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "wrong"
            r = self._run(out, "18935", {"FAKE_COMMIT": "abc1234",
                                         "BENCH_EXPECT_COMMIT": "deadbeef"})
            self.assertNotEqual(r.returncode, 0)
            self.assertFalse((out / "RUN_COMPLETE.json").exists())
            failed = json.loads((out / "RUN_FAILED.json").read_text())
            self.assertTrue(any("BENCH_EXPECT_COMMIT" in x for x in failed["problems"]))

    def test_the_expected_commit_passes(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "right"
            r = self._run(out, "18936", {"FAKE_COMMIT": "abc1234",
                                         "BENCH_EXPECT_COMMIT": "abc1234"})
            self.assertEqual(r.returncode, 0, r.stdout[-1500:] + r.stderr[-1500:])


class TeardownMustNotContaminateTheNextArm(unittest.TestCase):
    """`stop_server` waits for the driver to hand the memory back, and used to
    print a warning and carry on when it did not. The next arm then sized itself
    with `-fit on` against a stale free-memory reading, so the failure landed on
    the following arm and looked like that arm's fault."""

    def _rr(self):
        import importlib.util
        os.environ.update(LLAMA_SERVER_BIN="/bin/true", MODEL_TARGET="/dev/null",
                          BENCH_OUT="/tmp/_td_probe")
        spec = importlib.util.spec_from_file_location("_rr_td", RUNNER)
        rr = importlib.util.module_from_spec(spec); spec.loader.exec_module(rr)
        return rr

    def test_an_unsettled_teardown_fails_the_run(self):
        rr = self._rr()
        res = [{"arm": "a", "repeat": 0, "rows": [], "crashed": None,
                "teardown": {"settled": False, "mib_after": 21000,
                             "wait_s": 60.0, "readable": True}}]
        with tempfile.TemporaryDirectory() as t:
            probs = rr.validate_run(Path(t), ["a"], 1, res)
        self.assertTrue(any("21000 MiB" in x for x in probs), probs)

    def test_a_settled_teardown_raises_nothing(self):
        rr = self._rr()
        res = [{"arm": "a", "repeat": 0, "rows": [], "crashed": None,
                "teardown": {"settled": True, "mib_after": 30,
                             "wait_s": 1.0, "readable": True}}]
        with tempfile.TemporaryDirectory() as t:
            probs = rr.validate_run(Path(t), ["a"], 1, res)
        self.assertFalse([x for x in probs if "teardown" in x], probs)

    def test_a_host_without_nvidia_smi_is_not_treated_as_a_failure(self):
        """`readable` false means the reading was unavailable, not that the GPU
        held memory; CI has no GPU and must not fail on that."""
        rr = self._rr()
        res = [{"arm": "a", "repeat": 0, "rows": [], "crashed": None,
                "teardown": {"settled": True, "mib_after": None,
                             "wait_s": 0.0, "readable": False}}]
        with tempfile.TemporaryDirectory() as t:
            probs = rr.validate_run(Path(t), ["a"], 1, res)
        self.assertFalse([x for x in probs if "teardown" in x], probs)


class ArmsMustGenerateTheSameAmountOfWork(unittest.TestCase):
    """Speculation is not output-preserving on this build (ERRATA A11), so with
    thinking off the arms stop at different points and pooled throughput
    compares arms that generated different numbers of tokens. `BENCH_IGNORE_EOS`
    forces the hard cap; a run that asks for it and does not get it is not the
    run it claims to be."""

    FAKE = ROOT / "tests" / "fake_llama_server.py"

    def _run(self, out: Path, port: str, extra: dict | None = None):
        env = dict(os.environ)
        env.update({
            "LLAMA_SERVER_BIN": str(self.FAKE), "MODEL_TARGET": "/dev/null",
            "BENCH_ARMS": "baseline", "BENCH_REPEATS": "1",
            "BENCH_ORDER": "cyclic", "BENCH_OUT": str(out),
            "BENCH_PORT": port, "BENCH_MAX_TOKENS": "300", "BENCH_FIT": "off",
        })
        env.update(extra or {})
        return subprocess.run([sys.executable, str(RUNNER)], env=env,
                              capture_output=True, text=True, timeout=600)

    def test_a_short_generation_under_ignore_eos_fails_the_run(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "short"
            r = self._run(out, "18941", {"BENCH_IGNORE_EOS": "on",
                                         "FAKE_PREDICTED_N": "100"})
            self.assertNotEqual(r.returncode, 0)
            self.assertFalse((out / "RUN_COMPLETE.json").exists())
            failed = json.loads((out / "RUN_FAILED.json").read_text())
            self.assertTrue(any("BENCH_IGNORE_EOS" in x for x in failed["problems"]))

    def test_the_flag_reaches_the_server(self):
        """The stub stops early unless the request carries `ignore_eos`, so a
        run that passes with it on proves the field was actually sent."""
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "sent"
            r = self._run(out, "18942", {"BENCH_IGNORE_EOS": "on",
                                         "FAKE_SHORT_UNLESS_IGNORE_EOS": "1"})
            self.assertEqual(r.returncode, 0, r.stdout[-1500:] + r.stderr[-1500:])
            rows = json.loads((out / "baseline__rep0.json").read_text())["rows"]
            self.assertTrue(all(x["predicted_n"] == 300 for x in rows))

    def test_without_the_flag_the_short_generation_is_allowed(self):
        """Off by default: the think-on runs hit the cap anyway, and every
        archived run predates the flag."""
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "off"
            r = self._run(out, "18943", {"FAKE_PREDICTED_N": "100"})
            self.assertEqual(r.returncode, 0, r.stdout[-1500:] + r.stderr[-1500:])
            self.assertTrue((out / "RUN_COMPLETE.json").exists())
            man = json.loads((out / "manifest.json").read_text())
            self.assertFalse(man["ignore_eos"])


class StagingMustNotDestroyItsSource(unittest.TestCase):
    def test_stage_equal_to_source_is_rejected_and_nothing_is_removed(self):
        with tempfile.TemporaryDirectory() as t:
            src = Path(t) / "ckpt"
            src.mkdir()
            (src / "config.json").write_text("{}", encoding="utf-8")
            (src / "model.safetensors").write_text("x", encoding="utf-8")
            env = dict(os.environ, MTP_SRC=str(src), MTP_STAGE=str(src))
            r = subprocess.run([sys.executable, str(ROOT / "bench" / "stage_mtp_source.py")],
                               env=env, capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("must be outside", r.stdout + r.stderr)
            self.assertTrue((src / "model.safetensors").exists(),
                            "the source checkpoint must not be touched")

    def test_stage_nested_inside_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            src = Path(t) / "ckpt"
            src.mkdir()
            (src / "config.json").write_text("{}", encoding="utf-8")
            env = dict(os.environ, MTP_SRC=str(src), MTP_STAGE=str(src / "staged"))
            r = subprocess.run([sys.executable, str(ROOT / "bench" / "stage_mtp_source.py")],
                               env=env, capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)


class CoverageMustBePerFamily(unittest.TestCase):
    """Asserting the union of two families hides that one of them is short."""

    def test_union_would_pass_where_per_family_fails(self):
        records = [{"spec_type": "draft-dflash", "arm": f"spec-dflash-n{n}"} for n in (1, 2, 4, 6, 8, 16)]
        records += [{"spec_type": "draft-mtp", "arm": f"spec-mtp-n{n}"} for n in (1, 2, 8)]
        union = sorted({int(r["arm"].rsplit("n", 1)[1]) for r in records})
        self.assertEqual(union, [1, 2, 4, 6, 8, 16])
        per = {}
        for r in records:
            per.setdefault(r["spec_type"], set()).add(int(r["arm"].rsplit("n", 1)[1]))
        self.assertEqual(sorted(per["draft-dflash"]), [1, 2, 4, 6, 8, 16])
        self.assertEqual(sorted(per["draft-mtp"]), [1, 2, 8])
        self.assertNotEqual(sorted(per["draft-mtp"]), union,
                            "the union claim was true while MTP covered only 1/2/8")


class OutputDirectoryMustBeFresh(unittest.TestCase):
    def test_existing_arm_run_files_are_refused(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "out"
            out.mkdir()
            (out / "baseline__rep0.json").write_text("{}", encoding="utf-8")
            r = run_runner({}, out)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("already contains", r.stdout + r.stderr)


def _load_runner():
    import importlib.util
    os.environ.setdefault("LLAMA_SERVER_BIN", "/bin/true")
    os.environ.setdefault("MODEL_TARGET", "/dev/null")
    spec = importlib.util.spec_from_file_location("rr_under_test", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    unittest.main(verbosity=2)
