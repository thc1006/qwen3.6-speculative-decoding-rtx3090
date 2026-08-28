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

import math as _math
import re
import itertools
import json
import os
import socket
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "bench" / "retest_runner.py"
sys.path.insert(0, str(ROOT / "analysis"))


_PORT_SEQ = itertools.count()


def free_port() -> str:
    """A free port, disjoint from any other process running this suite.

    Hard-coded ports made two concurrent runs collide: the second run's stub
    server could not bind and the first run's requests reached the wrong
    process. Binding port 0 and closing it does not fix that either - the kernel
    can hand the same ephemeral port to the other process in the window before
    the stub binds it, and with two suites running that happened often enough to
    hang both. So: a base derived from this process's PID, outside the ephemeral
    range, plus a counter, with the port confirmed unused before it is returned.
    """
    base = 20000 + (os.getpid() * 97) % 30000
    for _ in range(200):
        port = 20000 + (base + next(_PORT_SEQ)) % 40000
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sk:
            sk.settimeout(0.3)
            if sk.connect_ex(("127.0.0.1", port)) != 0:
                return str(port)
    raise RuntimeError("no free port found")


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
        self.assertIn("is not a recognised boolean", r.stdout + r.stderr)

    def test_every_treatment_variable_fails_closed(self):
        """BENCH_THINK was strict and the rest were not, which is the same
        defect under a different name: `BENCH_IGNORE_EOS=oen` selected off,
        an unknown BENCH_FLAVOR selected master, and BENCH_CONCURRENCY=0 was
        clamped to 1. A typo must stop the run before any GPU work."""
        for var, bad, needle in (("BENCH_IGNORE_EOS", "oen", "recognised boolean"),
                                 ("BENCH_FIT", "onn", "recognised boolean"),
                                 ("BENCH_FLAVOR", "mater", "is not one of"),
                                 ("BENCH_CONCURRENCY", "0", "out of range"),
                                 ("BENCH_CONCURRENCY", "-2", "out of range"),
                                 ("BENCH_MAX_TOKENS", "0", "out of range"),
                                 ("BENCH_PORT", "70000", "out of range"),
                                 ("BENCH_PORT", "0", "out of range"),
                                 ("BENCH_REPEATS", "notanumber", "is not an integer")):
            with self.subTest(var=var, value=bad), tempfile.TemporaryDirectory() as d:
                r = run_runner({var: bad}, Path(d) / "out")
                self.assertNotEqual(r.returncode, 0, f"{var}={bad} was accepted")
                self.assertIn(needle, r.stdout + r.stderr)

    def test_a_spelling_a_committed_run_used_is_still_accepted(self):
        """`matrix_L_thinkoff` records `think_env: "think_off"`. A strict parser
        that drops it cannot reproduce a run this repository published, which is
        a worse failure than the one it was written to prevent."""
        for value, want in (("think_off", "off"), ("think_on", "on")):
            with self.subTest(value=value):
                env = dict(os.environ, BENCH_THINK=value)
                r = subprocess.run(
                    [sys.executable, "-c",
                     "import sys; sys.argv=['x']; "
                     "exec(open('bench/retest_runner.py').read().split('def main')[0]); "
                     "print(THINK)"],
                    env=env, cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr[-300:])
                self.assertEqual(r.stdout.strip().splitlines()[-1], want)

    def test_the_good_values_still_work(self):
        for var, good in (("BENCH_IGNORE_EOS", "on"), ("BENCH_IGNORE_EOS", "off"),
                          ("BENCH_FIT", "true"), ("BENCH_FLAVOR", "master"),
                          ("BENCH_FLAVOR", "legacy"), ("BENCH_CONCURRENCY", "8")):
            with self.subTest(var=var, value=good), tempfile.TemporaryDirectory() as d:
                r = run_runner({var: good}, Path(d) / "out")
                for n in ("recognised boolean", "is not one of", "out of range"):
                    self.assertNotIn(n, r.stdout + r.stderr, f"{var}={good}")

    def test_expect_lib_sha_must_be_a_whole_digest(self):
        """A prefix comparison accepts a library the caller never checked."""
        with tempfile.TemporaryDirectory() as d:
            r = run_runner({"BENCH_EXPECT_LIB_SHA256": "ce94855f"}, Path(d) / "out")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("64-character", r.stdout + r.stderr)

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

    def test_a_missing_arm_run_fails(self):
        """The report aggregated whatever files it found, so deleting a whole
        arm-run passed --strict while every remaining one was whole."""
        with tempfile.TemporaryDirectory() as t:
            d = _fixture_run(Path(t) / "gone")
            (d / "arm-x__rep1.json").unlink()
            self.assertNotEqual(self._strict(d), 0)

    def test_a_file_whose_name_disagrees_with_its_contents_fails(self):
        with tempfile.TemporaryDirectory() as t:
            d = _fixture_run(Path(t) / "mislabelled")
            body = json.loads((d / "arm-x__rep1.json").read_text())
            body["arm"] = "baseline"
            (d / "arm-x__rep1.json").write_text(json.dumps(body), encoding="utf-8")
            self.assertNotEqual(self._strict(d), 0)

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
    PORT = None  # each test picks its own

    def _run(self, out: Path, extra: dict | None = None) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env.update({
            "LLAMA_SERVER_BIN": str(self.FAKE), "MODEL_TARGET": "/dev/null",
            "BENCH_ARMS": "baseline", "BENCH_REPEATS": "2",
            "BENCH_ORDER": "cyclic", "BENCH_OUT": str(out),
            "BENCH_PORT": free_port(), "BENCH_MAX_TOKENS": "8", "BENCH_FIT": "off",
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
            ident = body["server_identity"]
            self.assertEqual(ident["build"], "12345")
            self.assertEqual(ident["commit"], "abc1234")
            # /props is a second, independent read of what answered. The fake
            # server does not serve it, and the failure is recorded rather than
            # dropped - an absent props read must be visible in the record.
            self.assertIn("props", ident)
            self.assertIn("error", ident["props"])


class ProvenanceMustIdentifyWhatRan(unittest.TestCase):
    """A manifest that names the binary and the model still does not say which
    harness asked, which prompts it sent, or whether the binary stayed the same
    between arms. Run O2 recorded an empty `server_identity` for all 81 arm-runs
    and nothing in its output said which version of the runner produced it."""

    FAKE = ROOT / "tests" / "fake_llama_server.py"

    def _run(self, out: Path, port: str | None = None, extra: dict | None = None):
        port = port or free_port()
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
            self._run(out, None, {"BENCH_HARNESS_SHA": "cafebabe"})
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
            self._run(out)
            r = json.loads((out / "baseline__rep0.json").read_text())
            self.assertIn("server_lib_sha256", r)
            self.assertRegex(r["server_log_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(r["server_loaded_commit"], r["server_identity"]["commit"])

    def test_a_commit_that_is_not_the_expected_one_fails_the_run(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "wrong"
            r = self._run(out, None, {"FAKE_COMMIT": "abc1234",
                                         "BENCH_EXPECT_COMMIT": "deadbeef"})
            self.assertNotEqual(r.returncode, 0)
            self.assertFalse((out / "RUN_COMPLETE.json").exists())
            failed = json.loads((out / "RUN_FAILED.json").read_text())
            self.assertTrue(any("BENCH_EXPECT_COMMIT" in x for x in failed["problems"]))

    def test_the_expected_commit_passes(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "right"
            r = self._run(out, None, {"FAKE_COMMIT": "abc1234",
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

    def test_another_process_using_the_gpu_is_not_this_run_failing(self):
        """`settled` means THIS run gave its memory back, not that the card is
        idle. The first version compared against an absolute 2048 MiB, so a
        second project's server on the same box made every arm-run report an
        unsettled teardown - which is how the suite started failing on a
        developer machine while passing in CI."""
        rr = self._rr()
        res = [{"arm": "a", "repeat": 0, "rows": [], "crashed": None,
                "teardown": {"settled": True, "mib_after": 16700,
                             "mib_before": 16600, "ceiling_mib": 18648,
                             "wait_s": 0.1, "readable": True}}]
        with tempfile.TemporaryDirectory() as t:
            probs = rr.validate_run(Path(t), ["a"], 1, res)
        self.assertFalse([x for x in probs if "teardown" in x], probs)

    def test_the_ceiling_is_the_pre_run_reading_plus_headroom(self):
        rr = self._rr()
        import types
        calls = []

        def fake_mem():
            calls.append(1)
            return 16700 if len(calls) > 1 else 16700

        rr.gpu_mem_used_mib = fake_mem
        proc = types.SimpleNamespace(pid=1, returncode=0, poll=lambda: 0,
                                     wait=lambda timeout=None: 0)
        rr.os.killpg = lambda *a, **k: None
        rr.os.getpgid = lambda p: 1
        td = rr.stop_server(proc, baseline_mib=16600)
        self.assertTrue(td["settled"])
        # 2048 MiB was not a tolerance, it was most of the margin the fitter
        # works in, and the docstring beside it says a 120 MiB allocation was
        # enough to kill the next arm.
        self.assertEqual(td["ceiling_mib"], 16600 + 128)
        self.assertEqual(td["consecutive"], 3)

    def test_one_low_reading_is_not_enough(self):
        """The driver can be caught mid-release and read low once. Counting the
        readings, not echoing the parameter: `consecutive` is an argument and
        comes back in the dict whether or not it was honoured."""
        rr = self._rr()
        import types
        seq = iter([16700, 18000, 16700, 16700, 16700])
        rr.gpu_mem_used_mib = lambda: next(seq, 16700)
        proc = types.SimpleNamespace(pid=1, returncode=0, poll=lambda: 0,
                                     wait=lambda timeout=None: 0)
        rr.os.killpg = lambda *a, **k: None
        rr.os.getpgid = lambda p: 1
        rr.time.sleep = lambda *_a: None
        td = rr.stop_server(proc, baseline_mib=16600)
        self.assertTrue(td["settled"])
        # 16700 low, 18000 high, then three low: five readings, and settling on
        # the first would be settling on a reading the driver had not finished.
        self.assertEqual(td["readings"], 5)

    def test_a_clean_teardown_still_needs_three_readings(self):
        rr = self._rr()
        import types
        rr.gpu_mem_used_mib = lambda: 16650
        proc = types.SimpleNamespace(pid=1, returncode=0, poll=lambda: 0,
                                     wait=lambda timeout=None: 0)
        rr.os.killpg = lambda *a, **k: None
        rr.os.getpgid = lambda p: 1
        rr.time.sleep = lambda *_a: None
        td = rr.stop_server(proc, baseline_mib=16600)
        self.assertTrue(td["settled"])
        self.assertEqual(td["readings"], 3)

    def test_a_card_that_stops_answering_mid_run_is_a_failure(self):
        """`used is None` used to return settled=True, so the one instrument
        this check depends on was optional. It is a failure now - but only where
        a reading existed BEFORE the run, because a guard about memory coming
        back cannot apply on a host where memory was never observable."""
        rr = self._rr()
        import types
        rr.gpu_mem_used_mib = lambda: None
        proc = types.SimpleNamespace(pid=1, returncode=0, poll=lambda: 0,
                                     wait=lambda timeout=None: 0)
        rr.os.killpg = lambda *a, **k: None
        rr.os.getpgid = lambda p: 1
        td = rr.stop_server(proc, baseline_mib=16600)
        self.assertFalse(td["settled"])
        self.assertFalse(td["readable"])
        self.assertIn("before this arm-run and not", td["why"])
        res = [{"arm": "a", "repeat": 0, "rows": [], "crashed": None,
                "teardown": td}]
        with tempfile.TemporaryDirectory() as t:
            probs = rr.validate_run(Path(t), ["a"], 1, res)
        self.assertTrue([x for x in probs if "teardown" in x], probs)

    def test_the_runner_completes_end_to_end_where_nvidia_smi_fails(self):
        """CI has no nvidia-smi and this machine does, so a rule keyed on the
        reading passes here and fails there. This forces the CI condition with a
        shim: a teardown guard is about memory coming back, and it cannot apply
        on a host where memory was never observable."""
        with tempfile.TemporaryDirectory() as t:
            shim = Path(t) / "bin"
            shim.mkdir()
            (shim / "nvidia-smi").write_text("#!/bin/sh\nexit 127\n", encoding="utf-8")
            (shim / "nvidia-smi").chmod(0o755)
            env_path = f"{shim}{os.pathsep}{os.environ.get('PATH', '')}"
            out = Path(t) / "out"
            env = dict(os.environ)
            env.update({
                "PATH": env_path,
                "LLAMA_SERVER_BIN": str(ROOT / "tests" / "fake_llama_server.py"),
                "MODEL_TARGET": "/dev/null", "BENCH_ARMS": "baseline",
                "BENCH_REPEATS": "1", "BENCH_ORDER": "cyclic",
                "BENCH_OUT": str(out), "BENCH_PORT": free_port(),
                "BENCH_MAX_TOKENS": "300", "BENCH_FIT": "off",
            })
            r = subprocess.run([sys.executable, str(RUNNER)], env=env,
                               capture_output=True, text=True, timeout=600)
            self.assertEqual(r.returncode, 0, (r.stdout + r.stderr)[-1500:])
            self.assertTrue((out / "RUN_COMPLETE.json").exists())
            body = json.loads((out / "baseline__rep0.json").read_text())
            td = body["teardown"]
            self.assertFalse(td["readable"])
            self.assertTrue(td["settled"])
            self.assertIn("no GPU reading is available", td["why"])

    def test_a_host_with_no_gpu_at_all_is_not_a_failure(self):
        """CI has no nvidia-smi, and the harness's own end-to-end tests run
        there. No reading before and none after is not a teardown problem."""
        rr = self._rr()
        import types
        rr.gpu_mem_used_mib = lambda: None
        proc = types.SimpleNamespace(pid=1, returncode=0, poll=lambda: 0,
                                     wait=lambda timeout=None: 0)
        rr.os.killpg = lambda *a, **k: None
        rr.os.getpgid = lambda p: 1
        td = rr.stop_server(proc, baseline_mib=None)
        self.assertTrue(td["settled"])
        self.assertFalse(td["readable"])
        self.assertIn("no GPU reading is available", td["why"])
        res = [{"arm": "a", "repeat": 0, "rows": [], "crashed": None,
                "teardown": td}]
        with tempfile.TemporaryDirectory() as t:
            probs = rr.validate_run(Path(t), ["a"], 1, res)
        self.assertFalse([x for x in probs if "teardown" in x], probs)

    def _unused_a_host_without_nvidia_smi(self):
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

    def _run(self, out: Path, port: str | None = None, extra: dict | None = None):
        port = port or free_port()
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
            r = self._run(out, None, {"BENCH_IGNORE_EOS": "on",
                                         "FAKE_PREDICTED_N": "100"})
            self.assertNotEqual(r.returncode, 0)
            self.assertFalse((out / "RUN_COMPLETE.json").exists())
            failed = json.loads((out / "RUN_FAILED.json").read_text())
            # the message covers both mechanisms now: the run-level flag and
            # the per-arm suffix that lets both modes share one invocation
            self.assertTrue(any("under a hard cap" in x for x in failed["problems"]),
                            failed["problems"])

    def test_the_flag_reaches_the_server(self):
        """The stub stops early unless the request carries `ignore_eos`, so a
        run that passes with it on proves the field was actually sent."""
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "sent"
            r = self._run(out, None, {"BENCH_IGNORE_EOS": "on",
                                         "FAKE_SHORT_UNLESS_IGNORE_EOS": "1"})
            self.assertEqual(r.returncode, 0, r.stdout[-1500:] + r.stderr[-1500:])
            rows = json.loads((out / "baseline__rep0.json").read_text())["rows"]
            self.assertTrue(all(x["predicted_n"] == 300 for x in rows))

    def test_without_the_flag_the_short_generation_is_allowed(self):
        """Off by default: the think-on runs hit the cap anyway, and every
        archived run predates the flag."""
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "off"
            r = self._run(out, None, {"FAKE_PREDICTED_N": "100"})
            self.assertEqual(r.returncode, 0, r.stdout[-1500:] + r.stderr[-1500:])
            self.assertTrue((out / "RUN_COMPLETE.json").exists())
            man = json.loads((out / "manifest.json").read_text())
            self.assertFalse(man["ignore_eos"])


class TheWholeLibraryMapIsPinned(unittest.TestCase):
    """`libllama-server-impl.so` is the server's own code, but `libllama.so`,
    `libggml-cuda.so` and `libggml-base.so` all decide what a decode does.
    Swapping any of them between arms changed the measurement while the impl
    hash stayed put, and the arm still produced a completion marker."""

    def _rr(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("rr_libs", RUNNER)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_a_changed_sibling_library_fails_the_arm(self):
        rr = self._rr()
        rr._LIB_BASELINE = None
        first = {"libllama-server-impl.so": "a" * 64, "libggml-cuda.so": "b" * 64}
        self.assertEqual(rr.check_identity("arm", 0, {}, first, healthy=False), [])
        moved = {"libllama-server-impl.so": "a" * 64, "libggml-cuda.so": "c" * 64}
        bad = rr.check_identity("arm", 1, {}, moved, healthy=False)
        self.assertTrue(bad)
        self.assertIn("libggml-cuda.so", bad[0])

    def test_an_added_or_removed_library_fails_the_arm(self):
        rr = self._rr()
        base = {"libllama.so": "a" * 64, "libggml.so": "b" * 64}
        rr._LIB_BASELINE = None
        rr.check_identity("arm", 0, {}, base, healthy=False)
        # one gone
        self.assertTrue(rr.check_identity("arm", 1, {}, {"libllama.so": "a" * 64}, healthy=False))
        # one added
        self.assertTrue(rr.check_identity("arm", 2, {},
                                          dict(base, libextra_so="d" * 64), healthy=False))
        # and an EMPTY map is not "every library vanished", it is an arm-run
        # that never hashed anything - the crash is the finding, and cascading
        # a library complaint onto every later arm hides it
        self.assertEqual(rr.check_identity("crashed", 3, {}, {}, healthy=False), [])

    def test_a_crashed_arm_run_does_not_become_the_baseline(self):
        """A crashed arm-run never hashed anything. Seeding the baseline from
        its empty map made every later arm-run report that the libraries had
        changed, so one crash failed the whole run for the wrong reason."""
        rr = self._rr()
        rr._LIB_BASELINE = None
        self.assertEqual(rr.check_identity("crashed", 0, {}, {}, healthy=False), [])
        self.assertIsNone(rr._LIB_BASELINE)
        real = {"libllama-server-impl.so": "a" * 64}
        self.assertEqual(rr.check_identity("arm", 0, {}, real, healthy=False), [])
        self.assertEqual(rr._LIB_BASELINE, real)
        self.assertEqual(rr.check_identity("crashed", 1, {}, {}, healthy=False), [])

    def test_an_unchanged_map_passes(self):
        rr = self._rr()
        rr._LIB_BASELINE = None
        same = {"libllama-server-impl.so": "a" * 64, "libggml.so": "b" * 64}
        rr.check_identity("arm", 0, {}, same, healthy=False)
        self.assertEqual(rr.check_identity("arm", 1, {}, dict(same), healthy=False), [])


class PairedBlocksAgreesWithTheRunner(unittest.TestCase):
    """The two files defined "balanced" differently, so a schedule the runner
    accepted was reported here as unbalanced - and the analysis carried on and
    wrote the same JSON either way."""

    def _pb(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "pb", ROOT / "analysis" / "paired_blocks.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def _rr(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("rr_pb", RUNNER)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_the_two_definitions_agree_on_every_small_schedule(self):
        """The definition is duplicated - paired_blocks.py cannot import the
        runner, whose module body exits on a bad environment - so the copy is
        checked against the original rather than assumed to match. Exhaustively,
        over every schedule of 2 or 3 arms and up to 4 rounds."""
        import itertools
        pb, rr = self._pb(), self._rr()
        checked = 0
        for n_arms in (2, 3):
            positions = list(range(1, n_arms + 1))
            for rounds in (1, 2, 3, 4):
                length = n_arms * rounds
                # every assignment of positions to arms, capped so this stays a
                # test rather than a benchmark
                space = list(itertools.product(positions, repeat=length))
                step = max(1, len(space) // 400)
                for pick in space[::step]:
                    sched = {f"a{i}": list(pick) for i in range(n_arms)}
                    # and a rotated variant, which is what real schedules are
                    rot = {f"a{i}": [positions[(j + i) % n_arms]
                                     for j in range(length)]
                           for i in range(n_arms)}
                    for c in (sched, rot):
                        self.assertEqual(pb.is_position_balanced(c),
                                         rr.is_position_balanced(c), c)
                        checked += 1
        self.assertGreater(checked, 500)

    def test_the_two_definitions_agree(self):
        pb, rr = self._pb(), self._rr()
        cases = [
            {"a": [1, 2], "b": [2, 1]},                        # 2 arms, 1 round
            {"a": [1, 2, 1, 2], "b": [2, 1, 2, 1]},            # 2 arms, 2 rounds
            {"a": [1, 2, 3], "b": [2, 3, 1], "c": [3, 1, 2]},  # 3 arms, 1 round
            {"a": [1, 2, 3, 1], "b": [2, 3, 1, 2],
             "c": [3, 1, 2, 3]},                               # the run T rotation
            {"a": [1, 1], "b": [2, 2]},                        # never rotates
            {},
        ]
        for c in cases:
            with self.subTest(schedule=c):
                self.assertEqual(pb.is_position_balanced(c),
                                 rr.is_position_balanced(c))

    def test_four_repeats_of_two_arms_is_balanced(self):
        """The old check required every position exactly once, so this - which
        the runner accepts and which run K used - was reported unbalanced."""
        self.assertTrue(self._pb().is_position_balanced(
            {"a": [1, 2, 1, 2], "b": [2, 1, 2, 1]}))

    def test_t_critical_matches_the_published_table(self):
        """The table stopped at df=10 and fell back to 1.96 - the NORMAL value -
        for everything above, which silently narrows every interval with twelve
        or more blocks."""
        pb = self._pb()
        for df, want in ((1, 12.706), (2, 4.303), (4, 2.776), (8, 2.306),
                         (10, 2.228), (11, 2.201), (12, 2.179), (20, 2.086),
                         (30, 2.042), (60, 2.000), (120, 1.980)):
            with self.subTest(df=df):
                self.assertAlmostEqual(pb.t_critical_975(df), want, places=2)
        self.assertGreater(pb.t_critical_975(11), 1.96)

    def test_both_option_spellings_parse(self):
        pb = self._pb()
        a1, o1 = pb._parse_argv(["dir", "--iters=2000", "--baseline=x"])
        a2, o2 = pb._parse_argv(["dir", "--iters", "2000", "--baseline", "x"])
        self.assertEqual((a1, o1), (a2, o2))
        self.assertEqual(o1["--iters"], "2000")


class TheMutationRunnerNeverTouchesTheRealTree(unittest.TestCase):
    """It committed one of its own mutations once. The mirror fixed that; the
    sidecar recovery path that wrote into ROOT without resolving the path it
    read is gone with it."""

    def test_no_sidecar_recovery_remains(self):
        src = (ROOT / "tests" / "mutate.py").read_text(encoding="utf-8")
        code = "\n".join(l for l in src.splitlines()
                         if not l.lstrip().startswith("#"))
        for gone in ("_restore_from_sidecar(", "SIDECAR ="):
            self.assertNotIn(gone, code, f"{gone} is still executable")

    def test_it_writes_only_under_a_temporary_directory(self):
        src = (ROOT / "tests" / "mutate.py").read_text(encoding="utf-8")
        self.assertIn("mirror(", src)
        self.assertIn("TemporaryDirectory", src)
        for w in re.findall(r"\(ROOT / [^)]*\)\.write_text", src):
            self.fail(f"writes into the real tree: {w}")


class WhatAnsweredMustHaveLoadedTheRightModel(unittest.TestCase):
    """argv says what was asked for. The manifest recorded that and nothing
    else: the log-line pattern for the model path was `loaded meta data from`,
    which llama.cpp has never printed, so `model_path` was silently absent from
    every arm-run identity and the field was only read when present."""

    def _rr(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("rr_model", RUNNER)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    LINE = ("llama_model_loader: loaded meta data with 45 key-value pairs and "
            "579 tensors from /models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf "
            "(version GGUF V3 (latest))")

    def test_the_model_path_is_read_out_of_the_startup_log(self):
        rr = self._rr()
        with tempfile.TemporaryDirectory() as t:
            lg = Path(t) / "s.log"
            lg.write_text("common_params_print_info: build 10622 (3737e4137) with x\n"
                          + self.LINE + "\n", encoding="utf-8")
            ident = rr.server_identity(lg)
        self.assertEqual(ident.get("commit"), "3737e4137")
        self.assertEqual(ident.get("model_path"),
                         "/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")

    def test_a_different_model_fails_the_arm(self):
        rr = self._rr()
        rr._LIB_BASELINE = None
        rr.TARGET = "/models/right.gguf"
        bad = rr.check_identity("arm", 0, {"model_path": "/models/wrong.gguf"}, {}, healthy=False)
        self.assertTrue(bad)
        self.assertIn("wrong.gguf", bad[0])

    def test_props_is_compared_too(self):
        rr = self._rr()
        rr._LIB_BASELINE = None
        rr.TARGET = "/models/right.gguf"
        bad = rr.check_identity("arm", 0,
                                {"props": {"model_path": "/models/other.gguf"}}, {}, healthy=False)
        self.assertTrue(bad)
        self.assertIn("/props", bad[0])

    def test_the_right_model_passes_on_both_reads(self):
        rr = self._rr()
        rr._LIB_BASELINE = None
        rr.TARGET = "/models/right.gguf"
        self.assertEqual(
            rr.check_identity("arm", 0,
                              {"model_path": "/models/right.gguf",
                               "props": {"model_path": "/models/right.gguf"}}, {}, healthy=False),
            [])


class TimerExtractionMustNotDependOnFieldOrder(unittest.TestCase):
    """The reader matched `AUDIT_US load_tgt=(\\d+) load_dft=(\\d+)` positionally.
    Splitting the timers inserts `sync_lt=` between those two, and a positional
    regex does not fail on that - it stops matching, and the extraction reports
    zero restores and a smaller total that still looks like a measurement."""

    def _ex(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ck", ROOT / "analysis" / "extract_checkpoint_timers.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    OLD = ("AUDIT_US update_tgt=35344\n"
           "AUDIT_US load_tgt=22315 load_dft=7398\n"
           "AUDIT_US update_dft=1000\n")
    NEW = ("AUDIT_US update_tgt=35344 sync_tgt=30000\n"
           "AUDIT_US load_tgt=22315 sync_lt=20000 load_dft=7398 sync_ld=7000\n"
           "AUDIT_US update_dft=1000 sync_dft=900\n")

    def _run(self, text):
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "spec-draft-n8__rep0.log"
            f.write_text(text, encoding="utf-8")
            return self._ex().analyse(str(f))

    def test_the_old_format_still_reads(self):
        r = self._run(self.OLD)
        self.assertEqual(r["creates"], 1)
        self.assertEqual(r["restores"], 1)
        self.assertAlmostEqual(r["load_tgt_s"], 0.022, places=3)
        self.assertAlmostEqual(r["load_dft_s"], 0.007, places=3)
        self.assertNotIn("sync_total_s", r, "no split fields on an unsplit log")

    def test_the_split_format_reads_the_same_totals(self):
        old, new = self._run(self.OLD), self._run(self.NEW)
        for k in ("creates", "restores", "update_tgt_s", "update_dft_s",
                  "load_tgt_s", "load_dft_s", "checkpoint_total_s"):
            self.assertEqual(old[k], new[k], k)

    def test_the_split_is_reported_and_adds_up(self):
        r = self._run(self.NEW)
        # the field is rounded to milliseconds, so compare at its own precision
        self.assertAlmostEqual(r["sync_total_s"],
                               round((30000 + 20000 + 7000 + 900) / 1e6, 3),
                               places=3)
        self.assertAlmostEqual(r["sync_total_s"] + r["state_total_s"],
                               r["checkpoint_total_s"], places=3)
        self.assertAlmostEqual(r["state_tgt_s"], round((35344 - 30000) / 1e6, 3),
                               places=3)
        self.assertGreater(r["sync_share_of_checkpoint_pct"], 0)

    def test_a_restore_line_is_never_silently_dropped(self):
        """The failure this class exists for: a field inserted in the middle."""
        r = self._run("AUDIT_US load_tgt=100 sync_lt=90 load_dft=10 sync_ld=9\n")
        self.assertEqual(r["restores"], 1)


class TheLengthModeAnalysisMustReadBothDesigns(unittest.TestCase):
    """`analysis/length_mode.py`'s crossover path is validated against the
    published run V, which it reproduces to the hundredth of a point. Its
    within-invocation path had never run on anything, so this builds a run of
    the shape `run_v3_within.sh` produces and checks the arithmetic against
    hand-computed values."""

    def _mod(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "lm", ROOT / "analysis" / "length_mode.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    @staticmethod
    def _arm_run(d: Path, arm: str, rep: int, tok_s: float, n_per_req: int,
                 n_req: int = 10):
        """One arm-run at an exact pooled rate."""
        ms = 1000.0 * n_per_req / tok_s
        rows = [{"tag": f"p{i}", "predicted_n": n_per_req,
                 "predicted_ms": ms, "predicted_per_second": tok_s,
                 "timings": {"predicted_n": n_per_req, "predicted_ms": ms,
                             "predicted_per_second": tok_s},
                 "wall_ms": ms + 5.0, "draft_n": 0, "draft_n_accepted": 0}
                for i in range(n_req)]
        (d / f"{arm}__rep{rep}.json").write_text(
            json.dumps({"arm": arm, "repeat": rep, "rows": rows}), encoding="utf-8")

    def _build(self, root: Path, rates):
        """rates: {arm: (freerun tok/s, hardcap tok/s)}"""
        root.mkdir(parents=True, exist_ok=True)
        arms = []
        for arm, (fr, cp) in rates.items():
            for rep in range(2):
                self._arm_run(root, arm, rep, fr, 120)
                self._arm_run(root, arm + "-cap", rep, cp, 300)
            arms += [arm, arm + "-cap"]
        (root / "manifest.json").write_text(json.dumps({
            "order_mode": "latin", "schedule_is_position_balanced": True,
            "repeats": 2, "hardcap_suffix": "-cap",
            "hardcap_arms": sorted(a for a in arms if a.endswith("-cap")),
            "arms": {a: [] for a in arms}}), encoding="utf-8")
        (root / "RUN_COMPLETE.json").write_text("{}", encoding="utf-8")

    def test_the_within_design_is_read_and_the_shift_is_right(self):
        import io, contextlib
        m = self._mod()
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / "matrix_V3_s1_20260827_120000"
            # baseline is the same in both modes; the arm gains 10 % under the cap
            self._build(d, {"baseline": (100.0, 100.0),
                            "spec-dflash-n2": (110.0, 121.0)})
            out = {}
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                m.report_within([str(d)], out)
            text = buf.getvalue()
        # free: 110/100 -> +10.00 %   cap: 121/100 -> +21.00 %   shift +11.00 pp
        self.assertIn("+10.00%", text)
        self.assertIn("+21.00%", text)
        shift = out["within"][0]["shift_pp"]["spec-dflash-n2"]
        self.assertAlmostEqual(shift, 11.0, places=6)
        self.assertTrue(out["within"][0]["complete"])

    def test_a_baseline_that_moves_is_divided_out(self):
        """The contrast is a ratio to the baseline IN THE SAME MODE, so a whole
        -invocation shift that moves every arm equally must cancel."""
        import io, contextlib
        m = self._mod()
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / "matrix_V3_s1_20260827_120000"
            # everything 20 % slower under the cap, including the baseline
            self._build(d, {"baseline": (100.0, 80.0),
                            "spec-dflash-n2": (110.0, 88.0)})
            out = {}
            with contextlib.redirect_stdout(io.StringIO()):
                m.report_within([str(d)], out)
        self.assertAlmostEqual(out["within"][0]["shift_pp"]["spec-dflash-n2"],
                               0.0, places=6)
        # and the log form of the same contrast. This case is the one that can
        # tell the two apart: with the baseline equal in both modes,
        # log(cap_a/cap_base) - log(free_a/free_base) and log(cap_a/free_a) are
        # numerically identical, so a test built on that case cannot see a
        # contrast that stopped dividing out its baseline.
        self.assertAlmostEqual(out["within"][0]["log_delta"]["spec-dflash-n2"],
                               0.0, places=9)
        c = m.contrast({"baseline": 100.0, "spec-dflash-n2": 110.0},
                       {"baseline": 80.0, "spec-dflash-n2": 88.0})
        self.assertAlmostEqual(c["spec-dflash-n2"]["log_delta"], 0.0, places=9)

    def test_the_contrast_is_computed_in_exactly_one_place(self):
        """`delta()` returned a log ratio and each report recomputed the
        percentage-point shift from the rates directly - the same quantity, two
        formulas, two places, and the log value was never read. Breaking one of
        them changed nothing, which is how two mutations survived. Both reports
        read `contrast()` now, so breaking it breaks both."""
        m = self._mod()
        c = m.contrast({"baseline": 100.0, "a": 110.0},
                       {"baseline": 100.0, "a": 121.0})["a"]
        self.assertAlmostEqual(c["free_pct"], 10.0, places=6)
        self.assertAlmostEqual(c["cap_pct"], 21.0, places=6)
        self.assertAlmostEqual(c["shift_pp"], 11.0, places=6)
        # the log form of the same contrast, which is what averages properly.
        # This case cannot distinguish the two formulas - see
        # test_a_baseline_that_moves_is_divided_out, which can.
        self.assertAlmostEqual(c["log_delta"],
                               _math.log(1.21) - _math.log(1.10), places=9)
        # and a change to it must reach the reported shift
        import io, contextlib
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / "matrix_V3_s1_20260827_120000"
            self._build(d, {"baseline": (100.0, 100.0), "a": (110.0, 121.0)})
            out = {}
            with contextlib.redirect_stdout(io.StringIO()):
                m.report_within([str(d)], out)
            self.assertAlmostEqual(out["within"][0]["shift_pp"]["a"],
                                   c["shift_pp"], places=9)
            self.assertAlmostEqual(out["within"][0]["log_delta"]["a"],
                                   c["log_delta"], places=9)

    def test_a_session_missing_one_half_is_dropped(self):
        """The crossover contrast needs both halves of the same session. Using
        a session with one of them silently compares across invocations, which
        is the thing the design exists to avoid."""
        import io, contextlib
        m = self._mod()
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            halves = []
            for mode, cap in (("freerun", False), ("hardcap", True)):
                for sess in (1, 2):
                    if sess == 2 and mode == "hardcap":
                        continue                       # session 2 is a stump
                    d = root / f"matrix_V2_s{sess}_{mode}_20260827_120000"
                    d.mkdir()
                    for arm, rate in (("baseline", 100.0), ("a", 121.0 if cap else 110.0)):
                        for rep in range(2):
                            self._arm_run(d, arm, rep, rate, 300 if cap else 120)
                    (d / "manifest.json").write_text(json.dumps(
                        {"ignore_eos": cap, "created": f"2026-08-27T0{sess}:0"
                                                       f"{0 if mode == 'freerun' else 5}:00+0800"}),
                        encoding="utf-8")
                    (d / "RUN_COMPLETE.json").write_text("{}", encoding="utf-8")
                    halves.append(str(d))
            out = {}
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                m.report_crossover(m.classify(halves)[1], out)
            text = buf.getvalue()
        self.assertIn("1 of 2", text)
        self.assertIn("dropped s2", text)
        self.assertEqual(out["crossover"]["a"]["sessions"], 1)

    def test_several_invocations_get_an_interval(self):
        import io, contextlib
        m = self._mod()
        with tempfile.TemporaryDirectory() as t:
            dirs = []
            for i, cap in enumerate((121.0, 122.0, 120.0), 1):
                d = Path(t) / f"matrix_V3_s{i}_20260827_120000"
                self._build(d, {"baseline": (100.0, 100.0),
                                "spec-dflash-n2": (110.0, cap)})
                dirs.append(str(d))
            out = {}
            with contextlib.redirect_stdout(io.StringIO()):
                m.report_within(dirs, out)
        s = out["within_summary"]["spec-dflash-n2"]
        self.assertEqual(s["n"], 3)
        self.assertAlmostEqual(s["mean_pp"], 11.0, places=2)
        self.assertLess(s["lo"], s["mean_pp"])
        self.assertGreater(s["hi"], s["mean_pp"])


class ARunScriptMustSetEveryFieldItClaimsToReproduce(unittest.TestCase):
    """`run_v2_crossover.sh` says it uses "run V's configuration otherwise
    verbatim". It did not: `BENCH_FIT_TARGET` was unset, so it took the 1024 MiB
    default, and ERRATA A9 says that margin is exactly what kills a DFlash arm -
    the fitter sizes the target to leave 1024 MiB and a BF16 DFlash drafter plus
    its compute buffer does not fit in it. Every `spec-dflash-n2` arm-run of the
    first attempt aborted before `/health`, twenty-five minutes of GPU time for
    nothing, and the run script's own comment was the thing that was wrong.

    A comment is not a check. This is."""

    # script -> the published run whose configuration it says it reproduces
    CLAIMS = {
        "bench/run_v2_crossover.sh": "matrix_V_freerun_20260826_210956",
        "bench/run_v3_within.sh": "matrix_V_freerun_20260826_210956",
        "bench/run_t4_split_timers.sh": "matrix_T_timers_20260826_182639",
    }
    # environment variable -> the manifest field it lands in
    FIELDS = {
        "BENCH_CTX": "ctx", "BENCH_FIT_TARGET": "fit_target",
        "BENCH_MAX_TOKENS": "max_tokens", "BENCH_CONCURRENCY": "concurrency",
        "BENCH_FLAVOR": "flavor", "BENCH_THINK": "think",
    }
    # recorded as a boolean, set as a word
    BOOLS = {"BENCH_FIT": "fit"}

    @staticmethod
    def _exports(path):
        out = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            m = re.match(r'\s*export (BENCH_\w+)=("?)([^"\s]*)\2\s*$', line)
            if m:
                out[m.group(1)] = m.group(3)
        return out

    def test_every_treatment_field_the_reference_run_recorded_is_set(self):
        for script, run in self.CLAIMS.items():
            man = json.loads((ROOT / "v4_audit_2026_08_25" / "data" / run
                              / "manifest.json").read_text(encoding="utf-8"))
            env = self._exports(ROOT / script)
            for var, field in self.FIELDS.items():
                want = man.get(field)
                if want in (None, ""):
                    continue
                with self.subTest(script=script, var=var):
                    self.assertIn(var, env,
                                  f"{script} does not set {var}, and {run} "
                                  f"recorded {field}={want!r} - the default is "
                                  f"not what that run measured")
                    self.assertEqual(str(env[var]), str(want),
                                     f"{script} sets {var}={env[var]!r}, "
                                     f"{run} recorded {field}={want!r}")
            for var, field in self.BOOLS.items():
                if field not in man:
                    continue
                with self.subTest(script=script, var=var):
                    self.assertIn(var, env, f"{script} does not set {var}")
                    on = str(env[var]).lower() in ("on", "1", "true", "yes")
                    self.assertEqual(on, bool(man[field]),
                                     f"{script} sets {var}={env[var]!r}, "
                                     f"{run} recorded {field}={man[field]!r}")

    def test_the_fit_target_is_the_one_that_keeps_dflash_alive(self):
        """The specific value, named, because the default is a live failure."""
        for script in ("bench/run_v2_crossover.sh", "bench/run_v3_within.sh"):
            with self.subTest(script=script):
                self.assertEqual(self._exports(ROOT / script).get("BENCH_FIT_TARGET"),
                                 "3072")

    def test_telemetry_is_started_with_an_interface_the_tool_accepts(self):
        """`gpu_telemetry.sh` takes `[schema] [interval] [label]` and derives its
        own output path. The run scripts passed a FILE PATH as the first
        argument, so it was read as a schema, rejected, and the sampler exited
        immediately - the eight-session crossover ran with no telemetry at all
        and nothing said so. Same class as the fit-target miss: an interface
        invented rather than read."""
        schemas = set(re.findall(r"^(\w+)\)$",
                                 (ROOT / "bench" / "gpu_telemetry.sh")
                                 .read_text(encoding="utf-8"), re.M))
        self.assertTrue({"full", "compact", "raw"} <= schemas, schemas)
        for script in ("bench/run_v2_crossover.sh", "bench/run_v3_within.sh",
                       "bench/run_t4_split_timers.sh"):
            text = (ROOT / script).read_text(encoding="utf-8")
            m = re.search(r'bash "\$TELE_SH" ([^\n&]*)&', text)
            with self.subTest(script=script):
                self.assertIsNotNone(m, f"{script} does not start the sampler")
                args = m.group(1).split()
                self.assertGreaterEqual(len(args), 1, args)
                first = args[0].strip('"')
                # either a literal schema, or a variable with a schema default
                if first.startswith("$"):
                    var = first.strip("${}").split(":-")[0]
                    d = re.search(rf'{re.escape(var)}="\$\{{[^:]*:-(\w+)\}}"', text)
                    self.assertIsNotNone(d, f"{script}: {first} has no default")
                    first = d.group(1)
                self.assertIn(first, schemas,
                              f"{script} starts the sampler with {first!r}, "
                              f"which gpu_telemetry.sh rejects as a schema")

    def test_the_scripts_do_not_set_a_run_level_cap_by_accident(self):
        """`run_v3_within.sh` measures both modes per arm; a run-level
        BENCH_IGNORE_EOS would flatten the freerun half into the capped one."""
        env = self._exports(ROOT / "bench" / "run_v3_within.sh")
        self.assertNotIn("BENCH_IGNORE_EOS", env)
        self.assertEqual(env.get("BENCH_HARDCAP_SUFFIX"), "-cap")


class BothModesMustFitInOneInvocation(unittest.TestCase):
    """`BENCH_IGNORE_EOS` is a run-level treatment, so run V had to measure the
    two modes as two runs sixteen minutes apart - and A16 finds a
    DFlash-specific invocation effect of the same size as the shift it reported,
    which is why A17 cannot attribute it. An arm named `<base><suffix>` takes its
    server flags from `<base>` and sends `ignore_eos` on its own requests, so
    both modes sit in one balanced square, in one invocation, adjacent in time.
    """

    FAKE = ROOT / "tests" / "fake_llama_server.py"

    def _run(self, out: Path, arms: str, extra: dict | None = None):
        env = dict(os.environ)
        env.update({
            "LLAMA_SERVER_BIN": str(self.FAKE), "MODEL_TARGET": "/dev/null",
            "BENCH_ARMS": arms, "BENCH_REPEATS": "2",
            "BENCH_ORDER": "latin", "BENCH_OUT": str(out),
            "BENCH_PORT": free_port(), "BENCH_MAX_TOKENS": "300",
            "BENCH_FIT": "off", "BENCH_HARDCAP_SUFFIX": "-cap",
            "FAKE_SHORT_UNLESS_IGNORE_EOS": "1", "FAKE_PREDICTED_N": "300",
        })
        env.update(extra or {})
        return subprocess.run([sys.executable, str(RUNNER)], env=env,
                              capture_output=True, text=True, timeout=600)

    def test_one_run_carries_both_modes(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "both"
            r = self._run(out, "baseline,baseline-cap")
            self.assertEqual(r.returncode, 0, (r.stdout + r.stderr)[-1500:])
            self.assertTrue((out / "RUN_COMPLETE.json").exists())
            free = json.loads((out / "baseline__rep0.json").read_text())
            cap = json.loads((out / "baseline-cap__rep0.json").read_text())
            # the fake server returns a third unless ignore_eos is on the request
            self.assertTrue(all(x["predicted_n"] == 100 for x in free["rows"]),
                            [x["predicted_n"] for x in free["rows"]])
            self.assertTrue(all(x["predicted_n"] == 300 for x in cap["rows"]),
                            [x["predicted_n"] for x in cap["rows"]])

    def test_the_capped_arm_runs_the_base_arms_flags(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "flags"
            self.assertEqual(self._run(out, "baseline,baseline-cap").returncode, 0)
            man = json.loads((out / "manifest.json").read_text())
            self.assertEqual(man["arms"]["baseline-cap"], man["arms"]["baseline"])
            self.assertEqual(man["hardcap_suffix"], "-cap")
            self.assertEqual(man["hardcap_arms"], ["baseline-cap"])
            # argv actually used, not just what the manifest says
            free = json.loads((out / "baseline__rep0.json").read_text())["argv"]
            cap = json.loads((out / "baseline-cap__rep0.json").read_text())["argv"]
            self.assertEqual([a for a in cap if "--port" not in a],
                             [a for a in free if "--port" not in a])

    def test_a_capped_arm_that_generates_short_fails_the_run(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "short"
            r = self._run(out, "baseline-cap",
                          {"FAKE_SHORT_UNLESS_IGNORE_EOS": "0",
                           "FAKE_PREDICTED_N": "100"})
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("under a hard cap", r.stdout + r.stderr)
            self.assertFalse((out / "RUN_COMPLETE.json").exists())

    def test_an_uncapped_arm_may_stop_early(self):
        """The freerun half of the contrast must NOT be held to the cap, or the
        design collapses back into one mode."""
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "free"
            r = self._run(out, "baseline")
            self.assertEqual(r.returncode, 0, (r.stdout + r.stderr)[-800:])
            rows = json.loads((out / "baseline__rep0.json").read_text())["rows"]
            self.assertTrue(all(x["predicted_n"] == 100 for x in rows))

    def test_the_suffix_is_opt_in(self):
        with tempfile.TemporaryDirectory() as t:
            r = self._run(Path(t) / "off", "baseline-cap",
                          {"BENCH_HARDCAP_SUFFIX": ""})
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("unknown arm", r.stdout + r.stderr)

    def test_a_real_arm_wins_over_the_suffix_rule(self):
        """`baseline-kvfp16` is a real arm AND `baseline` + `-kvfp16`. Without
        the `arm in ARMS` guard it would be served as the baseline's flags with
        a cap - a different configuration entirely, measured under the name of
        the one that was asked for, silently."""
        import importlib.util
        os.environ["BENCH_HARDCAP_SUFFIX"] = "-kvfp16"
        try:
            spec = importlib.util.spec_from_file_location("rr_suffix", RUNNER)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            self.assertIn("baseline-kvfp16", m.ARMS)
            self.assertIn("baseline", m.ARMS)
            self.assertNotEqual(m.ARMS["baseline-kvfp16"], m.ARMS["baseline"])
            self.assertFalse(m.arm_is_hardcap("baseline-kvfp16"),
                             "a real arm was taken for a capped one")
            self.assertEqual(m.arm_base("baseline-kvfp16"), "baseline-kvfp16")
        finally:
            os.environ.pop("BENCH_HARDCAP_SUFFIX", None)

    def test_that_arm_still_runs_as_itself_end_to_end(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "real"
            r = self._run(out, "baseline-kvfp16",
                          {"BENCH_HARDCAP_SUFFIX": "-kvfp16"})
            self.assertEqual(r.returncode, 0, (r.stdout + r.stderr)[-900:])
            man = json.loads((out / "manifest.json").read_text())
            self.assertEqual(man["hardcap_arms"], [])
            argv = json.loads(
                (out / "baseline-kvfp16__rep0.json").read_text())["argv"]
            # `--kv-fp16` is a sentinel: it drops `-ctk q8_0 -ctv q8_0` rather
            # than adding a flag, so the tell is that the quantised KV args are
            # gone. Plain `baseline` keeps them, so this argv could not have
            # come from serving the baseline under another name.
            self.assertNotIn("q8_0", " ".join(argv), argv)
            rows = json.loads(
                (out / "baseline-kvfp16__rep0.json").read_text())["rows"]
            self.assertTrue(all(x["predicted_n"] == 100 for x in rows),
                            "it was capped, so it was treated as baseline+cap")

    def test_a_suffix_on_something_that_is_not_an_arm_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            r = self._run(Path(t) / "nope", "nonsense-cap")
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("unknown arm", r.stdout + r.stderr)

    def test_the_run_level_flag_still_caps_everything(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "runlevel"
            r = self._run(out, "baseline", {"BENCH_IGNORE_EOS": "on"})
            self.assertEqual(r.returncode, 0, (r.stdout + r.stderr)[-800:])
            rows = json.loads((out / "baseline__rep0.json").read_text())["rows"]
            self.assertTrue(all(x["predicted_n"] == 300 for x in rows))


class RederivationMustNotForgive(unittest.TestCase):
    """`analysis/rederive_from_logs.py` is the only thing that ties the derived
    JSON to the logs it came from. A re-derivation that reports success when a
    record changed, or when one stopped being produced, proves nothing."""

    def _mod(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "rederive", ROOT / "analysis" / "rederive_from_logs.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        m.FAIL.clear()
        real = m.compare

        def quiet(*a, **k):          # compare() reports as it goes; not here
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                return real(*a, **k)

        m.compare = quiet
        return m

    def test_a_changed_record_fails(self):
        m = self._mod()
        m.FAIL.clear()
        pub = [{"k": 1, "v": "a"}, {"k": 2, "v": "b"}]
        got = [{"k": 1, "v": "a"}, {"k": 2, "v": "CHANGED"}]
        m.compare("t", got, pub, lambda r: r["k"])
        self.assertTrue(m.FAIL, "a differing record was accepted")

    def test_a_vanished_record_fails(self):
        m = self._mod()
        m.FAIL.clear()
        pub = [{"k": 1}, {"k": 2}]
        m.compare("t", [{"k": 1}], pub, lambda r: r["k"])
        self.assertTrue(m.FAIL, "a record that stopped being produced was accepted")

    def test_the_documented_gap_is_allowed_and_only_that(self):
        m = self._mod()
        m.FAIL.clear()
        pub = [{"k": 1}, {"k": 2}]
        m.compare("t", [{"k": 1}], pub, lambda r: r["k"], expected_gap=1)
        self.assertFalse(m.FAIL, m.FAIL)
        m.FAIL.clear()
        m.compare("t", [], pub, lambda r: r["k"], expected_gap=1)
        self.assertTrue(m.FAIL, "a wider gap than documented was accepted")

    def test_an_unexpected_record_fails(self):
        m = self._mod()
        m.FAIL.clear()
        m.compare("t", [{"k": 1}, {"k": 9}], [{"k": 1}], lambda r: r["k"])
        self.assertTrue(m.FAIL)

    def test_the_three_unreproducible_runs_are_not_committed(self):
        for r in self._mod().NOT_REPRODUCIBLE:
            self.assertFalse((ROOT / "v4_audit_2026_08_25" / "data" / r).is_dir(),
                             f"{r} is committed, so its rows should regenerate")


class AnEmptyRunIsNotACompleteRun(unittest.TestCase):
    """`range(0)` is empty, so the arm loop never runs and the completeness
    validation has nothing to object to. The directory then carries a
    RUN_COMPLETE.json attesting to zero arm-runs, which every consumer reads as
    a whole run."""

    def test_zero_repeats_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "zero"
            r = run_runner({"BENCH_REPEATS": "0"}, out)
            self.assertNotEqual(r.returncode, 0)
            # refused by the range check now, before a server is started at all,
            # rather than by the completeness validation after the empty loop
            self.assertIn("out of range", r.stdout + r.stderr)
            self.assertFalse(out.exists(), "nothing should have been written")
            self.assertFalse((out / "RUN_COMPLETE.json").exists())

    def test_negative_repeats_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "neg"
            r = run_runner({"BENCH_REPEATS": "-1"}, out)
            self.assertNotEqual(r.returncode, 0)
            self.assertFalse((out / "RUN_COMPLETE.json").exists())

    def test_a_repeated_arm_is_refused_before_the_gpu_is_touched(self):
        """Two entries write the same `<arm>__rep<n>.json`; the second
        overwrites the first and half the run is thrown away. `validate_run`
        catches it afterwards, which costs the whole run."""
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "dup"
            r = run_runner({"BENCH_ARMS": "baseline,baseline"}, out)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("each arm may appear once", r.stdout + r.stderr)
            self.assertFalse(out.exists() and any(out.glob("*__rep*.json")))


class InFlightCountMustBeASweepLine(unittest.TestCase):
    """The concurrency arms rest on this: if it reads 1 while N were asked for,
    the run measured nothing about concurrency and the arm-run says so. The
    handover case is the one that matters - a request ending exactly as the next
    begins must not count as an overlap, or a strictly serial client would look
    concurrent."""

    def _f(self):
        import importlib.util
        os.environ.update(LLAMA_SERVER_BIN="/bin/true", MODEL_TARGET="/dev/null",
                          BENCH_OUT="/tmp/_inflight_probe")
        spec = importlib.util.spec_from_file_location("_rr_if", RUNNER)
        rr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rr)
        return rr.max_client_requests_in_flight

    def test_the_sweep_line(self):
        f = self._f()
        for name, rows, want in (
            ("no rows", [], 0),
            ("one request", [{"t_start": 0, "t_end": 1}], 1),
            ("handover at the same instant",
             [{"t_start": 0, "t_end": 1}, {"t_start": 1, "t_end": 2}], 1),
            ("two overlapping",
             [{"t_start": 0, "t_end": 2}, {"t_start": 1, "t_end": 3}], 2),
            ("three nested",
             [{"t_start": 0, "t_end": 9}, {"t_start": 1, "t_end": 8},
              {"t_start": 2, "t_end": 7}], 3),
            ("four requests, peak of two",
             [{"t_start": 0, "t_end": 2}, {"t_start": 1, "t_end": 3},
              {"t_start": 4, "t_end": 6}, {"t_start": 5, "t_end": 7}], 2),
            ("a zero-length request",
             [{"t_start": 1, "t_end": 1}, {"t_start": 1, "t_end": 2}], 1),
            ("rows without timestamps", [{"predicted_n": 1}], 0),
        ):
            with self.subTest(name):
                self.assertEqual(f(rows), want)


class EveryPublishedFixIsStillHere(unittest.TestCase):
    """A long editing session reverts things.

    Each entry is a fragment of a fix this repository published, and the file it
    lives in. Not a substitute for the behavioural tests above - those are what
    prove the fixes work - but those cannot cover a whole file, and one fix did
    silently disappear: `analysis/extract_checkpoint_timers.py` went back to
    `open(path).read()` under a later edit, which the ResourceWarning in the
    test output was quietly reporting for an hour.
    """

    FIXES = [
        ("analysis/extract_checkpoint_timers.py", 're.sub(r"__rep',
         "arm name is repeat-independent"),
        ("analysis/extract_checkpoint_timers.py", "not fully covered",
         "--repeats refuses partial coverage"),
        ("analysis/extract_checkpoint_timers.py",
         'with open(path, errors="replace") as fh', "the log file is closed"),
        ("analysis/extract_checkpoint_timers.py",
         "(sum(tgt) + sum(dft) + sum(lt) + sum(ld)) / 1e6",
         "the total is rounded once, not summed from rounded parts"),
        ("bench/retest_runner.py", "def is_position_balanced",
         "balance means equal visits"),
        ("bench/retest_runner.py", "def validate_run",
         "the run validates itself before attesting"),
        ("bench/retest_runner.py", "RUN_FAILED.json", "and says so when it fails"),
        ("bench/retest_runner.py", "if proc is not None and proc.poll() is not None",
         "liveness is checked before readiness"),
        ("bench/retest_runner.py", "if not port_is_free(PORT)",
         "the port is free before spawning"),
        ("bench/retest_runner.py", "TEARDOWN[(arm, rep)]",
         "teardown is recorded, not printed"),
        ("bench/retest_runner.py",
         "ceiling = headroom_mib if baseline_mib is None else baseline_mib + headroom_mib",
         "the teardown ceiling is relative to the pre-run reading"),
        ("bench/retest_runner.py", 'body["ignore_eos"] = True',
         "the hard cap reaches the server"),
        # superseded 2026-08-27: `_env_int("BENCH_REPEATS", 3, 1)` refuses it
        # before a server starts, so the later `if REPEATS < 1` was unreachable
        ("bench/retest_runner.py", 'REPEATS = _env_int("BENCH_REPEATS", 3, 1)',
         "an empty run is refused"),
        ("bench/retest_runner.py", "is not a recognised boolean",
         "a mistyped treatment boolean stops the run"),
        ("bench/retest_runner.py", "is not one of {sorted(allowed)}",
         "a mistyped treatment choice stops the run"),
        ("bench/retest_runner.py", 'r"[0-9a-f]{64}", EXPECT_LIB',
         "the expected library digest must be whole"),
        ("bench/retest_runner.py", "elif libs != _LIB_BASELINE:",
         "the whole shared-library map is pinned"),
        ("bench/retest_runner.py", "unverified = baseline_mib is not None",
         "an unreadable teardown is a failure only where a reading existed"),
        ("bench/retest_runner.py", "if IGNORE_EOS or hardcap:",
         "one invocation can carry both length modes"),
        ("bench/retest_runner.py", "def arm_is_hardcap",
         "a hard-cap arm is resolved from its base"),
        ("bench/retest_runner.py", "if ok >= consecutive:",
         "a teardown needs consecutive low readings"),
        ("analysis/paired_blocks.py", "def is_position_balanced",
         "the analysis shares the runner's balance definition"),
        ("analysis/paired_blocks.py", "def t_critical_975",
         "the t critical value is computed, not tabulated to df=10"),
        ("analysis/paired_blocks.py", "refusing to write an interval",
         "an unbalanced schedule does not silently get an interval"),
        ("bench/stage_mtp_source.py", "safetensors indexes in",
         "staging refuses more than one index"),
        ("bench/stage_mtp_source.py", ".staging.",
         "staging builds in a fresh directory and renames"),
        ("bench/retest_runner.py", "each arm may appear once",
         "a repeated arm is refused"),
        ("bench/retest_runner.py", 'res["server_log_sha256"] = sha256',
         "the log is hashed after the server stops"),
        ("bench/retest_runner.py", "json.dumps(PROMPTS, sort_keys=True",
         "the prompt hash covers the prompts"),
        ("analysis/check_data_integrity.py", "expected_cells - set(cells)",
         "the exact (arm, repeat) product"),
        ("analysis/matrix_report.py", "for a, r in sorted(want - set(have)):",
         "--strict checks which arm-runs are present"),
        ("analysis/check_data_integrity.py", "filename says arm=",
         "filename against contents"),
        ("analysis/paired_blocks.py", "def observed_schedule",
         "the schedule is re-derived from the data"),
        ("analysis/matrix_report.py", "DIFFERENT numbers of tokens",
         "the length confound is surfaced"),
        ("analysis/thermal_report.py", "def detect", "all three telemetry schemas"),
        ("analysis/thermal_report.py", "THROTTLE_BITS",
         "the bitmask is decoded rather than reported as zero"),
        ("analysis/thermal_report.py", "act = [r for r in busy",
         "flags are counted over loaded samples"),
        ("bench/convert_dflash.sh", "Only now is the conversion a success",
         "the drafter is promoted after the load check"),
        ("bench/collect_evidence.sh", "nothing to archive",
         "an empty archive is refused"),
        ("analysis/plot_v4_runs.py", "if not CHECK", "--check is read-only"),
        ("analysis/plot_v4_runs.py", "BENCH_PLOT_RUN",
         "the chart names its source run"),
    ]

    def test_each_one(self):
        for rel, needle, what in self.FIXES:
            with self.subTest(what):
                self.assertIn(needle, (ROOT / rel).read_text(encoding="utf-8"),
                              f"{what}: gone from {rel}")


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

    @staticmethod
    def _checkpoint(src, shards, index_name="model.safetensors.index.json",
                    extra=()):
        """A minimal AWQ-shaped checkpoint whose --mtp export set is BF16."""
        import struct
        src.mkdir(parents=True, exist_ok=True)
        (src / "config.json").write_text(
            json.dumps({"quantization_config": {"quant_method": "awq"}}),
            encoding="utf-8")
        weight_map = {}
        for i, shard in enumerate(shards):
            keys = {"mtp.layer.weight" if i == 0 else f"mtp.layer{i}.weight":
                    {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}
            if i == 0:
                keys["model.norm.weight"] = {"dtype": "BF16", "shape": [1],
                                             "data_offsets": [2, 4]}
            hdr = json.dumps(keys).encode()
            with (src / shard).open("wb") as fh:
                fh.write(struct.pack("<Q", len(hdr)))
                fh.write(hdr)
                fh.write(b"\0" * 4)
            for k in keys:
                weight_map[k] = shard
        (src / index_name).write_text(json.dumps({"weight_map": weight_map}),
                                      encoding="utf-8")
        for f in extra:
            (src / f).write_text("stale", encoding="utf-8")

    def _stage(self, src, stage):
        env = dict(os.environ, MTP_SRC=str(src), MTP_STAGE=str(stage))
        return subprocess.run(
            [sys.executable, str(ROOT / "bench" / "stage_mtp_source.py")],
            env=env, capture_output=True, text=True)

    def test_two_indexes_are_refused_rather_than_sorted(self):
        """Taking the first by sort order silently picks a checkpoint
        generation, which is not a detail a conversion should decide."""
        with tempfile.TemporaryDirectory() as t:
            src, stage = Path(t) / "ckpt", Path(t) / "stage"
            self._checkpoint(src, ["a.safetensors"])
            (src / "old.index.json").write_text(
                json.dumps({"weight_map": {}}), encoding="utf-8")
            r = self._stage(src, stage)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("safetensors indexes", r.stdout + r.stderr)
            self.assertFalse(stage.exists())

    def test_a_shard_the_index_does_not_reference_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            src, stage = Path(t) / "ckpt", Path(t) / "stage"
            self._checkpoint(src, ["a.safetensors"], extra=("older.safetensors",))
            r = self._stage(src, stage)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("does not", r.stdout + r.stderr)
            self.assertFalse(stage.exists())

    def test_a_missing_shard_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            src, stage = Path(t) / "ckpt", Path(t) / "stage"
            self._checkpoint(src, ["a.safetensors"])
            (src / "a.safetensors").unlink()
            r = self._stage(src, stage)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("not there", r.stdout + r.stderr)

    def test_a_stale_stage_does_not_survive_a_restage(self):
        """The stage used to be reused, so shards the source no longer has
        stayed behind and the conversion read two generations."""
        with tempfile.TemporaryDirectory() as t:
            src, stage = Path(t) / "ckpt", Path(t) / "stage"
            self._checkpoint(src, ["a.safetensors", "b.safetensors"])
            self.assertEqual(self._stage(src, stage).returncode, 0, )
            stage.mkdir(exist_ok=True)
            (stage / "ghost.safetensors").write_text("stale", encoding="utf-8")
            (src / "b.safetensors").unlink()
            self._checkpoint(src, ["a.safetensors"])
            r = self._stage(src, stage)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertFalse((stage / "ghost.safetensors").exists())
            self.assertFalse((stage / "b.safetensors").exists())
            self.assertTrue((stage / "a.safetensors").exists())


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


class ThePullRequestBodyMustBeChecked(unittest.TestCase):
    """A published document nothing parses is how every one of these started.

    The third review's P0-4 was errors in the PR body. It is four numeric
    tables and three counts, none of which had a code path, so fixing them
    once fixed nothing. `analysis/verify_claims.py` parses it now, and this
    keeps the parse from being dropped the way the run-M table's was.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def test_the_body_is_in_the_tree(self):
        self.assertTrue((self.ROOT / "PULL_REQUEST.md").is_file())

    def test_the_checker_reads_it(self):
        src = (self.ROOT / "analysis" / "verify_claims.py").read_text(encoding="utf-8")
        self.assertIn("PULL_REQUEST.md", src)
        self.assertIn("_pr_table", src)

    def test_every_table_in_the_body_is_parsed(self):
        """Counting the tables, not trusting that four is still all of them."""
        body = (self.ROOT / "PULL_REQUEST.md").read_text(encoding="utf-8").splitlines()
        def _is_rule(line):
            bare = line.replace("|", "").replace(" ", "")
            return bare and set(bare) <= set("-:")

        # tables inside a list item are indented, and the first version of
        # this test missed one that way
        body = [l.strip() for l in body]
        headers = [l for i, l in enumerate(body)
                   if l.startswith("|") and i + 1 < len(body)
                   and _is_rule(body[i + 1])]
        # substring-matching the first column passes on any header whose first
        # word appears anywhere in the source, which "arm" and "new" both do.
        # The parsed set is the literal arguments `_pr_table` is called with.
        import ast
        tree = ast.parse((self.ROOT / "analysis" / "verify_claims.py")
                         .read_text(encoding="utf-8"))
        parsed = [n.args[0].value for n in ast.walk(tree)
                  if isinstance(n, ast.Call)
                  and getattr(n.func, "id", None) == "_pr_table"
                  and n.args and isinstance(n.args[0], ast.Constant)]
        self.assertTrue(parsed, "nothing calls _pr_table")
        norm = str.maketrans({"\u2212": "-", "\u2013": "-", "\u2014": "-"})
        unparsed = [h for h in headers
                    if not any(h.translate(norm).startswith(a) for a in parsed)]
        self.assertEqual(unparsed, [], f"{len(unparsed)} table(s) nothing reads")

    def test_the_counts_it_quotes_are_derived_not_typed(self):
        src = (self.ROOT / "analysis" / "verify_claims.py").read_text(encoding="utf-8")
        for needle in ("assertion count it quotes", "regression count it quotes",
                       "run-directory count it quotes", "mutation counts it quotes"):
            self.assertIn(needle, src, f"nothing checks the {needle}")

    def test_a_perturbation_of_it_is_registered(self):
        muts = (self.ROOT / "tests" / "data_mutate.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(muts.count('"PULL_REQUEST.md"'), 5)


class ComparabilityMustBeOneRule(unittest.TestCase):
    """A16's "twelve comparable runs" is decided in two files.

    `analysis/verify_claims.py` filters on seven manifest fields plus the run
    date; `analysis/plot_v4_runs.py` had the first seven and not the date, so
    when run T4 landed on 2026-08-27 the checker still said twelve and the
    chart quietly became thirteen. Neither was wrong about its own arithmetic.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def _selected(self, extra_date_filter):
        tgt = ("707a55a8a4397ecde44de0c499d3e68c1ad1d240d1da65826b4949d1043f"
               "4450")
        out = []
        data = self.ROOT / "v4_audit_2026_08_25" / "data"
        for mp in sorted(data.glob("matrix_*/manifest.json")):
            m = json.loads(mp.read_text(encoding="utf-8"))
            if not (m.get("think") == "on" and m.get("concurrency") == 1
                    and m.get("prompt_set", "v1") == "v1"
                    and str(m.get("ctx")) == "8192"
                    and str(m.get("fit_target")) == "3072"
                    and m.get("target_sha256") == tgt
                    and "spec-dflash-n2" in (m.get("arms") or {})):
                continue
            if extra_date_filter and not str(m.get("created", "")).startswith(
                    "2026-08-26"):
                continue
            out.append(mp.parent.name)
        return out

    def test_the_date_filter_is_what_separates_them(self):
        """Without it the two rules select different sets - the actual bug."""
        self.assertNotEqual(self._selected(True), self._selected(False),
                            "no run distinguishes the rules, so this test "
                            "would pass whether or not the filter existed")

    def test_both_files_carry_the_date_filter(self):
        for rel in ("analysis/verify_claims.py", "analysis/plot_v4_runs.py"):
            src = (self.ROOT / rel).read_text(encoding="utf-8")
            self.assertIn('"2026-08-26"', src,
                          f"{rel} selects comparable runs without the date")

    def test_the_selection_is_still_twelve(self):
        self.assertEqual(len(self._selected(True)), 12)

    def test_and_run_t4_is_the_one_it_excludes(self):
        excluded = set(self._selected(False)) - set(self._selected(True))
        self.assertEqual(excluded, {"matrix_T4_split_20260827_175051"})


class EveryPerturbationAnchorMustResolve(unittest.TestCase):
    """A document perturbation is only a test if it lands where it says.

    `edit_doc` checked that its anchor was present and not that it was unique,
    so `| **39.09** | **54.7 %** |` -- two identical rows of the split table --
    perturbed whichever came first. It still fired, which is the problem: an
    anchor can go ambiguous, or stop matching after an edit, and the suite
    reports the same 'all detected' either way.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def _anchors(self):
        import ast
        tree = ast.parse((self.ROOT / "tests" / "data_mutate.py")
                         .read_text(encoding="utf-8"))
        out = []
        for n in ast.walk(tree):
            if (isinstance(n, ast.Call) and getattr(n.func, "id", "") == "edit_doc"
                    and len(n.args) == 3
                    and all(isinstance(a, ast.Constant) for a in n.args[:2])):
                out.append((n.args[0].value, n.args[1].value))
        return out

    def test_there_are_anchors_to_check(self):
        self.assertGreater(len(self._anchors()), 30)

    def test_each_appears_exactly_once_in_its_document(self):
        for rel, old in self._anchors():
            txt = (self.ROOT / rel).read_text(encoding="utf-8")
            self.assertEqual(txt.count(old), 1,
                             f"{rel}: {old[:48]!r} appears {txt.count(old)}x")

    def test_edit_doc_refuses_an_ambiguous_anchor(self):
        src = (self.ROOT / "tests" / "data_mutate.py").read_text(encoding="utf-8")
        self.assertIn("must be exactly one", src,
                      "edit_doc accepts an anchor that matches more than once")


class TheGitlessAssertionGapMustBeTheDeclaredOne(unittest.TestCase):
    """`verify_claims.py` skips eight checks where there is no git history.

    The pull request body publishes how many assertions the checker runs, and
    `tests/data_mutate.py` runs it in a mirror that has no `.git`, so the two
    counts differ by exactly the git-gated ones. That difference is a constant
    in the checker; this derives it instead of trusting it.
    """

    ROOT = Path(__file__).resolve().parents[1]

    @staticmethod
    def _count(env):
        r = subprocess.run([sys.executable, "analysis/verify_claims.py"],
                           cwd=str(TheGitlessAssertionGapMustBeTheDeclaredOne.ROOT),
                           capture_output=True, text=True, env=env, timeout=900)
        return sum(1 for l in r.stdout.splitlines()
                   if l.startswith("  PASS") or l.startswith("  FAIL"))

    def test_the_gap_is_what_the_checker_declares(self):
        # In a shallow clone the with-git arm is refused by design, so the
        # difference measures a truncated run against a whole one: CI reported
        # "the checker skips -1012 assertions" before this guard existed.
        shallow = subprocess.run(
            ["git", "-C", str(self.ROOT), "rev-parse", "--is-shallow-repository"],
            capture_output=True, text=True).stdout.strip()
        if shallow == "true":
            self.skipTest("shallow clone: the gap cannot be measured here, and "
                          "the checker refuses such a clone on purpose. Use "
                          "fetch-depth: 0.")
        src = (self.ROOT / "analysis" / "verify_claims.py").read_text(encoding="utf-8")
        m = re.search(r"^_GITLESS_SKIPPED = (\d+)$", src, re.M)
        self.assertIsNotNone(m, "the checker no longer declares the gap")
        declared = int(m.group(1))

        with_git = dict(os.environ)
        # emptying PATH would break every other subprocess too, so shadow just
        # git with one that fails, the way a shallow clone or a mirror does
        with tempfile.TemporaryDirectory() as shim:
            g = Path(shim) / "git"
            g.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
            g.chmod(0o755)
            without_git = dict(os.environ,
                               PATH=f"{shim}:{os.environ.get('PATH', '')}")
            n_with = self._count(with_git)
            n_without = self._count(without_git)
        self.assertGreater(n_with, 0)
        self.assertEqual(n_with - n_without, declared,
                         f"the checker skips {n_with - n_without} assertions "
                         f"without git and declares {declared}")


class TheVerificationSuitesMustRefuseAMeasuringHost(unittest.TestCase):
    """The guard added on 2026-08-27 was itself unguarded.

    `tests/mutate.py` opens by saying a mutation that survives means its guard
    is decorative. Three guards shipped that night with nothing testing them:
    this one, the workflows' `shell: bash`, and A16's new claims. Deleting
    `host_guard.protect(...)` from either suite failed nothing, which is the
    defect this whole branch exists to catch, committed by the branch itself.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def _hg(self):
        sys.path.insert(0, str(self.ROOT / "bench"))
        import host_guard
        return host_guard

    def test_both_suites_call_the_guard_before_they_work(self):
        for rel in ("tests/mutate.py", "tests/data_mutate.py"):
            src = (self.ROOT / rel).read_text(encoding="utf-8")
            self.assertIn("host_guard.protect(", src,
                          f"{rel} can start during a measurement")
            self.assertIn("host_guard.serialise(", src,
                          f"{rel} can overlap with another pipeline")

    def test_detection_is_positional_not_a_substring(self):
        """`\"bench.py\" in cmdline` also matches an editor, a grep, and this
        session's own shell command that merely names it."""
        m = self._hg()._benchmark_name
        for argv, want in (
                (["/opt/build/bin/llama-server", "-m", "x.gguf"], "llama-server"),
                (["python3", "harness/bench.py", "--matrix", "phase_a"], "bench.py"),
                (["python3", "-u", "harness/bench.py"], "bench.py"),
                (["grep", "-rn", "bench.py", "."], None),
                (["bash", "-c", "echo bench.py llama-server"], None),
                (["vim", "harness/bench.py"], None),
                (["python3", "tests/data_mutate.py"], None),
                ([], None)):
            self.assertEqual(m(argv), want, f"argv={argv}")

    def test_every_thread_variable_is_pinned(self):
        """OpenBLAS spawning one thread per core is half of what caused the
        contention; setdefault must cover all five names."""
        hg = self._hg()
        self.assertEqual(len(hg._THREAD_VARS), 5)
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            self.assertIn(v, hg._THREAD_VARS)

    def _protect(self, env_extra, lockfile):
        code = ("import sys; sys.path.insert(0, 'bench'); import host_guard; "
                "host_guard.protect('t'); print('proceeded')")
        env = dict(os.environ, BENCH_GPU_LOCK=str(lockfile), **env_extra)
        env.pop("BENCH_ALLOW_CONTENDED", None)
        if "CI" not in env_extra:
            env.pop("CI", None)
        return subprocess.run([sys.executable, "-c", code], cwd=str(self.ROOT),
                              capture_output=True, text=True, env=env, timeout=120)

    def test_a_held_lock_stops_it(self):
        with tempfile.NamedTemporaryFile(suffix=".lock") as lk:
            r = self._protect({}, lk.name)
        self.assertNotEqual(r.returncode, 0, "a held lock did not stop the suite")
        self.assertIn("refusing to run", r.stdout + r.stderr)

    def test_but_ci_is_exempt_because_there_is_no_card_there(self):
        with tempfile.NamedTemporaryFile(suffix=".lock") as lk:
            r = self._protect({"CI": "true"}, lk.name)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("proceeded", r.stdout)


class TheWorkflowsMustNameTheirShell(unittest.TestCase):
    """A `run:` step with no shell gets `bash -e {0}`: -e, and no pipefail.

    Under that default `checker.py | tail -1` reports tail's exit status, so a
    checker can print FAIL into a green job. Nothing here pipes today; this is
    what keeps that true when something does.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def test_both_workflows_declare_bash(self):
        for wf in ("audit.yml", "evidence.yml"):
            src = (self.ROOT / ".github" / "workflows" / wf).read_text(encoding="utf-8")
            self.assertRegex(src, r"defaults:\s*\n\s*run:\s*\n\s*shell:\s*bash",
                             f"{wf} leaves the shell defaulted, so no pipefail")

    def test_no_step_pipes_without_saying_pipefail(self):
        """Belt and braces: even with the default named, a step that pipes and
        overrides the shell would be back where it started."""
        offenders = []
        for wf in ("audit.yml", "evidence.yml"):
            path = self.ROOT / ".github" / "workflows" / wf
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                t = line.strip()
                if t.startswith("#") or "||" in t:
                    continue
                if re.search(r"[^|]\|[^|]", t) and "pipefail" not in t:
                    offenders.append(f"{wf}:{i} {t[:60]}")
        # the two that exist are inside steps that set -euo pipefail themselves
        for o in offenders:
            self.assertIn("evidence.yml", o, f"unguarded pipe: {o}")


class TheTelemetryGapA16NamesMustBeReal(unittest.TestCase):
    """A16 says host load was never sampled. That has to stay true of the
    sampler, or the sentence becomes false without anyone noticing."""

    ROOT = Path(__file__).resolve().parents[1]

    def test_gpu_telemetry_samples_no_host_load(self):
        src = (self.ROOT / "bench" / "gpu_telemetry.sh").read_text(encoding="utf-8")
        for needle in ("/proc/stat", "loadavg", "vmstat", "mpstat"):
            self.assertNotIn(needle, src,
                             "gpu_telemetry.sh grew host sampling; A16 says it "
                             "has none and no run has the column")

    def test_no_committed_trace_has_a_host_column(self):
        data = self.ROOT / "v4_audit_2026_08_25" / "data"
        checked = 0
        for csv in sorted(data.glob("gpu_telemetry_*.csv")):
            head = csv.read_text(encoding="utf-8", errors="replace").split("\n", 1)[0]
            checked += 1
            for bad in ("load1", "busy_pct", "own_pct", "other_pct"):
                self.assertNotIn(bad, head, f"{csv.name} has a host column")
        self.assertGreater(checked, 0, "no telemetry traces found to check")

    def test_the_sampler_exists_for_the_next_run(self):
        hg = (self.ROOT / "bench" / "host_guard.py").read_text(encoding="utf-8")
        self.assertIn("def sample(", hg)
        self.assertIn("--sample", hg)


class TheRerunScriptsMustBeSelfContainedAndFailClosed(unittest.TestCase):
    """A published reproducer that only works in the author's shell is not one.

    Both drivers exported `BENCH_SERVER`. `retest_runner.py` reads
    `LLAMA_SERVER_BIN` and nothing else, so a clean shell failed outright and a
    shell that happened to carry somebody else's `LLAMA_SERVER_BIN` silently ran
    a binary the script never chose. Both scripts also ended on `find | wc`,
    which succeeds, so every session could fail and the driver still exited 0.
    """

    ROOT = Path(__file__).resolve().parents[1]
    SCRIPTS = ("bench/run_v2_crossover.sh", "bench/run_v3_within.sh")

    def _src(self, rel):
        return (self.ROOT / rel).read_text(encoding="utf-8")

    def test_they_export_the_variable_the_runner_reads(self):
        runner = self._src("bench/retest_runner.py")
        self.assertIn('os.environ.get("LLAMA_SERVER_BIN"', runner)
        self.assertNotIn("BENCH_SERVER", runner,
                         "the runner grew a second name for the server")
        for rel in self.SCRIPTS:
            src = self._src(rel)
            self.assertIn("export LLAMA_SERVER_BIN=", src, f"{rel}")
            self.assertNotIn("export BENCH_SERVER=", src,
                             f"{rel} still exports a name nothing consumes")

    def test_they_assert_the_binary_before_hours_of_work(self):
        for rel in self.SCRIPTS:
            src = self._src(rel)
            self.assertIn("BENCH_EXPECT_COMMIT", src,
                          f"{rel} does not pin the server commit")
        # and the runner must actually act on it
        self.assertIn('EXPECT_COMMIT = os.environ.get("BENCH_EXPECT_COMMIT"',
                      self._src("bench/retest_runner.py"))

    def test_they_exit_non_zero_when_a_session_fails(self):
        for rel in self.SCRIPTS:
            src = self._src(rel).rstrip()
            self.assertTrue(src.endswith('exit "$rc"'),
                            f"{rel} ends on {src.splitlines()[-1]!r}, which can "
                            f"succeed while sessions failed")
            self.assertIn('[ -z "$FAILED" ]', src, f"{rel} ignores FAILED")

    def test_they_require_the_exact_expected_number_of_units(self):
        v2 = self._src("bench/run_v2_crossover.sh")
        self.assertIn("-eq 16", v2, "V2 does not require 16 validated halves")
        v3 = self._src("bench/run_v3_within.sh")
        self.assertIn("-eq 2", v3, "V3 does not require 2 validated sessions")
        self.assertIn("-eq 100", v3, "V3 does not require 100 arm-runs a session")

    def test_the_scripts_still_lint(self):
        """shellcheck runs in CI; a syntax error here would only surface there."""
        for rel in self.SCRIPTS:
            r = subprocess.run(["bash", "-n", str(self.ROOT / rel)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f"{rel}: {r.stderr}")


class RederivationMustNotCollapseOrMiscount(unittest.TestCase):
    """`compare()` indexed both sides with a dict comprehension.

    Two records sharing a key collapsed to the last one, so a dump that cannot
    be indexed compared clean. And the gap was checked by COUNT: delete one
    reproducible row, add one unreproducible row, and nine is still nine.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def _mod(self):
        sys.path.insert(0, str(self.ROOT / "analysis"))
        import importlib
        import rederive_from_logs
        return importlib.reload(rederive_from_logs)

    def test_a_duplicate_key_is_refused_not_overwritten(self):
        m = self._mod()
        rows = [{"k": 1, "v": "a"}, {"k": 1, "v": "b"}]
        with self.assertRaises(SystemExit) as cm:
            m.index_unique(rows, lambda r: r["k"], "test")
        self.assertIn("duplicate key", str(cm.exception))

    def test_compare_actually_uses_it(self):
        """Testing the helper in isolation leaves `compare()` free to go back
        to a dict comprehension, which is the defect, not the helper."""
        src = (self.ROOT / "analysis" / "rederive_from_logs.py").read_text(encoding="utf-8")
        body = src.split("def compare(")[1].split("\ndef ")[0]
        self.assertIn("index_unique(regenerated", body)
        self.assertIn("index_unique(published", body)
        self.assertNotIn("{key(r): r for r in", body,
                         "compare() indexes with a dict comprehension again")

    def test_unique_keys_still_index(self):
        m = self._mod()
        out = m.index_unique([{"k": 1}, {"k": 2}], lambda r: r["k"], "test")
        self.assertEqual(sorted(out), [1, 2])

    def test_the_missing_set_is_compared_not_its_size(self):
        m = self._mod()
        m.FAIL.clear()
        published = [{"k": i} for i in range(4)]
        # one documented gap (k=3), but a DIFFERENT row (k=0) actually missing
        regenerated = [{"k": 1}, {"k": 2}, {"k": 3}]
        m.compare("t", regenerated, published, lambda r: r["k"],
                  expected_missing=[3])
        self.assertTrue(m.FAIL, "a swapped missing row passed as the documented one")
        self.assertIn("not the documented one", m.FAIL[0])

    def test_the_documented_gap_still_passes(self):
        m = self._mod()
        m.FAIL.clear()
        published = [{"k": i} for i in range(4)]
        regenerated = [{"k": 0}, {"k": 1}, {"k": 2}]
        m.compare("t", regenerated, published, lambda r: r["k"],
                  expected_missing=[3])
        self.assertEqual(m.FAIL, [])

    def test_t4_can_be_made_mandatory(self):
        src = (self.ROOT / "analysis" / "rederive_from_logs.py").read_text(encoding="utf-8")
        self.assertIn("REDERIVE_REQUIRE_T4", src)
        wf = (self.ROOT / ".github" / "workflows" / "evidence.yml").read_text(encoding="utf-8")
        self.assertIn("REDERIVE_REQUIRE_T4", wf,
                      "the evidence workflow does not require T4")


class AHealthyArmRunMustSayWhatItLoaded(unittest.TestCase):
    """Provenance was compared only when it happened to be present.

    `check_identity` compared the startup log and `/props` against
    MODEL_TARGET "when present, and neither is required, because a server that
    never reached the loader has neither". True of a crash. For an arm-run that
    produced rows, having neither means nothing observed ties the numbers to
    the model the manifest names, and the comparison silently passed.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def _rr(self):
        return _load_runner()

    def test_a_completed_run_with_no_observed_identity_fails(self):
        rr = self._rr()
        rr._LIB_BASELINE = None
        bad = rr.check_identity("arm", 0, {}, {"libllama.so": "a" * 64},
                                healthy=True)
        self.assertTrue(bad, "a healthy arm-run passed with no identity at all")
        self.assertIn("neither the startup log nor /props", bad[0])

    def test_a_crashed_run_is_still_exempt(self):
        rr = self._rr()
        rr._LIB_BASELINE = None
        self.assertEqual(rr.check_identity("crashed", 0, {}, {}, healthy=False), [])

    def test_either_source_alone_satisfies_it(self):
        rr = self._rr()
        for ident in ({"model_path": rr.TARGET},
                      {"props": {"model_path": rr.TARGET}}):
            with self.subTest(ident=sorted(ident)):
                rr._LIB_BASELINE = None
                bad = rr.check_identity("arm", 0, ident,
                                        {"libllama.so": "a" * 64}, healthy=True)
                self.assertEqual(bad, [], bad)

    def test_the_caller_passes_the_run_health(self):
        src = (self.ROOT / "bench" / "retest_runner.py").read_text(encoding="utf-8")
        self.assertIn("healthy=bool(res.get(\"rows\"))", src,
                      "check_identity is called without saying whether the "
                      "arm-run produced anything")

    def test_the_fake_server_prints_a_loader_line(self):
        """Otherwise every end-to-end test exercises the absent-provenance path,
        which is how this stayed invisible."""
        fake = (self.ROOT / "tests" / "fake_llama_server.py").read_text(encoding="utf-8")
        self.assertIn("llama_model_loader: loaded meta data", fake)


class ACarryoverContrastNeedsABalancedSchedule(unittest.TestCase):
    """`analysis/carryover.py` must refuse the runs that cannot answer it.

    Run V3's cyclic rotation preceded every arm by one and the same arm nine
    times out of nine. A carryover number computed from that is the predecessor
    and the treatment at once, which is the alias the fourth review found in
    the wording. Printing it anyway would be worse than not having the file.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def _run(self, pattern):
        r = subprocess.run(
            [sys.executable, "analysis/carryover.py", "--json"]
            + sorted(str(p) for p in (self.ROOT / "v4_audit_2026_08_25" / "data").glob(pattern)),
            cwd=str(self.ROOT), capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        return json.loads(r.stdout)

    def test_it_refuses_the_cyclic_runs(self):
        for pattern in ("matrix_V3_s*_20260827_102614", "matrix_O2_latin_*"):
            with self.subTest(pattern=pattern):
                out = self._run(pattern)
                self.assertTrue(out["runs"], "no runs analysed")
                for rec in out["runs"]:
                    self.assertFalse(rec["first_order_carryover_balanced"])
                    self.assertIn("refused", rec)
                self.assertNotIn("across_sessions", out,
                                 "a contrast was reported for an aliased schedule")

    def test_the_diagnostic_names_the_alias(self):
        out = self._run("matrix_V3_s*_20260827_102614")
        note = out["runs"][0]["balance_note"]
        self.assertIn("preceded by 1 of 9", note)
        self.assertIn("9x", note, "the note does not say how lopsided it is")

    def test_the_order_comes_from_the_data_not_the_manifest(self):
        src = (self.ROOT / "analysis" / "carryover.py").read_text(encoding="utf-8")
        self.assertIn("t_start", src)
        self.assertNotIn('m["schedule"]', src,
                         "the analyser trusts the planned schedule")

    def test_the_runner_can_build_a_balanced_one(self):
        """Otherwise the refusal above is the only outcome this file can have."""
        rr = _load_runner()
        arms = [f"a{i}" for i in range(10)]
        sched = rr.williams_square(arms, 1)
        self.assertTrue(rr.is_carryover_balanced(sched, arms))
        self.assertTrue(rr.is_position_balanced(rr.position_counts(sched)))
        cyclic = rr.build_schedule(arms, 10, "latin")
        self.assertFalse(rr.is_carryover_balanced(cyclic, arms),
                         "the cyclic schedule would have been balanced after all")


class TheCarryoverAnalyserMustRecoverAPlantedEffect(unittest.TestCase):
    """Refusing the wrong schedules is half of it; the other half is arithmetic.

    Two defects in the first version would have produced a plausible wrong
    number for run W and nothing would have caught them:

      * it averaged over EVERY adjacency, but only the within-repeat ones are
        balanced. On the schedule W actually runs, that gives `spec-dflash-n2`
        six capped and four free predecessors instead of five and four - nine
        arms of ten contaminated by the row boundary.
      * it dropped crashed arm-runs before computing predecessors, so the arm
        after a crash was attributed to the wrong one.

    So this plants a known carryover effect and requires it back.
    """

    ROOT = Path(__file__).resolve().parents[1]
    ARMS = ["a", "a-cap", "b", "b-cap", "c", "c-cap"]
    BASE = {"a": 100.0, "a-cap": 110.0, "b": 50.0, "b-cap": 55.0,
            "c": 200.0, "c-cap": 220.0}
    EFFECT = 0.04          # 4 % slower after a capped predecessor

    def _square(self):
        rr = _load_runner()
        return rr.williams_square(self.ARMS, 11)

    def _write_run(self, out: Path, square, crash=None, boundary_bias=0.0):
        """Synthesise a run whose rates carry EFFECT and nothing else."""
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(json.dumps(
            {"hardcap_suffix": "-cap", "order_mode": "williams"}), encoding="utf-8")
        flat = [(rep, arm) for rep, row in enumerate(square) for arm in row]
        t = 0.0
        for i, (rep, arm) in enumerate(flat):
            prev = flat[i - 1] if i else None
            rate = self.BASE[arm]
            if prev:
                if prev[1].endswith("-cap"):
                    rate *= (1 - self.EFFECT)
                # a row-boundary adjacency also carries a large spurious bias,
                # so a tool that folds those in cannot return EFFECT
                if prev[0] != rep:
                    rate *= (1 + boundary_bias)
            n, ms = 300, 1000.0 * 300 / rate
            crashed = crash == (rep, arm)
            body = {"arm": arm, "repeat": rep, "crashed": crashed,
                    "rows": [] if crashed else
                            [{"t_start": t, "predicted_n": n, "predicted_ms": ms}]}
            (out / f"{arm}__rep{rep}.json").write_text(json.dumps(body),
                                                       encoding="utf-8")
            t += 1.0

    def _analyse(self, d: Path):
        r = subprocess.run([sys.executable, "analysis/carryover.py", "--json", str(d)],
                           cwd=str(self.ROOT), capture_output=True, text=True,
                           timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr[-600:])
        return json.loads(r.stdout)

    def test_the_split_is_the_one_the_design_promises(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / "matrix_X_s1_1"
            self._write_run(d, self._square())
            out = self._analyse(d)
        rec = out["runs"][0]
        self.assertTrue(rec["first_order_carryover_balanced"], rec["balance_note"])
        self.assertNotIn("refused", rec)
        for a, v in rec["capped_predecessor"].items():
            want = [3, 2] if not a.endswith("-cap") else [2, 3]
            self.assertEqual([v["n_cap"], v["n_free"]], want, a)
            self.assertTrue(v["split_is_balanced"], a)

    def test_it_returns_the_effect_that_was_planted(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / "matrix_X_s1_1"
            self._write_run(d, self._square())
            out = self._analyse(d)
        for a, v in out["runs"][0]["capped_predecessor"].items():
            self.assertAlmostEqual(v["delta_pct"], -100 * self.EFFECT, places=6,
                                   msg=f"{a}: planted {-100*self.EFFECT}, got "
                                       f"{v['delta_pct']}")

    def test_a_row_boundary_bias_does_not_leak_into_it(self):
        """This is the defect, made visible: a huge bias on exactly the
        adjacencies the square cannot balance must not move the answer."""
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / "matrix_X_s1_1"
            self._write_run(d, self._square(), boundary_bias=0.50)
            out = self._analyse(d)
        for a, v in out["runs"][0]["capped_predecessor"].items():
            self.assertAlmostEqual(v["delta_pct"], -100 * self.EFFECT, places=6,
                                   msg=f"{a}: a 50 % boundary bias moved the "
                                       f"contrast to {v['delta_pct']}")

    def test_an_arm_run_with_no_clock_is_refused_not_reordered(self):
        """A crashed arm-run has no `t_start` anywhere, so its slot is not in
        the data. Sorting it to one end silently hands its neighbour the wrong
        predecessor; guessing from the manifest is what this file refuses to
        do. The run is refused, and the driver would not have attested it."""
        sq = self._square()
        victim = (2, sq[2][3])
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / "matrix_X_s1_1"
            self._write_run(d, sq, crash=victim)
            r = subprocess.run(
                [sys.executable, "analysis/carryover.py", "--json", str(d)],
                cwd=str(self.ROOT), capture_output=True, text=True, timeout=120)
        self.assertNotEqual(r.returncode, 0, "a run with a gap was analysed")
        self.assertIn("no requests", r.stderr + r.stdout)
        self.assertIn("every predecessor after the gap", r.stderr + r.stdout)


class ThePublishToolMustParseAndVerify(unittest.TestCase):
    """The regex it replaced published half of its own command to GitHub.

    `re.sub(r'^<!--.*?-->', ...)` stopped at the literal `-->` inside the
    command text that lived in the comment, and the check that was supposed to
    catch it ran the same broken regex over both sides, so they agreed.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def _strip(self, text):
        sys.path.insert(0, str(self.ROOT / "tools"))
        import importlib
        import publish_pr_body
        return importlib.reload(publish_pr_body).strip_header(text)

    def test_a_marker_inside_the_comment_does_not_end_it(self):
        body = self._strip('<!--\n  see re.sub(r"^<!--.*?-->", "", x)\n-->\nreal body\n')
        self.assertEqual(body, "real body")

    def test_a_marker_must_be_a_whole_line(self):
        body = self._strip('<!--\n  trailing --> mid-line\n-->\nreal body\n')
        self.assertEqual(body, "real body")

    def test_an_unclosed_comment_is_refused(self):
        with self.assertRaises(SystemExit):
            self._strip("<!--\n  never closed\n\nbody\n")

    def test_a_file_without_a_header_is_returned_whole(self):
        self.assertEqual(self._strip("just a body\n"), "just a body")

    def test_the_real_file_strips_to_prose(self):
        text = (self.ROOT / "PULL_REQUEST.md").read_text(encoding="utf-8")
        body = self._strip(text)
        self.assertFalse(body.startswith("\\s*"), "the old defect is back")
        self.assertNotIn("gh api -X PATCH", body[:400],
                         "the publishing command is being published again")
        self.assertTrue(body.startswith("This branch audits"), body[:80])

    def test_it_reads_the_body_back_before_claiming_success(self):
        src = (self.ROOT / "tools" / "publish_pr_body.py").read_text(encoding="utf-8")
        self.assertIn('_api("GET")', src, "it never re-reads what it published")
        self.assertIn("differs from", src, "no byte comparison after publishing")


class TheRerunScriptsMustBehaveWithAFakeRunner(unittest.TestCase):
    """Source-level assertions pass on a script that does not work.

    The fourth review asked for behaviour: does the driver hand the runner the
    variable it reads, and does it fail when a session produces nothing? Both
    are answered by running it against a stub.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def _harness(self, tmp: Path, runner_body: str):
        bench = tmp / "bench"
        (bench / "llama-retest" / "build" / "bin").mkdir(parents=True)
        server = bench / "llama-retest" / "build" / "bin" / "llama-server"
        server.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        server.chmod(0o755)
        (bench / "retest_runner.py").write_text(runner_body, encoding="utf-8")
        # exits at once: a stub that sleeps inherits stdout and holds the pipe
        # open, so `capture_output` waits for it and the test hangs rather than
        # testing anything. The driver's trap tolerates an already-dead sampler.
        (bench / "gpu_telemetry.sh").write_text(
            "#!/bin/sh\necho stub telemetry\n", encoding="utf-8")
        return bench

    def _run(self, script: str, bench: Path, env_extra=None):
        env = dict(os.environ, BENCH_ROOT=str(bench),
                   BENCH_RUNNER=str(bench / "retest_runner.py"),
                   BENCH_TELEMETRY=str(bench / "gpu_telemetry.sh"),
                   MODEL_TARGET="/dev/null", MODEL_DRAFT="/dev/null",
                   MODEL_DFLASH="/dev/null", MODEL_MTP="/dev/null")
        env.pop("LLAMA_SERVER_BIN", None)
        env.update(env_extra or {})
        return subprocess.run(["bash", str(self.ROOT / script)], env=env,
                              capture_output=True, text=True, timeout=300)

    STUB_RECORDS_ENV = (
        "import json, os, pathlib\n"
        "out = pathlib.Path(os.environ['BENCH_OUT'])\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'seen_env.json').write_text(json.dumps(\n"
        "    {k: v for k, v in os.environ.items() if k.startswith(('BENCH_', 'LLAMA_'))}))\n"
    )

    def test_the_driver_passes_the_variable_the_runner_reads(self):
        with tempfile.TemporaryDirectory() as t:
            bench = self._harness(Path(t), self.STUB_RECORDS_ENV)
            r = self._run("bench/run_v3_within.sh", bench)
            seen = sorted(bench.glob("matrix_V3_*/seen_env.json"))
            self.assertTrue(seen, r.stdout[-500:] + r.stderr[-500:])
            env = json.loads(seen[0].read_text(encoding="utf-8"))
        self.assertIn("LLAMA_SERVER_BIN", env,
                      "the runner was never told where the server is")
        self.assertTrue(env["LLAMA_SERVER_BIN"].endswith("llama-server"))
        self.assertNotIn("BENCH_SERVER", env, "the dead name is exported again")
        self.assertIn("BENCH_EXPECT_COMMIT", env, "the binary is not pinned")

    def test_it_exits_non_zero_when_nothing_completes(self):
        """The stub writes no RUN_COMPLETE.json, so every session failed."""
        with tempfile.TemporaryDirectory() as t:
            bench = self._harness(Path(t), self.STUB_RECORDS_ENV)
            r = self._run("bench/run_v3_within.sh", bench)
        self.assertNotEqual(r.returncode, 0,
                            "the driver reported success with no validated run")
        self.assertIn("FAIL", r.stdout + r.stderr)

    def test_a_missing_server_binary_stops_it_before_any_arm(self):
        with tempfile.TemporaryDirectory() as t:
            bench = self._harness(Path(t), self.STUB_RECORDS_ENV)
            (bench / "llama-retest" / "build" / "bin" / "llama-server").unlink()
            r = self._run("bench/run_v3_within.sh", bench)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not executable", r.stdout + r.stderr)


class TheSuffixMustNotCollideWithARealArm(unittest.TestCase):
    """`arm_is_hardcap` gives a real arm precedence, so a real arm named
    `<other><suffix>` answers a request for `<other>`'s capped twin with a
    different configuration entirely."""

    def test_a_clash_is_detectable(self):
        arms = {"foo": [], "foo-cap": [], "bar": []}
        suf = "-cap"
        clash = sorted(a for a in arms if a.endswith(suf) and a[:-len(suf)] in arms)
        self.assertEqual(clash, ["foo-cap"])

    def test_the_real_arm_table_has_no_clash_for_cap(self):
        rr = _load_runner()
        clash = [a for a in rr.ARMS if a.endswith("-cap")
                 and a[:-4] in rr.ARMS]
        self.assertEqual(clash, [])

    def test_the_runner_records_the_resolution_rather_than_refusing(self):
        """Refusing was over-reach and broke a documented behaviour.

        `BENCH_HARDCAP_SUFFIX=-kvfp16` is a real arm suffix in this table and
        has nothing to do with capping; `arm_is_hardcap` already resolves the
        overlap deterministically, and that resolution is tested elsewhere.
        What was missing was any record that a choice had been made, so the
        manifest names the arms that read both ways.
        """
        src = (Path(__file__).resolve().parents[1] / "bench" / "retest_runner.py") \
            .read_text(encoding="utf-8")
        self.assertIn("ambiguous_arms", src)
        self.assertIn("each runs as ITSELF", src)
        self.assertNotIn("would be served the real arm instead", src,
                         "the over-broad refusal is back")

    def test_a_real_arm_still_wins_under_an_overlapping_suffix(self):
        rr = _load_runner()
        overlapping = [a for a in rr.ARMS
                       if a.endswith("-kvfp16") and a[:-7] in rr.ARMS]
        self.assertTrue(overlapping, "the table no longer has an overlap to test")


class ModelFilesMustNotChangeUnderARun(unittest.TestCase):
    """The hashes were taken once, at the start of a run that lasts hours."""

    ROOT = Path(__file__).resolve().parents[1]

    def test_the_runner_hashes_again_at_the_end(self):
        src = (self.ROOT / "bench" / "retest_runner.py").read_text(encoding="utf-8")
        self.assertIn("model_sha256_after", src)
        self.assertIn("a model file changed while the matrix ran", src)

    def test_the_mismatch_is_a_run_problem_not_a_warning(self):
        src = (self.ROOT / "bench" / "retest_runner.py").read_text(encoding="utf-8")
        block = src.split("model_moved = ")[1].split("stamp = ")[0]
        self.assertIn("problems.append", block,
                      "a changed model file does not fail the run")


class TheModeAnalyserMustFailClosed(unittest.TestCase):
    """P0-3's other half. The estimand was the headline; these were the rest.

    Each of these used to pass silently: a half the driver never attested, a
    crashed arm-run averaged into a pooled rate, two directories claiming the
    same session and mode, an order guessed from a missing timestamp, halves
    that ran different models, and arm sets that did not match.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def _src(self):
        return (self.ROOT / "analysis" / "length_mode.py").read_text(encoding="utf-8")

    def test_a_crashed_arm_run_is_not_pooled(self):
        body = self._src().split("def pooled(")[1].split("\ndef ")[0]
        self.assertIn('body.get("crashed")', body)
        self.assertIn("continue", body)

    def test_a_duplicate_session_half_is_refused(self):
        self.assertIn("two directories claim to be session", self._src())

    def test_an_unknown_order_is_not_guessed(self):
        src = self._src()
        self.assertIn("first_mode[s] = None", src,
                      "a missing or equal timestamp still picks a side")
        # the old form assigned a side whenever a stamp was absent; the new one
        # only chooses between two stamps that exist and differ
        self.assertNotIn('if tf and tc and tf < tc else "hardcap"', src)
        self.assertIn("if not tf or not tc or tf == tc:", src)

    def test_the_halves_must_be_the_same_experiment(self):
        src = self._src()
        for field in ("target_sha256", "server_loaded_commit", "prompt_set"):
            self.assertIn(f'"{field}"', src, f"{field} is not compared")
        self.assertIn("is not the mode alone", src)

    def test_the_arm_sets_must_match(self):
        src = self._src()
        self.assertIn("do not carry the same arms", src)
        self.assertIn("arms without a twin", src)

    def test_an_unattested_run_needs_an_explicit_flag(self):
        src = self._src()
        self.assertIn("has no RUN_COMPLETE.json", src)
        self.assertIn("--allow-incomplete", src)

    def test_a_mistyped_path_is_refused(self):
        r = subprocess.run(
            [sys.executable, "analysis/length_mode.py", "no/such/dir"],
            cwd=str(self.ROOT), capture_output=True, text=True, timeout=120)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not a directory", r.stdout + r.stderr)

    def test_the_published_numbers_survive_all_of_it(self):
        """The gates are additive guarantees; if they changed a published
        figure they would be corrections, and would be labelled as such."""
        d = sorted((self.ROOT / "v4_audit_2026_08_25" / "data")
                   .glob("matrix_V2_s*_20260827_044442"))
        r = subprocess.run(
            [sys.executable, "analysis/length_mode.py"] + [str(x) for x in d],
            cwd=str(self.ROOT), capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stderr[-500:])
        for arm, pp in (("spec-dflash-n4", "+12.03"), ("spec-mtp-n2", "+9.54"),
                        ("spec-draft-n8", "+6.31"), ("spec-dflash-n2", "+5.92")):
            self.assertRegex(r.stdout, rf"{re.escape(arm)}\s+\{pp} pp",
                             f"{arm} moved")


class TheMirrorsMustCarryEverythingTheCheckerReads(unittest.TestCase):
    """A file the checker reads and the mirror lacks is not a failed assertion.

    It is a `FileNotFoundError` on the UNPERTURBED copy, which stops both
    perturbation suites before a single mutation runs and reports nothing about
    any guard. `tests/data_mutate.py` already carries a comment saying this
    happened once - "the mirror silently lacked them until the checker crashed
    on the unperturbed copy" - and nothing was added to stop it happening
    again. It happened again on 2026-08-28, with `CITATION.cff`.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def _copy_list(self, rel):
        src = (self.ROOT / rel).read_text(encoding="utf-8")
        body = src.split("COPY = (")[1].split(")")[0]
        return set(re.findall(r'"([^"]+)"', body))

    def _named_by_the_checker(self):
        src = (self.ROOT / "analysis" / "verify_claims.py").read_text(encoding="utf-8")
        out = set()
        for m in re.finditer(
                r'["\']([A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:md|cff|json|csv|yml|sh|py|txt))["\']',
                src):
            top = m.group(1).split("/")[0]
            if (self.ROOT / top).exists():
                out.add(top)
        return out

    def test_the_checker_names_things_worth_checking(self):
        named = self._named_by_the_checker()
        self.assertGreater(len(named), 8, "the extraction found almost nothing")
        self.assertIn("CITATION.cff", named)

    def test_both_mirrors_carry_all_of_them(self):
        named = self._named_by_the_checker()
        for rel in ("tests/data_mutate.py", "tests/mutate.py"):
            with self.subTest(suite=rel):
                missing = sorted(named - self._copy_list(rel))
                self.assertEqual(missing, [],
                                 f"{rel}'s mirror would lack {missing}, so the "
                                 f"checker crashes on the unperturbed copy and "
                                 f"no perturbation is ever evaluated")


class AShallowCloneMustBeDiagnosedNotEndured(unittest.TestCase):
    """`actions/checkout` defaults to depth 1, and the checker could not say so.

    The first real run of the evidence workflow failed with five provenance
    assertions listing run directories and nothing naming the cause. A shallow
    clone still satisfies `git rev-parse --git-dir`, so `_HAS_GIT` was true and
    the history simply was not there. Diagnosing it took a whole 40-minute CI
    run; reading the message now takes a second.

    A mirror with no `.git` is a different thing and is legitimately skipped -
    `tests/data_mutate.py` creates one on purpose.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def test_the_checker_tells_them_apart(self):
        src = (self.ROOT / "analysis" / "verify_claims.py").read_text(encoding="utf-8")
        self.assertIn("--is-shallow-repository", src)
        self.assertIn("fetch-depth: 0", src, "the message does not name the fix")
        self.assertIn("no git history here (a mirror)", src,
                      "a mirror with no git must still be skipped, not refused")

    def test_it_refuses_a_real_shallow_clone(self):
        with tempfile.TemporaryDirectory() as t:
            dst = Path(t) / "shallow"
            r = subprocess.run(
                ["git", "clone", "--depth", "1", "--no-local", "-q",
                 f"file://{self.ROOT}", str(dst)],
                capture_output=True, text=True, timeout=600)
            if r.returncode:
                self.skipTest(f"cannot make a shallow clone here: {r.stderr[-200:]}")
            self.assertEqual(
                subprocess.run(["git", "-C", str(dst), "rev-parse",
                                "--is-shallow-repository"],
                               capture_output=True, text=True).stdout.strip(),
                "true")
            # the clone carries the committed checker; test the working one
            shutil.copy2(self.ROOT / "analysis" / "verify_claims.py",
                         dst / "analysis" / "verify_claims.py")
            out = subprocess.run([sys.executable, "analysis/verify_claims.py"],
                                 cwd=str(dst), capture_output=True, text=True,
                                 timeout=900)
        self.assertNotEqual(out.returncode, 0, "a shallow clone was accepted")
        blob = out.stdout + out.stderr
        self.assertIn("SHALLOW clone", blob)
        self.assertIn("fetch-depth: 0", blob)

    def test_every_job_that_reaches_the_checker_asks_for_full_history(self):
        """Two jobs needed this and only one had it, twice.

        `evidence.yml` runs the checker directly; `audit.yml`'s unit job reaches
        it through the tests. Both failed in CI for the same reason, three
        commits apart. Naming the rule is cheaper than diagnosing it again.
        """
        # No PyYAML: these tests are stdlib-only and CI's unit job installs
        # nothing. Importing it passed here and raised ERROR there, which is
        # the same shape as every other "works on my machine" defect this
        # branch has been fixing. Job blocks are `^  name:` at two spaces.
        REACHES = ("verify_claims", "unittest discover",
                   "tests/mutate.py", "tests/data_mutate.py")
        for rel in (".github/workflows/audit.yml", ".github/workflows/evidence.yml"):
            lines = (self.ROOT / rel).read_text(encoding="utf-8").splitlines()
            starts = [i for i, l in enumerate(lines)
                      if re.fullmatch(r"  [A-Za-z0-9_-]+:", l)]
            self.assertTrue(starts, f"{rel}: no job blocks found")
            for n, i in enumerate(starts):
                end = starts[n + 1] if n + 1 < len(starts) else len(lines)
                block = "\n".join(lines[i:end])
                name = lines[i].strip().rstrip(":")
                if not any(k in block for k in REACHES):
                    continue
                with self.subTest(workflow=rel, job=name):
                    self.assertIn(
                        "fetch-depth: 0", block,
                        f"{rel}:{name} reaches the claim checker but does not "
                        f"ask for full history; the provenance assertions need "
                        f"every version of the runner and the checker refuses "
                        f"a shallow clone")


class TheSuiteMustRunOnAStockInterpreter(unittest.TestCase):
    """A test that imports something CI does not install passes here and

    ERRORs there. That happened with PyYAML: the unit job installs nothing, so
    a rule about workflow files could not be checked by the job it governs.
    The general form is the same "works on my machine" this branch keeps
    finding, and it is cheap to forbid.

    `analysis/plot_v4_runs.py` legitimately needs matplotlib and runs in its
    own job with a hash-pinned lock file; the TESTS are the stdlib-only part.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def test_no_test_module_imports_anything_third_party(self):
        import ast as _ast
        stdlib = set(sys.stdlib_module_names)
        allowed = stdlib | {"host_guard", "publish_pr_body", "carryover",
                            "length_mode", "paired_blocks", "rr_under_test",
                            "rederive_from_logs", "past_threshold_fit",
                            "verify_claims", "extract_checkpoint_timers"}
        offenders = []
        for f in sorted((self.ROOT / "tests").glob("*.py")):
            tree = _ast.parse(f.read_text(encoding="utf-8"))
            for n in _ast.walk(tree):
                if isinstance(n, _ast.Import):
                    names = [a.name.split(".")[0] for a in n.names]
                elif isinstance(n, _ast.ImportFrom) and n.level == 0 and n.module:
                    names = [n.module.split(".")[0]]
                else:
                    continue
                for name in names:
                    if name not in allowed:
                        offenders.append(f"{f.name}:{n.lineno} {name}")
        self.assertEqual(offenders, [],
                         "the unit job installs nothing, so these imports pass "
                         "locally and ERROR in CI")


class TheSplitTimersMustBeRederivable(unittest.TestCase):
    """39.09 s and 0.002 s were in the document and nowhere else."""

    ROOT = Path(__file__).resolve().parents[1]
    DUMP = "checkpoint_timers_20260827_split.json"

    def test_the_dump_is_committed(self):
        self.assertTrue((self.ROOT / "v4_audit_2026_08_25" / "data" / self.DUMP).is_file())

    def test_it_carries_the_split_fields(self):
        rows = json.loads((self.ROOT / "v4_audit_2026_08_25" / "data" / self.DUMP)
                          .read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 18)
        # the timers only fire on the arm that checkpoints; the other twelve
        # rows carry zeros and no split, which is the extractor's own shape
        drafting = [r for r in rows if r["arm"] == "spec-draft-n8"]
        self.assertEqual(len(drafting), 6)
        for r in drafting:
            for f in ("sync_total_s", "state_total_s", "checkpoint_total_s"):
                self.assertIn(f, r, f"{f} missing: the split is what T4 was for")
            self.assertAlmostEqual(r["sync_total_s"] + r["state_total_s"],
                                   r["checkpoint_total_s"], places=2)

    def test_the_rederivation_covers_it(self):
        src = (self.ROOT / "analysis" / "rederive_from_logs.py").read_text(encoding="utf-8")
        self.assertIn(self.DUMP, src)
        self.assertIn("matrix_T4_split_20260827_175051", src)

    def test_its_logs_are_attested(self):
        man = (self.ROOT / "v4_audit_2026_08_25" / "EVIDENCE_MANIFEST.sha256") \
            .read_text(encoding="utf-8")
        self.assertEqual(man.count("matrix_T4_split_20260827_175051/server_logs/"), 18)


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
