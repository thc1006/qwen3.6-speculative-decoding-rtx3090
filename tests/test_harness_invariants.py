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
        (d / "RUN_COMPLETE.json").write_text(json.dumps({"expected_arm_runs": len(arms) * repeats}),
                                             encoding="utf-8")
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
