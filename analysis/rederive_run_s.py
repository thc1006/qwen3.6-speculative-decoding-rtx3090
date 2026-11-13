#!/usr/bin/env python3
"""Verify plan S's design, then derive its two contrasts with the estimator it pledged.

    python3 analysis/rederive_run_s.py            # verify, derive, print
    python3 analysis/rederive_run_s.py --json     # the same, as JSON
    python3 analysis/rederive_run_s.py --conditions DIR   # only verify DIR

The third form is what `bench/run_s_siblings.sh` runs on the directory it has
just written, so the driver's check of its own run and the reading's check of
the committed one are one function.

Exits 0 with a message when no plan S run is committed, so the claims job can run
it from the day it is written rather than from the day the data lands. A script
wired in later is a script nobody wires in.

Written before the data. The first version was written while a first attempt at
the run was in progress, and committed after that attempt had been stopped, in
the commit that records why it was stopped (76d8583); nothing derived a contrast
from it, this file included, and no plan S run has produced a number since. What
that attempt's arm-runs showed before it was stopped is recorded in the plan's
reading section, by name.

WHY THIS VERIFIES BEFORE IT DERIVES
-----------------------------------
The first version read seven fields out of an arm-run record that carries
twenty-three. It never asked whether an arm got the processors it requested,
whether the three arms ran the same server, whether the thread count matched,
or whether exactly one of them carried the third arm's setting. It computed a ratio and
printed a verdict. A run where `packed` had quietly run on the distinct mask
would have passed it without a word.

The second version asked those questions and answered several of them on
absence: a set of `server_loaded_commit` values that is `{None}` has one member,
so a record that never carried the field passed the "one server" check. Every
check below now requires the field it reads, because an absent field is how the
first seventy eight runs here hid their threadpool parameters.

So: `conditions()` runs first and `derive()` refuses if anything in it failed.

WHAT IS PARSED AND WHAT IS DERIVED
----------------------------------
Nothing the plan decides is written down here a second time. The bands, the
masks, the runner's arm names and the block count are parsed out of the plan;
the spread the design assumed is `load_run_power`'s derivation; the definition
of a balanced schedule is `paired_blocks.is_position_balanced`. This repository
has removed the two-copies defect repeatedly -- the gate's step list, the mask
check, the measurement lock's two homes -- and the first version of this file
had three more of them, including the band edges, which it held without their
sign: a `packed` three per cent FASTER read as "the candidate stands".
"""
from __future__ import annotations

import json
import math
import pathlib
import re
import statistics as st
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from paired_blocks import (                                      # noqa: E402
    is_position_balanced, load_blocks, observed_schedule, t_critical_975)

ROOT = pathlib.Path(__file__).resolve().parents[1]
PLAN = ROOT / "v4_audit_2026_08_25" / "PROSPECTIVE_PLAN_SIBLINGS.md"
MINUS = "−"          # U+2212, which is what the documents carry
# Flags that set a threadpool, in llama.cpp's spelling at the pinned commit:
# `-C/--cpu-mask`, `-Cr/--cpu-range`, `--cpu-strict`, `--prio`, `--poll`, each
# with a `-batch` twin, and the `--spec-draft-` ones that land nowhere. No arm may
# carry one. The third arm first carried `--poll 0`, and the server these runs
# use is an OpenMP build, where ggml never reads it: what differs in the third
# arm is the server's environment, `OMP_WAIT_POLICY`, which the record reads back
# from `/proc` as `server_env`.
POOL_SHORT = ("-C", "-Cr", "-Cb", "-Crb")
POOL_WORDS = ("poll", "cpu-mask", "cpu-range", "cpu-strict", "prio")


def _pool_flag(a: str) -> bool:
    return a in POOL_SHORT or (a.startswith("--") and any(w in a for w in POOL_WORDS))


def _pct(v: float) -> str:
    return f"{MINUS if v < 0 else ''}{abs(v):.2f} %"


def cpus(spec) -> set:
    out = set()
    for part in str(spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out |= set(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def _plan_text() -> str:
    return PLAN.read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    m = re.search(rf"## {re.escape(heading)}(.*?)\n## ", text, re.S)
    if not m:
        raise SystemExit(f"{PLAN.name} has no '{heading}' section, so what this "
                         f"reads out of it cannot be found")
    return m[1]


def design() -> dict:
    """The arms, their masks and the block count, as the plan registers them.

    The masks come from the arm table, the runner's names from the sentence that
    says which name runs which condition, and the block count from the bold row
    of the design table. Each is checked for the shape the reading needs, so a
    plan edited into an inconsistent state is refused here rather than read.
    """
    text = _plan_text()
    arms_sec = _section(text, "The three arms")
    rows = re.findall(r"^\| `([^`]+)` \| `([0-9,]+)` \| (.*) \|$", arms_sec, re.M)
    if len(rows) != 3:
        raise SystemExit(f"{PLAN.name}: the arm table has {len(rows)} rows of "
                         f"the form | `name` | `mask` | and |, and the plan has "
                         f"three arms")
    # The server environment each arm is given, written in the table's third
    # column as `NAME=value`. Only the OpenMP runtime's variables can appear
    # there, and only the third arm may carry any.
    envs = [dict(re.findall(r"`((?:OMP|GOMP|KMP)_[A-Z_]+)=([^`\s]+)`", r[2]))
            for r in rows]
    if envs[0] or envs[1] or not envs[2]:
        raise SystemExit(f"{PLAN.name}: the arm table gives the environments "
                         f"{envs}; the first two arms take none and the third "
                         f"exactly the setting it differs by")
    flat = " ".join(arms_sec.split())
    m = re.search(r"It runs them as `([^`]+)`, `([^`]+)` and `([^`]+)`", flat)
    if not m:
        raise SystemExit(f"{PLAN.name}: the sentence naming the runner's arms "
                         f"('It runs them as `a`, `b` and `c`') is not there")
    names = list(m.groups())
    if len(set(names)) != 3:
        raise SystemExit(f"{PLAN.name}: the runner's arm names repeat: {names}")
    masks = {name: frozenset(cpus(mask)) for name, (_cond, mask, _and) in zip(names, rows)}
    sizes = {len(v) for v in masks.values()}
    if len(sizes) != 1:
        raise SystemExit(f"{PLAN.name}: the masks have {sorted(sizes)} processors; "
                         f"the plan holds the thread count fixed, so they must "
                         f"be the same size")
    distinct, packed, passive = names
    if masks[packed] != masks[passive]:
        raise SystemExit(f"{PLAN.name}: the second and third arms have different "
                         f"masks, and the third is meant to differ from the "
                         f"second by its environment alone")
    if masks[distinct] == masks[packed]:
        raise SystemExit(f"{PLAN.name}: the first two arms have the same mask, so "
                         f"the primary contrast is between nothing")

    dsec = _section(text, "The design")
    b = re.findall(r"^\| \*\*(\d+)\*\* \| \*\*(\d+)\*\* \|", dsec, re.M)
    if len(b) != 1:
        raise SystemExit(f"{PLAN.name}: the design table has {len(b)} bold rows, "
                         f"and exactly one of them names the chosen block count")
    blocks, arm_runs = int(b[0][0]), int(b[0][1])
    if blocks * len(names) != arm_runs:
        raise SystemExit(f"{PLAN.name}: the chosen row says {blocks} blocks and "
                         f"{arm_runs} arm-runs, which is not {len(names)} arms "
                         f"a block")

    # What the plan fixes for every arm alike: the order, the token cap and the
    # build. Read from the plan so a run with a different one is refused rather
    # than compared: the predecessor structure the plan states is `latin`'s, the
    # same GPU work is the cap's, and every line number the plan cites is the
    # build's.
    def one_of(pat, what, src=flat):
        found = re.findall(pat, src)
        if len(found) != 1:
            raise SystemExit(f"{PLAN.name}: {what} is written {len(found)} times "
                             f"in a form this can read (/{pat}/), and it has to be "
                             f"exactly once")
        return found[0]
    order = one_of(r"blocks of three in `(\w+)` order", "the order")
    cap = int(one_of(r"to a cap of `(\d+)` tokens", "the token cap"))
    commit = one_of(r"built from `([0-9a-f]{40})`", "the build",
                    " ".join(text.split()))
    return {"distinct": distinct, "packed": packed, "passive": passive,
            "arms": tuple(names), "masks": masks, "blocks": blocks,
            "threads": sizes.pop(), "env": dict(zip(names, envs)),
            "order": order, "cap": cap, "commit": commit}


def assumed_sd_pct() -> float:
    """The within-invocation spread the design's half widths were computed from.

    `load_run_power`'s derivation from run T4's adjacent differences, which is
    what `plan_siblings_power` feeds the design table. The first version quoted
    it as a literal, a second copy of a derived figure.
    """
    import load_run_power                                          # noqa: E402
    return load_run_power.measured()["adjacent_diff_sd"]


def assumed_effect_sd_pct() -> float:
    """The design's second spread: run Y's own twelve within-block log ratios.

    `plan_z_power` derives it and the plan's design table carries it beside the
    null spread, because a real effect widens the spread. Printed beside the
    observed spread so a reader sees which of the two the run resembled.
    """
    import plan_z_power                                            # noqa: E402
    return plan_z_power.design()[1]["sd_effect_pct"]


def bands() -> list:
    """The pre-registered reading, parsed out of the plan.

    Returns (lo, hi, text) over the SIGNED change, lowest first, covering the
    line. Parsed rather than copied, and the plan writes the edges as digits for
    exactly this reason. Which band an edge belongs to is `in_band`'s rule, which
    is the plan's: the band nearer zero.
    """
    sec = _section(_plan_text(), "The pre-registered reading")
    want = {
        "null": (r"\*\*Inside (-\d+) % to \+(\d+) %\*\*", "inside the null band: "
                 "hyperthread sharing is not the mechanism"),
        "predicted": (r"\*\*(-\d+) % to (-\d+) %, packed slower\*\*",
                      "the size a yielding spin predicts: the candidate stands"),
        "too_big": (r"\*\*Below (-\d+) %\*\*",
                    "too large for a yielding spin: something takes the core"),
        "faster": (r"\*\*Above \+(\d+) %, packed FASTER\*\*",
                   "packed faster, which nothing here predicts: the arms differ "
                   "in something this design did not control"),
    }
    got = {}
    for key, (pat, _label) in want.items():
        found = [mm.groups() for mm in re.finditer(pat, sec)]
        if len(found) != 1:
            raise SystemExit(
                f"{PLAN.name}: the {key!r} band is written {len(found)} times in "
                f"a form this can read, and it has to be exactly once. Expected "
                f"/{pat}/ in the pre-registered reading, so that the edges live "
                f"in the plan and not in a second copy here.")
        got[key] = tuple(int(x) for x in found[0])
    null_lo, null_hi = got["null"]
    pred_lo, pred_hi = got["predicted"]
    big = got["too_big"][0]
    fast = got["faster"][0]
    if not (big < pred_lo < pred_hi <= null_lo < 0 < null_hi <= fast):
        raise SystemExit(
            f"the plan's bands do not order: too_big {big}, predicted "
            f"{pred_lo}..{pred_hi}, null {null_lo}..{null_hi}, faster {fast}. "
            f"Overlapping or inverted bands would make the reading ambiguous.")
    gap = ("between the bands the plan reserved, which it says it did not "
           "reserve")
    out = [(-math.inf, float(big), want["too_big"][1])]
    # Only where there IS a gap: an empty interval in the table is one more
    # thing a reader has to work out is unreachable.
    for lo, hi, text in ((big, pred_lo, gap),
                         (pred_lo, pred_hi, want["predicted"][1]),
                         (pred_hi, null_lo, gap),
                         (null_lo, null_hi, want["null"][1]),
                         (null_hi, fast, gap)):
        if lo < hi:
            out.append((float(lo), float(hi), text))
    out.append((float(fast), math.inf, want["faster"][1]))
    return out


def in_band(lo: float, hi: float, x: float) -> bool:
    """An edge belongs to the band nearer zero, which is the plan's rule.

    Below zero that is the band above the edge, so the interval is [lo, hi);
    at or above zero it is the band below, (lo, hi]. Exactly -1 and exactly +1
    are then both the null band, which is the smaller finding in each direction.
    """
    return lo <= x < hi if x < 0 else lo < x <= hi


def interval(lo: float, hi: float) -> str:
    """The band as an interval, with the brackets `in_band` actually applies."""
    left = "(" if not math.isfinite(lo) or not in_band(lo, hi, lo) else "["
    right = ")" if not math.isfinite(hi) or not in_band(lo, hi, hi) else "]"
    return f"{left}{lo:g} %, {hi:g} %{right}"


def verdict(change_pct: float, table: list | None = None) -> str:
    """SIGNED. A gain and a loss of the same size are different findings."""
    if not math.isfinite(change_pct):
        raise ValueError(f"the change is {change_pct}, which no band can hold; "
                         f"the cause is upstream of the reading")
    hits = [text for lo, hi, text in (table or bands()) if in_band(lo, hi, change_pct)]
    if len(hits) != 1:
        raise ValueError(f"{change_pct} falls in {len(hits)} bands; the table "
                         f"is not a partition of the line")
    return hits[0]


def reached(lo_pct: float, hi_pct: float, table: list | None = None) -> list:
    """Every band an interval reaches, in order along the line.

    The bands are contiguous, so the ones an interval reaches are exactly those
    from its lower end's band to its upper end's. By position in the table, not
    by text: the gap's text appears twice.
    """
    table = table or bands()

    def at(x):
        idx = [i for i, (lo, hi, _t) in enumerate(table) if in_band(lo, hi, x)]
        if len(idx) != 1:
            raise ValueError(f"{x} falls in {len(idx)} bands")
        return idx[0]
    return [t for _lo, _hi, t in table[at(lo_pct):at(hi_pct) + 1]]


def _run_dir() -> pathlib.Path | None:
    found = sorted(ROOT.glob("v*/data/matrix_S_siblings_*"))
    if len(found) > 1:
        raise SystemExit(
            f"{len(found)} plan S run directories: "
            f"{', '.join(p.name for p in found)}. Which one the figures describe "
            f"would depend on which sorted first, so this refuses rather than "
            f"choosing.")
    return found[0] if found else None


def _normalised_argv(rec: dict) -> tuple[list | None, list]:
    """The argv with the designed difference taken out, and what was wrong.

    `taskset -c <mask>` is the one difference argv may show, and it must name the
    mask the record says was requested. The third arm's difference is its
    environment, which argv does not carry, so no threadpool flag may appear in
    any arm. What is left has to be the same for every arm-run, so a model, a
    drafter, a draft length, a context size or a thread count that differs
    anywhere shows up as a difference here.
    """
    bad = []
    argv = rec.get("argv")
    if not isinstance(argv, list) or not argv:
        return None, ["no argv, so what the server was started with is unrecorded"]
    argv = list(argv)
    if argv[:2] != ["taskset", "-c"] or len(argv) < 4:
        return None, [f"argv does not start with `taskset -c`: {argv[:3]}"]
    if cpus(argv[2]) != cpus(rec.get("cpus_requested")):
        bad.append(f"taskset was given {argv[2]!r} and the record says "
                   f"{rec.get('cpus_requested')!r} was requested")
    argv = argv[3:]
    left = [a for a in argv if _pool_flag(a)]
    if left:
        bad.append(f"threadpool flags {left} in argv, which an OpenMP build does "
                   f"not read and the plan does not vary")
    return argv, bad


def conditions(run: pathlib.Path, d: dict | None = None) -> tuple[list, dict]:
    """Everything the reading depends on, read out of the run itself.

    Not one of these is a style check. Each is a way the three arms could have
    differed in something other than the one thing the plan varies, or a way the
    run could be other than the one the plan registered, and each is answerable
    from a field the runner records. A field that is absent is a failure, never
    a pass.
    """
    d = d or design()
    bad: list[str] = []

    if not (run / "RUN_COMPLETE.json").is_file():
        why = " (RUN_FAILED.json is there)" if (run / "RUN_FAILED.json").exists() else ""
        bad.append(f"no RUN_COMPLETE.json{why}: the runner did not attest this "
                   f"directory as a whole run")
    if (run / "ABANDONED.txt").exists():
        bad.append("ABANDONED.txt is here: this is a run that was stopped")
    try:
        man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        man = {}
        bad.append(f"manifest.json unreadable: {e}")

    files = sorted(run.glob("*__rep*.json"))
    if not files:
        return bad + [f"{run.name} holds no arm-run files"], {}
    recs = []
    for f in files:
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            bad.append(f"{f.name}: does not parse: {e}")
            continue
        if not isinstance(r, dict):
            bad.append(f"{f.name}: is not an arm-run record")
            continue
        recs.append((f.name, r))
    # A refusal, not a crash: a record of the wrong shape further down would
    # raise, and the reasons gathered so far would be lost with it.
    shape = []
    for name, r in recs:
        if not isinstance(r.get("repeat"), int) or isinstance(r.get("repeat"), bool):
            shape.append(f"{name}: repeat is {r.get('repeat')!r}, not a block number")
        for x in r.get("rows") or []:
            miss = [k for k in ("tag", "predicted_n", "predicted_ms", "t_start")
                    if not isinstance(x, dict) or x.get(k) is None]
            if miss:
                shape.append(f"{name}: a request row has no {', '.join(miss)}")
                break
    if shape:
        return bad + shape, {}

    # What the plan fixes for every arm alike.
    if man.get("order_mode") != d["order"]:
        bad.append(f"the order is {man.get('order_mode')!r} and the plan "
                   f"registers {d['order']!r}; the predecessor structure the "
                   f"plan states is that order's")
    if man.get("max_tokens") != d["cap"]:
        bad.append(f"the token cap is {man.get('max_tokens')!r} and the plan "
                   f"registers {d['cap']}")

    crashed = [n for n, r in recs if r.get("crashed")]
    if crashed:
        bad.append(f"{len(crashed)} arm-run(s) are marked crashed: "
                   f"{', '.join(crashed[:3])}")

    # 0. exactly the plan's arms, each in every block once
    cells = Counter((r.get("arm"), r.get("repeat")) for _n, r in recs)
    seen = {a for a, _r in cells}
    if seen != set(d["arms"]):
        bad.append(f"the arms are {sorted(map(str, seen))} and the plan's are "
                   f"{sorted(d['arms'])}")
    for arm in d["arms"]:
        reps = sorted(r for a, r in cells if a == arm)
        if reps != list(range(d["blocks"])):
            bad.append(f"{arm} ran in blocks {reps}, not each of the "
                       f"{d['blocks']} once")
    dup = sorted(f"{a} rep{r}" for (a, r), n in cells.items() if n > 1)
    if dup:
        bad.append(f"two files claim one cell: {dup}")

    # 1. the mask each arm asked for, the mask the kernel gave, the thread count
    for name, r in recs:
        arm = r.get("arm")
        if arm not in d["masks"]:
            continue
        req, got = cpus(r.get("cpus_requested")), cpus(r.get("cpus_allowed"))
        if not req:
            bad.append(f"{name}: no cpus_requested, so the placement is unrecorded")
        elif req != d["masks"][arm]:
            bad.append(f"{name}: {arm} asked for {sorted(req)} and the plan "
                       f"assigns it {sorted(d['masks'][arm])}")
        if not got:
            bad.append(f"{name}: no cpus_allowed, so nothing read back what the "
                       f"kernel applied")
        elif got != req:
            bad.append(f"{name}: asked {sorted(req)}, the kernel gave {sorted(got)}")
        if r.get("threads") != d["threads"]:
            bad.append(f"{name}: threads {r.get('threads')!r}, and the plan fixes "
                       f"{d['threads']}, the size of every mask; a cpuset "
                       f"otherwise moves the thread count with it")

    # 2. one server: the same build, the same libraries beside it, the same
    #    shared objects actually mapped, and a fresh process every arm-run
    def one(field, project=lambda v: v):
        vals, missing = set(), []
        for name, r in recs:
            v = r.get(field)
            if not v or (isinstance(v, dict) and "error" in v):
                missing.append(name)
                continue
            vals.add(json.dumps(project(v), sort_keys=True))
        if missing:
            bad.append(f"{len(missing)} arm-run(s) carry no usable {field}, e.g. "
                       f"{missing[0]}; an absent field cannot show two arms ran "
                       f"the same server")
        if len(vals) > 1:
            bad.append(f"{len(vals)} different values of {field} across the "
                       f"arm-runs, so the arms did not run the same server")
    one("server_loaded_commit")
    commits = {str(r.get("server_loaded_commit") or "") for _n, r in recs}
    if any(len(c) < 7 or not d["commit"].startswith(c) for c in commits):
        bad.append(f"the server reports commit(s) {sorted(commits)}, and the plan "
                   f"is written against {d['commit'][:12]}: every source line it "
                   f"cites is that build's")
    one("server_lib_sha256")
    # Path and content, not inode or mtime: those identify a file, and the
    # question is whether the same code was mapped.
    one("server_mapped",
        lambda v: {p: (e or {}).get("sha256") for p, e in v.items()})
    pids = [r.get("server_pid") for _n, r in recs]
    if None in pids or len(set(pids)) != len(pids):
        bad.append("server_pid is missing or repeats, and the plan's argument "
                   "that nothing of one arm's placement survives into the next "
                   "rests on every arm-run starting a fresh server")

    # 3. the argv: the mask out, everything else identical, no pool flag
    shapes, recorded_poll, env_bad, gomp_bad, thr_bad = set(), [], [], [], []
    for name, r in recs:
        arm = r.get("arm")
        if arm not in d["masks"]:
            continue
        argv, why = _normalised_argv(r)
        bad += [f"{name}: {w}" for w in why]
        if argv is not None:
            shapes.add(json.dumps(argv))
        for field in ("poll", "poll_batch"):
            if field not in r:
                recorded_poll.append(f"{name}: no `{field}` field")
            elif r[field] is not None:
                recorded_poll.append(f"{name}: `{field}` is {r[field]!r}")
        # 3a. the third arm's difference, read back from the server's /proc
        env = r.get("server_env")
        if not isinstance(env, dict) or "error" in env:
            env_bad.append(f"{name}: server_env is {env!r}")
        elif env != d["env"][arm]:
            env_bad.append(f"{name}: {arm} was given {env} and the plan "
                           f"gives it {d['env'][arm]}")
        # 3b. the build is an OpenMP one, so the environment is the lever
        mapped = r.get("server_mapped")
        if isinstance(mapped, dict) and "error" not in mapped and not any(
                pathlib.PurePath(q).name.startswith("libgomp") for q in mapped):
            gomp_bad.append(name)
        # 3c. the observation the mechanism needs
        thr = r.get("server_threads")
        if (not isinstance(thr, dict) or "error" in thr
                or not isinstance(thr.get("threads"), list) or not thr["threads"]
                or not isinstance(thr.get("clk_tck"), int)
                or not all(isinstance(t, dict) and isinstance(t.get("ticks"), int)
                           and isinstance(t.get("main"), bool)
                           for t in thr["threads"])):
            thr_bad.append(f"{name}: server_threads is {thr!r}"[:160])
    if len(shapes) > 1:
        bad.append(f"{len(shapes)} different server command lines once the mask "
                   f"is taken out; the arms differ in something the plan does "
                   f"not vary")
    if recorded_poll:
        bad.append(f"the recorded poll does not match the design: "
                   f"{'; '.join(recorded_poll[:3])}")
    if env_bad:
        bad.append(f"the server environment does not match the design: "
                   f"{'; '.join(env_bad[:3])}")
    if gomp_bad:
        bad.append(f"{len(gomp_bad)} arm-run(s) mapped no libgomp, e.g. "
                   f"{gomp_bad[0]}: without OpenMP the third arm's setting "
                   f"reaches nothing")
    if thr_bad:
        bad.append(f"the per-thread processor time is missing: "
                   f"{'; '.join(thr_bad[:2])}")

    # 4. the same work: every prompt, every request run to the cap
    tags, cap = man.get("prompt_tags"), man.get("max_tokens")
    if not tags or not cap:
        bad.append("the manifest names no prompt set or token cap, so the "
                   "work the arms did cannot be compared with anything")
    else:
        short = []
        for name, r in recs:
            rows = r.get("rows") or []
            if sorted(str(x.get("tag")) for x in rows) != sorted(tags):
                bad.append(f"{name}: prompts "
                           f"{sorted(str(x.get('tag')) for x in rows)} against "
                           f"the manifest's {sorted(tags)}")
            short += [f"{name} {x.get('tag')} {x.get('predicted_n')}"
                      for x in rows if x.get("predicted_n") != cap]
        if short:
            bad.append(f"{len(short)} request(s) did not generate the cap of "
                       f"{cap} tokens, e.g. {short[0]}: the plan's arms do the "
                       f"same GPU work, and a request that stops early is less "
                       f"of it")

    # 5. position balance, rebuilt from t_start rather than read off a label,
    #    and judged by the runner's own definition of balanced
    info: dict = {"arm_runs": len(recs)}
    if crashed:
        bad.append("positions are not rebuilt from a run with a crashed arm-run")
        return bad, info
    try:
        sched, n_blocks = observed_schedule(run)
    except (SystemExit, KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
        bad.append(f"the order cannot be rebuilt: {type(e).__name__}: {e}")
        return bad, info
    counts = {a: dict(sorted(Counter(v).items())) for a, v in sorted(sched.items())}
    if not is_position_balanced(sched):
        bad.append(f"the observed positions are not balanced: {counts}. A "
                   f"position-balanced schedule puts every arm in every position "
                   f"the same number of times")
    info.update({"positions": counts, "blocks_observed": n_blocks})
    return bad, info


def other_threads_cpu_s(run: pathlib.Path, d: dict) -> dict:
    """Per arm, the mean processor seconds the server's threads OTHER than the
    main one spent while the requests ran.

    Descriptive, not part of any band. It is the observation the plan's
    candidate has never had: a worker that spins between graphs accrues
    processor time and one that sleeps does not. It does not separate
    libgomp's workers from the server's other threads, which carry the same
    name, so it is read as a difference between arms that share everything
    but the wait policy, the packed and passive arms, and not as a level.
    """
    per_arm: dict = {a: [] for a in d["arms"]}
    for f in sorted(run.glob("*__rep*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        thr = r["server_threads"]
        per_arm[r["arm"]].append(
            sum(t["ticks"] for t in thr["threads"] if not t["main"])
            / thr["clk_tck"])
    return {a: st.fmean(v) for a, v in per_arm.items()}


def spin_observed(cpu_s: dict, d: dict) -> bool:
    """Whether the workers were seen to spin, by the plan's own rule.

    The predicted band says the candidate stands only if the other threads'
    processor time in the passive arm is at most half of that in the packed arm,
    per arm-run. Parsed from the band's text rather than held here.
    """
    sec = " ".join(_section(_plan_text(), "The pre-registered reading").split())
    m = re.search(r"processor time in `packed-passive` at most (half) of that in "
                  r"`packed`, per arm-run", sec)
    if not m:
        raise SystemExit(f"{PLAN.name}: the predicted band does not state the "
                         f"spinning rule in a form this can read")
    packed, passive = cpu_s[d["packed"]], cpu_s[d["passive"]]
    return packed > 0 and passive <= 0.5 * packed


def derive(run: pathlib.Path | None = None) -> dict:
    run = run if run is not None else _run_dir()
    if run is None:
        raise SystemExit("no plan S run is committed, so there is nothing to derive")
    d = design()
    bad, info = conditions(run, d)
    if bad:
        raise SystemExit(
            f"{run.name}: the design is not what the plan registered, so no "
            f"contrast is derived from it:\n  " + "\n  ".join(bad))

    blocks = load_blocks(run)
    table = bands()
    rows = []
    for base, other in ((d["distinct"], d["packed"]), (d["packed"], d["passive"])):
        have = sorted(b for b, v in blocks.items() if base in v and other in v)
        if len(have) != d["blocks"]:
            raise SystemExit(
                f"{other} against {base}: {len(have)} complete blocks, not "
                f"{d['blocks']}. The interval's degrees of freedom are published, "
                f"so a different count is a different number, not a near miss.")
        b = [blocks[k][base] for k in have]
        o = [blocks[k][other] for k in have]
        for k, (x, y) in enumerate(zip(b, o)):
            if not (x > 0 and y > 0 and math.isfinite(x) and math.isfinite(y)):
                raise SystemExit(
                    f"block {have[k]} has a pooled rate that is not a positive "
                    f"number ({base} {x}, {other} {y}); a log ratio of that is "
                    f"not a number and the cause is upstream of this file")
        lr = [math.log(o[i] / b[i]) for i in range(len(have))]
        m = st.fmean(lr)
        sd = st.stdev(lr)
        se = sd / math.sqrt(len(lr))
        df = len(lr) - 1
        half = t_critical_975(df) * se
        rows.append({
            "base": base, "other": other,
            "base_mean": st.fmean(b), "other_mean": st.fmean(o),
            "change_pct": (math.exp(m) - 1) * 100,
            "ci_lo_pct": (math.exp(m - half) - 1) * 100,
            "ci_hi_pct": (math.exp(m + half) - 1) * 100,
            "half_width_pct": (math.exp(half) - 1) * 100,
            # The design tables' half width and spreads are mean log ratios
            # times a hundred, which is not the same scale as a back-transformed
            # percentage: 1.33 there is 1.34 here at twelve blocks. Both are
            # given, each labelled, so nothing is compared across the two.
            "half_width_log_x100": half * 100,
            "blocks": len(lr), "df": df, "mean_log_ratio": m,
            "sd_log_ratio": sd, "sd_log_x100": sd * 100,
        })
    primary = rows[0]
    cpu_s = other_threads_cpu_s(run, d)
    spun = spin_observed(cpu_s, d)
    band = verdict(primary["change_pct"], table)
    return {"run": run.name, "rows": rows,
            "other_threads_cpu_s": cpu_s, "spin_observed": spun,
            "candidate_stands": "candidate stands" in band and spun,
            "bands": [{"lo": lo if math.isfinite(lo) else None,
                       "hi": hi if math.isfinite(hi) else None, "reading": t}
                      for lo, hi, t in table],
            "verdict_primary": verdict(primary["change_pct"], table),
            "interval_reaches": reached(primary["ci_lo_pct"],
                                        primary["ci_hi_pct"], table),
            "assumed_sd_pct": assumed_sd_pct(),
            "assumed_effect_sd_pct": assumed_effect_sd_pct(), **info}


def main() -> int:
    args = sys.argv[1:]
    if args[:1] == ["--conditions"]:
        if len(args) != 2:
            print("usage: rederive_run_s.py --conditions DIR", file=sys.stderr)
            return 2
        bad, info = conditions(pathlib.Path(args[1]))
        for b in bad:
            print(f"FAIL: {b}", file=sys.stderr)
        if bad:
            return 1
        print(f"every condition plan S registers holds in {args[1]}: "
              f"{info.get('arm_runs')} arm-runs, positions {info.get('positions')}")
        return 0
    for a in args:
        if a != "--json":
            print(f"unknown argument {a!r}; the only ones are --json and "
                  f"--conditions DIR", file=sys.stderr)
            return 2
    as_json = "--json" in args
    d = design()
    table = bands()
    run = _run_dir()
    if run is None:
        # Not an error. This is wired into the claims job from the day it is
        # written, and it starts checking the moment a run is committed.
        print("no v*/data/matrix_S_siblings_* directory yet; plan S has no run "
              "committed, and what it would read is checked below")
        print(f"  arms {', '.join(d['arms'])}, {d['blocks']} blocks, "
              f"{d['threads']} threads")
        for lo, hi, text in table:
            print(f"  {interval(lo, hi):18s} {text}")
        print("  the plan's design parses, its bands partition the line, and an "
              "edge belongs to the band nearer zero")
        return 0
    out = derive(run)
    if as_json:
        print(json.dumps(out, indent=2, allow_nan=False))
        return 0

    print(f"plan S, {out['run']}, {out['arm_runs']} arm-runs")
    for r in out["rows"]:
        print(f"  {r['other']} against {r['base']}")
        print(f"     {r['base_mean']:8.2f} -> {r['other_mean']:8.2f} tok/s   "
              f"{_pct(r['change_pct'])}  "
              f"[{_pct(r['ci_lo_pct'])}, {_pct(r['ci_hi_pct'])}]")
        print(f"     {r['blocks']} blocks, df {r['df']}, half width "
              f"{_pct(r['half_width_pct'])}")
        print(f"     on the design table's scale, log ratio x100: half width "
              f"{r['half_width_log_x100']:.2f}, within-block sd "
              f"{r['sd_log_x100']:.3f} against the {out['assumed_sd_pct']:.3f} and "
              f"{out['assumed_effect_sd_pct']:.3f} the design was computed on")
    p = out["rows"][0]
    print(f"\n  primary, packed against distinct: {_pct(p['change_pct'])}")
    print(f"  pre-registered reading: {out['verdict_primary']}")
    if "candidate stands" in out["verdict_primary"] and not out["candidate_stands"]:
        print("  BUT the other threads' processor time does not show the workers "
              "spinning, so by the plan's rule the candidate does not stand: "
              "sharing a core costs this much and the spinning worker is not why")
    if len(out["interval_reaches"]) > 1:
        print("  its 95 % interval reaches: "
              + "; and ".join(out["interval_reaches"]))
    print("  secondary, passive against packed: reported above; the plan "
          "registers no numeric band for it")
    print("  processor seconds of the server's other threads, per arm-run:")
    for arm, v in out["other_threads_cpu_s"].items():
        print(f"    {arm:34s} {v:8.1f} s")
    print("\n  the design, verified from the arm-runs and not from a label:")
    for arm, pos in out["positions"].items():
        print(f"    {arm:34s} positions {pos}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
