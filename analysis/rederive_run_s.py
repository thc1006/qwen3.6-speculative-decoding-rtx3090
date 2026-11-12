#!/usr/bin/env python3
"""Verify plan S's design, then derive its two contrasts with the estimator it pledged.

    python3 analysis/rederive_run_s.py            # verify, derive, print
    python3 analysis/rederive_run_s.py --json     # the same, as JSON

Exits 0 with a message when no plan S run is committed, so the claims job can run
it from the day it is written rather than from the day the data lands. A script
wired in later is a script nobody wires in.

Written before the data: the first version was committed while a run that was
later abandoned sat in its third block, and no plan S run has produced a number
since. What that run's arm-runs showed before it was stopped is recorded in the
plan's reading section, by name.

WHY THIS VERIFIES BEFORE IT DERIVES
-----------------------------------
The first version read seven fields out of an arm-run record that carries
twenty-three. It never asked whether an arm got the processors it requested,
whether the three arms ran the same server, whether the thread count matched,
or whether exactly one of them carried `--poll 0`. It computed a ratio and
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
# with a `-batch` twin, and the `--spec-draft-` ones that land nowhere. Only the
# third arm may carry one, `--poll 0`.
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
    rows = re.findall(r"^\| `([^`]+)` \| `([0-9,]+)` \|", arms_sec, re.M)
    if len(rows) != 3:
        raise SystemExit(f"{PLAN.name}: the arm table has {len(rows)} rows of "
                         f"the form | `name` | `mask` |, and the plan has three arms")
    flat = " ".join(arms_sec.split())
    m = re.search(r"It runs them as `([^`]+)`, `([^`]+)` and `([^`]+)`", flat)
    if not m:
        raise SystemExit(f"{PLAN.name}: the sentence naming the runner's arms "
                         f"('It runs them as `a`, `b` and `c`') is not there")
    names = list(m.groups())
    if len(set(names)) != 3:
        raise SystemExit(f"{PLAN.name}: the runner's arm names repeat: {names}")
    masks = {name: frozenset(cpus(mask)) for name, (_cond, mask) in zip(names, rows)}
    sizes = {len(v) for v in masks.values()}
    if len(sizes) != 1:
        raise SystemExit(f"{PLAN.name}: the masks have {sorted(sizes)} processors; "
                         f"the plan holds the thread count fixed, so they must "
                         f"be the same size")
    distinct, packed, nopoll = names
    if masks[packed] != masks[nopoll]:
        raise SystemExit(f"{PLAN.name}: the second and third arms have different "
                         f"masks, and the third is meant to differ from the "
                         f"second by `--poll 0` alone")
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
    return {"distinct": distinct, "packed": packed, "nopoll": nopoll,
            "arms": tuple(names), "masks": masks, "blocks": blocks,
            "threads": sizes.pop()}


def assumed_sd_pct() -> float:
    """The within-invocation spread the design's half widths were computed from.

    `load_run_power`'s derivation from run T4's adjacent differences, which is
    what `plan_siblings_power` feeds the design table. The first version quoted
    it as a literal, a second copy of a derived figure.
    """
    import load_run_power                                          # noqa: E402
    return load_run_power.measured()["adjacent_diff_sd"]


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


def _normalised_argv(rec: dict, arm: str, d: dict) -> tuple[list | None, list]:
    """The argv with the two designed differences taken out, and what was wrong.

    `taskset -c <mask>` is the first designed difference and must name the mask
    the record says was requested; `--poll 0` is the second and must appear in
    the third arm exactly once and nowhere else. What is left has to be the same
    for every arm-run, so a model, a drafter, a draft length, a context size or
    a thread count that differs anywhere shows up as a difference here.
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
    if arm == d["nopoll"]:
        at = [i for i, a in enumerate(argv) if a == "--poll"]
        if len(at) != 1 or at[0] + 1 >= len(argv) or argv[at[0] + 1] != "0":
            bad.append(f"the third arm carries --poll {len(at)} time(s), and it "
                       f"must carry `--poll 0` exactly once")
        else:
            del argv[at[0]:at[0] + 2]
    left = [a for a in argv if _pool_flag(a)]
    if left:
        bad.append(f"threadpool flags {left} in argv, and the plan varies only "
                   f"`--poll` and only in the third arm")
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
            recs.append((f.name, json.loads(f.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError) as e:
            bad.append(f"{f.name}: does not parse: {e}")

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

    # 3. the argv: the designed differences out, everything else identical
    shapes, recorded_poll = set(), []
    for name, r in recs:
        arm = r.get("arm")
        if arm not in d["masks"]:
            continue
        argv, why = _normalised_argv(r, arm, d)
        bad += [f"{name}: {w}" for w in why]
        if argv is not None:
            shapes.add(json.dumps(argv))
        if "poll" not in r:
            recorded_poll.append(f"{name}: no `poll` field")
        elif r["poll"] != (0 if arm == d["nopoll"] else None):
            recorded_poll.append(f"{name}: `poll` is {r['poll']!r}")
    if len(shapes) > 1:
        bad.append(f"{len(shapes)} different server command lines once the mask "
                   f"and the third arm's `--poll 0` are taken out; the arms "
                   f"differ in something the plan does not vary")
    if recorded_poll:
        bad.append(f"the recorded poll does not match the design: "
                   f"{'; '.join(recorded_poll[:3])}")

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
    except SystemExit as e:
        bad.append(f"the order cannot be rebuilt: {e}")
        return bad, info
    counts = {a: dict(sorted(Counter(v).items())) for a, v in sorted(sched.items())}
    if not is_position_balanced(sched):
        bad.append(f"the observed positions are not balanced: {counts}. A "
                   f"position-balanced schedule puts every arm in every position "
                   f"the same number of times")
    info.update({"positions": counts, "blocks_observed": n_blocks})
    return bad, info


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
    for base, other in ((d["distinct"], d["packed"]), (d["packed"], d["nopoll"])):
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
            "blocks": len(lr), "df": df, "mean_log_ratio": m,
            "sd_log_ratio": sd, "sd_pct": (math.exp(sd) - 1) * 100,
        })
    primary = rows[0]
    return {"run": run.name, "rows": rows,
            "bands": [{"lo": lo if math.isfinite(lo) else None,
                       "hi": hi if math.isfinite(hi) else None, "reading": t}
                      for lo, hi, t in table],
            "verdict_primary": verdict(primary["change_pct"], table),
            "interval_reaches": reached(primary["ci_lo_pct"],
                                        primary["ci_hi_pct"], table),
            "assumed_sd_pct": assumed_sd_pct(), **info}


def main() -> int:
    args = sys.argv[1:]
    for a in args:
        if a != "--json":
            print(f"unknown argument {a!r}; the only one is --json",
                  file=sys.stderr)
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
              f"{_pct(r['half_width_pct'])}, within-block sd "
              f"{r['sd_pct']:.3f} % against the {out['assumed_sd_pct']:.3f} % the "
              f"design assumed")
    p = out["rows"][0]
    print(f"\n  primary, packed against distinct: {_pct(p['change_pct'])}")
    print(f"  pre-registered reading: {out['verdict_primary']}")
    if len(out["interval_reaches"]) > 1:
        print("  its 95 % interval reaches: "
              + "; and ".join(out["interval_reaches"]))
    print("  secondary, nopoll against packed: reported above; the plan registers "
          "no numeric band for it")
    print("\n  the design, verified from the arm-runs and not from a label:")
    for arm, pos in out["positions"].items():
        print(f"    {arm:34s} positions {pos}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
