#!/usr/bin/env python3
"""Decide BY MEASUREMENT which logical processors share a physical core.

    python3 bench/cpu_siblings.py --show          # what it would do, no timing
    python3 bench/cpu_siblings.py                 # measure, then compare to /sys
    python3 bench/cpu_siblings.py --pairs 0-1,0-2

Why this exists
---------------
`PROSPECTIVE_PLAN_SIBLINGS.md` asks what shares a physical core with llama.cpp's
main thread, and its arms are two affinity masks: one taking a single hyperthread
from each of eight cores, one taking both hyperthreads of four. That design rests
entirely on a claim about the hardware, and the plan's own prerequisite 3 says
nothing in this repository can check it. `/sys` states it, but `/sys` states what
the kernel was told, which on a virtual machine is whatever the hypervisor chose
to say. The development box this was written on reports eight processors and eight
distinct cores while its host is an i9 13900 with eight performance cores of two
threads each: if any two of those vCPUs are backed by one physical core, the
masks mean something other than what they say and nothing in the guest would
show it.

So the topology is measured rather than read, and then the measurement is
compared with `/sys`. Agreement makes the plan's masks checkable. Disagreement is
the more interesting outcome and the one worth finding before a run rather than
after.

How
---
`bench/cpu_ilp.c` is a throughput-bound integer kernel with no memory traffic;
its own header explains why that shape and not another, and why a latency-bound
loop would report every pair as distinct. Two hyperthreads of one core contend
for that core's multiplier, so:

    ratio(i, j) = rate(i and j together) / (rate(i alone) + rate(j alone))

is near one for processors on different cores and well under one for two threads
on the same core. The bands below are deliberately wide and the gap between them
is refused rather than guessed.

Both ends are anchored, which is the part that makes a number a measurement:

  * the NEGATIVE control is any pair `/sys` calls distinct, which must land near
    one. If those do not, the instrument is measuring something else -- a shared
    cache, a frequency change, another tenant -- and no verdict is issued.
  * the POSITIVE control is a pair where both threads are pinned to the SAME
    processor. One processor cannot do two threads' work, so that must land near
    one half. It is not hyperthread sharing, and it is not offered as such; it
    establishes that contention is detectable and that the arithmetic has the
    right sign. A run that reports every pair as distinct AND reports the
    same-processor control as distinct has failed, not found a topology.

The sibling band itself cannot be exercised on a machine with no siblings. Where
that is the case this says so and does not claim to have been validated against
one.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
KERNEL = ROOT / "bench" / "cpu_ilp.c"

# The lock is NOT asked about here. This file used to carry
# `LOCK = ROOT / ".gpu-in-use.lock"`, which is checkout-relative, while
# `bench/host_guard.py` searches absolute paths: two readers of one lock in one
# repository with two notions of where it is, and they could only ever agree at
# one location. A test that asserted they coincided passed in the working tree
# and failed in the gate's clean clone, which is the honest answer -- a clone
# under `~/ci_repro` is not a measurement host. So there is one reader now, and
# it is the guard.

# A pair at or below SHARED_MAX shares something that costs this much; a pair at
# or above DISTINCT_MIN shares nothing that this kernel can see. Between them is
# refused. Two hyperthreads of a modern x86 core on a throughput-bound integer
# loop come out around 0.55 to 0.70, and distinct cores come out within a few per
# cent of 1.0, so the gap is real rather than a rounding allowance.
SHARED_MAX = 0.80
DISTINCT_MIN = 0.92

# The same-processor control. Two threads sharing one processor get half each,
# and the scheduler's own overhead takes a little more, so the band is one-sided
# below 0.5 and tight above it.
SAME_CPU_MAX = 0.60

# Solo rates are re-measured around every pair rather than once at the start,
# because a long run drifts: this kernel saturates the multiplier, the part gets
# hot, and the clock comes down. On the box this was written on a two second
# window ran three per cent slower than a half second one. SPREAD_MAX is how much
# disagreement between a processor's own repeats is tolerated before the run is
# refused as unstable rather than reported as a topology.
SPREAD_MAX = 0.04


def classify(rate_i: float, rate_j: float, rate_pair: float) -> tuple[str, float]:
    """Pure. The timing is host-dependent and this is not.

    Separated out deliberately. The measurement cannot be reproduced in CI --
    a runner has four processors, no hyperthreads it will admit to, and a clock
    that moves under it -- so what CI can check is the arithmetic and the bands,
    with figures handed to them. An inverted comparison here would report every
    sibling pair as distinct and the run would look clean.
    """
    if rate_i <= 0.0 or rate_j <= 0.0 or rate_pair <= 0.0:
        return "refused: a rate was not positive", 0.0
    ratio = rate_pair / (rate_i + rate_j)
    if ratio <= SHARED_MAX:
        return "shared", ratio
    if ratio >= DISTINCT_MIN:
        return "distinct", ratio
    return "refused: between the bands", ratio


def validate_masks(distinct: set, packed: set, threads: int,
                   ceiling: dict, sibling: dict) -> tuple[list, dict]:
    """Pure. Why these two masks are not plan S's two conditions, and what they are.

    Separated from the driver for the same reason `classify` is separated from the
    timing: this part is host-INDEPENDENT and can therefore be tested, while
    reading `/sys` cannot. A runner has four processors, no `cpufreq` and no
    siblings it will admit to, so a check that lives only inside
    `bench/run_s_siblings.sh` is a check CI never executes. The driver reads the
    host and calls this; the suite calls it with topologies it builds.

    `ceiling` maps every processor ON THE HOST to its maximum clock in MHz, and
    `sibling` maps each processor to the set sharing its physical core. The host
    rather than the selection, because the clock TIERS are a property of the part:
    this one has three, 4300 for the efficiency cores, 5500 for the performance
    cores and 5800 for the two the turbo favours. The first version of this took
    the maximum over the SELECTION and called everything below ninety five per
    cent of it an efficiency core, which reported nine ordinary performance-core
    threads as efficiency cores, because 5500 over 5800 is 0.948. A threshold
    cannot separate three tiers. They have to be enumerated.
    """
    bad = []
    tiers = sorted(set(ceiling.values()))
    sel = sorted(distinct | packed)
    missing = [c for c in sel if c not in ceiling or c not in sibling]
    if missing:
        return ([f"processors {missing} are not in the host's own topology, so "
                 f"these masks name processors this machine does not have"],
                {"tiers": tiers})

    for name, s_ in (("distinct", distinct), ("packed", packed)):
        if len(s_) != threads:
            bad.append(f"the {name} set has {len(s_)} processors and the thread "
                       f"count is {threads}: the contrast would change the thread "
                       f"count too")

    if len(tiers) > 1:
        slow = sorted(c for c in sel if ceiling[c] == tiers[0])
        if slow:
            bad.append(f"processors {slow} are on this host's lowest clock tier "
                       f"({tiers[0]} MHz of {tiers}), so at least one set holds an "
                       f"efficiency core; that is run Y's contrast, not this one")

    fav = {c for c in sel if ceiling[c] == tiers[-1]}
    if len(tiers) > 1:
        n_d, n_p = len(distinct & fav), len(packed & fav)
        if n_d != n_p:
            bad.append(f"the distinct set holds {n_d} of the favoured processors "
                       f"{sorted(fav)} and the packed set holds {n_p}. They would "
                       f"differ in clock as well as in sharing, and the clock "
                       f"difference alone is larger than the band the plan "
                       f"reserves for deciding the mechanism is absent")

    d_cores = {frozenset(sibling[c]) for c in distinct} if distinct else set()
    if distinct and len(d_cores) != len(distinct):
        bad.append(f"the distinct set spans {len(d_cores)} physical cores for "
                   f"{len(distinct)} processors, so two of them share one and it "
                   f"is not the no-sharing condition")
    for c in sorted(packed):
        partners = sibling[c] - {c}
        if not (partners & packed):
            bad.append(f"cpu{c} is in the packed set and none of its siblings "
                       f"{sorted(partners) or 'none'} is, so it shares its core "
                       f"with nothing and the set is not fully packed")
            break
    p_cores = {frozenset(sibling[c]) for c in packed} if packed else set()
    if distinct and packed and len(p_cores) >= len(d_cores):
        bad.append(f"the packed set spans {len(p_cores)} physical cores and the "
                   f"distinct set {len(d_cores)}: packing has to span fewer or "
                   f"the two are the same condition written twice")
    return bad, {"tiers": tiers, "favoured": sorted(fav),
                 "favoured_in_distinct": len(distinct & fav),
                 "favoured_in_packed": len(packed & fav),
                 "distinct_cores": len(d_cores), "packed_cores": len(p_cores)}


def online_cpus() -> list[int]:
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:          # not Linux
        return []


def sys_siblings() -> dict[int, frozenset]:
    """What the kernel says, which on a guest is what the hypervisor said."""
    out = {}
    for c in online_cpus():
        p = pathlib.Path(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list")
        try:
            raw = p.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        got = set()
        for part in raw.split(","):
            if "-" in part:
                lo, hi = part.split("-", 1)
                got.update(range(int(lo), int(hi) + 1))
            else:
                got.add(int(part))
        out[c] = frozenset(got)
    return out


def compile_kernel(into: pathlib.Path) -> pathlib.Path:
    if not KERNEL.is_file():
        raise SystemExit(f"FAIL: {KERNEL} is missing; nothing to measure with")
    exe = into / "cpu_ilp"
    cc = os.environ.get("CC", "cc")
    r = subprocess.run([cc, "-O2", "-Wall", "-Wextra", "-o", str(exe), str(KERNEL)],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise SystemExit(f"FAIL: {cc} could not build the kernel:\n{r.stderr.strip()}")
    # A warning here is not cosmetic: the loop is only throughput bound if the
    # optimiser kept it, and the usual way to lose it is a warning about an
    # unused result.
    if r.stderr.strip():
        raise SystemExit(f"FAIL: the kernel built with warnings, so what it "
                         f"measures is not known:\n{r.stderr.strip()}")
    return exe


def _one(exe: pathlib.Path, cpus: list[int], seconds: float) -> list[float]:
    """Run one instance per entry in `cpus`, all at once, and return their rates.

    `cpus` may repeat a processor: that is the positive control, and it has to go
    through the same path as everything else or it controls nothing.
    """
    procs = []
    for c in cpus:
        procs.append(subprocess.Popen(
            ["taskset", "-c", str(c), str(exe), f"{seconds}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    rates = []
    for p, c in zip(procs, cpus):
        out, err = p.communicate(timeout=int(seconds) + 120)
        if p.returncode != 0:
            raise SystemExit(f"FAIL: the kernel on cpu {c} exited "
                             f"{p.returncode}: {err.strip()}")
        parts = out.split()
        if len(parts) != 3:
            raise SystemExit(f"FAIL: the kernel on cpu {c} printed {out!r}")
        blocks, elapsed = float(parts[0]), float(parts[1])
        if elapsed <= 0.0:
            raise SystemExit(f"FAIL: the kernel on cpu {c} reported {elapsed} s")
        # blocks over MEASURED seconds, never over the requested window: a
        # process that was descheduled took longer than it asked for, and
        # dividing by the request would hand it the gap as throughput.
        rates.append(blocks / elapsed)
    return rates


def solo(exe: pathlib.Path, cpu: int, seconds: float, repeats: int) -> float:
    """The median of `repeats`, and a refusal if they disagree."""
    got = sorted(_one(exe, [cpu], seconds)[0] for _ in range(repeats))
    mid = got[len(got) // 2]
    spread = (got[-1] - got[0]) / mid if mid > 0 else 1.0
    if spread > SPREAD_MAX:
        raise SystemExit(
            f"FAIL: cpu {cpu} measured {spread * 100:.1f} % apart across "
            f"{repeats} repeats, over the {SPREAD_MAX * 100:.0f} % this allows. "
            f"The clock is moving or something else is running, and a topology "
            f"read off an unstable rate is a guess with a decimal point.")
    return mid


def parse_pairs(spec: str, cpus: list[int]) -> list[tuple[int, int]]:
    if spec == "all":
        pairs = [(a, b) for i, a in enumerate(cpus) for b in cpus[i + 1:]]
        return pairs
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" not in item:
            raise SystemExit(f"FAIL: {item!r} is not a pair like 0-1")
        a, b = item.split("-", 1)
        try:
            pair = (int(a), int(b))
        except ValueError:
            raise SystemExit(f"FAIL: {item!r} is not a pair of numbers")
        for c in pair:
            if c not in cpus:
                raise SystemExit(f"FAIL: cpu {c} is not in this process's "
                                 f"affinity mask {cpus}")
        out.append(pair)
    if not out:
        raise SystemExit("FAIL: --pairs named nothing")
    return out


def refuse_if_busy() -> None:
    sys.path.insert(0, str(ROOT / "bench"))
    import host_guard                                        # noqa: E402

    held = host_guard.lock_held()
    if held is not None:
        raise SystemExit(
            f"FAIL: {held} exists, so a GPU measurement is in progress. This "
            f"saturates every processor it is given and would land in that run "
            f"as host load.")
    try:
        one_minute = float(pathlib.Path("/proc/loadavg").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return
    # Not a style rule: a busy machine makes a pair of DISTINCT cores look like
    # siblings, because the second thread is competing with something that is
    # not its partner. The failure is toward finding siblings that are not there.
    if one_minute > 1.0:
        raise SystemExit(
            f"FAIL: load average is {one_minute:.2f}. Other work makes distinct "
            f"cores look shared, so this would invent a topology. Wait for the "
            f"machine to be idle.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", default="all",
                    help="'all', or a list like 0-1,2-3")
    ap.add_argument("--seconds", type=float, default=1.0,
                    help="window per run (default 1.0)")
    ap.add_argument("--repeats", type=int, default=3,
                    help="repeats per solo rate (default 3)")
    ap.add_argument("--show", action="store_true",
                    help="say what would be measured and how long, then stop")
    a = ap.parse_args()

    cpus = online_cpus()
    if len(cpus) < 2:
        raise SystemExit(f"FAIL: this process may use {cpus}; a pair needs two")
    if not (0.1 <= a.seconds <= 60.0):
        raise SystemExit("FAIL: --seconds must be between 0.1 and 60")
    if not (1 <= a.repeats <= 25):
        raise SystemExit("FAIL: --repeats must be between 1 and 25")

    pairs = parse_pairs(a.pairs, cpus)
    # solo repeats for every processor, one run per pair, plus the control
    runs = len(cpus) * a.repeats + len(pairs) + 1
    budget = runs * a.seconds
    print(f"{len(cpus)} processors {cpus}, {len(pairs)} pair(s), "
          f"{runs} runs of {a.seconds} s is about {budget / 60:.1f} min")
    claimed = sys_siblings()
    groups = sorted({tuple(sorted(v)) for v in claimed.values()})
    print(f"  /sys claims {len(groups)} physical core(s): "
          f"{', '.join('+'.join(str(c) for c in g) for g in groups)}")
    if budget > 1800 and a.pairs == "all":
        raise SystemExit(
            f"FAIL: 'all' here is {budget / 60:.0f} minutes. Name the pairs that "
            f"answer the question instead, or lower --seconds.")
    if a.show:
        return 0

    refuse_if_busy()
    with tempfile.TemporaryDirectory(prefix="cpu_siblings.") as tmp:
        exe = compile_kernel(pathlib.Path(tmp))

        print("  solo rates:")
        rate = {}
        for c in cpus:
            rate[c] = solo(exe, c, a.seconds, a.repeats)
            print(f"    cpu {c:<3} {rate[c]:10.1f} blocks/s")

        # The positive control first, so a run that cannot detect contention at
        # all stops before it reports twenty-eight reassuring numbers.
        ctl_cpu = cpus[0]
        ctl_rates = _one(exe, [ctl_cpu, ctl_cpu], a.seconds)
        ctl = sum(ctl_rates) / (2.0 * rate[ctl_cpu])
        print(f"  control, two threads on cpu {ctl_cpu} alone: {ctl:.3f}")
        if ctl > SAME_CPU_MAX:
            raise SystemExit(
                f"FAIL: two threads pinned to one processor measured {ctl:.3f} of "
                f"its solo rate, and one processor cannot do two threads' work. "
                f"Either taskset is not binding or the kernel is not the "
                f"bottleneck, and in both cases every pair below would read as "
                f"distinct for the wrong reason.")

        print("  pairs:")
        measured = {}
        bad = []
        for (i, j) in pairs:
            rp = sum(_one(exe, [i, j], a.seconds))
            verdict, ratio = classify(rate[i], rate[j], rp)
            same_core = j in claimed.get(i, frozenset())
            agrees = ((verdict == "shared") == same_core) if verdict in (
                "shared", "distinct") else False
            mark = "" if agrees else "   <-- disagrees with /sys"
            if verdict.startswith("refused"):
                mark = f"   <-- {verdict}"
            print(f"    {i:>3} + {j:<3} {ratio:6.3f}  {verdict:<9}"
                  f"  /sys says {'same core' if same_core else 'different'}{mark}")
            measured[(i, j)] = (verdict, ratio, same_core)
            if not agrees:
                bad.append((i, j))

        # The negative control is every pair /sys calls distinct. If those do not
        # land near one, this kernel is measuring something that is not the core.
        neg = [r for (p, (v, r, same)) in measured.items() if not same]
        if neg and min(neg) <= SHARED_MAX and all(
                not measured[p][2] for p in measured):
            print("  note: a pair /sys calls distinct measured as shared. That is "
                  "either a topology /sys does not report, or contention this "
                  "kernel should not see.")

        print()
        exercised_sibling_band = any(same for (_, _, same) in measured.values())
        if not bad:
            print(f"measurement agrees with /sys on all {len(measured)} pair(s)")
        else:
            print(f"measurement DISAGREES with /sys on {len(bad)} pair(s): {bad}")
        if not exercised_sibling_band:
            print("no pair on this machine is a sibling pair by /sys, so the "
                  "shared band was never exercised here and this run does not "
                  "validate it. The same-processor control is what was checked.")
        return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
