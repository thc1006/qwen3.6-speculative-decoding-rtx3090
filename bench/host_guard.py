#!/usr/bin/env python3
"""Refuse to run CPU-heavy verification while this host is measuring.

Why this exists
---------------
On 2026-08-27 the verification pipeline in this repository was recorded by a
benchmark harness on the same machine as two `host_contended` incidents,
`python3 100 %` and `python3 97 %`, and the measurement they landed on had to
be re-run. The sibling repository's harness samples host load once per arm-pass
and refuses a result that logged any incident, so one stray burst costs a whole
rung.

What actually caused it, because the obvious answer was wrong
-------------------------------------------------------------
`tests/mutate.py` and `tests/data_mutate.py` are **sequential** - one
`verify_claims.py` or one `unittest` subprocess at a time, no pool, no threads.
Neither can produce six cores of load on its own. Two things could and did:

  1. Overlapping pipelines. Several full verification runs were launched
     back to back and at times one was still finishing while the next started,
     so several single-threaded processes were live at once. That is an
     operating error, and `serialise()` below makes it impossible rather than
     impolite.

  2. Unbounded BLAS threads. `analysis/plot_v4_runs.py` imports matplotlib,
     which imports numpy, whose OpenBLAS spawns one thread per core by
     default. Nothing in this repository pinned that. `limit_threads()` does.

Attributing the burst to "the mutation suite is parallel" would have been a
plausible story that no measurement supports, which is the mistake ERRATA A12
was written about. It is sequential; the load came from running several of them
at once and from a library nobody had bounded.

Use
---
    import host_guard
    host_guard.protect("data mutation suite")

Fail-open by design: anything unexpected lets the work proceed. The one hard
stop is an explicit lock, because that is a deliberate signal from a running
measurement.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Set in every child too, so a subprocess that imports numpy does not undo this.
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

# A measurement announces itself with a lock file. `BENCH_GPU_LOCK` overrides
# the search; otherwise these are the paths the machines here actually use.
_DEFAULT_LOCKS = (
    "~/.gpu-in-use.lock",
    "~/dev/qwen3.8-speculative-decoding-rtx3090/.gpu-in-use.lock",
    "~/bench/.gpu-in-use.lock",
)


def limit_threads(n: int = 1) -> None:
    """Pin every numeric library to `n` threads, for this process and children.

    Must run before numpy is imported to take effect in this process; it always
    takes effect in children, which is what the suites actually spawn.
    """
    for var in _THREAD_VARS:
        os.environ.setdefault(var, str(n))


def be_nice(increment: int = 10) -> None:
    """Yield to anything measuring. Cannot be undone, so only ever lowers."""
    try:
        os.nice(increment)
    except (OSError, AttributeError):
        pass


def lock_held() -> Path | None:
    """The lock file a measurement is holding, or None."""
    override = os.environ.get("BENCH_GPU_LOCK")
    candidates = (override,) if override else _DEFAULT_LOCKS
    for c in candidates:
        if not c:
            continue
        try:
            p = Path(c).expanduser()
            if p.exists():
                return p
        except OSError:
            continue
    return None


def _own_pids() -> set[int]:
    """This process and its descendants, so we never count ourselves."""
    try:
        children = {}
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / "stat").read_text(encoding="utf-8")
            except OSError:
                continue
            # comm can contain spaces and parentheses; ppid is the field after
            # the state letter, which follows the last ')'
            tail = fields[fields.rfind(")") + 1:].split()
            if len(tail) < 2:
                continue
            children.setdefault(int(tail[1]), set()).add(int(entry.name))
        own, stack = set(), [os.getpid()]
        while stack:
            pid = stack.pop()
            if pid in own:
                continue
            own.add(pid)
            stack.extend(children.get(pid, ()))
        return own
    except OSError:
        return {os.getpid()}


# Matched at argv positions, never as a substring of the whole command line.
_BENCH_EXE = ("llama-server", "llama-bench")
_BENCH_SCRIPTS = ("bench.py", "retest_runner.py")


def _benchmark_name(argv: list[str]) -> str | None:
    """The benchmark this argv *is*, or None. Position, not keyword.

    `"bench.py" in cmdline` matches an editor with the file open, a grep for
    it, and this session's own shell command that merely names it. A command
    guard on this machine narrowed the same way seven times and the lesson each
    time was the same: a substring match reads text and cannot recover intent,
    so precision has to come from position and boundaries. `argv[0]`'s basename
    is a position; the script being an actual argument is a boundary.
    """
    if not argv:
        return None
    exe = os.path.basename(argv[0])
    if exe in _BENCH_EXE:
        return exe
    if exe.split(".")[0].startswith("python"):
        # `python3 -u harness/bench.py ...` - the script is an argument near the
        # front, compared whole, not searched for inside a longer string
        for a in argv[1:4]:
            if os.path.basename(a) in _BENCH_SCRIPTS and not _in_scratch(a):
                return os.path.basename(a)
    return None


def _in_scratch(path: str) -> bool:
    """A harness copy under a throwaway mirror is a test, not a measurement.

    `tests/mutate.py` and `tests/data_mutate.py` each copy `bench/` into a
    temporary directory and run the harness there, to check that a broken one
    is caught. Started side by side on 2026-08-29 each saw the other's copy,
    `python3 /tmp/tmp*/work/bench/retest_runner.py`, and refused to run. The
    match was right about the name and right about the position and still
    wrong about what the process was: a measurement does not live under the
    temporary directory, and a copy of the harness running there is the suite
    that tests it.
    """
    try:
        real = os.path.realpath(path)
    except OSError:
        return False
    return real.startswith(os.path.realpath(tempfile.gettempdir()) + os.sep)


def measuring_processes() -> list[str]:
    """Benchmark processes running on this host that are not ours.

    Attribution is by descent, not by name: the sibling repository learned that
    matching on names is wrong in both directions - it recorded a run's own
    `nvidia-smi` sampler as competition, and never recorded an analysis script
    that genuinely was.
    """
    own = _own_pids()
    found = []
    try:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit() or int(entry.name) in own:
                continue
            try:
                raw = (entry / "cmdline").read_bytes()
            except OSError:
                continue
            argv = [a for a in raw.decode("utf-8", "replace").split("\0") if a]
            if _benchmark_name(argv):
                found.append(f"{entry.name} {' '.join(argv)[:70]}")
    except OSError:
        return []
    return found


def protect(what: str, allow_env: str = "BENCH_ALLOW_CONTENDED") -> None:
    """Bound this process's CPU footprint, and stop if a measurement is live.

    CI always proceeds: there is no card there to contend for, and the suites
    are the point of the job.
    """
    limit_threads()
    be_nice()
    if os.environ.get("CI") or os.environ.get(allow_env) == "1":
        return
    lock = lock_held()
    running = measuring_processes()
    if not lock and not running:
        return
    why = []
    if lock:
        why.append(f"a measurement holds {lock}")
    if running:
        why.append("benchmark processes are running: " + "; ".join(running[:3]))
    sys.exit(
        f"refusing to run {what}: " + ", and ".join(why) + ".\n"
        "  A CPU burst on a measuring host invalidates the arm-pass it lands\n"
        "  on, and this suite spawns a subprocess per mutation. Wait for the\n"
        "  run to finish, or set "
        f"{allow_env}=1 if you know the card is free."
    )


# Held for the life of the process. Module-level on purpose: a caller that has
# to keep a handle alive will one day drop it, and an unused local is also the
# kind of thing pyflakes rejects, so the module owns it instead.
_HELD = []


def serialise(name: str = "verify") -> None:
    """A whole-host lock so two verification pipelines cannot overlap.

    Overlapping pipelines are how several single-threaded processes became six
    cores of load, and nothing prevented it. The lock is released when this
    process exits.
    """
    import fcntl
    path = Path(os.environ.get("TMPDIR", "/tmp")) / f".qwen36-{name}.lock"
    try:
        fh = open(path, "w", encoding="utf-8")
    except OSError:
        return          # fail open: an unwritable tmp is not a reason to stop
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        sys.exit(
            f"another {name} pipeline already holds {path}.\n"
            "  Running two at once is what produced the CPU bursts a\n"
            "  benchmark on this host recorded as contention. Wait for it."
        )
    fh.write(f"{os.getpid()}\n")
    fh.flush()
    _HELD.append(fh)


def _cpu_totals() -> tuple[int, int]:
    """(busy_jiffies, total_jiffies) from /proc/stat's aggregate line."""
    with open("/proc/stat", encoding="utf-8") as fh:
        parts = [int(x) for x in fh.readline().split()[1:]]
    total = sum(parts)
    idle = parts[3] + (parts[4] if len(parts) > 4 else 0)   # idle + iowait
    return total - idle, total


def _proc_jiffies() -> dict[int, tuple[int, str, list[str]]]:
    """{pid: (utime+stime, cmdline, argv)} for every readable process.

    argv as well as the joined string, because picking the benchmark's root by
    searching that string is what `_benchmark_name` exists to avoid.
    """
    out = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            tail = stat[stat.rfind(")") + 1:].split()
            jiffies = int(tail[11]) + int(tail[12])          # utime, stime
            raw = (entry / "cmdline").read_bytes()
            argv = [a for a in raw.decode("utf-8", "replace").split("\0") if a]
            cmd = " ".join(argv) or f"[{entry.name}]"
        except (OSError, IndexError, ValueError):
            continue
        out[int(entry.name)] = (jiffies, cmd, argv)
    return out


def _starttime(pid: int) -> int | None:
    """Field 22 of `/proc/<pid>/stat`, in clock ticks since boot.

    A pid is reused; the pair (pid, starttime) is not, within a boot. Checking
    it each tick is what stops the sampler from following a stranger that
    happened to inherit the number the benchmark had.

    Index 19 into the tail after the last `)`: the tail begins at field 3, the
    state letter, so field 22 is 22 - 3 = 19. Verified against
    `/proc/stat`'s `btime` plus this value over `SC_CLK_TCK`.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    tail = stat[stat.rfind(")") + 1:].split()
    try:
        return int(tail[19])
    except (IndexError, ValueError):
        return None


def sample(out_path: str, interval: float = 5.0,
           root_pid: int | None = None) -> None:
    """Record host load beside the GPU trace, until killed.

    `bench/gpu_telemetry.sh` queries `nvidia-smi` and nothing else, so no run
    in this repository before 2026-08-27 can say what else its bench host was
    doing. ERRATA A16 turns on "nothing recorded distinguishes them", and host
    load was one of the things not recorded.

    Attribution is by DESCENT from the benchmark process, not by name. The
    sibling repository matched on names and got it wrong in both directions at
    once: it counted a run's own `nvidia-smi` sampler as competition, and never
    counted an analysis script that genuinely was competing.

    But descent needs a root, and picking that root by searching every cmdline
    for `"llama-server"` put the naming problem back one level: a `llama-server`
    belonging to somebody else became a root, its whole tree counted as OURS,
    and the interference this file exists to see went invisible in the one
    reading that mattered. `root_pid` is the answer -- the driver knows which
    process it started -- and it is checked against its start time each tick,
    because a pid is reused and the pair is not.

    Without `root_pid` the fallback picks roots by POSITION, through the same
    `_benchmark_name` the refusal path uses, so an editor or a grep naming the
    binary is not a root. It still cannot tell a stranger's real server from
    ours; the `attribution` column says which of the two modes produced the row,
    so nothing has to guess afterwards.
    """
    import time
    out = Path(out_path).expanduser()
    new = not out.exists()
    with open(out, "a", encoding="utf-8", buffering=1) as fh:
        if new:
            fh.write("wall_iso,busy_pct,load1,ncpu,own_pct,other_pct,"
                     "top_other_pct,attribution,top_other\n")
        prev_busy, prev_total = _cpu_totals()
        prev_proc = _proc_jiffies()
        root_start = _starttime(root_pid) if root_pid is not None else None
        if root_pid is not None and root_start is None:
            raise SystemExit(f"host_guard --sample: pid {root_pid} is not "
                             f"running, so there is no tree to attribute to")
        ncpu = os.cpu_count() or 1
        while True:
            time.sleep(interval)
            busy, total = _cpu_totals()
            proc = _proc_jiffies()
            d_total = total - prev_total
            if d_total <= 0:
                prev_busy, prev_total, prev_proc = busy, total, proc
                continue
            busy_pct = 100.0 * (busy - prev_busy) / d_total

            # which pids belong to the benchmark, by descent from a root
            # the driver named, or from a positional match if it named none
            if root_pid is not None:
                alive = _starttime(root_pid) == root_start
                roots = {root_pid} if alive else set()
                mode = "root-pid" if alive else "root-gone"
            else:
                roots = {pid for pid, (_, _c, argv) in proc.items()
                         if _benchmark_name(argv)}
                mode = "by-name"
            own_tree = _descendants(roots) if roots else set()

            own = other = 0.0
            top_pct, top_cmd = 0.0, ""
            for pid, (jif, cmd, _argv) in proc.items():
                d = jif - prev_proc.get(pid, (jif, "", []))[0]
                if d <= 0:
                    continue
                pct = 100.0 * d * ncpu / d_total
                if pid in own_tree:
                    own += pct
                else:
                    other += pct
                    if pct > top_pct:
                        top_pct, top_cmd = pct, cmd[:60].replace(",", " ")
            fh.write(f"{_now()},{busy_pct:.1f},{_load1():.2f},{ncpu},"
                     f"{own:.1f},{other:.1f},{top_pct:.1f},{mode},{top_cmd}\n")
            prev_busy, prev_total, prev_proc = busy, total, proc


def _descendants(roots: set[int]) -> set[int]:
    """Every pid whose parent chain reaches one of `roots`, plus the roots."""
    children: dict[int, set[int]] = {}
    try:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text(encoding="utf-8")
            except OSError:
                continue
            tail = stat[stat.rfind(")") + 1:].split()
            if len(tail) < 2:
                continue
            children.setdefault(int(tail[1]), set()).add(int(entry.name))
    except OSError:
        return set(roots)
    seen, stack = set(), list(roots)
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        stack.extend(children.get(pid, ()))
    return seen


def _now() -> str:
    import datetime
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _load1() -> float:
    try:
        with open("/proc/loadavg", encoding="utf-8") as fh:
            return float(fh.readline().split()[0])
    except (OSError, ValueError, IndexError):
        return -1.0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--sample":
        # bench/host_guard.py --sample <out.csv> [interval] [--root-pid N]
        _argv = sys.argv[2:]
        _root = None
        if "--root-pid" in _argv:
            _i = _argv.index("--root-pid")
            try:
                _root = int(_argv[_i + 1])
            except (IndexError, ValueError):
                raise SystemExit("--root-pid needs an integer pid")
            del _argv[_i:_i + 2]
        if not _argv:
            raise SystemExit("bench/host_guard.py --sample <out.csv> "
                             "[interval] [--root-pid N]")
        sample(_argv[0], float(_argv[1]) if len(_argv) > 1 else 5.0,
               root_pid=_root)
    else:
        lock = lock_held()
        running = measuring_processes()
        print(f"lock:      {lock or 'none'}")
        print(f"benchmark: {'; '.join(running) if running else 'none running'}")
        print(f"verdict:   {'would refuse' if (lock or running) else 'clear to run'}")
