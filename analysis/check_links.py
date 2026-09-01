"""Check every reference a fresh clone has to be able to resolve.

That is three kinds: relative links between the Markdown files, heading anchors
inside them, and the sibling modules the tracked scripts import.

Anchor generation follows github-slugger: lowercase, drop every character that
is not a word character (which INCLUDES the underscore), a hyphen, or a space,
then turn spaces into hyphens. Getting the underscore rule wrong silently
"fixes" links that were already correct, so it is spelled out here.

EXISTING ON THIS DISK IS NOT THE TEST. `analysis/figstyle.py` was imported by
both plotting scripts and never added to the index, and the coverage
attestations the checker reads were a directory git had never heard of. Both
resolved here and would have vanished in the checkout CI makes, which is where
the charts job runs the script that imports one of them. So every target is
required to be tracked as well as present, and a target that is present but
untracked is reported as its own kind of failure rather than folded into the
missing ones.

Run: python analysis/check_links.py   (exits non-zero if anything is broken)
"""
import pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def slug(heading: str) -> str:
    s = heading.strip().lower()
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)   # [text](url) -> text
    s = re.sub(r"[`*]", "", s)                        # code ticks and emphasis
    s = "".join(c for c in s if c.isalnum() or c in "-_ ")
    return s.replace(" ", "-")


def tracked_paths():
    """Every path in the index, plus every directory on the way to one.

    A check that cannot run must not read as green, so a git that will not
    answer is a failure here and not a skipped step.
    """
    r = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("  git ls-files failed, so the tracking check could not run:")
        print("   ", r.stderr.strip()[:200])
        sys.exit(1)
    files = {f for f in r.stdout.split("\0") if f}
    dirs = set()
    for f in files:
        d = pathlib.PurePosixPath(f).parent
        while str(d) not in (".", ""):
            dirs.add(str(d))
            d = d.parent
    return files, dirs


def main() -> None:
    files, dirs = tracked_paths()

    def is_tracked(target: pathlib.Path) -> bool:
        try:
            rel = target.relative_to(ROOT).as_posix()
        except ValueError:
            return True          # outside the repository; not ours to police
        return rel in files or rel in dirs

    mds = [p for p in ROOT.rglob("*.md") if ".git" not in p.parts]
    anchors = {p: {slug(m.group(2)) for m in
                   re.finditer(r"^(#{1,6})\s+(.*)$", p.read_text(encoding="utf-8"), re.M)}
               for p in mds}
    bad, untracked = [], []
    for p in mds:
        txt = p.read_text(encoding="utf-8")
        for m in re.finditer(r"\]\(([^)\s]*?)(#[^)\s]+)?\)", txt):
            tgt, anc = m.group(1), (m.group(2) or "")[1:]
            if tgt.startswith(("http", "mailto:", "sandbox:")):
                continue
            target = p if tgt == "" else (p.parent / tgt).resolve()
            if not target.exists():
                bad.append(f"{p.relative_to(ROOT)}: missing file {tgt}")
                continue
            if not is_tracked(target):
                untracked.append(f"{p.relative_to(ROOT)}: {tgt} is not tracked")
            if anc:
                if target not in anchors:
                    bad.append(f"{p.relative_to(ROOT)}: anchor into non-markdown {tgt}#{anc}")
                elif anc not in anchors[target]:
                    bad.append(f"{p.relative_to(ROOT)} -> {target.relative_to(ROOT)}#{anc}")

    # the same failure in the form the charts job hits: a sibling module that a
    # tracked script imports and the index does not hold
    imp = re.compile(r"^\s*(?:import|from)\s+([a-z_][a-z0-9_]*)", re.M)
    for rel in sorted(f for f in files if f.endswith(".py")):
        src = ROOT / rel
        for m in imp.finditer(src.read_text(encoding="utf-8")):
            sib = src.parent / f"{m.group(1)}.py"
            if sib.exists() and not is_tracked(sib):
                untracked.append(
                    f"{rel}: imports {sib.relative_to(ROOT)}, which is not tracked")

    total = sum(len(v) for v in anchors.values())
    print(f"  {len(mds)} markdown files, {total} headings, {len(files)} tracked paths")
    print("  broken:", "\n    ".join([""] + bad) if bad else "none")
    print("  present but untracked:",
          "\n    ".join([""] + untracked) if untracked else "none")
    sys.exit(1 if bad or untracked else 0)


if __name__ == "__main__":
    main()
