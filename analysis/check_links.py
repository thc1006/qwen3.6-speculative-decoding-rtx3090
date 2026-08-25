"""Check every relative link and heading anchor in the repository's Markdown.

Anchor generation follows github-slugger: lowercase, drop every character that
is not a word character (which INCLUDES the underscore), a hyphen, or a space,
then turn spaces into hyphens. Getting the underscore rule wrong silently
"fixes" links that were already correct, so it is spelled out here.

Run: python analysis/check_links.py   (exits non-zero if anything is broken)
"""
import pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def slug(heading: str) -> str:
    s = heading.strip().lower()
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)   # [text](url) -> text
    s = re.sub(r"[`*]", "", s)                        # code ticks and emphasis
    s = "".join(c for c in s if c.isalnum() or c in "-_ ")
    return s.replace(" ", "-")


def main() -> None:
    mds = [p for p in ROOT.rglob("*.md") if ".git" not in p.parts]
    anchors = {p: {slug(m.group(2)) for m in
                   re.finditer(r"^(#{1,6})\s+(.*)$", p.read_text(encoding="utf-8"), re.M)}
               for p in mds}
    bad = []
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
            if anc:
                if target not in anchors:
                    bad.append(f"{p.relative_to(ROOT)}: anchor into non-markdown {tgt}#{anc}")
                elif anc not in anchors[target]:
                    bad.append(f"{p.relative_to(ROOT)} -> {target.relative_to(ROOT)}#{anc}")
    total = sum(len(v) for v in anchors.values())
    print(f"  {len(mds)} markdown files, {total} headings")
    print("  broken:", "\n    ".join([""] + bad) if bad else "none")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
