#!/usr/bin/env python3
"""Publish `PULL_REQUEST.md` as pull request #2's body, and prove it landed.

    python tools/publish_pr_body.py            # check only, the default
    python tools/publish_pr_body.py --write    # publish, then read back

READ-ONLY unless `--write` is given. It used to be the other way round --
`check_only = "--check" in sys.argv` -- so any typo selected the PATCH path in
a tool whose target is a public pull request body. Unknown options are refused
rather than ignored.

Why this is a script and not a one-liner
----------------------------------------
It used to be a one-liner in `PULL_REQUEST.md`'s own header comment:

    re.sub(r'^<!--.*?-->\\s*', '', text, flags=re.S)

The non-greedy `.*?` stops at the first `-->` in the file, and that command
contains a literal `-->` inside its own pattern. So the strip ended in the
middle of the command and the rest of it was published as the opening of the
pull request body. It was live on GitHub from 2026-08-27 until the fourth
review found it.

Two things made that survive a check that was supposed to catch it. The
comparison ran the same broken regex over both sides, so they agreed - the
checker shared the defect with the thing it checked. And nothing ever read the
body back from GitHub; the only comparison was file-against-file.

So: the header is parsed line by line, a closing marker is a line that is
exactly `-->` after trimming, and after publishing the body is fetched back and
compared byte for byte.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "PULL_REQUEST.md"
REPO = "thc1006/qwen3.6-speculative-decoding-rtx3090"
PR = 2


def reflow(text: str) -> str:
    """Join the lines of each paragraph, because a pull request body is not a file.

    `PULL_REQUEST.md` is wrapped at eighty columns so a diff of it is readable
    line by line. GitHub Flavored Markdown renders issue and pull request bodies
    with newlines PRESERVED -- a single newline inside a paragraph becomes
    `<br>` -- so those wraps are not invisible there the way they are in a
    README. Rendering the file through GitHub's own markdown endpoint on
    2026-09-01 counted 412 of them, which is what the body looked like to anyone
    reading the pull request.

    So the file keeps its wrapping and the body is reflowed on the way out.
    Three things are structure rather than prose and survived a first version
    that did not know it, each caught by rendering the result and counting:

    INDENTATION. A table row indented inside a list item came out flush left,
    which ends the list item and starts a second table. Rows went from 48 to 54.
    A joined block now keeps the indentation of its own first line.

    A NUMBER IS NOT A LIST MARKER. The test for an ordered item was a leading
    digit with a `.` anywhere in the first four characters, so a wrapped line
    beginning `82.079 + 19.266` or `3.3 pp` or `1.96;` was read as opening a new
    item and the sentence was cut there. Six of the sixteen breaks left after
    the first fix were this, and the risk was never the break: it was prose
    being parsed as a list. A marker is digits then `.` or `)` then a space.

    BLOCKQUOTES WRAP TOO. `>` marks the line but what follows it is a paragraph,
    and GFM breaks inside a quote exactly as it does outside one. The marker is
    held aside, the content joined, and the marker put back.
    """
    marker = re.compile(r"^\d+[.)]\s")
    quote = re.compile(r"^(?:>\s?)+")
    out: list[str] = []
    para: list[str] = []
    prefix = ""

    def flush() -> None:
        nonlocal prefix
        if para:
            head = para[0]
            indent = head[:len(head) - len(head.lstrip())]
            out.append(prefix + indent + " ".join(x.strip() for x in para))
            para.clear()
        prefix = ""

    fenced = False
    for line in text.split("\n"):
        s = line.rstrip()
        st = s.lstrip()
        if st.startswith("```"):
            flush()
            fenced = not fenced
            out.append(s)
            continue
        if fenced:
            out.append(s)
            continue
        q = ""
        m = quote.match(st)
        if m:
            q = m.group(0)
            s = st[len(q):]
            st = s.lstrip()
            if not st:                       # a blank line inside the quote
                flush()
                out.append(q.rstrip())
                continue
            if para and prefix != q:         # a quote at a different depth
                flush()
        elif prefix:                         # the quote ended
            flush()
        if not st:
            flush()
            out.append("")
            continue
        # line-oriented: a table row, a heading and a rule are each one line,
        # and nothing continues them
        if st.startswith(("|", "#")) or st in ("---", "***", "___"):
            flush()
            out.append(q + s)
            continue
        # a list item opens a block that its wrapped lines continue
        if st[:2] in ("- ", "* ", "+ ") or marker.match(st):
            flush()
            prefix = q
            para.append(s)
            continue
        if not para:
            prefix = q
        para.append(s)
    flush()
    return "\n".join(out).strip() + "\n"


def strip_header(text: str) -> str:
    """Everything after the leading HTML comment, which must be well formed.

    Line based on purpose: `-->` inside a line is data, only a line that *is*
    `-->` closes the comment. A file with no leading comment is returned as is;
    a file that opens one and never closes it is an error, not a silent pass.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "<!--":
        return text.strip()
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "-->":
            return "\n".join(lines[i + 1:]).strip()
    raise SystemExit(f"{SOURCE.name}: header comment opens and never closes")


def _token() -> str:
    """The token for github.com, not whichever host is listed first.

    This returned the first `oauth_token` in the file. `hosts.yml` can hold
    several hosts -- a GitHub Enterprise entry above github.com is the ordinary
    case -- and this would then send that host's credential to api.github.com.
    """
    hosts = Path.home() / ".config" / "gh" / "hosts.yml"
    host, tokens = None, {}
    for line in hosts.read_text(encoding="utf-8").splitlines():
        if line[:1] not in (" ", "\t", "#", "") and line.rstrip().endswith(":"):
            host = line.strip().rstrip(":")
        elif "oauth_token" in line and host:
            tokens[host] = line.split()[-1].strip()
    if "github.com" not in tokens:
        raise SystemExit(f"no github.com oauth_token in {hosts}; hosts found: "
                         f"{sorted(tokens) or '(none)'}")
    return tokens["github.com"]


def _api(method: str, payload: dict | None = None) -> dict:
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/pulls/{PR}",
        method=method,
        data=json.dumps(payload).encode() if payload else None,
        headers={"Authorization": f"Bearer {_token()}",
                 "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as fh:
        return json.load(fh)


def main() -> None:
    # argparse, and READ-ONLY by default. `check_only = "--check" in sys.argv`
    # meant a typo -- `--chek` -- silently selected the PATCH path: the default
    # for a tool that writes to a public pull request was to write.
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true",
                   help="compare the published body with the source (default)")
    g.add_argument("--write", action="store_true",
                   help="PATCH the pull request body, then read it back and compare")
    args = ap.parse_args()

    # reflowed, because GFM renders a newline inside a paragraph of a pull
    # request body as <br>: the file's eighty-column wrapping showed up as 399
    # line breaks on the pull request page
    # Normalised exactly as the read-back below is, and for the reason this
    # module exists: the two sides of a comparison must be treated the same.
    # `got` was stripped and `want` was not, `reflow` ends its output with a
    # newline, and so publishing succeeded and then reported a one character
    # mismatch against the body it had just written. The docstring above
    # records the mirror image of this, a checker that shared a defect with the
    # thing it checked; different normalisation on the two sides is the same
    # error with the sign flipped.
    want = reflow(strip_header(SOURCE.read_text(encoding="utf-8"))).replace(
        "\r\n", "\n").strip()
    if not want:
        raise SystemExit(f"{SOURCE.name}: nothing to publish after the header")

    if args.write:
        _api("PATCH", {"body": want})

    got = (_api("GET").get("body") or "").replace("\r\n", "\n").strip()
    if got == want:
        # characters and bytes, both: the body carries 36 non-ASCII characters
        # (28 of them U+2212) and GitHub's API reports its length in bytes, so
        # printing one count labelled as the other invites exactly the
        # comparison that fails for no reason. Comparing the strings IS a
        # byte-for-byte comparison, because UTF-8 encodes them one way only.
        print(f"live body matches {SOURCE.name}: {len(want)} characters, "
              f"{len(want.encode('utf-8'))} bytes")
        return

    # say where, not just that. The index is into characters, which is what
    # the slices below are, so it is not called a byte offset.
    for i, (a, b) in enumerate(zip(want, got)):
        if a != b:
            raise SystemExit(
                f"live body differs from {SOURCE.name} at character {i}\n"
                f"  want: {want[max(0, i - 40):i + 40]!r}\n"
                f"  got:  {got[max(0, i - 40):i + 40]!r}")
    raise SystemExit(
        f"live body differs in length: file {len(want)} characters, "
        f"live {len(got)}\n"
        f"  tail of the shorter: {(want if len(want) < len(got) else got)[-120:]!r}")


if __name__ == "__main__":
    main()
