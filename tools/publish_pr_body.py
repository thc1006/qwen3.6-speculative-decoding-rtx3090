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
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "PULL_REQUEST.md"
REPO = "thc1006/qwen3.6-speculative-decoding-rtx3090"
PR = 2


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

    want = strip_header(SOURCE.read_text(encoding="utf-8"))
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
