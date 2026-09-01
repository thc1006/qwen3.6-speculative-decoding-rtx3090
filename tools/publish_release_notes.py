#!/usr/bin/env python3
"""Publish `RELEASE_NOTES_v4.2.md` as the release body, and prove it landed.

    python tools/publish_release_notes.py            # check only, the default
    python tools/publish_release_notes.py --write    # publish, then read back

Why this exists
---------------
A release body is rendered by the same GitHub Flavored Markdown as an issue or
a pull request body: newlines are PRESERVED, so a paragraph wrapped at eighty
columns for a readable diff becomes a paragraph full of `<br>`. `PULL_REQUEST.md`
had 412 of them and this repository has a tool and a commit about fixing it.

The release was then created with `gh release create --notes-file
RELEASE_NOTES_v4.2.md`, which hands GitHub the file verbatim, and the published
body rendered with **29** line breaks. The same defect, on the surface nobody had
written a tool for.

The file keeps its wrapping, because a diff of it is read line by line, and the
body is reflowed on the way out. `reflow` is imported from `publish_pr_body`
rather than copied: two implementations of one rule are two things to drift, and
the rule here is subtle enough that the first version of it dropped the
indentation off table rows inside list items.

READ-ONLY unless `--write` is given, for the reason the other publisher gives:
its default used to be to write, and a typo in the flag selected the PATCH path
on a public artefact.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from publish_pr_body import reflow, strip_header, _token   # noqa: E402

SOURCE = ROOT / "RELEASE_NOTES_v4.2.md"
REPO = "thc1006/qwen3.6-speculative-decoding-rtx3090"
TAG = "raw-evidence-2026-08-31-v4.2"


def _api(url: str, method: str = "GET", payload: dict | None = None) -> dict:
    req = urllib.request.Request(
        url, method=method,
        data=json.dumps(payload).encode() if payload else None,
        headers={"Authorization": f"Bearer {_token()}",
                 "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as fh:
        return json.load(fh)


def _breaks(text: str) -> int:
    """How many `<br>` GitHub's own renderer puts in this body.

    Asked of GitHub rather than counted from the source, because the question is
    what a reader sees and only the renderer answers that. The pull request body
    was "fixed" once against a local guess and was still full of breaks.
    """
    req = urllib.request.Request(
        "https://api.github.com/markdown", method="POST",
        data=json.dumps({"text": text, "mode": "gfm", "context": REPO}).encode(),
        headers={"Authorization": f"Bearer {_token()}",
                 "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as fh:
        return fh.read().decode().count("<br")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0] if __doc__ else None)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true",
                   help="compare the published body with the source (default)")
    g.add_argument("--write", action="store_true",
                   help="PATCH the release body, then read it back and compare")
    args = ap.parse_args()

    want = reflow(strip_header(SOURCE.read_text(encoding="utf-8")))
    rel = _api(f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}")
    live = (rel.get("body") or "").replace("\r\n", "\n")

    if not args.write:
        n = _breaks(live)
        same = live.strip() == want.strip()
        print(f"  live body: {len(live.splitlines())} lines, {n} <br> when rendered")
        print(f"  matches the reflowed source: {same}")
        if not same:
            for i, (a, b) in enumerate(zip(live, want)):
                if a != b:
                    print(f"  first difference at character {i}")
                    print(f"    live: {live[max(0, i - 40):i + 40]!r}")
                    print(f"    want: {want[max(0, i - 40):i + 40]!r}")
                    break
        raise SystemExit(0 if same and n == 0 else 1)

    _api(f"https://api.github.com/repos/{REPO}/releases/{rel['id']}",
         method="PATCH", payload={"body": want})
    back = (_api(f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}")
            .get("body") or "").replace("\r\n", "\n")
    n = _breaks(back)
    ok = back.strip() == want.strip()
    print(f"  published: {len(want.splitlines())} lines")
    print(f"  read back identical: {ok}")
    print(f"  rendered line breaks: {n}")
    raise SystemExit(0 if ok and n == 0 else 1)


if __name__ == "__main__":
    main()
