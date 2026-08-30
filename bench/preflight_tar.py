#!/usr/bin/env python3
"""Refuse an evidence archive before extracting it, not after.

The evidence workflow ran `tar -xf` and then checked the extracted files against
`EVIDENCE_MANIFEST.sha256`. That proves every file the manifest NAMES is intact.
It does not reject what the manifest does not name: a member with an absolute
path, one with `..` in it, a symlink or hardlink pointing outside the tree, a
device node, a duplicated member whose second copy silently overwrites the
first, or simply an extra file that lands in the working tree and is never
looked at. The pinned digests make that unlikely for the archives that exist;
the verifier is still not closed-world, and that is a property of the verifier
rather than of today's archives.

    preflight_tar.py MANIFEST ARCHIVE [ARCHIVE ...]

Every member must be a regular file, with a normalized relative path, appearing
once, and named by the manifest. Anything else is reported and exits non-zero,
before a single byte is written to disk.
"""
from __future__ import annotations

import posixpath
import sys
import tarfile


def manifest_paths(path: str) -> set[str]:
    out = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            # `sha256sum` format: "<64 hex>  <path>", the separator being two
            # spaces or a space and a binary marker
            parts = line.split(None, 1)
            if len(parts) == 2:
                out.add(parts[1].lstrip("*").strip())
    return out


def check(archive: str, named: set[str]) -> list[str]:
    bad, seen = [], set()
    # STREAM mode. The default seeks, and the workflow hands this a process
    # substitution -- `<(unzstd -c ...)` -- which is a FIFO: `Illegal seek`,
    # on the first run this code ever had. Local tests used real files and
    # never touched the path CI uses. `r|*` reads sequentially and detects
    # compression, so it works for both.
    with tarfile.open(archive, mode="r|*") as tf:
        # Iterate, not getmembers(): a stream cannot be rewound to build the
        # full list first, and iterating is what a sequential reader supports.
        for m in tf:
            n = m.name
            if not m.isfile():
                kind = ("symlink" if m.issym() else "hardlink" if m.islnk()
                        else "directory" if m.isdir() else "special")
                if m.isdir():
                    continue          # directories are harmless and expected
                bad.append(f"{archive}: {n!r} is a {kind}")
                continue
            if posixpath.isabs(n) or n.startswith("/"):
                bad.append(f"{archive}: {n!r} is an absolute path")
                continue
            norm = posixpath.normpath(n)
            if norm.startswith("..") or "/../" in f"/{norm}/":
                bad.append(f"{archive}: {n!r} escapes the extraction root")
                continue
            if norm != n.rstrip("/"):
                bad.append(f"{archive}: {n!r} is not normalized ({norm!r})")
                continue
            if n in seen:
                bad.append(f"{archive}: {n!r} appears twice; the second would "
                           f"overwrite the first")
                continue
            seen.add(n)
            if named and not any(p == n or p.endswith("/" + n) or n.endswith("/" + p)
                                 for p in named):
                bad.append(f"{archive}: {n!r} is not named by the manifest")
    return bad


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    named = manifest_paths(sys.argv[1])
    if not named:
        print(f"FAIL: {sys.argv[1]} names no files", file=sys.stderr)
        return 1
    bad = []
    for a in sys.argv[2:]:
        bad += check(a, named)
    for b in bad[:40]:
        print(f"FAIL: {b}", file=sys.stderr)
    if len(bad) > 40:
        print(f"FAIL: and {len(bad) - 40} more", file=sys.stderr)
    if not bad:
        print(f"{len(sys.argv) - 2} archive(s): every member is a normalized "
              f"relative regular file named by the manifest")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
