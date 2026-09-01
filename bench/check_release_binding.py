#!/usr/bin/env python3
"""Does a release tag name the commit whose verifier and manifest it publishes?

`raw-evidence-2026-08-27` points at `de6f33bf`, which is this pull request's
BASE. Its assets kept being added to -- a third tranche for run W among them --
while its source revision stayed behind, so the tarball a reader downloads from
that tag holds neither the current verifier, nor the current manifest, nor run
W's data, and yet shares one release identity with the assets that do. That
breaks the chain the whole evidence apparatus rests on:

    source revision <-> verifier <-> manifest <-> assets <-> published claims

The published tag is deliberately NOT retargeted: silently moving a tag people
may already have fetched destroys the one property a tag has. A final versioned
release is cut at the exact dataset-and-verifier commit instead, and this checks
that it really is that commit.

    check_release_binding.py TAG [REF]

Compares, between TAG's commit and REF (default HEAD), the files that decide
what the evidence means. Any difference is a release that does not publish the
tree it claims to.
"""
from __future__ import annotations

import subprocess
import sys

BOUND = (
    "v4_audit_2026_08_25/EVIDENCE_MANIFEST.sha256",
    "v4_audit_2026_08_25/EVIDENCE_REGISTRY.json",
    "v4_audit_2026_08_25/RUN_REGISTRY.json",
    "analysis/verify_claims.py",
    "analysis/rederive_from_logs.py",
    "bench/preflight_tar.py",
)


def blob(ref: str, path: str) -> str | None:
    r = subprocess.run(["git", "rev-parse", f"{ref}:{path}"],
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def main() -> int:
    if not 2 <= len(sys.argv) <= 3:
        print(__doc__, file=sys.stderr)
        return 2
    tag, ref = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "HEAD")
    tc = subprocess.run(["git", "rev-parse", f"{tag}^{{commit}}"],
                        capture_output=True, text=True)
    if tc.returncode:
        print(f"FAIL: {tag} is not a tag in this repository", file=sys.stderr)
        return 1
    tag_commit = tc.stdout.strip()
    bad = []
    for path in BOUND:
        a, b = blob(tag_commit, path), blob(ref, path)
        if a is None:
            bad.append(f"{path} does not exist at {tag}")
        elif a != b:
            bad.append(f"{path} differs: {tag} has {a[:12]}, {ref} has "
                       f"{(b or '(absent)')[:12]}")
    for x in bad:
        print(f"FAIL: {x}", file=sys.stderr)
    if bad:
        print(f"FAIL: {tag} -> {tag_commit[:12]} does not publish the tree it "
              f"names; cut the release at the dataset-and-verifier commit",
              file=sys.stderr)
        return 1
    print(f"{tag} -> {tag_commit[:12]}: verifier, manifest and both registries "
          f"are the ones at {ref}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
