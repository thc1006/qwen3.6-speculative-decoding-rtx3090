# Cutting the final evidence release

`raw-evidence-2026-08-27` is a mutable release whose tag points at
`de6f33bf71f1fe0fbac4d213b6aaf6a072da529d` — this branch's **base**. Assets kept
being added to it, a third tranche for run W among them, while its source
revision stayed where it was. A reader who downloads the source tarball from
that tag gets a tree that contains neither the current verifier, nor the current
manifest, nor run W's data: `bench/check_release_binding.py` reports that the
whole `v4_audit_2026_08_25` directory does not exist at that commit.

That breaks the chain everything here rests on:

    source revision <-> verifier <-> manifest <-> assets <-> published claims

## What is deliberately not done

The published tag is **not** retargeted. Silently moving a tag people may
already have fetched destroys the one property a tag has, and the release notes
say where it points and why it is left there.

## What is done instead

A final versioned release, cut at the exact commit that carries the dataset and
the verifier together, at the end rather than the middle:

    TAG=v4.2
    HEAD_SHA=$(git rev-parse HEAD)

    # 1. the tree must be green at that commit, all four gates
    python3 analysis/verify_claims.py
    python3 -m unittest discover -s tests -p 'test_*.py'
    python3 tests/mutate.py
    python3 tests/data_mutate.py

    # 2. the tag goes on that commit and nothing later. ANNOTATED, not signed:
    #    this repository has no signing key configured and none of its five
    #    existing tags is signed, so `-s` would have failed here and `-a` is
    #    what it has always actually done. Signing needs a key registered with
    #    GitHub as a signing key, which is an account action, and the release
    #    notes say the tag is unsigned rather than leaving a reader to check.
    git tag -a "$TAG" -m "evidence and verifier at $HEAD_SHA" "$HEAD_SHA"
    git push origin "$TAG"

    # 3. the binding is checked, not assumed
    python3 bench/check_release_binding.py "$TAG"

    # 4. and the release is created from that tag, immutable, with notes that
    #    list every asset with its bytes and SHA-256, the manifest's own
    #    SHA-256, the verifier commit, the exact log and trace counts, and
    #    which data can only be integrity-checked rather than re-derived
    # $ASSETS is wherever the eight archives are held on the operator's
    # machine. They were split across two directories while the fourth tranche
    # was being built, and one of those was on a tmpfs that filled: the
    # archives are 1.0 GB, the release is the only durable copy of them once it
    # exists, and until then they are worth keeping somewhere that survives a
    # reboot. Their digests are in RELEASE_NOTES_v4.2.md and are checked
    # against the files before this runs.
    #    NOT `--notes-file`. A release body is rendered by the same GFM as an
    #    issue body: newlines are preserved, so the file's eighty-column
    #    wrapping becomes `<br>`. Passing the file verbatim on 2026-09-01
    #    published a body with 29 of them, which is the defect this repository
    #    already has a tool and a commit about, on the one surface that had no
    #    tool. Create the release with a placeholder body and set the real one
    #    with `tools/publish_release_notes.py --write`, which reflows it, reads
    #    it back and renders it through GitHub's own markdown endpoint.
    #    No `--title` either. The name is the notes file's own first heading,
    #    set by the same tool that sets the body, so the two cannot disagree.
    #    Typed at the shell it read "Evidence and verifier, v4.2", the only one
    #    of this repository's seven releases that does not lead with its
    #    version, while the other six read "v1.0 ..." through "v3.0 ...".
    gh release create "$TAG" --target "$HEAD_SHA" --verify-tag \
       --title "$TAG" --notes "see below" \
       "$ASSETS"/raw_logs.tar.zst "$ASSETS"/raw_logs_20260827.tar.zst \
       "$ASSETS"/raw_logs_20260828.tar.zst "$ASSETS"/raw_logs_20260831.tar.zst \
       "$ASSETS"/telemetry.tar.zst "$ASSETS"/telemetry_20260827.tar.zst \
       "$ASSETS"/telemetry_20260828.tar.zst "$ASSETS"/telemetry_20260831.tar.zst

    # 5. the body, reflowed. The file keeps its wrapping because its diff is
    #    read line by line; the body is reflowed on the way out and the tool
    #    proves it landed: it reads the published body back and asks GitHub's
    #    renderer how many line breaks are in it. Zero, or it exits non-zero.
    python3 tools/publish_release_notes.py --write

## Why all eight assets, and not two

`raw-evidence-2026-08-27` keeps its six and is not retargeted. The v4.2 release
carries all eight so that one tag publishes one complete evidence set: a reader
who fetches it needs no second identity to verify the manifest, which is the
property the paragraph at the top of this file says was broken.

## What the notes must contain

Not a description of the assets — the assets themselves, enumerated:

- every asset, its size in bytes and its SHA-256, all eight of them: run W2's
  logs are a fourth tranche, published rather than left pending
- the SHA-256 of `EVIDENCE_MANIFEST.sha256`
- the verifier commit, which is the tag's own commit
- exact counts: server logs, telemetry traces, tranches, run directories
- **what is only integrity-checked**: the primary per-request benchmark JSON is
  not re-derived from the logs, because the logs carry no per-request timing
  rows. `EVIDENCE_REGISTRY.json` records that distinction and the four
  log-derived audit files that are re-derived.

## The check that stops this recurring

`bench/check_release_binding.py TAG` compares the manifest, both registries and
the three verifier scripts between the tag's commit and `HEAD`. A release whose
tag does not publish the tree it names fails it. Run it before creating the
release and again after, and in CI on any tag matching `raw-evidence-*`.
