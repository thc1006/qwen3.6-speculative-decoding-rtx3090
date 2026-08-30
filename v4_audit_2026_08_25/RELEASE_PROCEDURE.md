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

    TAG=raw-evidence-2026-08-30-v4.2
    HEAD_SHA=$(git rev-parse HEAD)

    # 1. the tree must be green at that commit, both gates
    python3 analysis/verify_claims.py
    python3 -m unittest discover -s tests -p 'test_*.py'

    # 2. the tag goes on that commit and nothing later
    git tag -s "$TAG" -m "evidence and verifier at $HEAD_SHA" "$HEAD_SHA"
    git push origin "$TAG"

    # 3. the binding is checked, not assumed
    python3 bench/check_release_binding.py "$TAG"

    # 4. and the release is created from that tag, immutable, with notes that
    #    list every asset with its bytes and SHA-256, the manifest's own
    #    SHA-256, the verifier commit, the exact log and trace counts, and
    #    which data can only be integrity-checked rather than re-derived
    gh release create "$TAG" --target "$HEAD_SHA" --verify-tag \
       --title "Evidence and verifier, v4.2" --notes-file RELEASE_NOTES_v4.2.md

## What the notes must contain

Not a description of the assets — the assets themselves, enumerated:

- every asset, its size in bytes and its SHA-256, all six of them
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
