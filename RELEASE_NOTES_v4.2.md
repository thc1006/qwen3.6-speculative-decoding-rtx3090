# v4.2 · Evidence and verifier at one commit, every published number probed

Tag `v4.2`. Named here because a release note that does
not say which release it belongs to leaves a reader holding the file with no
way to tell, and this one did not.

The v4 audit's dataset and the code that checks it, published together at one
commit. `raw-evidence-2026-08-27` is left exactly where it points, which is this
branch's base: retargeting a tag people may already have fetched destroys the
one property a tag has. That release's six assets kept being added to while its
source revision stayed behind, so the tarball a reader downloaded from it held
neither the current verifier nor the current manifest. This release exists so
that one tag names one tree and carries the whole evidence set for it.

**The tag is annotated and NOT signed.** This repository has no signing key
configured and none of its five earlier tags is signed. Signing would need a key
registered with GitHub, which is an account action rather than a repository one,
so the procedure says `-a` and this line says so rather than leaving a reader to
discover it.

## What is in the archives

| asset | bytes | sha256 |
|---|---:|---|
| `raw_logs.tar.zst` | 271 028 599 | `29c2401f100390268bbd52e43b5c2da9a61440bad3dabe502ca1684478771fd6` |
| `telemetry.tar.zst` | 162 320 | `8a29cc875e30bc66c6e83913b1bf40b075295218a37eef37897317805b47d03c` |
| `raw_logs_20260827.tar.zst` | 187 358 414 | `d56a7f88a099550bdab229ccb2bd36840f167550cea7689f575fd6d0f11da8ff` |
| `telemetry_20260827.tar.zst` | 46 603 | `db833395470aeaf842225a33fb544e933f1c0a3047cac161524aca1ac1aef061` |
| `raw_logs_20260828.tar.zst` | 221 242 327 | `5af671bf3cf47a20fa2ca78504c089642b9bb2ea249b8997577b4852caa7a5c2` |
| `telemetry_20260828.tar.zst` | 58 905 | `72e331bf0cbfaaee73619acf320523598cab4a46b7a7daa51fabbd6e2c455bf3` |
| `raw_logs_20260831.tar.zst` | 365 852 615 | `524ce5db75d494028b7b596f45e98b2384d8234726a924135775cce4f85b4cda` |
| `telemetry_20260831.tar.zst` | 114 587 | `4abdb6d701171bf0178ae36dad86f3190801cc047e88220d16f5442b7189949b` |

Four tranches, in the order they were cut. Each is a separate archive rather
than one rebuilt file, so the first one's digest keeps meaning what it meant
when it was published. Unpack all eight into one bench root and
`v4_audit_2026_08_25/EVIDENCE_MANIFEST.sha256` verifies every file.

## What the manifest holds

| what | count |
|---|---:|
| server logs | 3020 |
| telemetry traces | 23 |
| manifest entries | 3043 |
| tranches | 4 |
| release assets | 8 |
| committed run directories | 77 |
| arm-runs across them | 3005 |

`EVIDENCE_MANIFEST.sha256` itself hashes to `994f8fb6fcbfbe207cbe77e29a16a579cc2ab6d3359b165274f11c84dcd78ccb`.

## What is re-derived, and what is only integrity-checked

`analysis/rederive_from_logs.py <bench-root>` regenerates **four** log-derived
audit files from the archives and compares them with what is committed: the
acceptance counters, two sets of checkpoint timers and the speculative
accounting. `v4_audit_2026_08_25/EVIDENCE_REGISTRY.json` records which runs each
is expected to cover, held apart from the outputs being checked.

The primary per-request benchmark JSON is **not** regenerated. The server logs
carry no per-request timing rows, so those files can only be integrity-checked
against the manifest. That distinction is in the registry rather than in prose
alone. `.github/workflows/evidence.yml` performs that whole chain when it
runs. It has run once, on 2026-08-29, through a scoped `push` trigger, and
passed. Its `release: published` trigger reads the workflow from the TAG's own
ref rather than from the default branch, so publishing this release did re-run
the check by itself. This paragraph said the opposite when the release was cut,
and was wrong about it: the chain ran twice on 2026-09-01, on the tag as first
named, which stopped at the manifest, and on `v4.2`, which passed in full.

## The verifier

The tag's own commit. `python3 analysis/verify_claims.py` re-derives every
published figure from the committed data; `bench/check_release_binding.py <tag>`
checks that this release names the commit whose manifest, both registries and
three verifier scripts it publishes, rather than an earlier one.
