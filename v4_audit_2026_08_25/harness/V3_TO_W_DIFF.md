# What changed in the runner between V3 and W

Run W was described as "V3 verbatim except `BENCH_ORDER`", and as the schedule
being the only thing that could explain a difference between the two. The
manifests do not support that on their own: they record different runner
hashes, `harness_tree_sha.sha` is `null` in both, and neither pins an expected
commit. This file closes that gap with the diff itself rather than with a
claim about it.

| | SHA-256 | where it is |
|---|---|---|
| V3 runner | `506b060653b6117166d9f8d2e51840577a9f68b7516deb7e8558f769350a841a` | `bench/retest_runner.py` at commit `eddc81c77707b05619358d4d97ac2297870f2924` |
| W runner | `341a4a649c9215feb596561fd14c466274bc202c915455ab8d6a34fddd862f0b` | `v4_audit_2026_08_25/harness/retest_runner_W_20260828_104222.py`, archived beside this file |

Both hashes are the ones the run manifests record. `V3_to_W_runner.diff` is
`diff -u` between exactly those two blobs: **189 lines, 111 added, 3 removed,
nine hunks**. Every hunk, and what it is:

| hunk | change | class |
|---|---|---|
| 1 | `import random` | schedule builder |
| 2 | `BENCH_HARDCAP_SUFFIX` restricted to `[A-Za-z0-9_.-]+` and refused for `.`/`..`; `williams` added to `BENCH_ORDER`; `BENCH_SCHEDULE_SEED` read | provenance assertion + schedule |
| 3 | `check_identity` gains a `healthy` parameter | provenance assertion |
| 4 | an arm-run that produced rows and yielded no observed target identity is refused | provenance assertion |
| 5 | `carryover_counts`, `is_carryover_balanced`, `williams_square` | schedule builder |
| 6 | the `williams` branch of the scheduler, with its even-`n` and balance refusals | schedule builder |
| 7 | the `check_identity` call site passes `healthy` | provenance assertion |
| 8 | four manifest fields recording the schedule's balance, seed and adjacency counts | provenance record |
| 9 | the `williams` entry in the manifest's order-mode descriptions | provenance record |

**Nothing in the diff touches the request body, the server argv, timing
collection, teardown or aggregation.** `chat()` is byte-identical between the
two, as are the server launch, the per-request timing extraction and the
per-arm-run aggregation.

One change deserves naming rather than filing under "provenance". Hunk 4 adds a
failure mode V3 did not have: an arm-run with rows but no observed target
identity is now rejected. It can only reject more, never alter a measured
number, so it cannot move a figure -- but it means the two runners do not accept
exactly the same set of runs, and "verbatim" was the wrong word for that too.

## What this licenses, and what it does not

Supported: *W used the same treatment definitions, prompts, models and server
build as V3, with a Williams schedule and a later harness revision whose diff is
archived here and contains no request or timing semantic changes.*

Not supported: *the schedule is the only thing that can explain a difference
between W and V3.* The harness revision is a second difference. It is a small
one, it is enumerated above, and none of it reaches the measurement path -- but
it is not nothing, and the earlier wording asserted that it was.
