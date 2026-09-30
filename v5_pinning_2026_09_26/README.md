# Core placement on the bench host, from 2026-09-26

A round of its own, and deliberately not part of the `v4.2` archive.

`analysis/verify_claims.py` pins the number of run directories under
`v4_audit_2026_08_25/data` at seventy-seven and the attested ones at forty-eight,
and that checker is one of the six files `bench/check_release_binding.py` compares
between the `v4.2` tag and HEAD. Putting a new run there would change a count in a
frozen file, so the tag would stop publishing the tree that verifies it. The
archived rounds before it, `v2_3090_followup/` and `v3_dflash_2026_05_07/`, each
hold their own data for the same kind of reason, and this follows them.

The directory name is a round label and not a released version. There is no `v5`
release, and whether this data ever joins one is a separate decision.

## What is here

`data/matrix_Y_pinning_20260926_131110/` is run Y, pre-registered in
[`v4_audit_2026_08_25/PROSPECTIVE_PLAN_Y_CORE_PINNING.md`](../v4_audit_2026_08_25/PROSPECTIVE_PLAN_Y_CORE_PINNING.md)
before it ran. Forty-eight arm-runs: two arms, two pinnings, twelve blocks, one
invocation. Every arm-run records the cpu list the driver asked for and
`Cpus_allowed_list` as the kernel reported it.

`data/gpu_telemetry_Y_20260926_131110.csv` is the GPU trace for the invocation,
at one second.

Server logs are not committed, which is what `v4_audit_2026_08_25` does with its
own: forty-eight of them are 277 MB and they are raw evidence for a release
asset, not for a tree.

## Two things about the trace that a reader has to be told

**It is a window, not the whole file.** The sampler was not stopped when the run
ended. `bench/gpu_telemetry.sh`'s `raw` schema backgrounds `nvidia-smi -l`, which
is nvidia-smi's own loop, and every driver script here stops its sampler by
killing the shell -- which works for the two schemas that loop in bash and did not
work for this one. Two samplers ran at one hertz for four days, reparented to
init, and neither a GPU-lock check nor a load average names them. The trap that
fixes it is in the same commit as this file.

So the file on the bench host holds 363 434 rows spanning 2026-09-26 to
2026-09-30, of which 2 469 are inside the invocation. What is committed here is
those 2 469 rows and the header. The rest is an idle card. The untruncated file's
SHA-256 is

    1281b300225ccfb4acc01b607556285736256da38da70b6e7e759940e516f1e3

so the truncation is auditable rather than asserted.

**Two samplers were running during the run, not one.** The first attempt at run Y
died at once on an unknown arm, and its sampler was the one left orphaned; the
second attempt started its own. Both were sampling for the whole invocation. Each
is one `nvidia-smi` query per second, tens of milliseconds of one processor, and
the two conditions alternate within every block, so it cannot bias the contrast.
It is host work during a measurement, so it is recorded.

## The result

Pre-registered: for each arm, the twelve within-block log ratios of the slow
condition's pooled decode rate to the fast condition's, their mean, and a Student
t interval on eleven degrees of freedom. Primary reading the one-sided
ninety-five per cent upper limit on the slowdown.

| arm | fast | slow | change | 95 % interval | one-sided upper limit | plus or minus one per cent |
|---|---:|---:|---:|---|---:|---|
| `spec-dflash-n2` | 140.327 | 120.068 | −15.57 % | [−16.90 %, −14.25 %] | 16.66 % | moves |
| `baseline` | 114.664 | 97.763 | −15.94 % | [−16.76 %, −15.12 %] | 16.61 % | moves |

The pre-registration expected both arms to hold. Both move, by more than fifteen
per cent, and the bound is far above the 3.93 % step ERRATA A16 is about. That is
the plan's second branch: processor speed reaches this workload by enough to
matter.

Three things establish that this is speed and not different work, and that it is
the host and not the card:

- the output is identical to the token. 36 000 generated in both conditions;
  29 292 drafted and 21 192 accepted in both conditions of the speculative arm
- the GPU's SM clock is the same or slightly higher in the slow condition,
  1928.8 MHz against 1920.0, so it is not throttling
- its power, utilisation and temperature all fall in the slow condition, from
  326.8 W to 302.4, from 75.2 % to 56.1, from 73.8 °C to 71.2. The card is
  waiting

## What follows from it, and what does not

Established: a full move to the efficiency cores costs about sixteen per cent of
the decode rate on this host, on both arms, with the GPU idle-waiting for the
difference. About half of a token's time is host work that scales with processor
clock.

Not established, and the reason the next run exists: that this explains A16. A16
names a 3.93 % step, which is 0.23 of a full displacement, or roughly two of
eight threads landing on efficiency cores. Every published run here passes no
thread count and no affinity, llama.cpp chooses `n_threads = 8` of the thirty-two
logical processors, and each arm-run starts a fresh server and so draws its own
placement. That is the right shape and the right order of magnitude, and it is an
arithmetic coincidence until a run sets the number of displaced threads and
measures the response.
