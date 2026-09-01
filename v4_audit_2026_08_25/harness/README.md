# Archived harness versions

A run records `runner_sha256`, and `analysis/verify_claims.py` requires that
hash to resolve to a version of `bench/retest_runner.py` this repository has
held. Normally it resolves in git history.

`retest_runner_W_20260828_104222.py` does not, and the reason is a process
error worth recording rather than hiding: the runner was deployed to the bench
host from the working tree at 10:42 on 2026-08-28 and run W was started, and
the file was then edited twice more before being committed. So the exact source
that produced W's 500 arm-runs was never a commit.

This is that file, byte for byte, fetched back from the bench host and verified
to hash to the `341a4a64…` every one of W's five manifests names. It is
archived so the pin resolves to something real; it is not a substitute for
committing first, and `TheRunnerThatRanMustBeRecoverable` requires every
committed run's `runner_sha256` to resolve to git history **or** to a file in
this directory, so a run whose harness is neither cannot pass unnoticed.

The two edits that happened after W started, for the record: the hard-cap suffix
collision check was replaced with a recorded `ambiguous_arms` field, and the
end-of-run model-file re-hash was added. Neither can affect a run already in
progress on another machine, and W's manifests carry the pre-edit hash, which
is exactly why the pin is worth having.
