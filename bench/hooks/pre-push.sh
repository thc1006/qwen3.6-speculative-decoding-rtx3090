#!/usr/bin/env bash
# Refuse to push a commit that has not been reproduced green against CI.
#
# Install it, from the root of a checkout:
#
#     ln -sfn ../../bench/hooks/pre-push.sh .git/hooks/pre-push
#
# A SYMLINK, not a copy. This guard was a file in `.git/hooks/`, which no clone
# carries and nothing in this repository could see, while two comments in
# `bench/ci_faithful.sh` said a pre-push hook consumes the token it writes. That
# is a guard the tree names and does not hold, and it is the third time this
# repository has found its own tooling living where nothing checks it: the probe
# launcher, then the CI reproduction, then this. A copy would be the other half
# of that defect, because the installed one and the tracked one would drift.
#
# The `.sh` suffix is not decoration. `audit.yml`'s shellcheck step globs
# `-name '*.sh'`, and a hook called `pre-push` would sit outside it, which is the
# same narrow glob that step was widened to remove.
#
# "I will run the reproduction before pushing" is a promise. On 2026-09-01 a
# probe was launched on an eight core host because a number was carried over
# from a script written for a thirty-two core one, and every gate was green and
# none of them covered it. A promise is not a guard. This is.
#
# The token is written by bench/ci_faithful.sh, only when every job it
# reproduces passed, and it is named after the exact commit, so verifying one
# tree and pushing another does not satisfy it.
#
# Override, when you mean to and can say why:  CI_REPRO_OVERRIDE=1 git push ...
set -u
TOKENS=$HOME/.ci_repro_green
REMOTE_URL=${2:-}
status=0

# Only the remote CI actually watches. The probe is driven by pushing the
# commit to a second machine over ssh, and a guard that fires on that is a
# guard that gets overridden every day until it means nothing.
case $REMOTE_URL in
    *github.com*) ;;
    *) echo "pre-push: $REMOTE_URL is not the CI remote, not checking" >&2; exit 0 ;;
esac
# git feeds this  <local ref> <local sha> <remote ref> <remote sha>
# The first version named the FOURTH field `remote_ref_name` and matched it
# against `refs/tags/*`, so the tag exemption compared a SHA to a ref pattern
# and never fired: this hook would have refused the tag push that the release
# procedure starts with, and the reason would not have been visible.
while read -r _local_ref local_sha remote_ref _remote_sha; do
    case $local_sha in
        *[!0-9a-f]*|"") continue ;;                       # a deletion has no sha
    esac
    case $remote_ref in
        refs/tags/*) echo "pre-push: $remote_ref is a tag, not a tree" >&2; continue ;;
    esac
    if [ "${CI_REPRO_OVERRIDE:-0}" = "1" ]; then
        echo "pre-push: CI_REPRO_OVERRIDE=1, pushing ${local_sha:0:12} unverified" >&2
        continue
    fi
    # a token has to say what was reproduced. An empty file satisfied `-f`,
    # and an empty file is exactly what a failed copy leaves behind.
    if [ ! -s "$TOKENS/$local_sha" ] || ! grep -q "^python 3" "$TOKENS/$local_sha"; then
        echo "pre-push: REFUSED. No green reproduction recorded for ${local_sha:0:12}." >&2
        echo "  Run bench/ci_faithful.sh $local_sha on the host whose python" >&2
        echo "  matches the workflow, and push once it records the token." >&2
        status=1
    else
        echo "pre-push: ${local_sha:0:12} reproduced green ($(cat "$TOKENS/$local_sha"))" >&2
    fi
done
exit "$status"
