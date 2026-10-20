#!/usr/bin/env bash
# Reproduce audit.yml's jobs on pristine clones, the way CI checks them out.
#
#     ci_faithful.sh <commit>
#
# Every difference between this box and a runner that has already cost a red
# CI run is eliminated here rather than hoped about:
#
#   python   five of audit.yml's six jobs pin 3.12 and one pins 3.13. This
#            refuses to run on anything but the version the jobs it reproduces
#            pin, because the whole point is to stop testing on 3.13 and
#            reporting it as CI.
#   TZ       the runner is UTC. A naive nvidia-smi timestamp read in +0800 is
#            how the claims job went red on 2026-09-01.
#   depth    `static`, `data integrity` and `charts` are checked out at the
#            default depth of 1; `claims` and `unit and mutation` set
#            fetch-depth 0. A depth-1 clone fails every provenance assertion
#            and does not say why, so the two are built separately and each
#            job runs in the one the workflow gives it.
#   tree     a clone of the COMMIT, so an untracked or ignored file cannot make
#            a check pass here that fails on a fresh checkout.
#
# One difference it does NOT eliminate, and it has cost a red run: the HARDWARE.
# A runner has four processors. This host has thirty-two and the box these are
# written on has eight, so a test that named processors four, five and six passed
# in both places and failed on the runner, where `taskset` has nothing to pin to.
# Every job here can be green and a test that reads the processor count, the
# memory, or the disk can still be red there. Such a test has to derive what it
# needs from the host it is on, and say so when the host cannot supply it.
#
# What it does NOT reproduce is named at the end rather than left for a reader
# to notice: the `charts` job, which pins 3.13 and so has to be run separately
# on a 3.13 interpreter, and the evidence workflow, which needs a published
# release and is rehearsed against the real archives instead.
set -u

die() { echo "FAIL: $*" >&2; exit 2; }

SHA_IN=${1:?usage: bench/ci_faithful.sh <commit>}
# The repository this script is IN, not a path written down. It was
# `$HOME/dev/qwen3.6-speculative-decoding-rtx3090` while the script lived in a
# home directory; in a repository that spelling would clone some other checkout
# than the one the script came from, which is the wrong-copy defect that put this
# file here, arriving by the other door.
SRC=$(cd "$(dirname "$0")/.." && pwd)
[ -d "$SRC/.git" ] || die "$SRC is not a git repository; run this from a checkout"
W=${CI_REPRO_DIR:-$HOME/ci_repro}
WANT_PY=3.12
export TZ=UTC

got_py=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
[ "$got_py" = "$WANT_PY" ] || die "python is $got_py and the jobs reproduced here pin $WANT_PY.
      Run this where that python is. Reproducing CI on the wrong interpreter
      is the failure this script exists to prevent."

SHA=$(git -C "$SRC" rev-parse "$SHA_IN^{commit}" 2>/dev/null) \
  || die "$SHA_IN is not a commit in $SRC"

rm -rf "$W"; mkdir -p "$W"
fail=0
note() {
    printf "  %-46s %s\n" "$1" "$2"
    if [ "$2" != "OK" ]; then fail=$((fail + 1)); fi
}
run() {  # run LABEL -- cmd...
    local label
    label=$1
    shift
    if [ "${1:-}" = "--" ]; then shift; fi
    case $label in *[!A-Za-z0-9_]*) die "bad label '$label'";; esac
    if "$@" > "$W/$label.log" 2>&1; then
        note "$label" OK
    else
        note "$label" FAILED
        tail -4 "$W/$label.log"
    fi
}

# --- the two checkouts, each asserted to be what it claims to be -----------
# `git clone --depth` is silently ignored for a local path, and `--local` with
# `--no-local` is a contradiction git rejects. The first version of this did
# both and fell back to a FULL clone that it then called shallow: a control
# that quietly became the thing it was controlling against. This is what
# actions/checkout does, and the depth is asserted afterwards.
mk_checkout() {  # mk_checkout DIR DEPTH
    local dir depth got
    dir=$1 depth=$2
    mkdir -p "$dir"
    git -C "$dir" init --quiet
    git -C "$dir" remote add origin "file://$SRC"
    if [ "$depth" = "full" ]; then
        # TAGS TOO. `actions/checkout` at fetch-depth 0 fetches
        # `+refs/tags/*:refs/tags/*`, and this fetched one commit and no tags,
        # so a test that reads `git tag` saw an empty list here and six in CI.
        # It failed the reproduction on 2026-09-01 and the tree was fine: an
        # unfaithful control is a red that costs a diagnosis, and the next one
        # of those is a green that costs more.
        git -C "$dir" fetch --quiet --tags origin "$SHA" \
            || die "fetch failed in $dir"
    else
        git -C "$dir" fetch --quiet --depth "$depth" origin "$SHA" \
            || die "shallow fetch failed in $dir"
    fi
    git -C "$dir" checkout --quiet --detach FETCH_HEAD || die "checkout failed in $dir"
    got=$(git -C "$dir" rev-parse HEAD)
    [ "$got" = "$SHA" ] || die "$dir is at $got and should be at $SHA"
}

echo "python $got_py, TZ=$TZ, commit ${SHA:0:12}"
mk_checkout "$W/shallow" 1
mk_checkout "$W/full" full
n_shallow=$(git -C "$W/shallow" rev-list --count HEAD)
n_full=$(git -C "$W/full" rev-list --count HEAD)
echo "  shallow checkout: $n_shallow commit(s)   full checkout: $n_full commits"
[ "$n_shallow" -eq 1 ] || die "the shallow checkout has $n_shallow commits and is not shallow"
[ "$n_full" -gt 1 ] || die "the full checkout has $n_full commit(s) and is not full"

# --- the linters, at the versions the lock pins, never a reused venv -------
V=$W/lintvenv
python3 -m venv "$V" >/dev/null 2>&1 || die "could not create the lint venv"
"$V/bin/pip" install -q --require-hashes -r "$W/full/requirements-lint.lock" \
    || die "the pinned linters would not install"
for tool in pyflakes shellcheck; do
    [ -x "$V/bin/$tool" ] || die "$tool is not in the lock file's venv"
done
echo "  linters: $("$V/bin/pyflakes" --version 2>&1), shellcheck $("$V/bin/shellcheck" --version 2>&1 | awk '/^version:/{print $2}')"

echo "=== static (depth 1, as the workflow checks it out) ==="
cd "$W/shallow" || die "cannot enter the shallow checkout"
files=$(find . -name '*.py' -not -path './.git/*' -not -path './v2_3090_followup/*' \
          -not -path './v3_dflash_2026_05_07/*' -not -path './results/*' \
          -not -path './bench_runner.py' | sort)
[ -n "$files" ] || die "no python files found; the checkout is not what it should be"
# shellcheck disable=SC2086
run compileall -- python3 -m compileall -q $files
# shellcheck disable=SC2086
run pyflakes -- "$V/bin/pyflakes" $files
sh=$(find . -name '*.sh' -not -path './.git/*' -not -path './v2_3090_followup/*' \
       -not -path './v3_dflash_2026_05_07/*' -not -path './results/*' | sort)
[ -n "$sh" ] || die "no shell scripts found"
# shellcheck disable=SC2086
run shellcheck -- "$V/bin/shellcheck" --severity=style $sh
run check_links -- python3 analysis/check_links.py

echo "=== data integrity (depth 1) ==="
run check_data_integrity -- python3 analysis/check_data_integrity.py
# The job has two invocations: the default root and every round that keeps its
# own data. One of them was reproduced here and the other was not, which made
# `data integrity` a job this script claimed while running half of it.
run check_data_integrity_v5 -- python3 analysis/check_data_integrity.py v5_pinning_2026_09_26/data

echo "=== claims (full history) ==="
cd "$W/full" || die "cannot enter the full checkout"
python3 analysis/verify_claims.py > "$W/claims.log" 2>&1
if grep -q "^ALL CLAIMS VERIFIED" "$W/claims.log"; then
    note "verify_claims ($(grep -cE '^  (PASS|FAIL)  ' "$W/claims.log") assertions)" OK
else
    note "verify_claims" FAILED
    grep "^FAILURES:" "$W/claims.log" | head -2
fi

# audit.yml's claims job has five steps and this script ran one of them.
# The closing line says "4 of 5 jobs reproduced", which is only true if a
# job means all of its steps. `load_run_power.py --check` was added to that
# job on 2026-09-17 and is the one that re-derives the run X tables.
run table_coverage -- python3 analysis/table_coverage.py
run load_run_power -- python3 analysis/load_run_power.py --check
# Added to that job on 2026-10-01: it re-derives run Y's result table, the
# plan's power table and the figures the mechanism paragraph quotes, and the
# reason it exists is that nothing did and three columns of the first were the
# mean log ratio printed as a percentage.
run rederive_run_y -- python3 analysis/rederive_run_y.py
run plan_z_power -- python3 analysis/plan_z_power.py
run past_threshold -- python3 analysis/past_threshold_fit.py
# The sixth step of that job, which this script did not have either: every run
# directory carrying RUN_COMPLETE.json has to aggregate under `--strict`, and the
# count is checked against the tree rather than against a floor written down.
# Reproduced as a function so `run` still owns the log and the verdict.
# `run` invokes this through "$@", which shellcheck cannot follow, so every
# line of the body reads as unreachable to it. The alternative is a `bash -c`
# with the whole loop quoted inside it, which is worse to read and no safer.
# shellcheck disable=SC2317
aggregate_attested() {
    local n want d
    n=0
    for d in v4_audit_2026_08_25/data/*/; do
        if [ -f "$d/RUN_COMPLETE.json" ]; then
            python3 analysis/matrix_report.py --strict "$d" || return 1
            n=$((n + 1))
        fi
    done
    want=$(find v4_audit_2026_08_25/data -maxdepth 2 -name RUN_COMPLETE.json | wc -l)
    echo "$n attested runs aggregated, $want carry the marker"
    [ "$n" -eq "$want" ] || { echo "FAIL: aggregated $n of $want" >&2; return 1; }
    [ "$n" -ge 36 ] || { echo "FAIL: only $n attested runs" >&2; return 1; }
}
run matrix_report -- aggregate_attested

echo "=== unit and mutation (full history) ==="
# UNDER THE RUNNER'S PROCESSOR COUNT. A test that asked for processors four,
# five and six passed here on thirty-two of them and failed on a runner that has
# four, and this script's whole claim is that it does not hope about such a
# difference. `taskset` restricts the affinity, which is what
# `os.sched_getaffinity` reads, so a test deriving what it needs from the host
# derives what a runner would give it. `os.cpu_count()` ignores affinity and the
# two launchers read `getconf`, so the shard-count refusals still see the real
# machine, which is what they are about.
#
# Not written down as a number of its own: CI_RUNNER_CPUS says what a runner has,
# the default is the four that `ubuntu-latest` gives, and a host with fewer than
# that runs the suite unrestricted rather than pretending.
RUNNER_CPUS=${CI_RUNNER_CPUS:-0-3}
PIN=()
if command -v taskset > /dev/null 2>&1 \
   && [ "$(getconf _NPROCESSORS_ONLN)" -gt 4 ]; then
    PIN=(taskset -c "$RUNNER_CPUS")
    echo "  unit runs on processors $RUNNER_CPUS, as a runner has" >&2
fi
"${PIN[@]}" python3 -m unittest discover -s tests -p 'test_*.py' > "$W/unit.log" 2>&1
if grep -qE "^OK$" "$W/unit.log"; then
    note "unittest ($(grep -oE '^Ran [0-9]+ tests' "$W/unit.log"))" OK
else
    note "unittest" FAILED
    grep -E "^(FAIL|ERROR): " "$W/unit.log" | head -5
fi
run code_mutations -- python3 tests/mutate.py
# Sharded, one per processor, which is what the workflow does now too. It
# is the same eighty-four perturbations against the same clean mirrors and
# the launcher requires their caught counts to add up to the whole list; a
# shard that quietly did nothing fails the total. On this host it is the
# difference between fifty-six minutes and about two.
run data_perturbations -- bash bench/run_data_mutations.sh "$(getconf _NPROCESSORS_ONLN)"

# --- what this covered, against what the workflow declares ----------------
# Derived from audit.yml, not written down here: a job added to the workflow
# would otherwise be missing from this reproduction without a word, which is
# the shape this repository keeps removing.
echo
declared=$(sed -n 's/^    name: //p' "$W/full/.github/workflows/audit.yml" | sort)
covered="claims
code mutations
data integrity
data perturbations
static
unit"
missing=$(comm -23 <(printf '%s\n' "$declared") <(printf '%s\n' "$covered" | sort))
echo "audit.yml declares $(printf '%s\n' "$declared" | wc -l) jobs; this reproduces $(printf '%s\n' "$covered" | wc -l)."
if [ -n "$missing" ]; then
    echo "NOT reproduced here, and not claimed to be:"
    printf '%s\n' "$missing" | sed 's/^/    /'
fi
echo "    (and evidence.yml, which needs a published release)"

echo
if [ "$fail" -eq 0 ]; then
    n_cov=$(printf '%s\n' "$covered" | wc -l)
    n_dec=$(printf '%s\n' "$declared" | wc -l)
    echo "$n_cov of audit.yml's $n_dec jobs reproduced green on python $got_py, TZ=UTC,"
    echo "from a clean checkout of ${SHA:0:12}. That is a statement about those"
    echo "$n_cov jobs at that commit and about nothing else."
    # The token `bench/hooks/pre-push.sh` looks for. It names the commit, so
    # verifying one tree and pushing another cannot pass: the whole point of
    # running this is lost if the thing pushed is not the thing checked.
    mkdir -p "$HOME/.ci_repro_green"
    # the HOST too. This box runs python 3.13 and the jobs pin 3.12, so the
    # reproduction happens on the bench host and the token is copied back; a
    # token that does not say where it was produced is a token whose claim
    # cannot be checked, and the pre-push hook prints it verbatim.
    printf 'python %s TZ=%s host=%s jobs=%s\n' "$got_py" "$TZ" "$(hostname)" "$covered" \
        > "$HOME/.ci_repro_green/$SHA"
    echo "recorded: ~/.ci_repro_green/${SHA:0:12}"
else
    echo "$fail CHECK(S) FAILED -- do not push"
    rm -f "$HOME/.ci_repro_green/$SHA"
fi
exit "$fail"
