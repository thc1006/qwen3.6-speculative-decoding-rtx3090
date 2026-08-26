#!/usr/bin/env bash
# Build the raw-evidence manifest and archive for a set of run directories.
#
# The committed JSON in `v4_audit_2026_08_25/data/` is derived from ~3 GB of
# llama-server logs and GPU telemetry that are too large to commit. Without a
# manifest the derivation is unfalsifiable: a reader can check that the JSON is
# internally consistent, but not that it came from the logs it names. This
# writes a SHA-256 for every log and every telemetry CSV, so the archive can be
# published separately and still be tied to this repository byte for byte.
#
# Run it on the benchmark host, AFTER the last measurement finishes: hashing
# 3 GB competes for I/O with a running benchmark.
#
#   bash bench/collect_evidence.sh ~/bench <out-dir>
set -euo pipefail

SRC="${1:?usage: collect_evidence.sh <bench-root> [out-dir]}"
OUT="${2:-$SRC/evidence_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
cd "$SRC"

MAN="$OUT/EVIDENCE_MANIFEST.sha256"
: > "$MAN"
n_logs=0
for d in matrix_* smoke_*; do
    [ -d "$d/server_logs" ] || continue
    for f in "$d"/server_logs/*.log; do
        [ -f "$f" ] || continue
        sha256sum "$f" >> "$MAN"
        n_logs=$((n_logs + 1))
    done
done
n_tele=0
for f in gpu_telemetry_*.csv; do
    [ -f "$f" ] || continue
    sha256sum "$f" >> "$MAN"
    n_tele=$((n_tele + 1))
done
echo "manifest: $n_logs logs + $n_tele telemetry files -> $MAN"

# Telemetry is small enough to live in the repository; the logs are not.
TELE="$OUT/telemetry"
mkdir -p "$TELE"
cp -f gpu_telemetry_*.csv "$TELE"/ 2>/dev/null || true

# built as an array so a directory with a space in its name cannot split
dirs=()
for d in matrix_* smoke_*; do
    [ -d "$d/server_logs" ] && dirs+=("$d/server_logs")
done
if [ "${#dirs[@]}" -eq 0 ]; then
    echo "no server_logs directories under $SRC" >&2
    exit 1
fi

if command -v zstd >/dev/null 2>&1; then
    ARC="$OUT/raw_logs.tar.zst"
    tar --use-compress-program='zstd -19 -T0' -cf "$ARC" "${dirs[@]}"
else
    ARC="$OUT/raw_logs.tar.xz"
    tar -cJf "$ARC" "${dirs[@]}"
fi
sha256sum "$ARC" | tee "$OUT/raw_logs.sha256"
ls -l "$ARC"
echo "done: $OUT"
