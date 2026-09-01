#!/usr/bin/env bash
# Re-convert the z-lab DFlash drafter with post-merge llama.cpp master.
#
# Why this exists
# ---------------
# The archived v3 DFlash measurements used a drafter GGUF produced by PR
# #22105's PRE-MERGE converter, and post-merge master rejects that file:
#
#     error loading model hyperparameters:
#     DFlash model requires 'target_layers' in GGUF metadata
#
# Master's gguf-py carries `add_target_layers()` and the `{arch}.target_layers`
# key, so re-converting from the original HF safetensors should produce a
# loadable drafter. That unblocks a DFlash A/B on ONE pinned binary, which the
# archived v3 comparison never had - it compared b8889-bcb5eeb64 against
# b8942-67cb0d507 and read the difference as a DFlash effect (ERRATA D4).
#
# Do not run this while a benchmark is running on the same host. The conversion
# is CPU- and RAM-heavy, and CPU contention measurably depresses GPU decode
# rate: during the audit a concurrent compile pulled the no-speculation
# baseline from ~133 tok/s down to 116-123 tok/s on this very machine.
#
# Everything is overridable:
#   LLAMA_REPO   post-merge llama.cpp checkout   (default ~/bench/llama-retest)
#   CONVERT_VENV python env with torch/transformers/gguf
#   DFLASH_SRC   HF safetensors directory for the drafter
#   TARGET_META  directory holding the TARGET model's tokenizer + config
#   OUT_GGUF     where to write the converted drafter
#   TARGET_GGUF  target model, used only for the post-conversion load check

set -uo pipefail

LLAMA_REPO="${LLAMA_REPO:-$HOME/bench/llama-retest}"
CONVERT_VENV="${CONVERT_VENV:-$HOME/dflash_convert_venv}"
DFLASH_SRC="${DFLASH_SRC:-$HOME/models/qwen36-dflash}"
TARGET_META="${TARGET_META:-$HOME/models/qwen36-target-meta}"
OUT_GGUF="${OUT_GGUF:-$HOME/models/qwen36-dflash-master.gguf}"
TARGET_GGUF="${TARGET_GGUF:-$HOME/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
PORT="${PORT:-18190}"
LOG_DIR="${LOG_DIR:-$HOME/bench}"
CONVERT_LOG="$LOG_DIR/dflash_convert_$(date +%Y%m%d_%H%M%S).log"

die() { echo "FATAL: $*" >&2; exit 1; }

# ---- preconditions, all of them, before touching anything -------------------
# NB: a git worktree's .git is a FILE containing "gitdir: ...", not a
# directory, so `-d` is wrong here. Ask git instead.
git -C "$LLAMA_REPO" rev-parse --git-dir >/dev/null 2>&1 \
    || die "no llama.cpp checkout at $LLAMA_REPO"
[ -x "$CONVERT_VENV/bin/python" ]         || die "no python at $CONVERT_VENV/bin/python"
[ -f "$DFLASH_SRC/config.json" ]          || die "no HF drafter at $DFLASH_SRC"
[ -f "$TARGET_META/config.json" ]         || die "no target metadata at $TARGET_META"
[ -f "$LLAMA_REPO/convert_hf_to_gguf.py" ] || die "converter missing in $LLAMA_REPO"

if pgrep -f 'retest_runner.py|llama-server' >/dev/null 2>&1; then
    die "a benchmark or server is running on this host; CPU contention would bias it"
fi

CONVERTER_SHA="$(git -C "$LLAMA_REPO" rev-parse HEAD)"
echo "converter commit : $CONVERTER_SHA"
echo "converter subject: $(git -C "$LLAMA_REPO" log -1 --format=%s)"
echo "drafter source   : $DFLASH_SRC"
echo "target metadata  : $TARGET_META"
echo "output           : $OUT_GGUF"
echo "log              : $CONVERT_LOG"

"$CONVERT_VENV/bin/python" - <<'PY' || die "conversion deps missing"
import torch, transformers, gguf, safetensors
print(f"  deps: torch {torch.__version__}, transformers {transformers.__version__}")
PY

# Never health-check a half-written file, and never promote one the loader has
# not accepted: convert to a temp path, load-check the temp path, promote last.
TMP_GGUF="${OUT_GGUF}.partial"
rm -f "$TMP_GGUF"

# tee, not `| tail`, so a failure's real message survives; and take the
# converter's status, not the pager's.
set -o pipefail
"$CONVERT_VENV/bin/python" "$LLAMA_REPO/convert_hf_to_gguf.py" "$DFLASH_SRC" \
    --outtype bf16 --target-model-dir "$TARGET_META" --outfile "$TMP_GGUF" \
    2>&1 | tee "$CONVERT_LOG"
rc=${PIPESTATUS[0]}
echo "convert rc=$rc"
[ "$rc" -eq 0 ] || die "conversion failed, see $CONVERT_LOG"
[ -s "$TMP_GGUF" ] || die "conversion produced no output"

# ---- the key whose absence rejected the archived file ------------------------
echo "=== does the output carry {arch}.target_layers? ==="
# `if cmd; then` rather than testing $? afterwards: any command inserted
# between the two lines would silently break the check.
if ! PYTHONPATH="$LLAMA_REPO/gguf-py" "$CONVERT_VENV/bin/python" - "$TMP_GGUF" <<'PY'
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1], "r")
hit = [k for k in r.fields if k.endswith("target_layers")]
print("  target_layers keys:", hit or "NONE")
for k in hit:
    print(f"    {k} = {r.fields[k].contents()}")
sys.exit(0 if hit else 1)
PY
then
    die "converted file still lacks target_layers; master's converter did not write it"
fi

# The file stays at the temp path until the loader has accepted it. Promoting
# here - which is what this did - defeats the temp path entirely: a drafter that
# converts and carries `target_layers` but that the loader refuses is still
# sitting at the final path when the script dies, and the next benchmark run
# picks it up. `target_layers` present is not the success condition; loading is.

# ---- does the merged loader actually accept it? -----------------------------
echo "=== load check with --spec-type draft-dflash ==="
LOADLOG="$LOG_DIR/dflash_loadcheck_$(date +%Y%m%d_%H%M%S).log"
timeout 240 "$LLAMA_REPO/build/bin/llama-server" \
    -m "$TARGET_GGUF" -md "$TMP_GGUF" \
    --spec-type draft-dflash --spec-draft-n-max 8 -ngld 99 \
    -ngl 999 -c 4096 --jinja -fa on -ctk q8_0 -ctv q8_0 --no-webui \
    --host 127.0.0.1 --port "$PORT" -v > "$LOADLOG" 2>&1 &
srv=$!
ok=0
for _ in $(seq 1 200); do
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok; then ok=1; break; fi
    kill -0 "$srv" 2>/dev/null || break
    sleep 1
done
kill -TERM "$srv" 2>/dev/null; wait "$srv" 2>/dev/null

echo "health=$ok  log=$LOADLOG"
grep -iE "target_layers|dflash|speculative decoding context init|error" "$LOADLOG" | head -12
if [ "$ok" -ne 1 ]; then
    rm -f "$TMP_GGUF"
    die "loader still refuses the converted drafter; see $LOADLOG (the partial \
file has been removed, nothing was promoted to $OUT_GGUF)"
fi

# Only now is the conversion a success.
mv -f "$TMP_GGUF" "$OUT_GGUF"
sha="$(sha256sum "$OUT_GGUF" | awk '{print $1}')"
printf '%s  %s\nconverter_commit %s\n' "$sha" "$(basename "$OUT_GGUF")" "$CONVERTER_SHA" \
    > "${OUT_GGUF}.provenance"
ls -la "$OUT_GGUF"
echo "sha256 $sha  (provenance written next to the file)"
echo "OK: $OUT_GGUF loads under --spec-type draft-dflash on $CONVERTER_SHA"
