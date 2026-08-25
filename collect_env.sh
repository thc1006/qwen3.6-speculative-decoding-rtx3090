#!/usr/bin/env bash
# Snapshot exact machine + software state for reproducibility.
# Writes BENCHMARK_ENV.md with everything a reviewer needs to reproduce.
#
# Audited 2026-08-25 (see ERRATA.md). Two things this script does NOT capture,
# both of which cost this repository real claims:
#   * the CUDA *toolkit* version, if nvcc is not on PATH - `nvidia-smi`'s
#     "CUDA Version" is the driver's maximum supported runtime, not the build
#     toolkit (ERRATA F5)
#   * the binary's sha256 - `--version` can report a fork-point commit rather
#     than the build that ran, which is exactly how v3 ended up comparing two
#     different binaries while believing it had one (ERRATA D4)
# bench/retest_runner.py records both in its per-run manifest.
set -u

OUT="${OUT:-$(dirname "$0")/BENCHMARK_ENV.md}"

# BENCHMARK_ENV.md carries audit corrections added on 2026-08-25 (the -no-cnv
# description, the CUDA-version distinction, and the v3 --version-vs-build
# warning). This script rewrites the file wholesale, so refuse to clobber those
# unless the caller is explicit. Same failure mode as extract_exp2.py had.
if [ -f "$OUT" ] && grep -q "Audit addendum\|Corrected 2026-08-25" "$OUT" 2>/dev/null; then
    if [ "${FORCE_OVERWRITE:-0}" != "1" ]; then
        echo "refusing to overwrite $OUT: it contains audit corrections." >&2
        echo "write elsewhere with OUT=/path/to/new.md, or set FORCE_OVERWRITE=1" >&2
        exit 1
    fi
fi
LLAMA_REPO="${LLAMA_REPO:-$HOME/benchmarks/llama.cpp}"
MODEL_DIR="${MODEL_DIR:-$HOME/benchmarks/models}"

{
    echo "# Benchmark environment snapshot"
    echo ""
    echo "_Collected at $(date -Iseconds)_"
    echo ""
    echo "## Hardware"
    echo '```'
    nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv 2>&1
    echo ""
    echo "--- CPU ---"
    lscpu | grep -E "Model name|Socket|CPU\(s\)|Thread|MHz" 2>&1
    echo ""
    echo "--- RAM ---"
    free -h 2>&1
    echo '```'
    echo ""
    echo "## OS / kernel"
    echo '```'
    uname -a
    if command -v lsb_release >/dev/null 2>&1; then lsb_release -a 2>&1; fi
    cat /etc/os-release 2>/dev/null | head -5
    echo '```'
    echo ""
    echo "## CUDA / driver"
    echo '```'
    nvcc --version 2>&1 | tail -4 || echo "(nvcc not on PATH)"
    echo ""
    nvidia-smi 2>&1 | head -5
    echo '```'
    echo ""
    echo "## llama.cpp"
    if [ -d "$LLAMA_REPO/.git" ]; then
        echo '```'
        cd "$LLAMA_REPO" && \
        echo "commit    : $(git rev-parse HEAD)" && \
        echo "short     : $(git rev-parse --short HEAD)" && \
        echo "describe  : $(git describe --tags --dirty 2>/dev/null || echo N/A)" && \
        echo "authored  : $(git log -1 --format=%ci HEAD)" && \
        echo "subject   : $(git log -1 --format=%s HEAD)"
        echo '```'
    fi
    echo ""
    echo "## Models"
    echo '```'
    ls -lhS "$MODEL_DIR"/*/*.gguf 2>/dev/null | awk '{print $5, $9}'
    for f in "$MODEL_DIR"/*/*.gguf; do
        [ -f "$f" ] || continue
        echo "$(sha256sum "$f" | awk '{print $1"  "}')$(basename "$f")"
    done
    echo '```'
    echo ""
    echo "## Python packages (venv)"
    VENV_PY="${VENV_PY:-$(command -v python3)}"   # was a host-specific venv path
    if [ -x "$VENV_PY" ]; then
        echo '```'
        "$VENV_PY" --version
        "$VENV_PY" -m pip freeze | grep -iE "^(numpy|matplotlib|pandas|urllib3|requests|huggingface)" 2>&1 | sort
        echo '```'
    fi
    echo ""
    echo "## Build flags (for reference)"
    echo '```'
    echo "cmake flags used:"
    echo "  -DGGML_CUDA=ON"
    echo "  -DCMAKE_CUDA_ARCHITECTURES=86   # RTX 3090 SM 8.6"
    echo "  -DLLAMA_CURL=OFF"
    echo "  -DBUILD_SHARED_LIBS=OFF"
    echo "  CUDACXX=/usr/local/cuda-12.6/bin/nvcc"
    echo '```'
    echo ""
    echo "## Server invocation template"
    echo '```'
    echo "llama-server \\"
    echo "  -m Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \\"
    echo "  --host 127.0.0.1 --port 18123 \\"
    echo "  -ngl 999 -c 16384 --jinja \\"
    echo "  -fa on -ctk q8_0 -ctv q8_0 --no-webui"
    echo "  # + per-config spec-decode flags (see run_p0_matrix.sh / run_matrix.sh)"
    echo '```'
    echo ""
    echo "## Environment variables at bench time"
    echo '```'
    env | grep -E "^(CUDA|LD_LIBRARY|PATH|HOME|OLLAMA|GGML|LLAMA|HF|HUGGING)" 2>&1 | sort
    echo '```'
} > "$OUT"

echo "wrote $OUT ($(wc -l < "$OUT") lines)"
