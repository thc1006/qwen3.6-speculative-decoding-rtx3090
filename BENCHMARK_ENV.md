# Benchmark environment snapshot

> [!WARNING]
> **Corrected 2026-08-25.** Three descriptions in this file were wrong; they are
> marked inline below. Summary: (1) `llama-cli -st -no-cnv` is **not**
> "single-turn non-conversational" - `-no-cnv` is rejected by `llama-cli` on
> these builds; (2) `nvidia-smi`'s "CUDA Version: 13.0" is driver support, not
> the build toolkit, which was CUDA 12.6 for v1; (3) the v3 `llama-cli
> --version` string does not match the binary that produced the v3 logs. See
> [`ERRATA.md`](ERRATA.md) items D1, D2, D4, F5.

_v1 (original 19-config bench) collected at 2026-04-21T08:00:30+08:00 on
the s1 dual-3090 host. A v2 environment snapshot appears below for the
follow-up bench in `v2_3090_followup/` on the single-3090 `3090` host._
_v3 environment snapshot (DFlash via llama.cpp PR #22105) is also below;
v3 ran on the **same** physical `3090` host as v2 (single RTX 3090,
Tailscale name `3090`), but with llama.cpp PR #22105 checked out from
master and a separate Python venv for the HF→GGUF drafter convert step._

## Hardware
```
index, name, memory.total [MiB], driver_version, compute_cap
0, NVIDIA GeForce RTX 3090, 24576 MiB, 580.126.09, 8.6
1, NVIDIA GeForce RTX 3090, 24576 MiB, 580.126.09, 8.6

--- CPU ---
CPU(s):                                  16
On-line CPU(s) list:                     0-15
Model name:                              11th Gen Intel(R) Core(TM) i7-11700 @ 2.50GHz
Thread(s) per core:                      2
Socket(s):                               1
CPU(s) scaling MHz:                      74%
CPU max MHz:                             4900.0000
CPU min MHz:                             800.0000
NUMA node0 CPU(s):                       0-15

--- RAM ---
               total        used        free      shared  buff/cache   available
Mem:            62Gi       6.5Gi       5.9Gi       126Mi        50Gi        56Gi
Swap:          8.0Gi       904Ki       8.0Gi
```

## OS / kernel
```
Linux s1 6.17.0-20-generic #20~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Thu Mar 19 01:28:37 UTC 2 x86_64 x86_64 x86_64 GNU/Linux
Distributor ID:	Ubuntu
Description:	Ubuntu 24.04.4 LTS
Release:	24.04
Codename:	noble
PRETTY_NAME="Ubuntu 24.04.4 LTS"
NAME="Ubuntu"
VERSION_ID="24.04"
VERSION="24.04.4 LTS (Noble Numbat)"
VERSION_CODENAME=noble
```

## CUDA / driver
`nvidia-smi`'s "CUDA Version" field reports the **maximum CUDA runtime the
driver supports**, not the toolkit llama.cpp was built with. v1 was built with
**CUDA 12.6** (`CUDACXX=/usr/local/cuda-12.6/bin/nvcc`, see "Build flags"
below). `nvcc` was not on `PATH` when `collect_env.sh` ran, which is why the
toolkit version is absent from this capture.

```
collect_env.sh: line 35: nvcc: command not found

Tue Apr 21 08:00:30 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
```

## llama.cpp
```
commit    : 97895129e5f2bde94d13dc01ca41ee79e9b629f2
short     : 9789512
describe  : N/A
authored  : 2026-04-20 23:30:38 +0200
subject   : ggml-cuda: flush legacy pool on OOM and retry (#22155)
```

## Models
```
21G /home/reachym/benchmarks/models/qwen3.6-ud-q4kxl/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
508M /home/reachym/benchmarks/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf
379M /home/reachym/benchmarks/models/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf
ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a  Qwen3-0.6B-Q4_K_M.gguf
bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517  Qwen3.5-0.8B-Q4_K_M.gguf
707a55a8a4397ecde44de0c499d3e68c1ad1d240d1da65826b4949d1043f4450  Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
```

## Python packages (venv)
```
Python 3.12.3
huggingface_hub==1.3.0
matplotlib==3.10.8
numpy==2.4.4
requests==2.33.1
urllib3==2.6.3
```

## Build flags (for reference)
```
cmake flags used:
  -DGGML_CUDA=ON
  -DCMAKE_CUDA_ARCHITECTURES=86   # RTX 3090 SM 8.6
  -DLLAMA_CURL=OFF
  -DBUILD_SHARED_LIBS=OFF
  CUDACXX=/usr/local/cuda-12.6/bin/nvcc
```

## Server invocation template
```
llama-server \
  -m Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --host 127.0.0.1 --port 18123 \
  -ngl 999 -c 16384 --jinja \
  -fa on -ctk q8_0 -ctv q8_0 --no-webui
  # + per-config spec-decode flags (see run_p0_matrix.sh / run_matrix.sh)
```

## Environment variables at bench time
```
HOME=/home/reachym
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin
```

---

# v2 Benchmark Environment (follow-up bench, 2026-04-22)

_For the artefacts in `v2_3090_followup/`._

⚠️ **This is a different physical 3090 from the v1 bench.** v1 ran on
the s1 box (2× RTX 3090, i7-11700); v2 ran on a single-3090 box stood
up the same day to respond to the HF-discussion critique. Baseline
differs by +3 % (v2: 139.9 tok/s vs v1: 135.7) which is within normal
board-to-board variance and documented in the v2 reply.

## Hardware
```
index, name, memory.total [MiB], driver_version, compute_cap
0, NVIDIA GeForce RTX 3090, 24576 MiB, 580.126.09, 8.6
```

## GPU state at bench start
```
clocks.current.graphics : 1965 MHz
clocks.max.graphics     : 2100 MHz
clocks.current.memory   : 9751 MHz
clocks.max.memory       : 9751 MHz
power.limit             : 350.00 W
power.default_limit     : 350.00 W
```

**Stock clocks — no overclocking.** GPU is at the factory-default power
limit of 350 W. Relative measurements between configs are invariant to
OC within ~±20 %; absolute numbers would scale with memory-bandwidth OC.

## OS / toolchain
- Ubuntu 24.04
- gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
- CUDA 12.0.140 (system-installed via apt) - note this differs from v1's
  CUDA 12.6 build toolkit; v1 and v2 binaries are not the same build
- python 3.12.3

## llama.cpp commits tested
| tag | commit | notes |
|---|---|---|
| original | `97895129e5f2bde94d13dc01ca41ee79e9b629f2` | equivalent to v1's `9789512` short hash — same commit, post PR #19493 |
| master | `bcb5eeb64` (as of 2026-04-22) | includes PR #22227 `speculative-simple: add checkpoint support` — the most recent spec-decode change |

Both commits tested with identical configs; results agree within ±0.3 %
noise (see `v2_3090_followup/SUMMARY.md` for the cross-check table).

## Build flags
```
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -GNinja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j $(nproc) --target llama-cli
```

## Models (same files as v1)
```
21G  ~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
508M ~/models/Qwen3.5-0.8B-Q4_K_M.gguf
```
SHA-256 matches v1 (see `v2_3090_followup/v2_oleg_suggestions/01_baseline/p1.log`
header for the `general.architecture` / `tokenizer.ggml.tokens` fields).

## Bench tool

> [!CAUTION]
> **This section was wrong.** It read: "v2 uses `llama-cli -st -no-cnv`
> (single-turn non-conversational)". `-no-cnv` is **not supported by
> `llama-cli`** on this build. 61 of the 62 committed v2 logs contain
> `--no-conversation is not supported by llama-cli / please use
> llama-completion instead`, and the same logs then show `[Start thinking]`
> and a full reasoning trace despite the `/no_think` suffix. The measured v2
> workload is long chain-of-thought output in llama-cli's default mode. The
> working thinking switches on this build are `-rea off` and
> `--reasoning-budget 0`; the working non-conversational tool is
> `llama-completion`.

v2 uses `llama-cli` with `-st` rather than v1's `llama-server` + Python client.
Both report `[Prompt: X t/s | Generation: Y t/s]` at end-of-run. Differences in
observed means between v1 and v2 come from a different host, a different
prompt set, a different sampling temperature, and a different output cap - not
from one controlled variable.

## Script
`v2_3090_followup/bench_3090_oleg.sh` reproduces the v2 run on any
single-3090 box with the model files + a llama-cli binary present.

---

## v3 environment snapshot (DFlash, 2026-05-07)

_v3 (DFlash via llama.cpp PR #22105) collected at 2026-05-07T18:33:41Z on
the **same** physical `3090` host as v2 (Tailscale name `3090`, hostname
`3090`, IP `100.112.135.98`). What changed vs v2: llama.cpp commit
(`bcb5eeb64` master → `67cb0d507` on the PR #22105 branch), drafter
(z-lab DFlash drafter converted to GGUF added), and a separate convert
venv with transformers/torch/gguf for the one-shot HF→GGUF step._

### Hardware

```
$ nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv
index, name, memory.total [MiB], driver_version, compute_cap
0, NVIDIA GeForce RTX 3090, 24576 MiB, 580.126.09, 8.6

# Same single-3090 host as v2 (`3090` Tailscale node).

$ nvidia-smi --query-gpu=clocks.current.graphics,clocks.max.graphics,clocks.current.memory,clocks.max.memory,power.limit,power.default_limit,power.max_limit --format=csv
clocks.current.graphics [MHz], clocks.max.graphics [MHz], clocks.current.memory [MHz], clocks.max.memory [MHz], power.limit [W], power.default_limit [W], power.max_limit [W]
1965 MHz, 2100 MHz, 9751 MHz, 9751 MHz, 350.00 W, 350.00 W, 350.00 W
```

Stock card, no OC. Power limit pinned at default 350 W.

### Software

```
$ uname -srm
Linux 6.x.x x86_64

$ python3 --version
Python 3.12.3

$ nvcc --version | tail -1
Cuda compilation tools, release 12.0, V12.0.140
```

### llama.cpp build

```
$ cd ~/bench/llama.cpp
$ git remote -v
origin  https://github.com/ggml-org/llama.cpp.git (fetch)
origin  https://github.com/ggml-org/llama.cpp.git (push)

$ git fetch origin pull/22105/head:pr-22105
$ git checkout pr-22105
$ git log --oneline -1
67cb0d507 dflash: enable llama-cli & llama-server with np=1
$ git log --oneline -5
67cb0d507 dflash: enable llama-cli & llama-server with np=1
e344c4a71 dflash: remove redundant logic & correct bias naming
85a0089e6 dflash: add support for qwen3.5/3.6 moe models
0724d66e5 dflash: first working POC
91b03e4c9 Merge branch 'master' into pr/18039

$ cd build && cmake --build . --config Release -j$(nproc)
$ ls bin/
... llama-cli, llama-server, llama-speculative-simple, ...

$ ./bin/llama-cli --version | head -1
version: 8889 (bcb5eeb64) -- inherited from master at fork point, plus DFlash patch on top.
```

> [!CAUTION]
> **Do not trust this `--version` string.** The v3 run logs themselves report
> `build : b8942-67cb0d507` for the DFlash configs and `build : b8889-bcb5eeb64`
> for the baseline and Oleg configs. The `--version` above was captured before
> the DFlash rebuild. The logs are authoritative, and they show the v3
> comparison used **two different binaries** (ERRATA D4). A run manifest must
> record `git rev-parse HEAD` and the binary's `sha256sum`, not `--version`.

CUDA backend built with `-DGGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=86`,
`-DGGML_CCACHE=ON`. ccache cache reused from prior master build, so the
DFlash branch incremental rebuild took ~5 min (vs ~30+ min cold).

### Models

```
$ ls -la ~/models/
-rw-r--r--  1  21G  Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf       # target (same as v1/v2)
-rw-r--r--  1 508M  Qwen3.5-0.8B-Q4_K_M.gguf              # draft (same as v1/v2); vocab SIZE
                                                        # matches but llama.cpp's gate REJECTS the
                                                        # pair - see ERRATA A2
drwxr-xr-x  ...    qwen36-dflash/                         # DFlash drafter (HF safetensors)
-rw-r--r--  1 905M  qwen36-dflash.gguf                    # DFlash drafter, BF16 GGUF (converted)
drwxr-xr-x  ...    qwen36-target-meta/                    # tokenizer + config from Qwen/Qwen3.6-35B-A3B
                                                           # (used for `--target-model-dir` during
                                                           # convert_hf_to_gguf.py)
```

DFlash drafter source: `z-lab/Qwen3.6-35B-A3B-DFlash`, downloaded via
`~/.local/bin/hf download`.  Conversion HF → GGUF uses PR #22105's
modified `convert_hf_to_gguf.py` with the new `--target-model-dir` flag
(set to a directory containing the target model's `config.json`,
`tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`).

### Conversion venv (one-shot, CPU-only torch is fine)

```
$ python3 -m venv ~/dflash_convert_venv
$ ~/dflash_convert_venv/bin/pip install -r ~/bench/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
$ ~/dflash_convert_venv/bin/pip list | grep -iE "torch|transformers|gguf|sentencepiece|safetensors"
gguf               0.19.0
sentencepiece      0.2.1
torch              2.6.0+cpu
transformers       5.5.1
safetensors        0.7.0
```

### Disk

```
$ df -h ~/models/
Filesystem       Size  Used Avail Use% Mounted on
/dev/...         913G  ~620G ~250G  72% /
```

### Bench script

`bench_dflash.sh` (in `v3_dflash_2026_05_07/bench/`) — does NOT use
`set -euo pipefail` because `grep | tail` empty-match interaction with
`pipefail` killed the v2 master bench script when a config errored. The
v3 script tolerates per-config failures and continues to the next one.

### Reproduction smoke test

```
$ ~/bench/llama.cpp/build/bin/llama-cli \
    -m ~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    -md ~/models/qwen36-dflash.gguf \
    --dflash --draft-max 16 \
    -ngl 999 -c 4096 -fa on -ctk q8_0 -ctv q8_0 \
    -n 100 --temp 0.5 --seed 42 -no-cnv -st \
    -p "Why does the sky look blue? Answer in two sentences. /no_think"
...
[ Prompt: 449.4 t/s | Generation: 66.8 t/s ]
common_memory_breakdown_print: |   - CUDA0 (RTX 3090) | 24115 = 761 + (21445 = 20798 + 105 + 541) + 1908 |
```

GPU memory: ~20.8 GB model weights + ~1.9 GB CUDA workspace, fits comfortably in 24 GB.

---


---

## Audit addendum, 2026-08-25

State of the v2/v3 bench host (`3090`, Tailscale `100.112.135.98`) when the
audit re-probed it, read-only:

```
GPU        : 1 x NVIDIA GeForce RTX 3090, 24576 MiB, 82 MiB used, 0 % util
driver     : 580.173.02   (was 580.126.09 at v2/v3 bench time)
disk free  : 262 GiB
models     : Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf  707a55a8...f4450
             Qwen3.5-0.8B-Q4_K_M.gguf         bd258782...dc517
             qwen36-dflash.gguf               (959 MiB, BF16)
llama.cpp  : ~/bench/llama.cpp @ bcb5eeb64, branch pr-22105 present
toolchain  : nvcc, gcc, cmake, ninja present; upstream fetch works
```

Two facts established by execution rather than by reading:

1. `--spec-type` is registered `.set_examples({LLAMA_EXAMPLE_SERVER})` at both
   `97895129e` and `bcb5eeb64`. `llama-server --help` lists it;
   `llama-completion --help` does not. v3's "not in master, fail-fast" note was
   wrong (ERRATA D6).

2. The draft model's incompatibility is a single missing GGUF key.
   `Qwen/Qwen3.5-0.8B` has **no `generation_config.json`** upstream (HTTP 404),
   so `convert_hf_to_gguf.py` wrote no `tokenizer.ggml.bos_token_id`, and
   `llama-vocab.cpp:1838` substitutes the hard-coded GPT-2 legacy default `11`
   against the target's `248044`. Both models declare
   `add_bos_token = false`, so the field that gates speculation is one neither
   model uses when tokenising. Adding
   `--override-kv tokenizer.ggml.bos_token_id=int:248044` flips
   `vocab_cmpt` from `0` to `1`, which also proves the two token arrays are
   byte-identical - the per-token text comparison from id 5 to 248320 passes.
   See ERRATA A2.

---

# v4 controlled runs — environment snapshot (2026-08-25/26)

Runs A–L. Unlike v1–v3, every artefact below is hashed in each run's
`manifest.json`, so a claim about which binary or which model produced a number
is checkable from the committed data rather than from this document.

## Binaries and models

| artefact | sha256 |
|---|---|
| `llama-server` (`3737e4137`) | `b6a5c490bb932ffa9bf8a0d887f15eb0aade1d00a5e29b177a27249a2c539903` |
| `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (target) | `707a55a8a4397ecde44de0c499d3e68c1ad1d240d1da65826b4949d1043f4450` |
| `Qwen3.5-0.8B-Q4_K_M.gguf` (matched-vocabulary drafter) | `bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517` |
| `qwen36-dflash-master.gguf` (DFlash drafter, re-converted) | `0e6d95c6a2ef7a3aaa4f7d54c6307d8bea50b86621823ea90f5bf8b5d8446e30` |

Runs A and B additionally used `bcb5eeb64`; the abort evidence in
`v4_audit_2026_08_25/data/abort_evidence_bcb5eeb64.txt` names it.

## Memory policy, which is a variable here and not a constant

Derived from the committed manifests, not maintained by hand — the earlier
version of this table stopped at run L and did not name the policy that most of
the repository's runs, including every run behind the headline, actually used.

| runs | `-ngl` | `-c` | `--fit-target` | why |
|---|---|---|---|---|
| A, B, C, D, E, H, I2 | pinned `999` | 16384 | — | placement fixed by hand |
| J, J2 | unset (`-fit on`) | 16384 | default 1024 | the BF16 DFlash drafter only loads with `-ngl` unset; `common_fit_params` aborts when it is pinned |
| K, K1, L | unset (`-fit on`) | 8192 | 2048 | the default 1024 MiB margin is where the drafter has to live, and it does not fit — see the run K section |
| M1, M2, M3, M4, N, O, **O2**, P, Q, R, T, T3 | unset (`-fit on`) | 8192 | 3072 | 2048 still leaves the DFlash arm inside 630 MiB of headroom; 3072 is where every arm of the nine-method matrix starts reliably |

That last row is why runs K1 and L are excluded from cross-run comparisons of
`spec-dflash-n2`: they read about +21 % at `--fit-target 2048` where the 3072
runs read +25 to +27 %. Same binary, same drafter, different memory policy.

`-fit on` is applied to **every arm within a run**, baseline included, so
placement policy never differs between an arm and its control. Absolute rates
are therefore comparable within a run and not across runs that differ in this
table; every matrix carries its own baseline for that reason.

## Instrumentation

- continuous `nvidia-smi` trace including the throttle-reason bitmask for the
  whole of every session, one file per session. **Three schemas were used, and
  `bench/gpu_telemetry.sh` carried only one of them until 2026-08-26** — it
  produced one of the seventeen traces recorded during the audit; the other two
  forms lived inline in driver scripts. All three are in that file now, selected
  by its first argument, so any committed trace can be reproduced:

  | schema | fields | interval | traces | what reads it |
  |---|---:|---:|---:|---|
  | `full` | 19 | 5 s | 1 | ERRATA C4b's thermal table |
  | `compact` | 9 | 5 s | 12 | runs I/J through O2, and run T — including A16's thermal comparison |
  | `raw` | 10 | 1 s | 4 | runs T3, O3 and later |

  The trace belonging to a run shares its timestamp:
  `gpu_telemetry_<label>_<stamp>.csv` beside `matrix_<label>_<stamp>/`.
- per-request JSON with full `content` and `reasoning_content`, token ids via
  `logprobs`, `timings`, and a `thinking_suppressed` flag measured from the
  reasoning channel rather than assumed from the request flag
- request start/end timestamps, from which the achieved batch width
  (`max_in_flight`) is derived instead of taken from `--parallel`
- `-v` server logs per arm-run (kept on the bench host; not committed, they are
  ~100 MB per matrix)

## Harness

`bench/retest_runner.py`, driven by `BENCH_*` environment variables recorded in
each `manifest.json`: `BENCH_ARMS`, `BENCH_REPEATS`, `BENCH_MAX_TOKENS`,
`BENCH_THINK`, `BENCH_FIT`, `BENCH_FIT_TARGET`, `BENCH_CTX`,
`BENCH_CONCURRENCY`, `BENCH_FLAVOR`.

