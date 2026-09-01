# Instrumentation patches

Every measurement in this repository except run T was taken on a **stock**
llama.cpp build at `3737e4137`. Run T was not, and this directory holds the
difference so that anyone can see exactly what was changed and reproduce it.

## `checkpoint_timers.patch`

ERRATA A12 originally attributed a share of wall-clock to the speculative
checkpoint path using the interval between each checkpoint log line and the next
line in the log. That estimator is invalid, because the two messages sit on
opposite sides of the work they name:

| | order in `tools/server/server-context.cpp` |
|---|---|
| create | `ckpt.update_tgt()` at `:2965`, **then** the message at `:2970` |
| restore | the message at `:3822`, **then** `load_tgt()` at `:3824` and `load_dft()` at `:3827` |

so the same rule missed the create copy and captured the restore. The figure was
withdrawn rather than corrected, because log timestamps cannot recover it.

This patch puts `ggml_time_us()` around each of the four calls —
`update_tgt`, `update_dft`, `load_tgt`, `load_dft` — and prints the elapsed
microseconds on the existing log lines. Upstream had already left the create-side
timer in place, commented out, at `:2963` and `:2967`; this uncomments it and
adds the three it was missing.

15 insertions, 5 deletions, one file. It changes no control flow and no
arithmetic — only timing and logging.

## Applying it

```bash
cd <llama.cpp checkout at 3737e4137>
patch -p1 < checkpoint_timers.patch
cmake --build build -j --target llama-server
```

## What it does to provenance, and why that matters

`build/bin/llama-server` is an ~18 kB launcher. The server logic compiles into
`build/bin/libllama-server-impl.so`, and the launcher is **byte-identical**
between a stock build and an instrumented one:

```
instrumented libllama-server-impl.so : ce94855f4f2d82ba…
stock        libllama-server-impl.so : a0cbe4d04bcda3f8…
launcher, both                       : b6a5c490bb932ffa…
```

`b6a5c490…` is what every manifest in this repository recorded as
`server_sha256` before 2026-08-26. That field never identified the code that
answered the requests. The runner now hashes every shared object beside the
binary into `server_lib_sha256`, which is what distinguishes these two builds.
See ERRATA A15.

The llama.cpp working tree is restored to stock after the run. Nothing is
committed, pushed or proposed upstream from here.
