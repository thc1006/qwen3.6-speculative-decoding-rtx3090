#!/usr/bin/env python3
"""Split A12's checkpoint timers into the synchronisation wait and the state work.

A12 times `ckpt.update_tgt` / `update_dft` / `load_tgt` / `load_dft`. Those call
`llama_state_seq_get_data_ext` / `set_data_ext`, and at 3737e4137 both begin
with `ctx->synchronize()` (llama-context.cpp:4083). So the published 39.07 s is
elapsed INSIDE the checkpoint API calls, synchronisation included, and 54.7 % is
an attribution to that boundary rather than a measurement of state-copy cost.

This drains the queue explicitly, immediately before each call, and times the
drain. The call's own internal synchronize() then finds nothing outstanding, so
what remains is the state work. Total work is unchanged - the wait happened
either way, a microsecond earlier - and that is checkable: the new `update_tgt`
is still sync + state, so it must reproduce A12's 39.07 s.

Field names are kept: `update_tgt` etc. still mean THE WHOLE CALL, so the
existing extractor reads the new logs unchanged. The `sync_*` fields are new,
and state = <call> - <sync>.

    python3 apply_split_timers.py <llama.cpp-tree>
"""
import sys
from pathlib import Path

EDITS = [
    # ---- create, target -----------------------------------------------------
    ('''                if (use_ckpt_tgt) {
                    //const int64_t t_start = ggml_time_us();

                    ckpt.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                    //const int64_t t_total = ggml_time_us() - t_start;
                    //printf("checkpoint total: %f ms\\n", t_total / 1000.0);

                    SLT_DBG(slot, "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %d, size = %.3f MiB, draft = %.3f MiB)\\n",
                            ckpt.pos_min, ckpt.pos_max, slot.prompt.n_tokens(),
                            (float) ckpt.size() / 1024 / 1024,
                            (float) ckpt.data_dft.size() / 1024 / 1024);
                }''',
     '''                if (use_ckpt_tgt) {
                    const int64_t t_start = ggml_time_us();

                    // drain first, and time the drain. update_tgt() calls
                    // llama_state_seq_get_data_ext(), which begins with
                    // ctx->synchronize(); doing it here does not add work, it
                    // separates the wait from the copy.
                    llama_synchronize(ctx_tgt);
                    const int64_t t_sync = ggml_time_us() - t_start;

                    ckpt.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                    const int64_t t_total = ggml_time_us() - t_start;

                    SLT_DBG(slot, "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %d, size = %.3f MiB, draft = %.3f MiB) AUDIT_US update_tgt=%lld sync_tgt=%lld\\n",
                            ckpt.pos_min, ckpt.pos_max, slot.prompt.n_tokens(),
                            (float) ckpt.size() / 1024 / 1024,
                            (float) ckpt.data_dft.size() / 1024 / 1024,
                            (long long) t_total, (long long) t_sync);
                }'''),
    # ---- create, drafter ----------------------------------------------------
    ('''                if (use_ckpt_dft) {
                    ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }''',
     '''                if (use_ckpt_dft) {
                    const int64_t t_start_dft = ggml_time_us();
                    llama_synchronize(ctx_dft);
                    const int64_t t_sync_dft = ggml_time_us() - t_start_dft;
                    ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    SLT_DBG(slot, "checkpoint update_dft AUDIT_US update_dft=%lld sync_dft=%lld\\n",
                            (long long) (ggml_time_us() - t_start_dft),
                            (long long) t_sync_dft);
                }'''),
    # ---- restore ------------------------------------------------------------
    ('''                        ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                        if (slot.ctx_dft) {
                            ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        }''',
     '''                        const int64_t t_start_lt = ggml_time_us();
                        llama_synchronize(slot.ctx_tgt);
                        const int64_t t_sync_lt = ggml_time_us() - t_start_lt;
                        ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        const int64_t t_lt = ggml_time_us() - t_start_lt;

                        int64_t t_ld = 0;
                        int64_t t_sync_ld = 0;
                        if (slot.ctx_dft) {
                            const int64_t t_start_ld = ggml_time_us();
                            llama_synchronize(slot.ctx_dft);
                            t_sync_ld = ggml_time_us() - t_start_ld;
                            ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                            t_ld = ggml_time_us() - t_start_ld;
                        }
                        SLT_DBG(slot, "restored speculative checkpoint AUDIT_US load_tgt=%lld sync_lt=%lld load_dft=%lld sync_ld=%lld\\n",
                                (long long) t_lt, (long long) t_sync_lt,
                                (long long) t_ld, (long long) t_sync_ld);'''),
]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    f = Path(sys.argv[1]) / "tools" / "server" / "server-context.cpp"
    src = f.read_text(encoding="utf-8")
    if "AUDIT_US" in src:
        sys.exit("that tree already carries an AUDIT_US patch; check it out clean first")
    for i, (old, new) in enumerate(EDITS, 1):
        if src.count(old) != 1:
            sys.exit(f"edit {i}: found {src.count(old)} matches, expected exactly 1")
        src = src.replace(old, new, 1)
    f.write_text(src, encoding="utf-8")
    print(f"  patched {f}")
    print(f"  AUDIT_US sites: {src.count('AUDIT_US')}")
    print("  llama_synchronize calls added: 4")


if __name__ == "__main__":
    main()
