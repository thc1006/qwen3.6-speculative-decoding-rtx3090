"""Stage the AWQ checkpoint so `convert_hf_to_gguf.py --mtp` can read it.

Why this exists
---------------
`--mtp` exports only the multi-token-prediction head plus the shared embedding,
final norm and lm_head. In this checkpoint all 788 of those tensors are stored
as plain BF16 — nothing in the export set is AWQ-packed, while 89 856 tensors
elsewhere in the same checkpoint are. But `conversion/base.py:dequant_model()`
dispatches on `config.json`'s `quantization_config.quant_method`, not on the
tensors actually being exported, so it raises

    NotImplementedError: Quant method is not yet supported: 'awq'

before reaching them.

This script builds a staging directory that symlinks the shards and tokenizer
untouched and writes one modified `config.json` with `quantization_config`
removed. It does NOT modify llama.cpp and it does not modify the checkpoint.

It refuses to run unless every tensor in the export set is verified unquantised
first, because if any of them were packed, dropping the key would silently emit
garbage rather than fail.

Env: MTP_SRC (default ~/models/qwen36-awq), MTP_STAGE (default ~/models/qwen36-mtp-src)
"""
from __future__ import annotations

import json
import os
import struct
import sys
from collections import Counter
from pathlib import Path

SRC = Path(os.environ.get("MTP_SRC", Path.home() / "models/qwen36-awq"))
STAGE = Path(os.environ.get("MTP_STAGE", Path.home() / "models/qwen36-mtp-src"))
SHARED = ("lm_head.weight", "model.language_model.embed_tokens.weight",
          "model.language_model.norm.weight", "model.embed_tokens.weight",
          "model.norm.weight")


def export_set(index: dict[str, str]) -> list[str]:
    return [k for k in index if k.startswith("mtp.")] + [k for k in index if k in SHARED]


def dtypes_of(keys: list[str], index: dict[str, str]) -> Counter:
    by_file: dict[str, list[str]] = {}
    for k in keys:
        by_file.setdefault(index[k], []).append(k)
    seen = Counter()
    for fn, ks in by_file.items():
        with (SRC / fn).open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k in ks:
            seen[hdr[k]["dtype"]] += 1
    return seen


def main() -> None:
    idx_files = sorted(SRC.glob("*.index.json"))
    if not idx_files:
        sys.exit(f"no safetensors index in {SRC}")
    index = json.loads(idx_files[0].read_text())["weight_map"]

    keys = export_set(index)
    if not any(k.startswith("mtp.") for k in keys):
        sys.exit("checkpoint carries no mtp.* tensors; nothing to export")
    seen = dtypes_of(keys, index)
    packed = [k for k in keys if k.endswith(("qweight", "qzeros", "scales"))]
    print(f"  export set: {len(keys)} tensors, dtypes {dict(seen)}")
    if packed or set(seen) - {"BF16", "F16", "F32"}:
        sys.exit(f"REFUSING: {len(packed)} packed tensors / dtypes {dict(seen)} in the "
                 "export set. Dropping quantization_config would emit garbage.")
    print("  verified: every tensor in the export set is unquantised")

    cfg = json.loads((SRC / "config.json").read_text())
    removed = cfg.pop("quantization_config", None)
    if removed is None:
        print("  note: config.json had no quantization_config; staging is a no-op copy")

    STAGE.mkdir(parents=True, exist_ok=True)
    for f in SRC.iterdir():
        dst = STAGE / f.name
        if f.name == "config.json":
            continue
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(f.resolve())
    (STAGE / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (STAGE / "STAGING_NOTE.txt").write_text(
        "Symlinks to " + str(SRC) + " with one change: config.json has\n"
        "quantization_config removed (was quant_method=" 
        + str((removed or {}).get("quant_method")) + ").\n"
        "Justified because all " + str(len(keys)) + " tensors in the --mtp export set\n"
        "are BF16; the AWQ guard in conversion/base.py dispatches on the config key,\n"
        "not on the tensors being exported. llama.cpp itself is unmodified.\n")
    print(f"  staged {STAGE}  (quantization_config removed: {bool(removed)})")


if __name__ == "__main__":
    main()
