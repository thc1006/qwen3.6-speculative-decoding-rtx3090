"""Reconstruct llama.cpp's speculative-decoding accounting from a `-v` run log.

Why this script exists
----------------------
Until the 2026-08-25 audit, this repository reported "100 % draft acceptance"
as an empirical finding, citing the line

    draft acceptance rate = 1.00000 (  115 accepted /   115 generated)

from `v2_3090_followup/v2_oleg_suggestions/verbose.log`.

That ratio is a tautology for this model on this build, not a measurement.
In `tools/server/server-context.cpp` at commit 97895129e, the per-request
counters `n_draft_total` / `n_draft_accepted` are updated *after* an early
`continue`:

    if (accepted.size() < slot.spec_draft.size() + 1) {          // partial accept
        if (slot.ctx_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            slot.spec_draft = std::move(accepted);               // truncate
            ... restore checkpoint ...
            continue;                                            // <-- counters skipped
        }
    }
    ...
    slot.n_draft_accepted += ids.size() - 1;
    slot.n_draft_total    += n_draft;

Qwen3.6-35B-A3B is a hybrid Gated-DeltaNet/MoE model, so
`common_context_can_seq_rm()` returns COMMON_CONTEXT_SEQ_RM_TYPE_FULL ("the
target context does not support partial sequence removal"). Every partially
accepted round therefore takes the `continue` branch, is dropped from both
numerator and denominator, and is re-verified on the next pass against the
truncated — already-known-accepted — prefix. Only rounds where the whole draft
was accepted ever reach the counters, so the reported ratio can only be 1.0.

The drafter keeps its own, honest counters and prints them on the very next
line of the same log:

    statistics draft: ... #gen drafts = 81, #acc drafts = 33,
                          #gen tokens = 214, #acc tokens = 115, ...

Run: python analysis/verbose_accounting.py [path/to/verbose.log ...]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOGS = [ROOT / "v2_3090_followup/v2_oleg_suggestions/verbose.log"]
OUT_JSON = ROOT / "analysis" / "verbose_accounting.json"

RE_ATTEMPT = re.compile(r"update_slots: n_draft=(\d+), accepted=(\d+)")
RE_COUNTED = re.compile(r"accepted (\d+)/(\d+) draft tokens")
RE_REPORTED = re.compile(
    r"draft acceptance rate = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated\)")
RE_DRAFTER = re.compile(
    r"statistics draft: #calls\(b,g,a\) = (\d+) (\d+) (\d+), "
    r"#gen drafts = (\d+), #acc drafts = (\d+), "
    r"#gen tokens = (\d+), #acc tokens = (\d+), "
    r"dur\(b,g,a\) = ([\d.]+), ([\d.]+), ([\d.]+) ms")
RE_CKPT_NEW = re.compile(
    r"created speculative checkpoint \(pos_min = \d+, pos_max = \d+, "
    r"n_tokens = \d+, size = ([\d.]+) MiB\)")
RE_CKPT_OLD = re.compile(
    r"restoring speculative checkpoint \(pos_min = \d+, pos_max = \d+, size = (\d+)\)")
RE_RATE = re.compile(r"\[ Prompt: ([\d.]+) t/s \| Generation: ([\d.]+) t/s \]")
RE_SEQRM = re.compile(r"the target context does not support partial sequence removal")
RE_VOCAB = re.compile(r"the target and draft vocabs are not compatible")
RE_BUILD = re.compile(r"^build\s+:\s+(\S+)", re.M)
# `-n 200` is what every bench script in this repository passes, but derive the
# figure from the log instead of trusting that: the server prints
# "n_decoded = X, n_remaining = Y" per token, and X + Y is the request's total.
RE_DECODED = re.compile(r"n_decoded = (\d+), n_remaining = (\d+)")


def analyse(path: Path) -> dict:
    text = path.read_text(errors="replace")

    attempts = [(int(a), int(b)) for a, b in RE_ATTEMPT.findall(text)]
    counted = [(int(a), int(b)) for a, b in RE_COUNTED.findall(text)]
    full = [x for x in attempts if x[1] == x[0] + 1]
    partial = [x for x in attempts if x[1] < x[0] + 1]

    out: dict = {
        "log": str(path.relative_to(ROOT)),
        "build": (m.group(1) if (m := RE_BUILD.search(text)) else None),
        "hybrid_seq_rm_full": bool(RE_SEQRM.search(text)),
        "vocab_translation_fallback": bool(RE_VOCAB.search(text)),
        "verification_attempts": len(attempts),
        "attempts_fully_accepted": len(full),
        "attempts_partially_accepted": len(partial),
        "rounds_reaching_counters": len(counted),
    }

    if m := RE_REPORTED.search(text):
        out["reported"] = {
            "ratio": float(m.group(1)),
            "accepted": int(m.group(2)),
            "generated": int(m.group(3)),
            "meaning": "tokens re-verified after truncation to the accepted prefix; "
                       "ratio is 1.0 by construction on a SEQ_RM_TYPE_FULL context",
        }

    if m := RE_DRAFTER.search(text):
        gd, ad, gt, at = (int(m.group(i)) for i in (4, 5, 6, 7))
        out["drafter_own_counters"] = {
            "drafts_generated": gd,
            "drafts_accepted": ad,
            "draft_sequence_acceptance_pct": round(100 * ad / gd, 1) if gd else None,
            "draft_tokens_generated": gt,
            "draft_tokens_accepted": at,
            "draft_token_acceptance_pct": round(100 * at / gt, 1) if gt else None,
            "duration_begin_ms": float(m.group(8)),
            "duration_generate_ms": float(m.group(9)),
            "duration_accept_ms": float(m.group(10)),
        }

    created = [float(x) for x in RE_CKPT_NEW.findall(text)]
    restored = [int(x) for x in RE_CKPT_OLD.findall(text)]
    out["state_management"] = {
        "checkpoints_created": len(created),
        "checkpoint_mib_each": round(created[0], 1) if created else None,
        "checkpoint_gib_written": round(sum(created) / 1024, 2),
        "checkpoints_restored": len(restored),
        "checkpoint_gib_read_back": round(sum(restored) / 2 ** 30, 2),
    }

    if m := RE_RATE.search(text):
        gen_rate = float(m.group(2))
        out["reported_generation_tok_s"] = gen_rate
        dg = out.get("drafter_own_counters", {}).get("duration_generate_ms")
        dec = RE_DECODED.findall(text)
        if dec:
            n_gen = max(int(a) + int(b) for a, b in dec)
            out["n_generated_tokens_source"] = "n_decoded + n_remaining, read from the log"
        else:
            n_gen = 200
            out["n_generated_tokens_source"] = "assumed -n 200; no n_decoded lines in this log"
        out["n_generated_tokens"] = n_gen
        wall_ms = 1000 * n_gen / gen_rate
        out["generation_wall_ms"] = round(wall_ms)
        if dg:
            out["drafter_share_of_generation_pct"] = round(100 * dg / wall_ms, 1)

    return out


def render(rep: dict) -> None:
    print("=" * 74)
    print(f"speculative accounting :: {rep['log']}")
    print(f"build {rep['build']}  |  hybrid SEQ_RM_TYPE_FULL: {rep['hybrid_seq_rm_full']}"
          f"  |  vocab-translation fallback: {rep['vocab_translation_fallback']}")
    print("=" * 74)

    r, d = rep.get("reported"), rep.get("drafter_own_counters")
    if r:
        print(f"reported by the server counter : {r['ratio']:.5f} "
              f"({r['accepted']} accepted / {r['generated']} generated)")
    if d:
        print(f"drafter's own token counter    : "
              f"{d['draft_token_acceptance_pct']:.1f}% "
              f"({d['draft_tokens_accepted']} accepted / {d['draft_tokens_generated']} generated)")
        print(f"drafter's own sequence counter : "
              f"{d['draft_sequence_acceptance_pct']:.1f}% "
              f"({d['drafts_accepted']} accepted / {d['drafts_generated']} generated)")
    print()
    print(f"verification attempts          : {rep['verification_attempts']}")
    print(f"  fully accepted -> counted    : {rep['attempts_fully_accepted']}")
    print(f"  partial -> discarded + redone: {rep['attempts_partially_accepted']}")
    print()
    s = rep["state_management"]
    print(f"speculative checkpoints        : {s['checkpoints_created']} created "
          f"@ {s['checkpoint_mib_each']} MiB = {s['checkpoint_gib_written']} GiB written")
    print(f"                                 {s['checkpoints_restored']} restored "
          f"= {s['checkpoint_gib_read_back']} GiB read back")
    if d and "drafter_share_of_generation_pct" in rep:
        print()
        print(f"drafter generate() time        : {d['duration_generate_ms']:.0f} ms "
              f"= {rep['drafter_share_of_generation_pct']}% of the "
              f"{rep['generation_wall_ms']} ms generation wall-clock "
              f"({rep['reported_generation_tok_s']} tok/s x {rep['n_generated_tokens']} tokens, "
              f"{rep['n_generated_tokens_source']})")
    print()


def main() -> None:
    paths = [Path(a) for a in sys.argv[1:]] or DEFAULT_LOGS
    reports = []
    for p in paths:
        if not p.exists():
            print(f"  skip (missing): {p}", file=sys.stderr)
            continue
        rep = analyse(p)
        reports.append(rep)
        render(rep)
    if reports:
        OUT_JSON.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
