"""Build HARD-NEGATIVE training clips: real 'horizontal-but-normal' bodies labeled NORMAL.

The realfall model over-triggers "down" on any non-upright body (gymnastics inverted mid-air,
a leaning cyclist, a slip that recovers, someone reclining) — all 25 of its val+test false
alarms reported posture "horizontal-on-floor". This mines the OOPS labels build_oops.py
EXCLUDES — other(9), lie_down(5), lying(6), crawl(14) — which the OmniFall scheme separates
from the actual falls fall(1)/fallen(2), i.e. annotator-judged NON-fall activity. Labeled
NORMAL and added to training via continual-FT, these teach "horizontal/tilted body != fallen".

Leakage-free: only extracts from videos NOT backing any eval clip (data/real/oops/provenance).

    python3 scripts/build_oops_hardneg.py   # after the hardneg harvest completes
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_oops import cut_strip, load_tables

HARDNEG = {5: "lie_down", 6: "lying", 9: "other", 14: "crawl"}
PROMPT = ("You are a safety monitor. These are consecutive video frames (oldest first), "
          "possibly with more than one person. Report whether ANYONE has fallen, fainted, "
          "is lying immobile, or is in distress; else normal. Answer with JSON only.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", type=Path, default=Path("data/real/oops_hardneg"))
    ap.add_argument("--out", type=Path, default=Path("data/real/oops_hardneg_clips"))
    ap.add_argument("--max", type=int, default=0, help="cap #clips (0 = all)")
    args = ap.parse_args()

    seg, itw2oops = load_tables()
    eval_videos = set(json.load(open("data/real/oops/provenance.json"))["eval_videos"])

    rows = []
    prov = {}
    counts = {}
    for r in seg.itertuples():
        if r.label not in HARDNEG:
            continue
        itw = r.path if str(r.path).endswith(".mp4") else f"{r.path}.mp4"
        rel = itw2oops.get(itw)
        if not rel or rel in eval_videos:
            continue
        vpath = args.src_root / rel
        if not vpath.exists():
            continue
        kind = HARDNEG[r.label]
        idx = counts.get(kind, 0)
        cid = f"oopshn_{kind}_{idx:04d}"
        clip_dir = args.out / "clips" / cid
        frames = cut_strip(vpath, float(r.start), float(r.end), clip_dir)
        if len(frames) < 4:
            shutil.rmtree(clip_dir, ignore_errors=True)
            continue
        counts[kind] = idx + 1
        prov[cid] = {"video": rel, "start": float(r.start), "end": float(r.end),
                     "omnifall_label": kind}
        # posture varies but the verdict is always NORMAL — that is the whole point
        rows.append({"id": cid, "class": "normal", "frames": frames, "prompt": PROMPT,
                     "rationale": f"Real in-the-wild clip: body may be low/horizontal/dynamic "
                                  f"({kind}) but nobody has fallen -> normal.",
                     "answer": {"posture": "non-upright-normal", "status": "normal",
                                "confidence": 0.8, "person_down": False},
                     "lighting": "day", "split_key": "oops_hardneg"})
        if args.max and len(rows) >= args.max:
            break

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "hardneg_samples.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    (args.out / "provenance.json").write_text(json.dumps(
        {"excluded_eval_videos": len(eval_videos), "counts": counts, "clips": prov}, indent=2))
    print(f"hard-negative clips: {len(rows)} by kind {counts} -> "
          f"{args.out}/hardneg_samples.jsonl")


if __name__ == "__main__":
    main()
