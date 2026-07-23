"""Build REAL in-the-wild fall training data + a val-expansion eval set from harvested
OOPS source videos (data/real/oops_src/), excluding every video that backs an existing
eval clip (data/real/oops/provenance.json) — leakage-free at the source-video level.

The project's training set contained ZERO in-the-wild real fall positives (all synthetic
mannequins + 10 URFD lab clips) while the whole benchmark is in-the-wild OOPS — the
deep-review's binding-constraint diagnosis. This script is the direct attack.

Split discipline: source VIDEOS are assigned to either the val-expansion pool or the
training pool, never both. Every emitted clip records its source video + time window.

    python3 scripts/build_oops_train.py            # after the harvest download finishes
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_oops import DOWN, NORMAL, cut_strip, load_tables

PROMPT = ("You are a safety monitor. These are consecutive video frames (oldest first), "
          "possibly with more than one person. Report whether ANYONE has fallen, fainted, "
          "is lying immobile, or is in distress; else normal. Answer with JSON only.")
UPRIGHT = {0, 7, 8, 15}            # walk, stand_up, standing, jump
LOW = {3, 4, 10, 11, 12, 13}       # sit/kneel/squat family


def answer_for(raw_label: int) -> tuple[str, dict, str]:
    if raw_label in DOWN:
        return "fall", {"posture": "horizontal-on-floor", "status": "fall",
                        "confidence": 0.9, "person_down": True}, \
               "Real in-the-wild clip: person goes down and ends on the floor -> fall."
    posture = "upright-standing" if raw_label in UPRIGHT else "upright-low"
    return "normal", {"posture": posture, "status": "normal",
                      "confidence": 0.8, "person_down": False}, \
           "Real in-the-wild clip: activity continues without anyone down -> normal."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", type=Path, default=Path("data/real/oops_src"))
    ap.add_argument("--out", type=Path, default=Path("data/real/oops_train"))
    ap.add_argument("--val-down", type=int, default=75)
    ap.add_argument("--val-normal", type=int, default=75)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    seg, itw2oops = load_tables()
    eval_videos = set(json.load(open("data/real/oops/provenance.json"))["eval_videos"])

    by_video: dict[str, list] = {}
    for r in seg.itertuples():
        cls = "down" if r.label in DOWN else ("normal" if r.label in NORMAL else None)
        if cls is None:
            continue
        itw = r.path if str(r.path).endswith(".mp4") else f"{r.path}.mp4"
        rel = itw2oops.get(itw)
        if not rel or rel in eval_videos:
            continue
        if not (args.src_root / rel).exists():
            continue
        by_video.setdefault(rel, []).append((cls, int(r.label), float(r.start), float(r.end)))

    videos = sorted(by_video)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(videos)

    # assign whole videos to the val-expansion pool until its class targets are met
    val_videos, val_counts = set(), {"down": 0, "normal": 0}
    for v in videos:
        if val_counts["down"] >= args.val_down and val_counts["normal"] >= args.val_normal:
            break
        want = any((val_counts[c] < {"down": args.val_down, "normal": args.val_normal}[c])
                   for c, *_ in by_video[v])
        if not want:
            continue
        val_videos.add(v)
        for c, *_ in by_video[v]:
            val_counts[c] += 1

    train_rows, val_manifest, provenance = [], [], {}
    counts = {"train_down": 0, "train_normal": 0, "val_down": 0, "val_normal": 0}
    for v in videos:
        pool = "val" if v in val_videos else "train"
        for cls, raw_label, start, end in by_video[v]:
            key = f"{pool}_{cls}"
            cid = f"oops{pool}_{cls}_{counts[key]:04d}"
            clip_dir = args.out / ("val_clips" if pool == "val" else "clips") / cid
            frames = cut_strip(args.src_root / v, start, end, clip_dir)
            if len(frames) < 4:
                import shutil
                shutil.rmtree(clip_dir, ignore_errors=True)  # else a later clip with the
                continue                                     # same id inherits stale frames
            counts[key] += 1
            provenance[cid] = {"video": v, "start": start, "end": end, "label": cls, "pool": pool}
            if pool == "val":
                val_manifest.append({"id": cid, "frames_dir": str(clip_dir),
                                     "label": "fall" if cls == "down" else "normal",
                                     "split": "oops_itw_valx"})
            else:
                fine, ans, why = answer_for(raw_label)
                train_rows.append({"id": cid, "class": fine, "frames": frames,
                                   "prompt": PROMPT, "rationale": why, "answer": ans,
                                   "lighting": "day", "split_key": "oops_itw"})

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "train_samples.jsonl", "w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    Path("data/real/oops/oops_val_expansion.json").write_text(json.dumps(val_manifest, indent=2))
    (args.out / "provenance.json").write_text(json.dumps(
        {"seed": args.seed, "excluded_eval_videos": len(eval_videos), "counts": counts,
         "val_videos": sorted(val_videos), "clips": provenance}, indent=2))
    print(f"train rows: {len(train_rows)} ({counts['train_down']} fall / "
          f"{counts['train_normal']} normal) -> {args.out}/train_samples.jsonl")
    print(f"val-expansion: {len(val_manifest)} ({counts['val_down']}/{counts['val_normal']}) "
          f"-> data/real/oops/oops_val_expansion.json")
    print(f"videos: {len(videos)} harvested, {len(val_videos)} in val pool "
          f"(video-level split, eval-video exclusion={len(eval_videos)})")


if __name__ == "__main__":
    main()
