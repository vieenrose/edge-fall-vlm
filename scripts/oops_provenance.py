"""Reconstruct which OOPS source videos produced the existing 300 val/test clips.

build_oops.py selected clips deterministically (OOPS.csv shuffled with random_state=0,
first 150 down + 150 normal rows with a mapped source video), but oops_manifest.json never
recorded the source video per clip and the 47.9GB of sources were deleted after extraction.
This replays the exact selection to recover clip_id -> source video, so a future training
harvest can exclude those videos and stay leakage-free at the source-video level.

Writes data/real/oops/provenance.json:
  {"clips": {clip_id: {"video": oops_rel, "start": s, "end": e, "label": ...}},
   "eval_videos": sorted unique source videos backing the 300 eval clips}

    python3 scripts/oops_provenance.py
"""
from __future__ import annotations

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_oops import DOWN, NORMAL, load_tables

PER_CLASS = 150   # what the original extraction used
SEED = 0


def main():
    seg, itw2oops = load_tables()
    seg = seg.sample(frac=1.0, random_state=SEED)  # identical shuffle to build_oops.py

    clips, counts = {}, {"down": 0, "normal": 0}
    for r in seg.itertuples():
        if counts["down"] >= PER_CLASS and counts["normal"] >= PER_CLASS:
            break
        cls = "down" if r.label in DOWN else ("normal" if r.label in NORMAL else None)
        if cls is None or counts[cls] >= PER_CLASS:
            continue
        itw = r.path if str(r.path).endswith(".mp4") else f"{r.path}.mp4"
        oops_rel = itw2oops.get(itw)
        if not oops_rel:
            continue
        # original run had the full dataset on disk, so vpath.exists() was always True
        cid = f"{cls}_{counts[cls]:04d}"
        clips[cid] = {"video": oops_rel, "start": float(r.start), "end": float(r.end),
                      "label": "fall" if cls == "down" else "normal"}
        counts[cls] += 1

    # sanity: the replayed ids must exactly match the clip dirs on disk
    on_disk = {p.name for p in Path("data/real/oops/clips").iterdir() if p.is_dir()}
    replayed = set(clips)
    missing, extra = on_disk - replayed, replayed - on_disk
    print(f"replayed {len(clips)} clips {counts}; on disk {len(on_disk)}; "
          f"missing_from_replay={sorted(missing)[:5]} extra={sorted(extra)[:5]}")

    videos = sorted({c["video"] for c in clips.values()})
    by_split = {}
    for v in videos:
        by_split[v.split("/")[1]] = by_split.get(v.split("/")[1], 0) + 1
    print(f"eval clips draw on {len(videos)} unique source videos, by split: {by_split}")

    out = Path("data/real/oops/provenance.json")
    out.write_text(json.dumps({"seed": SEED, "per_class": PER_CLASS,
                               "replay_matches_disk": not missing and not extra,
                               "clips": clips, "eval_videos": videos}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
