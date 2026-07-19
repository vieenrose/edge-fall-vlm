"""Build an in-the-wild OOPS-Fall validation set from OmniFall's OOPS labels.

Cuts labeled segments out of the source OOPS videos (via cv2), samples a 6-frame strip
each, and writes a manifest for validate_real.py. This is the standardized in-the-wild
test OmniFall reports baselines on (VideoMAE Fallen sensitivity 0.21, frozen-I3D 0.68) —
so our number here is directly comparable.

Mapping to our down3 scheme:
  DOWN    = OmniFall {1 fall, 2 fallen}                      (person on the floor)
  NORMAL  = {0 walk, 3 sit_down, 4 sitting, 7 stand_up, 8 standing,
             10 kneel_down, 11 kneeling, 12 squat_down, 13 squatting, 15 jump}
  (excluded ambiguous: 5 lie_down, 6 lying, 9 other, 14 crawl)

    python scripts/build_oops.py --video-root data/real/oops --per-class 150
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import hf_hub_download

DOWN = {1, 2}
NORMAL = {0, 3, 4, 7, 8, 10, 11, 12, 13, 15}
STRIP = 6


def load_tables():
    seg = hf_hub_download("simplexsigil2/omnifall", "labels/OOPS.csv", repo_type="dataset")
    mp = hf_hub_download("simplexsigil2/omnifall", "data_files/oops_video_mapping.csv", repo_type="dataset")
    import pandas as pd
    seg = pd.read_csv(seg)
    mp = pd.read_csv(mp)
    # OOPS.csv path e.g. "falls/..24" ; mapping itw_path "falls/..24.mp4" -> oops_path
    itw2oops = {r.itw_path: r.oops_path for r in mp.itertuples()}
    return seg, itw2oops


def cut_strip(video_path: Path, start: float, end: float, out_dir: Path) -> list[str]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    f0 = max(0, int(start * fps))
    f1 = min(n - 1 if n else int(end * fps), int(end * fps))
    if f1 <= f0:
        f1 = f0 + STRIP
    idx = np.linspace(f0, f1, STRIP).astype(int)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for k, fi in enumerate(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        p = out_dir / f"f{k:02d}.png"
        cv2.imwrite(str(p), frame)
        paths.append(str(p))
    cap.release()
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--per-class", type=int, default=150)
    ap.add_argument("--out", type=Path, default=Path("data/real/oops"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seg, itw2oops = load_tables()
    rng = np.random.default_rng(args.seed)
    seg = seg.sample(frac=1.0, random_state=args.seed)  # shuffle

    manifest, counts = [], {"down": 0, "normal": 0}
    frames_root = args.out / "clips"
    for r in seg.itertuples():
        if counts["down"] >= args.per_class and counts["normal"] >= args.per_class:
            break
        cls = "down" if r.label in DOWN else ("normal" if r.label in NORMAL else None)
        if cls is None or counts[cls] >= args.per_class:
            continue
        itw = r.path if str(r.path).endswith(".mp4") else f"{r.path}.mp4"
        oops_rel = itw2oops.get(itw)
        if not oops_rel:
            continue
        vpath = args.video_root / oops_rel
        if not vpath.exists():
            continue
        cid = f"{cls}_{counts[cls]:04d}"
        strip = cut_strip(vpath, float(r.start), float(r.end), frames_root / cid)
        if len(strip) < 2:
            continue
        manifest.append({"id": cid, "frames_dir": str(frames_root / cid),
                         "label": "fall" if cls == "down" else "normal",
                         "split": "oops_itw"})
        counts[cls] += 1

    (args.out / "oops_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"OOPS in-the-wild: {len(manifest)} clips {counts} -> oops_manifest.json")


if __name__ == "__main__":
    main()
