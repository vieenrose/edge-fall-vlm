"""Turn 6-frame strips into single mosaic images + labels, for an image-CLASSIFIER
fall detector that can be exported to ONNX (optimum supports image-classification natively,
unlike VLMs) and run live in-browser via transformers.js WebGPU.

Each strip -> one 3x2 grid image (compose_mosaic, the project's existing tiling), labeled
down3 {down, distress, normal}. Emits an HF imagefolder layout: <out>/<label>/<id>.jpg.

    python3 scripts/build_mosaic_dataset.py --samples data/train_qwen_realhn_x1.jsonl --out data/mosaic/train
    python3 scripts/build_mosaic_dataset.py --manifest data/real/oops/oops_val.json --out data/mosaic/val
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.vlm_backend import compose_mosaic
from training.dataset import load_records, collapse_label
from scripts.validate_real import load_manifest_as_samples
from training.sft import _subsample

TILE = 224  # final mosaic is 3*TILE_w wide; each frame downscaled — keep small for WebGPU


def strip_to_mosaic(frame_paths, max_frames=6, tile=112):
    imgs = []
    for p in _subsample(list(frame_paths), max_frames):
        im = Image.open(p).convert("RGB").resize((tile, tile))
        imgs.append(np.asarray(im))
    return compose_mosaic(imgs, cols=3)   # 3x2 grid -> (3*tile, 2*tile)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, help="training jsonl (class field)")
    ap.add_argument("--manifest", type=Path, help="real eval manifest")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label-set", default="down3")
    ap.add_argument("--tile", type=int, default=112)
    args = ap.parse_args()

    if args.samples:
        samples = load_records(args.samples, label_set=args.label_set)
    else:
        samples = load_manifest_as_samples(args.manifest, args.label_set)

    counts = {}
    for s in samples:
        label = s.label
        d = args.out / label
        d.mkdir(parents=True, exist_ok=True)
        try:
            mosaic = strip_to_mosaic(s.frames, tile=args.tile)
        except Exception:
            continue
        mosaic.save(d / f"{s.id}.jpg", quality=88)
        counts[label] = counts.get(label, 0) + 1
    print(f"mosaic dataset -> {args.out}  {counts}")


if __name__ == "__main__":
    main()
