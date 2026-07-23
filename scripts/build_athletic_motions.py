"""Build a skewed enrichment motion set of ONLY athletic/inverted NORMAL poses
(handstand, cartwheel, bridge, vault) -- the horizontal-but-normal hard-negatives the
existing gen_normal set lacks. Follows the project's proven 'dedicated skewed enrichment'
pattern. Output feeds blender_dataset.py for 3D+projection rendering.

    python3 scripts/build_athletic_motions.py --n 240 --out data/synth3d/motions_athletic
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.convert_motion import gen_athletic_normal
from synthgen.rationale import JOINTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--out", type=Path, default=Path("data/synth3d/motions_athletic"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    mdir = args.out / "motions"; mdir.mkdir(parents=True, exist_ok=True)
    manifest, counts = [], {}
    for i in range(args.n):
        joints, sub = gen_athletic_normal(rng)
        clip_id = f"normal_{sub}_{i:04d}"
        path = mdir / f"{clip_id}.npz"
        np.savez(path, fps=30, intended_class="normal",
                 **{f"joint_{n}": joints[n] for n in JOINTS})
        manifest.append({"clip_id": clip_id, "path": str(path), "class": "normal", "fps": 30})
        counts[sub] = counts.get(sub, 0) + 1
    (args.out / "motion_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"built {len(manifest)} athletic-normal motions {counts} -> {args.out}")


if __name__ == "__main__":
    main()
