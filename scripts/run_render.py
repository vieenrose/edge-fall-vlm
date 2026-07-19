"""CLI entry for the synthetic render pipeline.

Run INSIDE BlenderProc (which supplies bpy):

    blenderproc run scripts/run_render.py \
        --manifest data/motion_manifest.json \
        --out data/synth_shards \
        --seed 0

The manifest is a JSON list of motion clips:
    [{"clip_id": "lafan1_fall_0001", "path": "motions/lafan1_fall_0001.npz",
      "class": "fall", "fps": 30}, ...]

For a dry-run of the pure logic without Blender, use scripts/dryrun.py instead.
"""
import argparse
import sys
from pathlib import Path

# make the repo importable when invoked via `blenderproc run`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthgen.config import DEFAULT, PipelineCfg
from synthgen.render import render_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = PipelineCfg(render=DEFAULT.render, lighting=DEFAULT.lighting,
                      domain=DEFAULT.domain, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    render_manifest(args.manifest, args.out, cfg)


if __name__ == "__main__":
    main()
