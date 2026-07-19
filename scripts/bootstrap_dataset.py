"""Build a Blender-free bootstrap training set from procedural motions.

motion .npz -> label/rationale (once) -> K sampled cameras -> GT projection ->
stick-figure PNG strips -> samples.jsonl (one record per view).

This produces a REAL, trainable mini-dataset (stick-figure imagery) that exercises the
entire data->train->eval plumbing before Blender/photoreal assets exist. Swap
skeleton_render for the Blender render and the sample schema is unchanged.

    python scripts/convert_motion.py --out data/bootstrap --scale 4
    python scripts/bootstrap_dataset.py --manifest data/bootstrap/motion_manifest.json \
        --out data/bootstrap/shards
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.convert_motion import load_world_joints
from synthgen.cameras import intrinsics_dict, project_points, sample_camera
from synthgen.config import DEFAULT, LabelClass, Projection
from synthgen.quality import (DistributionAudit, label_consistency, physical_plausibility,
                              visibility)
from synthgen.rationale import (JOINTS, build_answer, build_rationale, classify,
                                compute_features, detect_events)
from synthgen.render import _forced_projections, _strip_frame_indices, _subject_pixel_height
from synthgen.scene import sample_lighting
from synthgen.skeleton_render import render_strip

PROMPT = ("You are a safety monitor. Look at the frames (oldest to newest) and report "
          "whether a person has fallen, fainted, is lying immobile, in distress, or is "
          "acting normally. Answer with JSON only.")


def build(manifest_path: Path, out_dir: Path, cfg=DEFAULT, k=8, seed=0):
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = DistributionAudit()
    manifest = json.loads(Path(manifest_path).read_text())
    samples_path = out_dir / "samples.jsonl"
    n_written = 0
    with samples_path.open("w") as sf:
        for entry in manifest:
            joints, fps, intended_s = load_world_joints(Path(entry["path"]))
            intended = LabelClass(intended_s)
            feats = compute_features(joints, fps)
            if not physical_plausibility(feats).ok:
                audit.record(entry["clip_id"], None, None, None, False, False, ["implausible"])
                continue
            events = detect_events(feats)
            label = classify(feats, events, intended=intended)
            if not label_consistency(label, intended).ok:
                audit.record(entry["clip_id"], None, None, None, False, False,
                             [f"{intended.value}->{label.value}"])
                continue
            rationale = build_rationale(feats, events, label)
            answer = build_answer(label, feats, events)
            centroid = joints["pelvis"].mean(axis=0)
            strip = _strip_frame_indices(joints["pelvis"].shape[0], cfg)
            light = sample_lighting(rng, cfg.lighting)
            forced = _forced_projections(cfg)

            for i in range(k):
                force = forced[i] if i < len(forced) else None
                short = int(cfg.render.short_side_px.sample(rng))
                cam = sample_camera(rng, centroid, short, force_projection=force)
                kp = []
                for t in strip:
                    pts = np.stack([joints[n][t] for n in JOINTS], axis=0)
                    uv, valid = project_points(cam, pts)
                    kp.append({"uv": uv.tolist(), "valid": valid.tolist()})
                subj_h = _subject_pixel_height(kp, cam.res_y)
                vis = visibility(subj_h, occlusion_frac=0.0)
                is_night = light.mode == "night_ir"
                audit.record(entry["clip_id"], label.value, cam.projection.value,
                             cam.archetype.value, is_night, vis.ok, vis.reasons)
                if not vis.ok:
                    continue
                view_id = f"{entry['clip_id']}_cam{i:02d}"
                frame_dir = out_dir / "frames" / view_id
                frames = render_strip(kp, cam.res_x, cam.res_y, frame_dir, night=is_night)
                rec = {
                    "id": view_id,
                    "motion_id": entry["clip_id"],
                    "class": label.value,
                    "frames": frames,
                    "prompt": PROMPT,
                    "rationale": rationale,
                    "answer": answer,
                    "camera": {"archetype": cam.archetype.value,
                               "intrinsics": intrinsics_dict(cam)},
                    "lighting": light.mode,
                    "split_key": cam.archetype.value,  # used for cross-view splits in eval
                }
                sf.write(json.dumps(rec) + "\n")
                n_written += 1
    (out_dir / "audit.json").write_text(json.dumps(
        {"summary": audit.summary(), "warnings": audit.warnings()}, indent=2, default=str))
    print(f"wrote {n_written} samples -> {samples_path}")
    print("AUDIT:", audit.summary())
    return n_written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build(args.manifest, args.out, k=args.k, seed=args.seed)


if __name__ == "__main__":
    main()
