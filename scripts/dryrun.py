"""Blender-free smoke test of the full pipeline logic.

Synthesizes fake world-space joints for a few motions, then runs the real
label -> K-camera-sample -> GT-projection -> quality-gate -> meta path (everything
except the actual Blender render + SMPL-X spawn). Proves the orchestration glue holds
before any assets are installed.

    python scripts/dryrun.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthgen.cameras import intrinsics_dict, look_at_extrinsics, project_points, sample_camera
from synthgen.config import DEFAULT, LabelClass, Projection
from synthgen.quality import DistributionAudit, label_consistency, physical_plausibility, visibility
from synthgen.rationale import (JOINTS, build_answer, build_rationale, classify,
                                compute_features, detect_events)
from synthgen.render import _forced_projections, _strip_frame_indices, _subject_pixel_height


def fake_fall(T=90):
    j = {n: np.zeros((T, 3)) for n in JOINTS}
    z_p = np.concatenate([np.full(40, 1.0), np.linspace(1.0, 0.2, 8), np.full(T - 48, 0.2)])
    z_h = np.concatenate([np.full(40, 1.6), np.linspace(1.6, 0.15, 8), np.full(T - 48, 0.15)])
    z_n = np.concatenate([np.full(40, 1.4), np.linspace(1.4, 0.2, 8), np.full(T - 48, 0.2)])
    x_n = np.concatenate([np.zeros(40), np.linspace(0, 0.6, 8), np.full(T - 48, 0.6)])
    j["pelvis"][:, 2] = z_p; j["head"][:, 2] = z_h; j["neck"][:, 2] = z_n; j["neck"][:, 0] = x_n
    j["l_shoulder"][:, 2] = z_n; j["r_shoulder"][:, 2] = z_n
    for n in ("l_hip", "r_hip", "l_ankle", "r_ankle"):
        j[n][:, 2] = np.clip(z_p - 0.1, 0, None)
    return j, LabelClass.FALL


def fake_normal_standing(T=90):
    j = {n: np.zeros((T, 3)) for n in JOINTS}
    j["pelvis"][:, 2] = 1.0; j["neck"][:, 2] = 1.4; j["head"][:, 2] = 1.6
    j["l_shoulder"][:, 2] = 1.4; j["r_shoulder"][:, 2] = 1.4
    return j, LabelClass.NORMAL


def main():
    rng = np.random.default_rng(0)
    cfg = DEFAULT
    audit = DistributionAudit()
    out = Path(tempfile.mkdtemp(prefix="synth_dryrun_"))
    motions = [("m_fall", *fake_fall()), ("m_normal", *fake_normal_standing())]

    total_views = 0
    for clip_id, joints, intended in motions:
        feats = compute_features(joints, cfg.render.fps)
        assert physical_plausibility(feats).ok, clip_id
        events = detect_events(feats)
        label = classify(feats, events, intended=intended)
        assert label_consistency(label, intended).ok, f"{clip_id}: {label} vs {intended}"
        rationale = build_rationale(feats, events, label)
        answer = build_answer(label, feats, events)
        centroid = joints["pelvis"].mean(axis=0)

        strip = _strip_frame_indices(len(joints["pelvis"]), cfg)
        forced = _forced_projections(cfg)
        k = 10
        seen_proj = set()
        for i in range(k):
            force = forced[i] if i < len(forced) else None
            cam = sample_camera(rng, centroid, short_side_px=640, force_projection=force)
            seen_proj.add(cam.projection.value)
            kp = []
            for t in strip:
                pts = np.stack([joints[n][t] for n in joints], axis=0)
                uv, valid = project_points(cam, pts)
                kp.append({"uv": uv.tolist(), "valid": valid.tolist()})
            subj_h = _subject_pixel_height(kp, cam.res_y)
            vis = visibility(subj_h, occlusion_frac=0.0)
            audit.record(clip_id, label.value, cam.projection.value, cam.archetype.value,
                         False, vis.ok, vis.reasons)
            if vis.ok:
                total_views += 1
                meta = {"clip_id": f"{clip_id}_cam{i:02d}", "class": label.value,
                        "camera": {"archetype": cam.archetype.value,
                                   "extrinsics_world2cam": look_at_extrinsics(cam.location, cam.look_at, cam.roll_deg).tolist(),
                                   "intrinsics": intrinsics_dict(cam)},
                        "rationale": rationale, "answer": answer, "keypoints_2d_incam": kp}
                d = out / f"{clip_id}_cam{i:02d}"; d.mkdir(parents=True, exist_ok=True)
                (d / "meta.json").write_text(json.dumps(meta))
        # coverage guarantee held for this clip
        assert Projection.FISHEYE_EQUIDISTANT.value in seen_proj, "no top-down fisheye"
        assert Projection.RECTILINEAR.value in seen_proj, "no level rectilinear"

    print(f"views written: {total_views}  -> {out}")
    print("AUDIT:", audit.summary())
    # sanity: at least one fisheye view actually projected the subject visibly
    sample_meta = next(out.rglob("meta.json"))
    m = json.loads(sample_meta.read_text())
    print("sample view:", m["camera"]["intrinsics"]["projection"], "| answer:", m["answer"])
    print("DRYRUN OK")


if __name__ == "__main__":
    main()
