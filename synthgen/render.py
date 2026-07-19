"""Render orchestrator — one motion clip -> K labelled camera views + samples.

Runs INSIDE `blenderproc run` (imports bpy/blenderproc via the submodules). The pure
sampling/labelling/quality logic is exercised by the module self-tests without Blender;
this file is the glue that only executes on-machine once assets are installed.

Per motion clip it:
  1. spawns the animated SMPL-X body, reads back view-canonical world joints
  2. computes kinematic features + events + label + rationale ONCE (camera-independent)
  3. samples K cameras (guaranteeing >=1 top-down fisheye and >=1 level rectilinear)
  4. for each camera: configures the lens, renders the strip, projects GT joints into
     that (possibly fisheye) view, writes frames + meta.json, runs quality gates
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .cameras import (configure_camera_in_blender, intrinsics_dict, look_at_extrinsics,
                      project_points, sample_camera)
from .config import DEFAULT, PipelineCfg, Projection
from .quality import (DistributionAudit, label_consistency, physical_plausibility,
                      visibility)
from .rationale import (build_answer, build_rationale, classify, compute_features,
                        detect_events)


def _pick_camera_count(rng, cfg: PipelineCfg) -> int:
    r = cfg.render.cams_per_clip
    return int(rng.integers(int(r.lo), int(r.hi) + 1))


def _forced_projections(cfg: PipelineCfg) -> list[Projection | None]:
    """Coverage guarantee: force at least one top-down fisheye and one level rectilinear."""
    forced = []
    if cfg.require_topdown_fisheye:
        forced.append(Projection.FISHEYE_EQUIDISTANT)
    if cfg.require_level_rectilinear:
        forced.append(Projection.RECTILINEAR)
    return forced


def render_clip(motion, out_dir: Path, cfg: PipelineCfg, audit: DistributionAudit,
                rng: np.random.Generator):
    """Render one MotionClip. Imports bpy/blenderproc lazily via submodules."""
    from . import bodies, scene  # noqa: F401  (Blender-only)

    n_frames_total = None  # set from motion
    obj = bodies.spawn_animated_smplx(bodies.load_motion(motion.path))
    poses = bodies.load_motion(motion.path)["poses"]
    n_frames_total = len(poses)
    world_joints = bodies.readback_world_joints(obj, n_frames_total, motion.fps)
    centroid = bodies.subject_centroid(world_joints)

    # --- label once, camera-independent ---
    feats = compute_features(world_joints, motion.fps)
    plaus = physical_plausibility(feats)
    if not plaus.ok:
        audit.record(motion.clip_id, None, None, None, False, False, plaus.reasons)
        return
    events = detect_events(feats)
    from .config import LabelClass
    intended = LabelClass(motion.intended_class)
    label = classify(feats, events, intended=intended)
    lc = label_consistency(label, intended)
    if not lc.ok:
        audit.record(motion.clip_id, None, None, None, False, False, lc.reasons)
        return
    rationale = build_rationale(feats, events, label)
    answer = build_answer(label, feats, events)

    # --- lighting (per clip) ---
    light = scene.sample_lighting(rng, cfg.lighting)
    scene.load_scene_and_floor(rng, cfg.domain)
    scene.apply_lighting(light)

    # --- cameras ---
    short = int(cfg.render.short_side_px.sample(rng))
    k = _pick_camera_count(rng, cfg)
    forced = _forced_projections(cfg)
    strip_idx = _strip_frame_indices(n_frames_total, cfg)

    for i in range(k):
        force = forced[i] if i < len(forced) else None
        cam = sample_camera(rng, centroid, short, force_projection=force)
        configure_camera_in_blender(cam)

        frames = _render_strip(strip_idx)   # returns list of image paths (Blender)

        # GT joints projected into THIS lens (correct even for fisheye)
        kp_per_frame = []
        for t in strip_idx:
            pts = np.stack([world_joints[n][t] for n in world_joints], axis=0)
            uv, valid = project_points(cam, pts)
            kp_per_frame.append({"uv": uv.tolist(), "valid": valid.tolist()})

        subj_px_h = _subject_pixel_height(kp_per_frame, cam.res_y)
        vis = visibility(subj_px_h, occlusion_frac=0.0)  # occlusion filled by seg pass
        passed = vis.ok
        is_night = light.mode == "night_ir"
        audit.record(motion.clip_id, label.value, cam.projection.value,
                     cam.archetype.value, is_night, passed, vis.reasons)
        if not passed:
            continue

        clip_out = out_dir / f"{motion.clip_id}_cam{i:02d}"
        clip_out.mkdir(parents=True, exist_ok=True)
        _write_meta(clip_out, motion, cam, label, rationale, answer, light,
                    events, kp_per_frame, frames)


def _strip_frame_indices(n_total: int, cfg: PipelineCfg) -> list[int]:
    span = min(n_total, int(cfg.render.strip_span_s * cfg.render.fps))
    start = max(0, n_total - span)
    return list(np.linspace(start, n_total - 1, cfg.render.strip_frames).astype(int))


def _render_strip(strip_idx):
    """Render the selected frames with BlenderProc. Returns image paths.

    Integration point: set frame range to strip_idx and call bproc.renderer.render();
    write PNGs. Kept minimal here — depends on writer choice (PNG vs hdf5)."""
    import blenderproc as bproc  # noqa: F401
    raise NotImplementedError("Render strip_idx via bproc.renderer.render(); write frames.")


def _subject_pixel_height(kp_per_frame, res_y) -> float:
    hs = []
    for kp in kp_per_frame:
        uv = np.array(kp["uv"]); valid = np.array(kp["valid"])
        if valid.sum() >= 2:
            ys = uv[valid, 1]
            hs.append(ys.max() - ys.min())
    return float(np.median(hs)) if hs else 0.0


def _write_meta(clip_out, motion, cam, label, rationale, answer, light, events,
                kp_per_frame, frames):
    T_wc = look_at_extrinsics(cam.location, cam.look_at, cam.roll_deg)
    meta = {
        "clip_id": clip_out.name,
        "class": label.value,
        "motion_id": motion.clip_id,
        "camera": {
            "archetype": cam.archetype.value,
            "extrinsics_world2cam": T_wc.tolist(),
            "intrinsics": intrinsics_dict(cam),
        },
        "lighting": {"mode": light.mode, "temp_k": round(light.temp_k, 0), "lux": round(light.lux, 0)},
        "events": {"impact_frame": events.impact_frame, "immobile_from": events.immobile_from},
        "keypoints_2d_incam": kp_per_frame,
        "rationale": rationale,
        "answer": answer,
    }
    (clip_out / "meta.json").write_text(json.dumps(meta, indent=2))


def render_manifest(manifest_path: Path, out_dir: Path, cfg: PipelineCfg = DEFAULT):
    """Entry called from scripts/run_render.py inside `blenderproc run`."""
    from . import bodies
    rng = np.random.default_rng(cfg.seed)
    audit = DistributionAudit()
    manifest = json.loads(Path(manifest_path).read_text())
    for entry in manifest:
        motion = bodies.MotionClip(clip_id=entry["clip_id"], path=Path(entry["path"]),
                                   intended_class=entry["class"], fps=entry.get("fps", 30))
        try:
            render_clip(motion, out_dir, cfg, audit, rng)
        except NotImplementedError as e:
            print(f"[integration-stub] {motion.clip_id}: {e}")
    (out_dir / "audit.json").write_text(json.dumps({
        "summary": audit.summary(), "warnings": audit.warnings(),
        "dropped": audit.dropped}, indent=2, default=str))
    print("AUDIT:", audit.summary())
