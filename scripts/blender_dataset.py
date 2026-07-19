import blenderproc as bproc  # MUST be the first line (blenderproc re-execs the file)
# Photoreal-ish MULTI-PERSON synthetic dataset via Blender. Runs INSIDE `blenderproc run`.
# Renders 1-3 connected skin-mesh humans (not stick figures) through native fisheye; scene
# label = worst danger present (aggregated). Same samples.jsonl schema (+ n_people) so
# training/eval consume it unchanged.
#   CUDA_VISIBLE_DEVICES=0 blenderproc run scripts/blender_dataset.py -- \
#       --manifest data/synth3d/motions/motion_manifest.json --out data/synth3d/multi \
#       --scenes 200 --k 5 --short 384 --samples 20
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.convert_motion import load_world_joints  # noqa: E402
from synthgen import blender_render as br  # noqa: E402
from synthgen.cameras import intrinsics_dict, project_points, sample_camera  # noqa: E402
from synthgen.config import DEFAULT, LabelClass  # noqa: E402
from synthgen.quality import DistributionAudit, physical_plausibility, visibility  # noqa: E402
from synthgen.rationale import (JOINTS, build_answer, build_rationale, classify,  # noqa: E402
                                compute_features, detect_events)
from synthgen.render import _forced_projections, _strip_frame_indices, _subject_pixel_height  # noqa: E402
from synthgen.scene import sample_lighting  # noqa: E402
from synthgen.scene_compose import (SEVERITY, compose_scene, scene_answer,  # noqa: E402
                                    scene_rationale, transform_joints)

PROMPT = ("You are a safety monitor. These are consecutive video frames (oldest first), "
          "possibly with more than one person. Report whether ANYONE has fallen, fainted, "
          "is lying immobile, or is in distress; else normal. Answer with JSON only.")
_RANK = {c: i for i, c in enumerate(SEVERITY)}


def build_motion_index(manifest):
    idx = {"danger": [], "normal": []}
    for e in manifest:
        (idx["normal"] if e["class"] == "normal" else idx["danger"]).append(e)
    return idx


def make_picker(idx, rng_seedless):
    def pick(rng, prefer_danger):
        pool = idx["danger"] if (prefer_danger and idx["danger"]) else idx["normal"]
        if not pool:
            pool = idx["danger"] or idx["normal"]
        e = pool[int(rng.integers(len(pool)))]
        return e["clip_id"], e["path"], e["class"]
    return pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scenes", type=int, default=150)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--short", type=int, default=384)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bproc.init()
    br.configure_render(cycles_samples=args.samples, denoise=True)
    rng = np.random.default_rng(args.seed)
    cfg = DEFAULT
    audit = DistributionAudit()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text())
    picker = make_picker(build_motion_index(manifest), None)
    bodies = br.build_bodies(3)          # pool; extras hidden per scene
    floor = br.setup_scene_and_floor(rng)

    sf = (args.out / "samples.jsonl").open("w")
    n_written = 0
    for si in range(args.scenes):
        spec = compose_scene(rng, picker)
        # load + place each person's joints; derive per-person actual label from kinematics
        people_joints, per_labels, per_feats, per_events, per_rats = [], [], [], [], []
        ok = True
        for person in spec.people:
            j, fps, _ = load_world_joints(Path(person.motion_path))
            if not physical_plausibility(compute_features(j, fps)).ok:
                ok = False
                break
            jt = transform_joints(j, person.origin, person.yaw)
            feats = compute_features(jt, fps)
            ev = detect_events(feats)
            lab = classify(feats, ev, intended=person.label)
            people_joints.append(jt)
            per_labels.append(lab)
            per_feats.append(feats)
            per_events.append(ev)
            per_rats.append(build_rationale(feats, ev, lab))
        if not ok or not people_joints:
            audit.record(f"scene{si}", None, None, None, False, False, ["implausible"])
            continue

        # aggregate scene status from ACTUAL per-person labels (worst severity)
        danger = [(i, l) for i, l in enumerate(per_labels) if l != LabelClass.NORMAL]
        if danger:
            di, status = min(danger, key=lambda il: _RANK[il[1]])
        else:
            di, status = 0, LabelClass.NORMAL
        spec.status = status
        spec.danger_index = di if danger else None
        rationale = scene_rationale(spec, per_rats)
        base_answer = build_answer(status, per_feats[di], per_events[di])
        answer = scene_answer(spec, person_down=base_answer.get("person_down", False),
                              posture=base_answer.get("posture", "unknown"))

        # frame the scene: aim at the label-driving person's centroid
        centroid = people_joints[di]["pelvis"].mean(axis=0)
        strip = _strip_frame_indices(people_joints[0]["pelvis"].shape[0], cfg)
        light = sample_lighting(rng, cfg.lighting)
        br.setup_lighting(rng, light)
        forced = _forced_projections(cfg)

        for i in range(args.k):
            force = forced[i] if i < len(forced) else None
            cam = sample_camera(rng, centroid, args.short, force_projection=force)
            # visibility gate on the label-driving person
            kp = []
            for t in strip:
                pts = np.stack([people_joints[di][n][t] for n in JOINTS], axis=0)
                uv, val = project_points(cam, pts)
                kp.append({"uv": uv.tolist(), "valid": val.tolist()})
            subj_h = _subject_pixel_height(kp, cam.res_y)
            vis = visibility(subj_h, occlusion_frac=0.0)
            is_night = light.mode == "night_ir"
            audit.record(f"scene{si}", status.value, cam.projection.value,
                         cam.archetype.value, is_night, vis.ok, vis.reasons)
            if not vis.ok:
                continue
            view_id = f"scene{si:04d}_n{spec.n}_cam{i:02d}"
            frames = br.render_scene(bodies, people_joints, strip, cam,
                                     args.out / "frames" / view_id)
            rec = {"id": view_id, "class": status.value, "n_people": spec.n,
                   "frames": frames, "prompt": PROMPT, "rationale": rationale,
                   "answer": answer,
                   "camera": {"archetype": cam.archetype.value, "intrinsics": intrinsics_dict(cam)},
                   "lighting": light.mode, "split_key": cam.archetype.value}
            sf.write(json.dumps(rec) + "\n")
            n_written += 1
        br._clear_lights() if hasattr(br, "_clear_lights") else _clear_lights()

    sf.close()
    (args.out / "audit.json").write_text(json.dumps(
        {"summary": audit.summary(), "warnings": audit.warnings()}, indent=2, default=str))
    print(f"[blender_dataset] wrote {n_written} views -> {args.out}")
    print("AUDIT:", audit.summary())


def _clear_lights():
    import bpy
    for o in list(bpy.data.objects):
        if o.type == "LIGHT":
            bpy.data.objects.remove(o, do_unlink=True)


if __name__ == "__main__":
    main()
