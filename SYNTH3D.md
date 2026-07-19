# Fully-Synthetic-From-3D Dataset (the ungated pivot)

**Decision (this session):** generate the training data as **3D renders projected through
the camera model**, using only **un-gated** tooling — no SMPL-X / LAFAN1 / AMASS / OmniFall
registration required. This removes the asset-login gate from the critical path.

## Why
- Our pipeline was already "projected from 3D" (procedural 3D joints → camera/fisheye
  projection → GT + labels). The only weak link was the render layer: **stick-figures are
  too abstract for a pretrained VLM to read** (bootstrap run 3 learned seen-views but that
  wouldn't transfer to real pixels).
- Rendering the same 3D motion as a **textured 3D body** (volume, shading, cast shadows,
  self-occlusion, real fisheye) gives imagery the VLM backbone actually engages with.
- Evidence this transfers: BEDLAM (synthetic-only matches real for human perception),
  OmniFall OF-Synthetic (beat staged-real on in-the-wild falls).

## Implementation (all working, gpu0, no login)
- **Blender 4.2.1** (auto-installed via BlenderProc) + Cycles GPU (OptiX on RTX 5090 —
  confirmed working).
- **`synthgen/blender_render.py`** — volumetric **mannequin**: torso ellipsoid + head /
  hand / foot spheres + limb capsules, skinned in skin/cloth materials, driven by the same
  3D joint trajectories the rest of the pipeline produces. Arms hang from shoulders along
  the torso-down axis (the procedural skeleton has no arm joints). Renders as a clearly
  human artist's-mannequin figure — see `docs/img/synth3d_mannequin_sample.png` (standing,
  mid-fall, and overhead-fisheye views).
- **`synthgen/cameras.set_blender_camera_object`** — native panoramic fisheye
  (equidistant/equisolid) set directly per frame; GT keypoints still from
  `project_points` so labels are correct under distortion.
- **`scripts/blender_dataset.py`** — render driver (runs inside `blenderproc run`). Same
  `samples.jsonl` schema as the stick-figure path → `training/` + `eval` unchanged. Args:
  `--k` cams, `--limit` motions (manifest shuffled first for class mix), `--short` px,
  `--samples` Cycles samples.

## Run
```
CUDA_VISIBLE_DEVICES=0 blenderproc run scripts/blender_dataset.py -- \
  --manifest data/bootstrap/motion_manifest.json --out data/synth3d/batch \
  --k 5 --limit 110 --short 384 --samples 24 --seed 1
```

## Multi-person scenes (implemented)
`synthgen/scene_compose.py` + the updated `blender_render` / `blender_dataset` render
**1–3 people per scene** (weighted 0.55 / 0.32 / 0.13), each with its own motion and floor
position, distinct clothing colour, real inter-person **occlusion** from 3D.
- **Scenario bias:** person 0 is danger-biased, others usually bystanders (a person falls
  while others stand/walk), with occasional multi-danger.
- **Label = worst danger present** (severity fall > faint > immobile > distress > normal);
  the scene is an alert if ANYONE is in danger. `answer` JSON gains `n_people`; the
  rationale names which person drives the label.
- Camera frames the label-driving person; visibility gate is on that person.
- Verified: renders 1/2/3-person scenes with correct aggregation (e.g. distress+normal →
  scene `distress`). Sample: `docs/img/synth3d_multi_sample.png`.

## Body fidelity: skin-mesh now, MPFB2 next
- **Now:** `blender_render.build_body` builds a CONNECTED organic body via Blender's Skin
  modifier over a 15-point graph (9 joints + derived elbow/knee/hand for natural limb
  bend), posed purely by world vertex positions (no armature/rigging → low risk). A clear
  step up from floating capsules.
- **Next photoreal tier — MPFB2** (MakeHuman-for-Blender-2, GPLv3/CC0, no login, Blender
  4.2-native, scriptable via `HumanService.create_human` → `add_builtin_rig("game_engine")`).
  It produces real parametric human meshes but is posed by an **armature via IK/`pose_bone.
  matrix`**, which needs the skeleton to carry elbow/knee joints. Path: add elbow/knee to
  the motion skeleton, then drive the MPFB2 rig with IK-target empties at our joint
  positions. Quaternius CC0 glTF characters are a zero-install fallback. (MakeHuman
  standalone has no headless export; Mixamo needs a login; Human Generator is paid.)

## Status & fidelity ladder
- ✅ Pipeline validated end-to-end: readable 3D human, real lighting/shadows, native
  fisheye, GT + labels, gpu0-only, GGUF-compatible downstream.
- **Mannequin** is the current body — recognizably human, good enough to test whether the
  VLM learns from 3D (far better than stick-figures). It is NOT photoreal.
- **Next fidelity step (optional):** swap the mannequin for a rigged **MakeHuman**
  (fully scriptable, login-free) or **Mixamo** (free login) mesh on the same rig +
  real fall animations. Everything else (camera, GT, labels, eval) stays identical.

## Honest caveats
- Mannequin ≠ photoreal human; domain gap to real camera footage remains → still want a
  small **self-recorded** real fisheye validation set (ungated) to measure sim-to-real.
- Procedural motion is simple; real fall variety needs Mixamo animations + Blender ragdoll.
- Render cost: ~1–2 s/frame at 384px/24 samples on the RTX 5090.
