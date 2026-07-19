# synthgen — synthetic fall/danger data pipeline

Generates perspective- and fisheye-robust fall/faint/danger training clips with
pixel-exact labels, by rendering SMPL-X bodies (driven by real fall motion) from many
sampled cameras and lenses. Implements `SYNTHETIC_DATA_SPEC.md`.

## Why this exists
Real falls can only be captured one camera at a time. In a renderer, **camera pose and
lens are free parameters** — render one fall, emit it from N viewpoints × lens models
with exact ground truth. The renderer also gives 3D pose + camera, so labels and
**view-canonical rationales** (identical CoT target for the same fall across all views)
are free. That is the mechanism for camera-angle robustness.

## Module map
| File | Runs where | Purpose |
|---|---|---|
| `config.py` | anywhere | all sampling ranges (camera/lens/lighting/class-mix) — single source of truth |
| `cameras.py` | math anywhere; setup in Blender | camera+lens sampling, native fisheye setup, **GT projection incl. fisheye** |
| `rationale.py` | anywhere | 3D pose → kinematic features → label + view-canonical rationale + JSON answer |
| `quality.py` | anywhere | plausibility / visibility / label-consistency gates + distribution audit |
| `scene.py` | sampling anywhere; apply in Blender | lighting + domain-randomization sampling |
| `bodies.py` | Blender only | animated SMPL-X spawn + world-joint readback (integration skeleton) |
| `render.py` | Blender only | orchestrates one clip → K labelled views |

## Key design decisions
- **Native panoramic fisheye, not Brown-Conrady.** BlenderProc's `set_lens_distortion`
  is Brown-Conrady and can't do 150–200° fisheye faithfully, so fisheye uses Blender's
  native `PANO` camera and we project GT keypoints ourselves (`cameras.project_points`,
  equidistant `r=f·θ` / equisolid `r=2f·sin(θ/2)`). Rectilinear uses BlenderProc's
  K-matrix path.
- **Label once, camera-independent.** Kinematics/label/rationale are computed from world
  joints before any camera, so all K views of a clip share one view-canonical target.
- **Coverage guarantee.** Every clip forces ≥1 top-down fisheye and ≥1 level rectilinear.
- **Negatives outnumber falls** (`CLASS_MIX`) to control false-alarm rate.

## What's runnable now vs. needs assets
Runnable with just `numpy` (no Blender):
```
python -m synthgen.config       # sampling weights sanity
python -m synthgen.rationale    # fall → label + rationale + JSON
python -m synthgen.quality      # gates + audit
python -m synthgen.scene        # lighting mix
python scripts/dryrun.py        # FULL pipeline glue on synthetic joints, no Blender
```
Needs assets (SMPL-X model, LAFAN1/AMASS motion, Infinigen scenes) — see `scripts/install.sh`:
```
blenderproc run scripts/run_render.py --manifest <motions.json> --out <shards> --seed 0
```

## Open integration points (marked `NotImplementedError` in code)
1. `bodies.spawn_animated_smplx` — wire Meshcapade SMPL_blender_addon; key poses per frame.
2. `bodies.readback_world_joints` — confirm the addon's bone names / SMPL-X joint indices.
3. `scene.load_scene_and_floor` — point at Infinigen export / .blend room library.
4. `render._render_strip` — set frame range and call `bproc.renderer.render()`, write PNGs.
5. `scripts/convert_motion.py` (TODO) — retarget LAFAN1 + ragdoll sim → SMPL-X `.npz`.

## Validate-on-machine checklist (M2)
- [ ] BlenderProc downloads its Blender; `blenderproc quickstart` renders.
- [ ] One SMPL-X clip spawns + animates; joint readback matches `rationale.JOINTS`.
- [ ] Native fisheye render visually matches a real fisheye reference; if not, escalate
      to Isaac-Sim Kannala-Brandt per spec §0.
- [ ] `project_points` 2D keypoints overlay correctly on rendered fisheye frames.
- [ ] Distribution audit warnings clear on a real batch.
