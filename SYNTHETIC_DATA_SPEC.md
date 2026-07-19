# AI-Care-2 — Synthetic Fall Data Generation Spec (M2)

Buildable spec for the engine-render pipeline that produces perspective- and fisheye-robust fall/danger training data with pixel-exact labels. Runs on **gpu0 (RTX 5090 32GB)**. Companion: `VLM_TRAINING_PLAN.md` §2b.

**Design goal:** render each motion event from many camera poses × lens models × scenes × lighting, with ground-truth 3D pose + camera exported alongside every clip so labels and view-canonical rationales are free.

---

## 0. Stack decision

| Layer | Choice | Why |
|---|---|---|
| Body model | **SMPL-X** (neutral + male/female, betas sampled) | Standard, AMASS/LAFAN-compatible, drives BEDLAM-class realism |
| Motion source | **LAFAN1** (fall/recovery) + **ragdoll physics** (impact) + AMASS (ADL/negatives) | AMASS is fall-poor; LAFAN1 has real falls; ragdoll for impact dynamics |
| Renderer | **BlenderProc 2.x** (Blender 4.x / Cycles) as primary; **Isaac Sim/Replicator** as escalation for calibrated Kannala-Brandt fisheye | BlenderProc = scriptable headless SDG + native lens distortion + GT writers; Isaac only if we must match a *specific* real camera calibration |
| Scenes | **Infinigen-Indoors** + a small set of hand-built rooms + HDRI backgrounds | Photoreal randomized indoor environments; no humans (we add SMPL-X) |
| Body→Blender | **Meshcapade SMPL_blender_addon** (imports SMPL-X, AMASS .npz) | Direct AMASS/LAFAN import + shape/pose correctives |

Primary path is **BlenderProc**. Only escalate to Isaac Sim per-camera if BlenderProc fisheye fidelity proves insufficient against a real reference lens (decided empirically in M3).

---

## 1. Motion library (the hardest input — build first)

Target **~1,500–3,000 distinct motion clips**, 2–6 s each @ 30fps, split:

| Class | Count target | Source |
|---|---|---|
| `fall` (forward/backward/lateral/from-chair/from-stairs/slump) | ~800 | LAFAN1 fall/recovery + ragdoll variants; ≥6 mechanisms × directions |
| `faint-collapse` (vertical crumple, no protective reaction) | ~250 | Ragdoll with limp muscle model + slow-onset keyframes |
| `lying-immobile` (already down, still) | ~200 | Hold end-poses of falls + prone/supine mocap |
| `distress` (clutching, struggling to rise, slow slide-down) | ~250 | Hand-authored + mocap |
| `normal / hard-negative` (sit fast, lie on sofa/bed, bend/pick-up, exercise, kneel, crouch) | ~1,000+ | AMASS ADL + authored — **must outnumber falls to control false alarms** |

**Ragdoll recipe:** SMPL-X skeleton → Blender rigid-body/Rigify ragdoll, initial pose + push impulse sampled, gravity sim, capture 2–6 s. Randomize: impulse direction/magnitude, initial standing pose, self-collision friction, floor height. Discard clips that fail a physical-plausibility filter (§6).

**Critical constraint:** fall physics fidelity gates everything downstream. Budget iteration here; a wrong-looking fall teaches a wrong cue.

---

## 2. Camera sampling — the perspective-robustness engine

Per motion clip, render from **K = 8–16 sampled cameras**. Each camera independently sampled:

### 2.1 Placement (spherical around subject, subject at scene centroid ± jitter)
| Param | Range | Notes |
|---|---|---|
| Mount archetype | {ceiling, high-corner, wall-mid, low-shelf, body-worn} weighted | covers real install types |
| Height above floor | 0.8 – 3.2 m | ceiling ~2.6–3.2, corner ~2.0–2.6, shelf ~0.8–1.5 |
| Horizontal distance to subject | 1.5 – 8 m | near for wide-angle, far for tele |
| Pitch (down-tilt) | 0° (level) – 90° (nadir/top-down) | top-down is the fisheye stress case |
| Yaw (azimuth) | 0 – 360° uniform | full ring — never bias to frontal |
| Roll | −15° – +15° (−180–180 for body-worn) | cameras are rarely perfectly level |
| Position jitter | ±0.3 m per axis | breaks perfect centering |

### 2.2 Lens / projection (sampled per camera)
| Projection | Weight | Params |
|---|---|---|
| Rectilinear (perspective) | 0.35 | HFOV 45–90° |
| Wide rectilinear | 0.20 | HFOV 90–120° |
| **Fisheye equidistant** (r = f·θ) | 0.20 | FOV 150–200°, Blender Cycles `PANO`/fisheye equidistant |
| **Fisheye equisolid** (r = 2f·sin(θ/2)) | 0.20 | lens 8–16 mm, sensor 36 mm, matches real lenses |
| Calibrated Kannala-Brandt (Isaac path) | 0.05 | k0–k4 sampled near real reference cameras |

Also randomize: optical-center offset (±5% of frame — fisheye circles are rarely centered), sensor resolution rendered at **512–768 px** short side (matches VLM input; avoids wasting render on detail the model never sees), mild radial vignetting.

**Coverage guarantee:** stratify sampling so every motion clip is seen by at least one top-down fisheye and one level rectilinear camera across its K renders — the two extremes must both appear in training.

---

## 3. Scene, lighting, domain randomization

| Axis | Range |
|---|---|
| Scene | Infinigen-Indoors rooms + ~10 hand-built (living room, bedroom, bathroom, hallway, kitchen) + HDRI-only fallback |
| Floor material | tile/wood/carpet/concrete, randomized albedo + roughness |
| Lighting | 500–6500 K color temp; 20–1000 lux; 1–4 sources; window daylight vs lamp vs **IR/near-mono night** (CEPDOF-style) |
| Time-of-day | day / dusk / night-IR — night is ~30% of samples (falls happen at night) |
| Occluders | furniture between camera and subject (0–40% body occlusion) sampled |
| Distractors | 0–3 extra people (ADL), pets, moving objects — teaches multi-occupant robustness |
| Clothing/texture | randomized SMPL-X textures + CLO-style garments if available |
| Body shape | betas sampled across BMI/height/age proxy; skin tone across Monk scale |
| Camera noise | sensor noise, motion blur, mild compression artifacts (deployment realism) |

---

## 4. Output format (per rendered clip)

```
clip_<uuid>/
  frames/        # PNG or mp4, 30fps, 512–768px, the K-th camera view
  meta.json
```

`meta.json`:
```json
{
  "clip_id": "...",
  "class": "fall|faint-collapse|lying-immobile|distress|normal",
  "motion_id": "lafan1_fall_0123|ragdoll_...",
  "camera": {
    "archetype": "ceiling",
    "extrinsics": [[...]],           // 4x4 world->cam
    "projection": "fisheye_equidistant",
    "intrinsics": {"fov_deg": 187, "cx": 0.48, "cy": 0.52, "dist": [...]}
  },
  "scene_id": "infinigen_room_44",
  "lighting": {"temp_k": 3200, "lux": 120, "mode": "night_ir"},
  "gt_pose_3d": "poses.npz",         // SMPL-X params + joints, per frame
  "gt_pose_2d_incam": "kp2d.npz",    // projected into THIS distorted view
  "events": {"impact_frame": 47, "immobile_from": 55},
  "occlusion_frac_max": 0.22
}
```

The `gt_pose_3d` + `camera` fields are what make rationale generation free (§5).

---

## 5. Auto-generating VLM training samples from renders

Each clip → one or more instruction samples for the VLM. Two derived fields:

**(a) View-canonical rationale (chain-of-thought target).** Computed from `gt_pose_3d` in a **gravity-aligned canonical frame** (independent of camera), so the *same fall* yields a *similar* rationale across all K cameras:
> "Person standing at t=0; torso pitches to ~85° from vertical between frames 40–47; head descends below hip height; vertical velocity peaks then arrests at frame 47 (impact); body remains horizontal and still through frame 90 → **fall followed by immobility**."

Template-fill from pose kinematics (torso-vertical angle, head-hip height delta, COM vertical velocity, stillness window). Optionally paraphrase with the Qwen2.5-VL-32B teacher for language diversity.

**(b) Short JSON answer (the deployment output):**
```json
{"status":"fall","confidence":0.94,"person_down":true}
```

Input to the model = **N-frame strip** (e.g. 4–8 frames spanning ~3–4 s, temporally subsampled) matching the on-Pi inference format, rendered from that clip's camera.

---

## 6. Quality gates (run before a clip enters the training set)

1. **Physical-plausibility filter:** reject if COM trajectory violates gravity bounds, limbs interpenetrate > threshold, or foot-skate exceeds limit. (Auto from `gt_pose_3d`.)
2. **Visibility filter:** subject must be ≥ min pixel height and ≤ max occlusion for the clip's label to hold.
3. **Label-consistency check:** kinematic auto-label must match the intended motion class, else flag for review.
4. **Distribution audit:** log per-batch histograms of viewpoint/lens/lighting/class — enforce the §2 coverage guarantee and the negatives-outnumber-falls rule.
5. **Human spot-check:** sample ~1% for eyeball review each batch; log what was dropped (no silent truncation).

---

## 7. Scale & compute budget (gpu0)

- Target v1 corpus: **~2,000 motions × ~12 cameras ≈ 24k clips** (~4–8 frames each for VLM strips; full 30fps mp4 optional for the video-native model).
- Cycles render at 512–768px, denoised, ~1–3 s/frame on RTX 5090 → strips (few frames) are cheap; budget a multi-day headless batch. Parallelize across motion clips.
- Store as WebDataset/tar shards for training throughput.
- Escalate resolution/samples only for a held-out "high-fidelity" eval slice.

---

## 8. Build order (M2)

1. **Motion library** — LAFAN1 import + ragdoll rig + physical-plausibility filter. *(highest risk, do first)*
2. **SMPL-X → BlenderProc** single-clip render with one camera; verify GT pose/camera export round-trips.
3. **Camera + lens sampler** (§2) incl. Blender fisheye equidistant/equisolid; visual sanity vs a real fisheye reference frame.
4. **Scene/lighting/DR** (§3) via Infinigen + HDRI.
5. **Rationale + JSON sample generator** (§5) from GT.
6. **Quality gates + distribution audit** (§6).
7. Scale to full batch; hand off shards to Stage A/B training.

**Definition of done:** a single command renders a labeled, quality-gated, VLM-ready shard from a motion list, with enforced viewpoint/lens/lighting coverage and exported GT — reproducible and re-runnable to grow the corpus.

---

## 9. Open decisions to resolve during build

- BlenderProc fisheye vs Isaac Kannala-Brandt: is BlenderProc's distortion close enough to our real target lens? (Decide in M3 against a real fisheye calibration.)
- LAFAN1 licensing for our use — confirm before shipping any derived data.
- How much diffusion (Wan/CogVideoX) appearance augmentation to blend in vs pure engine — tune by measuring OOPS-Fall transfer.
- Strip length / frame count: the sweet spot between temporal signal and Pi latency (co-decided with the on-Pi M1 benchmark).
