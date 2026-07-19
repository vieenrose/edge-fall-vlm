# AI-Care-2 — Project Status

**Goal:** ONE tiny VLM on Raspberry Pi 5 (4GB), 24/7, detecting human fall / faint /
danger from an RGB camera, robust to perspective, lighting, and fisheye/ultra-wide
distortion. Training on **gpu0 only** (RTX 5090 32GB); gpu1 is reserved.

## Documents
| Doc | What |
|---|---|
| `RESEARCH_PLAN.md` | Phase-0 evidence base (25 sources, adversarially verified) |
| `VLM_TRAINING_PLAN.md` | Single-VLM recipe, on-device numbers, staged plan |
| `SYNTHETIC_DATA_SPEC.md` | Synthetic fisheye data-gen spec (the perspective solution) |
| `M1_BENCHMARK.md` | On-device VLM latency/RAM results + decisions (base model, strip format, fps-robustness) |
| `synthgen/README.md` | Pipeline code guide |

## Code map
```
synthgen/         data generation (spec impl)
  config.py       all sampling ranges (single source of truth)          [tested]
  cameras.py      camera+lens sampling, NATIVE fisheye, GT projection    [tested]
  rationale.py    3D pose -> label + view-canonical CoT + JSON answer     [tested]
  quality.py      plausibility/visibility/consistency gates + audit       [tested]
  scene.py        lighting + domain-randomization sampling                [tested]
  skeleton_render.py  Blender-free stick-figure PNG renderer (bootstrap)  [tested]
  bodies.py       SMPL-X spawn + joint readback              [Blender integration stub]
  render.py       one clip -> K labelled views orchestration [Blender integration stub]
scripts/
  convert_motion.py   procedural motion generator (+ LAFAN1 retarget TODO) [tested]
  bootstrap_dataset.py motions -> images -> samples.jsonl (no Blender)     [tested]
  dryrun.py       full pipeline glue without Blender                       [tested]
  run_render.py   `blenderproc run` entry                        [needs Blender+assets]
  run_tests.sh    all Blender-free self-tests                              [green]
  install.sh      env + asset checklist
training/
  dataset.py      samples.jsonl -> chat samples, cross-view/subject splits [tested]
  sft.py          SmolVLM2 fine-tune (full or LoRA), gpu0        [smoke-trained on gpu0]
  eval.py         sensitivity/specificity/F1, cross-view gap, FA/day       [tested]
  export_gguf.py  Stage-D GGUF/mmproj export for llama.cpp                 [tested]
deploy/
  monitor.py      motion-gate + strip buffer + alert hysteresis loop       [tested]
  vlm_backend.py  llama.cpp mtmd backend + mosaic + JSON parse             [tested]
  bench.py        M1 on-Pi latency/RAM/thermal harness                 [stub-tested]
```

## What is proven to run
- `bash scripts/run_tests.sh` → **ALL TESTS PASSED** (no Blender, no assets, no Pi).
- Full data path on procedural data: motions → K cameras (incl. true fisheye) → GT
  projection → stick-figure PNG strips → 399 labelled training samples, coverage +
  class-balance audit clean.
- **SmolVLM2-500M really fine-tuned on gpu0** (2-step smoke, loss 4.52, ~1.5 s/it; the
  image-splitting OOM is fixed via `do_image_splitting=False` + frame cap).
- Deploy loop converts noisy verdicts into stable RAISE/CLEAR alerts; bench projects
  false-alarms/day from cadence×(1−specificity).

## M1 benchmark — DONE (x86 proxy) — see M1_BENCHMARK.md
- Real GGUF inference measured via llama.cpp mtmd (Ryzen @4 threads = Pi proxy):
  **SmolVLM2-500M Q8_0 → 787 MB RSS, ~0.3–0.4 s compute; 256M → 663 MB, ~0.3 s.** Both
  fit 4GB. Pi5 projected ~1–2 s/inference (persistent process) → 0.5–1 Hz cadence fine.
- Key finding: subprocess-per-call reloads the model (~1 s here, ~3–5 s on Pi) →
  **24/7 must use a persistent server** (`llama-server`, not yet built).
- Decisions: base = SmolVLM2-500M-Video (256M fallback); native multi-frame strip N≈6
  (mosaic fallback if latency-bound); both in `deploy.vlm_backend`.
- **fps-robustness implemented + tested**: time-based `StripBuffer` (fixed ~3.5 s window
  by timestamp — 5 fps & 15 fps both give 6 frames) + `temporal_augment` (varied
  frame-count/spacing in SFT) + alert persistence. Robust to thermal-throttle fps drift.
- Models downloaded under `models/` (not committed as bulk); `~/llama.cpp` built with
  `llama-mtmd-cli`. Run `scripts/pi_bench.sh` on the real Pi for thermal + server numbers.

## Training loop — VALIDATED on gpu0 — see TRAINING_NOTES.md
- Real SFT of SmolVLM2-500M on the stick-figure bootstrap set (1,243 samples), gpu0 only,
  cross-view holdout (ceiling+low_shelf), stratified eval.
- **Balanced answer-first model learns**: seen-view acc **0.885** (sens 0.93), held-out
  acc **0.59** → **cross-view gap 0.295** — our pipeline empirically reproduces the exact
  perspective-generalization failure the project targets, and the eval detects it.
- Design decisions locked in: **answer-first JSON target** (parseable within token budget),
  **class balancing** (`--balance`, else model collapses to majority prior), cross-view
  holdout as the headline metric. All in `training/`.
- Stick-figures validate the loop + metric, not absolute accuracy — photoreal renders +
  real datasets remain the gate for a deployable detector.

## Fully-synthetic-from-3D pivot — WORKING (see SYNTH3D.md)
- Removes the SMPL-X/LAFAN1/AMASS/OmniFall login gate: training data is 3D renders
  projected through the camera model, using only Blender (auto-installed) + our code.
- `synthgen/blender_render.py` (volumetric human mannequin) + `set_blender_camera_object`
  (native fisheye) + `scripts/blender_dataset.py` (driver, same samples.jsonl schema).
- Cycles GPU (OptiX) works on the RTX 5090; renders a **recognizable human** with real
  lighting/shadows/fisheye — `docs/img/synth3d_mannequin_sample.png`.
- First 3D batch (243 views) trained + cross-view eval vs stick-figure baseline: 3D model
  **cross-view gap ~0.006 vs 0.295** (generalizes across views) but low absolute accuracy
  on a small, imbalanced set — promising, not conclusive (see TRAINING_NOTES.md).
- **Multi-person + body upgrade (done):** `scene_compose.py` renders 1–3 people/scene
  (own motion, position, colour, real occlusion), scene label = worst danger present
  (`answer` gains `n_people`). Body swapped capsules → connected **skin-mesh** humanoid
  with derived elbow/knee/hand. Sample: `docs/img/synth3d_multi_sample.png`.
- Next photoreal tier: **MPFB2** rigged mesh (needs elbow/knee joints on the skeleton +
  armature IK) — recipe in SYNTH3D.md.

## What is gated (needs assets / compute / a Pi)
1. **Assets** (licences cleared for detector training): SMPL-X model + Meshcapade addon,
   LAFAN1 retarget → SMPL-X (`convert_motion.retarget_lafan1_to_smplx` TODO), AMASS,
   Infinigen scenes. Then wire `bodies.py` / `scene.py` / `render._render_strip`.
2. **Blender photoreal render** replaces `skeleton_render` (schema identical).
3. **Real datasets:** OmniFall (`OF-Staged`+`OF-Synthetic`) + OOPS-Fall for the honest
   eval splits.
4. **Full training run** on gpu0, then **GGUF export + post-quant eval** (the only score
   that counts).
5. **M1 on-Pi benchmark:** real latency/RAM/thermal for SmolVLM2-256M/500M vs
   LFM2-VL-450M vs FastVLM-0.5B → pins base model + strip length.

## Recommended next actions (in order)
1. Acquire SMPL-X + LAFAN1; implement `retarget_lafan1_to_smplx`; render a first
   photoreal fisheye batch via BlenderProc; validate `project_points` overlays on it.
2. Download OmniFall + OOPS-Fall; add loaders alongside `bootstrap_dataset` schema.
3. M1 Pi bench (can run in parallel — it only needs the base GGUF models, not our data).
4. Stage A→D training on gpu0; GGUF export; post-quant OOPS-Fall + cross-view eval.

## Environment notes
- Installed this session: `num2words`, `accelerate>=1.1`, (needs `blenderproc`, `peft`
  for LoRA, `trl` optional). torch 2.10+cu128, transformers 5.13, datasets 4.7 present.
- **Always set `CUDA_VISIBLE_DEVICES=0`** for any GPU job (gpu1 reserved).
