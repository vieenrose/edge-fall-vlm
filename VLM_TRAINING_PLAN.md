# AI-Care-2 — Single-VLM Training Plan (v1)

**Decision (2026-07-18):** The deliverable is **ONE VLM** that performs fall/faint/danger detection on RPi5 4GB end-to-end. Pose datasets/annotations may be used as **training supervision only** — no explicit pose detector ships in, or is built as part of, the final system.

Companion doc: `RESEARCH_PLAN.md` (Phase-0 evidence base). Training runs on **gpu0 only** (RTX 5090 32GB).

---

## 1. On-device reality (measured numbers, RPi5 / llama.cpp GGUF Q4)

| Model | Latency per image | RAM | Verdict for 4GB 24/7 |
|---|---|---|---|
| **SmolVLM-256M** | ~0.4s encode + short decode ≈ **1–2s** | ~0.5GB | ✅ primary candidate |
| **SmolVLM-500M** | ~0.8s encode ≈ **2–3s** | ~0.9GB | ✅ primary candidate |
| LFM2-VL-450M | no Pi numbers published; 230M text sibling does 42 tok/s on Pi 5 | ~<1GB est. | 🟡 benchmark ourselves |
| FastVLM-0.5B | no Pi numbers (fast on phone SoC); cheap vision encoder = right design point | ? | 🟡 benchmark ourselves |
| Moondream-0.5B | ~9s/image (Station stack wants 8GB) | high | ❌ |
| Qwen2-VL-2B / Moondream-2B | ~8–20s/image, ~1.5GB+ | marginal | ❌ for always-on |

Consequences baked into the design:
- **Cadence, not framerate.** The VLM runs at ~0.5–1 Hz (motion-gated), not per-frame. Fall→alert latency target: **< 10s**, which is clinically fine (what matters is detecting the person *down*, not the 300ms of falling).
- **Short structured outputs.** Decode dominates after encode; we train the model to answer in ~5–10 tokens (`{"status":"fall","conf":0.93}`), keeping total latency ≈ encode cost.
- **Temporal input, single inference.** Falls are temporal; the model gets an **N-frame strip/mosaic spanning the last ~3–4s** (SmolVLM2 is video-capable) so one forward pass sees the motion history. A trivial pixel-diff motion gate (not a model) triggers burst sampling.

## 2. Base model choice

**Primary: SmolVLM2-500M-Video-Instruct.** Video-native, Apache-2.0, proven GGUF/llama.cpp path, fits 4GB with big headroom, ~2–3s/inference on Pi.
**Fallback if too slow after quantization:** SmolVLM2-256M.
**Challengers to benchmark on the actual Pi in week 1:** LFM2-VL-450M, FastVLM-0.5B. Whichever hits the best latency×accuracy stays; the training recipe below is base-agnostic.

## 2b. Synthetic data generation — the primary lever for perspective + fisheye robustness

Rationale: falls cannot be collected from many angles/lenses in reality (rare, risky, one camera per take). In a renderer, **camera pose and lens are free parameters** — render one fall, emit it from N viewpoints and lens models with pixel-exact labels. Evidence this transfers: BEDLAM (synthetic-only human perception matches real, 46.6mm PA-MPJPE on real 3DPW); OmniFall's `OF-Synthetic` (12k diffusion-generated fall videos) **beat** staged-real on in-the-wild falls (F1 64.2 vs 61.2). Confirmed gap we fill ourselves: **no synthetic fisheye fall dataset exists.**

Two complementary generators:

**(i) Engine rendering — geometry + fisheye (primary).**
- Bodies: **SMPL-X** driven by fall motion capture. AMASS is fall-poor → source fall/recovery motion from **LAFAN1** (Ubisoft) + **ragdoll physics** for impact dynamics.
- Renderer: **Blender/BlenderProc** (native equidistant & equisolid fisheye, scriptable calibrated lens distortion + GT) or **Isaac Sim / Omniverse Replicator** (Kannala-Brandt / F-theta up to ~200° FOV) to match a specific real camera's calibration.
- Backgrounds/scenes: **Infinigen** indoor scenes + heavy domain randomization (lighting, textures, camera height/pitch/roll, distractors).
- **Free bonus:** the renderer *is* the ground-truth pose + camera oracle → the view-canonical pose-grounded rationales (Stage B) come out of the render pass, no separate pose model needed for synthetic clips.

**(ii) Video diffusion — appearance diversity (secondary).**
- Extend OmniFall's approach with **Wan 2.2 / CogVideoX** for varied people/clothing/environments. Weakness: no calibrated fisheye GT, occasional physics violations → use for appearance, not geometry; filter physically-implausible clips.

Curriculum (converged across BEDLAM / OmniFall / Phase-0 fisheye work): **synthetic (engine fisheye + diffusion appearance) → fine-tune on a small real fisheye fall set** (few dozen self-recorded clips) to close the residual sim-to-real gap.

Risks: engine pipeline is ~1–2 weeks of real build effort and the highest-risk-highest-reward item; fall *physics* fidelity matters (a wrong-looking fall teaches a wrong cue) → tune ragdoll / prefer real fall mocap.

## 3. Training recipe (gpu0, RTX 5090 32GB)

At 256M–500M params, we can **full-fine-tune** (no LoRA compromise needed) in bf16 with room to spare; batch via frame-strip packing.

### Stage A — Domain adaptation (fisheye + surveillance viewpoint)
- **Primary data: our synthetic engine renders (§2b)** — SMPL-X falls/ADL from many viewpoints and fisheye lens models with exact labels; the highest-leverage source for perspective robustness.
- Data: CEPDOF / WEPDTOF / HABBOF / MW-R (overhead fisheye people) turned into QA pairs from their box annotations: "How many people? Where? Anyone on the floor?" 
- Apply **FED fisheye augmentation** (equidistant r = f·θ) to perspective datasets to synthesize distortion; strong lighting/night augmentation (CEPDOF has real low-light/IR).
- Purpose: teach the encoder distorted, overhead, low-light humans before any fall semantics.

### Stage B — Fall/danger SFT with pose-grounded rationales
- Data: **OmniFall (`OF-Staged` + `OF-Synthetic` 12k) + our engine renders (§2b)** → clips → N-frame strips → instruction pairs.
- **Pose as supervision, not runtime:** for real clips, run an offline pose model (gpu0) to auto-generate grounded rationales — "body horizontal, hips at floor level, rapid descent between frames 2–4" — as chain-of-thought targets; for synthetic clips the renderer emits exact 3D pose + camera, so rationales are **view-canonicalized for free**. The pose model is a data-factory tool, discarded after training.
- **Teacher distillation:** Qwen2.5-VL-32B (quantized, fits gpu0) labels ambiguous/unlabeled clips with rich descriptions; distill into the small model.
- Label schema: `fall / faint-collapse / lying-immobile / distress / normal`, JSON short-form output.

### Stage C — Hard negatives & preference tuning
- The killer confusions: sitting down fast, lying down deliberately, exercise, pets, occluded lower body. Mine these from OmniFall ADL segments + OOPS non-fall clips.
- DPO/KTO pass on (correct vs false-alarm) answer pairs to push false-alarms/day down — the metric that decides real-world usability (older in-home systems averaged ~5.4 false alarms/day).

### Stage D — Quantization-aware finish
- Export GGUF (Q4_K_M weights; keep vision encoder Q8 if accuracy drops); re-eval post-quant on the full test suite — small models degrade more from quantization, so the post-quant score is the only score that counts.

## 4. Evaluation (honest, per Phase-0 findings)

1. **OOPS-Fall** (in-the-wild) sensitivity/specificity — primary metric; staged accuracy is known to collapse in the wild (VideoMAE 0.78→0.21).
2. **Cross-view splits** on OmniFall (29 views) — never report cross-subject only.
3. **Fisheye holdout:** self-recorded fisheye clips (staged falls in our own space) + FED-distorted OOPS.
4. **On-Pi soak:** 24h+ continuous run — latency, RAM ceiling, temperature (throttle at 80°C, active cooler mandatory), false-alarms/day on continuous normal-life footage.

## 5. Milestones

| # | Deliverable | When |
|---|---|---|
| M1 | Pi bench harness: SmolVLM2-256M/500M, LFM2-VL-450M, FastVLM-0.5B GGUF latency/RAM/thermal on the actual RPi5 4GB; pick base | wk 1 |
| M2 | Data factory: OmniFall+fisheye datasets downloaded; FED aug; strip-builder; pose-rationale + teacher labeling pipelines on gpu0 | wk 2–3 |
| M3 | Stage A+B trained; first OOPS-Fall / cross-view numbers | wk 4–5 |
| M4 | Stage C+D; quantized model beating false-alarm target (<1/day) on soak footage | wk 6–7 |
| M5 | 24/7 Pi deployment demo: camera → motion gate → VLM → alert, 24h soak report | wk 8 |

## 6. Known risks (accepted with the single-VLM choice)

- **No published precedent** for continuous VLM fall-detection on Pi-class hardware — we are first; M1 exists to fail fast on latency.
- ~1 Hz cadence means very brief events rely on the multi-frame strip capturing them; mitigated by motion-gated burst sampling and "person on floor" persistence (a fallen person stays fallen).
- Tiny-VLM quantization loss is unknown for this task until M4 — hence quantized-eval-only policy.
