# AI-Care — Edge Fall / Danger Detection with a Single VLM

A research prototype: **one compact vision-language model** that detects human
**falls, fainting, and distress** from a real-time RGB camera, designed to run **24/7 on a
Raspberry Pi 5 (4 GB)** and be robust to perspective, lighting, and fisheye/ultra-wide
distortion — trained almost entirely on **synthetic 3D data**.

## Headline results

| What | Result |
|---|---|
| In-the-wild fall recall (OOPS benchmark) | **0.83** — beats VideoMAE (0.21) and frozen-I3D (0.68) baselines |
| Real overhead falls (URFD, after small real-negative fine-tune) | **0.90 recall / 1.0 specificity** |
| Sim-to-real transfer (synthetic-only → real falls) | **0.95 recall** before any real data |
| On-device | **SmolVLM2-2.2B, Q6_K GGUF ≈ 2.2 GB, ~1.8 GB RAM** — fits RPi5 4 GB |

Model: single-stage VLM (no separate pose detector). Trained on synthetic renders +
a small real-negative set. See [`TRAINING_NOTES.md`](TRAINING_NOTES.md) for the full arc.

## The recipe (what actually worked)

1. **Synthetic 3D data → perspective/fisheye robustness.** Render 3D humans from many
   viewpoints and lens models (incl. native fisheye) with exact labels — camera pose is a
   free parameter. ([`SYNTHETIC_DATA_SPEC.md`](SYNTHETIC_DATA_SPEC.md), [`SYNTH3D.md`](SYNTH3D.md))
2. **A 2.2B VLM, not 500M.** Capacity was the bottleneck; 500M couldn't separate "fallen"
   from "crouching/sitting". 2.2B can, and still fits RPi5 at Q6_K.
3. **Synthetic falls transfer to real** out of the box (0.95 recall).
4. **A small real *negative* set fixes specificity** (0 → 1.0 on held-out real).
5. **A verification stack** (motion gate → temporal confirm → persistence timer → optional
   human) is required to hit the commercial false-alarm bar.
   ([`COMMERCIAL_BAR.md`](COMMERCIAL_BAR.md))

## Repo layout

```
synthgen/      synthetic 3D data generation (Blender/BlenderProc, fisheye, mannequin)
scripts/       data build, training driver, real-dataset validation, GGUF export
training/      dataset loader, SFT (SmolVLM2), eval (cross-view, confusion, person-down)
deploy/        RPi5 loop: motion gate, strip buffer, alert hysteresis, verification stack
docs/          sample renders
*.md           research plan, training notes, deployment/commercial analysis
```

## Reproduce

```bash
pip install -r requirements.txt            # torch, transformers, blenderproc, ...
python scripts/convert_motion.py --out data/synth3d/motions --scale 6   # procedural motions
blenderproc run scripts/blender_dataset.py -- --manifest ... --out ...   # render 3D data
python training/sft.py --base HuggingFaceTB/SmolVLM2-2.2B-Instruct ...    # train (gpu)
python scripts/validate_real.py --model runs/... --manifest ...          # validate on real
```

Blender-free tests: `bash scripts/run_tests.sh`.

## Models & demo
- **Model:** https://huggingface.co/Luigi/edge-fall-vlm-2.2b (SmolVLM2-2.2B fine-tune + GGUF)
- **Demo:** ZeroGPU Gradio app in [`space/`](space/) — `app.py` + example clips, ready to deploy (HF Gradio hosting requires a PRO plan)

## Honest status
Research prototype. Strong sim-to-real transfer and competitive in-the-wild recall, but
**not a product**: validated on small test sets (40–300 clips), no certification, and the
false-alarm rate at 24/7 scale needs the verification layer + real-site data. See
[`COMMERCIAL_BAR.md`](COMMERCIAL_BAR.md) for the gap to a commercial system.

## License
Apache-2.0 (inherits the SmolVLM2 base). Third-party datasets used for evaluation (URFD,
OmniFall/OOPS) are **not** redistributed here — see their original sources for terms.
