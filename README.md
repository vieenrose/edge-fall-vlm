# AI-Care — Fall / Danger Detection with a Single VLM

A research prototype: **one vision-language model** that detects human
**falls, fainting, and distress** from a short strip of video frames — trained almost
entirely on **synthetic 3D data**, robust to perspective, lighting, and fisheye/ultra-wide
distortion. Two sibling fine-tunes exist with a genuine trade-off (see below); the project
now prioritizes **capability/quality over on-device footprint** (the original Raspberry Pi
5 / 4 GB constraint has been dropped in favor of that priority).

## Headline results

| What | Result |
|---|---|
| In-the-wild fall recall (OOPS benchmark, SmolVLM2-2.2B) | **0.83** — beats VideoMAE (0.21) and frozen-I3D (0.68) baselines |
| Real overhead falls (URFD, after small real-negative fine-tune) | **0.90 recall / 1.0 specificity** |
| Sim-to-real transfer (synthetic-only → real falls) | **0.95 recall** before any real data |

## Two deployed models — a genuine trade-off, not a strict winner

| Model | Accuracy (n=210) | Recall (n=210) | Hard real clip* |
|---|---|---|---|
| [`edge-fall-vlm-2.2b`](https://huggingface.co/Luigi/edge-fall-vlm-2.2b) (SmolVLM2-2.2B) | 0.800 | **0.897** | Missed |
| [`edge-fall-vlm-qwen3.5-2b`](https://huggingface.co/Luigi/edge-fall-vlm-qwen3.5-2b) (Qwen3.5-2B, **currently deployed in the demo**) | **0.847** | 0.784 | **Detected** |

*A real user-submitted clip (elderly person falling from a bent/reaching position near a
chair, cluttered/low-light room) was missed by every SmolVLM2-2.2B variant tried, including
three rounds of synthetic-data fidelity improvements. Same training recipe/data on a
different backbone (Qwen3.5-2B) caught it. SmolVLM2-2.2B still has meaningfully better
general recall on the in-the-wild benchmark. Pick based on your priority — see each model's
card for the full honest write-up.

Model: single-stage VLM (no separate pose detector). Trained on synthetic renders +
a small real-negative set. See [`TRAINING_NOTES.md`](TRAINING_NOTES.md) for the full arc.

## The recipe (what actually worked)

1. **Synthetic 3D data → perspective/fisheye robustness.** Render 3D humans from many
   viewpoints and lens models (incl. native fisheye) with exact labels — camera pose is a
   free parameter. ([`SYNTHETIC_DATA_SPEC.md`](SYNTHETIC_DATA_SPEC.md), [`SYNTH3D.md`](SYNTH3D.md))
2. **A 2.2B-class VLM, not 500M.** Capacity was the bottleneck; 500M couldn't separate
   "fallen" from "crouching/sitting". 2.2B can.
3. **Synthetic falls transfer to real** out of the box (0.95 recall).
4. **A small real *negative* set fixes specificity** (0 → 1.0 on held-out real).
5. **Backbone choice is itself a lever, not just size/data.** Two models trained with the
   same recipe on the same data can disagree on hard real cases — see the trade-off table
   above.
6. **A verification stack** (motion gate → temporal confirm → persistence timer → optional
   human) is required to hit the commercial false-alarm bar.
   ([`COMMERCIAL_BAR.md`](COMMERCIAL_BAR.md))

## Repo layout

```
synthgen/      synthetic 3D data generation (Blender/BlenderProc, fisheye, mannequin, clutter/HDRI)
scripts/       data build, training driver, real-dataset validation, GGUF export
training/      dataset loader, SFT (sft.py = SmolVLM2, sft_qwen35.py = Qwen3.5), eval
deploy/        motion gate, strip buffer, alert hysteresis, verification stack
docs/          sample renders
*.md           research plan, training notes, deployment/commercial analysis
```

## Reproduce

```bash
pip install -r requirements.txt            # torch, transformers, blenderproc, ...
python scripts/convert_motion.py --out data/synth3d/motions --scale 6   # procedural motions
blenderproc run scripts/blender_dataset.py -- --manifest ... --out ...   # render 3D data
python training/sft.py --base HuggingFaceTB/SmolVLM2-2.2B-Instruct ...    # train SmolVLM2 (gpu)
python training/sft_qwen35.py --base Qwen/Qwen3.5-2B ...                  # train Qwen3.5 (gpu)
python scripts/validate_real.py --model runs/... --manifest ...          # validate on real
```

Blender-free tests: `bash scripts/run_tests.sh`.

## Models & demo
- **Model (currently deployed):** https://huggingface.co/Luigi/edge-fall-vlm-qwen3.5-2b (Qwen3.5-2B fine-tune)
- **Sibling model:** https://huggingface.co/Luigi/edge-fall-vlm-2.2b (SmolVLM2-2.2B fine-tune + GGUF, better general recall)
- **Demo:** ZeroGPU Gradio app, live at https://huggingface.co/spaces/Luigi/edge-fall-vlm-demo — code in [`space/`](space/)

## Honest status
Research prototype. Strong sim-to-real transfer and competitive in-the-wild recall, but
**not a product**: validated on small test sets (60–300 clips), no certification, and the
false-alarm rate at 24/7 scale needs the verification layer + real-site data. See
[`COMMERCIAL_BAR.md`](COMMERCIAL_BAR.md) for the gap to a commercial system.

## License
Apache-2.0 (inherits the SmolVLM2 base). Third-party datasets used for evaluation (URFD,
OmniFall/OOPS) are **not** redistributed here — see their original sources for terms.
