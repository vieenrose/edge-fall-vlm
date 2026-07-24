---
title: Edge Fall / Danger Detection VLM — Static Demo
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
short_description: Fall / person-down / distress detection (static)
---

# Edge Fall / Danger Detection VLM — static demo

A **static** showcase of a single-stage **Qwen3.5-2B** vision-language model that flags
**falls, a person down, or distress** from a short strip of video frames — trained mostly on
**synthetic 3D data** plus real in-the-wild fall footage.

Because this is a static Space (no backend GPU), the example verdicts shown here are the
**real outputs of the deployed model, pre-computed** — not live inference. To run the model
yourself on new clips, use the model directly:

- Model: https://huggingface.co/Luigi/edge-fall-vlm-qwen3.5-2b
- Sibling (SmolVLM2-2.2B + GGUF): https://huggingface.co/Luigi/edge-fall-vlm-2.2b
- Code: https://github.com/vieenrose/edge-fall-vlm

The "upload a clip" tool below runs entirely in your browser: it extracts the 6-frame strip
the model would see, so you can preview the exact input — the verdict step needs the model
(linked above), since a 2B VLM can't run in a static page.

Example footage: UR Fall Detection dataset (Kwolek & Kepski, 2014), used for illustration.
Research prototype — **not a medical/safety device.**
