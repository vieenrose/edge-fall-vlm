---
title: Edge Fall / Danger Detection VLM
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
python_version: "3.12"
short_description: Fall / person-down / distress from a video clip
startup_duration_timeout: 1h
---

# Edge Fall / Danger Detection VLM

Single-stage **Qwen3.5-2B** fine-tune that flags **falls, a person down, or distress**
from a short video clip. Trained mostly on **synthetic 3D data** plus real in-the-wild
fall footage.

- Model: https://huggingface.co/Luigi/edge-fall-vlm-qwen3.5-2b
- Sibling model (SmolVLM2-2.2B + GGUF): https://huggingface.co/Luigi/edge-fall-vlm-2.2b
- Code: https://github.com/vieenrose/edge-fall-vlm

Upload a clip or try an example. Research prototype — **not a medical/safety device**.

## Rebuilding this Space from GitHub
This directory is a complete, self-contained HuggingFace Space definition (the YAML header
above configures a Gradio ZeroGPU Space). To recreate the demo:
1. Create a new Space (SDK: Gradio, hardware: ZeroGPU) on HuggingFace.
2. Upload the contents of this `space/` directory (`app.py`, `requirements.txt`, this
   `README.md`, and `examples/`) to the Space repo root.
The model is loaded by ID from the Hub (`Luigi/edge-fall-vlm-qwen3.5-2b` in `app.py`), so
no weights are stored here — the Space pulls them at runtime. `gradio` is provided by the
SDK and `spaces` (ZeroGPU) by the runtime, so neither is pinned in `requirements.txt`.

## Example footage attribution
The example clips are from the **UR Fall Detection (URFD) dataset** — B. Kwolek & M.
Kepski, *"Human fall detection on embedded platform using depth maps and wireless
accelerometer"*, Computer Methods and Programs in Biomedicine, 2014.
Source: http://fenix.ur.edu.pl/~mkepski/ds/uf.html (held-out test clips, used for
illustration/evaluation).
