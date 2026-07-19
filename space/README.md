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

Single-stage **SmolVLM2-2.2B** fine-tune that flags **falls, a person down, or distress**
from a short video clip. Trained mostly on **synthetic 3D data**; runs on a Raspberry Pi 5
at Q6_K GGUF.

- Model: https://huggingface.co/Luigi/edge-fall-vlm-2.2b
- Code: https://github.com/vieenrose/edge-fall-vlm

Upload a clip or try an example. Research prototype — **not a medical/safety device**.

## Example footage attribution
The example clips are from the **UR Fall Detection (URFD) dataset** — B. Kwolek & M.
Kepski, *"Human fall detection on embedded platform using depth maps and wireless
accelerometer"*, Computer Methods and Programs in Biomedicine, 2014.
Source: http://fenix.ur.edu.pl/~mkepski/ds/uf.html (held-out test clips, used for
illustration/evaluation).
