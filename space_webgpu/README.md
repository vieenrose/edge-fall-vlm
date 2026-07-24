---
title: Edge Fall Detection VLM — WebGPU (in-browser)
emoji: 🛡️
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
short_description: Fall / person-down detection VLM, live in-browser via WebGPU
---

# 🛡️ Fall / Danger Detection VLM — live in your browser (WebGPU)

This runs the **actual fine-tuned Qwen3.5-2B fall-detection VLM** entirely in your browser,
GPU-accelerated via **WebGPU** — no server, no upload of your video anywhere. Powered by
[wllama](https://github.com/ngxson/wllama) (llama.cpp compiled to WASM + WebGPU) loading the
model's GGUF weights.

- Model (GGUF): https://huggingface.co/Luigi/edge-fall-vlm-qwen3.5-2b-gguf
- Full-precision model: https://huggingface.co/Luigi/edge-fall-vlm-qwen3.5-2b
- Code: https://github.com/vieenrose/edge-fall-vlm

**Requirements:** a WebGPU-capable browser (Chrome/Edge 121+, or Firefox Nightly). First load
downloads ~1.9 GB of weights (cached afterwards). Everything runs locally on your GPU.

*Research prototype — not a medical/safety device.*
