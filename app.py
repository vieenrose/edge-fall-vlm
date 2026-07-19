import spaces  # must precede torch / transformers (ZeroGPU monkey-patches torch.cuda)

import json
import re

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "Luigi/edge-fall-vlm-2.2b"
N_FRAMES = 6
IMG_SIZE = 384

PROMPT = ("You are a safety monitor. These are consecutive video frames (oldest first), "
          "possibly with more than one person. Report whether ANYONE has fallen, fainted, "
          "is lying immobile, or is in distress; else normal. Answer with JSON only.")

# Load on CPU at module scope. We move to GPU inside @spaces.GPU when ZeroGPU quota is
# available, and fall back to CPU inference when it is exhausted (see analyze()).
processor = AutoProcessor.from_pretrained(MODEL_ID, do_image_splitting=False,
                                          size={"longest_edge": IMG_SIZE})
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16).eval()

_JSON = re.compile(r"\{[^{}]*\}")
_LABEL = {"down": "🔴 PERSON DOWN (fall / fainted / lying immobile)",
          "distress": "🟠 DISTRESS", "normal": "🟢 normal"}


def _canon(s: str) -> str:
    s = (s or "").lower()
    if "distress" in s: return "distress"
    if any(k in s for k in ("fall", "fallen", "down", "immobil", "lying", "faint", "collaps")):
        return "down"
    return "normal"


def _sample_frames(video_path: str) -> list[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    if n <= 0:
        while True:
            ok, f = cap.read()
            if not ok: break
            frames.append(f)
    else:
        for i in np.linspace(0, n - 1, N_FRAMES).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok: frames.append(f)
    cap.release()
    if not frames:
        raise gr.Error("Could not read any frames from the video.")
    if len(frames) > N_FRAMES:
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, N_FRAMES).astype(int)]
    return [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]


def _infer(frames, device: str) -> str:
    """Run the model on `device` ('cuda' or 'cpu'). Moves weights as needed."""
    model.to(device)
    msgs = [{"role": "user", "content": [{"type": "image"} for _ in frames] +
             [{"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    batch = processor(text=[text], images=[frames], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**batch, max_new_tokens=64, do_sample=False)
    return processor.batch_decode(out[:, batch["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0]


@spaces.GPU(duration=40)
def _infer_gpu(frames) -> str:
    return _infer(frames, "cuda")


def _format(gen: str, used: str):
    m = list(_JSON.finditer(gen))
    parsed = {}
    if m:
        try: parsed = json.loads(m[-1].group(0))
        except json.JSONDecodeError: pass
    status = _canon(parsed.get("status", gen))
    parsed.setdefault("status", status)
    parsed["_compute"] = used
    return _LABEL[status], parsed


def analyze(video_path):
    """Detect fall / person-down / distress in a short clip. Uses ZeroGPU when quota is
    available, else falls back to (slower) CPU inference."""
    if not video_path:
        raise gr.Error("Please provide a video clip.")
    frames = _sample_frames(video_path)
    try:
        gen = _infer_gpu(frames)          # ZeroGPU
        used = "ZeroGPU"
    except Exception as e:                 # quota exhausted / no GPU -> CPU fallback
        msg = str(e).lower()
        if any(k in msg for k in ("quota", "gpu", "zerogpu", "exceeded", "abort")):
            gr.Info("ZeroGPU quota unavailable — running on CPU (slower, ~30–60 s).")
            gen = _infer(frames, "cpu")
            used = "CPU fallback"
        else:
            raise
    label, parsed = _format(gen, used)
    return label, parsed, frames


with gr.Blocks(title="Edge Fall / Danger Detection VLM") as demo:
    gr.Markdown(
        "# 🛡️ Edge Fall / Danger Detection — single VLM\n"
        "A **SmolVLM2-2.2B** fine-tune ([`Luigi/edge-fall-vlm-2.2b`]"
        "(https://huggingface.co/Luigi/edge-fall-vlm-2.2b)) that flags **falls, a person "
        "down, or distress** from a short clip — trained mostly on **synthetic 3D data**, "
        "sized to run on a **Raspberry Pi 5**. "
        "[Code](https://github.com/vieenrose/edge-fall-vlm).\n\n"
        "Upload a short clip (or try an example — **real footage** from the UR Fall Detection "
        "dataset [Kwolek & Kepski, 2014]). Runs on **ZeroGPU**; if your daily GPU quota is "
        "exhausted it falls back to **CPU** (slower). Note: trained mostly on synthetic data, "
        "so it may false-alarm on out-of-distribution real scenes. "
        "*Research prototype — not a medical/safety device.*")
    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            vid = gr.Video(label="Video clip", sources=["upload"], height=440,
                           autoplay=True)
            btn = gr.Button("Analyze", variant="primary")
            gr.Examples(
                examples=[["examples/real_fall.mp4"], ["examples/real_person_down.mp4"],
                          ["examples/real_normal_walk.mp4"], ["examples/real_normal_sit.mp4"]],
                inputs=vid, label="Example clips (real footage — UR Fall Detection dataset)")
        with gr.Column(scale=2):
            verdict = gr.Label(label="Verdict")
            raw = gr.JSON(label="Model output")
            gallery = gr.Gallery(label="Sampled frames (oldest → newest)", columns=6, height=160)
    btn.click(analyze, inputs=vid, outputs=[verdict, raw, gallery])

if __name__ == "__main__":
    demo.launch(mcp_server=True)
