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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
N_FRAMES = 6
IMG_SIZE = 384

PROMPT = ("You are a safety monitor. These are consecutive video frames (oldest first), "
          "possibly with more than one person. Report whether ANYONE has fallen, fainted, "
          "is lying immobile, or is in distress; else normal. Answer with JSON only.")

# Load at module scope; ZeroGPU packs weights and streams them on first @spaces.GPU call.
processor = AutoProcessor.from_pretrained(MODEL_ID, do_image_splitting=False,
                                          size={"longest_edge": IMG_SIZE})
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE).eval()

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
    if n <= 0:  # fallback: read sequentially
        while True:
            ok, f = cap.read()
            if not ok: break
            frames.append(f)
    else:
        idx = np.linspace(0, n - 1, N_FRAMES).astype(int)
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok: frames.append(f)
    cap.release()
    if not frames:
        raise gr.Error("Could not read any frames from the video.")
    if len(frames) > N_FRAMES:
        idx = np.linspace(0, len(frames) - 1, N_FRAMES).astype(int)
        frames = [frames[i] for i in idx]
    return [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]


@spaces.GPU(duration=60)
def analyze(video_path):
    """Detect fall / person-down / distress in a short video clip and return the verdict."""
    if not video_path:
        raise gr.Error("Please provide a video clip.")
    frames = _sample_frames(video_path)
    msgs = [{"role": "user", "content": [{"type": "image"} for _ in frames] +
             [{"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    batch = processor(text=[text], images=[frames], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**batch, max_new_tokens=64, do_sample=False)
    gen = processor.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    m = list(_JSON.finditer(gen))
    parsed = {}
    if m:
        try: parsed = json.loads(m[-1].group(0))
        except json.JSONDecodeError: pass
    status = _canon(parsed.get("status", gen))
    label = _LABEL[status]
    parsed.setdefault("status", status)
    return label, parsed, frames


with gr.Blocks(title="Edge Fall / Danger Detection VLM") as demo:
    gr.Markdown(
        "# 🛡️ Edge Fall / Danger Detection — single VLM\n"
        "A **SmolVLM2-2.2B** fine-tune ([`Luigi/edge-fall-vlm-2.2b`]"
        "(https://huggingface.co/Luigi/edge-fall-vlm-2.2b)) that flags **falls, a person "
        "down, or distress** from a short clip — trained mostly on **synthetic 3D data**, "
        "sized to run on a **Raspberry Pi 5**. "
        "[Code](https://github.com/vieenrose/edge-fall-vlm).\n\n"
        "Upload a short clip (or try an example). The model samples 6 frames and returns a "
        "JSON verdict. *Research prototype — not a medical/safety device.*")
    with gr.Row():
        with gr.Column():
            vid = gr.Video(label="Video clip", sources=["upload"])
            btn = gr.Button("Analyze", variant="primary")
            gr.Examples(
                examples=[["examples/fall.mp4"], ["examples/person_down.mp4"],
                          ["examples/distress.mp4"], ["examples/normal.mp4"]],
                inputs=vid, label="Example clips (synthetic)")
        with gr.Column():
            verdict = gr.Label(label="Verdict")
            raw = gr.JSON(label="Model output")
            gallery = gr.Gallery(label="Sampled frames (oldest → newest)", columns=6, height=140)
    btn.click(analyze, inputs=vid, outputs=[verdict, raw, gallery])

if __name__ == "__main__":
    demo.launch(mcp_server=True)
