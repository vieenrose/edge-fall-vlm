import spaces  # must precede torch / transformers (ZeroGPU monkey-patches torch.cuda)

import json
import re

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_IDS = {
    "2.2B — best accuracy (recommended)": "Luigi/edge-fall-vlm-2.2b",
    "500M — faster, weaker": "Luigi/edge-fall-vlm-500m",
    "256M — fastest, weakest": "Luigi/edge-fall-vlm-256m",
}
DEFAULT_MODEL = "2.2B — best accuracy (recommended)"
N_FRAMES = 6
IMG_SIZE = 384

PROMPT = ("You are a safety monitor. These are consecutive video frames (oldest first), "
          "possibly with more than one person. Report whether ANYONE has fallen, fainted, "
          "is lying immobile, or is in distress; else normal. Answer with JSON only.")

_cache = {}   # model_id -> (processor, model on CPU)


def _get(model_id):
    if model_id not in _cache:
        proc = AutoProcessor.from_pretrained(model_id, do_image_splitting=False,
                                             size={"longest_edge": IMG_SIZE})
        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16).eval()
        _cache[model_id] = (proc, model)
    return _cache[model_id]


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


_SEV = {"down": 3, "distress": 2, "normal": 0}


def _domain_aug(frames):
    """Compact real-video-style augmentation for test-time augmentation (self-contained)."""
    from PIL import ImageEnhance, ImageFilter
    import random
    b, c, s = random.uniform(0.7, 1.4), random.uniform(0.7, 1.4), random.uniform(0.6, 1.3)
    blur = random.random() < 0.4
    out = []
    for im in frames:
        im = ImageEnhance.Brightness(im).enhance(b)
        im = ImageEnhance.Contrast(im).enhance(c)
        im = ImageEnhance.Color(im).enhance(s)
        if blur:
            im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.4, 1.3)))
        w, h = im.size
        d = random.uniform(0.5, 1.0)
        if d < 1.0:
            im = im.resize((max(8, int(w * d)), max(8, int(h * d)))).resize((w, h))
        out.append(im)
    return out


def _status_of(gen: str) -> str:
    m = list(_JSON.finditer(gen))
    if m:
        try: return _canon(json.loads(m[-1].group(0)).get("status", gen))
        except json.JSONDecodeError: pass
    return _canon(gen)


def _generate(proc, model, frames, device) -> str:
    msgs = [{"role": "user", "content": [{"type": "image"} for _ in frames] +
             [{"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    batch = proc(text=[text], images=[frames], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**batch, max_new_tokens=64, do_sample=False)
    return proc.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def _run(proc, model, frames, device, k):
    """k=0: single pass. k>0: test-time augmentation — original + k augmented views,
    keep the most severe verdict. Recovers small-model recall (256M 0.16→0.47,
    500M 0.31→0.55 in-the-wild; see SIZE_COMPARISON.md)."""
    if k <= 0:
        return _generate(proc, model, frames, device)
    views = [frames] + [_domain_aug(frames) for _ in range(k)]
    gens = [_generate(proc, model, v, device) for v in views]
    best_g, best_sev = gens[0], -1
    for g in gens:
        sev = _SEV.get(_status_of(g), 0)
        if sev > best_sev:
            best_g, best_sev = g, sev
    return best_g


@spaces.GPU(duration=90)
def _infer_gpu(frames, model_id, k) -> str:
    proc, model = _get(model_id)
    model.to("cuda")
    try:
        return _run(proc, model, frames, "cuda", k)
    finally:
        model.to("cpu")   # keep model on CPU between calls (ZeroGPU releases the GPU)


def _format(gen: str, used: str, size: str):
    m = list(_JSON.finditer(gen))
    parsed = {}
    if m:
        try: parsed = json.loads(m[-1].group(0))
        except json.JSONDecodeError: pass
    status = _canon(parsed.get("status", gen))
    parsed.setdefault("status", status)
    parsed["_model"] = size.split(" ")[0]
    parsed["_compute"] = used
    return _LABEL[status], parsed


def analyze(video_path, size, tta):
    """Detect fall / person-down / distress in a short clip with the chosen model size.
    `tta` enables test-time augmentation (higher recall for small models, ~5x slower).
    Uses ZeroGPU when quota is available, else falls back to (slower) CPU."""
    if not video_path:
        raise gr.Error("Please provide a video clip.")
    model_id = MODEL_IDS[size]
    k = 4 if tta else 0
    frames = _sample_frames(video_path)
    try:
        gen = _infer_gpu(frames, model_id, k)
        used = "ZeroGPU" + (" +TTA" if tta else "")
    except Exception as e:
        msg = str(e).lower()
        if any(x in msg for x in ("quota", "gpu", "zerogpu", "exceeded", "abort")):
            gr.Info("ZeroGPU quota unavailable — running on CPU (slower).")
            proc, model = _get(model_id)
            gen = _run(proc, model, frames, "cpu", k)
            used = "CPU fallback" + (" +TTA" if tta else "")
        else:
            raise
    label, parsed = _format(gen, used, size)
    return label, parsed, frames


with gr.Blocks(title="Edge Fall / Danger Detection VLM") as demo:
    gr.Markdown(
        "# 🛡️ Edge Fall / Danger Detection — single VLM\n"
        "A **SmolVLM2** fine-tune that flags **falls, a person down, or distress** from a "
        "short clip — trained mostly on **synthetic 3D data**, sized to run on a "
        "**Raspberry Pi 5**. Pick a **model size** to compare accuracy vs size "
        "([details](https://github.com/vieenrose/edge-fall-vlm/blob/master/SIZE_COMPARISON.md)). "
        "[Code](https://github.com/vieenrose/edge-fall-vlm).\n\n"
        "Try an example (**real footage**, UR Fall Detection dataset [Kwolek & Kepski, 2014]) "
        "or upload a clip. **Note:** smaller models miss most *in-the-wild* falls; the 2.2B "
        "is the only size usable for the real task. Trained on synthetic data, so it may "
        "false-alarm on out-of-distribution real scenes. *Research prototype — not a "
        "medical/safety device.*")
    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            size = gr.Radio(choices=list(MODEL_IDS), value=DEFAULT_MODEL, label="Model size")
            tta = gr.Checkbox(value=False, label="Test-time augmentation (boosts small-model "
                              "recall — 256M 0.16→0.47, 500M 0.31→0.55 in-the-wild; ~5× slower)")
            vid = gr.Video(label="Video clip", sources=["upload"], height=440, autoplay=True)
            btn = gr.Button("Analyze", variant="primary")
            gr.Examples(
                examples=[["examples/real_fall.mp4"], ["examples/real_person_down.mp4"],
                          ["examples/real_normal_walk.mp4"], ["examples/real_normal_sit.mp4"]],
                inputs=vid, label="Example clips (real footage — UR Fall Detection dataset)")
        with gr.Column(scale=2):
            verdict = gr.Label(label="Verdict")
            raw = gr.JSON(label="Model output")
            gallery = gr.Gallery(label="Sampled frames (oldest → newest)", columns=6, height=160)
    btn.click(analyze, inputs=[vid, size, tta], outputs=[verdict, raw, gallery])

if __name__ == "__main__":
    demo.launch(mcp_server=True)
