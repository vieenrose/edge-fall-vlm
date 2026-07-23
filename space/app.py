import spaces  # must precede torch / transformers (ZeroGPU monkey-patches torch.cuda)

import json
import re

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "Luigi/edge-fall-vlm-qwen3.5-2b"
N_FRAMES = 6
MIN_PIXELS = 64 * 64
MAX_PIXELS = 384 * 384

# A fall is a brief event (~1-4s). Sampling 6 frames uniformly across a long clip dilutes
# the fall frames among the surrounding normal activity and washes the verdict out to
# "normal" (this is what made IMG_9144-style ~60s clips read as normal). Instead we slide
# a short window across the clip, score each window, and take the max-severity verdict --
# a fall ANYWHERE in the clip raises the alert. This mirrors the real streaming deployment
# (deploy/monitor.py), where each ~3.5s window is inferred separately and persistence
# raises the alert. Bounded to MAX_WINDOWS so the ZeroGPU per-call budget holds.
WIN_SEC = 3.0
MAX_WINDOWS = 10
_SEVERITY = {"down": 2, "distress": 1, "normal": 0}

PROMPT = ("You are a safety monitor. These are consecutive video frames (oldest first), "
          "possibly with more than one person. Report whether ANYONE has fallen, fainted, "
          "is lying immobile, or is in distress; else normal. Answer with JSON only.")

_cache = {}   # model_id -> (processor, model on CPU)


def _get(model_id=MODEL_ID):
    if model_id not in _cache:
        proc = AutoProcessor.from_pretrained(model_id, do_image_splitting=False,
                                             size={"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS})
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


def _read_all_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise gr.Error("Could not read any frames from the video.")
    return frames, (fps if fps > 0 else 20.0)


def _sample_windows(video_path: str):
    """Return a list of windows, each a list of N_FRAMES PIL images, plus per-window start
    times (seconds). Short clips -> one window (single pass, unchanged behavior). Long
    clips -> up to MAX_WINDOWS short windows spanning the clip, so a brief fall is not
    diluted by the surrounding normal activity."""
    frames, fps = _read_all_frames(video_path)
    n = len(frames)
    win = max(N_FRAMES, int(WIN_SEC * fps))
    if n <= int(win * 1.5):
        starts = [0]
        spans = [(0, n - 1)]
    else:
        nwin = min(MAX_WINDOWS, max(2, int(np.ceil(n / win))))
        starts = np.linspace(0, n - win, nwin).astype(int).tolist()
        spans = [(s, s + win - 1) for s in starts]
    windows, times = [], []
    for a, b in spans:
        idxs = np.linspace(a, b, N_FRAMES).astype(int)
        windows.append([frames[i] for i in idxs])
        times.append(round(a / fps, 1))
    return windows, times


def _generate(proc, model, frames, device) -> str:
    msgs = [{"role": "user", "content": [{"type": "image"} for _ in frames] +
             [{"type": "text", "text": PROMPT}]}]
    # enable_thinking=False is required for Qwen-family chat templates -- without it, the
    # template injects a <think> block and generation rambles into open-ended reasoning
    # instead of the trained JSON format. Harmless no-op for non-Qwen models.
    try:
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
    except TypeError:
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    batch = proc(text=[text], images=[frames], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**batch, max_new_tokens=64, do_sample=False)
    return proc.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)[0]


@spaces.GPU(duration=60)
def _infer_gpu(windows) -> list:
    proc, model = _get()
    model.to("cuda")
    try:
        return [_generate(proc, model, w, "cuda") for w in windows]
    finally:
        model.to("cpu")   # keep model on CPU between calls (ZeroGPU releases the GPU)


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
    """Detect fall / person-down / distress in a clip. Slides a short window across the
    clip and takes the MAX-severity verdict (a fall anywhere = alert) -- so a brief fall
    in a long clip is not diluted to 'normal'. Uses ZeroGPU, else CPU fallback."""
    if not video_path:
        raise gr.Error("Please provide a video clip.")
    windows, times = _sample_windows(video_path)
    try:
        gens = _infer_gpu(windows)
        used = "ZeroGPU"
    except Exception as e:
        msg = str(e).lower()
        if any(x in msg for x in ("quota", "gpu", "zerogpu", "exceeded", "abort")):
            gr.Info("ZeroGPU quota unavailable — running on CPU (slower).")
            proc, model = _get()
            gens = [_generate(proc, model, w, "cpu") for w in windows]
            used = "CPU fallback"
        else:
            raise

    # pick the most severe window as the verdict
    best_i, best_label, best_parsed = 0, "normal", {}
    for i, gen in enumerate(gens):
        label, parsed = _format(gen, used)
        if _SEVERITY.get(parsed.get("status", "normal"), 0) > _SEVERITY.get(best_label, 0):
            best_i, best_label, best_parsed = i, parsed.get("status", "normal"), parsed
    if not best_parsed:
        _, best_parsed = _format(gens[0], used)
    if len(windows) > 1:
        best_parsed["_windows"] = len(windows)
        if best_label != "normal":
            best_parsed["_event_at_sec"] = times[best_i]
    return _LABEL[_canon(best_parsed.get("status", "normal"))], best_parsed, windows[best_i]


with gr.Blocks(title="Edge Fall / Danger Detection VLM") as demo:
    gr.Markdown(
        "# 🛡️ Fall / Danger Detection — single VLM\n"
        "A **Qwen3.5-2B** fine-tune that flags **falls, a person down, or distress** "
        "from a short clip — trained mostly on **synthetic 3D data**. Chosen for "
        "**capability on hard real cases over raw benchmark recall** — see the model "
        "card for the honest trade-off vs. the SmolVLM2 sibling model. "
        "[Code](https://github.com/vieenrose/edge-fall-vlm) · "
        "[Model card](https://huggingface.co/Luigi/edge-fall-vlm-qwen3.5-2b).\n\n"
        "Try an example (**real footage**, UR Fall Detection dataset [Kwolek & Kepski, 2014]) "
        "or upload a clip. Trained on synthetic data, so it may false-alarm on out-of-"
        "distribution real scenes. *Research prototype — not a medical/safety device.*")
    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
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
    btn.click(analyze, inputs=[vid], outputs=[verdict, raw, gallery])

if __name__ == "__main__":
    demo.launch(mcp_server=True)
