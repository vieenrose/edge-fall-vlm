"""System-level benchmark: windowed inference on UNTRIMMED source videos.

The trimmed OOPS benchmark (bench_full.py) scores the model on 6-frame strips cut tightly
around each labeled event — it cannot see what windowing does, because every clip is
already one window. This benchmark scores the DEPLOYED behavior: slide a window across a
full untrimmed video and aggregate, exactly like space/app.py / deploy/monitor.py.

It reports, for each aggregation rule, system recall on fall videos and the false-alarm
rate on normal videos (videos with no fall segment anywhere) — the tradeoff the trimmed
benchmark is blind to:
  - max@1  : alert if ANY window fires down (space/app.py's current rule)
  - Nof M  : alert only if >= N windows fire down (deploy/monitor.py persistence)

    CUDA_VISIBLE_DEVICES=0 python3 scripts/bench_system_longclip.py \
        --model runs/sft-qwen35-2b-realfall --videos data/real/oops_syseval \
        --categories data/real/oops/syseval_videos.json --out bench_system_longclip.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.eval import parse_answer, DANGER

PROMPT = ("You are a safety monitor. These are consecutive video frames (oldest first), "
          "possibly with more than one person. Report whether ANYONE has fallen, fainted, "
          "is lying immobile, or is in distress; else normal. Answer with JSON only.")
N_FRAMES = 6
WIN_SEC = 3.0
MAX_WINDOWS = 10


def load_windows(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) < N_FRAMES:
        return []
    n = len(frames)
    win = max(N_FRAMES, int(WIN_SEC * (fps if fps > 0 else 20.0)))
    if n <= int(win * 1.5):
        spans = [(0, n - 1)]
    else:
        nwin = min(MAX_WINDOWS, max(2, int(np.ceil(n / win))))
        spans = [(s, s + win - 1) for s in np.linspace(0, n - win, nwin).astype(int)]
    from PIL import Image
    return [[Image.fromarray(frames[i]) for i in np.linspace(a, b, N_FRAMES).astype(int)]
            for a, b in spans]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--categories", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("bench_system_longclip.json"))
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor
    try:
        proc = AutoProcessor.from_pretrained(args.model, do_image_splitting=False,
                                             size={"longest_edge": 384})
    except (ValueError, TypeError):
        proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16).to("cuda").eval()

    def down_count(windows):
        """Return (#windows firing danger, #windows total)."""
        d = 0
        for w in windows:
            msgs = [{"role": "user", "content": [{"type": "image"} for _ in w] +
                     [{"type": "text", "text": PROMPT}]}]
            try:
                text = proc.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            batch = proc(text=[text], images=[w], return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**batch, max_new_tokens=64, do_sample=False)
            gen = proc.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            if parse_answer(gen).get("status") in DANGER:
                d += 1
        return d, len(windows)

    cats = json.loads(args.categories.read_text())
    results = {}   # video -> {label, fired, nwin}
    for label, key in [("fall", "fall_only"), ("normal", "normal_only")]:
        vids = cats[key]
        for i, v in enumerate(vids):
            p = args.videos / v
            if not p.exists():
                continue
            windows = load_windows(p)
            if not windows:
                continue
            fired, nwin = down_count(windows)
            results[v] = {"label": label, "fired": fired, "nwin": nwin}
            if (i + 1) % 20 == 0:
                print(f"  {key}: {i+1}/{len(vids)}", flush=True)

    fall = [r for r in results.values() if r["label"] == "fall"]
    norm = [r for r in results.values() if r["label"] == "normal"]

    def eval_rule(min_fire):
        rec = sum(1 for r in fall if r["fired"] >= min_fire) / len(fall) if fall else float("nan")
        fa = sum(1 for r in norm if r["fired"] >= min_fire) / len(norm) if norm else float("nan")
        return round(rec, 3), round(fa, 3)

    report = {
        "model": args.model, "n_fall_videos": len(fall), "n_normal_videos": len(norm),
        "avg_windows": round(np.mean([r["nwin"] for r in results.values()]), 1),
        "rules": {f"min_fire_{k}": {"recall": eval_rule(k)[0], "false_alarm_rate": eval_rule(k)[1]}
                  for k in (1, 2, 3)},
        "per_video": results,
    }
    args.out.write_text(json.dumps(report, indent=2))
    print("\n=== SYSTEM benchmark (untrimmed video, windowed) ===")
    print(f"fall videos={len(fall)} normal videos={len(norm)} avg_windows={report['avg_windows']}")
    for k in (1, 2, 3):
        rec, fa = eval_rule(k)
        print(f"  >={k} window(s) fire down:  recall {rec}   false-alarm rate {fa}")


if __name__ == "__main__":
    main()
