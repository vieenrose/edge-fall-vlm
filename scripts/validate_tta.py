"""Iteration 3: test-time augmentation (TTA) — retraining-free recall boost for small models.

For each clip, run the model on the original + K augmented views and aggregate by max
severity (any view says down/distress -> that). Targets the small models' conservatism:
if a fall is visible in ANY view, we catch it. Trades inference cost (K+1 passes) for recall.

    CUDA_VISIBLE_DEVICES=0 python scripts/validate_tta.py --model runs/sft-500m-real \
        --manifest data/real/oops/oops_val.json --k 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validate_real import load_manifest_as_samples
from training.dataset import domain_augment
from training.eval import parse_answer, score
from training.sft import _subsample

SEV = {"down": 3, "fall": 3, "faint-collapse": 3, "lying-immobile": 3,
       "distress": 2, "danger": 3, "normal": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--label-set", default="down3")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from training.dataset import load_images

    samples = load_manifest_as_samples(args.manifest, args.label_set)[:args.n]
    proc = AutoProcessor.from_pretrained(args.model, do_image_splitting=False,
                                         size={"longest_edge": 384})
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
    PROMPT = samples[0].prompt
    rng = np.random.default_rng(0)

    def run(imgs):
        msgs = [{"role": "user", "content": [{"type": "image"} for _ in imgs] +
                 [{"type": "text", "text": PROMPT}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        b = proc(text=[text], images=[imgs], return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = model.generate(**b, max_new_tokens=48, do_sample=False)
        gen = proc.batch_decode(o[:, b["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return parse_answer(gen).get("status", "normal")

    preds, golds = [], []
    for s in samples:
        base = _subsample(load_images(s), 6)
        views = [base] + [domain_augment(base, rng) for _ in range(args.k)]
        statuses = [run(v) for v in views]
        # aggregate: highest severity across views
        best = max(statuses, key=lambda st: SEV.get(st, 0))
        preds.append(best); golds.append(s.label)
    m = score(preds, golds)
    print(f"=== TTA (k={args.k}) {args.model} on {args.manifest.name} ===")
    print(f"person-down recall={m.person_down_recall} spec={m.binary_specificity} acc={m.accuracy}")


if __name__ == "__main__":
    main()
