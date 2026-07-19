"""Iteration 2 probe: run the 2.2B teacher over the training clips and (a) report how much
it DISAGREES with ground-truth (if ~0, sequence-distillation ≡ GT training → skip), and
(b) write a teacher-labeled samples file for distillation if it's worth it.

    CUDA_VISIBLE_DEVICES=0 python scripts/distill_label.py --n 400
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.dataset import Sample, collapse_label, load_records
from training.eval import parse_answer
from training.sft import _subsample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="runs/sft-2b-real")
    ap.add_argument("--samples", default="data/train_mixed.jsonl")
    ap.add_argument("--out", default="data/distill_teacher.jsonl")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--label-set", default="down3")
    args = ap.parse_args()

    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from training.dataset import load_images

    # dedupe by id (train_mixed oversamples)
    raw = [json.loads(l) for l in Path(args.samples).read_text().splitlines() if l.strip()]
    seen, uniq = set(), []
    for r in raw:
        if r["id"] in seen: continue
        seen.add(r["id"]); uniq.append(r)
    import random; random.Random(0).shuffle(uniq)
    uniq = uniq[:args.n]

    proc = AutoProcessor.from_pretrained(args.teacher, do_image_splitting=False,
                                         size={"longest_edge": 384})
    model = AutoModelForImageTextToText.from_pretrained(args.teacher, dtype=torch.bfloat16).to("cuda").eval()
    PROMPT = ("You are a safety monitor. These are consecutive video frames (oldest first), "
              "possibly with more than one person. Report whether ANYONE has fallen, fainted, "
              "is lying immobile, or is in distress; else normal. Answer with JSON only.")

    agree = disagree = t_down_gt_normal = 0
    conf = Counter()
    out_lines = []
    for r in uniq:
        gt = collapse_label(r["class"], args.label_set)
        s = Sample(id=r["id"], frames=r["frames"], prompt=PROMPT, rationale="",
                   answer={}, label=gt, split_key="", lighting="day")
        imgs = _subsample(load_images(s), 6)
        msgs = [{"role": "user", "content": [{"type": "image"} for _ in imgs] +
                 [{"type": "text", "text": PROMPT}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        batch = proc(text=[text], images=[imgs], return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = model.generate(**batch, max_new_tokens=48, do_sample=False)
        gen = proc.batch_decode(o[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        t = collapse_label(parse_answer(gen).get("status", "normal"), args.label_set)
        conf[(gt, t)] += 1
        if t == gt: agree += 1
        else:
            disagree += 1
            if gt == "normal" and t in ("down", "distress"): t_down_gt_normal += 1
        r2 = dict(r); r2["class"] = t
        a = dict(r.get("answer", {})); a["status"] = t; r2["answer"] = a
        out_lines.append(json.dumps(r2))

    n = agree + disagree
    Path(args.out).write_text("\n".join(out_lines) + "\n")
    print(f"teacher vs GT on {n} clips: agree {agree} ({agree/n:.2%}), disagree {disagree} ({disagree/n:.2%})")
    print(f"teacher says DANGER where GT=normal: {t_down_gt_normal}")
    print("confusion (gt -> teacher):", {f"{k[0]}->{k[1]}": v for k, v in sorted(conf.items(), key=lambda x:-x[1])[:8]})
    print(f"-> teacher labels written to {args.out}")


if __name__ == "__main__":
    main()
