"""Soft-label knowledge distillation: 2.2B teacher -> small student (deployment-valid;
student ships single-pass). The SmolVLM2 family shares the SmolLM2 tokenizer, so we match
the teacher's and student's next-token DISTRIBUTIONS on the answer tokens via KL (the
"dark knowledge" beyond the hard label), combined with CE on ground truth.

Loss = (1-a)*CE(student, GT) + a*T^2*KL(softmax(student/T) || softmax(teacher/T))
on the answer-token span (same token IDs in both, shared vocab).

Manual bs=1 loop (teacher and student have different image-token counts, so we align the
last-K answer positions rather than pad-batching). Domain augmentation on inputs so the
teacher's soft targets are demonstrated on OOD-like views too.

    CUDA_VISIBLE_DEVICES=0 python scripts/distill_train.py \
        --student HuggingFaceTB/SmolVLM2-256M-Video-Instruct --out runs/sft-256m-distill \
        --samples data/train_mixed.jsonl --steps 1500
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.dataset import Sample, collapse_label, domain_augment, load_images, target_text
from training.sft import _subsample

PROMPT = ("You are a safety monitor. These are consecutive video frames (oldest first), "
          "possibly with more than one person. Report whether ANYONE has fallen, fainted, "
          "is lying immobile, or is in distress; else normal. Answer with JSON only.")


def build_inputs(proc, frames, answer_text, device):
    """Teacher-force (images + prompt + answer). Returns (batch, prompt_len) where the
    answer span is input_ids[prompt_len:]. Computing prompt_len WITH images correctly
    locates the answer regardless of chat-template wrapper / image-token count."""
    msgs = [{"role": "user", "content": [{"type": "image"} for _ in frames] +
             [{"type": "text", "text": PROMPT}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer_text}]}]
    full = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    prompt_only = proc.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True)
    batch = proc(text=[full], images=[frames], return_tensors="pt").to(device)
    p = proc(text=[prompt_only], images=[frames], return_tensors="pt")
    prompt_len = p["input_ids"].shape[1]
    return batch, prompt_len


def answer_logits_and_targets(model, batch, prompt_len, V):
    """Logits at positions predicting the answer tokens, and the answer target ids."""
    ids = batch["input_ids"][0]
    n_full = ids.shape[0]
    logits = model(**batch).logits[0, :, :V]
    ans_logits = logits[prompt_len - 1:n_full - 1].float()   # predict ids[prompt_len:]
    ans_targets = ids[prompt_len:n_full]
    return ans_logits, ans_targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", default="runs/sft-2b-real")
    ap.add_argument("--samples", default="data/train_mixed.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--alpha", type=float, default=0.7, help="KD weight vs CE")
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--label-set", default="down3")
    ap.add_argument("--accum", type=int, default=4)
    args = ap.parse_args()

    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from training.dataset import load_records

    t_proc = AutoProcessor.from_pretrained(args.teacher, do_image_splitting=False, size={"longest_edge": 384})
    s_proc = AutoProcessor.from_pretrained(args.student, do_image_splitting=False, size={"longest_edge": 384})
    teacher = AutoModelForImageTextToText.from_pretrained(args.teacher, dtype=torch.bfloat16).to("cuda").eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    student = AutoModelForImageTextToText.from_pretrained(args.student, dtype=torch.bfloat16).to("cuda").train()

    samples = load_records(args.samples, label_set=args.label_set)
    # balance by class
    from collections import defaultdict
    by = defaultdict(list)
    for s in samples: by[s.label].append(s)
    rng = np.random.default_rng(0)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)

    V = min(teacher.config.get_text_config().vocab_size, student.config.get_text_config().vocab_size) \
        if hasattr(teacher.config, "get_text_config") else student.config.text_config.vocab_size

    step = 0; opt.zero_grad()
    running = 0.0
    while step < args.steps:
        labels = list(by)
        s = rng.choice(by[labels[step % len(labels)]])   # class-balanced sampling
        frames = _subsample(load_images(s), 6)
        if rng.random() < 0.6:
            frames = domain_augment(frames, rng)          # OOD-like views for the teacher to label
        ans = target_text(s, with_rationale=False)         # short JSON status target

        with torch.no_grad():
            tb, t_plen = build_inputs(t_proc, frames, ans, "cuda")
            t_l, _ = answer_logits_and_targets(teacher, tb, t_plen, V)
        sb, s_plen = build_inputs(s_proc, frames, ans, "cuda")
        s_l, ans_ids = answer_logits_and_targets(student, sb, s_plen, V)

        k = min(s_l.shape[0], t_l.shape[0])
        s_l, t_l, ans_ids = s_l[-k:], t_l[-k:], ans_ids[-k:]
        ce = F.cross_entropy(s_l, ans_ids)
        # soft KD
        kd = F.kl_div(F.log_softmax(s_l / args.temp, -1),
                      F.softmax(t_l / args.temp, -1), reduction="batchmean") * (args.temp ** 2)
        loss = (1 - args.alpha) * ce + args.alpha * kd
        (loss / args.accum).backward()
        running += loss.item()
        if (step + 1) % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        if (step + 1) % 100 == 0:
            print(f"step {step+1}/{args.steps} loss {running/100:.4f} (ce {ce.item():.3f} kd {kd.item():.3f})", flush=True)
            running = 0.0
        step += 1

    Path(args.out).mkdir(parents=True, exist_ok=True)
    student.save_pretrained(args.out)
    s_proc.save_pretrained(args.out)
    print("distilled student saved ->", args.out)


if __name__ == "__main__":
    main()
