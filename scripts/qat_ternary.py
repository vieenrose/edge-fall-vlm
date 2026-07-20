"""QAT toward ternary (or binary) weights for the 2.2B text decoder, via self-distillation.

"Lossless" ternary is not a real thing (ternary is ~1.6 bit/weight vs fp16's 16 bit — a ~10x
compression, fundamentally lossy). This is the honest alternative: quantization-aware training
with a straight-through estimator (STE), so the fp16 *master* weights are nudged during
fine-tuning toward values that survive ternary rounding well. The frozen original checkpoint
acts as a self-distillation teacher (KL on answer-token logits) so the student is trained to
reproduce the SAME behavior under the ternary constraint, not just refit ground truth.

Only training/text_model Linear layers are QAT-wrapped (1.71B/2.25B params, 168 layers) —
that's what a GGUF TQ1_0/TQ2_0 export would actually compress. Vision tower + connector +
lm_head stay frozen fp16 (small fraction of params, and lm_head/embeddings are known-fragile
under extreme quantization).

    CUDA_VISIBLE_DEVICES=0 python scripts/qat_ternary.py --mode ternary --out runs/sft-2b-qat-ternary
    CUDA_VISIBLE_DEVICES=1 python scripts/qat_ternary.py --mode binary  --out runs/sft-2b-qat-binary
"""
from __future__ import annotations

import argparse
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.dataset import domain_augment, load_images, target_text
from training.sft import _subsample

PROMPT = ("You are a safety monitor. These are consecutive video frames (oldest first), "
          "possibly with more than one person. Report whether ANYONE has fallen, fainted, "
          "is lying immobile, or is in distress; else normal. Answer with JSON only.")


def fake_quant(w: torch.Tensor, mode: str) -> torch.Tensor:
    """BitNet-b1.58-style absmean scale, per-tensor. STE via w + (wq - w).detach()."""
    scale = w.detach().abs().mean().clamp_min(1e-5)
    if mode == "ternary":
        wq = (w / scale).round().clamp(-1, 1) * scale
    elif mode == "binary":
        wq = w.sign() * scale
        wq = torch.where(w == 0, torch.zeros_like(wq), wq)
    else:
        raise ValueError(mode)
    return w + (wq - w).detach()


MLP_ROLES = ("gate_proj", "up_proj", "down_proj")
ATTN_ROLES = ("q_proj", "k_proj", "v_proj", "o_proj")

SCOPES = {
    "all": lambda name: True,
    "mlp": lambda name: name.endswith(MLP_ROLES),
    "attn": lambda name: name.endswith(ATTN_ROLES),
    "mlp_gateup": lambda name: name.endswith(("gate_proj", "up_proj")),
}


def qat_wrap(root: nn.Module, mode: str, scope: str = "all") -> int:
    """Monkey-patch matching nn.Linear.forward under `root` to fake-quantize its weight.
    Keeps the real nn.Parameter as the trainable master weight. Non-matching Linears stay
    plain fp16/bf16 (still trainable — lets them compensate for the quantized subset)."""
    match = SCOPES[scope]
    n = 0
    for name, mod in root.named_modules():
        if isinstance(mod, nn.Linear) and match(name):
            def make_forward(m):
                def forward(self, x):
                    w = fake_quant(self.weight, mode)
                    return F.linear(x, w, self.bias)
                return forward
            mod.forward = types.MethodType(make_forward(mod), mod)
            n += 1
    return n


def build_inputs(proc, frames, answer_text, device):
    msgs = [{"role": "user", "content": [{"type": "image"} for _ in frames] +
             [{"type": "text", "text": PROMPT}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer_text}]}]
    full = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    prompt_only = proc.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True)
    batch = proc(text=[full], images=[frames], return_tensors="pt").to(device)
    p = proc(text=[prompt_only], images=[frames], return_tensors="pt")
    prompt_len = p["input_ids"].shape[1]
    return batch, prompt_len


def answer_logits_and_targets(model, batch, prompt_len):
    ids = batch["input_ids"][0]
    n_full = ids.shape[0]
    logits = model(**batch).logits[0]
    ans_logits = logits[prompt_len - 1:n_full - 1].float()
    ans_targets = ids[prompt_len:n_full]
    return ans_logits, ans_targets


def ternary_fit_report(model, mode: str, scope: str = "all") -> dict:
    """How much L2 error would REAL (non-STE) rounding introduce right now, vs a fresh
    (non-QAT'd) fp16 checkpoint would show. Lower = weights have moved toward quantization
    grid points during QAT."""
    match = SCOPES[scope]
    tot_err, tot_norm = 0.0, 0.0
    for name, mod in model.model.text_model.named_modules():
        if isinstance(mod, nn.Linear) and match(name):
            w = mod.weight.detach()
            scale = w.abs().mean().clamp_min(1e-5)
            if mode == "ternary":
                wq = (w / scale).round().clamp(-1, 1) * scale
            else:
                wq = w.sign() * scale
            tot_err += (w - wq).pow(2).sum().item()
            tot_norm += w.pow(2).sum().item()
    return {"relative_rounding_error": (tot_err / tot_norm) ** 0.5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="runs/sft-2b-real", help="KD reference (always original fp16)")
    ap.add_argument("--init", default=None, help="student init checkpoint (default: --teacher); "
                    "pass a previous QAT output dir to continue/anneal training")
    ap.add_argument("--mode", choices=["ternary", "binary"], required=True)
    ap.add_argument("--samples", default="data/train_mixed.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--alpha", type=float, default=0.5, help="KD weight vs CE")
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--label-set", default="down3")
    ap.add_argument("--scope", choices=list(SCOPES), default="all",
                    help="which Linear roles to QAT-wrap: all/mlp(gate,up,down)/attn(q,k,v,o)")
    args = ap.parse_args()

    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from training.dataset import load_records

    proc = AutoProcessor.from_pretrained(args.teacher, do_image_splitting=False, size={"longest_edge": 384})
    teacher = AutoModelForImageTextToText.from_pretrained(args.teacher, dtype=torch.bfloat16).to("cuda").eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    student = AutoModelForImageTextToText.from_pretrained(args.init or args.teacher, dtype=torch.bfloat16).to("cuda")
    n_wrapped = qat_wrap(student.model.text_model, args.mode, args.scope)
    print(f"QAT-wrapped {n_wrapped} Linear layers in text_model ({args.mode}, scope={args.scope})", flush=True)
    before = ternary_fit_report(student, args.mode, args.scope)
    print("rounding error BEFORE QAT:", before, flush=True)

    # only the text_model needs updating (that's what's being QAT'd); freeze vision/connector/lm_head
    for p in student.parameters(): p.requires_grad_(False)
    for p in student.model.text_model.parameters(): p.requires_grad_(True)
    student.train()

    samples = load_records(args.samples, label_set=args.label_set)
    from collections import defaultdict
    by = defaultdict(list)
    for s in samples: by[s.label].append(s)
    labels = list(by)
    rng = np.random.default_rng(0)
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr)

    step = 0; opt.zero_grad(); running = 0.0
    while step < args.steps:
        s = rng.choice(by[labels[step % len(labels)]])
        frames = _subsample(load_images(s), 6)
        if rng.random() < 0.6:
            frames = domain_augment(frames, rng)
        ans = target_text(s, with_rationale=False)

        with torch.no_grad():
            tb, t_plen = build_inputs(proc, frames, ans, "cuda")
            t_l, _ = answer_logits_and_targets(teacher, tb, t_plen)
        sb, s_plen = build_inputs(proc, frames, ans, "cuda")
        s_l, ans_ids = answer_logits_and_targets(student, sb, s_plen)

        ce = F.cross_entropy(s_l, ans_ids)
        kd = F.kl_div(F.log_softmax(s_l / args.temp, -1),
                      F.softmax(t_l / args.temp, -1), reduction="batchmean") * (args.temp ** 2)
        loss = (1 - args.alpha) * ce + args.alpha * kd
        (loss / args.accum).backward()
        running += loss.item()
        if (step + 1) % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        if (step + 1) % 200 == 0:
            print(f"step {step+1}/{args.steps} loss {running/200:.4f} (ce {ce.item():.3f} kd {kd.item():.3f})", flush=True)
            running = 0.0
        if (step + 1) % 500 == 0:
            r = ternary_fit_report(student, args.mode, args.scope)
            print(f"  [rounding error @ step {step+1}]: {r}", flush=True)
        step += 1

    after = ternary_fit_report(student, args.mode, args.scope)
    print("rounding error AFTER QAT:", after, flush=True)

    # Save with REAL (non-STE) rounding baked in — this is the model that will actually be
    # exported/deployed at ternary/binary precision, not the STE fp16 proxy.
    match = SCOPES[args.scope]
    with torch.no_grad():
        for name, mod in student.model.text_model.named_modules():
            if isinstance(mod, nn.Linear) and match(name):
                w = mod.weight.data
                scale = w.abs().mean().clamp_min(1e-5)
                if args.mode == "ternary":
                    wq = (w / scale).round().clamp(-1, 1) * scale
                else:
                    wq = w.sign() * scale
                mod.weight.data.copy_(wq)
                mod.forward = nn.Linear.forward.__get__(mod, nn.Linear)  # unwrap QAT hook

    Path(args.out).mkdir(parents=True, exist_ok=True)
    student.save_pretrained(args.out)
    proc.save_pretrained(args.out)
    import json
    (Path(args.out) / "qat_report.json").write_text(json.dumps(
        {"mode": args.mode, "scope": args.scope, "n_wrapped_layers": n_wrapped, "steps": args.steps,
         "rounding_error_before": before, "rounding_error_after": after}, indent=2))
    print("QAT model saved (weights ARE REAL", args.mode, "values now) ->", args.out)


if __name__ == "__main__":
    main()
