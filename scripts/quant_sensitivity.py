"""Layer-role/depth sensitivity sweep for mixed-precision ternary quantization.

Naive RTN ternary quantization (no retraining) applied to ONE group of Linear layers at a
time in an otherwise-fp16 2.2B checkpoint, evaluated on a held-out val slice. This finds
which weight ROLES (q/k/v/o/gate/up/down proj) and which DEPTH band (early/mid/late layers)
actually cost accuracy when compressed, vs which are "free" to ternary-ize.

Groups are applied and then reverted (in-memory) between experiments so we load the base
model once. Uses the OOPS VAL split (not test) to avoid contaminating the final report.

    CUDA_VISIBLE_DEVICES=0 python scripts/quant_sensitivity.py --n 50
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validate_real import load_manifest_as_samples
from training.eval import parse_answer, score
from training.sft import _subsample

ROLES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
N_LAYERS = 24  # SmolVLM2-2.2B text decoder depth


def ternary_rtn_(w: torch.Tensor) -> torch.Tensor:
    scale = w.abs().mean().clamp_min(1e-5)
    return (w / scale).round().clamp(-1, 1) * scale


def find_linears(text_model, match_fn):
    return [(name, mod) for name, mod in text_model.named_modules()
            if isinstance(mod, nn.Linear) and match_fn(name)]


def layer_idx(name: str) -> int | None:
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def apply_ternary(targets):
    """Returns backup list of (module, original_weight_data) for later restore."""
    backup = []
    for _, mod in targets:
        backup.append((mod, mod.weight.data.clone()))
        mod.weight.data.copy_(ternary_rtn_(mod.weight.data))
    return backup


def restore(backup):
    for mod, w in backup:
        mod.weight.data.copy_(w)


def evaluate(model, proc, samples, device, max_frames=6, max_new_tokens=48):
    preds, golds = [], []
    from training.dataset import load_images
    for s in samples:
        imgs = _subsample(load_images(s), max_frames)
        msgs = [{"role": "user", "content": [{"type": "image"} for _ in imgs] +
                 [{"type": "text", "text": s.prompt}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        batch = proc(text=[text], images=[imgs], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
        gen = proc.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        preds.append(parse_answer(gen).get("status", "normal"))
        golds.append(s.label)
    return score(preds, golds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runs/sft-2b-real")
    ap.add_argument("--manifest", default="data/real/oops/oops_val.json")
    ap.add_argument("--label-set", default="down3")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("quant_sensitivity_report.json"))
    args = ap.parse_args()

    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(args.base, do_image_splitting=False, size={"longest_edge": 384})
    model = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.bfloat16).to(args.device).eval()
    text_model = model.model.text_model

    samples = load_manifest_as_samples(Path(args.manifest), args.label_set)[:args.n]
    print(f"sensitivity sweep on {len(samples)} val clips", flush=True)

    results = {}

    # 0. sanity: pure fp16 (no quantization)
    m = evaluate(model, proc, samples, args.device)
    results["fp16_baseline"] = m.__dict__
    print("fp16_baseline:", m, flush=True)

    # 1. per-role sweep: quantize ALL layers of one role, keep rest fp16
    for role in ROLES:
        targets = find_linears(text_model, lambda n, r=role: n.endswith(r))
        backup = apply_ternary(targets)
        m = evaluate(model, proc, samples, args.device)
        restore(backup)
        results[f"role_{role}"] = m.__dict__
        print(f"role={role} (n_layers_hit={len(targets)}):", m, flush=True)

    # 2. per-depth sweep: quantize ALL roles but only in one depth third
    thirds = {"early_0-7": range(0, 8), "mid_8-15": range(8, 16), "late_16-23": range(16, 24)}
    for name, rng in thirds.items():
        targets = find_linears(text_model, lambda n, rng=rng: (layer_idx(n) in rng))
        backup = apply_ternary(targets)
        m = evaluate(model, proc, samples, args.device)
        restore(backup)
        results[f"depth_{name}"] = m.__dict__
        print(f"depth={name} (n_layers_hit={len(targets)}):", m, flush=True)

    # 3. reference: uniform full ternary, no QAT (expected to be badly broken, like Q4_K_M)
    targets = find_linears(text_model, lambda n: True)
    backup = apply_ternary(targets)
    m = evaluate(model, proc, samples, args.device)
    restore(backup)
    results["full_ternary_naive_no_qat"] = m.__dict__
    print(f"full_ternary_naive (n_layers_hit={len(targets)}):", m, flush=True)

    args.out.write_text(json.dumps(results, indent=2, default=str))
    print("\n=== SUMMARY (accuracy / recall / spec vs fp16_baseline) ===")
    base = results["fp16_baseline"]
    for k, v in results.items():
        print(f"{k:<28} acc={v['accuracy']:<7} recall(down)={v.get('person_down_recall'):<7} "
              f"spec={v['binary_specificity']:<7} "
              f"(delta acc vs fp16: {round(v['accuracy']-base['accuracy'],3)})")


if __name__ == "__main__":
    main()
