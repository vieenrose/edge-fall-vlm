"""Full benchmark: accuracy/recall/precision + peak RSS + inference/prefill time, in one pass,
for any HF-format model dir (fp16/bf16 checkpoints — the QAT-ternary/binary ones included, since
they're stored as plain SmolVLM safetensors, just with weight VALUES at ternary/binary grid
points). Runs on this GPU box for a same-hardware relative comparison across sizes/quant
schemes; absolute Pi-CPU numbers still need deploy/bench.py run ON the Pi with the GGUF export.

Prefill = one forward pass over the full (images + prompt) context, no generation — this is
the "time to first token" the images/vision-tower cost dominates. Inference = the full
generate() call (prefill + all decode steps). Peak RSS = this process's high-water-mark
resident memory (resource.RUSAGE_SELF), sampled after each model's full run.

    CUDA_VISIBLE_DEVICES=0 python scripts/bench_full.py \\
        --model runs/sft-2b-real=2.2B-fp16 \\
        --model runs/sft-2b-qat-ternary=2.2B-ternary-QAT \\
        --manifest data/real/oops/oops_test.json --n 100 --device cuda
"""
from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validate_real import load_manifest_as_samples
from training.eval import parse_answer, score
from training.sft import _subsample


def peak_rss_mb() -> float:
    """High-water-mark RSS of THIS process (Linux: ru_maxrss is in KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def bench_one(model_dir: str, samples, device: str, max_frames: int, img_size: int,
              max_new_tokens: int) -> dict:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from training.dataset import load_images

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    try:
        proc = AutoProcessor.from_pretrained(model_dir, do_image_splitting=False,
                                             size={"longest_edge": img_size})
    except (ValueError, TypeError):
        # non-SmolVLM processors (e.g. Qwen3.5's Qwen2VLImageProcessorFast, sized by pixel
        # area) don't take these kwargs -- the saved model dir already has correct sizing.
        proc = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=dtype).to(device).eval()

    prefill_t, gen_t = [], []
    preds, golds, records = [], [], []
    for i, s in enumerate(samples):
        imgs = _subsample(load_images(s), max_frames)
        msgs = [{"role": "user", "content": [{"type": "image"} for _ in imgs] +
                 [{"type": "text", "text": s.prompt}]}]
        try:
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
        except TypeError:
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        batch = proc(text=[text], images=[imgs], return_tensors="pt").to(device)

        if device == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(**batch)   # prefill only: one forward pass, no decode loop
        if device == "cuda": torch.cuda.synchronize()
        prefill_t.append(time.perf_counter() - t0)

        if device == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
        if device == "cuda": torch.cuda.synchronize()
        gen_t.append(time.perf_counter() - t0)

        gen = proc.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        d = parse_answer(gen)
        preds.append(d.get("status", "normal"))
        golds.append(s.label)
        records.append({"id": s.id, "gold": s.label, "pred": preds[-1],
                        "parse": d.get("_parse", "json"), "confidence": d.get("confidence"),
                        "raw": gen})
        if (i + 1) % 25 == 0:
            print(f"  {model_dir}: {i+1}/{len(samples)}", flush=True)

    m = score(preds, golds)
    del model
    if device == "cuda":
        import torch as _t
        _t.cuda.empty_cache()

    def stats(xs):
        return {"p50_s": round(statistics.median(xs), 3),
                "p95_s": round(sorted(xs)[int(0.95 * len(xs)) - 1], 3),
                "mean_s": round(statistics.mean(xs), 3)}

    return {
        "n": m.n, "accuracy": m.accuracy,
        "binary_sensitivity_recall": m.binary_sensitivity,
        "binary_specificity": m.binary_specificity,
        "binary_precision": _precision_from_confusion(m),
        "binary_f1": m.binary_f1,
        "person_down_recall": m.person_down_recall,
        "parse_failures": sum(1 for r in records if r["parse"] != "json"),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "prefill_time": stats(prefill_t),
        "inference_time_full_generate": stats(gen_t),
        # per-clip records make paired tests (McNemar), ensemble fusion, and parse-failure
        # audits possible — aggregates alone cannot resolve the deltas this project gates on
        "predictions": records,
    }


def _precision_from_confusion(m) -> float:
    from training.eval import DANGER
    tp = fp = 0
    for g, row in (m.confusion or {}).items():
        for p, c in row.items():
            if p in DANGER and g in DANGER: tp += c
            elif p in DANGER and g not in DANGER: fp += c
    return round(tp / (tp + fp), 3) if (tp + fp) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="path=label, repeatable")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--label-set", default="down3")
    ap.add_argument("--n", type=int, default=100, help="clip count; 0 = the full manifest")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-frames", type=int, default=6)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("bench_full_report.json"))
    args = ap.parse_args()

    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    samples = load_manifest_as_samples(args.manifest, args.label_set)
    if args.n > 0:
        samples = samples[:args.n]
    print(f"benchmarking {len(samples)} clips from {args.manifest}", flush=True)

    report = {}
    for spec in args.model:
        path, _, label = spec.partition("=")
        label = label or path
        print(f"=== {label} ({path}) ===", flush=True)
        report[label] = bench_one(path, samples, args.device, args.max_frames,
                                  args.img_size, args.max_new_tokens)
        print(json.dumps({k: v for k, v in report[label].items() if k != "predictions"},
                         indent=2), flush=True)

    args.out.write_text(json.dumps(report, indent=2))
    print("\n=== SUMMARY ===")
    hdr = f"{'model':<24}{'acc':>7}{'recall':>8}{'prec':>7}{'spec':>7}{'down_rec':>10}{'RSS_MB':>9}{'prefill_p50':>13}{'infer_p50':>11}"
    print(hdr)
    for label, r in report.items():
        print(f"{label:<24}{r['accuracy']:>7}{r['binary_sensitivity_recall']:>8}"
              f"{r['binary_precision']:>7}{r['binary_specificity']:>7}"
              f"{str(r['person_down_recall']):>10}{r['peak_rss_mb']:>9}"
              f"{r['prefill_time']['p50_s']:>13}{r['inference_time_full_generate']['p50_s']:>11}")


if __name__ == "__main__":
    main()
