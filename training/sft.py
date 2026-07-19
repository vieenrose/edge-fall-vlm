"""Supervised fine-tuning of a tiny VLM (SmolVLM2-500M default) on synth fall shards.

Runs on gpu0 (RTX 5090). Uses HF Trainer (trl optional). At 256M-500M params we
full-fine-tune in bf16; LoRA is available via --lora for larger bases.

    CUDA_VISIBLE_DEVICES=0 python training/sft.py \
        --samples data/bootstrap/shards/samples.jsonl \
        --base HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
        --out runs/sft-bootstrap --epochs 1 --bs 2

--smoke runs 2 optimizer steps to validate the pipeline without a full run.

NOTE: respects the gpu1 reservation via CUDA_VISIBLE_DEVICES — always set it to 0.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.dataset import Sample, load_images, load_records, target_text


def _subsample(items: list, k: int) -> list:
    if len(items) <= k:
        return items
    import numpy as np
    idx = np.linspace(0, len(items) - 1, k).astype(int)
    return [items[i] for i in idx]


def build_collator(processor, max_frames: int = 6, fps_augment: bool = True):
    import numpy as np
    from training.dataset import temporal_augment
    rng = np.random.default_rng(0)

    def collate(samples: list[Sample]):
        import torch
        texts, images_batch = [], []
        for s in samples:
            imgs = load_images(s)
            if fps_augment:
                imgs = temporal_augment(imgs, rng, max_n=max_frames)
            imgs = _subsample(imgs, max_frames)
            msgs = [{"role": "user",
                     "content": [{"type": "image"} for _ in imgs] + [{"type": "text", "text": s.prompt}]},
                    {"role": "assistant",
                     "content": [{"type": "text", "text": target_text(s, with_rationale=True)}]}]
            texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
            images_batch.append(imgs)
        batch = processor(text=texts, images=images_batch, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        # mask image tokens from the loss if the processor exposes the id
        img_tok = getattr(processor.tokenizer, "image_token_id", None)
        if img_tok is not None:
            labels[labels == img_tok] = -100
        batch["labels"] = labels
        return batch
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--base", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    ap.add_argument("--out", type=Path, default=Path("runs/sft"))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--img-size", type=int, default=384, help="processor longest_edge")
    ap.add_argument("--max-frames", type=int, default=6, help="cap frames/sample fed to VLM")
    ap.add_argument("--holdout-views", default="", help="comma-sep camera archetypes held out for cross-view eval")
    ap.add_argument("--label-set", default="fine", help="fine (5-class) | down3 (normal/down/distress) | binary")
    ap.add_argument("--balance", action="store_true", help="oversample minority classes to majority count")
    ap.add_argument("--eval-after", action="store_true", help="run cross-view eval after training")
    ap.add_argument("--eval-n", type=int, default=160, help="cap eval samples (generation is slow)")
    ap.add_argument("--smoke", action="store_true", help="2 steps to validate the pipeline")
    args = ap.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # never touch gpu1

    import torch
    from transformers import (AutoModelForImageTextToText, AutoProcessor, Trainer,
                              TrainingArguments)

    # Multi-frame strips: DISABLE image splitting (each frame would else tile into many
    # sub-images and blow up sequence length / VRAM). One patch-set per frame is plenty
    # for fall geometry. Cap the longest side too.
    processor = AutoProcessor.from_pretrained(
        args.base, do_image_splitting=False, size={"longest_edge": args.img_size})
    model = AutoModelForImageTextToText.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map=None)

    if args.lora:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()

    all_samples = load_records(args.samples, label_set=args.label_set)
    holdout = {v.strip() for v in args.holdout_views.split(",") if v.strip()}
    if holdout:
        from training.dataset import cross_view_split
        train_samples, test_samples = cross_view_split(all_samples, holdout)
        print(f"cross-view: train {len(train_samples)} (views != {holdout}) / "
              f"test {len(test_samples)} (held-out {holdout})")
    else:
        train_samples, test_samples = all_samples, []
    samples = train_samples
    if args.balance:
        # oversample minority classes to the majority count. The first bootstrap runs
        # showed a weak-signal model collapsing to the majority ("normal") class prior;
        # balancing removes that shortcut so the loss must actually use the images.
        from collections import Counter
        import numpy as np
        by = {}
        for s in samples:
            by.setdefault(s.label, []).append(s)
        target = max(len(v) for v in by.values())
        rng2 = np.random.default_rng(0)
        balanced = []
        for v in by.values():
            reps = [v[int(i)] for i in rng2.integers(0, len(v), target)]
            balanced += reps
        rng2.shuffle(balanced)
        print(f"balanced: {dict(Counter(s.label for s in samples))} -> "
              f"{dict(Counter(s.label for s in balanced))}")
        samples = balanced
    collate = build_collator(processor, max_frames=args.max_frames)

    targs = TrainingArguments(
        output_dir=str(args.out),
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=5,
        save_strategy="no",          # final model saved explicitly; avoids multi-GB epoch checkpoints
        save_total_limit=1,
        remove_unused_columns=False,
        max_steps=2 if args.smoke else -1,
        report_to=[],
    )
    trainer = Trainer(model=model, args=targs, data_collator=collate, train_dataset=samples)
    trainer.train()
    if not args.smoke:
        if args.lora:
            # merge the adapter into the base so eval / GGUF export load a normal model
            merged = model.merge_and_unload()
            merged.save_pretrained(str(args.out))
        else:
            trainer.save_model(str(args.out))
        processor.save_pretrained(str(args.out))
    print("SFT done ->", args.out)

    if args.eval_after and test_samples:
        import json as _json
        from training.eval import cross_view_gap, run_model, score
        print(f"\n=== cross-view eval on held-out {holdout} ===")
        # held-out (unseen-view) test
        te = test_samples[:args.eval_n]
        te_preds, te_golds = run_model(str(args.out), te, max_frames=args.max_frames,
                                       img_size=args.img_size)
        te_m = score(te_preds, te_golds)
        # in-distribution (seen-view) reference, same size
        tr_ref = train_samples[:args.eval_n]
        tr_preds, tr_golds = run_model(str(args.out), tr_ref, max_frames=args.max_frames,
                                       img_size=args.img_size)
        tr_m = score(tr_preds, tr_golds)
        report = {
            "seen_view": tr_m.__dict__,
            "held_out_view": te_m.__dict__,
            "cross_view_gap_accuracy": cross_view_gap(tr_m, te_m),
        }
        (args.out / "eval.json").write_text(_json.dumps(report, indent=2, default=str))
        print("SEEN-view   :", tr_m)
        print("HELDOUT-view:", te_m)
        print("cross-view accuracy gap:", report["cross_view_gap_accuracy"])


if __name__ == "__main__":
    main()
