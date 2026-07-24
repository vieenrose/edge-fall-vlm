"""Fine-tune a small image CLASSIFIER on frame-mosaic images (down3: down/distress/normal).

This is the WebGPU-deployable path: unlike VLMs (which no released optimum can export to
ONNX), image classifiers export cleanly via optimum and run live in-browser through
transformers.js's image-classification pipeline. Input = one 3x2 mosaic of the 6-frame strip
(compose_mosaic), so temporal info is preserved in a single image.

    CUDA_VISIBLE_DEVICES=0 python3 training/sft_mosaic_clf.py \
        --train data/mosaic/train --base WinKawaks/vit-small-patch16-224 --out runs/mosaic-vit-small
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True, help="imagefolder dir (class subdirs)")
    ap.add_argument("--base", default="WinKawaks/vit-small-patch16-224")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=6.0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    args = ap.parse_args()

    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import (AutoImageProcessor, AutoModelForImageClassification,
                              Trainer, TrainingArguments)

    ds = load_dataset("imagefolder", data_dir=str(args.train))["train"]
    labels = sorted(ds.features["label"].names)
    id2label = {i: n for i, n in enumerate(ds.features["label"].names)}
    label2id = {n: i for i, n in id2label.items()}
    print("classes:", id2label)
    from collections import Counter
    print("counts:", Counter(ds["label"]))

    proc = AutoImageProcessor.from_pretrained(args.base)
    # non-square mosaic -> processor resizes to the model's square input (some h-squish, ok)
    size = proc.size.get("height", proc.size.get("shortest_edge", 224))
    mean, std = proc.image_mean, proc.image_std

    def transform(batch):
        import torchvision.transforms as T
        aug = T.Compose([T.Resize((size, size)),
                         T.RandomHorizontalFlip(),
                         T.ColorJitter(0.2, 0.2, 0.2),
                         T.ToTensor(), T.Normalize(mean, std)])
        batch["pixel_values"] = [aug(im.convert("RGB")) for im in batch["image"]]
        return batch

    ds.set_transform(transform)

    model = AutoModelForImageClassification.from_pretrained(
        args.base, num_labels=len(id2label), id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True)

    # class-weighted loss (distress is rare) via a custom Trainer
    cnt = Counter(ds["label"])
    w = torch.tensor([1.0 / max(1, cnt[i]) for i in range(len(id2label))])
    w = (w / w.sum() * len(id2label)).float()

    class WTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.functional.cross_entropy(out.logits, labels, weight=w.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    def collate(ex):
        return {"pixel_values": torch.stack([e["pixel_values"] for e in ex]),
                "labels": torch.tensor([e["label"] for e in ex])}

    targs = TrainingArguments(
        output_dir=str(args.out), per_device_train_batch_size=args.bs,
        num_train_epochs=args.epochs, learning_rate=args.lr, bf16=True,
        logging_steps=20, save_strategy="no", remove_unused_columns=False, report_to=[])
    trainer = WTrainer(model=model, args=targs, data_collator=collate, train_dataset=ds)
    trainer.train()
    trainer.save_model(str(args.out))
    proc.save_pretrained(str(args.out))
    print("mosaic classifier saved ->", args.out)


if __name__ == "__main__":
    main()
