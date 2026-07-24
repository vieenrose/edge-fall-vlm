"""Evaluate the mosaic image-classifier on an OOPS mosaic eval set (down3 binary danger)."""
import argparse, json
from pathlib import Path
from collections import Counter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DANGER = {"down", "distress"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", type=Path, required=True, help="mosaic imagefolder (label subdirs)")
    args = ap.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    proc = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForImageClassification.from_pretrained(args.model).to("cuda").eval()
    id2label = model.config.id2label
    size = proc.size.get("height", proc.size.get("shortest_edge", 224))
    import torchvision.transforms as T
    tf = T.Compose([T.Resize((size, size)), T.ToTensor(),
                    T.Normalize(proc.image_mean, proc.image_std)])
    preds, golds = [], []
    for lbldir in sorted(Path(args.eval).iterdir()):
        gold = lbldir.name
        for img in lbldir.glob("*.jpg"):
            x = tf(Image.open(img).convert("RGB")).unsqueeze(0).to("cuda")
            with torch.no_grad():
                logit = model(pixel_values=x).logits
            preds.append(id2label[int(logit.argmax(-1))]); golds.append(gold)
    n = len(golds)
    pos = [i for i in range(n) if golds[i] in DANGER]; neg = [i for i in range(n) if golds[i] not in DANGER]
    tp = sum(1 for i in pos if preds[i] in DANGER); fa = sum(1 for i in neg if preds[i] in DANGER)
    acc = sum(1 for i in range(n) if (preds[i] in DANGER) == (golds[i] in DANGER)) / n
    print(f"{args.eval.name}: n={n}  acc={acc:.3f}  recall={tp}/{len(pos)}={tp/len(pos):.3f}  "
          f"FA={fa}/{len(neg)} (spec={(len(neg)-fa)/len(neg):.3f})")

if __name__ == "__main__":
    main()
