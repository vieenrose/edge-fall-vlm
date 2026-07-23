"""Validate a trained VLM on REAL video clips (sim-to-real). Runs on gpu0.

Takes a manifest of real clips (frame dirs + true label + optional split) and runs the
model exactly like the synthetic eval, then reports recall / specificity / confusion using
the same scheme-agnostic metrics. Maps arbitrary source labels to our schemes via
`--label-set`. Frames per clip are subsampled to a 6-frame strip spanning the clip.

    CUDA_VISIBLE_DEVICES=0 python scripts/validate_real.py --model runs/sft-2b-scale \
        --manifest data/real/urfd/manifest.json --label-set down3 --n 400
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.dataset import Sample, collapse_label
from training.eval import run_model, score

# map common source-dataset labels -> our fine 5-class, then collapse via label_set
SOURCE_MAP = {
    "fall": "fall", "falling": "fall", "fallen": "fall", "adl": "normal",
    "not_fall": "normal", "no_fall": "normal", "normal": "normal", "walk": "normal",
    "sit": "normal", "lie": "lying-immobile", "lying": "lying-immobile",
    "faint": "faint-collapse", "distress": "distress", "person": "normal",
}


def load_manifest_as_samples(manifest_path: Path, label_set: str) -> list[Sample]:
    rows = json.loads(Path(manifest_path).read_text())
    out = []
    for r in rows:
        frames = sorted(str(p) for p in Path(r["frames_dir"]).glob("*.png"))
        if not frames:
            frames = sorted(str(p) for p in Path(r["frames_dir"]).glob("*.jpg"))
        if len(frames) < 2:
            continue
        fine = SOURCE_MAP.get(r["label"], r["label"])
        lbl = collapse_label(fine, label_set)
        out.append(Sample(id=r["id"], frames=frames, prompt=r.get("prompt", PROMPT),
                          rationale="", answer={"status": lbl}, label=lbl,
                          split_key=r.get("split", "real"), lighting="day"))
    return out


PROMPT = ("You are a safety monitor. These are consecutive video frames (oldest first), "
          "possibly with more than one person. Report whether ANYONE has fallen, fainted, "
          "is lying immobile, or is in distress; else normal. Answer with JSON only.")


def extract_zip(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--label-set", default="down3")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--max-frames", type=int, default=6)
    ap.add_argument("--img-size", type=int, default=384)
    args = ap.parse_args()

    samples = load_manifest_as_samples(args.manifest, args.label_set)[:args.n]
    from collections import Counter
    print(f"real clips: {len(samples)}  labels={dict(Counter(s.label for s in samples))}")
    preds, golds, records = run_model(args.model, samples, max_frames=args.max_frames,
                                      img_size=args.img_size, return_records=True)
    m = score(preds, golds)
    print(f"\n=== REAL validation ({args.manifest.parent.name}) ===")
    print(f"acc={m.accuracy} binary_danger_sens={m.binary_sensitivity} "
          f"spec={m.binary_specificity} PERSON-DOWN recall={m.person_down_recall}")
    print("per-class recall:", m.per_class_recall)
    print("confusion:\n" + m.confusion_str())
    Path(args.model, "eval_real.json").write_text(json.dumps(
        {"manifest": str(args.manifest), "metrics": m.__dict__,
         "parse_failures": sum(1 for r in records if r["parse"] != "json"),
         "predictions": records}, indent=2, default=str))


if __name__ == "__main__":
    main()
