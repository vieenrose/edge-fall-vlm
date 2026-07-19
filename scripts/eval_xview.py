"""Cross-view evaluation on a trained model with a STRATIFIED, shuffled sample.

Fixes the naive first-N slicing (which, because samples.jsonl is written class-by-class,
drew only danger samples and left specificity undefined). Here we shuffle and take a
class-balanced subset that includes normals, so sensitivity AND specificity are real.

    CUDA_VISIBLE_DEVICES=0 python scripts/eval_xview.py --model runs/sft-xview \
        --samples data/bootstrap/shards/samples.jsonl --holdout ceiling,low_shelf --n 200
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.dataset import cross_view_split, load_records
from training.eval import cross_view_gap, run_model, score


def stratified(samples, n, seed=0):
    by = defaultdict(list)
    for s in samples:
        by[s.label].append(s)
    rnd = random.Random(seed)
    for v in by.values():
        rnd.shuffle(v)
    out, i = [], 0
    labels = list(by)
    while len(out) < n and any(by.values()):
        lab = labels[i % len(labels)]
        if by[lab]:
            out.append(by[lab].pop())
        i += 1
    rnd.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--holdout", default="ceiling,low_shelf")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--label-set", default="fine")
    ap.add_argument("--max-frames", type=int, default=6)
    ap.add_argument("--img-size", type=int, default=384)
    args = ap.parse_args()

    holdout = {v.strip() for v in args.holdout.split(",") if v.strip()}
    samples = load_records(args.samples, label_set=args.label_set)
    train_s, test_s = cross_view_split(samples, holdout)

    seen = stratified(train_s, args.n)
    held = stratified(test_s, args.n)
    print(f"seen-view eval n={len(seen)} labels={_dist(seen)}")
    print(f"held-out-view eval n={len(held)} labels={_dist(held)}")

    sp, sg = run_model(args.model, seen, max_frames=args.max_frames, img_size=args.img_size)
    hp, hg = run_model(args.model, held, max_frames=args.max_frames, img_size=args.img_size)
    sm, hm = score(sp, sg), score(hp, hg)
    report = {"holdout_views": sorted(holdout),
              "seen_view": sm.__dict__, "held_out_view": hm.__dict__,
              "cross_view_gap_accuracy": cross_view_gap(sm, hm),
              "seen_false_alarms_per_day@1Hz_motiongated0.1": sm.false_alarms_per_day(360),
              "held_false_alarms_per_day@1Hz_motiongated0.1": hm.false_alarms_per_day(360)}
    out = Path(args.model) / "eval_xview.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    for tag, m in (("SEEN-view", sm), ("HELDOUT-view", hm)):
        print(f"\n=== {tag} ===")
        print(f"acc={m.accuracy} binary_danger_sens={m.binary_sensitivity} "
              f"spec={m.binary_specificity} PERSON-DOWN recall={m.person_down_recall}")
        print("per-class recall:", m.per_class_recall)
        print("confusion (gold rows -> pred cols):")
        print(m.confusion_str())
    print("\ncross-view accuracy gap:", report["cross_view_gap_accuracy"])
    print("wrote", out)


def _dist(samples):
    d = defaultdict(int)
    for s in samples:
        d[s.label] += 1
    return dict(d)


if __name__ == "__main__":
    main()
