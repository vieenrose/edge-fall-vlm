"""Measure a pose-geometry VETO layered on the VLM (Approach B).

For every oops_val clip: take the VLM's per-clip prediction (from a bench report), compute
pose-geometry features, and test veto rules that flip a VLM "down" -> "normal" only when the
geometry confidently says the body is NOT in a floor collapse (inverted / feet-up). Reports
false-alarm reduction vs recall cost for each rule, so we can see how much of the false-alarm
problem pose geometry can safely remove.

    CUDA_VISIBLE_DEVICES=0 python3 scripts/pose_veto_eval.py bench_val150_hardneg.json hardneg
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.pose_geometry import clip_features
from training.eval import DANGER

REPORT, KEY = sys.argv[1], sys.argv[2]
MANIFEST = "data/real/oops/oops_val.json"

preds = {p["id"]: p for p in json.load(open(REPORT))[KEY]["predictions"]}
rows = json.loads(Path(MANIFEST).read_text())
byid = {r["id"]: r for r in rows}

feats = {}
for cid in preds:
    d = byid[cid]["frames_dir"]
    frs = sorted(str(p) for p in Path(d).glob("*.png"))
    feats[cid] = clip_features(frs) if frs else {}

def is_down(cid): return preds[cid]["pred"] in DANGER
def gold_down(cid): return preds[cid]["gold"] in DANGER

# candidate veto predicates (flip down->normal when true)
def inverted(f): return f.get("ankle_above_hip", 0) >= 0.5 or f.get("head_below_hip", 0) >= 0.5
def feet_up(f):  return f.get("ankle_above_hip", 0) >= 0.5
def upright(f):  return (f.get("torso_angle_deg") or 90) < 20   # NOTE: unsafe (falls detect upright too)

rules = {"inverted(feet-up OR head-down)": inverted, "feet-up only": feet_up,
         "upright-torso (unsafe ref)": upright}

pos = [c for c in preds if gold_down(c)]; neg = [c for c in preds if not gold_down(c)]
base_fa = sum(is_down(c) for c in neg); base_miss = sum(not is_down(c) for c in pos)
print(f"baseline (VLM={KEY}): FA {base_fa}/{len(neg)}, recall {(len(pos)-base_miss)/len(pos):.3f}")
print()
for name, pred in rules.items():
    # veto only applies to clips the VLM called down
    fa = sum(1 for c in neg if is_down(c) and not pred(feats[c]))
    miss = sum(1 for c in pos if (not is_down(c)) or pred(feats[c]))  # missed if VLM missed OR veto flipped a true fall
    vetoed_fa = sum(1 for c in neg if is_down(c) and pred(feats[c]))
    vetoed_tp = sum(1 for c in pos if is_down(c) and pred(feats[c]))
    print(f"{name:34} FA {base_fa}->{fa} (removed {vetoed_fa})  recall {(len(pos)-base_miss)/len(pos):.3f}->{(len(pos)-miss)/len(pos):.3f} (falls wrongly vetoed {vetoed_tp})")
