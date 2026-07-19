"""Evaluation harness — the honest metrics from RESEARCH_PLAN.md.

Metrics:
  - per-class + binary (danger vs normal) sensitivity / specificity / F1
  - cross-view GAP: train-view accuracy minus held-out-view accuracy (the perspective
    robustness number that must be reported, not cross-subject alone)
  - false-alarms/day estimate from specificity + an assumed inference cadence

Two entry points:
  - score(preds, golds): pure metric computation (unit-testable)
  - run_model(model_dir, samples): generate predictions with a trained VLM (GPU)

Parsing tolerates the model's "rationale ... ANSWER: {json}" format and bare JSON.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

# scheme-agnostic: danger = anything not "normal"; DOWN = on-the-floor classes (incl the
# collapsed "down"/"danger" labels) — the core person-on-floor alert.
DANGER = {"fall", "faint-collapse", "lying-immobile", "distress", "down", "danger"}
DOWN = {"fall", "faint-collapse", "lying-immobile", "down", "danger"}
_JSON_RE = re.compile(r"\{[^{}]*\}")


CANON = ("fall", "faint-collapse", "lying-immobile", "distress", "normal")


def normalize_status(s: str) -> str:
    """Map model output variants (falling/fainting/lying immobile/laid-off/...) to the
    canonical label set. SmolVLM's language prior drifts toward natural words; this
    canonicalizes them so scoring reflects the intended class, not surface form."""
    s = (s or "").strip().lower()
    if s in CANON:
        return s
    if s == "down" or s == "danger":        # collapsed-scheme labels, keep as-is
        return s
    if "faint" in s or "collaps" in s:
        return "faint-collapse"
    if "distress" in s or "struggl" in s:
        return "distress"
    if "fall" in s:                        # "fall", "falling", "fallen"
        return "fall"
    if "immobil" in s or "lying" in s or "laid" in s or "on the floor" in s or "prone" in s:
        return "lying-immobile"
    if "normal" in s or "fine" in s or "ok" in s:
        return "normal"
    return "normal"


def parse_answer(text: str) -> dict:
    """Extract the JSON verdict from a model generation. Robust to extra prose."""
    m = None
    for m in _JSON_RE.finditer(text):
        pass  # take the LAST json object (after the rationale)
    if m:
        try:
            d = json.loads(m.group(0))
            d["status"] = normalize_status(d.get("status", ""))
            return d
        except json.JSONDecodeError:
            pass
    # fallback: keyword scan (already canonical via normalize_status)
    low = text.lower()
    for kw in ("faint", "distress", "fall", "immobile", "lying", "normal"):
        if kw in low:
            return {"status": normalize_status(kw), "confidence": 0.5,
                    "person_down": normalize_status(kw) in DOWN}
    return {"status": "normal", "confidence": 0.0, "person_down": False}


@dataclass
class Metrics:
    n: int
    binary_sensitivity: float   # danger recall
    binary_specificity: float   # normal recall
    binary_f1: float
    per_class_recall: dict
    accuracy: float
    confusion: dict = None      # gold -> {pred: count}
    person_down_recall: float = None   # of truly-down clips, how many flagged down

    def false_alarms_per_day(self, inferences_per_hour: float) -> float:
        fp_rate = 1.0 - self.binary_specificity
        return round(fp_rate * inferences_per_hour * 24, 2)

    def confusion_str(self) -> str:
        if not self.confusion:
            return ""
        classes = sorted({k for k in self.confusion} |
                         {p for v in self.confusion.values() for p in v})
        rows = ["gold\\pred  " + " ".join(f"{c[:6]:>6}" for c in classes)]
        for g in classes:
            row = self.confusion.get(g, {})
            rows.append(f"{g[:9]:>9}  " + " ".join(f"{row.get(c,0):>6}" for c in classes))
        return "\n".join(rows)


def score(preds: list[str], golds: list[str]) -> Metrics:
    assert len(preds) == len(golds) and preds
    n = len(golds)
    tp = fp = tn = fn = 0
    per_class = {}
    confusion = {}
    correct = 0
    down_tot = down_hit = 0
    for p, g in zip(preds, golds):
        per_class.setdefault(g, {"hit": 0, "tot": 0})
        per_class[g]["tot"] += 1
        confusion.setdefault(g, {})
        confusion[g][p] = confusion[g].get(p, 0) + 1
        if p == g:
            per_class[g]["hit"] += 1
            correct += 1
        p_danger, g_danger = p in DANGER, g in DANGER
        if g_danger and p_danger:
            tp += 1
        elif g_danger and not p_danger:
            fn += 1
        elif not g_danger and p_danger:
            fp += 1
        else:
            tn += 1
        # person-down: gold is a down-class, did we predict ANY down-class?
        if g in DOWN:
            down_tot += 1
            down_hit += int(p in DOWN)
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    pcr = {c: round(v["hit"] / v["tot"], 3) for c, v in per_class.items()}
    pdr = round(down_hit / down_tot, 3) if down_tot else None
    return Metrics(n, round(sens, 3), round(spec, 3), round(f1, 3), pcr,
                   round(correct / n, 3), confusion=confusion, person_down_recall=pdr)


def cross_view_gap(train_view_metrics: Metrics, holdout_view_metrics: Metrics) -> float:
    """Positive = performance drops on unseen viewpoints (the risk we track)."""
    return round(train_view_metrics.accuracy - holdout_view_metrics.accuracy, 3)


# ---- GPU generation path ----
def run_model(model_dir: str, samples, max_frames=6, img_size=384, max_new_tokens=96):
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from training.dataset import load_images
    from training.sft import _subsample

    proc = AutoProcessor.from_pretrained(model_dir, do_image_splitting=False,
                                         size={"longest_edge": img_size})
    model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.bfloat16).to("cuda").eval()
    preds, golds = [], []
    for s in samples:
        imgs = _subsample(load_images(s), max_frames)
        msgs = [{"role": "user", "content": [{"type": "image"} for _ in imgs] +
                 [{"type": "text", "text": s.prompt}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        batch = proc(text=[text], images=[imgs], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
        gen = proc.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        preds.append(parse_answer(gen).get("status", "normal"))
        golds.append(s.label)
    return preds, golds


if __name__ == "__main__":
    # unit test the metric math with synthetic predictions
    golds = ["fall", "fall", "normal", "normal", "distress", "normal", "fall", "normal"]
    preds = ["fall", "normal", "normal", "fall", "distress", "normal", "fall", "normal"]
    m = score(preds, golds)
    print("metrics:", m)
    print("false alarms/day @0.5Hz:", m.false_alarms_per_day(inferences_per_hour=1800))
    # parse robustness
    assert parse_answer("torso tips... ANSWER: {\"status\":\"fall\",\"confidence\":0.9}")["status"] == "fall"
    assert parse_answer("the person appears to have fainted")["status"] == "faint-collapse"
    assert parse_answer("{\"status\":\"normal\"}")["status"] == "normal"
    # cross-view gap
    train_m = score(["fall"] * 8, ["fall"] * 8)
    hold_m = score(["fall"] * 4 + ["normal"] * 4, ["fall"] * 8)
    print("cross-view gap:", cross_view_gap(train_m, hold_m))
    print("eval OK")
