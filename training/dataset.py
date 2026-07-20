"""Load synth shards (samples.jsonl) into chat-formatted VLM training samples.

Consumes the schema emitted by scripts/bootstrap_dataset.py (and, later, the Blender
render writer): each record has frames[], prompt, rationale, answer, class, split_key.

Produces messages in the SmolVLM2 / HF chat format:
  user: [image, image, ..., text prompt]
  assistant: rationale + JSON answer   (rationale trains the reasoning; JSON is the output)

Pure-python + PIL; no GPU needed. `load_records` / `to_chat` are unit-testable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class Sample:
    id: str
    frames: list[str]
    prompt: str
    rationale: str
    answer: dict
    label: str
    split_key: str
    lighting: str = "day"


# Label schemes. The 5-class "fine" set proved unlearnable at current fidelity (falls,
# faints and lying-immobile look identical in end-frames); "down3" collapses the three
# on-the-floor classes into one actionable "down" alert, which the model CAN learn
# (person-down recall was already 0.73). distress/normal kept.
LABEL_COLLAPSE = {
    "down3": {"fall": "down", "faint-collapse": "down", "lying-immobile": "down",
              "distress": "distress", "normal": "normal"},
    "binary": {"fall": "danger", "faint-collapse": "danger", "lying-immobile": "danger",
               "distress": "danger", "normal": "normal"},
}


def collapse_label(label: str, scheme: str | None) -> str:
    if not scheme or scheme == "fine":
        return label
    return LABEL_COLLAPSE[scheme].get(label, label)


def load_records(samples_jsonl: Path, label_set: str | None = None) -> list[Sample]:
    out = []
    for line in Path(samples_jsonl).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        label = collapse_label(r["class"], label_set)
        answer = dict(r["answer"])
        answer["status"] = collapse_label(answer.get("status", r["class"]), label_set)
        out.append(Sample(id=r["id"], frames=r["frames"], prompt=r["prompt"],
                          rationale=r["rationale"], answer=answer, label=label,
                          split_key=r.get("split_key", "unknown"),
                          lighting=r.get("lighting", "day")))
    return out


def domain_augment(frames: list, rng) -> list:
    """Simulate real-video domain shift so small models generalize to in-the-wild footage
    instead of memorizing the clean synthetic look. Applied per-STRIP with consistent
    params (a clip has one lighting/compression), except noise which varies per frame.
    Targets the exact gap that tanked small-model in-the-wild recall (0.13/0.31)."""
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter
    import cv2
    # per-strip params
    bright = rng.uniform(0.6, 1.5)
    contrast = rng.uniform(0.6, 1.5)
    sat = rng.uniform(0.5, 1.4)
    do_gray = rng.random() < 0.12
    do_blur = rng.random() < 0.4
    blur_r = rng.uniform(0.4, 1.6)
    jpeg_q = int(rng.integers(25, 75)) if rng.random() < 0.6 else None
    down = rng.uniform(0.4, 1.0) if rng.random() < 0.5 else 1.0   # low-res web video
    noise_sd = rng.uniform(0, 14)
    # per-channel white-balance cast (warm/cool/greenish CCTV-style color response) --
    # distinct from saturation: this is a DIRECTIONAL shift, not a colorfulness scale.
    do_tint = rng.random() < 0.4
    ch_gain = rng.uniform(0.82, 1.22, size=3) if do_tint else np.ones(3)
    out = []
    for im in frames:
        im = ImageEnhance.Brightness(im).enhance(bright)
        im = ImageEnhance.Contrast(im).enhance(contrast)
        im = ImageEnhance.Color(im).enhance(sat)
        if do_gray:
            im = im.convert("L").convert("RGB")
        if do_blur:
            im = im.filter(ImageFilter.GaussianBlur(blur_r))
        w, h = im.size
        if down < 1.0:
            im = im.resize((max(8, int(w * down)), max(8, int(h * down)))).resize((w, h))
        a = np.asarray(im).astype(np.float32)
        if do_tint:
            a = a * ch_gain[np.newaxis, np.newaxis, :]
        if noise_sd > 0:
            a = a + rng.normal(0, noise_sd, a.shape)
        a = np.clip(a, 0, 255).astype(np.uint8)
        if jpeg_q is not None:
            ok, enc = cv2.imencode(".jpg", cv2.cvtColor(a, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
            if ok:
                a = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        out.append(Image.fromarray(a))
    return out


def temporal_augment(frames: list, rng, min_n=4, max_n=8, jitter=0.35,
                     drop_p=0.15) -> list:
    """Make training fps-robust by randomizing how the strip is temporally sampled.

    A strip captured at one cadence is re-sampled to a random frame count with jittered
    (non-uniform) spacing and occasional dropped frames, so the model learns to read the
    fall from body-STATE change rather than a fixed frame rhythm. This is the training
    counterpart to StripBuffer's time-based sampling on the device.
    """
    import numpy as np
    if len(frames) <= 2:
        return frames
    n = int(rng.integers(min_n, max_n + 1))
    # jittered positions across the clip, then dedup/sort
    base = np.linspace(0, len(frames) - 1, n)
    noise = (rng.random(n) - 0.5) * 2 * jitter * (len(frames) / max(n, 1))
    pos = np.clip(np.round(base + noise), 0, len(frames) - 1).astype(int)
    pos = sorted(set(pos.tolist()))
    kept = [frames[i] for i in pos if rng.random() > drop_p]
    if len(kept) < 2:                       # a temporal strip needs >= 2 frames
        kept = [frames[0], frames[-1]]
    return kept


def target_text(s: Sample, with_rationale: bool = True) -> str:
    """ANSWER-FIRST format: the JSON verdict is emitted immediately so it is always
    within the on-device token budget and always parseable; the rationale (optional)
    follows for interpretability. A 500M model can't reliably reason-then-answer, and
    CoT-before-answer is expensive at inference — answer-first is the deployment-correct
    choice (diagnosed from the first bootstrap run: models stalled in rationale and never
    reached the JSON)."""
    ans = json.dumps(s.answer, separators=(",", ":"))
    if with_rationale:
        return f"{ans}\nWHY: {s.rationale}"
    return ans


def to_chat(s: Sample, with_rationale: bool = True) -> dict:
    """One HF chat sample. Images referenced by path; the collator loads them."""
    content = [{"type": "image", "path": f} for f in s.frames]
    content.append({"type": "text", "text": s.prompt})
    return {
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": target_text(s, with_rationale)}]},
        ],
        "label": s.label,
        "split_key": s.split_key,
    }


def load_images(s: Sample) -> list[Image.Image]:
    return [Image.open(f).convert("RGB") for f in s.frames]


# ---- cross-view / cross-subject split (eval honesty, per RESEARCH_PLAN) ----
def cross_view_split(samples: list[Sample], holdout_views: set[str]) -> tuple[list, list]:
    """Hold out ENTIRE camera archetypes for test (the honest perspective metric)."""
    train = [s for s in samples if s.split_key not in holdout_views]
    test = [s for s in samples if s.split_key in holdout_views]
    return train, test


def cross_subject_split(samples: list[Sample], holdout_frac=0.2, seed=0) -> tuple[list, list]:
    """Hold out entire motions (by motion_id prefix) — no clip leakage across split."""
    import random
    motions = sorted({s.id.rsplit("_cam", 1)[0] for s in samples})
    rnd = random.Random(seed)
    rnd.shuffle(motions)
    n_test = max(1, int(len(motions) * holdout_frac))
    test_m = set(motions[:n_test])
    train = [s for s in samples if s.id.rsplit("_cam", 1)[0] not in test_m]
    test = [s for s in samples if s.id.rsplit("_cam", 1)[0] in test_m]
    return train, test


if __name__ == "__main__":
    import sys
    p = Path(sys.argv[1] if len(sys.argv) > 1 else "data/bootstrap/shards/samples.jsonl")
    samples = load_records(p)
    print(f"loaded {len(samples)} samples; labels:",
          {l: sum(s.label == l for s in samples) for l in {s.label for s in samples}})
    chat = to_chat(samples[0])
    n_img = sum(c["type"] == "image" for c in chat["messages"][0]["content"])
    print(f"chat sample: {n_img} images + prompt; target head:",
          chat["messages"][1]["content"][0]["text"][:80].replace("\n", " "))
    tr, te = cross_view_split(samples, {"ceiling"})
    print(f"cross-view holdout=ceiling: train {len(tr)} / test {len(te)}")
    trs, tes = cross_subject_split(samples)
    assert not ({s.id.rsplit('_cam',1)[0] for s in trs} & {s.id.rsplit('_cam',1)[0] for s in tes}), "leak!"
    print(f"cross-subject: train {len(trs)} / test {len(tes)} (no motion leakage)")
    # fps-robustness augmentation: varied frame counts across draws
    import numpy as np
    rng = np.random.default_rng(0)
    counts = {len(temporal_augment(list(range(8)), rng)) for _ in range(50)}
    print(f"temporal_augment yields varied frame counts: {sorted(counts)}")
    assert len(counts) > 1, "augment should vary frame count"
    print("dataset OK")
