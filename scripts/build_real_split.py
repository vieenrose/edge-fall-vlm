"""Split URFD into a real TRAIN pool (negatives + falls for fine-tuning) and a held-out
real TEST set, with NO clip overlap. Converts train clips into our samples.jsonl schema so
they can be mixed with synthetic data for fine-tuning.

Split: test = fall-01..20 + adl-01..20 ; train = fall-21..30 + adl-21..40.

    python scripts/build_real_split.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("data/real/urfd")
STRIP = 8   # frames kept per real clip (evenly spaced); collator further subsamples

PROMPT = ("You are a safety monitor. These are consecutive video frames (oldest first), "
          "possibly with more than one person. Report whether ANYONE has fallen, fainted, "
          "is lying immobile, or is in distress; else normal. Answer with JSON only.")


def clip_num(name: str) -> int:
    # fall-07-cam1-rgb -> 7
    return int(name.split("-")[1])


def even_frames(frames_dir: Path, k: int) -> list[str]:
    import numpy as np
    fs = sorted(str(p) for p in frames_dir.glob("*.png"))
    if len(fs) <= k:
        return fs
    idx = np.linspace(0, len(fs) - 1, k).astype(int)
    return [fs[i] for i in idx]


def to_sample(entry: dict) -> dict:
    is_fall = entry["label"] == "fall"
    fine = "fall" if is_fall else "normal"
    posture = "horizontal-on-floor" if is_fall else "upright-standing"
    answer = {"posture": posture, "status": fine,
              "confidence": 0.9 if is_fall else 0.6,
              "person_down": is_fall, "n_people": 1}
    return {"id": "real_" + entry["id"], "class": fine,
            "frames": even_frames(Path(entry["frames_dir"]), STRIP),
            "prompt": PROMPT, "rationale": "", "answer": answer,
            "camera": {"archetype": "real", "intrinsics": {}},
            "lighting": "day", "split_key": entry.get("split", "real")}


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    train, test = [], []
    for e in manifest:
        n = clip_num(e["id"])
        (test if n <= 20 else train).append(e)

    # real train -> samples.jsonl (mix into synthetic for fine-tuning)
    with (ROOT / "real_train_samples.jsonl").open("w") as f:
        for e in train:
            s = to_sample(e)
            if len(s["frames"]) >= 2:
                f.write(json.dumps(s) + "\n")
    # real test -> validation manifest
    (ROOT / "real_test_manifest.json").write_text(json.dumps(test, indent=2))

    n_train_fall = sum(e["label"] == "fall" for e in train)
    n_test_fall = sum(e["label"] == "fall" for e in test)
    print(f"real TRAIN: {len(train)} clips ({n_train_fall} fall / {len(train)-n_train_fall} adl)"
          f" -> real_train_samples.jsonl")
    print(f"real TEST:  {len(test)} clips ({n_test_fall} fall / {len(test)-n_test_fall} adl)"
          f" -> real_test_manifest.json")


if __name__ == "__main__":
    main()
