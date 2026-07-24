"""Extract real INDOOR ADL hard-negatives from GMDCSA24 (CC-BY-4.0, Zenodo 13354453).

GMDCSA24 is staged indoor home falls + ADL (10 subjects). Its ADL/non-fall segments (walking,
sitting, bending, lying on a bed/floor) are real indoor horizontal-but-normal hard-negatives,
complementary to OOPS's outdoor comedic falls and closer to the elder-care product domain.
Uses OmniFall's gmdcsa24 segment labels; extracts every NON-fall segment as NORMAL.

Separate dataset from the OOPS eval set -> no leakage. Label 1=fall / 2=fallen excluded.

    python3 scripts/build_gmdcsa_hardneg.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_oops import cut_strip

ROOT = Path("data/real/gmdcsa/extracted")
VIDROOT = next(ROOT.glob("ekramalam-*"))            # the single top dir
OUT = Path("data/real/gmdcsa_hardneg")
NAMES = {0:'walk',3:'sit_down',4:'sitting',5:'lie_down',6:'lying',7:'stand_up',8:'standing',
         9:'other',10:'kneel_down',11:'kneeling',12:'squat_down',13:'squatting',14:'crawl',15:'jump'}
FALL = {1, 2}
PROMPT = ("You are a safety monitor. These are consecutive video frames (oldest first), "
          "possibly with more than one person. Report whether ANYONE has fallen, fainted, "
          "is lying immobile, or is in distress; else normal. Answer with JSON only.")


def main():
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    rows = []
    for split in ['train', 'validation', 'test']:
        p = hf_hub_download('simplexsigil2/omnifall', f'parquet/gmdcsa24/{split}-00000-of-00001.parquet',
                            repo_type='dataset')
        rows += pq.read_table(p).to_pylist()

    out_rows, prov, counts = [], {}, {}
    for r in rows:
        if r['label'] in FALL:
            continue
        kind = NAMES.get(r['label'], 'other')
        # parquet path "Subject_1/Fall/01" -> "Subject 1/Fall/01.mp4"
        rel = r['path'].replace('_', ' ') + '.mp4'
        vpath = VIDROOT / rel
        if not vpath.exists():
            continue
        idx = counts.get(kind, 0)
        cid = f"gmdcsa_{kind}_{idx:04d}"
        clip_dir = OUT / "clips" / cid
        frames = cut_strip(vpath, float(r['start']), float(r['end']), clip_dir)
        if len(frames) < 4:
            import shutil; shutil.rmtree(clip_dir, ignore_errors=True); continue
        counts[kind] = idx + 1
        prov[cid] = {"video": rel, "start": float(r['start']), "end": float(r['end']), "kind": kind}
        posture = "non-upright-normal" if r['label'] in (5, 6, 14) else "upright-normal"
        out_rows.append({"id": cid, "class": "normal", "frames": frames, "prompt": PROMPT,
                         "rationale": f"Real indoor ADL clip ({kind}) — nobody has fallen -> normal.",
                         "answer": {"posture": posture, "status": "normal",
                                    "confidence": 0.8, "person_down": False},
                         "lighting": "day", "split_key": "gmdcsa_indoor"})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "hardneg_samples.jsonl", "w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")
    (OUT / "provenance.json").write_text(json.dumps({"counts": counts, "clips": prov}, indent=2))
    print(f"GMDCSA24 indoor hard-negatives: {len(out_rows)} by kind {counts} -> {OUT}")


if __name__ == "__main__":
    main()
