"""Pose-geometry features for false-alarm suppression (Approach B).

Runs YOLO-pose on a clip's frames and computes geometric features that distinguish a genuine
floor collapse (horizontal body AT floor level) from horizontal-but-not-fallen bodies
(gymnast inverted mid-air, cyclist leaning, person on a bed). The VLM confuses these because
it keys on body-axis-horizontal; pose geometry adds the cues it lacks: inversion, limb
support, and vertical placement.

Features per clip (aggregated over frames, using the largest/most-confident person):
  torso_angle_deg : angle of shoulder->hip axis from vertical (90 = fully horizontal)
  head_below_hip  : fraction of frames where head (nose) is BELOW hips in image-y (inverted)
  ankle_above_hip : fraction of frames where ankles are ABOVE hips (feet up: handstand/vault)
  vert_center     : median vertical position of the pose centroid in the frame (0=top,1=bottom)
  bbox_aspect     : median width/height of the keypoint bbox (>1 = horizontal spread)
  n_people_med    : median detected people
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        _MODEL = YOLO("yolo11n-pose.pt")
    return _MODEL


# COCO-17 indices
NOSE, L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK = 0, 5, 6, 11, 12, 15, 16


def clip_features(frame_paths: list[str]) -> dict:
    m = _model()
    torso, headlow, anklehigh, vcenter, aspect, npeople = [], [], [], [], [], []
    for fp in frame_paths:
        res = m(fp, verbose=False)[0]
        if res.keypoints is None or len(res.keypoints.data) == 0:
            npeople.append(0)
            continue
        kps = res.keypoints.data.cpu().numpy()   # (n_people, 17, 3): x,y,conf
        npeople.append(len(kps))
        # pick the person with the largest keypoint spread (closest/most prominent)
        spans = [(_valid_span(k), i) for i, k in enumerate(kps)]
        k = kps[max(spans)[1]]
        conf = k[:, 2]
        def pt(i):
            return k[i, :2] if conf[i] > 0.3 else None
        sh = _mid(pt(L_SH), pt(R_SH)); hip = _mid(pt(L_HIP), pt(R_HIP))
        nose = pt(NOSE); ank = _mid(pt(L_ANK), pt(R_ANK))
        if sh is not None and hip is not None:
            v = sh - hip
            ang = np.degrees(np.arctan2(abs(v[0]), abs(v[1]) + 1e-6))  # 0 vertical, 90 horiz
            torso.append(ang)
        if nose is not None and hip is not None:
            headlow.append(1.0 if nose[1] > hip[1] else 0.0)   # image-y grows downward
        if ank is not None and hip is not None:
            anklehigh.append(1.0 if ank[1] < hip[1] else 0.0)
        valid = k[conf > 0.3, :2]
        if len(valid):
            h = res.orig_shape[0]; w = res.orig_shape[1]
            vcenter.append(float(np.median(valid[:, 1]) / h))
            bw = np.ptp(valid[:, 0]); bh = np.ptp(valid[:, 1]) + 1e-6
            aspect.append(bw / bh)
    def med(x, d=np.nan): return float(np.median(x)) if x else d
    return {
        "torso_angle_deg": med(torso), "head_below_hip": med(headlow, 0.0),
        "ankle_above_hip": med(anklehigh, 0.0), "vert_center": med(vcenter),
        "bbox_aspect": med(aspect), "n_people_med": med(npeople, 0),
    }


def _valid_span(k):
    v = k[k[:, 2] > 0.3, :2]
    return 0.0 if len(v) < 2 else float(np.ptp(v[:, 0]) + np.ptp(v[:, 1]))


def _mid(a, b):
    pts = [p for p in (a, b) if p is not None]
    return np.mean(pts, axis=0) if pts else None


if __name__ == "__main__":
    import os
    for d in sys.argv[1:]:
        frs = sorted(str(p) for p in Path(d).glob("*.png"))
        if not frs:
            frs = sorted(str(p) for p in Path(d).glob("*.jpg"))
        f = clip_features(frs)
        print(f"{os.path.basename(d):16} " + "  ".join(f"{k}={v:.2f}" for k, v in f.items()))
