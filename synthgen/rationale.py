"""Turn ground-truth 3D pose into a view-canonical rationale + the short JSON answer.

Because the renderer gives us exact 3D joints in a gravity-aligned world frame, the
kinematic description of a given motion is the SAME regardless of which camera saw it.
That is precisely what trains viewpoint invariance: the same fall gets the same
chain-of-thought target across all K cameras.

Pure numpy — unit-testable outside Blender. Joint indexing here uses a small named
subset; map your SMPL-X joint indices to these names in bodies.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LabelClass

# minimal joint set we need for fall kinematics (world coords, +Z up)
JOINTS = ("head", "neck", "pelvis", "l_hip", "r_hip", "l_ankle", "r_ankle", "l_shoulder", "r_shoulder")


@dataclass
class KinFeatures:
    torso_vertical_deg: np.ndarray   # per-frame angle of torso from vertical (0=upright,90=horizontal)
    head_height: np.ndarray          # per-frame head z
    hip_height: np.ndarray           # per-frame pelvis z
    com_z: np.ndarray                # per-frame approx COM z
    com_vz: np.ndarray               # vertical velocity of COM (m/s)
    fps: int


def compute_features(joints: dict[str, np.ndarray], fps: int) -> KinFeatures:
    """joints[name] -> (T,3) world positions. Returns per-frame kinematic features."""
    pelvis = joints["pelvis"]
    neck = joints["neck"]
    head = joints["head"]
    torso = neck - pelvis
    torso_n = torso / (np.linalg.norm(torso, axis=1, keepdims=True) + 1e-9)
    vertical = np.array([0.0, 0.0, 1.0])
    cos_up = np.clip(torso_n @ vertical, -1, 1)
    torso_vertical_deg = np.degrees(np.arccos(cos_up))   # 0 upright -> 90 horizontal

    hip_h = pelvis[:, 2]
    head_h = head[:, 2]
    # crude COM proxy: mean of pelvis+shoulders+head
    com = np.mean([joints["pelvis"], joints["l_shoulder"], joints["r_shoulder"], joints["head"]], axis=0)
    com_z = com[:, 2]
    com_vz = np.gradient(com_z) * fps
    return KinFeatures(torso_vertical_deg, head_h, hip_h, com_z, com_vz, fps)


@dataclass
class Events:
    impact_frame: int | None      # frame of peak downward deceleration
    immobile_from: int | None     # first frame of sustained stillness after impact
    min_com_vz: float             # most negative vertical velocity (m/s)


def detect_events(f: KinFeatures, still_thresh_mps=0.15, still_window=8) -> Events:
    impact = None
    if len(f.com_vz) > 2:
        # impact = frame after the strongest downward velocity where it arrests
        i_min = int(np.argmin(f.com_vz))
        # find where velocity returns toward ~0 after i_min
        after = np.where(np.abs(f.com_vz[i_min:]) < still_thresh_mps)[0]
        impact = int(i_min + after[0]) if len(after) else i_min
    immobile_from = None
    speed = np.abs(f.com_vz)
    for t in range(0, len(speed) - still_window):
        if np.all(speed[t:t + still_window] < still_thresh_mps) and (impact is None or t >= impact):
            immobile_from = t
            break
    return Events(impact, immobile_from, float(f.com_vz.min()) if len(f.com_vz) else 0.0)


def classify(f: KinFeatures, ev: Events, intended: LabelClass | None = None) -> LabelClass:
    """Kinematic auto-label. `intended` (the motion's design class) is used as a prior
    and for the consistency gate in quality.py; here we return the kinematic verdict."""
    tail = min(8, len(f.torso_vertical_deg))
    horizontal_end = f.torso_vertical_deg[-tail:].mean() > 60
    low_end = f.hip_height[-1] < 0.5           # pelvis near floor
    fast_descent = ev.min_com_vz < -1.0        # strong downward velocity
    ended_still = ev.immobile_from is not None
    # oscillating COM velocity in the tail = clonic jerking, not stillness -- the kinematic
    # signature distinguishing a seizure's convulsing end-state from calm immobility.
    tail_jitter = float(np.std(f.com_vz[-tail:])) if len(f.com_vz) >= tail else 0.0

    if horizontal_end and low_end and intended == LabelClass.SEIZURE and tail_jitter > 0.3:
        return LabelClass.SEIZURE
    if horizontal_end and low_end and ended_still:
        if fast_descent:
            return LabelClass.FALL
        return LabelClass.FAINT if intended == LabelClass.FAINT else LabelClass.IMMOBILE
    if horizontal_end and low_end:
        return LabelClass.IMMOBILE
    if intended == LabelClass.DISTRESS:
        return LabelClass.DISTRESS
    return LabelClass.NORMAL


def build_rationale(f: KinFeatures, ev: Events, label: LabelClass) -> str:
    parts = []
    t0v, t1v = f.torso_vertical_deg[0], f.torso_vertical_deg[-1]
    parts.append(f"Torso angle from vertical goes {t0v:.0f} deg -> {t1v:.0f} deg")
    dh = f.head_height[0] - f.head_height[-1]
    parts.append(f"head descends {dh:.2f} m (to {'below' if f.head_height[-1] < f.hip_height[-1] else 'above'} hip height)")
    if ev.impact_frame is not None:
        parts.append(f"vertical velocity peaks at {ev.min_com_vz:.1f} m/s then arrests at frame {ev.impact_frame} (impact)")
    if ev.immobile_from is not None:
        parts.append(f"body remains still from frame {ev.immobile_from}")
    verdict = {
        LabelClass.FALL: "-> fall followed by immobility",
        LabelClass.FAINT: "-> collapse without protective reaction (faint)",
        LabelClass.IMMOBILE: "-> person lying immobile",
        LabelClass.DISTRESS: "-> distress / struggling posture",
        LabelClass.SEIZURE: "-> rigid fall followed by convulsive jerking (seizure)",
        LabelClass.NORMAL: "-> normal activity, no danger",
    }[label]
    return "; ".join(parts) + " " + verdict


def posture(f: KinFeatures) -> str:
    """Coarse body posture from GT kinematics (end of clip) — the discriminative cue the
    VLM struggles to extract on its own. Supervising this as an explicit output field
    (pose-assist, model still ships alone) forces attention to TORSO ORIENTATION, which is
    exactly what separates 'down' (horizontal on floor) from 'low-but-normal'
    (crouch/sit/kneel = upright torso, low)."""
    tail = min(6, len(f.torso_vertical_deg))
    torso = float(f.torso_vertical_deg[-tail:].mean())   # 0=upright, 90=horizontal
    hip = float(f.hip_height[-tail:].mean())
    if torso >= 55 and hip < 0.55:
        return "horizontal-on-floor"        # fallen / lying  -> DOWN
    if torso < 50 and hip < 0.7:
        return "upright-low"                # crouch/sit/kneel -> NORMAL
    if torso >= 45:
        return "leaning"
    return "upright-standing"


def build_answer(label: LabelClass, f: KinFeatures, ev: Events) -> dict:
    """The short JSON the deployed VLM must emit."""
    danger = label in (LabelClass.FALL, LabelClass.FAINT, LabelClass.IMMOBILE,
                       LabelClass.DISTRESS, LabelClass.SEIZURE)
    person_down = bool(f.hip_height[-1] < 0.5 and f.torso_vertical_deg[-1] > 55)
    # confidence proxy from signal strength (real training uses this as a soft target)
    conf = 0.6
    if label == LabelClass.FALL and ev.min_com_vz < -1.5 and ev.immobile_from is not None:
        conf = 0.95
    elif danger and person_down:
        conf = 0.85
    return {"posture": posture(f), "status": label.value,
            "confidence": round(conf, 2), "person_down": person_down}


if __name__ == "__main__":
    # synthetic fall: upright -> horizontal, head drops, fast descent, then still
    T, fps = 90, 30
    joints = {}
    z_pelvis = np.concatenate([np.full(40, 1.0), np.linspace(1.0, 0.2, 8), np.full(42, 0.2)])
    z_head = np.concatenate([np.full(40, 1.6), np.linspace(1.6, 0.15, 8), np.full(42, 0.15)])
    for name in JOINTS:
        joints[name] = np.zeros((T, 3))
    joints["pelvis"][:, 2] = z_pelvis
    joints["head"][:, 2] = z_head
    joints["neck"][:, 2] = np.concatenate([np.full(40, 1.4), np.linspace(1.4, 0.2, 8), np.full(42, 0.2)])
    joints["neck"][:, 0] = np.concatenate([np.zeros(40), np.linspace(0, 0.6, 8), np.full(42, 0.6)])  # tips over in X
    joints["l_shoulder"][:, 2] = joints["neck"][:, 2]
    joints["r_shoulder"][:, 2] = joints["neck"][:, 2]
    f = compute_features(joints, fps)
    ev = detect_events(f)
    label = classify(f, ev, intended=LabelClass.FALL)
    print("label:", label.value)
    print("rationale:", build_rationale(f, ev, label))
    print("answer:", build_answer(label, f, ev))
    assert label == LabelClass.FALL, label
    print("rationale OK")
