"""Quality gates (SYNTHETIC_DATA_SPEC.md §6). Pure numpy; testable outside Blender.

A clip must pass ALL gates to enter the training set. We also accumulate a
distribution audit so the coverage guarantees (viewpoint/lens/lighting spread and
negatives-outnumber-falls) are enforced at the batch level, with dropped clips logged
rather than silently discarded.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .config import LabelClass
from .rationale import KinFeatures


@dataclass
class GateResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)


def physical_plausibility(f: KinFeatures, max_gravity_mps2=25.0, max_updraft_mps=2.0) -> GateResult:
    """Reject clips whose COM violates gravity bounds (ragdoll blow-ups)."""
    reasons = []
    if len(f.com_vz) > 2:
        acc = np.gradient(f.com_vz) * f.fps
        if np.nanmax(np.abs(acc)) > max_gravity_mps2 * 3:   # allow impact spikes, cap blow-ups
            reasons.append("implausible COM acceleration")
        if np.nanmax(f.com_vz) > max_updraft_mps:           # body shouldn't fly upward
            reasons.append("upward COM velocity (body launched)")
    if np.nanmin(f.hip_height) < -0.2:                      # sank through the floor
        reasons.append("subject below floor")
    return GateResult(len(reasons) == 0, reasons)


def visibility(subject_px_height: float, occlusion_frac: float,
               min_px=48, max_occ=0.6) -> GateResult:
    reasons = []
    if subject_px_height < min_px:
        reasons.append(f"subject too small ({subject_px_height:.0f}px)")
    if occlusion_frac > max_occ:
        reasons.append(f"over-occluded ({occlusion_frac:.2f})")
    return GateResult(len(reasons) == 0, reasons)


def label_consistency(kin_label: LabelClass, intended: LabelClass) -> GateResult:
    """Kinematic auto-label must be compatible with the motion's design intent.
    Falls/faints/immobile are mutually confusable and allowed to swap; a clip
    designed as NORMAL that reads as FALL (or vice-versa) is flagged."""
    down = {LabelClass.FALL, LabelClass.FAINT, LabelClass.IMMOBILE}
    if intended in down and kin_label in down:
        return GateResult(True)
    if intended == kin_label:
        return GateResult(True)
    if intended == LabelClass.DISTRESS and kin_label in (LabelClass.DISTRESS, LabelClass.NORMAL):
        return GateResult(True)
    if intended == LabelClass.NORMAL and kin_label == LabelClass.NORMAL:
        return GateResult(True)
    return GateResult(False, [f"intended {intended.value} but reads as {kin_label.value}"])


@dataclass
class DistributionAudit:
    labels: Counter = field(default_factory=Counter)
    projections: Counter = field(default_factory=Counter)
    archetypes: Counter = field(default_factory=Counter)
    night: int = 0
    total: int = 0
    dropped: list[tuple[str, list[str]]] = field(default_factory=list)  # (clip_id, reasons)

    def record(self, clip_id, label, projection, archetype, is_night, passed, reasons):
        if not passed:
            self.dropped.append((clip_id, reasons))
            return
        self.total += 1
        self.labels[label] += 1
        self.projections[projection] += 1
        self.archetypes[archetype] += 1
        self.night += int(is_night)

    def warnings(self) -> list[str]:
        w = []
        if self.total == 0:
            return ["no clips passed"]
        falls = self.labels.get(LabelClass.FALL.value, 0)
        negs = self.labels.get(LabelClass.NORMAL.value, 0)
        if negs <= falls:
            w.append(f"negatives ({negs}) do not outnumber falls ({falls})")
        night_frac = self.night / self.total
        if night_frac < 0.20:
            w.append(f"night fraction low ({night_frac:.2f}, want ~0.30)")
        fish = sum(v for k, v in self.projections.items() if "fisheye" in k)
        if fish / self.total < 0.25:
            w.append(f"fisheye fraction low ({fish/self.total:.2f})")
        return w

    def summary(self) -> str:
        return (f"kept={self.total} dropped={len(self.dropped)} "
                f"labels={dict(self.labels)} proj={dict(self.projections)} "
                f"night={self.night} warnings={self.warnings()}")


if __name__ == "__main__":
    from synthgen.rationale import compute_features, detect_events, JOINTS
    T, fps = 60, 30
    joints = {n: np.zeros((T, 3)) for n in JOINTS}
    joints["pelvis"][:, 2] = 1.0
    joints["neck"][:, 2] = 1.4
    joints["head"][:, 2] = 1.6
    f = compute_features(joints, fps)
    ev = detect_events(f)
    print("plausibility (standing still):", physical_plausibility(f))
    # blow-up: launch the whole body upward (all COM joints)
    for n in ("pelvis", "head", "l_shoulder", "r_shoulder"):
        joints[n][:, 2] = np.linspace(1.0, 12.0, T)
    f2 = compute_features(joints, fps)
    res = physical_plausibility(f2)
    print("plausibility (launched):", res)
    assert not res.ok, "launch should be rejected"
    a = DistributionAudit()
    a.record("c1", LabelClass.FALL.value, "fisheye_equidistant", "ceiling", True, True, [])
    a.record("c2", LabelClass.NORMAL.value, "rectilinear", "wall_mid", False, True, [])
    print(a.summary())
    print("quality OK")
