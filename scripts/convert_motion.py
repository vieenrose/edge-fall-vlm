"""Motion library builder.

Two roles:
  1. PROCEDURAL BOOTSTRAP (works now, no assets): generate plausible fall / faint /
     immobile / distress / normal world-joint trajectories as .npz + a manifest, so the
     whole pipeline is exercisable before LAFAN1/AMASS are wired. These are kinematic
     stick-figure motions (world joints only), enough to drive skeleton_render and the
     training plumbing.
  2. REAL RETARGET (TODO, asset-gated): convert LAFAN1 / ragdoll sims / AMASS into SMPL-X
     params (.npz with poses/trans/betas). Stub signature provided; wire in M2.

The .npz schema consumed downstream:
    world_joints: dict-like -> we store one (T,3) array per JOINTS name
    fps: int
    intended_class: str
Bootstrap writes world joints directly (bodies.readback_world_joints is bypassed for the
no-Blender path); the real retarget writes SMPL-X params and Blender computes joints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synthgen.config import LabelClass
from synthgen.rationale import JOINTS


def _base_standing(T):
    j = {n: np.zeros((T, 3)) for n in JOINTS}
    j["pelvis"][:, 2] = 1.0
    j["l_hip"][:, 2] = 0.95; j["r_hip"][:, 2] = 0.95
    j["l_hip"][:, 0] = -0.1; j["r_hip"][:, 0] = 0.1
    j["l_ankle"][:, 2] = 0.08; j["r_ankle"][:, 2] = 0.08
    j["l_ankle"][:, 0] = -0.1; j["r_ankle"][:, 0] = 0.1
    j["neck"][:, 2] = 1.4
    j["l_shoulder"][:, 2] = 1.38; j["r_shoulder"][:, 2] = 1.38
    j["l_shoulder"][:, 0] = -0.18; j["r_shoulder"][:, 0] = 0.18
    j["head"][:, 2] = 1.6
    return j


def _tip_over(j, start, dur, direction, floor_z=0.15, rng=None):
    """Rotate upper body toward horizontal between [start, start+dur), then hold."""
    T = j["pelvis"].shape[0]
    dx, dy = direction
    for t in range(start, T):
        f = min(1.0, (t - start) / max(1, dur))
        # upper joints descend and translate in fall direction
        for n, h0 in (("head", 1.6), ("neck", 1.4), ("l_shoulder", 1.38), ("r_shoulder", 1.38)):
            j[n][t, 2] = h0 * (1 - f) + floor_z * f
            j[n][t, 0] += dx * f * 0.7
            j[n][t, 1] += dy * f * 0.7
        j["pelvis"][t, 2] = 1.0 * (1 - f) + (floor_z + 0.05) * f
        j["pelvis"][t, 0] += dx * f * 0.3
        j["pelvis"][t, 1] += dy * f * 0.3
    return j


def gen_fall(rng, T=90, fast=True):
    j = _base_standing(T)
    start = rng.integers(30, 45)
    dur = rng.integers(6, 10) if fast else rng.integers(18, 30)
    ang = rng.uniform(0, 2 * np.pi)
    return _tip_over(j, int(start), int(dur), (np.cos(ang), np.sin(ang)), rng=rng)


def gen_faint(rng, T=90):
    # vertical crumple: slower, less horizontal translation, collapses in place
    j = _base_standing(T)
    start = int(rng.integers(30, 45))
    dur = int(rng.integers(10, 16))
    for t in range(start, T):
        f = min(1.0, (t - start) / dur)
        for n, h0 in (("head", 1.6), ("neck", 1.4), ("l_shoulder", 1.38),
                      ("r_shoulder", 1.38), ("pelvis", 1.0)):
            j[n][t, 2] = h0 * (1 - f) + 0.2 * f
    return j


def gen_immobile(rng, T=90):
    # already lying down, still — randomized orientation/side/position for variety
    j = _base_standing(T)
    ang = rng.uniform(0, 2 * np.pi)            # which way the body points on the floor
    dx, dy = np.cos(ang), np.sin(ang)
    ox, oy = rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4)   # position on floor
    side = rng.choice([0.0, 0.08, -0.08])      # supine / on-side lift
    # body laid out along (dx,dy): head at one end, ankles at the other
    layout = {"head": 0.9, "neck": 0.6, "l_shoulder": 0.5, "r_shoulder": 0.5,
              "pelvis": 0.0, "l_hip": -0.05, "r_hip": -0.05, "l_ankle": -0.8, "r_ankle": -0.8}
    for n, along in layout.items():
        j[n][:, 0] = ox + dx * along
        j[n][:, 1] = oy + dy * along
        j[n][:, 2] = (0.12 if "ankle" in n else 0.18) + side * rng.uniform(0, 1)
    return j


def gen_distress_var(rng, T=90):
    return gen_distress(rng, T), "distress"


def gen_distress(rng, T=90):
    # crouched/struggling: mid-height, jittery, not fully down
    j = _base_standing(T)
    for n in JOINTS:
        j[n][:, 2] *= 0.6
        j[n][:, 0] += 0.03 * np.sin(np.linspace(0, 12 * np.pi, T) + rng.uniform(0, 6))
    j["head"][:, 2] = 1.0 + 0.05 * np.sin(np.linspace(0, 8 * np.pi, T))
    return j


def gen_normal(rng, T=90):
    # HARD negatives dominate: poses that resemble "down" but are normal daily activity.
    # These are what drive specificity (the model was calling half of normals "fall").
    kind = rng.choice(["stand", "sit_fast", "lie_sofa", "bend", "walk",
                       "crouch", "sit_floor", "reach_down", "squat_exercise", "kneel"])
    j = _base_standing(T)
    if kind == "crouch":                             # low but stable, upright torso
        for t in range(T):
            f = 0.4 + 0.15 * np.sin(t / T * 2 * np.pi)
            for n in ("pelvis", "l_hip", "r_hip"):
                j[n][t, 2] *= (1 - f)
            j["l_ankle"][t, 2] = j["r_ankle"][t, 2] = 0.08
        return j, kind
    if kind == "sit_floor":                          # sitting ON the floor, torso upright
        for n, h in (("pelvis", 0.2), ("l_hip", 0.2), ("r_hip", 0.2),
                     ("neck", 0.75), ("head", 0.95), ("l_shoulder", 0.72), ("r_shoulder", 0.72)):
            j[n][:, 2] = h
        j["l_ankle"][:, 2] = j["r_ankle"][:, 2] = 0.08
        j["l_ankle"][:, 0] = 0.4; j["r_ankle"][:, 0] = 0.4     # legs out front
        return j, kind
    if kind == "reach_down":                         # bends fully to pick something up, returns
        start = int(rng.integers(20, 40))
        for t in range(start, min(T, start + 30)):
            f = np.sin((t - start) / 30 * np.pi)
            j["head"][t, 2] = 1.6 - 1.2 * f; j["neck"][t, 2] = 1.4 - 1.0 * f
            j["l_shoulder"][t, 2] = j["r_shoulder"][t, 2] = 1.38 - 1.0 * f
            j["pelvis"][t, 2] = 1.0 - 0.3 * f
            for n in ("head", "neck", "l_shoulder", "r_shoulder"):
                j[n][t, 0] += 0.4 * f
        return j, kind
    if kind == "squat_exercise":                     # repeated up/down, upright
        for t in range(T):
            f = 0.35 * (0.5 + 0.5 * np.sin(t / T * 6 * np.pi))
            for n in ("pelvis", "neck", "head", "l_shoulder", "r_shoulder", "l_hip", "r_hip"):
                j[n][t, 2] *= (1 - f)
        return j, kind
    if kind == "kneel":                              # kneeling, torso upright
        for n, h in (("l_ankle", 0.1), ("r_ankle", 0.1), ("l_hip", 0.45), ("r_hip", 0.45),
                     ("pelvis", 0.5), ("neck", 0.9), ("head", 1.1),
                     ("l_shoulder", 0.88), ("r_shoulder", 0.88)):
            j[n][:, 2] = h
        return j, kind
    if kind == "sit_fast":
        start = int(rng.integers(30, 50))
        for t in range(start, T):
            f = min(1.0, (t - start) / 6)
            j["pelvis"][t, 2] = 1.0 - 0.45 * f          # hips drop to ~0.55 (chair), stays upright
    elif kind == "lie_sofa":
        for n in JOINTS:                                 # horizontal but elevated (sofa ~0.5m)
            j[n][:, 2] = np.clip(j[n][:, 2] * 0.0 + 0.55, 0.5, 0.6)
        j["head"][:, 0] = 0.6
    elif kind == "bend":
        start = int(rng.integers(25, 45))
        for t in range(start, min(T, start + 20)):
            f = np.sin((t - start) / 20 * np.pi)         # bend down and back up
            j["head"][t, 2] = 1.6 - 0.7 * f
            j["neck"][t, 2] = 1.4 - 0.5 * f
    elif kind == "walk":
        for n in JOINTS:
            j[n][:, 0] += np.linspace(0, 1.5, T)
            j[n][:, 2] += 0.02 * np.sin(np.linspace(0, 10 * np.pi, T))
    return j, kind


GEN = {
    LabelClass.FALL: lambda rng: (gen_fall(rng, fast=True), "fall"),
    LabelClass.FAINT: lambda rng: (gen_faint(rng), "faint"),
    LabelClass.IMMOBILE: lambda rng: (gen_immobile(rng), "immobile"),
    LabelClass.DISTRESS: lambda rng: (gen_distress(rng), "distress"),
    LabelClass.NORMAL: gen_normal,
}


def build(out_dir: Path, n_per_class: dict[LabelClass, int], seed=0):
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    mdir = out_dir / "motions"; mdir.mkdir(exist_ok=True)
    manifest = []
    for cls, n in n_per_class.items():
        for i in range(n):
            res = GEN[cls](rng)
            joints, sub = res if isinstance(res, tuple) else (res, cls.value)
            clip_id = f"{cls.value}_{sub}_{i:04d}"
            path = mdir / f"{clip_id}.npz"
            np.savez(path, fps=30, intended_class=cls.value,
                     **{f"joint_{n}": joints[n] for n in JOINTS})
            manifest.append({"clip_id": clip_id, "path": str(path),
                             "class": cls.value, "fps": 30})
    (out_dir / "motion_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"built {len(manifest)} motions -> {out_dir/'motion_manifest.json'}")
    return manifest


def load_world_joints(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    joints = {n: d[f"joint_{n}"] for n in JOINTS}
    return joints, int(d["fps"]), str(d["intended_class"])


# ---- REAL retarget stub (asset-gated) ----
def retarget_lafan1_to_smplx(bvh_path: Path, out_npz: Path):
    raise NotImplementedError(
        "Retarget LAFAN1 BVH -> SMPL-X params. Use a retargeting tool (e.g. SMPL-X fit "
        "or Rokoko/anim retarget), write poses(T,165)/trans(T,3)/betas. See spec 1.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/bootstrap"))
    ap.add_argument("--scale", type=int, default=4, help="multiplier on per-class counts")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    # boosted minority counts so post-render class balance is even (immobile/distress/
    # faint were starved before; falls+normal dominate after kinematic re-labeling + drops)
    # normals must DOMINATE and be varied (10 hard-negative kinds) so the model learns the
    # normal-vs-down boundary instead of over-predicting danger (specificity was 0.47).
    base = {LabelClass.FALL: 5, LabelClass.FAINT: 4, LabelClass.IMMOBILE: 5,
            LabelClass.DISTRESS: 5, LabelClass.NORMAL: 18}
    n = {k: v * args.scale for k, v in base.items()}
    build(args.out, n, seed=args.seed)


if __name__ == "__main__":
    main()
