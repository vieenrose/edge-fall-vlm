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


BRACE_STRENGTH = {"forward": 0.9, "lateral": 0.6, "backward": 0.15}   # Robinovitch et al.:
# real LTC fall videos show arm-bracing in 79% forward / 71% sideways / 46% backward falls.


def _tip_over(j, start, dur, direction, floor_z=0.15, rng=None, fall_dir="forward",
             destabilize=True, rotate=False, start_heights=None):
    """Rotate upper body toward horizontal between [start, start+dur), then hold.

    Biomechanically-informed (Robinovitch et al., real long-term-care fall videos):
    a brief destabilization sway precedes the topple; arm-bracing (approximated here as
    shoulder reach/spread, since our joint set has no separate hand joints) is stronger
    for forward/lateral falls and weak for backward falls; ~40% of forward falls rotate
    to land sideways/backward rather than landing in their initial direction; a short
    decaying settle follows impact rather than an instant hard stop."""
    T = j["pelvis"].shape[0]
    dx, dy = direction
    rng = rng or np.random.default_rng()
    brace = BRACE_STRENGTH.get(fall_dir, 0.5)
    h0 = start_heights or {"head": 1.6, "neck": 1.4, "l_shoulder": 1.38, "r_shoulder": 1.38}
    pelvis0 = (start_heights or {}).get("pelvis", 1.0)

    if destabilize:   # brief pre-topple sway, not an instant straight-line start
        pre = max(2, int(dur * 0.4))
        for t in range(max(0, start - pre), start):
            f = (t - (start - pre)) / max(1, pre)
            sway = 0.05 * np.sin(f * np.pi)
            for n in ("head", "neck", "l_shoulder", "r_shoulder", "pelvis"):
                j[n][t, 0] += dx * sway
                j[n][t, 1] += dy * sway

    land_dx, land_dy = dx, dy
    if rotate:   # descent rotates away from the initial destabilization direction
        base_ang = np.arctan2(dy, dx) + rng.uniform(np.pi / 3, np.pi) * rng.choice([-1, 1])
        land_dx, land_dy = np.cos(base_ang), np.sin(base_ang)

    for t in range(start, T):
        f = min(1.0, (t - start) / max(1, dur))
        cx, cy = dx * (1 - f) + land_dx * f, dy * (1 - f) + land_dy * f
        for n, h in h0.items():
            j[n][t, 2] = h * (1 - f) + floor_z * f
            j[n][t, 0] += cx * f * 0.7
            j[n][t, 1] += cy * f * 0.7
        j["pelvis"][t, 2] = pelvis0 * (1 - f) + (floor_z + 0.05) * f
        j["pelvis"][t, 0] += cx * f * 0.3
        j["pelvis"][t, 1] += cy * f * 0.3
        # bracing: shoulders reach/spread toward the fall direction, peaking mid-descent,
        # strongly direction-dependent (see BRACE_STRENGTH)
        peak = np.sin(np.clip(f, 0, 1) * np.pi)
        j["l_shoulder"][t, 0] += cx * peak * brace * 0.25
        j["l_shoulder"][t, 1] += cy * peak * brace * 0.25 + 0.15 * peak * brace
        j["r_shoulder"][t, 0] += cx * peak * brace * 0.25
        j["r_shoulder"][t, 1] += cy * peak * brace * 0.25 - 0.15 * peak * brace

    settle_start = start + dur   # decaying post-impact settle, not an instant hard stop
    for t in range(settle_start, min(T, settle_start + 6)):
        decay = max(0.0, 1 - (t - settle_start) / 6)
        j["pelvis"][t, 2] += 0.02 * decay * np.sin((t - settle_start) * 3)
    return j


def gen_fall(rng, T=90, fast=True):
    j = _base_standing(T)
    start = rng.integers(30, 45)
    dur = rng.integers(6, 10) if fast else rng.integers(18, 30)
    # forward falls dominate real incidents; backward/lateral less common (Robinovitch)
    fall_dir = rng.choice(["forward", "backward", "lateral"], p=[0.45, 0.25, 0.30])
    base_ang = {"forward": 0.0, "backward": np.pi}.get(fall_dir, rng.choice([np.pi / 2, -np.pi / 2]))
    ang = base_ang + rng.uniform(-0.3, 0.3)
    rotate = fall_dir == "forward" and rng.random() < 0.4   # ~40% of forward falls rotate on descent
    return _tip_over(j, int(start), int(dur), (np.cos(ang), np.sin(ang)), rng=rng,
                     fall_dir=fall_dir, rotate=rotate)


def _low_pose_heights(kind):
    """Starting joint heights for a bent/crouched/reaching/kneeling pose -- the SAME
    pose family as gen_normal's hard-negative low poses. Returns dict name->height.
    Includes hip (kept close to pelvis, as gen_normal's crouch/kneel do) -- leaving hip
    frozen at standing height while pelvis is at crouch/kneel height breaks the skin-mesh
    bone chain pelvis->hip->knee->ankle into a physically-impossible configuration that
    renders as a degenerate spike/blade (the root cause of a real visual bug found by
    inspecting actual training renders)."""
    if kind == "crouch":
        return {"head": 1.0, "neck": 0.86, "l_shoulder": 0.84, "r_shoulder": 0.84,
               "pelvis": 0.6, "hip": 0.56}
    if kind == "reach_down":
        return {"head": 0.85, "neck": 0.75, "l_shoulder": 0.73, "r_shoulder": 0.73,
               "pelvis": 0.7, "hip": 0.66}
    if kind == "kneel":
        return {"head": 1.1, "neck": 0.9, "l_shoulder": 0.88, "r_shoulder": 0.88,
               "pelvis": 0.5, "hip": 0.45}
    raise ValueError(kind)


def gen_fall_from_low(rng, T=90):
    """Fall that starts from an ALREADY-LOW bent/crouched/reaching/kneeling pose (not
    standing), toppling the rest of the way to the floor. Real-world falls often start
    from exactly this kind of low pose (someone bent over a chair/table who loses
    balance) -- visually it's a SMALL net height change plus a fast topple/kick dynamic,
    not the big standing->floor silhouette drop gen_fall produces. Without this variant
    the training data teaches "low pose = normal" and "big height drop = fall" but never
    "low pose that topples = fall", exactly the case a real low-starting-pose fall is."""
    kind = rng.choice(["crouch", "reach_down", "kneel"])
    heights = _low_pose_heights(kind)
    j = _base_standing(T)
    for n, h in heights.items():
        if n == "hip":
            j["l_hip"][:, 2] = j["r_hip"][:, 2] = h
        else:
            j[n][:, 2] = h
    if kind == "reach_down":
        for n in ("head", "neck", "l_shoulder", "r_shoulder"):
            j[n][:, 0] += 0.35   # leaning forward, reaching
    elif kind == "kneel":
        j["l_ankle"][:, 2] = j["r_ankle"][:, 2] = 0.1

    start = int(rng.integers(30, 45))
    dur = int(rng.integers(6, 12))
    ang = rng.uniform(0, 2 * np.pi)
    dx, dy = np.cos(ang), np.sin(ang)
    fall_dir = rng.choice(["forward", "backward", "lateral"], p=[0.5, 0.2, 0.3])
    ankle0 = 0.1 if kind != "crouch" else 0.08
    _tip_over(j, start, dur, (dx, dy), rng=rng, fall_dir=fall_dir,
             start_heights={n: heights[n] for n in ("head", "neck", "l_shoulder",
                                                     "r_shoulder", "pelvis")})
    for t in range(start, T):
        f = min(1.0, (t - start) / max(1, dur))
        # transient leg-kick: ankles swing UP above resting height mid-topple, then
        # settle back down -- the kinematic signature a fast topple actually has,
        # regardless of how small the head/torso height drop is.
        kick = np.sin(np.clip(f, 0, 1) * np.pi) * 0.5   # 0->peak mid-topple->0
        for n in ("l_ankle", "r_ankle"):
            j[n][t, 2] = ankle0 * (1 - f) + ankle0 * f + kick
            j[n][t, 0] += -dx * f * 0.3
            j[n][t, 1] += -dy * f * 0.3
        # keep hips consistent with pelvis throughout -- frozen-at-standing-height hips
        # while pelvis descends breaks the pelvis->hip->knee->ankle bone chain
        for n in ("l_hip", "r_hip"):
            j[n][t, 2] = heights["hip"] * (1 - f) + 0.18 * f
            j[n][t, 0] += dx * f * 0.3
            j[n][t, 1] += dy * f * 0.3
    return j, f"fall_from_{kind}"


def gen_fall_from_seated(rng, T=90):
    """Fall off a chair/bed edge -- documented industry-wide blind spot (Tanwar et al.,
    Healthcare 2022): public fall datasets almost never script falls starting from a
    seated/bed posture despite these being common real geriatric incidents (bed-exit
    falls, chair transfers). Starts from a sitting pose (chair-height pelvis), topples
    sideways or forward off the seat to the floor."""
    j = _base_standing(T)
    sit_h = rng.uniform(0.45, 0.55)     # chair/bed-edge seat height
    hip_h = sit_h - 0.05
    heights = {"head": 1.15, "neck": 0.95, "l_shoulder": 0.93, "r_shoulder": 0.93,
              "pelvis": sit_h}
    for n, h in heights.items():
        j[n][:, 2] = h
    j["l_hip"][:, 2] = j["r_hip"][:, 2] = hip_h   # keep hip consistent with seated pelvis
                                                    # (frozen-standing hip breaks the bone chain)
    j["l_ankle"][:, 2] = j["r_ankle"][:, 2] = 0.08
    j["l_ankle"][:, 0] = 0.25; j["r_ankle"][:, 0] = 0.25   # feet forward, seated

    start = int(rng.integers(30, 45))
    dur = int(rng.integers(8, 14))
    fall_dir = rng.choice(["forward", "lateral"], p=[0.4, 0.6])  # sideways-off-chair common
    ang = {"forward": 0.0}.get(fall_dir, rng.choice([np.pi / 2, -np.pi / 2])) + rng.uniform(-0.3, 0.3)
    dx, dy = np.cos(ang), np.sin(ang)
    _tip_over(j, start, dur, (dx, dy), rng=rng, fall_dir=fall_dir,
             start_heights=heights)
    for t in range(start, T):
        f = min(1.0, (t - start) / max(1, dur))
        for n in ("l_hip", "r_hip"):
            j[n][t, 2] = hip_h * (1 - f) + 0.18 * f
            j[n][t, 0] += dx * f * 0.3
            j[n][t, 1] += dy * f * 0.3
    return j, "fall_from_seated"


def gen_faint(rng, T=90):
    """Syncope: per clinical literature, LOC precedes the collapse, so there is a
    conscious PRODROME (presyncope: swaying, reaching for support) followed by a PASSIVE
    "boneless" drop with NO protective bracing reflex -- unlike gen_fall's direction-
    conditional arm-reach, syncope's defining kinematic marker is the ABSENCE of that
    reflex. Slower/more vertical than a mechanical fall, but once LOC hits, the actual
    drop is passive, not a controlled sit-down."""
    j = _base_standing(T)
    sway_start = int(rng.integers(15, 30))
    sway_dur = int(rng.integers(15, 30))         # conscious prodrome: seconds of swaying
    for t in range(sway_start, min(T, sway_start + sway_dur)):
        f = (t - sway_start) / sway_dur
        sway = 0.08 * np.sin(f * 5 * np.pi)
        for n in ("head", "neck", "l_shoulder", "r_shoulder"):
            j[n][t, 0] += sway
        j["l_shoulder"][t, 0] += 0.15 * f    # reaching for support before LOC
        j["l_shoulder"][t, 1] += 0.08 * f
    start = sway_start + sway_dur
    dur = int(rng.integers(8, 14))               # once LOC hits, the drop itself is fast
    for t in range(start, T):
        f = min(1.0, (t - start) / dur)
        for n, h0 in (("head", 1.6), ("neck", 1.4), ("l_shoulder", 1.38),
                      ("r_shoulder", 1.38), ("pelvis", 1.0)):
            j[n][t, 2] = h0 * (1 - f) + 0.2 * f
            # NO bracing offset here (contrast with gen_fall) -- passive collapse
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


def gen_distress(rng, T=90):
    """Distress covers several medically-distinct acute events (per clinical research):
    general struggling (original), choking (universal sign: hands at throat -- approximated
    here as shoulders pulled up/inward, person stays UPRIGHT unlike the crouched struggle
    variant), and a cardiac event (chest-clutch: torso hunches forward, shoulders pull in,
    motion settles to stillness rather than continued jitter)."""
    kind = rng.choice(["struggle", "choking", "cardiac"], p=[0.5, 0.25, 0.25])
    j = _base_standing(T)
    if kind == "struggle":
        for n in JOINTS:
            j[n][:, 2] *= 0.6
            j[n][:, 0] += 0.03 * np.sin(np.linspace(0, 12 * np.pi, T) + rng.uniform(0, 6))
        j["head"][:, 2] = 1.0 + 0.05 * np.sin(np.linspace(0, 8 * np.pi, T))
    elif kind == "choking":
        # stays standing (not crouched) -- shoulders pulled up/inward toward throat
        for n in ("l_shoulder", "r_shoulder"):
            j[n][:, 2] = 1.5
            j[n][:, 0] += 0.05 * (1 if n == "l_shoulder" else -1)
        j["neck"][:, 2] = 1.42
        panic = 0.02 * np.sin(np.linspace(0, 14 * np.pi, T) + rng.uniform(0, 6))
        j["head"][:, 0] += panic
        j["head"][:, 1] += panic
    else:  # cardiac
        start = int(rng.integers(15, 30))
        for t in range(start, T):
            f = min(1.0, (t - start) / 15)
            j["neck"][t, 2] = 1.4 - 0.35 * f     # hunch forward
            j["head"][t, 2] = 1.6 - 0.4 * f
            for n in ("head", "neck"):
                j[n][t, 0] += 0.25 * f
            for n in ("l_shoulder", "r_shoulder"):
                j[n][t, 2] = 1.38 - 0.25 * f
                j[n][t, 0] += 0.15 * f            # arms pull in toward chest
        j["pelvis"][:, 2] = 0.95   # slight sink, e.g. leaning on furniture, not fully down
    return j, kind


def gen_normal(rng, T=90):
    # HARD negatives dominate: poses that resemble "down" but are normal daily activity.
    # These are what drive specificity (the model was calling half of normals "fall").
    kind = rng.choice(["stand", "sit_fast", "lie_sofa", "bend", "walk",
                       "crouch", "sit_floor", "reach_down", "squat_exercise", "kneel",
                       "near_fall"])
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
            j["pelvis"][t, 2] = 1.0 - 0.45 * f          # pelvis drops to ~0.55 (chair), stays upright
            j["l_hip"][t, 2] = j["r_hip"][t, 2] = 0.95 - 0.45 * f   # keep hip consistent
                                                                     # with pelvis (bone chain)
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
    elif kind == "near_fall":
        # documented CV/medical confuser: destabilize + successful arm/step recovery,
        # resolves BACK to upright -- unlike gen_fall, the CoM trajectory gets arrested.
        start = int(rng.integers(25, 45))
        dip = int(rng.integers(8, 14))
        recover = int(rng.integers(10, 18))
        ang = rng.uniform(0, 2 * np.pi)
        dx, dy = np.cos(ang), np.sin(ang)
        for t in range(start, min(T, start + dip)):
            f = (t - start) / dip
            for n2, h0 in (("head", 1.6), ("neck", 1.4), ("l_shoulder", 1.38), ("r_shoulder", 1.38)):
                j[n2][t, 2] = h0 * (1 - 0.35 * f)
                j[n2][t, 0] += dx * f * 0.3
                j[n2][t, 1] += dy * f * 0.3
            # arm-reach recovery gesture (shoulder extends out, bracing against a fall)
            j["l_shoulder"][t, 0] += dx * f * 0.3
            j["r_shoulder"][t, 0] += dx * f * 0.3
        rec_start = start + dip
        for t in range(rec_start, min(T, rec_start + recover)):
            f = (t - rec_start) / recover
            for n2, h0 in (("head", 1.6), ("neck", 1.4), ("l_shoulder", 1.38), ("r_shoulder", 1.38)):
                j[n2][t, 2] = h0 * (1 - 0.35 * (1 - f)) if t < T else h0
                j[n2][t, 0] += dx * (1 - f) * 0.3 * 0.3
        # settles back fully upright for the remaining tail
        tail_start = min(T, rec_start + recover)
        for t in range(tail_start, T):
            for n2, h0 in (("head", 1.6), ("neck", 1.4), ("l_shoulder", 1.38), ("r_shoulder", 1.38)):
                j[n2][t, 2] = h0
    return j, kind


def gen_seizure(rng, T=90):
    """Tonic-clonic seizure: rigid full-body stiffening then falling like a plank (NO
    protective bracing at all -- more rigid than even syncope's passive drop), followed
    by rhythmic clonic jerking that continues through the rest of the clip. Kinematically
    distinct from both fall (bracing, settles still) and faint (slower, no jerk tail)."""
    j = _base_standing(T)
    start = int(rng.integers(20, 35))
    dur = int(rng.integers(9, 14))     # rigid plank-fall, same pace as a fast mechanical
                                        # fall -- the "no bracing" is what distinguishes it,
                                        # not an even-faster descent (which trips the
                                        # physical-plausibility gate's acceleration bound)
    ang = rng.uniform(0, 2 * np.pi)
    dx, dy = np.cos(ang), np.sin(ang)
    for t in range(start, T):
        f = min(1.0, (t - start) / dur)
        for n, h0 in (("head", 1.6), ("neck", 1.4), ("l_shoulder", 1.38), ("r_shoulder", 1.38)):
            j[n][t, 2] = h0 * (1 - f) + 0.18 * f
            j[n][t, 0] += dx * f * 0.6
            j[n][t, 1] += dy * f * 0.6
        j["pelvis"][t, 2] = 1.0 * (1 - f) + 0.2 * f
        j["pelvis"][t, 0] += dx * f * 0.3
        j["pelvis"][t, 1] += dy * f * 0.3
    jerk_start = start + dur
    for t in range(jerk_start, T):
        # amplitude/frequency tuned to register as COM-level velocity oscillation (the
        # marker classify() uses for clonic jerking) while staying under the physical-
        # plausibility gate's COM-acceleration/updraft bounds (synthgen/quality.py).
        jerk = 0.07 * np.sin((t - jerk_start) * 1.8) * rng.uniform(0.8, 1.2)
        for n in ("l_ankle", "r_ankle", "head", "neck", "l_shoulder", "r_shoulder", "pelvis"):
            j[n][t, 2] += abs(jerk)
            j[n][t, 0] += jerk
    return j


def gen_fall_mixed(rng):
    # standing-topple / low-pose-topple / seated(chair-bed)-topple -- covers "fell from
    # standing", "fell from an already-bent/low posture", and "fell from a chair/bed",
    # closing the industry-documented gap of only scripting standing-start falls.
    r = rng.random()
    if r < 0.5:
        return gen_fall(rng, fast=True), "fall_standing"
    if r < 0.8:
        return gen_fall_from_low(rng)
    return gen_fall_from_seated(rng)


GEN = {
    LabelClass.FALL: gen_fall_mixed,
    LabelClass.FAINT: lambda rng: (gen_faint(rng), "faint"),
    LabelClass.IMMOBILE: lambda rng: (gen_immobile(rng), "immobile"),
    LabelClass.SEIZURE: lambda rng: (gen_seizure(rng), "seizure"),
    LabelClass.DISTRESS: gen_distress,
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
            LabelClass.DISTRESS: 5, LabelClass.SEIZURE: 3, LabelClass.NORMAL: 18}
    n = {k: v * args.scale for k, v in base.items()}
    build(args.out, n, seed=args.seed)


if __name__ == "__main__":
    main()
