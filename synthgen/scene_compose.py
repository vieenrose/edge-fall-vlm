"""Multi-person scene composition + danger-label aggregation.

A scene has 1-3 people, each with its own motion clip and floor position. Realistic
scenarios: a lone person, someone falling while bystanders stand/walk, a person on the
floor with another checking on them, two people acting normally. 3D rendering gives real
inter-person occlusion for free.

Scene label = the WORST danger present (a scene is an alert if ANY person is in danger).
Pure logic; no bpy. Testable standalone.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import LabelClass

# severity order: the scene status is the worst danger present
SEVERITY = [LabelClass.FALL, LabelClass.FAINT, LabelClass.IMMOBILE,
            LabelClass.DISTRESS, LabelClass.NORMAL]
_RANK = {c: i for i, c in enumerate(SEVERITY)}

# how many people per scene (weighted toward 1)
N_PEOPLE_WEIGHTS = {1: 0.55, 2: 0.32, 3: 0.13}


@dataclass
class Person:
    clip_id: str
    motion_path: str
    label: LabelClass
    origin: np.ndarray          # (x,y) floor offset for this person
    yaw: float                  # facing rotation about Z (rad)


@dataclass
class SceneSpec:
    people: list[Person]
    status: LabelClass                 # aggregated scene label
    danger_index: int | None           # which person drives the label (None if all normal)

    @property
    def n(self) -> int:
        return len(self.people)


def _place(rng, n: int, spread=1.6, min_sep=0.6) -> list[np.ndarray]:
    """Sample n non-overlapping floor origins within +/-spread metres."""
    pts = []
    for _ in range(n):
        for _try in range(30):
            p = np.array([rng.uniform(-spread, spread), rng.uniform(-spread, spread)])
            if all(np.linalg.norm(p - q) >= min_sep for q in pts):
                pts.append(p)
                break
        else:
            pts.append(p)  # give up on separation for this one
    return pts


def sample_n_people(rng) -> int:
    ks = list(N_PEOPLE_WEIGHTS)
    return int(rng.choice(ks, p=[N_PEOPLE_WEIGHTS[k] for k in ks]))


def compose_scene(rng, pick_motion) -> SceneSpec:
    """Build a scene. `pick_motion(rng, prefer_danger: bool) -> (clip_id, path, label)`
    returns a motion of roughly the requested kind. We bias the first person toward a
    danger class often, the rest toward normal (bystanders), but allow multi-danger."""
    n = sample_n_people(rng)
    origins = _place(rng, n)
    people = []
    # ~half the scenes should be fully normal (a multi-person scene is "danger" if ANYONE
    # is down, so a high per-person danger bias starves the normal-SCENE class and tanks
    # specificity). Lower person-0 bias to ~0.5 to balance danger vs normal scenes.
    for i in range(n):
        prefer_danger = (i == 0 and rng.random() < 0.5) or (i > 0 and rng.random() < 0.15)
        clip_id, path, label = pick_motion(rng, prefer_danger)
        people.append(Person(clip_id, path, LabelClass(label), origins[i],
                             yaw=rng.uniform(0, 2 * np.pi)))
    danger = [(i, p) for i, p in enumerate(people) if p.label != LabelClass.NORMAL]
    if danger:
        di, dp = min(danger, key=lambda ip: _RANK[ip[1].label])   # worst severity
        status = dp.label
    else:
        di, status = None, LabelClass.NORMAL
    return SceneSpec(people, status, di)


def scene_answer(spec: SceneSpec, person_down: bool, posture: str = "unknown") -> dict:
    # posture FIRST (pose-assist: forces the model to commit to torso orientation before
    # the status) — it is the cue that separates down from low-but-normal.
    return {"posture": posture,
            "status": spec.status.value,
            "confidence": 0.9 if spec.status != LabelClass.NORMAL else 0.6,
            "person_down": bool(person_down),
            "n_people": spec.n}


def scene_rationale(spec: SceneSpec, per_person_rationale: list[str]) -> str:
    head = f"{spec.n} person(s) in view. "
    if spec.danger_index is None:
        return head + "All acting normally -> normal."
    dp = spec.people[spec.danger_index]
    who = "the person" if spec.n == 1 else f"person {spec.danger_index + 1} of {spec.n}"
    return head + f"{who}: {per_person_rationale[spec.danger_index]} -> {spec.status.value}."


def transform_joints(joints: dict, origin: np.ndarray, yaw: float) -> dict:
    """Rotate a person's joints by yaw about Z and translate by origin (x,y)."""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    off = np.array([origin[0], origin[1], 0.0])
    return {k: (v @ R.T) + off for k, v in joints.items()}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # stub motion picker
    pool = {True: [("fall_1", "p", "fall"), ("faint_1", "p", "faint-collapse"),
                   ("dist_1", "p", "distress")],
            False: [("norm_1", "p", "normal"), ("norm_2", "p", "normal")]}
    def pick(rng, prefer_danger):
        opts = pool[prefer_danger]
        return opts[int(rng.integers(len(opts)))]
    counts = {}
    saw_multi = saw_fall_with_bystander = False
    for _ in range(400):
        sp = compose_scene(rng, pick)
        counts[sp.n] = counts.get(sp.n, 0) + 1
        if sp.n > 1:
            saw_multi = True
            if sp.status == LabelClass.FALL and any(p.label == LabelClass.NORMAL for p in sp.people):
                saw_fall_with_bystander = True
        # severity aggregation sanity
        labels = [p.label for p in sp.people]
        if LabelClass.FALL in labels:
            assert sp.status == LabelClass.FALL
    print("n-people distribution:", counts)
    assert saw_multi and saw_fall_with_bystander
    # transform sanity: yaw=pi/2 rotates +x joint to +y
    j = {"a": np.array([[1.0, 0, 0]])}
    t = transform_joints(j, np.array([0.0, 0.0]), np.pi / 2)
    assert abs(t["a"][0, 1] - 1.0) < 1e-6 and abs(t["a"][0, 0]) < 1e-6
    print("scene_compose OK")
