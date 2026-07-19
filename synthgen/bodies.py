"""SMPL-X animated bodies from a motion clip. Runs INSIDE `blenderproc run` (needs bpy).

BlenderProc's built-in load_AMASS only loads a single static CMU frame, so it is NOT
enough for temporal falls. We drive an SMPL-X mesh per-frame from a motion clip
(LAFAN1 retargeted to SMPL-X, ragdoll sim, or AMASS ADL) using the Meshcapade
SMPL_blender_addon, then read back world-space joints for the rationale/GT.

This module is an integration skeleton: the marked steps depend on the SMPL-X model
files + addon being installed and on the exact motion representation. Validate on-machine
in M2 build-order step 2. The joint-readback contract (returns dict[name]->(T,3)) is
what the rest of the pipeline consumes and is stable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .rationale import JOINTS

# Map our named joints -> SMPL-X joint indices (SMPL-X body joint order).
# These are the standard SMPL-X indices; confirm against your addon's skeleton.
SMPLX_JOINT_INDEX = {
    "pelvis": 0, "l_hip": 1, "r_hip": 2, "l_ankle": 7, "r_ankle": 8,
    "neck": 12, "l_shoulder": 16, "r_shoulder": 17, "head": 15,
}


@dataclass
class MotionClip:
    clip_id: str
    path: Path            # .npz with per-frame SMPL-X params (poses, trans, betas, gender)
    intended_class: str   # design-time label (fall/faint/.../normal)
    fps: int = 30


def load_motion(path: Path) -> dict:
    """Load a motion .npz. Expected keys: poses (T,165), trans (T,3), betas (10..), gender.
    LAFAN1/ragdoll must be pre-retargeted to SMPL-X params by the offline converter
    (see scripts/convert_motion.py TODO)."""
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def spawn_animated_smplx(motion: dict):
    """Create an SMPL-X mesh in Blender and key its pose per frame. Returns the object.

    Integration point (needs SMPL_blender_addon):
      1. bpy.ops.scene.smplx_add_gender(...) or addon's create operator
      2. for each frame t: set body_pose/global_orient/transl from motion['poses'][t],
         insert keyframes
      3. set shape (betas)
    Kept as a guarded stub so the module imports cleanly outside Blender for testing.
    """
    import bpy  # noqa: F401  (only present inside blenderproc run)
    raise NotImplementedError(
        "Wire to SMPL_blender_addon: create SMPL-X body, key poses per frame from "
        "motion['poses']/'trans'/'betas'. See bodies.py docstring.")


def readback_world_joints(smplx_obj, n_frames: int, fps: int) -> dict[str, np.ndarray]:
    """Read world-space positions of our named joints for every frame.

    Uses the armature bone head positions (matrix_world @ bone.head). Returns
    dict[name]->(T,3). This is the GT that feeds rationale.compute_features and the
    fisheye projection — it is camera-independent, hence view-canonical.
    """
    import bpy
    scene = bpy.context.scene
    arm = smplx_obj if smplx_obj.type == "ARMATURE" else smplx_obj.find_armature()
    out = {name: np.zeros((n_frames, 3)) for name in JOINTS}
    # bone names in the addon skeleton; adjust mapping if the rig differs
    name_to_bone = {
        "pelvis": "pelvis", "l_hip": "left_hip", "r_hip": "right_hip",
        "l_ankle": "left_ankle", "r_ankle": "right_ankle", "neck": "neck",
        "l_shoulder": "left_shoulder", "r_shoulder": "right_shoulder", "head": "head",
    }
    for t in range(n_frames):
        scene.frame_set(t)
        mw = arm.matrix_world
        for name in JOINTS:
            bone = arm.pose.bones.get(name_to_bone[name])
            if bone is None:
                continue
            out[name][t] = np.array(mw @ bone.head)
    return out


def subject_centroid(world_joints: dict[str, np.ndarray]) -> np.ndarray:
    """Mean pelvis position over the clip — the point cameras are aimed around."""
    return world_joints["pelvis"].mean(axis=0)
