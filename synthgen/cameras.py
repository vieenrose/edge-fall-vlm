"""Camera + lens sampling and ground-truth projection.

This is the perspective-robustness engine. Two responsibilities:

  1. Sample a camera pose + lens per SYNTHETIC_DATA_SPEC.md §2 and configure it in
     Blender. Rectilinear lenses use a normal PERSP camera (BlenderProc-native).
     True 150-200 deg fisheye uses Blender's NATIVE panoramic camera
     (cam.type='PANO'), because BlenderProc's set_lens_distortion is Brown-Conrady
     and cannot represent >~150 deg fisheye faithfully.

  2. Project ground-truth 3D joints into whatever lens was used, so we get correct
     2D keypoints even under fisheye (Blender/BlenderProc's GT assumes pinhole).

The projection math here is the reason we can auto-label fisheye views for free.

`configure_camera` / native-fisheye setup import bpy and only run inside Blender.
The projection functions are pure numpy and unit-testable outside Blender.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import (LENSES, MOUNTS, OPTICAL_CENTER_OFFSET, POSITION_JITTER,
                     LensProfile, MountArchetype, MountProfile, Projection)


# ---------------------------------------------------------------------------
# Sampled camera description (pure data; no bpy).
# ---------------------------------------------------------------------------
@dataclass
class CameraSample:
    archetype: MountArchetype
    location: np.ndarray        # (3,) world, metres
    look_at: np.ndarray         # (3,) world point the camera aims at
    roll_deg: float
    projection: Projection
    # lens params (only the relevant ones are set)
    hfov_deg: float | None = None
    fisheye_fov_deg: float | None = None
    fisheye_lens_mm: float | None = None
    sensor_mm: float = 36.0
    res_x: int = 640
    res_y: int = 640
    cx_frac: float = 0.5        # principal point as fraction of width
    cy_frac: float = 0.5

    @property
    def is_fisheye(self) -> bool:
        return self.projection in (Projection.FISHEYE_EQUIDISTANT,
                                   Projection.FISHEYE_EQUISOLID)


# ---------------------------------------------------------------------------
# Weighted choice helpers.
# ---------------------------------------------------------------------------
def _weighted_choice(rng, items, weights):
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return items[int(rng.choice(len(items), p=w))]


def _sample_mount(rng) -> tuple[MountArchetype, MountProfile]:
    keys = list(MOUNTS.keys())
    prof = _weighted_choice(rng, keys, [MOUNTS[k].weight for k in keys])
    return prof, MOUNTS[prof]


def _sample_lens(rng, force: Projection | None = None) -> LensProfile:
    if force is not None:
        return next(l for l in LENSES if l.projection == force)
    return _weighted_choice(rng, list(LENSES), [l.weight for l in LENSES])


# ---------------------------------------------------------------------------
# Sample a full camera around a subject centroid.
# ---------------------------------------------------------------------------
def sample_camera(rng, subject_centroid: np.ndarray, short_side_px: int,
                  force_projection: Projection | None = None) -> CameraSample:
    archetype, mount = _sample_mount(rng)
    lens = _sample_lens(rng, force=force_projection)

    # spherical placement around the (jittered) subject centroid
    target = subject_centroid + np.array([POSITION_JITTER.sample(rng) for _ in range(3)])
    az = math.radians(rng.uniform(0, 360))          # full ring — never bias frontal
    pitch = math.radians(mount.pitch_deg.sample(rng))
    dist = mount.distance.sample(rng)
    height = mount.height.sample(rng)

    # horizontal offset from target by distance*cos(pitch), vertical fixed by mount height
    horiz = dist * math.cos(pitch)
    loc = np.array([
        target[0] + horiz * math.cos(az),
        target[1] + horiz * math.sin(az),
        height,
    ])
    look_at = target.copy()

    # square-ish sensor; res from short side, keep 4:3-ish for non-fisheye, 1:1 for fisheye
    if lens.projection in (Projection.FISHEYE_EQUIDISTANT, Projection.FISHEYE_EQUISOLID):
        res_x = res_y = short_side_px
    else:
        res_y = short_side_px
        res_x = int(round(short_side_px * 4 / 3))

    cam = CameraSample(
        archetype=archetype,
        location=loc,
        look_at=look_at,
        roll_deg=mount.roll_deg.sample(rng),
        projection=lens.projection,
        res_x=res_x,
        res_y=res_y,
        cx_frac=0.5 + OPTICAL_CENTER_OFFSET.sample(rng),
        cy_frac=0.5 + OPTICAL_CENTER_OFFSET.sample(rng),
    )
    if lens.hfov_deg is not None:
        cam.hfov_deg = lens.hfov_deg.sample(rng)
    if lens.fisheye_fov_deg is not None:
        cam.fisheye_fov_deg = lens.fisheye_fov_deg.sample(rng)
    if lens.fisheye_lens_mm is not None:
        cam.fisheye_lens_mm = lens.fisheye_lens_mm.sample(rng)
    return cam


# ---------------------------------------------------------------------------
# World -> camera transform (OpenCV convention: +Z forward, +X right, +Y down).
# ---------------------------------------------------------------------------
def look_at_extrinsics(location: np.ndarray, look_at: np.ndarray, roll_deg: float) -> np.ndarray:
    """Return 4x4 world->camera (OpenCV frame)."""
    fwd = look_at - location
    fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(fwd, world_up)) > 0.999:      # looking straight down/up: pick a stable up
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, world_up); right /= np.linalg.norm(right) + 1e-9
    up = np.cross(right, fwd)
    # apply roll about the forward axis
    r = math.radians(roll_deg)
    right2 = math.cos(r) * right + math.sin(r) * up
    up2 = -math.sin(r) * right + math.cos(r) * up
    # OpenCV: x=right, y=down, z=forward
    R_wc = np.stack([right2, -up2, fwd], axis=0)   # rows map world->cam axes
    t = -R_wc @ location
    T = np.eye(4)
    T[:3, :3] = R_wc
    T[:3, 3] = t
    return T


# ---------------------------------------------------------------------------
# Project 3D world points to 2D pixels for each lens model.
# Returns (uv[N,2], valid[N]) where valid marks points in front of the lens / in FOV.
# ---------------------------------------------------------------------------
def project_points(cam: CameraSample, pts_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = look_at_extrinsics(cam.location, cam.look_at, cam.roll_deg)
    pts_h = np.concatenate([pts_world, np.ones((len(pts_world), 1))], axis=1)
    pc = (T @ pts_h.T).T[:, :3]        # camera-frame points, +Z forward
    cx = cam.cx_frac * cam.res_x
    cy = cam.cy_frac * cam.res_y

    if cam.projection in (Projection.RECTILINEAR, Projection.WIDE_RECTILINEAR):
        f_px = (cam.res_x / 2.0) / math.tan(math.radians(cam.hfov_deg) / 2.0)
        z = pc[:, 2]
        valid = z > 1e-6
        z_safe = np.where(valid, z, 1.0)
        u = cx + f_px * pc[:, 0] / z_safe
        v = cy + f_px * pc[:, 1] / z_safe
        uv = np.stack([u, v], axis=1)
        valid &= (u >= 0) & (u < cam.res_x) & (v >= 0) & (v < cam.res_y)
        return uv, valid

    # ---- fisheye: angle theta from optical axis maps to radius r(theta) ----
    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
    r3 = np.sqrt(x * x + y * y + z * z) + 1e-9
    theta = np.arccos(np.clip(z / r3, -1.0, 1.0))    # angle from +Z (optical axis)
    phi = np.arctan2(y, x)                            # azimuth in image plane
    half_fov = math.radians((cam.fisheye_fov_deg or 180.0) / 2.0)

    # focal length in px so that r(half_fov) == image radius (short side / 2)
    img_radius = min(cam.res_x, cam.res_y) / 2.0
    if cam.projection == Projection.FISHEYE_EQUIDISTANT:      # r = f*theta
        f_px = img_radius / max(half_fov, 1e-6)
        r_px = f_px * theta
    else:                                                     # equisolid r = 2f*sin(theta/2)
        f_px = img_radius / (2.0 * math.sin(half_fov / 2.0))
        r_px = 2.0 * f_px * np.sin(theta / 2.0)

    u = cx + r_px * np.cos(phi)
    v = cy + r_px * np.sin(phi)
    uv = np.stack([u, v], axis=1)
    valid = (theta <= half_fov) & (u >= 0) & (u < cam.res_x) & (v >= 0) & (v < cam.res_y)
    return uv, valid


def intrinsics_dict(cam: CameraSample) -> dict:
    """Serializable intrinsics for meta.json (see spec §4)."""
    d = {"projection": cam.projection.value,
         "res_x": cam.res_x, "res_y": cam.res_y,
         "cx_frac": round(cam.cx_frac, 4), "cy_frac": round(cam.cy_frac, 4)}
    if cam.hfov_deg is not None:
        d["hfov_deg"] = round(cam.hfov_deg, 2)
    if cam.fisheye_fov_deg is not None:
        d["fisheye_fov_deg"] = round(cam.fisheye_fov_deg, 2)
    if cam.fisheye_lens_mm is not None:
        d["fisheye_lens_mm"] = round(cam.fisheye_lens_mm, 2)
    return d


# ---------------------------------------------------------------------------
# Blender-side configuration. Only call inside `blenderproc run`.
# ---------------------------------------------------------------------------
def configure_camera_in_blender(cam: CameraSample):
    """Set up the active Blender camera to match `cam`. Requires bpy + blenderproc.

    Rectilinear -> BlenderProc K-matrix path.
    Fisheye     -> native Cycles panoramic camera (raw bpy), which BlenderProc's
                   K-matrix API does not model. GT keypoints for fisheye come from
                   project_points() above, NOT from BlenderProc's pinhole GT.
    """
    import bpy
    import blenderproc as bproc

    bproc.camera.set_resolution(cam.res_x, cam.res_y)
    cam_data = bpy.context.scene.camera.data

    if cam.projection in (Projection.RECTILINEAR, Projection.WIDE_RECTILINEAR):
        cam_data.type = "PERSP"
        f_px = (cam.res_x / 2.0) / math.tan(math.radians(cam.hfov_deg) / 2.0)
        K = np.array([[f_px, 0, cam.cx_frac * cam.res_x],
                      [0, f_px, cam.cy_frac * cam.res_y],
                      [0, 0, 1.0]])
        bproc.camera.set_intrinsics_from_K_matrix(K, cam.res_x, cam.res_y)
    else:
        # native panoramic fisheye
        cam_data.type = "PANO"
        # panorama_type name differs across Blender versions; guard it.
        pano_attr = "panorama_type" if hasattr(cam_data, "panorama_type") else "cycles.panorama_type"
        if cam.projection == Projection.FISHEYE_EQUIDISTANT:
            _set_nested(cam_data, pano_attr, "FISHEYE_EQUIDISTANT")
            _set_nested(cam_data, "fisheye_fov", math.radians(cam.fisheye_fov_deg))
        else:
            _set_nested(cam_data, pano_attr, "FISHEYE_EQUISOLID")
            _set_nested(cam_data, "fisheye_lens", cam.fisheye_lens_mm)
            _set_nested(cam_data, "fisheye_fov", math.radians(cam.fisheye_fov_deg or 180.0))
            cam_data.sensor_width = cam.sensor_mm

    # pose: convert our OpenCV world->cam to Blender cam->world
    T_wc = look_at_extrinsics(cam.location, cam.look_at, cam.roll_deg)
    T_cw = np.linalg.inv(T_wc)
    T_cw = bproc.math.change_source_coordinate_frame_of_transformation_matrix(
        T_cw, ["X", "-Y", "-Z"])   # OpenCV -> Blender camera frame
    bproc.camera.add_camera_pose(T_cw)


def _set_nested(obj, dotted_attr, value):
    """Set possibly-nested attribute like 'cycles.panorama_type'."""
    parts = dotted_attr.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    if hasattr(obj, parts[-1]):
        setattr(obj, parts[-1], value)


def _apply_lens(cam_data, cam: CameraSample, res_x: int, res_y: int):
    """Set Blender camera-data lens fields for a CameraSample. Native PANO for fisheye."""
    import math as _m
    if cam.projection in (Projection.RECTILINEAR, Projection.WIDE_RECTILINEAR):
        cam_data.type = "PERSP"
        cam_data.sensor_fit = "HORIZONTAL"
        cam_data.sensor_width = cam.sensor_mm
        # focal length from HFOV: f = (sensor/2) / tan(hfov/2)
        cam_data.lens = (cam.sensor_mm / 2.0) / _m.tan(_m.radians(cam.hfov_deg) / 2.0)
    else:
        cam_data.type = "PANO"
        pano_attr = "panorama_type" if hasattr(cam_data, "panorama_type") else "cycles.panorama_type"
        if cam.projection == Projection.FISHEYE_EQUIDISTANT:
            _set_nested(cam_data, pano_attr, "FISHEYE_EQUIDISTANT")
            _set_nested(cam_data, "fisheye_fov", _m.radians(cam.fisheye_fov_deg or 180.0))
        else:
            _set_nested(cam_data, pano_attr, "FISHEYE_EQUISOLID")
            _set_nested(cam_data, "fisheye_lens", cam.fisheye_lens_mm)
            _set_nested(cam_data, "fisheye_fov", _m.radians(cam.fisheye_fov_deg or 180.0))
            cam_data.sensor_width = cam.sensor_mm


def set_blender_camera_object(cam: CameraSample):
    """Directly set the active camera object's lens + world transform (NO keyframing),
    for manual per-frame rendering (blender_render). Returns the camera object."""
    import bpy
    import mathutils
    scene = bpy.context.scene
    scene.render.resolution_x = cam.res_x
    scene.render.resolution_y = cam.res_y
    cam_obj = scene.camera
    _apply_lens(cam_obj.data, cam, cam.res_x, cam.res_y)
    # our OpenCV world->cam -> Blender cam->world (Blender cam looks down -Z, up +Y)
    T_wc = look_at_extrinsics(cam.location, cam.look_at, cam.roll_deg)
    T_cw = np.linalg.inv(T_wc)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])   # OpenCV -> Blender camera axes
    T_cw_blender = T_cw @ flip
    cam_obj.matrix_world = mathutils.Matrix(T_cw_blender.tolist())
    return cam_obj
