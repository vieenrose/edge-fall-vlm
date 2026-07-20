"""Blender photoreal-ish renderer: volumetric mannequin on our joint rig + native fisheye.

Runs INSIDE `blenderproc run` (provides bpy + blenderproc). Replaces skeleton_render's
2D stick-figure with a 3D textured body (spheres at joints, capsules along bones), lit,
shadowed, occluded, in a randomized room, seen through a sampled camera/lens including
TRUE panoramic fisheye. Fully un-gated: no SMPL-X / mocap assets — bodies are built from
primitives driven by the same 3D joint trajectories the rest of the pipeline already uses.

The mannequin is a stand-in that is far more readable to a pretrained VLM than stick
figures (volume, shading, cast shadows, self-occlusion). Swap `build_body` for a MakeHuman
/ Mixamo mesh on the same rig later without touching the camera / GT / label / eval code.

Contract: render_view(joints_at_frames, cam, lighting, out_dir) -> list[png_paths].
Camera GT keypoints still come from cameras.project_points (correct under fisheye).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .config import Projection
from .rationale import JOINTS

# Limb capsules as (pointA_key, pointB_key, radius_m). Keys may be real joints OR derived
# points computed in pose_body (l_hand/r_hand hang from shoulders along the torso-down
# axis, since the 9-joint procedural skeleton has no arm joints). No shoulder->hip struts
# (those made the old tripod look); arms hang, legs are thick, torso is a volume.
_BONES = [
    ("neck", "head", 0.06),
    ("l_shoulder", "r_shoulder", 0.05),           # shoulder bar
    ("l_shoulder", "l_hand", 0.045), ("r_shoulder", "r_hand", 0.045),  # hanging arms
    ("pelvis", "l_hip", 0.075), ("pelvis", "r_hip", 0.075),
    ("l_hip", "l_ankle", 0.07), ("r_hip", "r_ankle", 0.07),            # thick legs
]
_JOINT_R = {"head": 0.11, "l_hand": 0.05, "r_hand": 0.05,
            "l_ankle": 0.06, "r_ankle": 0.06}
_DEFAULT_JOINT_R = 0.045
# spheres to actually draw (skip pelvis/neck/shoulders — covered by torso volume)
_DRAW_SPHERES = ("head", "l_hand", "r_hand", "l_ankle", "r_ankle")


def _bpy():
    import bpy
    return bpy


def _mat(name, rgb, rough=0.7):
    bpy = _bpy()
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    return m


def _derived_points(j: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Add arm (elbow/hand) and knee points so limbs read naturally, even though the
    procedural skeleton has only 9 joints. Arms hang along the torso-down axis with a
    slight forward elbow bend; knees bend forward between hip and ankle."""
    pelvis, neck, head = np.asarray(j["pelvis"]), np.asarray(j["neck"]), np.asarray(j["head"])
    down = pelvis - neck
    if np.linalg.norm(down) < 0.15:
        # pelvis and neck can nearly coincide (e.g. mid-fall torso compression, or for
        # some fall-direction/starting-pose combinations even in the settled end pose --
        # head and neck already converge to the same point when lying flat by design, so
        # this isn't just transient). Normalizing a near-zero vector sends the derived
        # hand/elbow shooting off arbitrarily (a degenerate spike in the rendered mesh).
        down = pelvis - head            # first fallback: a longer, more stable reference
        if np.linalg.norm(down) < 0.15:
            down = np.array([0.0, 0.0, -1.0])   # last resort: arms hang straight down
    down = down / (np.linalg.norm(down) + 1e-6)
    # a horizontal 'forward' roughly perpendicular to the torso, for limb bend
    fwd = np.cross(down, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(fwd) < 1e-3:
        fwd = np.array([1.0, 0.0, 0.0])
    fwd = fwd / (np.linalg.norm(fwd) + 1e-6)
    out = dict(j)
    arm = 0.30
    for side in ("l", "r"):
        sh = np.asarray(j[f"{side}_shoulder"])
        hand = sh + down * (arm * 1.8)
        elbow = sh + down * arm + fwd * 0.06
        out[f"{side}_elbow"] = elbow
        out[f"{side}_hand"] = hand
        hip = np.asarray(j[f"{side}_hip"])
        ankle = np.asarray(j[f"{side}_ankle"])
        out[f"{side}_knee"] = (hip + ankle) / 2.0 + fwd * 0.05
    return out


# Skin-mesh graph: ordered vertices (joints + derived) and edges (bones), with a
# per-vertex skin radius. Produces a single CONNECTED organic body via the Skin modifier,
# driven purely by vertex world positions (no armature/rigging).
_VERTS = ["pelvis", "neck", "head", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
          "l_hand", "r_hand", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
_VIDX = {n: i for i, n in enumerate(_VERTS)}
_MESH_EDGES = [("pelvis", "neck"), ("neck", "head"),
               ("neck", "l_shoulder"), ("neck", "r_shoulder"),
               ("l_shoulder", "l_elbow"), ("l_elbow", "l_hand"),
               ("r_shoulder", "r_elbow"), ("r_elbow", "r_hand"),
               ("pelvis", "l_hip"), ("pelvis", "r_hip"),
               ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
               ("r_hip", "r_knee"), ("r_knee", "r_ankle")]
_SKIN_R = {"pelvis": 0.17, "neck": 0.13, "head": 0.115,
           "l_shoulder": 0.11, "r_shoulder": 0.11, "l_elbow": 0.06, "r_elbow": 0.06,
           "l_hand": 0.05, "r_hand": 0.05, "l_hip": 0.12, "r_hip": 0.12,
           "l_knee": 0.08, "r_knee": 0.08, "l_ankle": 0.06, "r_ankle": 0.06}

_PALETTE = [(0.2, 0.35, 0.6), (0.6, 0.25, 0.25), (0.3, 0.5, 0.3),
            (0.5, 0.45, 0.2), (0.4, 0.3, 0.5)]


def build_body(idx=0, cloth_rgb=None):
    """Create one CONNECTED skin-mesh human. Returns a handle with the mesh object and
    its vertex map, re-posed per frame by pose_body. idx varies clothing colour so
    multiple people are visually distinct."""
    bpy = _bpy()
    verts = [(0.0, 0.0, 0.0)] * len(_VERTS)
    edges = [(_VIDX[a], _VIDX[b]) for a, b in _MESH_EDGES]
    me = bpy.data.meshes.new(f"body{idx}")
    me.from_pydata(verts, edges, [])
    me.update()
    obj = bpy.data.objects.new(f"Body{idx}", me)
    bpy.context.scene.collection.objects.link(obj)
    skin = obj.modifiers.new("Skin", "SKIN")
    sub = obj.modifiers.new("Subsurf", "SUBSURF")
    sub.levels = sub.render_levels = 1
    # per-vertex skin radius + root at pelvis
    sv = me.skin_vertices[0].data
    for n, i in _VIDX.items():
        r = _SKIN_R[n]
        sv[i].radius = (r, r)
    sv[_VIDX["pelvis"]].use_root = True
    rgb = cloth_rgb or _PALETTE[idx % len(_PALETTE)]
    me.materials.append(_mat(f"cloth{idx}", rgb, 0.75))
    return {"obj": obj, "mesh": me}


def build_bodies(n: int):
    return [build_body(idx=i) for i in range(n)]


def pose_body(body, joints_t: dict[str, np.ndarray]):
    """Update the skin-mesh vertex positions for one frame. joints_t[name]->(3,)."""
    p = _derived_points(joints_t)
    me = body["mesh"]
    for n, i in _VIDX.items():
        me.vertices[i].co = tuple(p[n])
    me.update()


def _quat_from_z(z):
    import mathutils
    return mathutils.Vector((0, 0, 1)).rotation_difference(mathutils.Vector(tuple(z)))


def setup_scene_and_floor(rng, floor_rgb=None):
    bpy = _bpy()
    # floor plane
    bpy.ops.mesh.primitive_plane_add(size=12, location=(0, 0, 0))
    floor = bpy.context.active_object
    rgb = floor_rgb or (rng.uniform(0.15, 0.6),) * 3
    floor.data.materials.append(_mat("floor", rgb, rng.uniform(0.4, 0.95)))
    return floor


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
TEXTURE_FILES = sorted((ASSETS_DIR / "textures").glob("*.jpg")) if (ASSETS_DIR / "textures").is_dir() else []
HDRI_FILES = sorted((ASSETS_DIR / "hdris").glob("*.exr")) if (ASSETS_DIR / "hdris").is_dir() else []


def _real_texture_mat(name, tex_path, scale=1.0, rough=0.85):
    """Material using a REAL CC0 photo texture (fabric/wood/carpet, downloaded from
    Poly Haven) instead of a procedural pattern -- a genuinely photoreal stand-in for
    the "busy patterned blanket/upholstery" camouflage case a real missed fall exposed,
    rather than a synthetic checker approximation of it."""
    bpy = _bpy()
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    img = bpy.data.images.load(str(tex_path), check_existing=True)
    tex_node = nt.nodes.new("ShaderNodeTexImage")
    tex_node.image = img
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (scale, scale, scale)
    coord = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], tex_node.inputs["Vector"])
    nt.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = rough
    return m


def apply_hdri_lighting(rng, strength=None):
    """Light the scene with a REAL indoor HDRI environment (Poly Haven, CC0) instead of
    the procedural point/area lights -- targets the warm/dim real-CCTV-footage look a
    real missed fall exposed (procedural point lights + color-temp tint is a coarser
    approximation of this than an actual photographed indoor lighting environment).
    Returns True if an HDRI was applied, False if none are available (falls back to the
    caller's existing procedural lighting)."""
    if not HDRI_FILES:
        return False
    bpy = _bpy()
    path = HDRI_FILES[int(rng.integers(len(HDRI_FILES)))]
    world = bpy.context.scene.world
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    bg = nt.nodes.new("ShaderNodeBackground")
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(str(path), check_existing=True)
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value = (0, 0, rng.uniform(0, 2 * math.pi))
    coord = nt.nodes.new("ShaderNodeTexCoord")
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = strength if strength is not None else rng.uniform(0.4, 1.4)
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    return True


def _checker_mat(name, rgb_a, rgb_b, scale, rough=0.8):
    """Procedural checker material -- a cheap, asset-free way to approximate a "busy
    patterned" surface (patterned blanket/upholstery/rug) that can visually camouflage a
    person against it, without needing external texture/fabric image files."""
    bpy = _bpy()
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    checker = nt.nodes.new("ShaderNodeTexChecker")
    checker.inputs["Color1"].default_value = (*rgb_a, 1.0)
    checker.inputs["Color2"].default_value = (*rgb_b, 1.0)
    checker.inputs["Scale"].default_value = scale
    nt.links.new(checker.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = rough
    return m


def add_clutter_props(rng, subject_positions, n_range=(1, 4), near_prob=0.6):
    """Spawn simple furniture-like occluders (boxes standing in for a low table, chair,
    nightstand, or a patterned blanket/throw) around the scene, some deliberately placed
    near a subject's resting position so they partially occlude the person from camera --
    the real-world failure mode found in a held-out clip (person ends up tucked behind
    furniture, partially hidden, blended into a busy patterned blanket). Not tied to any
    specific clip's exact geometry (that would be cheating/overfitting to one sample) --
    randomized size/position/material every scene so the model has to generalize to
    "danger despite clutter/occlusion/camouflage" broadly, not memorize one layout.

    subject_positions: list of (x, y) floor positions (e.g. each person's final resting
    spot) to bias clutter placement toward -- empty list means fully random placement.
    Returns the list of created object NAMES (for remove_clutter_props next scene --
    raw object references can go stale between scenes under BlenderProc)."""
    bpy = _bpy()
    n = int(rng.integers(n_range[0], n_range[1] + 1))
    objs = []
    for i in range(n):
        near_subject = subject_positions and rng.random() < near_prob
        if near_subject:
            cx, cy = subject_positions[int(rng.integers(len(subject_positions)))]
            ox = cx + rng.uniform(-0.5, 0.5)
            oy = cy + rng.uniform(-0.5, 0.5)
        else:
            ox, oy = rng.uniform(-2.5, 2.5), rng.uniform(-2.5, 2.5)
        # Blender's default cube is 2x2x2 (extends -1..1), so `scale` doubles these --
        # halved here so the RENDERED object matches the intended real-world size range.
        kind = rng.choice(["low_table", "chair", "box", "blanket"])
        if kind == "low_table":
            sx, sy, sz = rng.uniform(0.175, 0.3), rng.uniform(0.175, 0.3), rng.uniform(0.175, 0.25)
        elif kind == "chair":
            sx, sy, sz = rng.uniform(0.125, 0.2), rng.uniform(0.125, 0.2), rng.uniform(0.2, 0.425)
        elif kind == "blanket":  # flat, wide, low -- classic camouflage-by-clutter case
            sx, sy, sz = rng.uniform(0.25, 0.5), rng.uniform(0.25, 0.5), rng.uniform(0.015, 0.04)
        else:  # box / nightstand
            sx, sy, sz = rng.uniform(0.125, 0.225), rng.uniform(0.125, 0.225), rng.uniform(0.125, 0.275)
        bpy.ops.mesh.primitive_cube_add(location=(ox, oy, sz))
        obj = bpy.context.active_object
        obj.scale = (sx, sy, sz)
        bpy.ops.object.transform_apply(scale=True)
        patterned = kind == "blanket" or rng.random() < 0.35   # patterned furniture too
        if TEXTURE_FILES and rng.random() < 0.7:
            tex_path = TEXTURE_FILES[int(rng.integers(len(TEXTURE_FILES)))]
            mat = _real_texture_mat(f"clutter{i}", tex_path, scale=rng.uniform(0.6, 2.5))
        elif patterned:
            rgb_a = tuple(rng.uniform(0.2, 0.9, size=3))
            rgb_b = tuple(rng.uniform(0.2, 0.9, size=3))
            mat = _checker_mat(f"clutter{i}", rgb_a, rgb_b, rng.uniform(4, 16))
        else:
            rgb = tuple(rng.uniform(0.15, 0.7, size=3))
            mat = _mat(f"clutter{i}", rgb, rng.uniform(0.3, 0.95))
        obj.data.materials.append(mat)
        objs.append(obj.name)
    return objs


def remove_clutter_props(obj_names):
    """Delete the previous scene's clutter objects (by name -- BlenderProc's internal
    bookkeeping between scenes can invalidate raw Python object references, so we look
    each one up fresh rather than holding onto the object handle) before spawning the
    next scene's; the Blender process is reused across scenes (not restarted), so stale
    clutter would otherwise accumulate. Silently skips names that no longer resolve."""
    bpy = _bpy()
    for name in obj_names:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)


def reset_world_background():
    """Clear the World node tree back to a plain Background->Output pair. Needed before
    both setup_lighting and apply_hdri_lighting, since the Blender process is reused
    across scenes -- without this, a previous scene's HDRI environment texture would
    otherwise linger as the background for a later non-HDRI-lit scene (only its
    Strength would change, not its Color, since setup_lighting only touched Strength)."""
    bpy = _bpy()
    world = bpy.context.scene.world
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    bg = nt.nodes.new("ShaderNodeBackground")
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    return bg


def setup_lighting(rng, light):
    """light: scene.LightingSample. Creates sun/area lights matching mode/lux/temp.
    Returns the list of created light objects (for cleanup next scene)."""
    bpy = _bpy()
    bg = reset_world_background()
    amb = 0.02 if light.mode == "night_ir" else rng.uniform(0.1, 0.5)
    bg.inputs["Strength"].default_value = amb
    n = light.n_sources
    created = []
    for _ in range(max(1, n)):
        bpy.ops.object.light_add(type="AREA",
                                 location=(rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(1.8, 3.0)))
        L = bpy.context.active_object
        L.data.energy = float(light.lux) * rng.uniform(0.3, 0.8)
        L.data.size = rng.uniform(0.5, 2.0)
        # warm/cool tint from temp
        t = (light.temp_k - 2500) / 4000
        L.data.color = (1.0, 0.75 + 0.2 * t, 0.5 + 0.5 * t)
        created.append(L.name)
    return created


def remove_lights(light_names):
    """By name, same rationale as remove_clutter_props -- raw object references can go
    stale between scenes under BlenderProc."""
    bpy = _bpy()
    for name in light_names:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)


def configure_render(cycles_samples=48, denoise=True):
    bpy = _bpy()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    try:
        scene.cycles.device = "GPU"      # gpu0 (set CUDA_VISIBLE_DEVICES=0 outside)
    except Exception:
        pass
    scene.cycles.samples = cycles_samples
    scene.cycles.use_denoising = denoise
    scene.render.image_settings.file_format = "PNG"


def apply_camera(cam):
    """Set the active Blender camera (lens + world transform) directly, no keyframing,
    for manual per-frame rendering."""
    from .cameras import set_blender_camera_object
    set_blender_camera_object(cam)


def render_view(body, joints_seq, strip_idx, cam, out_dir: Path) -> list[str]:
    """Single-person convenience wrapper (kept for back-compat)."""
    return render_scene([body], [joints_seq], strip_idx, cam, out_dir)


def render_scene(bodies, people_joints, strip_idx, cam, out_dir: Path) -> list[str]:
    """Render the strip for a multi-person scene from one camera.

    bodies: pool of skin-mesh bodies (>= len(people_joints)); extras are hidden.
    people_joints: list of per-person joint sequences (name->(T,3)), already placed in
    world (origin/yaw applied). One body per person, posed independently each frame.
    """
    bpy = _bpy()
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_camera(cam)
    npeople = len(people_joints)
    for i, b in enumerate(bodies):
        b["obj"].hide_render = (i >= npeople)
    paths = []
    for k, t in enumerate(strip_idx):
        for b, pj in zip(bodies, people_joints):
            pose_body(b, {j: pj[j][t] for j in JOINTS})
        p = out_dir / f"f{k:02d}.png"
        bpy.context.scene.render.filepath = str(p)
        bpy.ops.render.render(write_still=True)
        paths.append(str(p))
    return paths
