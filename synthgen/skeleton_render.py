"""Stick-figure renderer: draw projected 2D keypoints to PNG (numpy/PIL, no Blender).

Purpose: close the motion -> camera -> IMAGE -> training-sample loop WITHOUT Blender, so
the whole stack is CI-testable on procedural data. Blender photoreal render swaps in for
production (render._render_strip); the training/deploy code consumes identical PNG strips
either way, so this is a faithful smoke-test substitute — not production imagery.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .rationale import JOINTS

# skeleton edges over the named JOINTS subset
_EDGES = [
    ("head", "neck"), ("neck", "l_shoulder"), ("neck", "r_shoulder"),
    ("neck", "pelvis"), ("pelvis", "l_hip"), ("pelvis", "r_hip"),
    ("l_hip", "l_ankle"), ("r_hip", "r_ankle"),
]
_JLIST = list(JOINTS)


def render_frame(uv: np.ndarray, valid: np.ndarray, res_x: int, res_y: int,
                 bg=(18, 18, 22), night: bool = False) -> Image.Image:
    """uv: (J,2) pixel coords for JOINTS order; valid: (J,) bool."""
    img = Image.new("RGB", (res_x, res_y), bg if not night else (8, 8, 10))
    d = ImageDraw.Draw(img)
    idx = {n: i for i, n in enumerate(_JLIST)}
    col = (200, 210, 220) if not night else (90, 95, 100)
    for a, b in _EDGES:
        ia, ib = idx[a], idx[b]
        if valid[ia] and valid[ib]:
            d.line([tuple(uv[ia]), tuple(uv[ib])], fill=col, width=max(2, res_x // 160))
    r = max(2, res_x // 120)
    for i in range(len(_JLIST)):
        if valid[i]:
            x, y = uv[i]
            d.ellipse([x - r, y - r, x + r, y + r], fill=(230, 120, 90))
    return img


def render_strip(kp_per_frame: list[dict], res_x: int, res_y: int, out_dir: Path,
                 night: bool = False) -> list[str]:
    """Write one PNG per strip frame from precomputed keypoints. Returns file paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for t, kp in enumerate(kp_per_frame):
        uv = np.array(kp["uv"], dtype=float)
        valid = np.array(kp["valid"], dtype=bool)
        img = render_frame(uv, valid, res_x, res_y, night=night)
        p = out_dir / f"f{t:02d}.png"
        img.save(p)
        paths.append(str(p))
    return paths


if __name__ == "__main__":
    import tempfile
    # a fake standing skeleton in a rectilinear-ish layout
    J = len(_JLIST)
    uv = np.zeros((J, 2)); valid = np.ones(J, bool)
    layout = {"head": (320, 120), "neck": (320, 180), "l_shoulder": (280, 190),
              "r_shoulder": (360, 190), "pelvis": (320, 300), "l_hip": (300, 305),
              "r_hip": (340, 305), "l_ankle": (300, 440), "r_ankle": (340, 440)}
    for n, (x, y) in layout.items():
        uv[_JLIST.index(n)] = (x, y)
    img = render_frame(uv, valid, 640, 480)
    p = Path(tempfile.mkdtemp()) / "skel.png"
    img.save(p)
    assert p.exists() and p.stat().st_size > 0
    print("skeleton_render OK ->", p)
