"""Sampling configuration for synthetic fall-data generation.

All perspective/lens/lighting ranges from SYNTHETIC_DATA_SPEC.md live here so the
whole pipeline has a single source of truth. Nothing in here imports bpy, so it can
be unit-tested with plain `python -m synthgen.config`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


class LabelClass(str, Enum):
    FALL = "fall"
    FAINT = "faint-collapse"
    IMMOBILE = "lying-immobile"
    DISTRESS = "distress"
    NORMAL = "normal"


class Projection(str, Enum):
    RECTILINEAR = "rectilinear"          # Blender PERSP
    WIDE_RECTILINEAR = "wide_rectilinear"  # Blender PERSP, high FOV
    FISHEYE_EQUIDISTANT = "fisheye_equidistant"  # PANO, r = f*theta
    FISHEYE_EQUISOLID = "fisheye_equisolid"      # PANO, r = 2f*sin(theta/2)


class MountArchetype(str, Enum):
    CEILING = "ceiling"
    HIGH_CORNER = "high_corner"
    WALL_MID = "wall_mid"
    LOW_SHELF = "low_shelf"
    BODY_WORN = "body_worn"


@dataclass(frozen=True)
class Range:
    lo: float
    hi: float

    def sample(self, rng) -> float:
        return float(rng.uniform(self.lo, self.hi))


# ---------------------------------------------------------------------------
# Camera placement, per mount archetype (height in metres above floor).
# distance/pitch/yaw/roll are the spherical placement around the subject.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MountProfile:
    height: Range
    distance: Range          # horizontal distance to subject centroid (m)
    pitch_deg: Range         # 0 = level, 90 = straight down (nadir)
    roll_deg: Range          # camera tilt; body-worn allows full range
    weight: float


MOUNTS: dict[MountArchetype, MountProfile] = {
    MountArchetype.CEILING:     MountProfile(Range(2.6, 3.2), Range(1.5, 4.0), Range(55, 90), Range(-15, 15), 0.30),
    MountArchetype.HIGH_CORNER: MountProfile(Range(2.0, 2.6), Range(2.5, 8.0), Range(20, 55), Range(-15, 15), 0.30),
    MountArchetype.WALL_MID:    MountProfile(Range(1.4, 2.0), Range(2.0, 6.0), Range(0, 30),  Range(-15, 15), 0.20),
    MountArchetype.LOW_SHELF:   MountProfile(Range(0.8, 1.5), Range(1.5, 5.0), Range(-10, 20), Range(-15, 15), 0.10),
    MountArchetype.BODY_WORN:   MountProfile(Range(0.9, 1.6), Range(1.5, 4.0), Range(-20, 40), Range(-180, 180), 0.10),
}

# subject-centroid position jitter (m, per axis) so the person is not perfectly centred
POSITION_JITTER = Range(-0.3, 0.3)


# ---------------------------------------------------------------------------
# Lens / projection sampling.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LensProfile:
    projection: Projection
    hfov_deg: Range | None = None   # for rectilinear paths
    fisheye_fov_deg: Range | None = None  # for PANO fisheye (total FOV)
    fisheye_lens_mm: Range | None = None  # equisolid physical lens
    weight: float = 0.0


LENSES: Sequence[LensProfile] = (
    LensProfile(Projection.RECTILINEAR,        hfov_deg=Range(45, 90),  weight=0.35),
    LensProfile(Projection.WIDE_RECTILINEAR,   hfov_deg=Range(90, 120), weight=0.20),
    LensProfile(Projection.FISHEYE_EQUIDISTANT, fisheye_fov_deg=Range(150, 200), weight=0.20),
    LensProfile(Projection.FISHEYE_EQUISOLID,   fisheye_lens_mm=Range(8, 16),    weight=0.20),
    # 0.05 reserved for calibrated Kannala-Brandt via the Isaac-Sim escalation path.
)

# optical-centre offset as fraction of frame (fisheye circles are rarely centred)
OPTICAL_CENTER_OFFSET = Range(-0.05, 0.05)


# ---------------------------------------------------------------------------
# Render / output.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RenderCfg:
    short_side_px: Range = Range(512, 768)  # rendered short side; matches VLM input
    fps: int = 30
    strip_frames: int = 6          # frames per VLM training strip (matches M1 / deployment)
    strip_span_s: float = 3.5      # wall-clock span the strip covers
    cycles_samples: int = 64       # denoised; keep low, VLM never sees fine detail
    denoise: bool = True
    # cameras rendered per motion clip
    cams_per_clip: Range = Range(8, 16)


# ---------------------------------------------------------------------------
# Scene / lighting / domain randomization.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LightingCfg:
    color_temp_k: Range = Range(2500, 6500)
    lux: Range = Range(20, 1000)
    n_sources: tuple[int, int] = (1, 4)
    night_ir_fraction: float = 0.30   # ~30% of samples are night / near-mono IR


@dataclass(frozen=True)
class DomainRand:
    occlusion_frac: Range = Range(0.0, 0.40)   # furniture between cam and subject
    n_distractors: tuple[int, int] = (0, 3)    # extra people/pets/objects
    floor_materials: tuple[str, ...] = ("tile", "wood", "carpet", "concrete")
    add_sensor_noise: bool = True
    add_motion_blur: bool = True


# ---------------------------------------------------------------------------
# Class mix — negatives MUST outnumber falls to control false-alarm rate.
# ---------------------------------------------------------------------------
CLASS_MIX: dict[LabelClass, float] = {
    LabelClass.FALL: 0.28,
    LabelClass.FAINT: 0.09,
    LabelClass.IMMOBILE: 0.07,
    LabelClass.DISTRESS: 0.09,
    LabelClass.NORMAL: 0.47,   # hard negatives dominate
}


@dataclass(frozen=True)
class PipelineCfg:
    render: RenderCfg = field(default_factory=RenderCfg)
    lighting: LightingCfg = field(default_factory=LightingCfg)
    domain: DomainRand = field(default_factory=DomainRand)
    seed: int = 0
    # coverage guarantee: every motion clip must yield at least one of each below
    require_topdown_fisheye: bool = True
    require_level_rectilinear: bool = True


DEFAULT = PipelineCfg()


if __name__ == "__main__":
    # sanity: weights sum to ~1
    assert abs(sum(m.weight for m in MOUNTS.values()) - 1.0) < 1e-6, "mount weights"
    assert abs(sum(l.weight for l in LENSES) - 0.95) < 1e-6, "lens weights (0.05 reserved)"
    assert abs(sum(CLASS_MIX.values()) - 1.0) < 1e-6, "class mix"
    print("config OK; classes:", [c.value for c in LabelClass])
