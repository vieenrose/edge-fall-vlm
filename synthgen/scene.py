"""Scene, lighting and domain randomization. Runs INSIDE `blenderproc run`.

Integration skeleton: the calls below are the BlenderProc API surface we use; the
Infinigen scene hookup and material library are wired in M2 build-order step 4.
Sampling logic (which values to pick) is pure and testable via sample_lighting().
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DomainRand, LightingCfg


@dataclass
class LightingSample:
    mode: str          # "day" | "dusk" | "night_ir"
    temp_k: float
    lux: float
    n_sources: int


def sample_lighting(rng, cfg: LightingCfg) -> LightingSample:
    is_night = rng.random() < cfg.night_ir_fraction
    if is_night:
        mode = "night_ir"
        temp_k = rng.uniform(3000, 4500)     # IR/near-mono look handled at material stage
        lux = rng.uniform(cfg.lux.lo, 120)
    else:
        mode = "day" if rng.random() < 0.6 else "dusk"
        temp_k = rng.uniform(cfg.color_temp_k.lo, cfg.color_temp_k.hi)
        lux = rng.uniform(200, cfg.lux.hi)
    n = int(rng.integers(cfg.n_sources[0], cfg.n_sources[1] + 1))
    return LightingSample(mode, temp_k, lux, n)


def apply_lighting(light: LightingSample):
    """Create BlenderProc lights matching the sample. Needs bpy/blenderproc."""
    import blenderproc as bproc
    for i in range(light.n_sources):
        l = bproc.types.Light()
        l.set_type("POINT")
        # crude lux->watt proxy; calibrate against a reference render in M2
        l.set_energy(float(light.lux) * 0.5)
        l.set_color(_temp_to_rgb(light.temp_k))
        l.set_location(bproc.sampler.shell(center=[0, 0, 1.2], radius_min=1.5,
                                           radius_max=4.0, elevation_min=10, elevation_max=85))
    if light.mode == "night_ir":
        # desaturate world / emulate IR at compositor stage (TODO: node setup)
        pass


def _temp_to_rgb(temp_k: float) -> list[float]:
    """Very rough blackbody -> linear RGB. Replace with a proper CCT curve if needed."""
    t = np.clip(temp_k, 2000, 6500)
    r = 1.0
    g = np.clip(0.55 + (t - 2000) / 9000, 0, 1)
    b = np.clip((t - 2500) / 4000, 0, 1)
    return [float(r), float(g), float(b)]


def load_scene_and_floor(rng, domain: DomainRand):
    """Load an Infinigen/hand-built room + randomized floor material. Needs bpy.

    Integration point: point this at your Infinigen export dir or a .blend room library,
    then randomize floor material from domain.floor_materials. Returns the scene objects.
    """
    import blenderproc as bproc  # noqa: F401
    raise NotImplementedError(
        "Wire to Infinigen room export / .blend library; randomize floor material "
        "from DomainRand.floor_materials. See scene.py docstring.")


def add_distractors(rng, domain: DomainRand):
    """Optionally add 0-N extra people/pets/objects for multi-occupant robustness."""
    n = int(rng.integers(domain.n_distractors[0], domain.n_distractors[1] + 1))
    return n  # TODO: spawn secondary SMPL-X actors / props; return their handles


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    cfg = LightingCfg()
    modes = [sample_lighting(rng, cfg).mode for _ in range(1000)]
    night_frac = modes.count("night_ir") / len(modes)
    print(f"night fraction over 1000 samples: {night_frac:.2f} (target ~{cfg.night_ir_fraction})")
    assert 0.24 < night_frac < 0.36
    print("scene OK")
