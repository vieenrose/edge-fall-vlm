# Assets

CC0 (public domain) textures and HDRIs from [Poly Haven](https://polyhaven.com), used to
replace procedural checker patterns / point-light approximations with real photographed
materials and indoor lighting environments in `synthgen/blender_render.py`
(`_real_texture_mat`, `apply_hdri_lighting`). No attribution required by license, but
credit is good practice: all files sourced from polyhaven.com, downloaded via their public
API (`api.polyhaven.com`).

- `textures/`: fabric/wood/carpet diffuse maps (1k JPG) — patterned ones (checkered
  blanket, gingham, boucle, jacquard) specifically target the "busy patterned surface can
  visually camouflage a person" failure mode found via a real missed-fall clip.
- `hdris/`: indoor environment maps (1k EXR) — bedroom/hospital-room/lounge/attic, for
  realistic warm/dim indoor lighting instead of procedural point lights.
