#!/usr/bin/env bash
# Set up the synthetic-data generation environment on gpu0 (RTX 5090).
# BlenderProc pulls its own matching Blender on first `blenderproc run`.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

echo "[1/5] python venv"
python3 -m venv .venv-synth
# shellcheck disable=SC1091
source .venv-synth/bin/activate
python -m pip install -U pip wheel

echo "[2/5] core deps"
pip install numpy blenderproc
# BlenderProc downloads a pinned Blender into its data dir on first run:
blenderproc quickstart || true   # verifies the Blender download works

echo "[3/5] pure-logic self-tests (no Blender needed)"
python -m synthgen.config
python -m synthgen.rationale
python -m synthgen.quality
python -m synthgen.scene
python scripts/dryrun.py

cat <<'EOF'

[4/5] MANUAL ASSETS (licences already cleared for detector training):
  - SMPL-X model + Meshcapade SMPL_blender_addon
        https://smpl-x.is.tue.mpg.de  /  https://github.com/Meshcapade/SMPL_blender_addon
    -> place model .npz/.pkl under assets/smplx/
  - LAFAN1 motion (Ubisoft)  https://github.com/ubisoft/ubisoft-laforge-animation-dataset
    -> retarget to SMPL-X params with scripts/convert_motion.py (TODO), write motions/*.npz
  - AMASS (ADL / negatives)  https://amass.is.tue.mpg.de
  - Infinigen-Indoors scenes https://github.com/princeton-vl/infinigen
        -> export rooms under assets/scenes/

[5/5] RUN (after assets are in place):
  source .venv-synth/bin/activate
  blenderproc run scripts/run_render.py --manifest data/motion_manifest.json --out data/synth_shards --seed 0

EOF
echo "install: core + pure-logic OK. Wire assets, then render."
