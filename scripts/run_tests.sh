#!/usr/bin/env bash
# All Blender-free self-tests. Proves the data-gen -> train-plumbing -> eval -> deploy
# chain end to end without gated assets or a Pi. GPU touches gpu0 only.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "== synthgen pure-logic =="
python3 -m synthgen.config
python3 -m synthgen.rationale
python3 -m synthgen.quality
python3 -m synthgen.scene
python3 -m synthgen.skeleton_render

echo "== full pipeline glue (no Blender) =="
python3 scripts/dryrun.py | tail -1

echo "== bootstrap data (motions -> images -> samples) =="
python3 scripts/convert_motion.py --out data/bootstrap --scale 2 | tail -1
python3 scripts/bootstrap_dataset.py --manifest data/bootstrap/motion_manifest.json \
    --out data/bootstrap/shards --k 6 | tail -1

echo "== training plumbing =="
python3 training/dataset.py data/bootstrap/shards/samples.jsonl | tail -1
python3 -m training.eval | tail -1
python3 -m training.export_gguf | tail -1

echo "== deploy logic =="
python3 deploy/monitor.py | tail -1
python3 -m deploy.vlm_backend | tail -1
python3 -m deploy.bench --stub --iters 10 >/dev/null && echo "bench OK"

echo "ALL TESTS PASSED"
