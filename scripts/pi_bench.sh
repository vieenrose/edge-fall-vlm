#!/usr/bin/env bash
# M1 benchmark on the ACTUAL Raspberry Pi 5 (4GB). Measures per-inference latency, RAM,
# and SoC temperature under a sustained soak, for the candidate GGUF VLMs.
#
# Prereqs on the Pi:
#   - build llama.cpp: cmake -B build -DGGML_NATIVE=ON && cmake --build build -j4
#       targets: llama-mtmd-cli (bench) and llama-server (production persistent path)
#   - download models (same as x86): models/smolvlm2-500m/*.gguf, models/smolvlm-256m/*.gguf
#   - active cooler attached (throttle starts at 80C)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CLI=${CLI:-./llama.cpp/build/bin/llama-mtmd-cli}
THREADS=${THREADS:-4}
ITERS=${ITERS:-100}

command -v vcgencmd >/dev/null || echo "WARN: vcgencmd not found — no thermal readings"

for TAG in smolvlm2-500m smolvlm-256m; do
  W=$(ls models/$TAG/*Instruct*Q8_0.gguf models/$TAG/*Video-Instruct-Q8_0.gguf 2>/dev/null | grep -v mmproj | head -1)
  MM=$(ls models/$TAG/mmproj*Q8_0.gguf | head -1)
  [ -f "$W" ] || { echo "skip $TAG (no weights)"; continue; }
  echo "=== $TAG  weights=$(basename "$W") ==="
  echo "idle temp: $(vcgencmd measure_temp 2>/dev/null || echo n/a)"
  CLI=$CLI python3 -m deploy.bench --cli "$CLI" --model "$W" --mmproj "$MM" \
      --threads "$THREADS" --iters "$ITERS" --out "bench_pi_${TAG}.json"
  echo "final temp: $(vcgencmd measure_temp 2>/dev/null || echo n/a)"
  echo "throttled?: $(vcgencmd get_throttled 2>/dev/null || echo n/a)  (0x0 = never throttled)"
done

echo
echo "NOTE: deploy.bench uses subprocess-per-call (reloads model each time) => latency"
echo "includes model LOAD. For the production 24/7 number, run the persistent server:"
echo "  ./llama.cpp/build/bin/llama-server -m <weights> --mmproj <mmproj> -t $THREADS --port 8080"
echo "and time repeated /v1/chat/completions calls with image_url content."
