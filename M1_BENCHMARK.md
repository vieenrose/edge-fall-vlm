# M1 — On-Device VLM Benchmark (first results)

Goal: pin the base model, the frame-strip format, and the inference architecture for the
RPi5 4GB deployment. Measured this session on an **x86 proxy** (AMD Ryzen 9950X3D,
**capped to 4 threads** to emulate the Pi's 4 usable cores) with real GGUF models via
`llama.cpp` `llama-mtmd-cli`. Pi5 (Cortex-A76 @2.4GHz) numbers are projected — build
`llama-server` and run `scripts/pi_bench.sh` on the actual Pi to replace them.

## Models tested (Q8_0 GGUF)
| Model | Weights | mmproj | Peak RSS (real) | Fits 4GB? |
|---|---|---|---|---|
| SmolVLM2-500M-Video | 417 MB | 104 MB | **787 MB** | ✅ big headroom |
| SmolVLM-256M | 167 MB | 99 MB | **663 MB** | ✅ |

RSS is the child-process resident set (the number that matters for the 4GB budget), with
room left for the OS, camera pipeline, and page cache.

## Latency — compute only (excludes model load), Ryzen @ 4 threads
Each frame = **64 image tokens** in SmolVLM2 at ~384px (no image splitting).

| Config | Image tokens | Prefill | + decode (~60 tok) | ≈ per-inference compute |
|---|---|---|---|---|
| 500M, mosaic (6→1 img) | 64 | ~90 ms | ~220 ms | **~0.3 s** |
| 500M, native 6 frames | 384 | ~166 ms | ~220 ms | **~0.4 s** |
| 256M, mosaic | 64 | ~90 ms | ~180 ms | **~0.27 s** |
| 256M, native 6 frames | 384 | ~168 ms | ~180 ms | **~0.35 s** |

## The load-vs-compute finding (architecture-critical)
Subprocess-per-frame **reloads the model every call**: measured wall-clock was ~1.1 s of
which **~0.8–1.0 s is just model load**, only ~0.3 s is inference. On the Pi, load will be
~3–5 s.

➡ **24/7 deployment MUST keep the model resident** (persistent `llama-server` or a
long-lived `llama-cpp-python`/mtmd process). `deploy/monitor.py` already assumes an
injected long-lived backend; `deploy.vlm_backend.LlamaCppBackend` is the one-shot bench
tool, not the production path. Building `llama-server` is the first M1 follow-up.

## Pi5 projection (persistent process, 4 threads)
Zen5→A76 per-core factor ≈ 3–5×:
| Config | Projected Pi5 compute |
|---|---|
| 500M native 6-frame | ~1.2–2.0 s |
| 500M mosaic | ~0.9–1.5 s |
| 256M native 6-frame | ~1.0–1.7 s |

All comfortably support the design cadence of **0.5–1 Hz** (motion-gated). Thermal
behavior under sustained load is the remaining unknown — only `scripts/pi_bench.sh` on
real hardware answers it (throttle starts at 80 °C; active cooler mandatory).

## Decisions
1. **Base model: SmolVLM2-500M-Video.** Video-native, only ~0.1–0.3 s slower than 256M,
   fits RAM easily. Keep **256M as the thermal/latency fallback** if the Pi soak shows
   throttling.
2. **Strip format: native multi-frame (N≈6) by default.** SmolVLM2 is a video model —
   use ordered frames for true temporal reasoning. Prefill scales with N (64 tok/frame);
   if the Pi soak is latency-bound, switch to **mosaic** (constant 64 tokens) via
   `LlamaCppBackend(mode="mosaic")`. Both are implemented.
3. **Persistent server, not subprocess-per-frame.** Non-negotiable for 24/7.
4. **fps-robustness is built in** (see below), so effective-fps drift from thermal
   throttling does not break detection.

## fps-robustness (answers "robust to fps variation?": yes)
Two mechanisms, both implemented and tested:
- **Device — time-based strip** (`deploy/monitor.StripBuffer`): the strip always spans a
  fixed wall-clock window (~3.5 s) with N frames chosen by timestamp, not frame index.
  Verified: 5 fps and 15 fps capture both yield 6 frames over ~3.5 s. So whether the
  camera fps is fixed OR the Pi throttles and the rate drifts, the model sees a consistent
  temporal window.
- **Training — temporal augmentation** (`training/dataset.temporal_augment`): randomizes
  frame count (2–8) and spacing with dropout per sample, so the model reads a fall from
  body-STATE change rather than a fixed frame rhythm. Wired into the SFT collator
  (`fps_augment=True`).
- Plus **alert persistence** (`AlertState`): a fallen/immobile person stays detectable at
  any cadence, so even a missed fall *motion* is caught by the persistent down *state*.

## Winning 2.2B model — GGUF deployment (validated)
The deployable model (`runs/sft-2b-real`, 0.90 real recall / 1.0 real spec) exported to
GGUF and benchmarked on the 4-thread proxy:

| Quant | LM size | +mmproj = total | Post-quant accuracy | On-proxy |
|---|---|---|---|---|
| F16 | 3.6GB | 4.5GB | perfect | exceeds 4GB ✗ |
| Q8_0 | 1.9GB | 2.8GB | fall✓ normal✓ | fits, tighter |
| **Q6_K (recommended)** | 1.4GB | **2.2GB** | **fall✓ normal✓** | **1.83GB RSS, ~1.2s compute** |
| Q5_K_M | 1.3GB | 2.2GB | degrading ✗ | — |
| Q4_K_M | 1.1GB | 2.0GB | **BREAKS fine-tune** ✗ | — |

- **Key finding: Q4 silently breaks the fine-tuned 2.2B** (empty/garbled output); Q6_K is
  the sweet spot — accurate AND fits 4GB with ~2GB headroom for OS+camera. Always
  post-quant eval a fine-tuned VLM before trusting the GGUF.
- Q6_K: 1.83GB peak RSS, ~1.2s compute (proxy) -> **~4-6s/inference projected on RPi5 A76**
  — acceptable at the low-duty-cycle (motion-gated ~0.5Hz) cadence.
- mmproj kept f16 (0.83GB) for vision accuracy; could Q8 it to save ~0.4GB if needed.
- Files: export/2b/model-Q6_K.gguf + mmproj-f16.gguf. Use persistent server on Pi (avoid
  per-call reload); re-bench on real Pi hardware for thermal + true A76 latency.

## Reproduce
```
# x86 proxy (this session)
python -m deploy.bench --cli ~/llama.cpp/build/bin/llama-mtmd-cli \
  --model models/smolvlm2-500m/SmolVLM2-500M-Video-Instruct-Q8_0.gguf \
  --mmproj models/smolvlm2-500m/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf \
  --threads 4 --iters 20

# real Pi
bash scripts/pi_bench.sh    # add vcgencmd thermal, persistent server
```
