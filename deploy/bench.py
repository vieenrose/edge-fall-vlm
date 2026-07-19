"""M1 on-Pi benchmark harness: latency / RAM / thermal for a VLM backend.

Run on the actual RPi5. Measures per-inference latency, process RSS, and (on Pi) SoC
temperature via vcgencmd, over a soak of N inferences, and reports the sustainable
cadence + projected false-alarms/day given a measured specificity.

    python deploy/bench.py --cli ./llama-mtmd-cli --model smolvlm2-500m-q4.gguf \
        --mmproj mmproj.gguf --threads 4 --iters 200

Produces bench.json. Works with --stub for a dry structure test off-Pi.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

from deploy.vlm_backend import LlamaCppBackend, StubBackend


def soc_temp_c() -> float | None:
    try:
        out = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=5)
        return float(out.stdout.strip().split("=")[1].split("'")[0])
    except Exception:
        return None


def proc_rss_mb() -> float:
    """Peak RSS of spawned CHILD processes (the llama.cpp subprocess), in MB.
    RUSAGE_CHILDREN accumulates the max RSS of waited-for children — for the
    subprocess backend this is the model's real memory footprint, which is what the
    4GB Pi budget cares about (the Python parent is negligible)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0
    except Exception:
        return float("nan")


def run(backend, iters: int, strip_frames: int = 6, res: int = 384) -> dict:
    rng = np.random.default_rng(0)
    lat, temps = [], []
    t_start = time.time()
    for i in range(iters):
        strip = [(rng.random((res, res, 3)) * 255).astype("uint8") for _ in range(strip_frames)]
        t0 = time.perf_counter()
        backend.infer(strip)
        lat.append(time.perf_counter() - t0)
        tc = soc_temp_c()
        if tc is not None:
            temps.append(tc)
        if i % 20 == 0:
            print(f"  iter {i}/{iters} last={lat[-1]:.2f}s temp={tc}")
    wall = time.time() - t_start
    p50 = statistics.median(lat)
    p95 = sorted(lat)[int(0.95 * len(lat)) - 1]
    cadence_hz = 1.0 / p50 if p50 else 0.0
    report = {
        "iters": iters,
        "latency_s": {"p50": round(p50, 3), "p95": round(p95, 3),
                      "min": round(min(lat), 3), "max": round(max(lat), 3)},
        "sustainable_cadence_hz": round(cadence_hz, 3),
        "peak_rss_mb": round(proc_rss_mb(), 1),
        "thermal": {"max_c": max(temps) if temps else None,
                    "final_c": temps[-1] if temps else None,
                    "throttle_warn": bool(temps and max(temps) >= 80.0)},
        "wall_s": round(wall, 1),
    }
    return report


def project_false_alarms(cadence_hz: float, specificity: float, motion_gated_frac=0.1) -> float:
    """Effective inferences/day * (1-specificity), with motion-gating cutting the rate."""
    infer_per_day = cadence_hz * 3600 * 24 * motion_gated_frac
    return round(infer_per_day * (1 - specificity), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli"); ap.add_argument("--model"); ap.add_argument("--mmproj")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("bench.json"))
    ap.add_argument("--stub", action="store_true")
    args = ap.parse_args()

    if args.stub or not (args.cli and args.model):
        backend = StubBackend(lambda strip: (time.sleep(0.01),
                              {"status": "normal", "confidence": 0.1})[1])
        print("[stub] structural run (no llama.cpp)")
    else:
        backend = LlamaCppBackend(args.cli, args.model, args.mmproj, n_threads=args.threads)

    rep = run(backend, args.iters)
    rep["projected_false_alarms_per_day@spec0.99"] = project_false_alarms(
        rep["sustainable_cadence_hz"], 0.99)
    args.out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
