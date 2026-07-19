"""VLM inference backends for the RPi5 monitor.

LlamaCppBackend shells out to llama.cpp's multimodal CLI (llama-mtmd-cli) with a GGUF
model + mmproj, passing the strip frames as a tiled mosaic (llama.cpp mtmd handles one
image per call most reliably, so we compose the N-frame strip into a single grid image).

StubBackend is for tests. The GGUF path is run-gated (needs llama.cpp built + model on
the Pi); its command construction and output parsing are unit-testable here.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from training.eval import parse_answer  # reuse the tolerant JSON parser

PROMPT = ("You are a safety monitor. The image is a time-ordered grid of frames "
          "(top-left oldest). Report whether a person has fallen, fainted, is lying "
          "immobile, in distress, or is acting normally. Answer with JSON only: "
          '{"status": "...", "confidence": 0.0, "person_down": false}')


def compose_mosaic(strip: list[np.ndarray], cols: int | None = None) -> Image.Image:
    """Tile strip frames into one grid image (oldest top-left)."""
    imgs = [Image.fromarray(np.asarray(f).astype("uint8")) if not isinstance(f, Image.Image)
            else f for f in strip]
    n = len(imgs)
    cols = cols or int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    w = max(i.width for i in imgs)
    h = max(i.height for i in imgs)
    grid = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))
    for i, im in enumerate(imgs):
        grid.paste(im.resize((w, h)), ((i % cols) * w, (i // cols) * h))
    return grid


class StubBackend:
    def __init__(self, fn):
        self.fn = fn  # strip -> verdict dict

    def infer(self, strip):
        return self.fn(strip)


class LlamaCppBackend:
    """One-shot subprocess backend (reloads model each call — fine for the M1 bench, NOT
    for 24/7; use a persistent server in production, see M1_BENCHMARK.md).

    mode:
      "native" — pass frames as N ordered images (SmolVLM2 video reasoning; 64 tokens
                 PER frame, so prefill scales with N). Preferred for temporal fidelity.
      "mosaic" — tile frames into ONE grid image (constant 64 tokens regardless of N;
                 lower per-frame detail). Preferred when latency-bound.
    """

    def __init__(self, cli: str, model: str, mmproj: str, n_threads: int = 4,
                 n_predict: int = 96, mode: str = "native", extra_args: tuple = ()):
        self.cli = cli
        self.model = model
        self.mmproj = mmproj
        self.n_threads = n_threads
        self.n_predict = n_predict
        self.mode = mode
        self.extra_args = extra_args

    def build_cmd(self, image_paths: list[str]) -> list[str]:
        img_args = []
        for p in image_paths:
            img_args += ["--image", p]
        return [self.cli, "-m", self.model, "--mmproj", self.mmproj, *img_args,
                "-t", str(self.n_threads), "-n", str(self.n_predict), "--temp", "0",
                "-p", PROMPT, *self.extra_args]

    def infer(self, strip) -> dict:
        with tempfile.TemporaryDirectory() as d:
            if self.mode == "mosaic":
                p = Path(d) / "strip.png"
                compose_mosaic(strip).save(p)
                paths = [str(p)]
            else:
                paths = []
                for i, f in enumerate(strip):
                    p = Path(d) / f"f{i:02d}.png"
                    (f if isinstance(f, Image.Image) else Image.fromarray(np.asarray(f).astype("uint8"))).save(p)
                    paths.append(str(p))
            out = subprocess.run(self.build_cmd(paths), capture_output=True,
                                 text=True, timeout=120)
        return parse_answer(out.stdout)


if __name__ == "__main__":
    # mosaic + command construction are testable without llama.cpp
    strip = [np.full((64, 64, 3), i * 30, dtype="uint8") for i in range(6)]
    grid = compose_mosaic(strip)
    print("mosaic size:", grid.size, "(6 frames -> 3x2 grid of 64px)")
    assert grid.size == (192, 128)
    b = LlamaCppBackend("llama-mtmd-cli", "model.gguf", "mmproj.gguf", n_threads=4)
    cmd = b.build_cmd(["/tmp/a.png", "/tmp/b.png"])
    assert cmd.count("--image") == 2 and cmd[cmd.index("-t") + 1] == "4"
    print("native cmd images:", cmd.count("--image"))
    print("vlm_backend OK")
