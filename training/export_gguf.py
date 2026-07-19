"""Stage D: export a fine-tuned SmolVLM2 to GGUF + mmproj for llama.cpp on the Pi.

Small VLMs lose more to quantization, so the POST-QUANT eval is the only score that
counts (see RESEARCH_PLAN). This wraps llama.cpp's converters; it does not reimplement
them. It emits the exact commands and (optionally) runs them if llama.cpp is on PATH.

    python training/export_gguf.py --model runs/sft-bootstrap --llama-cpp ../llama.cpp \
        --quant Q4_K_M --out export/

Then re-run training/eval.py through deploy/vlm_backend.LlamaCppBackend on the SAME test
split to measure post-quant sensitivity/specificity BEFORE trusting the model.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def commands(model_dir: Path, llama_cpp: Path, quant: str, out: Path) -> list[list[str]]:
    """llama.cpp multimodal export: LM to GGUF (+quantize), vision encoder to mmproj.
    Keep the vision encoder at higher precision (Q8/F16) — it degrades accuracy most."""
    out.mkdir(parents=True, exist_ok=True)
    conv = llama_cpp / "convert_hf_to_gguf.py"
    f16 = out / "model-f16.gguf"
    quantized = out / f"model-{quant}.gguf"
    quant_bin = llama_cpp / "build" / "bin" / "llama-quantize"
    return [
        # 1. language model -> f16 gguf
        ["python", str(conv), str(model_dir), "--outfile", str(f16), "--outtype", "f16"],
        # 2. quantize LM weights
        [str(quant_bin), str(f16), str(quantized), quant],
        # 3. vision projector (mmproj) at f16 — keep precision on the encoder
        ["python", str(conv), str(model_dir), "--outfile", str(out / "mmproj-f16.gguf"),
         "--mmproj", "--outtype", "f16"],
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--llama-cpp", type=Path, required=True)
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--out", type=Path, default=Path("export"))
    ap.add_argument("--run", action="store_true", help="actually execute (else just print)")
    args = ap.parse_args()

    cmds = commands(args.model, args.llama_cpp, args.quant, args.out)
    for c in cmds:
        print("$", " ".join(c))
        if args.run:
            subprocess.run(c, check=True)
    if not args.run:
        print("\n(dry run — pass --run to execute; needs llama.cpp built with llama-quantize)")
    print("\nNEXT: post-quant eval is mandatory —")
    print("  from deploy.vlm_backend import LlamaCppBackend")
    print("  from training.eval import run_model-style loop on the SAME test split")


if __name__ == "__main__":
    # unit-test the command construction (no llama.cpp needed)
    cmds = commands(Path("runs/m"), Path("../llama.cpp"), "Q4_K_M", Path("/tmp/exp_test"))
    assert any("convert_hf_to_gguf.py" in " ".join(c) for c in cmds)
    assert any("--mmproj" in c for c in cmds)
    assert any("Q4_K_M" in c for c in cmds)
    print("export_gguf OK; steps:", len(cmds))
