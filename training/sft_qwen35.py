"""SFT driver for Qwen3.5 (0.8B/2B) VLM bases -- same recipe/data as the SmolVLM2 lineup
(training/sft.py) for an apples-to-apples comparison. Native transformers support (no
custom code, no version conflicts, unlike InternVL3.5), and full GGUF export support
already in llama.cpp (LLM_ARCH_QWEN35 + Qwen3VLVisionModel mmproj).

Only real difference from sft.py: Qwen3VLProcessor sizes images by PIXEL AREA
(min_pixels/max_pixels), not side-length like SmolVLM's `size={"longest_edge": N}` --
bounded here to keep per-frame token count comparable to SmolVLM2's ~81 tokens/frame.

    CUDA_VISIBLE_DEVICES=0 python training/sft_qwen35.py \
        --samples data/train_mixed.jsonl --base Qwen/Qwen3.5-0.8B \
        --out runs/sft-qwen35-0.8b --epochs 1 --bs 2
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.dataset import Sample, load_images, load_records, target_text
from training.sft import _subsample

# ~384x384 area budget, matches the SmolVLM2 recipe's per-frame token cost
MAX_PIXELS = 384 * 384
MIN_PIXELS = 64 * 64


def build_collator(processor, max_frames: int = 6, fps_augment: bool = True,
                   domain_augment: bool = False, with_rationale: bool = True):
    import numpy as np
    from training.dataset import temporal_augment
    from training.dataset import domain_augment as _domain_aug
    rng = np.random.default_rng(0)

    def collate(samples: list[Sample]):
        texts, images_batch = [], []
        for s in samples:
            imgs = load_images(s)
            if fps_augment:
                imgs = temporal_augment(imgs, rng, max_n=max_frames)
            imgs = _subsample(imgs, max_frames)
            if domain_augment:
                imgs = _domain_aug(imgs, rng)
            msgs = [{"role": "user",
                     "content": [{"type": "image"} for _ in imgs] + [{"type": "text", "text": s.prompt}]},
                    {"role": "assistant",
                     "content": [{"type": "text", "text": target_text(s, with_rationale=with_rationale)}]}]
            texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False,
                                                        enable_thinking=False))
            images_batch.append(imgs)
        batch = processor(text=texts, images=images_batch, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        img_tok = getattr(processor.tokenizer, "image_token_id", None)
        if img_tok is not None:
            labels[labels == img_tok] = -100
        batch["labels"] = labels
        return batch
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--base", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--out", type=Path, default=Path("runs/sft-qwen35"))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--deepspeed", default=None,
                    help="path to a DeepSpeed config JSON (e.g. ZeRO-2 CPU-offloaded "
                    "optimizer) -- lets a full fine-tune of a model too big for plain "
                    "bf16+8bit-Adam on one GPU fit by offloading optimizer state to RAM")
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-mlp", action="store_true",
                    help="also target gate/up/down_proj -- more capacity to override a "
                    "large base model's default verbose generation style, still far "
                    "cheaper than full FT")
    ap.add_argument("--qlora", action="store_true",
                    help="4-bit NF4 frozen base (bitsandbytes) + LoRA on top -- lets a much "
                    "bigger/broader LoRA fit in memory than plain LoRA-on-bf16-base, closer "
                    "to full-FT quality for models too big to full-FT on one GPU")
    ap.add_argument("--max-frames", type=int, default=6)
    ap.add_argument("--label-set", default="down3")
    ap.add_argument("--domain-augment", action="store_true")
    ap.add_argument("--balance", action="store_true")
    ap.add_argument("--rationale", action="store_true", default=True)
    ap.add_argument("--no-rationale", dest="rationale", action="store_false",
                    help="train on JSON-only target (no WHY: tail) -- forces an immediate "
                    "answer instead of open-ended reasoning that can run past the token budget")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments

    processor = AutoProcessor.from_pretrained(
        args.base, size={"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS})

    if args.qlora:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                     bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForImageTextToText.from_pretrained(
            args.base, dtype=torch.bfloat16, device_map={"": 0}, quantization_config=bnb_cfg)
    else:
        model = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.bfloat16, device_map=None)

    if args.lora or args.qlora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if args.qlora:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if args.lora_mlp or args.qlora:
            targets += ["gate_proj", "up_proj", "down_proj"]
        cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
                         target_modules=targets)
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()

    samples = load_records(args.samples, label_set=args.label_set)
    if args.balance:
        from collections import Counter
        import numpy as np
        by = {}
        for s in samples:
            by.setdefault(s.label, []).append(s)
        target = max(len(v) for v in by.values())
        rng2 = np.random.default_rng(0)
        balanced = []
        for v in by.values():
            reps = [v[int(i)] for i in rng2.integers(0, len(v), target)]
            balanced += reps
        rng2.shuffle(balanced)
        print(f"balanced: {dict(Counter(s.label for s in samples))} -> "
              f"{dict(Counter(s.label for s in balanced))}")
        samples = balanced
    collate = build_collator(processor, max_frames=args.max_frames, domain_augment=args.domain_augment,
                             with_rationale=args.rationale)

    # full FT of a 2B model with Qwen's large vocab OOMs on 32GB with plain fp32 AdamW
    # states (~17.6GB) + bf16 weights/grads (~8.8GB) + activations. 8-bit AdamW cuts
    # optimizer memory ~4x; gradient checkpointing trades the activation memory for compute.
    targs = TrainingArguments(
        output_dir=str(args.out),
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        save_total_limit=1,
        remove_unused_columns=False,
        max_steps=2 if args.smoke else -1,
        report_to=[],
        optim="adamw_torch" if (args.lora or args.qlora or args.deepspeed) else "adamw_bnb_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=args.deepspeed,
    )
    model.config.use_cache = False
    if args.lora or args.qlora:
        model.enable_input_require_grads()  # needed for grad checkpointing through frozen base + LoRA
    trainer = Trainer(model=model, args=targs, data_collator=collate, train_dataset=samples)
    trainer.train()
    if not args.smoke:
        if args.lora or args.qlora:
            merged = model.merge_and_unload()
            merged.save_pretrained(str(args.out))
        else:
            trainer.save_model(str(args.out))
        processor.save_pretrained(str(args.out))
    print("SFT done ->", args.out)


if __name__ == "__main__":
    main()
