# Mixed-precision ternary quantization of the 2.2B text decoder

Starting question: can the 2.2B model's weights be made ternary ({-1,0,1}, ~1.6 bit/weight)
to shrink it further than Q6_K? "Lossless" ternary isn't a real thing — it's a ~10x
compression vs fp16 — but quantization-aware training (QAT) with a straight-through
estimator (STE) can get a *usable* model at that precision, and a mixed-precision layout
(only some layers ternary) can recover most of the accuracy back.

## Method
- Self-distillation QAT: frozen fp16 `sft-2b-real` as teacher, a trainable copy of the same
  checkpoint as student, with target Linear layers wrapped in a fake-ternary STE quantizer
  (`scripts/qat_ternary.py`). Loss = CE(GT) + KL(student/teacher answer-token logits),
  8000 steps, lr 3e-4, on the same synthetic+real training mix used for the base model.
- Sensitivity check first (`scripts/quant_sensitivity.py`): naive post-training ternary
  rounding (no retraining) of even ONE weight role (24/168 Linear layers, ~14%) — attention
  Q/K/V/O or any single MLP matrix — completely collapses generation (model always predicts
  "normal", output effectively garbled). This holds uniformly across all 7 roles and all 3
  depth bands tested. **Naive PTQ ternary has no safe layer to exploit; only QAT rescues it.**

## Results (OOPS in-the-wild test, n=60, same harness/hardware — `scripts/bench_full.py`)

| Config | Layers ternary | Accuracy | Recall (danger) | Precision | Specificity |
|---|---|---|---|---|---|
| fp16 baseline | 0 | 0.867 | 0.938 | 0.833 | 0.786 |
| Full ternary QAT | 168/168 (100%) | 0.667 | 0.531 | 0.773 | 0.821 |
| Full binary QAT | 168/168 (100%) | 0.667 | 0.688 | 0.688 | 0.643 |
| Attention-only ternary QAT | 96/168 (25% of params) | 0.783 | 0.656 | 0.913 | 0.929 |
| **MLP-only ternary QAT** | 72/168 (**75% of params**) | 0.783 | **0.844** | 0.771 | 0.714 |

**MLP-only wins**: it compresses the *larger* share of parameters (MLP = gate/up/down proj,
75% of the text decoder) while leaving attention (Q/K/V/O, the routing/mixing layers) at
full precision. Recall recovers to 0.844 vs the uniform ternary model's 0.531 — attention
appears far more sensitive to ternary noise than MLP, plausibly because attention errors
corrupt *which* tokens/frames information flows between, while MLP errors are more like
per-channel noise the network's redundant hidden width can partially absorb.

## Real GGUF export (not just bf16-storage proxy)
Exported MLP-only-ternary checkpoint to GGUF via `convert_hf_to_gguf.py` (F16) then
`llama-quantize --tensor-type "ffn_gate=tq1_0" --tensor-type "ffn_up=tq1_0" --tensor-type
"ffn_down=tq1_0" ... Q8_0` (MLP → real packed ternary `TQ1_0`, attention/embeddings/output →
`Q8_0`). Verified with real `llama-mtmd-cli` inference (not just training-loss numbers) on
held-out real clips — correct JSON verdict on a real fall clip (`status:down`) and a real
normal-walk clip (`status:normal`); no garbling (unlike naive Q4_K_M, which broke completely).

| Package | Text decoder | Vision tower (F16, unchanged) | **Total** |
|---|---|---|---|
| Currently deployed (Q6_K, uniform) | 1489 MB | 872 MB | **2.31 GB** |
| Mixed (MLP→TQ1_0, attn→Q8_0) | 899 MB | 872 MB | **1.77 GB** |

~25% smaller total package, in exchange for recall dropping from ~0.94 (Q6_K/fp16) to ~0.84.

## Conclusion
This is a real, working compression option — not a null result like the earlier
distillation/augmentation experiments (see `SIZE_COMPARISON.md`). It is not *required*:
Q6_K already fits the Pi 5's 4GB budget with headroom, so this trades real recall for extra
headroom, useful if you need to run other processes alongside the model or want margin for
a larger vision tower later. **Recommendation: keep Q6_K as the default deployed model;
treat MLP-only mixed ternary as an available option if the 4GB budget becomes tight.**

Scripts: `scripts/qat_ternary.py` (`--mode ternary|binary --scope all|mlp|attn`),
`scripts/quant_sensitivity.py`, `scripts/bench_full.py`.
