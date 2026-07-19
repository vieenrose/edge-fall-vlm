# Accuracy vs VLM Size (SmolVLM2 256M / 500M / 2.2B)

Controlled comparison: same two-stage recipe (synthetic pretrain on `scale2b` → +real
fine-tune on `train_mixed`), same LoRA/`down3` settings, same real test sets. **Model size
is the only variable.**

| Model | Params | On-device (est.) | URFD held-out (easy, in-distribution) | **OOPS in-the-wild (hard, out-of-distribution)** |
|---|---|---|---|---|
| | | | recall / specificity | **person-down recall** / specificity |
| SmolVLM2-**256M** | 0.26B | ~0.5 GB, ~1–2 s/inf on Pi | 1.00 / 1.00 | **0.13** / 0.94 |
| SmolVLM2-**500M** | 0.51B | ~1 GB | 1.00 / 1.00 | **0.31** / 0.91 |
| SmolVLM2-**2.2B** | 2.26B | 2.2 GB Q6_K, ~4–6 s/inf on Pi | 0.90 / 1.00 | **0.83** / 0.72 |

## The finding: size barely matters on easy data, and dominates on hard data

1. **On the easy in-distribution test (URFD, 40 clips), all sizes saturate** — even 256M
   hits 1.0/1.0. This test does NOT discriminate model size; it's too easy and the models
   trained on URFD-like negatives. *(Lesson: don't judge a fall detector on a small staged
   benchmark — it hides the capacity gap.)*

2. **On the hard in-the-wild test (OOPS, 300 clips), recall scales steeply with size:**
   **0.13 → 0.31 → 0.83** from 256M → 500M → 2.2B. The small models MISS most real
   uncontrolled falls (256M catches 13%, 500M 31%), because they collapse to the
   conservative "normal" prediction (note their high specificity 0.91–0.94 is *because*
   they rarely say "down"). The 2.2B is the only one that actually catches falls in the
   wild (0.83), at the cost of lower specificity (0.72).

3. **Capacity buys generalization, not in-distribution fit.** All sizes fit the training
   distribution; only the 2.2B generalizes to novel real scenes. For a safety detector —
   where a *missed* fall is the costly error — recall on out-of-distribution real footage
   is the metric that matters, and it drops ~2.7× (2.2B→500M) to ~6.4× (2.2B→256M).

## Deployment trade-off
The smaller models are much cheaper on the Pi (256M/500M ≈ 1–2 s vs the 2.2B's ~4–6 s, and
0.5–1 GB vs 2.2 GB), but they would miss 70–87% of real in-the-wild falls. **The 2.2B is
the smallest size that is actually usable for the safety task**, and it still fits RPi5 at
Q6_K. Going smaller trades away exactly the capability (real-world fall recall) the product
exists to provide.

Models: `runs/sft-256m-real`, `runs/sft-500m-real`, `runs/sft-2b-real`.


## Iteration: test-time augmentation (autoresearch loop) — a real, retraining-free win
The size gap is capacity-bound (augmentation retraining = null; distillation probe showed
95% teacher/GT agreement → would be null, skipped). But **test-time augmentation** (run K=4
augmented views + original, aggregate "any view sees a fall → fall") recovers latent signal
WITHOUT retraining. Verified on a held-out OOPS-test split (val agreed → no overfit):

| Model (OOPS in-the-wild, held-out test) | single-pass recall | **+TTA recall** | TTA spec |
|---|---|---|---|
| 256M | 0.16 | **0.47** (~3x) | 0.79 |
| 500M | 0.31 | **0.55** (~1.8x) | 0.77 |
| 2.2B | 0.83 | _(see below)_ | |

TTA nearly triples the 256M's in-the-wild recall and lifts the 500M by ~1.8x, with
specificity holding ~0.77. Cost: K+1 forward passes at inference — cheap for the small
models on-device (they're fast). Anti-overfit discipline held throughout: OOPS was split
val/test, only val drove decisions, test was reported once, and augmentation was DROPPED
because its val/test moved opposite directions.
