# Evaluation Protocol — pre-registered promotion rules

Written 2026-07-23, **before** the Qwen3.5-4B full-FT val number existed, after a
statistical audit of the val-gate found most historical accept/reject deltas were below
what unpaired comparison at n=150 can resolve (MDE ~13pt accuracy / ~21pt recall; the
continual-FT "regression" was p=0.28, one recall clip apart; the 9B-LoRA rejection was
sound, p=0.009).

## The instrument

- **Gate:** full `data/real/oops/oops_val.json` (n=150, 75 down / 75 normal), never a
  head-truncated subset. `bench_full.py --n 0` runs the whole manifest.
- **Confirm:** full `data/real/oops/oops_test.json` (n=150). Promotion requires a
  sign-consistent delta here, not a second significance pass.
- **Per-clip records** (`predictions` in every bench report since 2026-07-23) enable
  McNemar paired testing via `scripts/paired_test.py`. Greedy decode + fixed frame
  sampling means zero within-run variance; all noise is clip-sampling variance, which
  pairing removes.
- **Parse failures** are counted separately (`parse_failures`; per-clip `parse` field).
  A parse failure defaults the prediction to "normal" — it is a harness artifact, not a
  perception miss. Precondition for any verdict: parse failures < 5% of clips for both
  models; otherwise fix the harness and re-run first.

## Promotion rule (candidate vs current champion)

Run both on identical val clips with per-clip logs. **PROMOTE** only if ALL of:

1. McNemar exact p < 0.05 on binary danger-correctness (`scripts/paired_test.py`);
2. net paired delta >= +5 clips in the candidate's favor;
3. the candidate loses <= 3 net positive (gold-danger) clips vs the champion — recall
   must not be traded away silently for accuracy;
4. sign-consistent delta on the full oops_test (n=150).

**Point estimate <= champion** = keep champion, recorded as **"no measurable
improvement"** — NOT "regressed" unless McNemar p < 0.05 in the wrong direction.
The status quo is free; only promotion needs evidence.

Unpaired fallback (only when per-clip logs are unavailable for one side): a delta is a
win/loss only beyond ~8pts accuracy / ~10pts recall at n=150; anything less is
inconclusive.

## Qwen3.5-4B full-FT adjudication (pre-registered 2026-07-23, run in flight)

Baseline champion: `runs/sft-qwen35-2b-fullft` — val 128/150 acc (0.853), 58/75 recall
(0.773), 70/75 spec (0.933).

- **DEPLOY** if the promotion rule above passes.
- **SHELVE** if acc in 0.80–0.90 AND recall in 0.72–0.85 — within noise of the 2B.
  Verdict recorded as "no measurable improvement"; per the capacity stopping rule below,
  backbone scaling is then CLOSED.
- **Hard floor:** recall < 54/75 (0.72) = shelve regardless of accuracy.
- **Salvage clause:** even if shelved solo, keep the checkpoint iff the cross-model
  disagreement analysis shows it uniquely catches >= 5 val positives no sibling catches
  (ensemble-member value); otherwise delete to reclaim disk.

## Capacity stopping rule

256M -> 500M -> 2.2B scaling gave 0.13 -> 0.31 -> 0.83 OOPS recall; above ~2B no full-FT
gain has been shown (9B was LoRA — an optimization confound, not capacity evidence). If
the 4B full-FT lands within noise of the 2B on the same data, capacity is declared
non-binding above 2B and backbone scaling ends, absent a unique-clip population in the
disagreement set.
