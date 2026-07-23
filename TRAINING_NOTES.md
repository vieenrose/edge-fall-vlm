# Training — Bootstrap Runs & Findings

Real SFT runs of **SmolVLM2-500M-Video** on **gpu0** (RTX 5090, gpu1 never touched), over
the **stick-figure bootstrap dataset** (1,243 samples, procedural motions → projected
skeletons, no Blender). Purpose: validate the full train→eval→cross-view→false-alarm loop
on real hardware and surface pipeline bugs before photoreal data exists. **These are
plumbing/diagnostic runs — not a usable detector.**

Cross-view protocol: hold out entire camera archetypes (`ceiling`, `low_shelf`) from
training; eval on those unseen views vs. a seen-view reference. Eval is stratified +
shuffled so it includes normals (sensitivity AND specificity both defined).

## Runs
| Run | Target format | Balance | Seen acc | Held acc | Held sens / spec | Note |
|---|---|---|---|---|---|---|
| xview  | rationale-first | no | 0.22 | 0.28 | 0.65 / 0.38 | predicts danger indiscriminately; JSON never reached in token budget → keyword-parse noise |
| xview2 | **answer-first** | no | 0.29 | 0.30 | 0.03 / 0.97 | clean parseable JSON ✓ but collapses to majority "normal" prior |
| xview3 | answer-first | **yes** | **0.885** | **0.59** | 0.77 / 0.46 | balancing → model actually learns; **cross-view gap 0.295** |

## Findings (the useful part)
1. **The whole loop works on real hardware.** Full fine-tune (loss 4.5→1.87, ~5.6 min),
   GGUF-ready checkpoint, stratified cross-view eval, false-alarm projection — all run
   end-to-end, gpu0-only. Bugs found and fixed: multi-image OOM (image-splitting off +
   frame cap), first-N eval slicing that excluded normals (→ stratified sampler).
2. **Answer-first format is the right design and is now the default**
   (`training/dataset.target_text`): the JSON verdict is emitted first so it is always
   within the on-device token budget and always parseable. Diagnosed directly: rationale-
   first generations stalled and never produced JSON. Kept for the real training + it is
   cheaper at inference (no long CoT before the answer).
3. **Class balancing matters** (`--balance`, oversample minorities): the unbalanced
   answer-first model won by always predicting the 54%-majority "normal", ignoring the
   images. Added as a knob for the real run.
4. **With balancing, the model DOES learn — and reproduces the cross-view gap.** Balanced
   answer-first (xview3) reaches **0.885 accuracy on seen views** (0.93 sensitivity) but
   only **0.59 on the two held-out camera archetypes** — a **0.295 cross-view accuracy
   gap** and a specificity collapse (0.79 → 0.46, i.e. many false alarms on unseen views).
   This is the single most important result: our pipeline empirically reproduces the exact
   perspective-generalization failure the project exists to solve, AND our cross-view eval
   protocol reliably detects it. It is the baseline the synthetic multi-view + fisheye
   (FED) data and temporal/geometric augmentation must shrink.
5. **Stick-figures are a plumbing substrate, not a real detector.** Learnable enough to
   validate the loop and the metric; not photoreal enough to trust the absolute numbers.
   Photoreal SMPL-X renders + OmniFall/OOPS-Fall remain the gate for a deployable model.

## 3D-render run vs stick-figure baseline (first fully-synthetic-from-3D result)
Same protocol (answer-first, balanced, holdout ceiling+low_shelf), trained on the Blender
mannequin renders (`data/synth3d/batch`, 243 views) instead of stick-figures.

| Data | Seen acc | Held-out acc | cross-view gap |
|---|---|---|---|
| Stick-figure (1,243 views) | 0.885 | 0.59 | **0.295** |
| **3D mannequin (243 views)** | 0.467 | 0.461 | **0.006** |

### Bigger balanced 3D run (524 views, class-balanced at source)
Re-ran with `convert_motion.py` minority counts boosted (motions 30/30/42/42/42) → 524
rendered views (fall 157 / normal 152 / distress 109 / immobile 106), holdout ceiling+
low_shelf, balanced, answer-first, 4 epochs.

| Data | Seen acc | Held-out acc | cross-view gap | held fall recall | held distress recall |
|---|---|---|---|---|---|
| 3D mannequin, 243 views | 0.467 | 0.461 | 0.006 | — | — |
| **3D mannequin, 524 balanced** | 0.46 | 0.39 | **0.07** | **0.14** | **0.0** |

- **Cross-view gap stays small (0.07)** on the properly-powered balanced run — the 3D→
  view-robustness signal is real and reproducible (vs stick-figure 0.295).
- **But absolute accuracy is low and fall/distress recall is poor** (fall 0.14, distress
  0.0 on held-out). The model does normal (0.67) and immobile (0.67) but MISSES falls and
  distress. Likely causes: (a) a fall's later frames look like `immobile` (both end on the
  floor) so the crude mannequin's *descent motion* isn't salient at 6×384px; (b) distress
  (crouch/jitter) is too subtle for the primitive body; (c) 322 train samples is small.
- **Honest conclusion: the mannequin + procedural-motion substrate is enough to validate
  the pipeline and confirm cross-view robustness, but NOT enough for an accurate detector.
  Getting real accuracy needs fidelity — photoreal bodies (MakeHuman/Mixamo) + realistic
  fall motion (ragdoll/mocap) — not more mannequin data.** Further mannequin runs would be
  polishing a substrate that has hit its ceiling.

Reading (careful — the 243-view run was underpowered):
- The stick-figure model's high seen-accuracy was largely **2D-pattern overfit that did
  not transfer across views** (0.295 gap). The 3D model's **cross-view gap collapses to
  ~0** — what it learns from 3D generalizes evenly across unseen camera angles. That is
  the encouraging signal for perspective robustness.
- BUT 3D absolute accuracy is low (~0.46) and the numbers are **noisy / not conclusive**:
  only 243 views, 154 train after holdout, and severe class imbalance (3 `lying-immobile`
  motions oversampled ~28×, `distress` held-out recall 0.0). Seen vs held-out sensitivity
  even flip (0.26 vs 0.79), a small-sample artifact.
- **Do not conclude "3D fixed the gap."** Conclude: the full 3D pipeline works end-to-end
  and the first signal favors cross-view consistency; a larger, better-balanced render is
  needed for real numbers.

### Concrete next steps for a conclusive 3D result
1. Bigger render: ~1–2k+ views (render cost ~1–2 s/frame on the RTX 5090).
2. Fix motion balance at the source: `convert_motion.py` makes too few `lying-immobile`
   / `distress` motions — generate more variety so oversampling isn't 28×.
3. Higher-fidelity body (MakeHuman/Mixamo mesh on the same rig) — mannequin is readable
   but not photoreal; the real sim-to-real test needs it + a small self-recorded real set.

## Accuracy investigation (multi-person, label normalization, 3-class)
Fixing the fall-recall problem the user prioritized. Findings, in order:

1. **Label-string bug (fixed):** the model emitted "falling"/"fainting"/"lying immobile"
   variants that didn't match canonical labels → per-class numbers were corrupt. Added
   `eval.normalize_status`. After normalization the multi-person model's true numbers were
   **person-down recall 0.73, specificity 0.47** (not the 0.14 fall-recall it looked like).
2. **The real blocker is specificity, not recall:** the model called ~half of NORMAL
   clips "fall" → 53% false-alarm rate, unusable for 24/7.
3. **3-class reframe (`down3`: normal/down/distress) + harder, dominant normals**
   (added crouch/sit-floor/reach-down/squat/kneel; normals 108/222 motions): specificity
   **0.47 → 0.86** (held) / 1.0 (seen) — false alarms largely solved. BUT person-down
   recall **dropped to 0.44** — the model went conservative, now missing >half of downs.

**Conclusion — fidelity ceiling reached.** Label/data engineering only slides the model
along a precision-recall tradeoff; it can't raise both, because at current body fidelity
the model **cannot visually separate "fallen" (horizontal torso on floor) from "crouching
/ sitting on floor / kneeling" (upright torso, low)**. The distinguishing cue is torso
orientation, which the crude skin-mesh body doesn't convey at 384px through fisheye. Also
distress (0.0) is never learned. **The next real lever is FIDELITY, not more label tricks:**
- MPFB2 photoreal rigged bodies (clear torso orientation) — needs elbow/knee joints + IK.
- Blender ragdoll fall physics (distinct fall dynamics vs deliberate lowering).
- possibly denser temporal sampling (a ~0.5s fall descent is under-sampled by 6 frames/3.5s).

Instrumentation added this round: confusion matrix, binary-danger + person-down-recall
metrics (`eval.score`), and `--label-set fine|down3|binary` (`dataset.collapse_label`).

## The precision-recall frontier (all cheap levers exhausted)
| Config | person-down recall | specificity | x-view gap |
|---|---|---|---|
| 5-class, 384px | **0.73** | 0.47 | 0.03 |
| 3-class down3 + hard normals, 384px | 0.44 | **0.86** | 0.11 |
| 3-class down3 + hard normals, 512px | 0.22 | 0.85 | 0.20 |
| 3-class + **pose-assist** (posture as supervised output field), 384px | 0.12 | 0.96 | 0.02 |

Pose-assist result (respects single-VLM: posture is a supervised OUTPUT, no pose model at
inference): did NOT move the curve — recall collapsed to 0.12, model reverted to predicting
"normal"/pretrained words. Comparable data (294 views) rules out pure under-training.
Interpretation: **the model can't extract posture from the imagery in the first place, so
forcing it to predict posture just adds an unlearnable field.** This localizes the
bottleneck to the VISUAL SIGNAL, not the training target.

Reading: the frontier is stuck — you can have recall ~0.73 at spec ~0.47, OR spec ~0.85 at
recall ~0.2–0.44, but **not both**. Labels, class-collapse, hard-negative normals, class
balance, and input resolution were all tried; none move the whole curve. 512px made recall
WORSE and the cross-view gap BIGGER (higher-res overfits seen views). One positive: at
512px distress became partly learnable (recall 0.68 seen / 0.33 held).

**Consolidated conclusion.** A 500M VLM on crude synthetic bodies cannot separate
"person down" from "low-but-normal" (crouch/sit/kneel) at deployment quality — exactly the
ceiling Phase-0 research predicted for a tiny VLM on this task. Cheap iteration is
exhausted. The remaining levers are all BIG investments requiring a direction decision:
1. **Fidelity:** MPFB2 photoreal bodies + ragdoll fall physics + more/varied data +
   real-scene backgrounds. Biggest synthetic-side effort; may still not transfer.
2. **Real data:** self-recorded fisheye falls (ungated) or OmniFall/OOPS-Fall (gated) —
   the sim-to-real gap is untested and could dominate.
3. **Architecture:** reconsider the single-VLM constraint. Phase-0's top recommendation
   was a pose-assisted pipeline. Pose as an auxiliary TRAINING signal was tried and did
   NOT help (the model can't extract posture from the imagery). The remaining variant is
   pose as INPUT — a tiny pose/keypoint front-end feeding the VLM (2-stage; breaks
   "single model"). This bypasses the extraction bottleneck by GIVING the model the
   torso-orientation cue. Phase-0: FallNet (YOLO-pose+LSTM) hit 92%P/97%R on RPi3B+.

## BIGGER ARCH (SmolVLM2-2.2B) — the frontier finally moves
Same down3 protocol; 2.2B via LoRA (fits RPi5 at Q4 ~1.3GB, the largest that does).

| Model / run | seen recall | seen spec | seen distress | held recall | held spec | x-view gap |
|---|---|---|---|---|---|---|
| 500M, down3 | 0.44 | 0.86 | 0.0 | 0.44 | 0.86 | 0.11 |
| 2.2B, same data | 0.39 | 0.72 | 0.81 | 0.47 | 0.24 | 0.11 |
| **2.2B, resourced** (balanced scenes, 462 views, 6 ep) | **0.855** | **0.71** | **0.94** | 0.775 | 0.33 | **0.25** |

Findings:
- **Capacity WAS a real bottleneck.** On seen views 2.2B gets recall 0.855 AND spec 0.71
  AND distress 0.94 — the 500M model could never lift more than one at a time. The bigger
  vision encoder can actually read torso orientation + the distress pose.
- **New bottleneck: cross-view OVERFITTING.** Held-out specificity collapses (0.33) and the
  cross-view gap grows to 0.25 — the bigger model overfits training viewpoints, calling
  new-angle normals "distress". (500M generalized evenly because it learned only coarse
  features.) Fix = more viewpoint/data diversity (the render pipeline's strength) + LoRA
  regularization; possibly per-install calibration for the specific deployment angle.
- Scene-composition fix applied: ~46% fully-normal scenes (was ~25%) so the normal class
  isn't starved (a multi-person scene is "danger" if anyone is down).

### Scale-up (2.2B, 1126 views, 6 viewpoints) — the recipe works
| Held-out metric | 2.2B @462 | 2.2B @1126 |
|---|---|---|
| person-down recall | 0.775 | **0.882** |
| specificity | 0.33 | **0.66** |
| danger sensitivity | 0.92 | 0.93 |
| cross-view gap | 0.246 | 0.204 |

Seen views essentially solved: recall 0.988 / spec 0.929 / distress 1.0 / acc 0.971.
**2x data → held-out specificity DOUBLED (0.33→0.66) and recall rose to 0.88.** The
cross-view overfitting is a data-quantity problem, and the render pipeline makes data
cheap. Validated recipe: **2.2B VLM + large viewpoint-diverse synthetic set**. Remaining
held-out specificity leak (normal→distress on extreme unseen angles) should keep shrinking
with more data; per-install calibration (few frames from the deployment camera) is the
pragmatic closer since real cameras are a fixed archetype, not "held out".
- **Deployment cost:** 2.2B Q4 fits 4GB but is slower on RPi5 (~8-20 s/inference per
  Phase-0 research) — acceptable only at low duty cycle; re-benchmark on real Pi.

## FIRST REAL-WORLD VALIDATION (sim-to-real) — the pivotal result
Synthetic-only 2.2B (runs/sft-2b-scale) run on REAL URFD footage (20 overhead falls +
20 frontal ADL), down3 scheme:

| Metric | Value |
|---|---|
| **Real fall recall (person-down)** | **0.95** (19/20 real falls detected) |
| **Real specificity** | **0.0** (every real ADL clip false-alarmed) |

Confusion: real normal → 17 "down", 3 "distress", 0 correct.

**The asymmetry IS the finding: synthetic FALLS transfer, synthetic NORMALS don't.** The
"person down" concept generalizes sim-to-real almost perfectly (0.95 on real overhead
falls — validates the synthetic approach for the core alert). But "normal" was learned on
crude mannequins doing procedural crouch/sit/walk, nothing like real ADL, so real normal
activity reads as danger. This matches OmniFall's finding exactly (synthetic transfers for
falls; you need REAL NEGATIVES for specificity).
- Confound: falls=overhead(cam1), ADL=frontal(cam0) — part of the 0.0 spec is view
  mismatch; disentangle with real OVERHEAD normal footage (LOAF/CEPDOF).
- **Clear fix: add real normal footage to training** (easy to get — LOAF/CEPDOF/WEPDTOF
  real overhead-fisheye people, no fall labels needed; or OmniFall staged ADL). Keep the
  synthetic falls (they transfer). Expected: specificity recovers, 0.95 recall holds.
- Harness: scripts/validate_real.py (real clips -> model -> recall/spec/confusion),
  scripts/download_urfd.py.

## THE FIX WORKS — synthetic + small real negatives (validated on real)
Continue-trained the synthetic 2.2B on a mix = synthetic scale2b + real URFD train
(10 falls + 20 ADL, oversampled 15x -> ~29% real), LoRA, then validated on the HELD-OUT
real test (20 falls + 20 ADL, no clip overlap):

| | real fall recall | real specificity | acc |
|---|---|---|---|
| synthetic-only 2.2B | 0.95 | **0.0** | 0.48 |
| **+ real negatives** | 0.90 | **1.0** | **0.95** |

Confusion (held-out real): down 18/20 correct, normal 20/20 correct. **20 real normal
clips took specificity 0 -> 1.0** while recall held at 0.90. The synthetic-then-small-real
curriculum (Phase-0 / BEDLAM / OmniFall all predicted this) is proven on real data.

### The validated recipe
1. Large **viewpoint-diverse synthetic 3D** data -> perspective robustness (x-view gap small).
2. **2.2B VLM** (not 500M) -> enough capacity to read pose (RPi5-deployable at Q4).
3. Synthetic **falls transfer** to real out-of-the-box (0.95 recall).
4. A **small real NEGATIVE set** fixes specificity (0->1.0). Real normals are the cheap,
   easy-to-collect kind (any everyday footage).

### Real cross-VIEW fall generalization is partial (honest limitation)
Real-fixed model (sft-2b-real) on real URFD falls by view:
- **Overhead (cam1, trained view): 0.90 recall**
- **Frontal (cam0, NEVER in real training): 0.55 recall**

So perspective robustness is weaker on REAL data than the synthetic cross-view gaps
(0.03-0.20) suggested — sim-to-real and cross-view COMPOUND. The deployment geometry is
overhead-fisheye, so the 0.90 overhead number is the relevant one, but this shows the
model is NOT uniformly view-robust on real footage and reinforces the per-install / real-
negatives-from-the-actual-view recommendation. (Caveat: frontal test is recall-only, no
ADL, 20 clips.)

### Honest caveats (this is a first probe, not a shipping metric)
- Small test (40 clips) — effect is huge/unambiguous but N is small.
- URFD is staged; real deployment + in-the-wild (OmniFall OOPS) is a harder test still to run.
- Real train/test are same-dataset (URFD) same-distribution (clip-split clean). Deployment
  footage differs -> expect to need real negatives FROM the actual install (per-install).
- View confound partly remains (falls overhead / ADL frontal) but both classes now work.

## IN-THE-WILD BENCHMARK vs published baselines (OOPS-Fall)
Ran sft-2b-real on 300 sampled OOPS in-the-wild segments (150 fall/fallen + 150 hard
normals: walk/sit/stand/kneel/squat/jump), down3 mapping:

| Model | in-the-wild fall recall | specificity |
|---|---|---|
| VideoMAE-K400 (OmniFall baseline) | 0.21 | ~0.96 |
| Frozen I3D (OmniFall baseline) | 0.68 | — |
| **Ours (synthetic + small real)** | **0.83** (124/150) | **0.72** (108/150) |

**Our model beats both published baselines on in-the-wild fall recall (~4x VideoMAE).**
Critical for safety: the baselines get high specificity by predicting "no fall" most of
the time (VideoMAE 0.21 sens BECAUSE spec ~0.96 — misses 4/5 falls). Ours is balanced
(0.83 recall / 0.72 spec) — the right tradeoff when a missed fall is the costly error.

Caveats (directional, not identical-protocol): 300-clip sample not full OmniFall splits;
our down=fall+fallen; metric defs may differ slightly from OmniFall's balanced-accuracy;
OOPS is handheld/varied web video (domain shift from both our synthetic training AND the
overhead-fisheye deployment) — so 0.83 transferring here is a strong generalization signal.
Tools: scripts/build_oops.py + validate_real.py; data/real/oops.

## Bottom line after the accuracy campaign
Every single-VLM-on-synthetic lever (labels, class-collapse, hard negatives, balance,
resolution, pose-assist-output) has been tried. The frontier did not improve past
recall~0.73@spec~0.47 or spec~0.85@recall~0.2-0.44. The bottleneck is the VISUAL SIGNAL:
a 500M VLM cannot read torso orientation off crude synthetic mannequins at surveillance
distance/fisheye. Highest-confidence path to a deployable detector is a **pose front-end
feeding the VLM (2-stage)** — which is what Phase-0 recommended and what the "single VLM
only" constraint was trading away. The single-VLM-on-synthetic approach has a low ceiling
for this task; further tuning of it is not productive.

## Carried into the real training recipe
- Keep **answer-first** target; short JSON is the deployment output, `WHY:` optional.
- Use **`--balance`** (or loss weighting) — never train on the raw class mix.
- Keep **cross-view holdout** as the headline metric; report specificity + projected
  false-alarms/day, not just accuracy.
- The pipeline is ready; swap stick-figure frames for BlenderProc photoreal renders and
  add OmniFall/OOPS-Fall loaders (same `samples.jsonl` schema).

## Reproduce
```
CUDA_VISIBLE_DEVICES=0 python training/sft.py \
  --samples data/bootstrap/shards/samples.jsonl \
  --base HuggingFaceTB/SmolVLM2-500M-Video-Instruct --out runs/sft-xview3 \
  --epochs 3 --bs 2 --lr 2e-5 --max-frames 6 --img-size 384 --balance \
  --holdout-views ceiling,low_shelf
python scripts/eval_xview.py --model runs/sft-xview3 \
  --samples data/bootstrap/shards/samples.jsonl --holdout ceiling,low_shelf --n 200
```

## 2026-07-23 — Deep-review reorientation: capacity CLOSED, data is the constraint

A statistical audit (EVAL_PROTOCOL.md, pre-registered) + per-clip logging landed before
the Qwen3.5-4B full-FT finished. Sweep on the full oops_val n=150, per-clip records,
zero parse failures on every model:

| model | acc | recall | spec | verdict |
|---|---|---|---|---|
| qwen2b-fullft (champion) | 0.853 | 0.773 | 0.933 | deployed |
| qwen4b-fullft (fresh)    | 0.753 | 0.627 | 0.880 | SHELVED: below 0.72 recall floor; McNemar vs 2B p=0.0041, net -15 clips; 0 unique catches -> deleted |
| smol2b-real              | 0.773 | 0.880 | 0.680 | kept — 7 UNIQUE positive catches |
| qwen9b-lora              | 0.767 | 0.573 | 0.960 | 1 unique catch (<5) -> deleted |

**Backbone scaling is CLOSED**: 2B > 4B > 9B on recall with identical training data.
A clean full-FT 4B (no LoRA confound, no parse issues) is *significantly worse* than the
2B. Consistent with the review diagnosis: training has zero in-the-wild real fall
positives, so extra capacity just preserves a stronger conservative prior.

**Measured ensemble fusion** (same 150 clips — replaces the old 0.92–1.0 bound):
- OR(qwen2b, smol2b): recall **0.947**, spec 0.667 — a recall no single model reaches;
  the two backbones are genuinely complementary (smol2b uniquely catches 7 positives).
- AND(qwen2b, smol2b): spec **0.947** at recall 0.707 — high-precision confirmer.
- MAJ-2-of-3 (+4B): 0.827/0.907 — no better than the pair; 4B adds nothing.
- Only 2/75 positives are missed by ALL models (down_0015, down_0048): the OR-system
  ceiling with current checkpoints is 0.973 recall.

Report: bench_val150_perclip_4models.json (has per-clip predictions; paired tests via
scripts/paired_test.py).

**p_lying probe** (same val clips, only the LAST 3 frames per strip): recall drops for
every model (qwen2b 0.773->0.693, smol2b 0.880->0.653) — detection leans on the fall
transient, not the aftermath. Caveat: OOPS fallers often stand back up, so some tail-3
"misses" are correct reads of a recovered person; still, do not assume post-fall lying is
the easy case. The real-fall harvest includes OmniFall `fallen` (lying) segments, which
directly train this. Report: bench_val150_tail3.json.

## 2026-07-23 — REAL fall data WINS: the reorientation validated

Continual-FT of the champion (runs/sft-qwen35-2b-fullft) on train_mixed (1576) + **1692
real in-the-wild OOPS fall clips** (scripts/build_oops_train.py, leakage-free at the
source-video level) -> runs/sft-qwen35-2b-realfall. lr 3e-6, 1 epoch, train_loss 0.018.
First time the project trained on real ITW falls beyond the 10 URFD lab clips.

| set | model | acc | recall | spec |
|---|---|---|---|---|
| oops_val (150)  | champion | 0.853 | 0.773 | 0.933 |
| oops_val (150)  | **realfall** | **0.887** | **0.920** | 0.853 |
| oops_test (150) | champion | 0.807 | 0.787 | 0.827 |
| oops_test (150) | **realfall** | **0.853** | **0.893** | 0.813 |

Paired McNemar (per-clip, 0 parse failures anywhere):
- **RECALL, pooled 300 clips: realfall catches 19 falls the champion misses, loses 0.
  p < 0.00001.** Significant on val (p=0.001) AND test (p=0.008) independently.
- SPEC: −6 on val (p=0.03), −1 on test (p=1.0) — small, inconsistent cost.
- COMBINED danger-correctness, pooled: net +12, **p=0.043** (significant). Val-only was
  +5/p=0.33 — underpowered at n=150, exactly as the eval audit predicted; pooling resolves it.

**Verdict: PROMOTE.** The real-fall model is the new best on the metric that matters for a
safety detector (fall recall 0.773->~0.91), significant and replicated across both held-out
sets, at a small specificity cost the verification stack absorbs. This confirms the review's
binding-constraint diagnosis: the ceiling was zero real fall positives in training, not
model capacity. Reports: bench_val150_realfall_vs_champion.json, bench_test150_*.json.

## 2026-07-23 — Real fall data helps EVERY size (small models partially rehabilitated)

Applied the same real-fall continual-FT (train_qwen_real, lr 3e-6, 1ep) to the 500M and
256M SmolVLM2 baselines. oops_val n=150, per-clip, paired McNemar vs each baseline:

| model | recall before | recall after | Δfalls (caught/lost) | McNemar p | spec before→after |
|---|---|---|---|---|---|
| 256M | 0.093 | **0.360** | +20 / 0 | <0.0001 | 0.96 → 0.75 |
| 500M | 0.427 | **0.520** | +7 / 0  | 0.016 | 0.95 → 0.89 |
| 2B (ref) | 0.773 | 0.920 | +19 / 0 | <1e-5 | 0.93 → 0.83 |

Findings:
1. **Real data lifts recall at every size, significantly, losing zero falls** — the effect
   is not capacity-specific.
2. **It helps the SMALLEST model the most in relative terms** (256M recall ~4x: 0.09→0.36).
   This REVISES the earlier "small models are capacity-bound" conclusion: they were *also*
   data-starved (zero real falls). Capacity and data were BOTH binding.
3. **But real data does NOT close the capacity gap.** Even post-fix, 256M (0.36) and 500M
   (0.52) recall stay far below 2B (0.92) and below any deployable bar for a safety
   detector; specificity also drops (same recall/spec trade as the 2B). The RPi5 edge path
   is more viable than the SIZE_COMPARISON.md numbers suggested, but 2B remains the smallest
   size that reaches usable recall.

Report: bench_val150_smallmodels_realfall.json. Models: runs/sft-{256m,500m}-realfall.

## 2026-07-23 — Durable false-alarm fix: hard-negatives work (concept validated)

The realfall model over-triggers "down" on non-upright bodies that aren't fallen (gymnastics
inverted mid-air, a leaning cyclist, a slip that recovers — all 25 of its false alarms report
posture "horizontal-on-floor"). Fix: mine real "horizontal-but-normal" clips from the OOPS
labels build_oops excludes — other(9)/lie_down(5)/lying(6)/crawl(14), which OmniFall separates
from actual falls, i.e. annotator-judged non-falls — label them NORMAL, continual-FT.

Concept test on a PARTIAL harvest (96 hard-neg clips, 3x oversampled) -> runs/sft-qwen35-2b-hardneg:

| set | model | acc | recall | spec |
|---|---|---|---|---|
| val (150)  | realfall | 0.887 | 0.920 | 0.853 |
| val (150)  | **hardneg** | **0.907** | 0.920 | **0.893** |
| test (150) | realfall | 0.853 | 0.893 | 0.813 |
| test (150) | **hardneg** | **0.887** | **0.920** | **0.853** |

Pooled val+test (300 clips, paired): FALSE ALARMS 25 -> 19 (9 fixed, 3 new, net -6;
McNemar p=0.146), RECALL 0.907 -> 0.920 (actually +2 falls, none lost). **Strictly better
on both axes** — fewer false alarms AND slightly higher recall — from just 96 hard-negatives.
FA clips fixed incl. normal_0014 (lake-slip), normal_0125 (cyclist), normal_0115, normal_0149.
Lone holdout: normal_0127 (inverted gymnast mid-air, the most extreme non-upright pose).

Not yet significant (only 96 clips) — harvesting the full 446-video / ~682-clip set (robust
chunked fetch that survives the OOPS throttle) to strengthen it before deploying. Reports:
bench_{val,test}150_hardneg.json.

## 2026-07-23 — Hard-negative FA fix is capacity-dependent (small-model negative result)
Applied the hardneg fix to 256M/500M (both GPUs in parallel). Unlike the 2B (FA down, recall
held), the small models got WORSE specificity: 256M spec 0.75->0.57 (FA 19->32), 500M spec
0.89->0.72 (FA 8->21), while recall rose. The small models lack capacity to learn the
"horizontal-but-normal != fallen" distinction; the added normal clips just shift the balance
toward more "down". The durable FA fix requires ~2B capacity — it does not transfer to edge
models. Reports: bench_val150_{256m,500m}_hardneg.json.
