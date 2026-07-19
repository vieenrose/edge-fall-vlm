# AI-Care-2 — Research Plan (Phase 0: First Research)

**Goal:** ONE compact vision model running 24/7 on a Raspberry Pi 5 (4GB) that detects human falls, fainting, and other danger events from a real-time RGB camera — robust to perspective, camera angle, lighting, and optical distortion up to ultra-wide / fisheye lenses.

**Training hardware:** single RTX 5090 32GB (**gpu0 only** — gpu1 is reserved for another project).

**Date:** 2026-07-18. Evidence below comes from a 25-source, adversarially-verified deep-research pass (24/25 claims confirmed 3-0, 1 refuted).

---

## 1. Headline conclusion

A **pose-based specialist pipeline** (fisheye-robust person/pose detector + lightweight temporal head over keypoint trajectories) is the evidence-backed primary architecture — **not** a tiny VLM running per-frame. A tiny VLM (SmolVLM-256M/500M) fits the RAM budget only as an **event-triggered secondary verifier** to suppress false alarms.

Why not a VLM as the main loop:
- No verified source demonstrates multi-FPS VLM inference on RPi5; SmolVLM-256M needs <1GB (0.8GB bf16 GPU; GGUF CPU smaller) so it *fits*, but continuous real-time throughput on Pi CPU is unproven.
- Pose pipelines are proven on far weaker hardware: **FallNet** (YOLOv8n-pose + LSTM, 3.4M params, 6.4 GFLOPs) hit **13 FPS / ~100ms on a Raspberry Pi 3B+** with 92% precision / 97% recall / 94% F1 [Biomedical Signal Processing & Control 2025]. RPi5 is several times faster.
- Skeleton-based processing is also privacy-friendlier (log keypoints, not appearance).

## 2. Recommended architecture (v0 hypothesis)

```
RGB camera (wide / fisheye), continuous
  └─ Stage A — Person/pose detector (~3–6M params, int8/NCNN or Hailo)
     YOLOv8n/v11n-pose class, trained with:
       • FED synthetic fisheye augmentation (equidistant projection r = f·θ)
       • RAPiD-style rotation-aware boxes (people appear rotated in overhead fisheye)
       • Two-stage curriculum: synthetic-fisheye epochs → real-fisheye epochs
       • Pretrain / domain-adapt on CEPDOF + WEPDTOF + HABBOF + MW-R
  └─ Stage B — Temporal head over keypoint trajectories (LSTM or light GCN, <1M params)
     Classes: fall, faint/collapse, lying-immobile, distress, normal
     + rule layer: prolonged immobility timer, on-floor duration
  └─ Stage C — (optional) SmolVLM-256M/500M GGUF, event-triggered only
     Verifies candidate alarm clips (a few frames) to cut false positives;
     low duty cycle → respects the 4GB RAM and thermal budget
```

Train with **cross-view splits** and evaluate on **in-the-wild** data — this is the core scientific risk (see §4).

## 3. Verified evidence highlights

| Finding | Numbers | Confidence |
|---|---|---|
| FallNet (YOLOv8n-pose+LSTM) edge-feasible | 3.4M params, 6.4 GFLOPs, 92%P/97%R, 13 FPS on Pi 3B+ | High |
| Pose+GCN near-ceiling on staged benchmarks | VIRA-GCN 99.81% acc on Le2i (staged — treat as upper bound) | Medium |
| Staged→wild collapse is THE risk | VideoMAE: 0.78 bal-acc staged → **0.21 sensitivity** on OOPS-Fall; I3D cross-view 0.44 vs cross-subject 0.72 | High |
| OmniFall unified corpus exists | 8 staged datasets, ~42h multiview, 101 subjects, 29 views + OOPS-Fall ~1.3k wild segments | High |
| Overhead-fisheye people datasets | CEPDOF 25,504 frames (incl. low-light/IR), WEPDTOF 16 in-the-wild clips/188 IDs, HABBOF 5,837, MW-R 8,752 — **no fall labels** | High |
| Fisheye: augment, don't rectify | FED augmentation + synth→real curriculum ranked 5th/62 (F1 0.6397, 2nd in awards) at AI City Challenge 2025 fisheye track, no architecture change | High |
| Rotation-aware detection | RAPiD periodic angle loss, single-pass, ~YOLO speed on MW-R/HABBOF/CEPDOF | High |
| Tiny VLM memory fits | SmolVLM-256M <1GB, 500M 1.2GB, 2.2B 4.9GB (bf16 GPU figures; Pi speed unproven) | Medium |
| RTX 5090 32GB is ample | 7–11B VLMs QLoRA-tune on 24GB; our specialists are 100–1000× smaller | High |
| Single-frame fall detectors exist but insufficient | LFD-YOLO 5.7M params — no temporal modeling, can't split fall vs lying down | Medium |

**Refuted claim (do not rely on):** VIRA-GCN's two-axis skeleton-rotation augmentation conferring rotated/occluded-view robustness failed adversarial verification (1-2). Viewpoint augmentation alone is not a proven fix for camera-angle robustness — real multi-view data + cross-view evaluation is.

## 4. Key risks & how the plan addresses them

1. **Staged→wild generalization collapse** (the #1 risk). Staged 99% ≠ real-world. → Train on OmniFall's unified corpus with cross-subject AND cross-view splits; hold out OOPS-Fall as the honest test; report sensitivity/specificity there, not staged accuracy.
2. **No public fisheye *fall* dataset verified** (CVFD/FES unconfirmed). → Bridge with: FED-distortion of OmniFall videos + fisheye person-detection pretraining (CEPDOF/WEPDTOF) + self-recorded fisheye fall clips for validation.
3. **False alarms in 24/7 deployment.** Older in-home trials averaged ~5.4 false alarms/day (window lighting, multiple occupants). → temporal hysteresis, on-floor duration logic, and the VLM verifier stage; track false-alarms/day as a first-class metric.
4. **RPi5 thermals.** Throttle starts at 80°C (2.4→1.5GHz at 85°C). → active cooler mandatory; int8 quantization; adaptive frame rate (e.g., 5–10 FPS detector, full rate only during motion); optional Hailo-8L (~27 FPS YOLOv8s-pose int8 vs ~0.5 FPS CPU per Seeed benchmark) if CPU-only proves too slow — but CPU-only is the design target.
5. **Fainting / non-fall danger has no labeled datasets.** → define labels ourselves (collapse-without-impact, prolonged immobility, distress posture); mine OOPS + synthetic generation; this is a genuine research contribution area.

## 5. Work plan

**Phase 1 — Data foundation (week 1–2)**
- Download OmniFall (8 staged sets), OOPS-Fall, CEPDOF, WEPDTOF, HABBOF, MW-R.
- Build unified label schema (fall / faint-collapse / lying-immobile / distress / normal) + conversion scripts.
- Implement FED fisheye augmentation (equidistant r = f·θ) with keypoint/box transform; visual sanity checks.

**Phase 2 — Baseline pipeline on gpu0 (week 2–4)**
- Train YOLO-pose (n-size) with FED aug, two-stage synth→real curriculum; fisheye person-detection eval on CEPDOF/WEPDTOF.
- Train temporal head (LSTM and light GCN, compare) on keypoint tracks from OmniFall, cross-view splits.
- Honest eval: OOPS-Fall sensitivity/specificity + cross-view balanced accuracy.

**Phase 3 — Edge port & 24/7 bench (week 4–6)**
- Export int8: compare NCNN vs ONNX Runtime vs LiteRT on RPi5; measure FPS, RAM, sustained temperature over 24h+.
- Frame-sampling / motion-gating strategy; false-alarms/day measurement on continuous footage.

**Phase 4 — VLM verifier + hard cases (week 6–8)**
- SmolVLM-256M/500M GGUF on Pi: latency per event-clip verification (<5–10s target); fine-tune with QLoRA on gpu0 using fall/no-fall clips if zero-shot is weak.
- Fainting/immobility label mining + retraining; occlusion and low-light hard-case rounds.

## 6. Open questions to answer experimentally

1. Actual RPi5 FPS/RAM/thermal for our exact stack (no verified public benchmark exists) — CPU-only vs Hailo-8L, and which runtime wins.
2. Does FED fisheye augmentation transfer from traffic detection (where it's proven) to human pose + fall classification?
3. Is event-triggered SmolVLM verification fast and accurate enough on Pi CPU to pay for its RAM?
4. How to label/detect faint & non-fall danger — no existing dataset covers this.

## 7. Primary sources

- FallNet: sciencedirect.com/science/article/abs/pii/S1746809425014533
- OmniFall + OOPS-Fall: arxiv.org/html/2505.19889v1
- RAPiD + CEPDOF: ar5iv.labs.arxiv.org/html/2005.11623 · vip.bu.edu/projects/vsns/cossy/datasets/cepdof/
- WEPDTOF: vip.bu.edu/projects/vsns/cossy/datasets/wepdtof/
- Fisheye FED augmentation (AI City 2025): openaccess.thecvf.com/content/ICCV2025W/AICity/papers/Pham_Data_Augmentation_Is_All_You_Need_For_Robust_Fisheye_Object_ICCVW_2025_paper.pdf
- FishEye8K: arxiv.org/abs/2305.17449
- SmolVLM: arxiv.org/html/2504.05299v1
- VIRA-GCN: pmc.ncbi.nlm.nih.gov/articles/PMC12609388/
- LFD-YOLO: ncbi.nlm.nih.gov/pmc/articles/PMC11814241/
- QLoRA VLM on 24GB: ncbi.nlm.nih.gov/pmc/articles/PMC12669176/
- RPi5 thermals: raspberrypi.com/news/heating-and-cooling-raspberry-pi-5/
