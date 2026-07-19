# The Commercial Bar for a Smart Security/Safety Monitoring Product

Evidence-backed target thresholds a camera+AI 24/7 monitoring product must hit to be
viable, by market segment — and how our fall-detection prototype measures up. Sourced from
a 22-source, adversarially-verified research pass (July 2026).

## The one metric that kills products: FALSE-ALARM RATE

False alarms are **94–98% of all alarm calls** to police and cost emergency services
**~$1.8B/year** — so the entire regulatory/commercial system is built to punish them:
- **NPCC "three-strikes" (UK): 3 unconfirmed false alarms in 12 months → police response
  (URN) WITHDRAWN.** This is the hard death line for a monitored product.
- **Non-response policies (US):** police won't dispatch on an *unverified* alarm at all.
- **Municipal fines:** ~$150 avg/false alarm; LA $267→$717 escalating; Seattle $115;
  Boston up to $200/day until fixed; permit revocation for repeat offenders.
- **Central station burden:** ~15s to assess each false alarm; thousands/day → operator
  fatigue and missed real alerts.

**What "good" looks like:** the best AI-analytics vendors claim **90–95% false-alarm
REDUCTION** (Actuate 95%+, Ambient.ai 90–95%). Note: bigger claims are marketing —
Scylla's "99.95%" and a "93%" case study were **refuted** in verification. And tellingly,
**no major vendor (Verkada, Ambient, Deep Sentinel) publishes an absolute false-alarm rate
or detection-accuracy (POD) number** — the industry sells reduction %, human verification,
and SLAs, not raw model accuracy.

## Concrete target thresholds by segment

| Requirement | Residential | Commercial (EN 50131 Grade 3) | Industrial / Perimeter | Safety-critical (fall/hazard) |
|---|---|---|---|---|
| **Detection (POD/recall)** | "high", unpublished | unpublished; dual-tech sensor confirmation | **POD > 95%** (fiber PIDS: climb/cut/lift) | effectively ~100%; missed event = liability |
| **False alarms** | avoid 3-strikes / fines | insurer + police-response grade | documented FAR in RFP; NAR bounded | near-zero tolerated |
| **Alarm verification** | video/human verify to get response | required (confirmed alarms only) | required | required |
| **Time-to-response** | guard <30–60s (Deep Sentinel SLA) | monitoring-center SLA (UL 827) | early detection | fast, but <10s clinically OK for falls |
| **Certification** | none / basic | **EN 50131 Grade 3, UL 827, SIA CP-01** | CPNI/PIDS, cyber (UL 2900) | **IEC 61508 SIL** and/or **FDA/CE medical device** |
| **Uptime/tamper** | best-effort | tamper detection, 2 signalling paths, 12–24h battery | MTBF documented, EMI-immune | high-integrity |
| **Environment** | indoor | — | **IP66–69, IK10/11, −50…+65°C, WDR, IR night** | — |
| **Privacy** | consumer notice | **GDPR: DPIA (legal), Art 6 basis, data minimisation** | + | + medical data rules |

Key structural facts:
- **A system's EN 50131 grade = its lowest-graded component.** One weak link caps the whole
  product. Grade 3 (dual-tech sensors, two signalling paths, tamper, police response) is
  the common bar for an insurer-approved commercial system.
- **Safety-critical is a different universe.** IEC 61508 SIL for *continuous-mode* functions
  demands a dangerous-failure rate of **1e-5 to 1e-9 per hour** (SIL1→4) — i.e. one
  dangerous failure per 100,000 to 1,000,000,000 operating hours. A fall detector marketed
  as a *safety* function faces this plus **medical-device regulation (FDA/CE)**.
- **VLM-based safety detection is modest even at SOTA:** the best PPE-compliance VLM
  (Clip2Safety) hits only **~72% accuracy, AUC 0.76** — so our numbers are not out of line
  with the research frontier, but the frontier itself is below the commercial bar.

## Where OUR prototype stands vs the bar

| Metric | Ours (best real number) | Commercial bar | Gap |
|---|---|---|---|
| Recall / POD | 0.83 in-the-wild, 0.90 overhead | >0.95 (industrial); ~1.0 (safety) | **below** |
| Specificity / false alarms | 0.72 in-the-wild, ~1.0 tiny test | must avoid 3-strikes/12mo → effectively per-decision FP ≪ 1% at 24/7 rates | **orders of magnitude short** |
| Certification | none | EN 50131 G3 / UL / SIL / FDA-CE | **none** |
| Environmental | indoor synthetic only | IP66-69, −50…+65°C, night, occlusion | **untested** |
| Validation scale | 40–300 clips | thousands of hrs, real elderly/site data | **far short** |

**The binding constraint is the false-alarm rate at 24/7 scale, exactly as flagged.** At
0.72–0.95 specificity and even one inference/second motion-gated, a single camera produces
far more than 3 false alarms/year — it would lose police response and rack up fines. The
commercial system solves this NOT with a better single model but with **layered
verification**: motion gating + multi-model/temporal confirmation + **human-in-the-loop
guards** (Deep Sentinel/Ambient all use human verification) + per-install tuning. Raw model
specificity is necessary but never sufficient.

## What this means for the product path
1. **You cannot ship on model accuracy alone.** Every viable competitor wraps the model in
   human verification and/or dual-technology confirmation to hit the false-alarm bar. Plan
   for a verification layer, not just a better VLM.
2. **Pick the regulatory lane deliberately.** "Safety/medical fall detection" invokes
   FDA/CE + potentially SIL — very expensive. "Caregiver alerting aid / situational
   awareness" avoids that lane and is the realistic entry.
3. **Certification is a system property, not a model property** (EN 50131 lowest-component
   rule; UL 827 for monitoring). Budget for it early.
4. **The synthetic-data recipe is still the differentiator** — it addresses the
   *environmental/viewpoint robustness* the bar demands (IP-rated cameras see infinite
   scenes) more cheaply than collecting real data per site.

## Verification stack + achievable false-alarm rate (modelled)
Since raw model accuracy can't hit the bar, we model the layered stack
(`scripts/false_alarm_model.py`, implemented in `deploy/verification.py`). Assuming a home
camera whose motion gate fires ~200x/day:

| Scenario | After full automated stack | + human-in-loop |
|---|---|---|
| Our in-the-wild spec (0.72) | ~178 FP/year | — |
| Our synthetic-held-out spec (0.93) | ~23 FP/year | **~0 (meets ≤3/yr)** |
| Improved spec (0.98) | ~9 FP/year | **~0 (meets)** |

Layer leverage (per-inference FP → alert):
1. **Motion gating** — tens-of-thousands → ~200 inferences/day.
2. **N-of-M temporal confirmation** — ~3–11x (bigger when spec is already high).
3. **Confidence gate** — ~3x.
4. **Persistence timer (person must STAY down ~20s)** — the fall-vs-transient
   discriminator and highest-leverage *automated* layer; a real fall persists on the floor,
   bending/sitting/kneeling doesn't. Costs ~20–26s latency (clinically fine for falls).
5. **Per-install calibration** — ~5–6x; removes site-specific persistent confusers.
6. **Human-in-the-loop verification** — the only thing that reliably clears the ≤3/yr bar;
   a guard reviews the few candidates/day and dismisses false ones. This is what Deep
   Sentinel / Ambient actually do.

**Conclusion:** pure automation gets from ~56 FP/day to well under 1/day, but reaching the
≤3/YEAR bar reliably needs either near-perfect specificity (~0.99, which we don't have) OR
a human-verification layer. The realistic product is **layered automation that filters to a
few candidates/day + light human confirmation** — matching every viable competitor. The
persistence timer is the cheapest high-impact win and is implemented in
`deploy/verification.py`.

## Verified sources
SIA CP-01 (securityindustry.org) · UL 827 (intertek.com) · EN 50131 grades
(businesswatchgroup.co.uk, norbain.com) · false-alarm fines/3-strikes
(deepsentinel.com/blogs) · POD>95% fiber PIDS (bandweaver.com) · IEC 61508 SIL
(en.wikipedia.org/wiki/Safety_integrity_level) · Ambient.ai / Actuate / Deep Sentinel /
Verkada / Avigilon claims (vendor sites) · UK GDPR CCTV (ico.org.uk) · PPE VLM
(arxiv.org/pdf/2408.07146) · Pelco extreme-weather (pelco.com).
