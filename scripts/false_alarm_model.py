"""False-alarm cascade model — estimate achievable false-alarms/day/year for the fall
detector wrapped in a verification stack, from measured per-inference specificity.

Commercial bar (COMMERCIAL_BAR.md): ~<=3 false alarms / camera / YEAR to keep police
response / avoid fines. We start from our measured per-inference specificity and apply the
verification layers, honestly separating TRANSIENT confusers (bend/sit/kneel — killed by
temporal + persistence) from PERSISTENT scene confusers (a coat-rack that looks like a
person — only killed by per-install calibration or a human).

Runnable: `python scripts/false_alarm_model.py`
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Scenario:
    name: str
    spec: float               # measured per-inference specificity on NORMAL activity
    motion_events_per_day: int  # times the motion gate fires (each -> a burst inference)
    # what fraction of per-inference false positives are TRANSIENT (gone in seconds)
    transient_frac: float = 0.85
    # --- verification layer knobs ---
    n_of_m: tuple[int, int] = (3, 4)   # need N danger verdicts out of M consecutive
    conf_threshold_gain: float = 3.0   # FP reduction from requiring high confidence
    persistence_reduction: float = 20.0  # 'down must persist T s' kills transient FPs
    calibration_reduction: float = 6.0   # per-install 'normal' removes persistent FPs
    human_in_loop: bool = False          # guard reviews candidates -> ~0 dispatched FPs
    human_catch: float = 0.98            # fraction of remaining FPs a guard dismisses


def n_of_m_reduction(p_fp: float, n: int, m: int) -> float:
    """Approx reduction factor from requiring >=n danger verdicts in m consecutive
    inferences, assuming TRANSIENT FPs are ~independent across inferences. Returns the
    multiplicative FP-rate factor (p_fp^n scaling)."""
    from math import comb
    # P(>=n of m) for independent Bernoulli(p_fp)
    p_conf = sum(comb(m, k) * p_fp**k * (1 - p_fp) ** (m - k) for k in range(n, m + 1))
    return p_conf / max(p_fp, 1e-9)     # factor relative to single-inference p_fp


def estimate(s: Scenario) -> dict:
    p_fp = 1 - s.spec
    base_fp_day = p_fp * s.motion_events_per_day
    # split into transient vs persistent
    trans = base_fp_day * s.transient_frac
    persist = base_fp_day * (1 - s.transient_frac)

    steps = [("baseline (motion-gated)", base_fp_day)]

    # temporal N-of-M confirmation: strong on transient, weak on persistent (correlated)
    n, m = s.n_of_m
    nm = n_of_m_reduction(p_fp, n, m)
    trans *= nm
    persist *= min(nm * 0.5 + 0.5, 1.0)   # persistent FPs recur -> confirmation helps little
    steps.append((f"+ {n}-of-{m} temporal confirm", trans + persist))

    # confidence threshold: applies to both
    trans /= s.conf_threshold_gain
    persist /= s.conf_threshold_gain
    steps.append(("+ confidence threshold", trans + persist))

    # persistence timer ('person still down after T s'): kills transient, not persistent
    trans /= s.persistence_reduction
    steps.append(("+ persistence timer (down must last)", trans + persist))

    # per-install calibration: removes persistent scene confusers
    persist /= s.calibration_reduction
    steps.append(("+ per-install calibration", trans + persist))

    fp_day = trans + persist
    if s.human_in_loop:
        fp_day *= (1 - s.human_catch)
        steps.append(("+ human-in-loop verify", fp_day))

    return {"fp_day": fp_day, "fp_year": fp_day * 365, "steps": steps,
            "meets_bar": fp_day * 365 <= 3}


def show(s: Scenario):
    r = estimate(s)
    print(f"\n=== {s.name} (per-inference spec={s.spec}) ===")
    prev = None
    for label, v in r["steps"]:
        arrow = f"  ({prev/v:.1f}x)" if prev and v > 0 else ""
        print(f"  {label:38s} {v:8.2f} FP/day{arrow}")
        prev = v
    print(f"  => {r['fp_day']:.3f} FP/day  =  {r['fp_year']:.0f} FP/year   "
          f"{'MEETS <=3/yr bar' if r['meets_bar'] else 'above bar'}")


if __name__ == "__main__":
    # motion_events_per_day: a home camera's gate fires on real activity; assume ~200/day
    show(Scenario("A. our in-the-wild spec, no human", spec=0.72, motion_events_per_day=200))
    show(Scenario("B. our synthetic held-out spec, no human", spec=0.93, motion_events_per_day=200))
    show(Scenario("C. spec 0.93 + human-in-loop", spec=0.93, motion_events_per_day=200,
                  human_in_loop=True))
    show(Scenario("D. improved spec 0.98 + human-in-loop", spec=0.98, motion_events_per_day=200,
                  human_in_loop=True))
    print("\nNote: 'motion_events_per_day' and reduction factors are engineering estimates;"
          "\nthe model shows WHICH layers matter, not a guaranteed number. Persistence timer"
          "\nand per-install calibration are the highest-leverage automated layers; the"
          "\nhuman-in-loop is what competitors use to actually clear the <=3/yr bar.")
