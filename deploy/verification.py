"""Verification stack that wraps the VLM to hit the commercial false-alarm bar.

Raw model specificity is necessary but never sufficient (COMMERCIAL_BAR.md): every viable
competitor layers verification. This composes the layers, highest-leverage first:

  1. confidence gate     - ignore low-confidence danger verdicts
  2. N-of-M temporal     - need N danger verdicts in the last M inferences (kills flicker)
  3. persistence timer   - the person must STAY down for T seconds before we alert. This is
                           the fall-vs-transient discriminator: a real fall persists on the
                           floor; bending/sitting/kneeling does not. Highest-leverage
                           automated layer (see false_alarm_model.py).
  4. human-review queue  - passing candidates are emitted for optional human confirmation
                           (what Deep Sentinel / Ambient use to actually clear the bar).

Pure logic; injected clock; unit-testable. Feeds off monitor.Monitor verdicts.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

DANGER = {"down", "fall", "faint-collapse", "lying-immobile", "distress", "danger"}


@dataclass
class VerifyConfig:
    min_conf: float = 0.7           # confidence gate
    n_of_m: tuple[int, int] = (3, 4)  # >=N danger verdicts in last M inferences
    persist_seconds: float = 20.0    # danger state must hold this long before alerting
    require_human: bool = True       # emit CANDIDATE for human review vs auto-ALERT


@dataclass
class VerifyState:
    cfg: VerifyConfig = field(default_factory=VerifyConfig)
    _window: deque = field(default_factory=lambda: deque(maxlen=8))  # recent (is_danger)
    _danger_since: float | None = None   # when the current danger run started
    alerted: bool = False

    def update(self, verdict: dict, t: float) -> dict:
        """Feed one model verdict at time t. Returns {action, ...} where action is
        'none' | 'candidate' (queue for human) | 'alert' (auto-dispatch)."""
        status = verdict.get("status", "normal")
        conf = float(verdict.get("confidence", 0.0))
        is_danger = status in DANGER and conf >= self.cfg.min_conf

        m = self.cfg.n_of_m[1]
        self._window = deque(list(self._window)[-(m - 1):] + [is_danger], maxlen=m)
        n_recent = sum(self._window)
        confirmed = n_recent >= self.cfg.n_of_m[0]

        # persistence tracking on the confirmed danger state
        if confirmed:
            if self._danger_since is None:
                self._danger_since = t
            held = t - self._danger_since
        else:
            self._danger_since = None
            self.alerted = False
            held = 0.0

        action = "none"
        if confirmed and held >= self.cfg.persist_seconds and not self.alerted:
            self.alerted = True
            action = "candidate" if self.cfg.require_human else "alert"
        return {"action": action, "status": status, "held_s": round(held, 1),
                "n_recent": n_recent}


if __name__ == "__main__":
    cfg = VerifyConfig(min_conf=0.7, n_of_m=(3, 4), persist_seconds=20.0, require_human=False)

    def run(script, dt=2.0):
        st = VerifyState(cfg=cfg)
        t = 0.0; actions = []
        for v in script:
            r = st.update(v, t); actions.append(r["action"]); t += dt
        return actions

    D = {"status": "down", "confidence": 0.95}
    N = {"status": "normal", "confidence": 0.9}
    LOW = {"status": "down", "confidence": 0.4}   # low-conf danger -> ignored

    # transient confuser: person bends down for ~6s then normal -> NO alert
    transient = [N, D, D, D, N, N, N]            # 3 danger @2s = 6s < 20s persist
    assert "alert" not in run(transient), "transient should not alert"

    # real fall: person goes down and STAYS down -> alert after ~20s
    fall = [N] + [D] * 15                          # 15 danger @2s = 30s > 20s
    acts = run(fall)
    assert "alert" in acts, "sustained down should alert"
    print("first alert at inference index:", acts.index("alert"), "(~",
          acts.index("alert") * 2, "s)")

    # low-confidence flicker -> never confirmed
    assert "alert" not in run([LOW] * 15), "low-conf should be gated"
    print("verification OK: transient rejected, sustained fall alerts, low-conf gated")
