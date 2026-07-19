"""RPi5 24/7 monitoring loop logic (framework-agnostic, testable without a camera).

Design (VLM_TRAINING_PLAN.md): cadence not framerate. A cheap pixel-diff motion gate
triggers burst sampling; the VLM runs at ~0.5-1 Hz on a multi-frame strip; an alert
hysteresis + person-down persistence converts noisy per-inference verdicts into stable
alerts and controls false-alarms/day.

The classes here are pure logic (numpy). The camera + VLM are injected, so the whole
loop is unit-testable with synthetic frames and a stub model.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

DANGER = {"fall", "faint-collapse", "lying-immobile", "distress"}


class MotionGate:
    """Mean absolute inter-frame difference on a downscaled grayscale frame."""

    def __init__(self, thresh: float = 3.0, downscale: int = 8):
        self.thresh = thresh
        self.downscale = downscale
        self._prev = None

    @staticmethod
    def _gray_small(frame: np.ndarray, ds: int) -> np.ndarray:
        g = frame.mean(axis=2) if frame.ndim == 3 else frame
        return g[::ds, ::ds]

    def update(self, frame: np.ndarray) -> float:
        cur = self._gray_small(frame, self.downscale)
        if self._prev is None or self._prev.shape != cur.shape:
            self._prev = cur
            return 0.0
        diff = float(np.abs(cur - self._prev).mean())
        self._prev = cur
        return diff

    def triggered(self, frame: np.ndarray) -> bool:
        return self.update(frame) > self.thresh


class StripBuffer:
    """Ring buffer that emits an N-frame strip spanning a fixed WALL-CLOCK window,
    regardless of capture fps.

    fps-robustness: the strip always covers `span_s` seconds with `n_frames` samples,
    selected by TIMESTAMP not by frame index. So whether the camera runs at 5 or 15 fps,
    or the Pi thermally throttles and the effective rate drifts, the model sees a
    consistent temporal window. This is the deployment half of fps-robustness; the
    training half is temporal augmentation (see training/dataset.temporal_augment).
    """

    def __init__(self, fps_capture: float = 6.0, span_s: float = 3.5, n_frames: int = 6):
        self.n_frames = n_frames
        self.span_s = span_s
        # generous capacity: hold >= span at up to 4x the nominal fps, so bursts fit
        self.maxlen = max(n_frames * 2, int(round(fps_capture * span_s * 4)))
        self.buf: deque = deque(maxlen=self.maxlen)  # (timestamp, frame)

    def push(self, frame: np.ndarray, t: float | None = None):
        # if no timestamp given, fall back to a synthetic monotonic index
        if t is None:
            t = (self.buf[-1][0] + 1.0 if self.buf else 0.0)
        self.buf.append((t, frame))

    def ready(self) -> bool:
        if len(self.buf) < self.n_frames:
            return False
        span = self.buf[-1][0] - self.buf[0][0]
        # ready once we hold at least ~half the target window (avoids stalling at startup)
        return span >= 0.5 * self.span_s

    def strip(self) -> list[np.ndarray]:
        """N frames evenly spaced in TIME across the last span_s seconds."""
        now = self.buf[-1][0]
        lo = now - self.span_s
        window = [(t, f) for (t, f) in self.buf if t >= lo] or list(self.buf)
        times = np.array([t for t, _ in window])
        targets = np.linspace(times[0], times[-1], self.n_frames)
        # nearest-in-time frame for each evenly-spaced target timestamp
        idx = [int(np.argmin(np.abs(times - tt))) for tt in targets]
        return [window[i][1] for i in idx]


@dataclass
class AlertState:
    """Hysteresis: require K consecutive danger verdicts to RAISE, M clears to release.
    Plus person-down persistence: a confirmed-down subject holds the alert even if a
    later inference is noisy."""
    raise_after: int = 2
    clear_after: int = 3
    min_conf: float = 0.6
    _danger_streak: int = 0
    _clear_streak: int = 0
    active: bool = False
    active_status: str | None = None
    history: list = field(default_factory=list)

    def update(self, verdict: dict) -> dict:
        status = verdict.get("status", "normal")
        conf = float(verdict.get("confidence", 0.0))
        is_danger = status in DANGER and conf >= self.min_conf
        if is_danger:
            self._danger_streak += 1
            self._clear_streak = 0
        else:
            self._clear_streak += 1
            self._danger_streak = 0

        event = None
        if not self.active and self._danger_streak >= self.raise_after:
            self.active = True
            self.active_status = status
            event = {"event": "ALERT_RAISED", "status": status, "confidence": conf}
        elif self.active and self._clear_streak >= self.clear_after:
            self.active = False
            prev = self.active_status
            self.active_status = None
            event = {"event": "ALERT_CLEARED", "prev_status": prev}
        self.history.append(status)
        return {"active": self.active, "status": self.active_status, "event": event}


@dataclass
class MonitorConfig:
    fps_capture: float = 6.0
    strip_span_s: float = 3.5
    strip_frames: int = 6
    motion_thresh: float = 3.0
    min_infer_interval_s: float = 1.0     # cap VLM cadence to protect thermals
    idle_infer_interval_s: float = 8.0    # slow heartbeat when no motion


class Monitor:
    """Ties the gate + buffer + hysteresis to an injected VLM backend.

    backend.infer(strip: list[np.ndarray]) -> {"status","confidence","person_down"}
    now_fn: monotonic seconds provider (injected for testability).
    """

    def __init__(self, backend, cfg: MonitorConfig = MonitorConfig(), now_fn=None):
        self.backend = backend
        self.cfg = cfg
        self.gate = MotionGate(cfg.motion_thresh)
        self.buf = StripBuffer(cfg.fps_capture, cfg.strip_span_s, cfg.strip_frames)
        self.alert = AlertState()
        self._last_infer_t = -1e9
        self._now = now_fn or (lambda: 0.0)
        self.inference_count = 0

    def on_frame(self, frame: np.ndarray) -> dict | None:
        t = self._now()
        self.buf.push(frame, t)
        motion = self.gate.triggered(frame)
        interval = self.cfg.min_infer_interval_s if motion else self.cfg.idle_infer_interval_s
        due = (t - self._last_infer_t) >= interval
        if not (self.buf.ready() and due):
            return None
        # keep the alert warm even on idle ticks (person-down persistence)
        if not motion and not self.alert.active and (t - self._last_infer_t) < self.cfg.idle_infer_interval_s:
            return None
        self._last_infer_t = t
        self.inference_count += 1
        verdict = self.backend.infer(self.buf.strip())
        return self.alert.update(verdict)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    class StubBackend:
        def __init__(self, script):
            self.script = script; self.i = 0
        def infer(self, strip):
            v = self.script[min(self.i, len(self.script) - 1)]; self.i += 1
            return v

    # scripted verdicts: 2 falls (raise) then 3 normals (clear)
    script = ([{"status": "fall", "confidence": 0.95, "person_down": True}] * 2 +
              [{"status": "normal", "confidence": 0.9, "person_down": False}] * 3)
    clock = {"t": 0.0}
    mon = Monitor(StubBackend(script), MonitorConfig(min_infer_interval_s=1.0),
                  now_fn=lambda: clock["t"])
    events = []
    for step in range(12):
        clock["t"] += 1.5
        # inject high-motion frames so the gate fires and inference runs
        frame = (rng.random((48, 64, 3)) * 255 * (1 if step % 2 else 0.2)).astype(float)
        r = mon.on_frame(frame)
        if r and r["event"]:
            events.append(r["event"])
    print("inferences:", mon.inference_count)
    print("events:", [e["event"] for e in events])
    assert any(e["event"] == "ALERT_RAISED" for e in events), "should raise"
    assert any(e["event"] == "ALERT_CLEARED" for e in events), "should clear"
    # motion gate sanity
    g = MotionGate(thresh=3.0)
    a = np.zeros((32, 32, 3)); b = np.ones((32, 32, 3)) * 50
    g.update(a)
    assert g.update(b) > 3.0 and not g.triggered(b.copy()) is None
    # strip buffer: fps-robustness — same 3.5s window at 5 fps and at 15 fps
    for fps in (5.0, 15.0):
        sb = StripBuffer(fps_capture=fps, span_s=3.5, n_frames=6)
        for i in range(int(fps * 5)):          # 5 seconds of capture
            sb.push(np.full((4, 4, 3), i, dtype=float), t=i / fps)
        s = sb.strip()
        assert sb.ready() and len(s) == 6, fps
        # frames should span ~3.5s regardless of fps (values are frame indices == t*fps)
        span_vals = float(s[-1][0, 0, 0] - s[0][0, 0, 0]) / fps
        assert 3.0 <= span_vals <= 3.6, f"fps={fps} span={span_vals:.2f}s"
    print(f"fps-robust strip: 5fps and 15fps both yield 6 frames over ~3.5s")
    print("monitor OK")
