from __future__ import annotations

import time

import numpy as np

from pool_fool.desktop.vision.balls import DetectedBall


class BallTracker:
    """Match detections frame-to-frame; smooth centers; assign stable track_id."""

    def __init__(self, alpha: float = 0.35, match_gate_mm: float = 80.0) -> None:
        self.alpha = alpha
        self.match_gate_mm = match_gate_mm
        self._state: dict[int, np.ndarray] = {}
        self._next_id = 0

    def update(self, balls: list[DetectedBall]) -> list[DetectedBall]:
        if not balls:
            return balls
        used: set[int] = set()
        out: list[DetectedBall] = []
        for b in balls:
            best_id: int | None = None
            best_d = self.match_gate_mm
            for tid, pos in self._state.items():
                if tid in used:
                    continue
                d = float(np.linalg.norm(pos - b.center_mm))
                if d < best_d:
                    best_d = d
                    best_id = tid
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
                self._state[best_id] = b.center_mm.copy()
            else:
                used.add(best_id)
                prev = self._state[best_id]
                smoothed = self.alpha * b.center_mm + (1.0 - self.alpha) * prev
                self._state[best_id] = smoothed
                b.center_mm = smoothed
            b.track_id = best_id
            out.append(b)
        return out


class StationaryGate:
    """
    True when tracked balls have been slow for several consecutive frames.
    Uses track_id (not rounded position) to avoid Hough jitter false motion.
    """

    def __init__(
        self,
        velocity_threshold_mm_s: float,
        *,
        still_frames_required: int = 4,
        velocity_ema_alpha: float = 0.25,
    ) -> None:
        self.threshold = velocity_threshold_mm_s
        self.still_frames_required = still_frames_required
        self.velocity_ema_alpha = velocity_ema_alpha
        self.stationary = False
        self.max_velocity_mm_s = 0.0
        self._prev: dict[int, tuple[np.ndarray, float]] = {}
        self._vel_ema: dict[int, float] = {}
        self._still_streak = 0

    def update(self, balls: list[DetectedBall]) -> bool:
        now = time.monotonic()
        active_ids = {b.track_id for b in balls if b.track_id >= 0}

        if not balls or not active_ids:
            self._still_streak = 0
            self.stationary = False
            self.max_velocity_mm_s = 0.0
            self._prev = {k: v for k, v in self._prev.items() if k in active_ids}
            return False

        max_v = 0.0
        a = self.velocity_ema_alpha
        for b in balls:
            if b.track_id < 0:
                continue
            tid = b.track_id
            if tid in self._prev:
                prev_pos, prev_t = self._prev[tid]
                dt = now - prev_t
                if dt > 1e-4:
                    v_raw = float(np.linalg.norm(b.center_mm - prev_pos)) / dt
                    prev_ema = self._vel_ema.get(tid, v_raw)
                    v_smooth = a * v_raw + (1.0 - a) * prev_ema
                    self._vel_ema[tid] = v_smooth
                    max_v = max(max_v, v_smooth)
            self._prev[tid] = (b.center_mm.copy(), now)

        self._prev = {k: v for k, v in self._prev.items() if k in active_ids}
        self._vel_ema = {k: v for k, v in self._vel_ema.items() if k in active_ids}
        self.max_velocity_mm_s = max_v

        if max_v < self.threshold:
            self._still_streak += 1
        else:
            self._still_streak = 0

        self.stationary = self._still_streak >= self.still_frames_required
        return self.stationary
