import time

import numpy as np

from pool_fool.desktop.vision.balls import DetectedBall
from pool_fool.desktop.vision.tracking import StationaryGate


def _ball(tid: int, x: float, y: float) -> DetectedBall:
    return DetectedBall(
        center_px=(x, y),
        center_mm=np.array([x, y]),
        radius_px=10.0,
        is_cue=False,
        brightness=100.0,
        track_id=tid,
    )


def test_stationary_after_still_frames():
    gate = StationaryGate(50.0, still_frames_required=3)
    pos = np.array([100.0, 200.0])
    for _ in range(5):
        gate.update([_ball(0, float(pos[0]), float(pos[1]))])
        time.sleep(0.02)
    assert gate.stationary


def test_moving_on_jump():
    gate = StationaryGate(30.0, still_frames_required=2)
    gate.update([_ball(0, 0.0, 0.0)])
    time.sleep(0.02)
    gate.update([_ball(0, 200.0, 0.0)])
    assert not gate.stationary
