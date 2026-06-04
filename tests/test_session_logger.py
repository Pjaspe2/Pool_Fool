import json
from pathlib import Path

import numpy as np

from pool_fool.desktop.physics.pocket_shot import compute_pocket_shot
from pool_fool.desktop.session.logger import SessionLogger
from pool_fool.desktop.vision.balls import DetectedBall
from pool_fool.shared.table import TableSpec
from pool_fool.shared.table_layout import PocketSpec


TABLE = TableSpec(width_mm=1000.0, length_mm=2000.0, ball_radius_mm=28.575)


def _ball(x: float, y: float, *, is_cue: bool = False, track_id: int = 0) -> DetectedBall:
    c = np.array([x, y], dtype=np.float64)
    return DetectedBall(
        center_px=(0.0, 0.0),
        center_mm=c,
        radius_px=10.0,
        is_cue=is_cue,
        brightness=200.0,
        track_id=track_id,
    )


def test_session_logger_writes_tracks_and_events(tmp_path: Path):
    logger = SessionLogger(tmp_path, meta={"test": True})
    balls = [_ball(100, 500, is_cue=True), _ball(800, 500, track_id=1)]
    logger.maybe_log_tracks(balls, stationary=True, interval_s=0.0)
    logger.maybe_log_tracks(balls, stationary=True, interval_s=0.0)

    cue = np.array([100.0, 500.0])
    obj = np.array([800.0, 500.0])
    pocket = PocketSpec("corner_tr", (1900.0, 100.0), "corner")
    shot = compute_pocket_shot(cue, obj, pocket, TABLE, object_index=0, other_balls=[obj])
    logger.log_shot_outcome("made", target_pocket_id="corner_tr", shot=shot, balls=balls)
    logger.close()

    tracks = logger.paths.tracks_path.read_text(encoding="utf-8").strip().splitlines()
    events = logger.paths.events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(tracks) == 2
    assert len(events) == 1
    ev = json.loads(events[0])
    assert ev["outcome"] == "made"
    assert ev["target_pocket_id"] == "corner_tr"
    assert ev["prediction"]["valid"] == shot.valid
    meta = json.loads(logger.paths.meta_path.read_text(encoding="utf-8"))
    assert meta["test"] is True
