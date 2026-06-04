from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pool_fool.desktop.physics.pocket_shot import PocketShotResult
from pool_fool.desktop.vision.balls import DetectedBall


def _vec2(v: np.ndarray) -> list[float]:
    return [float(v[0]), float(v[1])]


def _shot_dict(shot: PocketShotResult) -> dict[str, Any]:
    return {
        "valid": shot.valid,
        "pocket_id": shot.pocket_id,
        "object_index": shot.object_index,
        "cut_angle_deg": round(shot.cut_angle_deg, 2),
        "blocked": shot.blocked,
        "message": shot.message,
        "cue_mm": _vec2(shot.cue),
        "ghost_mm": _vec2(shot.ghost),
        "object_mm": _vec2(shot.object_ball),
        "pocket_mm": _vec2(shot.pocket),
    }


def _balls_dict(balls: list[DetectedBall]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for b in balls:
        out.append(
            {
                "track_id": int(b.track_id),
                "is_cue": bool(b.is_cue),
                "x_mm": float(b.center_mm[0]),
                "y_mm": float(b.center_mm[1]),
            }
        )
    return out


@dataclass(frozen=True)
class SessionPaths:
    session_dir: Path
    tracks_path: Path
    events_path: Path
    meta_path: Path


class SessionLogger:
    """Semi-automated B1 logging: periodic tracks + shot outcome events."""

    def __init__(self, log_root: Path, *, meta: dict[str, Any] | None = None) -> None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.paths = SessionPaths(
            session_dir=log_root / f"session_{stamp}",
            tracks_path=log_root / f"session_{stamp}" / "tracks.jsonl",
            events_path=log_root / f"session_{stamp}" / "events.jsonl",
            meta_path=log_root / f"session_{stamp}" / "meta.json",
        )
        self.paths.session_dir.mkdir(parents=True, exist_ok=True)
        self._tracks_fp = self.paths.tracks_path.open("a", encoding="utf-8")
        self._events_fp = self.paths.events_path.open("a", encoding="utf-8")
        self._last_track_mono = 0.0
        self.event_count = 0
        self.track_count = 0
        if meta:
            self.paths.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def close(self) -> None:
        self._tracks_fp.close()
        self._events_fp.close()

    def maybe_log_tracks(
        self,
        balls: list[DetectedBall],
        *,
        stationary: bool,
        interval_s: float,
        when_moving: bool = False,
    ) -> None:
        if not balls:
            return
        if not stationary and not when_moving:
            return
        now = time.monotonic()
        if now - self._last_track_mono < interval_s:
            return
        self._last_track_mono = now
        record = {
            "t_ms": int(time.time() * 1000),
            "stationary": stationary,
            "balls": _balls_dict(balls),
        }
        self._write_line(self._tracks_fp, record)
        self.track_count += 1

    def log_shot_outcome(
        self,
        outcome: str,
        *,
        target_pocket_id: str,
        shot: PocketShotResult,
        balls: list[DetectedBall],
    ) -> dict[str, Any]:
        if outcome not in ("made", "missed"):
            raise ValueError(f"outcome must be made or missed, got {outcome!r}")
        record: dict[str, Any] = {
            "t_ms": int(time.time() * 1000),
            "outcome": outcome,
            "target_pocket_id": target_pocket_id,
            "prediction": _shot_dict(shot),
            "balls": _balls_dict(balls),
        }
        self._write_line(self._events_fp, record)
        self.event_count += 1
        return record

    @staticmethod
    def _write_line(fp, record: dict[str, Any]) -> None:
        fp.write(json.dumps(record, separators=(",", ":")) + "\n")
        fp.flush()
