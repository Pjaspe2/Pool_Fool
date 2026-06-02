from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class BallState:
    x_mm: float
    y_mm: float
    is_cue: bool = False
    label: str = ""

    def to_array(self) -> np.ndarray:
        return np.array([self.x_mm, self.y_mm], dtype=np.float64)


@dataclass
class ShotGuide:
    """Polyline segments in table mm: cue -> ghost -> object."""

    valid: bool
    cue_mm: list[float] = field(default_factory=list)
    ghost_mm: list[float] = field(default_factory=list)
    object_mm: list[float] = field(default_factory=list)
    object_index: int = -1
    blocked: bool = False
    message: str = ""


@dataclass
class OverlayMessage:
    """Wire format desktop -> Pi for rasterization or direct draw."""

    timestamp_ms: int
    stationary: bool
    shot: ShotGuide
    balls: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> OverlayMessage:
        d = json.loads(raw)
        shot = ShotGuide(**d["shot"])
        return cls(
            timestamp_ms=d["timestamp_ms"],
            stationary=d["stationary"],
            shot=shot,
            balls=d.get("balls", []),
        )


def shot_from_arrays(
    cue: np.ndarray,
    ghost: np.ndarray,
    obj: np.ndarray,
    valid: bool,
    object_index: int = 0,
    blocked: bool = False,
    message: str = "",
) -> ShotGuide:
    return ShotGuide(
        valid=valid,
        cue_mm=cue.tolist(),
        ghost_mm=ghost.tolist(),
        object_mm=obj.tolist(),
        object_index=object_index,
        blocked=blocked,
        message=message,
    )
