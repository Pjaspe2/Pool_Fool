from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pool_fool.shared.table import TableSpec


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v / n


def _perp_distance_to_ray(origin: np.ndarray, direction: np.ndarray, point: np.ndarray) -> float:
    """Distance from point to infinite ray origin + t*direction, t >= 0."""
    d = _unit(direction)
    w = point - origin
    t = float(np.dot(w, d))
    if t < 0:
        return float(np.linalg.norm(point - origin))
    closest = origin + t * d
    return float(np.linalg.norm(point - closest))


@dataclass(frozen=True)
class GhostBallResult:
    valid: bool
    cue: np.ndarray
    ghost: np.ndarray
    object_ball: np.ndarray
    object_index: int
    blocked: bool
    message: str

    def polyline(self) -> list[np.ndarray]:
        if not self.valid:
            return []
        return [self.cue.copy(), self.ghost.copy(), self.object_ball.copy()]


def pick_object_ball(
    cue: np.ndarray,
    aim: np.ndarray,
    objects: list[np.ndarray],
    *,
    ball_radius_mm: float,
    angle_threshold_deg: float = 12.0,
    exclude_indices: set[int] | None = None,
) -> int | None:
    """
    Choose object ball nearest to aim ray ahead of cue, within lateral threshold.
    """
    if not objects:
        return None
    u = _unit(aim)
    best_idx: int | None = None
    best_t = float("inf")
    cos_thresh = float(np.cos(np.radians(angle_threshold_deg)))
    exclude = exclude_indices or set()

    for i, obj in enumerate(objects):
        if i in exclude:
            continue
        w = obj - cue
        t = float(np.dot(w, u))
        if t <= ball_radius_mm * 0.5:
            continue
        dist = _perp_distance_to_ray(cue, u, obj)
        if dist > 2.0 * ball_radius_mm + 1.0:
            continue
        # Must be roughly in front
        wn = _unit(w)
        if float(np.dot(wn, u)) < cos_thresh:
            continue
        if t < best_t:
            best_t = t
            best_idx = i

    return best_idx


def segment_blocked_by_balls(
    a: np.ndarray,
    b: np.ndarray,
    centers: list[np.ndarray],
    ball_radius_mm: float,
    *,
    ignore_indices: set[int] | None = None,
) -> bool:
    """True if any ball (inflated by diameter) intersects segment a-b."""
    ignore = ignore_indices or set()
    ab = b - a
    len_ab = float(np.linalg.norm(ab))
    if len_ab < 1e-6:
        return False
    u = ab / len_ab
    inflate = 2.0 * ball_radius_mm

    for i, c in enumerate(centers):
        if i in ignore:
            continue
        ac = c - a
        t = float(np.clip(np.dot(ac, u), 0.0, len_ab))
        closest = a + t * u
        if float(np.linalg.norm(c - closest)) < inflate - 0.5:
            return True
    return False


def compute_ghost_ball(
    cue: np.ndarray,
    aim: np.ndarray,
    object_ball: np.ndarray,
    table: TableSpec,
    *,
    other_balls: list[np.ndarray] | None = None,
    object_index: int = 0,
    cue_index: int = -1,
) -> GhostBallResult:
    """
    Ghost-ball center G = O - 2*r*u where u is unit aim from cue toward object.
    """
    r = table.ball_radius_mm
    u = _unit(aim)
    if np.linalg.norm(u) < 1e-6:
        return GhostBallResult(
            valid=False,
            cue=cue,
            ghost=cue,
            object_ball=object_ball,
            object_index=object_index,
            blocked=False,
            message="invalid aim vector",
        )

    # Aim should point from cue toward object region
    to_obj = object_ball - cue
    if float(np.dot(u, to_obj)) < 0:
        u = -u

    ghost = object_ball - 2.0 * r * u
    dist_cg = float(np.linalg.norm(cue - ghost))
    if dist_cg < r * 0.25:
        return GhostBallResult(
            valid=False,
            cue=cue,
            ghost=ghost,
            object_ball=object_ball,
            object_index=object_index,
            blocked=False,
            message="cue too close to ghost position",
        )

    others = list(other_balls or [])
    ignore = {object_index}
    if cue_index >= 0:
        ignore.add(cue_index)

    blocked = segment_blocked_by_balls(cue, ghost, others, r, ignore_indices=ignore)
    if blocked:
        return GhostBallResult(
            valid=False,
            cue=cue,
            ghost=ghost,
            object_ball=object_ball,
            object_index=object_index,
            blocked=True,
            message="line of sight blocked",
        )

    ghost = table.clip_point(ghost)
    return GhostBallResult(
        valid=True,
        cue=table.clip_point(cue),
        ghost=ghost,
        object_ball=table.clip_point(object_ball),
        object_index=object_index,
        blocked=False,
        message="",
    )


def solve_shot(
    cue: np.ndarray,
    aim: np.ndarray,
    object_balls: list[np.ndarray],
    table: TableSpec,
    *,
    angle_threshold_deg: float = 12.0,
    object_index: int | None = None,
) -> GhostBallResult:
    if object_index is None:
        picked = pick_object_ball(
            cue,
            aim,
            object_balls,
            ball_radius_mm=table.ball_radius_mm,
            angle_threshold_deg=angle_threshold_deg,
        )
        if picked is None:
            return GhostBallResult(
                valid=False,
                cue=cue,
                ghost=cue,
                object_ball=cue,
                object_index=-1,
                blocked=False,
                message="no object ball on aim",
            )
        object_index = picked

    if object_index < 0 or object_index >= len(object_balls):
        return GhostBallResult(
            valid=False,
            cue=cue,
            ghost=cue,
            object_ball=cue,
            object_index=-1,
            blocked=False,
            message="invalid object index",
        )

    obj = object_balls[object_index]
    return compute_ghost_ball(
        cue,
        aim,
        obj,
        table,
        other_balls=object_balls,
        object_index=object_index,
    )
