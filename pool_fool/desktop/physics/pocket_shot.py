from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pool_fool.desktop.physics.ghost_ball import segment_blocked_by_balls
from pool_fool.shared.table import TableSpec
from pool_fool.shared.table_layout import PocketSpec, TableLayout


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return v
    return v / n


def pick_nearest_object_to_cue(
    cue: np.ndarray,
    objects: list[np.ndarray],
) -> int | None:
    """PoC: object ball closest to the cue ball."""
    if not objects:
        return None
    best_idx: int | None = None
    best_d = float("inf")
    for i, obj in enumerate(objects):
        d = float(np.linalg.norm(obj - cue))
        if d < best_d:
            best_d = d
            best_idx = i
    return best_idx


def cut_angle_deg(cue: np.ndarray, obj: np.ndarray, pocket: np.ndarray) -> float:
    """0° = straight pot; larger = thinner cut (angle at object: cue path vs pocket path)."""
    v_in = _unit(obj - cue)
    v_out = _unit(pocket - obj)
    dot = float(np.clip(np.dot(v_in, v_out), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def validate_cut_geometry(
    cue: np.ndarray,
    ghost: np.ndarray,
    obj: np.ndarray,
    pocket: np.ndarray,
    table: TableSpec,
    *,
    max_cut_angle_deg: float = 48.0,
) -> tuple[bool, str, float]:
    """
    Reject ghost-ball layouts that are not realistically cuttable (thin cuts, wrong side).
    """
    cut = cut_angle_deg(cue, obj, pocket)
    if cut > max_cut_angle_deg:
        return False, f"cut too thin ({cut:.0f}°)", cut

    u_out = _unit(pocket - obj)
    # Cue must not sit past the object toward the pocket (would need masse / jump)
    proj_cue = float(np.dot(cue - obj, u_out))
    if proj_cue > table.ball_radius_mm and cut > 20.0:
        return False, "cue past object (need cushion)", cut

    # Ghost must sit ahead of the cue toward the object (not behind the cue)
    to_obj = obj - cue
    if float(np.dot(ghost - cue, to_obj)) <= 0:
        return False, "ghost behind cue", cut

    # Cue → ghost → object should be one line: cue-to-ghost aligns with ghost-to-object
    v_aim = _unit(ghost - cue)
    v_shot_line = _unit(obj - ghost)
    if float(np.dot(v_aim, v_shot_line)) < 0.5:
        return False, "bad cue approach", cut

    return True, "", cut


@dataclass(frozen=True)
class PocketShotResult:
    valid: bool
    cue: np.ndarray
    ghost: np.ndarray
    object_ball: np.ndarray
    pocket: np.ndarray
    pocket_id: str
    object_index: int
    blocked: bool
    message: str
    cut_angle_deg: float = 0.0

    def polyline(self) -> list[np.ndarray]:
        if not self.valid:
            return []
        return [
            self.cue.copy(),
            self.ghost.copy(),
            self.object_ball.copy(),
            self.pocket.copy(),
        ]


def compute_pocket_shot(
    cue: np.ndarray,
    object_ball: np.ndarray,
    pocket: PocketSpec,
    table: TableSpec,
    *,
    object_index: int,
    other_balls: list[np.ndarray],
    max_cut_angle_deg: float = 48.0,
) -> PocketShotResult:
    """Ghost-ball position to send object toward pocket center (straight-line PoC)."""
    r = table.ball_radius_mm
    pocket_pt = np.array(pocket.center_mm, dtype=np.float64)
    u_out = _unit(pocket_pt - object_ball)
    if float(np.linalg.norm(u_out)) < 1e-6:
        return PocketShotResult(
            valid=False,
            cue=cue,
            ghost=object_ball,
            object_ball=object_ball,
            pocket=pocket_pt,
            pocket_id=pocket.id,
            object_index=object_index,
            blocked=False,
            message="object on pocket",
        )

    ghost = object_ball - 2.0 * r * u_out
    cut = cut_angle_deg(cue, object_ball, pocket_pt)
    ok_geom, geom_msg, cut = validate_cut_geometry(
        cue, ghost, object_ball, pocket_pt, table, max_cut_angle_deg=max_cut_angle_deg
    )
    if not ok_geom:
        return PocketShotResult(
            valid=False,
            cue=cue,
            ghost=ghost,
            object_ball=object_ball,
            pocket=pocket_pt,
            pocket_id=pocket.id,
            object_index=object_index,
            blocked=False,
            message=geom_msg,
            cut_angle_deg=cut,
        )

    ignore = {object_index}
    blocked_cue = segment_blocked_by_balls(cue, ghost, other_balls, r, ignore_indices=ignore)
    blocked_obj = segment_blocked_by_balls(
        object_ball, pocket_pt, other_balls, r, ignore_indices=ignore
    )

    if blocked_cue:
        return PocketShotResult(
            valid=False,
            cue=table.clip_point(cue),
            ghost=table.clip_point(ghost),
            object_ball=table.clip_point(object_ball),
            pocket=pocket_pt,
            pocket_id=pocket.id,
            object_index=object_index,
            blocked=True,
            message="cue path blocked",
            cut_angle_deg=cut,
        )
    if blocked_obj:
        return PocketShotResult(
            valid=False,
            cue=table.clip_point(cue),
            ghost=table.clip_point(ghost),
            object_ball=table.clip_point(object_ball),
            pocket=pocket_pt,
            pocket_id=pocket.id,
            object_index=object_index,
            blocked=True,
            message="object→pocket blocked",
            cut_angle_deg=cut,
        )

    dist_cg = float(np.linalg.norm(cue - ghost))
    if dist_cg < r * 0.25:
        return PocketShotResult(
            valid=False,
            cue=cue,
            ghost=ghost,
            object_ball=object_ball,
            pocket=pocket_pt,
            pocket_id=pocket.id,
            object_index=object_index,
            blocked=False,
            message="cue too close to ghost",
            cut_angle_deg=cut,
        )

    return PocketShotResult(
        valid=True,
        cue=cue.copy(),
        ghost=ghost.copy(),
        object_ball=object_ball.copy(),
        pocket=pocket_pt,
        pocket_id=pocket.id,
        object_index=object_index,
        blocked=False,
        message="",
        cut_angle_deg=cut,
    )


def score_pocket_candidate(shot: PocketShotResult, object_ball: np.ndarray) -> float:
    """Lower is easier. Straight, short pots score best."""
    if not shot.valid:
        return float("inf")
    dist_op = float(np.linalg.norm(shot.pocket - object_ball))
    return dist_op + 0.35 * shot.cut_angle_deg


def _best_shot_among_objects(
    cue: np.ndarray,
    object_balls: list[np.ndarray],
    pockets: tuple[PocketSpec, ...],
    table: TableSpec,
    *,
    max_cut_angle_deg: float,
) -> PocketShotResult | None:
    """Try every object × pocket; return lowest-score valid shot."""
    best_shot: PocketShotResult | None = None
    best_score = float("inf")
    all_centers = list(object_balls)

    for object_index, obj in enumerate(object_balls):
        for pocket in pockets:
            shot = compute_pocket_shot(
                cue,
                obj,
                pocket,
                table,
                object_index=object_index,
                other_balls=all_centers,
                max_cut_angle_deg=max_cut_angle_deg,
            )
            if not shot.valid:
                continue
            score = score_pocket_candidate(shot, obj)
            if score < best_score:
                best_score = score
                best_shot = shot
    return best_shot


def solve_poc_pocket_shot(
    cue: np.ndarray,
    object_balls: list[np.ndarray],
    layout: TableLayout,
    table: TableSpec,
    *,
    object_index: int | None = None,
    max_cut_angle_deg: float = 48.0,
) -> PocketShotResult:
    """
    Best object ball + pocket (distance + cut angle, unblocked paths).
    """
    if not object_balls:
        return PocketShotResult(
            valid=False,
            cue=cue,
            ghost=cue,
            object_ball=cue,
            pocket=cue,
            pocket_id="",
            object_index=-1,
            blocked=False,
            message="no object balls",
        )

    if object_index is not None:
        obj = object_balls[object_index]
        best_shot: PocketShotResult | None = None
        best_score = float("inf")
        for pocket in layout.pockets:
            shot = compute_pocket_shot(
                cue,
                obj,
                pocket,
                table,
                object_index=object_index,
                other_balls=list(object_balls),
                max_cut_angle_deg=max_cut_angle_deg,
            )
            score = score_pocket_candidate(shot, obj)
            if score < best_score:
                best_score = score
                best_shot = shot
    else:
        best_shot = _best_shot_among_objects(
            cue, object_balls, layout.pockets, table, max_cut_angle_deg=max_cut_angle_deg
        )

    if best_shot is None or not best_shot.valid:
        fallback_idx = pick_nearest_object_to_cue(cue, object_balls) or 0
        obj = object_balls[fallback_idx]
        return PocketShotResult(
            valid=False,
            cue=cue,
            ghost=cue,
            object_ball=obj,
            pocket=obj,
            pocket_id="",
            object_index=fallback_idx,
            blocked=True,
            message="no open pocket",
        )
    return best_shot


def solve_target_pocket_shot(
    cue: np.ndarray,
    object_balls: list[np.ndarray],
    layout: TableLayout,
    table: TableSpec,
    pocket: PocketSpec,
    *,
    object_index: int | None = None,
    max_cut_angle_deg: float = 48.0,
) -> PocketShotResult:
    """Nearest object ball → user-selected pocket (session / practice mode)."""
    if not object_balls:
        return PocketShotResult(
            valid=False,
            cue=cue,
            ghost=cue,
            object_ball=cue,
            pocket=np.array(pocket.center_mm, dtype=np.float64),
            pocket_id=pocket.id,
            object_index=-1,
            blocked=False,
            message="no object balls",
        )

    if object_index is not None:
        return compute_pocket_shot(
            cue,
            object_balls[object_index],
            pocket,
            table,
            object_index=object_index,
            other_balls=list(object_balls),
            max_cut_angle_deg=max_cut_angle_deg,
        )

    best_shot: PocketShotResult | None = None
    best_score = float("inf")
    for object_index, obj in enumerate(object_balls):
        shot = compute_pocket_shot(
            cue,
            obj,
            pocket,
            table,
            object_index=object_index,
            other_balls=list(object_balls),
            max_cut_angle_deg=max_cut_angle_deg,
        )
        if not shot.valid:
            continue
        score = score_pocket_candidate(shot, obj)
        if score < best_score:
            best_score = score
            best_shot = shot

    if best_shot is not None:
        return best_shot

    fallback_idx = pick_nearest_object_to_cue(cue, object_balls)
    if fallback_idx is None:
        return PocketShotResult(
            valid=False,
            cue=cue,
            ghost=cue,
            object_ball=cue,
            pocket=np.array(pocket.center_mm, dtype=np.float64),
            pocket_id=pocket.id,
            object_index=-1,
            blocked=False,
            message="no object ball",
        )
    return compute_pocket_shot(
        cue,
        object_balls[fallback_idx],
        pocket,
        table,
        object_index=fallback_idx,
        other_balls=list(object_balls),
        max_cut_angle_deg=max_cut_angle_deg,
    )
