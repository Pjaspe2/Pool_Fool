import numpy as np
import pytest

from pool_fool.desktop.physics.ghost_ball import (
    compute_ghost_ball,
    pick_object_ball,
    segment_blocked_by_balls,
    solve_shot,
)
from pool_fool.shared.table import TableSpec


TABLE = TableSpec(width_mm=1270.0, length_mm=2540.0, ball_radius_mm=28.575)


def test_ghost_ball_collinear():
    cue = np.array([100.0, 635.0])
    obj = np.array([500.0, 635.0])
    aim = np.array([1.0, 0.0])
    r = TABLE.ball_radius_mm
    result = compute_ghost_ball(cue, aim, obj, TABLE, other_balls=[])
    assert result.valid
    expected_ghost = obj - 2 * r * np.array([1.0, 0.0])
    np.testing.assert_allclose(result.ghost, expected_ghost, atol=0.5)
    np.testing.assert_allclose(result.cue, cue)
    np.testing.assert_allclose(result.object_ball, obj)


def test_pick_nearest_on_ray():
    cue = np.array([0.0, 0.0])
    aim = np.array([1.0, 0.0])
    objects = [
        np.array([800.0, 50.0]),
        np.array([400.0, 5.0]),
        np.array([600.0, 0.0]),
    ]
    idx = pick_object_ball(cue, aim, objects, ball_radius_mm=28.575, angle_threshold_deg=15.0)
    assert idx == 1


def test_blocked_by_third_ball():
    cue = np.array([100.0, 100.0])
    ghost = np.array([300.0, 100.0])
    blocker = np.array([200.0, 100.0])
    assert segment_blocked_by_balls(cue, ghost, [blocker], 28.575)


def test_solve_shot_no_object():
    cue = np.array([100.0, 100.0])
    aim = np.array([0.0, 1.0])
    result = solve_shot(cue, aim, [], TABLE)
    assert not result.valid


def test_solve_shot_full():
    cue = np.array([200.0, 635.0])
    obj = np.array([600.0, 635.0])
    objects = [obj, np.array([1200.0, 635.0])]
    aim = obj - cue
    result = solve_shot(cue, aim, objects, TABLE)
    assert result.valid
    assert len(result.polyline()) == 3
