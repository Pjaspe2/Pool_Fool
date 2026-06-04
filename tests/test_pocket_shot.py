import numpy as np

from pool_fool.desktop.physics.pocket_shot import (
    compute_pocket_shot,
    pick_nearest_object_to_cue,
    solve_poc_pocket_shot,
    solve_target_pocket_shot,
    validate_cut_geometry,
)
from pool_fool.shared.table import TableSpec
from pool_fool.shared.table_layout import PocketSpec, TableLayout, default_six_pockets


TABLE = TableSpec(width_mm=1000.0, length_mm=2000.0, ball_radius_mm=28.575)


def _layout() -> TableLayout:
    pockets = default_six_pockets(2000.0, 1000.0, inset_mm=80.0)
    return TableLayout(spec=TABLE, pockets=pockets)


def test_pick_nearest_object():
    cue = np.array([0.0, 500.0])
    objects = [np.array([400.0, 500.0]), np.array([200.0, 500.0])]
    assert pick_nearest_object_to_cue(cue, objects) == 1


def test_straight_pocket_shot_valid():
    cue = np.array([100.0, 500.0])
    obj = np.array([600.0, 500.0])
    pocket = PocketSpec("corner_tr", (1200.0, 500.0), "corner")
    shot = compute_pocket_shot(cue, obj, pocket, TABLE, object_index=0, other_balls=[obj])
    assert shot.valid
    assert shot.cut_angle_deg < 5.0
    assert float(np.linalg.norm(shot.ghost - obj)) > 2 * TABLE.ball_radius_mm - 1


def test_reject_thin_cut():
    cue = np.array([100.0, 200.0])
    obj = np.array([600.0, 500.0])
    pocket = np.array([650.0, 900.0])
    ghost = obj - 2 * TABLE.ball_radius_mm * _unit(pocket - obj)
    ok, msg, cut = validate_cut_geometry(
        cue, ghost, obj, pocket, TABLE, max_cut_angle_deg=25.0
    )
    assert not ok
    assert "cut" in msg.lower()
    assert cut > 25.0


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def test_target_pocket_skips_blocked_object():
    """Pick an object with a valid cue path when another is blocked."""
    cue = np.array([100.0, 500.0])
    near = np.array([400.0, 500.0])
    far = np.array([1500.0, 500.0])
    blocker = np.array([1100.0, 480.0])
    layout = _layout()
    pocket = layout.pocket_by_id("corner_tr")
    shot = solve_target_pocket_shot(
        cue,
        [near, far, blocker],
        layout,
        TABLE,
        pocket,
        max_cut_angle_deg=48.0,
    )
    assert shot.valid
    assert shot.object_index == 0


def test_solve_target_pocket_uses_selected_pocket():
    cue = np.array([100.0, 500.0])
    obj = np.array([800.0, 500.0])
    layout = _layout()
    target = layout.pocket_by_id("side_left")
    shot = solve_target_pocket_shot(
        cue, [obj], layout, TABLE, target, max_cut_angle_deg=48.0
    )
    assert shot.pocket_id == "side_left"


def test_solve_poc_picks_open_pocket():
    cue = np.array([100.0, 500.0])
    obj = np.array([800.0, 500.0])
    objects = [obj]
    shot = solve_poc_pocket_shot(cue, objects, _layout(), TABLE)
    assert shot.valid
    assert shot.pocket_id
    assert len(shot.polyline()) == 4
