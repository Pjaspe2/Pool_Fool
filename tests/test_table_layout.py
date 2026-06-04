import numpy as np

from pool_fool.shared.table_layout import TableLayout, default_six_pockets, pockets_from_play_quad
from pool_fool.shared.table import TableSpec


def test_six_pockets_inside_playing_surface():
    spec = TableSpec(width_mm=1270.0, length_mm=2540.0, ball_radius_mm=28.575)
    pockets = default_six_pockets(spec.length_mm, spec.width_mm, inset_mm=57.0)
    layout = TableLayout(spec=spec, pockets=pockets)
    assert len(layout.pockets) == 6
    for p in layout.pockets:
        x, y = p.center_mm
        assert 0 < x < spec.length_mm
        assert 0 < y < spec.width_mm


def test_pockets_from_play_quad_use_corners():
    corners = np.array(
        [[0.0, 0.0], [2000.0, 0.0], [2000.0, 900.0], [0.0, 900.0]],
        dtype=np.float64,
    )
    pockets = pockets_from_play_quad(corners, inset_fraction=0.1)
    assert len(pockets) == 6
    tl = next(p for p in pockets if p.id == "corner_tl")
    assert tl.center_mm[0] > 0 and tl.center_mm[1] > 0
    assert tl.center_mm[0] < 200 and tl.center_mm[1] < 90


def test_from_play_region_matches_border():
    from pool_fool.shared.play_region import PlayRegion

    corners = np.array([[10, 20], [1010, 25], [1005, 520], [8, 515]], dtype=np.float64)
    region = PlayRegion(corners)
    layout = TableLayout.from_play_region(
        region,
        {
            "table": {
                "width_mm": 900,
                "length_mm": 2000,
                "ball_radius_mm": 28,
                "pockets": {"inset_fraction": 0.05},
            }
        },
    )
    np.testing.assert_array_almost_equal(layout.border_corners_mm, corners)


def test_nearest_pocket_corner():
    spec = TableSpec(width_mm=1000.0, length_mm=2000.0, ball_radius_mm=28.0)
    layout = TableLayout.from_config(
        {
            "table": {
                "width_mm": 1000.0,
                "length_mm": 2000.0,
                "ball_radius_mm": 28.0,
                "pockets": {"center_inset_mm": 50.0},
            }
        }
    )
    near_tl = layout.nearest_pocket(np.array([60.0, 60.0]))
    assert near_tl.id == "corner_tl"
