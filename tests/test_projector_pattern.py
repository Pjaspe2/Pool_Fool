from pool_fool.desktop.calibrate.cli import _projector_pattern_corners_px


def test_pattern_corners_inset_from_edges():
    cfg = {
        "projector": {
            "display_width": 1920,
            "display_height": 1080,
            "pattern_inset_fraction": 0.12,
        }
    }
    corners = _projector_pattern_corners_px(cfg)
    assert corners[0][0] > 0 and corners[0][1] > 0
    assert corners[1][0] < 1919
    assert corners[2][0] < 1919 and corners[2][1] < 1079
