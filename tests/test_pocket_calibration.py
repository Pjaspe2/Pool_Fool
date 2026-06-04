from pathlib import Path

import numpy as np

from pool_fool.shared.pocket_calibration import load_calibrated_pockets, save_calibrated_pockets
from pool_fool.shared.table_layout import PocketSpec, TableLayout


def test_save_load_pockets(tmp_path: Path):
    pockets = (
        PocketSpec("corner_tl", (10.0, 20.0), "corner"),
        PocketSpec("side_right", (500.0, 450.0), "side"),
    )
    path = tmp_path / "pockets.npz"
    save_calibrated_pockets(path, pockets)
    loaded = load_calibrated_pockets(path)
    assert loaded is not None
    assert len(loaded) == 2
    assert loaded[0].id == "corner_tl"
    assert loaded[1].center_mm == (500.0, 450.0)


def test_resolve_prefers_calibrated_pockets(tmp_path: Path):
    cfg = {
        "paths": {"pockets": "pockets.npz"},
        "table": {
            "length_mm": 2000,
            "width_mm": 1000,
            "ball_radius_mm": 28,
            "use_calibrated_pockets": True,
            "pockets_from_play_region": True,
        },
    }
    pockets = (PocketSpec("corner_tl", (99.0, 88.0), "corner"),)
    save_calibrated_pockets(tmp_path / "pockets.npz", pockets)
    layout = TableLayout.resolve(cfg, tmp_path, None)
    assert layout.pockets[0].center_mm == (99.0, 88.0)
