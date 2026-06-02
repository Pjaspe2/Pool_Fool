import numpy as np

from pool_fool.desktop.vision.balls import DetectedBall
from pool_fool.desktop.vision.yolo_support import cluster_detections, filter_ball_sizes
from pool_fool.shared.table import TableSpec


def _ball(x_mm: float, y_mm: float, r_px: float = 12.0) -> DetectedBall:
    return DetectedBall(
        center_px=(100.0, 100.0),
        center_mm=np.array([x_mm, y_mm]),
        radius_px=r_px,
        is_cue=False,
        brightness=200.0,
        bbox_px=(90.0, 90.0, 110.0, 110.0),
    )


def test_cluster_merges_nearby():
    a = _ball(0, 0)
    b = _ball(15, 10)
    out = cluster_detections([a, b], merge_mm=40.0)
    assert len(out) == 1


def test_filter_rejects_wrong_size():
    table = TableSpec(1270.0, 2540.0, 28.575)
    H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    H_inv = np.linalg.inv(H)
    tiny = _ball(100, 100, r_px=2.0)
    huge = _ball(200, 200, r_px=80.0)
    ok = _ball(300, 300, r_px=22.0)
    out = filter_ball_sizes([tiny, huge, ok], H_inv=H_inv, table=table)
    assert len(out) == 1
    assert out[0].center_mm[0] == 300
