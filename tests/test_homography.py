import numpy as np

from pool_fool.shared.homography import compute_table_homography, image_to_table
from pool_fool.shared.table import TableSpec


def test_homography_identity_corners():
    corners = [(0, 0), (100, 0), (100, 50), (0, 50)]
    dst = [(0, 0), (2540, 0), (2540, 1270), (0, 1270)]
    H = compute_table_homography(corners, dst)
    pt = image_to_table(H, (50, 25))
    assert 1200 < pt[0] < 1300
    assert 600 < pt[1] < 700


def test_table_clip():
    t = TableSpec(1270, 2540, 28.575)
    clipped = t.clip_point(np.array([3000, -10]))
    assert clipped[0] == 2540
    assert clipped[1] == 0
