import numpy as np

from pool_fool.shared.play_region import PlayRegion
from pool_fool.shared.table import TableSpec


def test_play_region_expand():
    table = TableSpec(1270.0, 2540.0, 28.575)
    corners = np.array([[100, 100], [400, 100], [400, 300], [100, 300]], dtype=np.float64)
    region = PlayRegion(corners)
    big = region.expanded(1.2, table)
    assert np.linalg.norm(big.corners_mm[0] - corners[0]) > 1.0
