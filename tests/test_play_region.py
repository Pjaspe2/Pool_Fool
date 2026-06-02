import numpy as np

from pool_fool.shared.play_region import PlayRegion


def test_contains():
    corners = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float64)
    r = PlayRegion(corners)
    assert r.contains(np.array([50, 25]))
    assert not r.contains(np.array([150, 25]))
