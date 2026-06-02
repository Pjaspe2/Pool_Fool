import numpy as np

from pool_fool.desktop.vision.cue import CueLine
from pool_fool.desktop.vision.fusion import fuse_cue_direction


def test_fuse_single():
    a = CueLine(direction_mm=np.array([1.0, 0.0]), confidence=0.9)
    out = fuse_cue_direction(a, None)
    assert out is not None
    np.testing.assert_allclose(out.direction_mm, [1.0, 0.0])


def test_fuse_with_numpy_tip():
    """tip_mm is ndarray; must not use 'or' on arrays."""
    tip = np.array([100.0, 200.0])
    a = CueLine(direction_mm=np.array([1.0, 0.0]), confidence=0.9, tip_mm=tip)
    b = CueLine(direction_mm=np.array([0.0, 1.0]), confidence=0.5, tip_mm=None)
    out = fuse_cue_direction(a, b)
    assert out is not None
    np.testing.assert_allclose(out.tip_mm, tip)
