import numpy as np

from pool_fool.desktop.calibrate.screen_warp import ScreenWarp


def test_warp_is_active():
    w = ScreenWarp()
    assert not w.is_active
    w.set_corners([(0, 0), (100, 0), (100, 80), (0, 80)])
    assert w.is_active


def test_warp_bool_check():
    w = ScreenWarp()
    w.set_corners([(0, 0), (10, 1), (11, 10), (1, 9)])
    assert w.is_active
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    out = w.warp(img)
    assert out is not None
    assert out.shape[1] == 900
