import pytest

from pool_fool.shared.calibration_regions import parse_dst_corners_mm, table_corners_for_region
from pool_fool.shared.table import TableSpec


TABLE = TableSpec(width_mm=1270.0, length_mm=2540.0, ball_radius_mm=28.575)


def test_half_near():
    c = table_corners_for_region(TABLE, "half_near")
    assert c[0] == (0.0, 0.0)
    assert c[1] == (1270.0, 0.0)


def test_parse_dst():
    c = parse_dst_corners_mm("0,0;10,0;10,5;0,5")
    assert len(c) == 4
    assert c[2] == (10.0, 5.0)


def test_bad_region():
    with pytest.raises(ValueError):
        table_corners_for_region(TABLE, "nope")
