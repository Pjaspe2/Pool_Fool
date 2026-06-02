from __future__ import annotations

from pool_fool.shared.table import TableSpec


def table_corners_for_region(
    table: TableSpec,
    region: str,
) -> list[tuple[float, float]]:
    """
    Table-mm corners matching image click order: TL, TR, BR, BL (playing surface).

    Use when the camera cannot see the full table — click the four corners of the
    *visible* felt quad; we map them to the corresponding table coordinates.
    """
    L, W = table.length_mm, table.width_mm
    regions: dict[str, list[tuple[float, float]]] = {
        "full": [(0.0, 0.0), (L, 0.0), (L, W), (0.0, W)],
        # Camera at foot end, sees near half (0 .. L/2 along length)
        "half_near": [(0.0, 0.0), (L / 2, 0.0), (L / 2, W), (0.0, W)],
        # Far half along length
        "half_far": [(L / 2, 0.0), (L, 0.0), (L, W), (L / 2, W)],
        # Center patch (good for desk tests)
        "center": [(L / 4, W / 4), (3 * L / 4, W / 4), (3 * L / 4, 3 * W / 4), (L / 4, 3 * W / 4)],
    }
    key = region.lower().strip()
    if key not in regions:
        raise ValueError(
            f"Unknown region {region!r}. Choose: {', '.join(regions)}"
        )
    return regions[key]


def parse_dst_corners_mm(text: str) -> list[tuple[float, float]]:
    """Parse 'x1,y1;x2,y2;x3,y3;x4,y4' (TL, TR, BR, BL)."""
    parts = [p.strip() for p in text.split(";")]
    if len(parts) != 4:
        raise ValueError("Need exactly 4 corners separated by ';'")
    out: list[tuple[float, float]] = []
    for p in parts:
        xy = [x.strip() for x in p.split(",")]
        if len(xy) != 2:
            raise ValueError(f"Bad corner {p!r}, use x,y")
        out.append((float(xy[0]), float(xy[1])))
    return out
