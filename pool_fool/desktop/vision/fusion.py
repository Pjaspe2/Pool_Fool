from __future__ import annotations

import numpy as np

from pool_fool.desktop.vision.cue import CueLine


def fuse_cue_direction(
    overhead: CueLine | None,
    side: CueLine | None,
    *,
    overhead_weight: float = 0.4,
) -> CueLine | None:
    """
    Blend cue aim vectors from two cameras; prefer higher confidence.
    """
    if overhead is None and side is None:
        return None
    if overhead is None:
        return side
    if side is None:
        return overhead

    w_o = overhead_weight * overhead.confidence
    w_s = (1.0 - overhead_weight) * side.confidence
    total = w_o + w_s
    if total < 1e-6:
        return overhead if overhead.confidence >= side.confidence else side

    d = (w_o * overhead.direction_mm + w_s * side.direction_mm) / total
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        return overhead
    d = d / n
    conf = min(1.0, (overhead.confidence + side.confidence) * 0.5 + 0.1)
    tip = overhead.tip_mm if overhead.tip_mm is not None else side.tip_mm
    return CueLine(direction_mm=d, confidence=conf, tip_mm=tip)
