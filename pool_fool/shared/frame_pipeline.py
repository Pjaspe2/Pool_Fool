from __future__ import annotations

from pathlib import Path

import numpy as np

from pool_fool.shared.config import load_config, resolve_path
from pool_fool.shared.lens import LensCorrector


def build_lens_corrector(cfg: dict, root: Path) -> LensCorrector | None:
    if not cfg.get("cameras", {}).get("undistort", False):
        return None
    path = resolve_path(cfg, "lens_calibration", root)
    if not path.exists():
        return None
    return LensCorrector(path)


def preprocess_frame(frame: np.ndarray, corrector: LensCorrector | None) -> np.ndarray:
    if corrector is None:
        return frame
    return corrector.apply(frame)
