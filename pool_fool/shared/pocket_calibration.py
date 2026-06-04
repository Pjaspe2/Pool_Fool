from __future__ import annotations

from pathlib import Path

import numpy as np

from pool_fool.shared.table_layout import PocketSpec


POCKET_CALIBRATION_ORDER: tuple[tuple[str, str, str], ...] = (
    ("corner_tl", "corner", "Corner near table TL (your 1st table corner)"),
    ("corner_tr", "corner", "Corner near table TR"),
    ("corner_br", "corner", "Corner near table BR"),
    ("corner_bl", "corner", "Corner near table BL"),
    ("side_left", "side", "Side pocket on long rail (left in image)"),
    ("side_right", "side", "Side pocket on long rail (right in image)"),
)


def save_calibrated_pockets(path: Path, pockets: tuple[PocketSpec, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.array([p.id for p in pockets], dtype=object)
    kinds = np.array([p.kind for p in pockets], dtype=object)
    centers = np.array([p.center_mm for p in pockets], dtype=np.float64)
    np.savez(path, pocket_ids=ids, pocket_kinds=kinds, centers_mm=centers)


def load_calibrated_pockets(path: Path) -> tuple[PocketSpec, ...] | None:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    ids = data["pocket_ids"]
    kinds = data["pocket_kinds"]
    centers = data["centers_mm"]
    return tuple(
        PocketSpec(str(ids[i]), (float(centers[i, 0]), float(centers[i, 1])), str(kinds[i]))
        for i in range(len(ids))
    )
