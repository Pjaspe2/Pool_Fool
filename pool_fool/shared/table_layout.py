from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from pathlib import Path

from pool_fool.shared.table import TableSpec

if TYPE_CHECKING:
    from pool_fool.shared.play_region import PlayRegion


@dataclass(frozen=True)
class PocketSpec:
    """Aim target at pocket center in table mm (x = along long rail, y = along short rail)."""

    id: str
    center_mm: tuple[float, float]
    kind: str  # "corner" | "side"


@dataclass(frozen=True)
class TableLayout:
    """
    Playing surface rectangle + six pocket centers in table coordinates.

    Origin (0, 0) is the corner you clicked first in table calibration (TL),
    x runs along the long side (length_mm), y along the short side (width_mm).
    """

    spec: TableSpec
    pockets: tuple[PocketSpec, ...]
    border_corners_mm: np.ndarray | None = None  # (4, 2) TL→TR→BR→BL; None = config rectangle

    @property
    def length_mm(self) -> float:
        return self.spec.length_mm

    @property
    def width_mm(self) -> float:
        return self.spec.width_mm

    def border_polygon_mm(self) -> np.ndarray:
        if self.border_corners_mm is not None:
            return self.border_corners_mm
        return np.array(
            [
                [0.0, 0.0],
                [self.length_mm, 0.0],
                [self.length_mm, self.width_mm],
                [0.0, self.width_mm],
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> TableLayout:
        t = cfg["table"]
        spec = TableSpec(
            width_mm=float(t["width_mm"]),
            length_mm=float(t["length_mm"]),
            ball_radius_mm=float(t["ball_radius_mm"]),
        )
        pocket_cfg = t.get("pockets", {}) if isinstance(t.get("pockets"), dict) else {}
        inset = float(pocket_cfg.get("center_inset_mm", 57.0))
        pockets = default_six_pockets(spec.length_mm, spec.width_mm, inset_mm=inset)
        return cls(spec=spec, pockets=pockets, border_corners_mm=None)

    @classmethod
    def from_play_region(cls, region: PlayRegion, cfg: dict[str, Any]) -> TableLayout:
        """Pockets and border aligned to play-region calibration (orange quad)."""
        t = cfg["table"]
        spec = TableSpec(
            width_mm=float(t["width_mm"]),
            length_mm=float(t["length_mm"]),
            ball_radius_mm=float(t["ball_radius_mm"]),
        )
        pocket_cfg = t.get("pockets", {}) if isinstance(t.get("pockets"), dict) else {}
        frac = float(pocket_cfg.get("inset_fraction", 0.06))
        corners = region.corners_mm.copy()
        pockets = pockets_from_play_quad(corners, inset_fraction=frac)
        return cls(spec=spec, pockets=pockets, border_corners_mm=corners)

    @classmethod
    def from_calibrated_pockets(
        cls,
        pockets: tuple[PocketSpec, ...],
        cfg: dict[str, Any],
        *,
        border_corners_mm: np.ndarray | None = None,
    ) -> TableLayout:
        t = cfg["table"]
        spec = TableSpec(
            width_mm=float(t["width_mm"]),
            length_mm=float(t["length_mm"]),
            ball_radius_mm=float(t["ball_radius_mm"]),
        )
        return cls(spec=spec, pockets=pockets, border_corners_mm=border_corners_mm)

    @classmethod
    def resolve(
        cls,
        cfg: dict[str, Any],
        root: Path,
        play_region: PlayRegion | None,
    ) -> TableLayout:
        """
        Pick pocket layout: clicked pockets.npz > play-region formula > config rectangle.
        """
        from pool_fool.shared.config import resolve_path
        from pool_fool.shared.pocket_calibration import load_calibrated_pockets

        t = cfg.get("table", {})
        use_clicked = bool(t.get("use_calibrated_pockets", True))
        border = play_region.corners_mm.copy() if play_region is not None else None

        if use_clicked:
            pocket_path = resolve_path(cfg, "pockets", root)
            loaded = load_calibrated_pockets(pocket_path)
            if loaded:
                return cls.from_calibrated_pockets(loaded, cfg, border_corners_mm=border)

        if play_region is not None and bool(t.get("pockets_from_play_region", True)):
            return cls.from_play_region(play_region, cfg)

        return cls.from_config(cfg)

    def pocket_centers(self) -> list[np.ndarray]:
        return [np.array(p.center_mm, dtype=np.float64) for p in self.pockets]

    def nearest_pocket(self, point_mm: np.ndarray) -> PocketSpec:
        best = self.pockets[0]
        best_d = float("inf")
        for p in self.pockets:
            c = np.array(p.center_mm, dtype=np.float64)
            d = float(np.linalg.norm(point_mm - c))
            if d < best_d:
                best_d = d
                best = p
        return best

    def distance_to_pocket(self, point_mm: np.ndarray, pocket_id: str) -> float:
        for p in self.pockets:
            if p.id == pocket_id:
                return float(np.linalg.norm(point_mm - np.array(p.center_mm, dtype=np.float64)))
        raise KeyError(pocket_id)

    def pocket_by_id(self, pocket_id: str) -> PocketSpec:
        for p in self.pockets:
            if p.id == pocket_id:
                return p
        raise KeyError(pocket_id)

    def pocket_at_index(self, index: int) -> PocketSpec:
        pockets = self.pockets
        if not pockets:
            raise IndexError("no pockets in layout")
        return pockets[int(index) % len(pockets)]

    def pocket_index(self, pocket_id: str) -> int:
        for i, p in enumerate(self.pockets):
            if p.id == pocket_id:
                return i
        raise KeyError(pocket_id)


def default_six_pockets(
    length_mm: float,
    width_mm: float,
    *,
    inset_mm: float,
) -> tuple[PocketSpec, ...]:
    """
    Standard 6-pocket layout on a rectangular playing surface.

    Side pockets sit on the long rails (length_mm sides) at mid-table.
    Corner pockets inset from each playing-surface corner along both axes.
    """
    L, W, d = length_mm, width_mm, inset_mm
    return (
        PocketSpec("corner_tl", (d, d), "corner"),
        PocketSpec("corner_tr", (L - d, d), "corner"),
        PocketSpec("corner_br", (L - d, W - d), "corner"),
        PocketSpec("corner_bl", (d, W - d), "corner"),
        PocketSpec("side_left", (d, W / 2.0), "side"),
        PocketSpec("side_right", (L - d, W / 2.0), "side"),
    )


def pockets_from_play_quad(
    corners_mm: np.ndarray,
    *,
    inset_fraction: float,
) -> tuple[PocketSpec, ...]:
    """
    Six pockets on the play-region quad (same TL→TR→BR→BL order as calibration).

    inset_fraction: move each target from corner/edge toward the quad center
    (e.g. 0.06 = 6% of the way from corner to center — scales with your quad).
    """
    c = corners_mm.astype(np.float64)
    if c.shape != (4, 2):
        raise ValueError("corners_mm must be (4, 2)")
    f = float(np.clip(inset_fraction, 0.01, 0.35))
    center = c.mean(axis=0)

    def toward_center(pt: np.ndarray) -> np.ndarray:
        return pt + f * (center - pt)

    corner_ids = ("corner_tl", "corner_tr", "corner_br", "corner_bl")
    corners = tuple(
        PocketSpec(corner_ids[i], tuple(toward_center(c[i])), "corner") for i in range(4)
    )

    edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    lengths = [float(np.linalg.norm(c[j] - c[i])) for i, j in edges]
    ranked = sorted(range(4), key=lambda k: lengths[k], reverse=True)
    long_edges = [edges[ranked[0]], edges[ranked[1]]]

    side_pockets: list[PocketSpec] = []
    for idx, (i, j) in enumerate(long_edges):
        mid = 0.5 * (c[i] + c[j])
        pos = toward_center(mid)
        pid = "side_left" if idx == 0 else "side_right"
        side_pockets.append(PocketSpec(pid, (float(pos[0]), float(pos[1])), "side"))

    return corners + tuple(side_pockets)
