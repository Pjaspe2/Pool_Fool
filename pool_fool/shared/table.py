from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TableSpec:
    width_mm: float
    length_mm: float
    ball_radius_mm: float

    @property
    def ball_diameter_mm(self) -> float:
        return 2.0 * self.ball_radius_mm

    def clip_point(self, p: np.ndarray) -> np.ndarray:
        x = float(np.clip(p[0], 0.0, self.length_mm))
        y = float(np.clip(p[1], 0.0, self.width_mm))
        return np.array([x, y], dtype=np.float64)

    def clip_segment(
        self, a: np.ndarray, b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Cohen–Sutherland style clip of segment to table rectangle."""
        x_min, y_min = 0.0, 0.0
        x_max, y_max = self.length_mm, self.width_mm

        def code(p: np.ndarray) -> int:
            c = 0
            if p[0] < x_min:
                c |= 1
            elif p[0] > x_max:
                c |= 2
            if p[1] < y_min:
                c |= 4
            elif p[1] > y_max:
                c |= 8
            return c

        p0, p1 = a.copy(), b.copy()
        c0, c1 = code(p0), code(p1)
        for _ in range(8):
            if c0 == 0 and c1 == 0:
                return p0, p1
            if c0 & c1:
                return None
            c_out = c0 if c0 else c1
            if c_out & 8:
                p0 = p0 + (p1 - p0) * (y_max - p0[1]) / (p1[1] - p0[1] + 1e-12)
            elif c_out & 4:
                p0 = p0 + (p1 - p0) * (y_min - p0[1]) / (p1[1] - p0[1] + 1e-12)
            elif c_out & 2:
                p0 = p0 + (p1 - p0) * (x_max - p0[0]) / (p1[0] - p0[0] + 1e-12)
            elif c_out & 1:
                p0 = p0 + (p1 - p0) * (x_min - p0[0]) / (p1[0] - p0[0] + 1e-12)
            if c_out == c0:
                c0 = code(p0)
            else:
                p1 = p0
                c1 = code(p1)
        return None
