from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def undistort_frame(
    frame: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    h, w = frame.shape[:2]
    new_cam, _roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=0.0, newImgSize=(w, h)
    )
    return cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_cam)


def load_lens_calibration(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]] | None:
    if not path.exists():
        return None
    data = np.load(path)
    return data["camera_matrix"], data["dist_coeffs"], tuple(data["image_size"])


def save_lens_calibration(
    path: Path,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=np.array(image_size, dtype=np.int32),
    )


class LensCorrector:
    """Apply saved intrinsics before table homography / vision."""

    def __init__(self, path: Path) -> None:
        loaded = load_lens_calibration(path)
        if loaded is None:
            raise FileNotFoundError(path)
        self.camera_matrix, self.dist_coeffs, self.image_size = loaded
        self._map1: np.ndarray | None = None
        self._map2: np.ndarray | None = None

    def _ensure_maps(self, w: int, h: int) -> None:
        if self._map1 is not None:
            return
        new_cam, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), alpha=0.0, newImgSize=(w, h)
        )
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            None,
            new_cam,
            (w, h),
            cv2.CV_16SC2,
        )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        self._ensure_maps(w, h)
        assert self._map1 is not None and self._map2 is not None
        return cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)
