from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def undistort_frame(
    frame: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    alpha: float = 0.0,
) -> np.ndarray:
    h, w = frame.shape[:2]
    new_cam, _roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=float(alpha), newImgSize=(w, h)
    )
    return cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_cam)


def mean_reprojection_error_px(
    obj_points: list[np.ndarray],
    img_points: list[np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
) -> float:
    total = 0.0
    n_pts = 0
    for obj, img, rvec, tvec in zip(obj_points, img_points, rvecs, tvecs, strict=True):
        proj, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
        proj = proj.reshape(-1, 2)
        img2 = img.reshape(-1, 2)
        total += float(np.linalg.norm(img2 - proj, axis=1).sum())
        n_pts += len(proj)
    return total / max(n_pts, 1)


def load_lens_calibration(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], float | None] | None:
    if not path.exists():
        return None
    data = np.load(path)
    reproj = float(data["reprojection_error_px"]) if "reprojection_error_px" in data else None
    return (
        data["camera_matrix"],
        data["dist_coeffs"],
        tuple(data["image_size"]),
        reproj,
    )


def save_lens_calibration(
    path: Path,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int],
    *,
    reprojection_error_px: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra: dict = {}
    if reprojection_error_px is not None:
        extra["reprojection_error_px"] = float(reprojection_error_px)
    np.savez(
        path,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=np.array(image_size, dtype=np.int32),
        **extra,
    )


class LensCorrector:
    """Apply saved intrinsics before table homography / vision."""

    def __init__(self, path: Path, *, alpha: float = 0.0) -> None:
        loaded = load_lens_calibration(path)
        if loaded is None:
            raise FileNotFoundError(path)
        self.camera_matrix, self.dist_coeffs, self.image_size, self.reprojection_error_px = loaded
        self.alpha = float(alpha)
        self._map1: np.ndarray | None = None
        self._map2: np.ndarray | None = None

    def _ensure_maps(self, w: int, h: int) -> None:
        if self._map1 is not None:
            return
        new_cam, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.dist_coeffs,
            (w, h),
            alpha=self.alpha,
            newImgSize=(w, h),
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
