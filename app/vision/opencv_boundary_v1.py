from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np


@dataclass(frozen=True)
class BoundaryV1Config:
    # ROI crop ratios (v1: simple fixed crop)
    x_left_ratio: float = 0.30
    x_right_ratio: float = 0.70
    y_top_ratio: float = 0.05
    y_bottom_ratio: float = 0.95

    # HSV threshold for "pink/magenta"
    # OpenCV HSV: H 0-179, S/V 0-255
    h_min: int = 140
    h_max: int = 179
    s_min: int = 40
    v_min: int = 40

    # Morphology for noise removal
    morph_kernel: int = 3
    morph_iter: int = 1

    # Row scan threshold
    row_ratio_threshold: float = 0.003  # 0.3% of ROI width
    min_pink_pixels: int = 10           # absolute minimum per row


@dataclass(frozen=True)
class BoundaryV1Result:
    found: bool
    y: Optional[int] = None
    y_in_roi: Optional[int] = None
    roi: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    debug: Optional[Dict[str, Any]] = None


def _compute_roi(h: int, w: int, cfg: BoundaryV1Config) -> Tuple[int, int, int, int]:
    x1 = int(w * cfg.x_left_ratio)
    x2 = int(w * cfg.x_right_ratio)
    y1 = int(h * cfg.y_top_ratio)
    y2 = int(h * cfg.y_bottom_ratio)

    # clamp + ensure non-empty
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))
    return x1, y1, x2, y2


def _pink_mask_hsv(bgr_roi: np.ndarray, cfg: BoundaryV1Config) -> np.ndarray:
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)

    lower = np.array([cfg.h_min, cfg.s_min, cfg.v_min], dtype=np.uint8)
    upper = np.array([cfg.h_max, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)  # 0 or 255

    k = max(1, int(cfg.morph_kernel))
    kernel = np.ones((k, k), dtype=np.uint8)

    if cfg.morph_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=cfg.morph_iter)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=cfg.morph_iter)

    return mask


def find_pink_boundary_y_v1(bgr: np.ndarray, cfg: BoundaryV1Config = BoundaryV1Config()) -> BoundaryV1Result:
    if bgr is None or bgr.size == 0:
        return BoundaryV1Result(found=False, debug={"reason": "empty_image"})

    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = _compute_roi(h, w, cfg)
    roi_bgr = bgr[y1:y2, x1:x2]

    mask = _pink_mask_hsv(roi_bgr, cfg)
    roi_h, roi_w = mask.shape[:2]

    row_threshold = max(int(roi_w * cfg.row_ratio_threshold), cfg.min_pink_pixels)

    row_counts = np.count_nonzero(mask, axis=1)  # per-row pink pixel counts
    hit_rows = np.where(row_counts >= row_threshold)[0]

    if hit_rows.size == 0:
        return BoundaryV1Result(
            found=False,
            roi=(x1, y1, x2, y2),
            debug={
                "row_threshold": int(row_threshold),
                "max_row_count": int(row_counts.max()) if row_counts.size else 0,
                "roi_hw": (int(roi_h), int(roi_w)),
            },
        )

    y_roi = int(hit_rows[0])
    y_abs = int(y1 + y_roi)

    return BoundaryV1Result(
        found=True,
        y=y_abs,
        y_in_roi=y_roi,
        roi=(x1, y1, x2, y2),
        debug={
            "row_threshold": int(row_threshold),
            "max_row_count": int(row_counts.max()),
            "roi_hw": (int(roi_h), int(roi_w)),
        },
    )
