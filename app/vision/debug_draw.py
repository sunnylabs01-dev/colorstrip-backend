# app/vision/debug_draw.py
from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple


def draw_contour(
    img_bgr: np.ndarray,
    contour: np.ndarray,
    *,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    out = img_bgr.copy()
    cv2.drawContours(out, [contour], -1, color, thickness)
    return out


def draw_bbox(
    img_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    *,
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    label: Optional[str] = None,
) -> np.ndarray:
    x, y, w, h = bbox
    out = img_bgr.copy()
    cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    if label:
        cv2.putText(
            out,
            label,
            (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return out
