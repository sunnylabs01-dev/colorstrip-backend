# app/vision/preprocess.py
from __future__ import annotations

import cv2
import numpy as np


def resize_long_edge(img: np.ndarray, long_edge: int = 1080) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= long_edge:
        return img
    scale = long_edge / float(max(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def to_edges(
    img_bgr: np.ndarray,
    *,
    blur_ksize: int = 5,
    canny_low: int = 50,
    canny_high: int = 150,
) -> np.ndarray:
    """
    Return edges (uint8) image for contour detection.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(gray, canny_low, canny_high)

    # 연결 끊긴 edge 보완 (가벼운 closing)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    return edges
