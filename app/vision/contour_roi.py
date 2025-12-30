# app/vision/contour_roi.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .preprocess import resize_long_edge, to_edges


@dataclass(frozen=True)
class TubeRoiResult:
    roi_bgr: np.ndarray
    bbox: Tuple[int, int, int, int]          # (x, y, w, h) in resized image coords
    contour: np.ndarray
    resized_bgr: np.ndarray
    edges: np.ndarray


def _is_tube_candidate(cnt: np.ndarray, img_shape: Tuple[int, int, int]) -> bool:
    h_img, w_img = img_shape[:2]
    x, y, w, h = cv2.boundingRect(cnt)

    area = cv2.contourArea(cnt)
    img_area = float(h_img * w_img)
    aspect = h / max(w, 1)

    # 너무 작은 것 제거 + 너무 납작한 것 제거
    if area < 0.03 * img_area:
        return False
    if aspect < 2.2:
        return False

    # 너무 위/아래에만 있거나 화면을 거의 다 덮는 것 방지(초기 안전장치)
    if h < 0.35 * h_img:
        return False

    return True


def _pick_best_tube_contour(
    contours: list[np.ndarray],
    img_shape: Tuple[int, int, int],
) -> Optional[np.ndarray]:
    candidates = []
    for cnt in contours:
        if _is_tube_candidate(cnt, img_shape):
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            aspect = h / max(w, 1)
            # 점수: 면적 + 길쭉함 가중
            score = area * (aspect ** 0.5)
            candidates.append((score, cnt))

    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def extract_tube_roi(
    img_bgr: np.ndarray,
    *,
    long_edge: int = 1080,
    canny_low: int = 50,
    canny_high: int = 150,
    pad_ratio: float = 0.02,
) -> TubeRoiResult:
    """
    Find the tube contour and return ROI cropped by its bounding box (+ padding).
    Raises ValueError if tube contour is not found.
    """
    resized = resize_long_edge(img_bgr, long_edge=long_edge)
    edges = to_edges(resized, canny_low=canny_low, canny_high=canny_high)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tube_cnt = _pick_best_tube_contour(contours, resized.shape)
    if tube_cnt is None:
        raise ValueError("Tube contour not found (no candidate passed filters).")

    x, y, w, h = cv2.boundingRect(tube_cnt)

    # padding 추가
    pad_x = int(round(w * pad_ratio))
    pad_y = int(round(h * pad_ratio))

    h_img, w_img = resized.shape[:2]
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(w_img, x + w + pad_x)
    y1 = min(h_img, y + h + pad_y)

    roi = resized[y0:y1, x0:x1].copy()
    bbox = (x0, y0, x1 - x0, y1 - y0)

    return TubeRoiResult(
        roi_bgr=roi,
        bbox=bbox,
        contour=tube_cnt,
        resized_bgr=resized,
        edges=edges,
    )
