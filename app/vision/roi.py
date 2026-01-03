# app/vision/roi.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .preprocess import resize_long_edge, to_edges


@dataclass(frozen=True)
class RoiResult:
    roi_bgr: np.ndarray
    bbox_xywh: Tuple[int, int, int, int]  # (x, y, w, h) in ORIGINAL image coords
    meta: dict


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

    # 너무 짧은 것 제거(초기 안전장치)
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


def _clip_bbox_xywh(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return x, y, w, h


class RoiExtractor:
    """
    v4 ROI extractor (tube contour based).

    Responsibility:
      - Find tube-like contour and return ROI crop for downstream color-signal analysis.
      - NO boundary logic here.
    """

    def __init__(
        self,
        long_edge: int = 1080,
        canny_low: int = 50,
        canny_high: int = 150,
        pad_ratio: float = 0.02,
        min_roi_size: Tuple[int, int] = (32, 64),  # (min_w, min_h)
    ) -> None:
        self.long_edge = long_edge
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.pad_ratio = pad_ratio
        self.min_roi_size = min_roi_size

    def extract(self, img_bgr: np.ndarray, *, debug: bool = False) -> RoiResult:
        """
        Returns ROI in ORIGINAL image coords.

        meta includes resized debug artifacts to help tuning ROI extraction.
        """
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("img_bgr is empty")

        h0, w0 = img_bgr.shape[:2]

        # 1) resize for stable contour extraction
        resized = resize_long_edge(img_bgr, long_edge=self.long_edge)
        hr, wr = resized.shape[:2]

        # scale factors: original <- resized
        sx = w0 / float(wr)
        sy = h0 / float(hr)

        # 2) edges & contours on resized
        edges = to_edges(resized, canny_low=self.canny_low, canny_high=self.canny_high)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        tube_cnt = _pick_best_tube_contour(contours, resized.shape)
        if tube_cnt is None:
            raise ValueError("Tube contour not found (no candidate passed filters).")

        xr, yr, wr_box, hr_box = cv2.boundingRect(tube_cnt)

        # 3) padding in resized coords
        pad_x = int(round(wr_box * self.pad_ratio))
        pad_y = int(round(hr_box * self.pad_ratio))

        x0r = max(0, xr - pad_x)
        y0r = max(0, yr - pad_y)
        x1r = min(wr, xr + wr_box + pad_x)
        y1r = min(hr, yr + hr_box + pad_y)

        # 4) map bbox to original coords
        x0 = int(round(x0r * sx))
        y0 = int(round(y0r * sy))
        x1 = int(round(x1r * sx))
        y1 = int(round(y1r * sy))

        x0 = max(0, min(x0, w0 - 1))
        y0 = max(0, min(y0, h0 - 1))
        x1 = max(x0 + 1, min(x1, w0))
        y1 = max(y0 + 1, min(y1, h0))

        bbox = (x0, y0, x1 - x0, y1 - y0)

        # 5) crop ROI from original image
        roi = img_bgr[y0:y1, x0:x1].copy()

        # 6) validate size (optional but good to fail fast)
        min_w, min_h = self.min_roi_size
        if roi.shape[1] < min_w or roi.shape[0] < min_h:
            raise ValueError(f"ROI too small: got {roi.shape[1]}x{roi.shape[0]}, require >= {min_w}x{min_h}")

        meta = {
            "debug": debug,
            # resized debug artifacts (useful for tuning)
            "resized_bgr": resized if debug else None,
            "edges": edges if debug else None,
            "contour": tube_cnt if debug else None,
            "bbox_resized_xywh": (x0r, y0r, x1r - x0r, y1r - y0r),
            "scale": {"sx": sx, "sy": sy, "resized_shape": (hr, wr), "orig_shape": (h0, w0)},
        }

        return RoiResult(roi_bgr=roi, bbox_xywh=bbox, meta=meta)
