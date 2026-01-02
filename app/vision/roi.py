# app/vision/roi.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class RoiResult:
    roi_bgr: np.ndarray
    bbox_xywh: Tuple[int, int, int, int]  # (x, y, w, h)
    meta: dict


class RoiExtractor:
    """
    v4 ROI extractor.

    Responsibility:
      - Extract a vertical strip ROI for analyzing color transition.
      - MUST NOT encode any boundary definition or color-transition logic.
    """

    def __init__(
        self,
        padding: int = 8,
        min_roi_size: Tuple[int, int] = (32, 64),
    ) -> None:
        self.padding = padding
        self.min_roi_size = min_roi_size

    def extract(self, image_bgr: np.ndarray, *, debug: bool = False) -> RoiResult:
        """
        Returns:
          RoiResult containing ROI crop (BGR), bbox (x,y,w,h), and debug metadata.

        Notes:
          - Implementation can reuse existing contour-based ROI logic.
          - Keep this module "color-agnostic".
        """
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("image_bgr is empty")

        # TODO(v4): implement ROI extraction (reuse contour_roi.py logic or simplified pipeline)
        h, w = image_bgr.shape[:2]
        x, y, bw, bh = 0, 0, w, h
        roi = image_bgr.copy()

        meta = {"debug": debug, "note": "TODO: replace with real ROI extraction"}
        return RoiResult(roi_bgr=roi, bbox_xywh=(x, y, bw, bh), meta=meta)
