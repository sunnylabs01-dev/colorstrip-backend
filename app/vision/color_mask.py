from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class PinkMaskConfig:
    # OpenCV HSV: H in [0,179]
    h_min1: int = 150
    h_max1: int = 179
    h_min2: int = 0
    h_max2: int = 12

    s_min: int = 35
    v_min: int = 35

    # optional: low-saturation (glare/white) suppression
    suppress_low_sat: bool = True
    s_low_max: int = 20  # if S <= this, consider it "near-white"

    # morphology
    close_ksize: int = 5
    close_iter: int = 1
    open_ksize: int = 3
    open_iter: int = 1


def make_pink_mask_hsv(roi_bgr: np.ndarray, cfg: PinkMaskConfig = PinkMaskConfig()) -> np.ndarray:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    lower1 = np.array([cfg.h_min1, cfg.s_min, cfg.v_min], dtype=np.uint8)
    upper1 = np.array([cfg.h_max1, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower1, upper1)

    lower2 = np.array([cfg.h_min2, cfg.s_min, cfg.v_min], dtype=np.uint8)
    upper2 = np.array([cfg.h_max2, 255, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    mask = cv2.bitwise_or(mask1, mask2)

    if cfg.suppress_low_sat:
        # glare/white-ish regions often have low S; remove them from mask
        s = hsv[:, :, 1]
        low_sat = (s <= cfg.s_low_max).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(low_sat))

    # morphology: close (fill holes) -> open (remove speckles)
    if cfg.close_ksize > 1:
        k = np.ones((cfg.close_ksize, cfg.close_ksize), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=cfg.close_iter)

    if cfg.open_ksize > 1:
        k = np.ones((cfg.open_ksize, cfg.open_ksize), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=cfg.open_iter)

    return mask
