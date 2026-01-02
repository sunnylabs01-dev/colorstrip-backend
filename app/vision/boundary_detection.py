from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .color_mask import PinkMaskConfig, make_pink_mask_hsv
from .projection import (
    ProjectionConfig,
    PeakBoundaryConfig,
    row_pixel_counts,
    first_row_meeting_m_of_k,
    find_peak_onset_boundary,
    smooth_1d,
)


@dataclass(frozen=True)
class BoundaryResult:
    boundary_y: int                 # ROI 기준 y (0=top). not found -> -1
    confidence: float               # 0~1 (heuristic)
    row_counts: np.ndarray
    row_ratio: np.ndarray
    row_ratio_smooth: np.ndarray
    mask: np.ndarray | None         # debug 옵션
    method: str                     # "peak_onset" | "m_of_k" | "none"
    debug: dict                     # method-specific debug info


def _confidence_from_window(ratio: np.ndarray, y: int, thr: float) -> float:
    n = len(ratio)
    w = 24
    window = ratio[y:min(n, y + w)]
    if len(window) == 0:
        return 0.0
    mean_strength = float(window.mean() / max(1e-6, thr * 3.0))
    stability = float((window >= thr).mean())
    return float(min(1.0, 0.5 * min(1.0, mean_strength) + 0.5 * stability))


def detect_pink_boundary(
    roi_bgr: np.ndarray,
    *,
    mask_cfg: PinkMaskConfig = PinkMaskConfig(),
    proj_cfg: ProjectionConfig = ProjectionConfig(),
    peak_cfg: PeakBoundaryConfig = PeakBoundaryConfig(),
    return_mask: bool = True,
) -> BoundaryResult:
    mask = make_pink_mask_hsv(roi_bgr, mask_cfg)
    counts, ratio = row_pixel_counts(mask, proj_cfg)

    # smooth ratio for debug and peak/onset
    ratio_s = smooth_1d(ratio, peak_cfg.smooth_window)

    # --- primary: peak/onset ---
    y, peak_dbg = find_peak_onset_boundary(ratio, peak_cfg)

    if y is not None:
        conf = _confidence_from_window(ratio_s, y, peak_cfg.onset_ratio)
        return BoundaryResult(
            boundary_y=int(y),
            confidence=conf,
            row_counts=counts,
            row_ratio=ratio,
            row_ratio_smooth=ratio_s,
            mask=mask if return_mask else None,
            method="peak_onset",
            debug=peak_dbg,
        )

    # --- fallback: m-of-k ---
    y2 = first_row_meeting_m_of_k(counts, ratio, proj_cfg)
    if y2 is not None:
        conf = _confidence_from_window(ratio_s, y2, proj_cfg.min_row_ratio)
        return BoundaryResult(
            boundary_y=int(y2),
            confidence=conf,
            row_counts=counts,
            row_ratio=ratio,
            row_ratio_smooth=ratio_s,
            mask=mask if return_mask else None,
            method="m_of_k",
            debug={"reason": "peak_failed", **peak_dbg},
        )

    # --- not found ---
    return BoundaryResult(
        boundary_y=-1,
        confidence=0.0,
        row_counts=counts,
        row_ratio=ratio,
        row_ratio_smooth=ratio_s,
        mask=mask if return_mask else None,
        method="none",
        debug={"reason": "not_found", **peak_dbg},
    )
