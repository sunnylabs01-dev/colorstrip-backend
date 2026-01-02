from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .color_mask import PinkMaskConfig, make_pink_mask_hsv
from .projection import ProjectionConfig, row_pixel_counts, first_row_meeting_m_of_k


@dataclass(frozen=True)
class BoundaryResult:
    boundary_y: int                 # ROI 기준 y (0=top). not found -> -1
    confidence: float               # 0~1 (heuristic)
    row_counts: np.ndarray
    row_ratio: np.ndarray
    mask: np.ndarray | None         # debug 옵션


def detect_pink_boundary(
    roi_bgr: np.ndarray,
    *,
    mask_cfg: PinkMaskConfig = PinkMaskConfig(),
    proj_cfg: ProjectionConfig = ProjectionConfig(),
    return_mask: bool = True,
) -> BoundaryResult:
    mask = make_pink_mask_hsv(roi_bgr, mask_cfg)
    counts, ratio = row_pixel_counts(mask, proj_cfg)
    y = first_row_meeting_m_of_k(counts, ratio, proj_cfg)

    if y is None:
        return BoundaryResult(
            boundary_y=-1,
            confidence=0.0,
            row_counts=counts,
            row_ratio=ratio,
            mask=mask if return_mask else None
        )

    # confidence heuristic:
    # 1) boundary 이후 window에서 ratio가 얼마나 안정적으로 유지되는지
    # 2) threshold 대비 평균 강도
    n = len(ratio)
    w = 20
    window = ratio[y:min(n, y + w)]
    if len(window) == 0:
        conf = 0.1
    else:
        mean_strength = float(window.mean() / max(1e-6, proj_cfg.min_row_ratio * 3.0))
        stability = float((window >= proj_cfg.min_row_ratio).mean())  # 0~1
        conf = float(min(1.0, 0.5 * min(1.0, mean_strength) + 0.5 * stability))

    return BoundaryResult(
        boundary_y=int(y),
        confidence=conf,
        row_counts=counts,
        row_ratio=ratio,
        mask=mask if return_mask else None
    )
