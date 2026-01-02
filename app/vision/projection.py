from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ProjectionConfig:
    # projection band (reduce edge noise): use center portion of ROI width
    use_center_band: bool = True
    band_left_ratio: float = 0.2   # ignore left 20%
    band_right_ratio: float = 0.2  # ignore right 20%

    # thresholds
    min_row_ratio: float = 0.002   # e.g. 0.2% of band width
    # alternative: also allow absolute pixel threshold for very small ROIs
    min_pixels_per_row: int = 6

    # robust boundary rule: in a window of k rows, at least m rows meet threshold
    window_k: int = 8
    require_m: int = 3


def _center_band(mask: np.ndarray, cfg: ProjectionConfig) -> np.ndarray:
    if not cfg.use_center_band:
        return mask
    h, w = mask.shape[:2]
    x0 = int(round(w * cfg.band_left_ratio))
    x1 = int(round(w * (1.0 - cfg.band_right_ratio)))
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    if x1 <= x0:
        return mask  # fallback
    return mask[:, x0:x1]


def row_pixel_counts(mask: np.ndarray, cfg: ProjectionConfig = ProjectionConfig()) -> tuple[np.ndarray, np.ndarray]:
    """
    mask: uint8 0/255
    return: (per-row count of non-zero pixels, per-row ratio w.r.t effective width)
    """
    band = _center_band(mask, cfg)
    counts = (band > 0).sum(axis=1).astype(np.int32)
    width = band.shape[1]
    ratio = counts.astype(np.float32) / float(max(1, width))
    return counts, ratio


def first_row_meeting_m_of_k(
    counts: np.ndarray,
    ratio: np.ndarray,
    cfg: ProjectionConfig = ProjectionConfig()
) -> int | None:
    """
    Find earliest y such that among rows [y, y+k), at least m rows satisfy:
      (ratio >= min_row_ratio) OR (counts >= min_pixels_per_row)
    """
    k = cfg.window_k
    m = cfg.require_m
    n = len(counts)
    if n < k:
        return None

    ok = (ratio >= cfg.min_row_ratio) | (counts >= cfg.min_pixels_per_row)

    # sliding window count of ok's (simple O(n*k) is fine for ROI heights ~ few hundred)
    for y in range(0, n - k + 1):
        if int(ok[y:y+k].sum()) >= m:
            return y
    return None
