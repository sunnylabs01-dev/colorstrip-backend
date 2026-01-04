# app/vision/ticks.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence

import numpy as np


TickSource = Literal["vision", "opencv"]


@dataclass(frozen=True)
class TickCandidate:
    """
    A single tick detection result.

    Coordinate system:
      - y is in ROI pixel coordinates (0 at top, increasing downward).
    """
    value: int
    y: float
    confidence: float  # 0.0 ~ 1.0
    source: TickSource = "vision"
    bbox_xyxy: Optional[tuple[float, float, float, float]] = None  # (x1,y1,x2,y2) in ROI coords


@dataclass(frozen=True)
class TickSet:
    """
    Collection of tick candidates + quality signals.
    """
    ticks: list[TickCandidate]
    quality: float  # 0.0 ~ 1.0
    failure_reason: Optional[str] = None
    debug: dict[str, Any] = None  # keep it JSON-serializable


@dataclass(frozen=True)
class TickDetectionConfig:
    """
    Config for tick detection & refinement.

    Keep it small and stable; model/prompt settings can live elsewhere later.
    """
    allowed_values: Sequence[int] = tuple(range(0, 101, 10))  # default: 0,10,...,100
    min_required_ticks: int = 2
    duplicate_y_tol_px: float = 6.0  # merge/choose duplicates if y is within this range
    outlier_iqr_multiplier: float = 1.5  # spacing outlier rule-of-thumb
    min_tick_confidence: float = 0.25  # drop very low confidence detections


class TickDetector:
    """
    High-level tick detection interface.

    This class is intentionally thin:
      - detect(): calls underlying Vision (and/or OpenCV fallback) to produce raw candidates
      - refine(): makes candidates "measurement-ready" (dedupe, monotonicity, spacing checks)
    """

    def __init__(self, config: TickDetectionConfig | None = None) -> None:
        self.config = config or TickDetectionConfig()

    def detect(self, roi_bgr: np.ndarray) -> TickSet:
        """
        Detect tick numbers and estimate their y positions.

        Args:
          roi_bgr: ROI image in BGR (OpenCV convention). Shape (H, W, 3).

        Returns:
          TickSet: may be empty with failure_reason.
        """
        # TODO:
        # 1) Call Vision/OCR to get (value, bbox, conf)
        # 2) Convert bbox -> y (e.g., center_y or baseline)
        # 3) Wrap into TickCandidates
        # 4) Return TickSet(ticks=raw, quality=raw_quality)
        return TickSet(ticks=[], quality=0.0, failure_reason="NOT_IMPLEMENTED", debug={})

    def refine(self, tickset: TickSet, roi_height: int) -> TickSet:
        """
        Refine tick candidates:
          - filter allowed values
          - drop low conf
          - dedupe same value
          - infer direction & enforce monotonicity
          - spacing outlier removal
          - compute final quality score

        Args:
          tickset: raw detection output
          roi_height: ROI image height, used for sanity checks/clamping

        Returns:
          refined TickSet
        """
        # TODO:
        # Implement the refinement pipeline in small private helpers.
        # Start conservative: do not over-prune early, but always compute debug metadata.
        return tickset


# -------------------------
# Helper functions (skeleton)
# -------------------------

def infer_value_y_direction(ticks: Sequence[TickCandidate]) -> Literal["value_increases_down", "value_increases_up", "unknown"]:
    """
    Infer whether tick values increase as y increases.

    Returns:
      - value_increases_down: larger value at larger y
      - value_increases_up: larger value at smaller y
      - unknown: insufficient data
    """
    # TODO: implement Spearman correlation (or simple pairwise comparisons).
    return "unknown"


def compute_tickset_quality(ticks: Sequence[TickCandidate], *, failure_reason: str | None = None) -> float:
    """
    Compute a 0~1 quality score for the refined tick set.
    """
    # TODO:
    # Combine: count, confidence stats, spacing stability, direction certainty.
    if failure_reason:
        return 0.0
    return 0.0
