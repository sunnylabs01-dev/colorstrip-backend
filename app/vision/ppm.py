# app/vision/ppm.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .ticks import TickCandidate, TickSet


@dataclass(frozen=True)
class PpmResult:
    """
    Final ppm mapping result.

    Coordinate system:
      - boundary_y is in ROI pixel coordinates (0 at top, increasing downward).
    """
    ppm: Optional[float]
    lower_tick: Optional[int]
    upper_tick: Optional[int]
    lower_y: Optional[float]
    upper_y: Optional[float]
    relative_position: Optional[float]  # 0~1 in the interval
    confidence: float  # 0~1
    failure_reason: Optional[str] = None
    debug: dict[str, Any] = None


@dataclass(frozen=True)
class PpmMappingConfig:
    """
    Deterministic mapping configuration.
    """
    clamp_out_of_range: bool = False
    min_interval_px: float = 3.0  # avoid division by ~0 when ticks collapse
    default_unit: str = "ppm"


def compute_ppm_from_ticks(
    *,
    boundary_y: float,
    tickset: TickSet,
    config: PpmMappingConfig | None = None,
    boundary_confidence: float = 1.0,
) -> PpmResult:
    """
    Compute ppm using linear interpolation between adjacent ticks around boundary_y.

    Requirements:
      - tickset should already be refined (deduped, filtered, monotonic-consistent).

    Strategy:
      1) Find the two ticks (lower/upper) such that boundary_y lies between their y.
      2) Interpolate ppm linearly.
      3) Compute confidence = f(boundary_confidence, tickset.quality, local interval quality)

    Returns:
      PpmResult (ppm may be None if mapping is not possible)
    """
    cfg = config or PpmMappingConfig()

    # TODO:
    # 1) Validate tickset: must have at least 2 ticks
    # 2) Sort ticks by y (NOT value) for interval search
    # 3) Find bracket interval containing boundary_y
    # 4) Interpolate
    # 5) Optionally clamp if out of range and cfg.clamp_out_of_range
    # 6) Compose confidence and debug metadata

    return PpmResult(
        ppm=None,
        lower_tick=None,
        upper_tick=None,
        lower_y=None,
        upper_y=None,
        relative_position=None,
        confidence=0.0,
        failure_reason="NOT_IMPLEMENTED",
        debug={},
    )


# -------------------------
# Local helpers (skeleton)
# -------------------------

def _local_interval_quality(lower: TickCandidate, upper: TickCandidate, *, min_interval_px: float) -> float:
    """
    Quality of the chosen interpolation interval.
    """
    # TODO:
    # - penalize tiny pixel interval
    # - combine (lower.conf, upper.conf)
    return 0.0
