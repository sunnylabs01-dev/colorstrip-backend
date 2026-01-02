# app/vision/change_point.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

ModelType = Literal["piecewise_constant", "piecewise_linear"]


@dataclass(frozen=True)
class ChangePointConfig:
    model: ModelType = "piecewise_constant"
    min_segment_length: int = 20
    search_range: Optional[Tuple[int, int]] = None  # (start, end) inclusive/exclusive
    expect_decrease: Optional[bool] = True          # b* often decreases when yellow ends (heuristic)
    direction_margin: float = 0.0                   # allow small violations if needed


@dataclass(frozen=True)
class ChangePointResult:
    index: int                 # row index (y) of detected change point
    score: float               # confidence/contrast score (bigger = better)
    aux: dict


def detect_change_point(
    signal: np.ndarray,
    *,
    config: ChangePointConfig = ChangePointConfig(),
) -> ChangePointResult:
    """
    Detect a single change point in a 1D signal.

    MVP approach:
      - piecewise_constant: choose k minimizing SSE of two means.
      - apply constraints: min_segment_length, optional search_range.
      - compute a simple confidence score (mean contrast / robust std).
    """
    if signal is None or signal.size == 0:
        raise ValueError("signal is empty")

    s = signal.astype(np.float32, copy=False)
    n = int(s.size)

    start = 0
    end = n
    if config.search_range is not None:
        start = max(0, int(config.search_range[0]))
        end = min(n, int(config.search_range[1]))
    min_len = max(1, int(config.min_segment_length))

    # feasible k range
    k_min = start + min_len
    k_max = end - min_len
    if k_min >= k_max:
        raise ValueError("No feasible change point with given constraints")

    # TODO(v4): implement real piecewise constant/linear fit.
    # For skeleton: pick center k.
    k = (k_min + k_max) // 2

    # dummy score
    score = 0.0
    aux = {"note": "TODO: implement SSE-based CPD", "config": config, "k_range": (k_min, k_max)}
    return ChangePointResult(index=k, score=score, aux=aux)
