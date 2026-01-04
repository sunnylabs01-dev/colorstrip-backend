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
    search_range: Optional[Tuple[int, int]] = None  # (start, end) in [0, n], end exclusive
    expect_decrease: Optional[bool] = True          # True: mean(top) > mean(bottom)
    direction_margin: float = 0.0                   # allow small violation (in signal units)
    return_sse_curve: bool = True                   # helpful for debugging


@dataclass(frozen=True)
class ChangePointResult:
    index: int                 # row index (y) of detected change point
    score: float               # confidence score (bigger = better)
    aux: dict                  # debug artifacts (e.g., sse_curve, means, constraints)


def _robust_std(x: np.ndarray, eps: float = 1e-6) -> float:
    x = x.astype(np.float32, copy=False)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return float(1.4826 * mad + eps)


def _prefix_sums(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # float64 for numerical stability
    x64 = x.astype(np.float64, copy=False)
    ps = np.cumsum(x64)
    ps2 = np.cumsum(x64 * x64)
    return ps, ps2


def _range_sum(ps: np.ndarray, i0: int, i1: int) -> float:
    # sum over [i0, i1)
    if i0 <= 0:
        return float(ps[i1 - 1])
    return float(ps[i1 - 1] - ps[i0 - 1])


def _range_sumsq(ps2: np.ndarray, i0: int, i1: int) -> float:
    if i0 <= 0:
        return float(ps2[i1 - 1])
    return float(ps2[i1 - 1] - ps2[i0 - 1])


def _segment_sse(ps: np.ndarray, ps2: np.ndarray, i0: int, i1: int) -> tuple[float, float]:
    """
    Return (sse, mean) for segment [i0, i1)
    SSE = sum(x^2) - n * mean^2
    """
    n = i1 - i0
    if n <= 0:
        return 0.0, 0.0
    s = _range_sum(ps, i0, i1)
    ss = _range_sumsq(ps2, i0, i1)
    m = s / float(n)
    sse = ss - float(n) * (m * m)
    return float(sse), float(m)


def detect_change_point(
    signal: np.ndarray,
    *,
    config: ChangePointConfig = ChangePointConfig(),
) -> ChangePointResult:
    """
    Detect a single change point in a 1D signal.

    MVP:
      - piecewise_constant: choose k minimizing SSE of two means.
      - constraints: min_segment_length, optional search_range, optional direction check.
      - score: |mean_left - mean_right| / robust_std(signal)
    """
    if signal is None or signal.size == 0:
        raise ValueError("signal is empty")

    if config.model != "piecewise_constant":
        raise NotImplementedError("Only piecewise_constant is implemented in v4 MVP.")

    s = signal.astype(np.float32, copy=False)
    n = int(s.size)

    min_len = max(1, int(config.min_segment_length))

    start = 0
    end = n
    if config.search_range is not None:
        start = max(0, int(config.search_range[0]))
        end = min(n, int(config.search_range[1]))
    if end - start < 2 * min_len + 1:
        raise ValueError("No feasible change point with given constraints/search_range.")

    k_min = start + min_len
    k_max = end - min_len  # k in [k_min, k_max)

    ps, ps2 = _prefix_sums(s)

    # Pre-allocate curve for debugging
    sse_curve = np.full((n,), np.nan, dtype=np.float32)

    best_k = -1
    best_val = float("inf")
    best_means = (0.0, 0.0)

    # Brute force over feasible k (fast enough for ROI heights)
    for k in range(k_min, k_max):
        sse0, m0 = _segment_sse(ps, ps2, 0, k)
        sse1, m1 = _segment_sse(ps, ps2, k, n)
        total = sse0 + sse1

        # Direction check (optional)
        if config.expect_decrease is True:
            # expect m0 > m1 (top higher than bottom)
            if (m0 - m1) < -float(config.direction_margin):
                total = float("inf")
        elif config.expect_decrease is False:
            # expect m0 < m1
            if (m1 - m0) < -float(config.direction_margin):
                total = float("inf")

        sse_curve[k] = total if np.isfinite(total) else np.nan

        if total < best_val:
            best_val = total
            best_k = k
            best_means = (m0, m1)

    if best_k < 0 or not np.isfinite(best_val):
        raise ValueError("Failed to find a valid change point (constraints too strict?)")

    m0, m1 = best_means
    denom = _robust_std(s)
    score = float(abs(m0 - m1) / denom)

    aux = {
        "model": config.model,
        "k_range": (k_min, k_max),
        "best_total_sse": float(best_val),
        "mean_top": float(m0),
        "mean_bottom": float(m1),
        "robust_std": float(denom),
    }
    if config.return_sse_curve:
        aux["sse_curve"] = sse_curve

    return ChangePointResult(index=int(best_k), score=score, aux=aux)
