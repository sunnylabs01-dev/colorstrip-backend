# app/vision/color_signal.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

Channel = Literal["l", "a", "b"]
RowStat = Literal["median", "trimmed_mean", "percentile"]


@dataclass(frozen=True)
class ColorSignalConfig:
    """
    v4 Color signal config.

    channel:
      - 'b' is recommended for yellow-ness (yellow reaction region).
      - 'a' can be used as auxiliary (pink/red tendency).
    stat:
      - trimmed_mean is recommended default (robust against reflections/printing).
    """
    channel: Channel = "b"
    stat: RowStat = "trimmed_mean"
    trim_ratio: float = 0.15           # used for trimmed_mean
    percentile: float = 0.70           # used for percentile
    clip_l_range: Optional[Tuple[int, int]] = None  # optional highlight suppression using L*


@dataclass(frozen=True)
class ColorSignalResult:
    signal: np.ndarray               # shape: (H,)
    aux: dict                        # debug/extra arrays


def extract_rowwise_color_signal(
    roi_bgr: np.ndarray,
    *,
    config: ColorSignalConfig = ColorSignalConfig(),
) -> ColorSignalResult:
    """
    Convert ROI to Lab, then summarize per-row channel values into a 1D signal.

    Returns:
      ColorSignalResult(signal=..., aux={"lab": ..., "channel_map": ...})
    """
    if roi_bgr is None or roi_bgr.size == 0:
        raise ValueError("roi_bgr is empty")

    # TODO(v4): implement BGR->Lab conversion (OpenCV cvtColor),
    # and compute row-wise stat robustly.
    h, _w = roi_bgr.shape[:2]
    dummy = np.linspace(0.0, 1.0, h, dtype=np.float32)

    aux = {
        "note": "TODO: replace dummy signal with Lab-based row-wise signal",
        "config": config,
    }
    return ColorSignalResult(signal=dummy, aux=aux)
