# app/vision/signal_processing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

SmoothMethod = Literal["moving_average", "savgol", "none"]
NormalizeMethod = Literal["robust_z", "minmax", "none"]


@dataclass(frozen=True)
class SignalProcessingConfig:
    smooth: SmoothMethod = "moving_average"
    smooth_window: int = 9

    normalize: NormalizeMethod = "robust_z"
    eps: float = 1e-6


@dataclass(frozen=True)
class ProcessedSignal:
    raw: np.ndarray
    smoothed: np.ndarray
    normalized: np.ndarray
    aux: dict


def process_signal(
    signal: np.ndarray,
    *,
    config: SignalProcessingConfig = SignalProcessingConfig(),
) -> ProcessedSignal:
    """
    Prepare a boundary-friendly signal: smoothing + robust normalization.

    Notes:
      - Keep it deterministic and lightweight.
      - Avoid heavy dependencies in MVP (NumPy only is fine).
    """
    if signal is None or signal.size == 0:
        raise ValueError("signal is empty")

    raw = signal.astype(np.float32, copy=False)

    # --- smoothing ---
    if config.smooth == "none":
        smoothed = raw
    elif config.smooth == "moving_average":
        k = int(config.smooth_window)
        k = max(1, k | 1)  # force odd window
        pad = k // 2
        padded = np.pad(raw, (pad, pad), mode="edge")
        kernel = np.ones(k, dtype=np.float32) / float(k)
        smoothed = np.convolve(padded, kernel, mode="valid")
    elif config.smooth == "savgol":
        # TODO(v4): optionally implement Savitzky–Golay (or keep moving_average for MVP)
        smoothed = raw
    else:
        raise ValueError(f"Unknown smoothing method: {config.smooth}")

    # --- normalization ---
    if config.normalize == "none":
        normalized = smoothed
    elif config.normalize == "minmax":
        mn, mx = float(np.min(smoothed)), float(np.max(smoothed))
        normalized = (smoothed - mn) / (mx - mn + config.eps)
    elif config.normalize == "robust_z":
        med = float(np.median(smoothed))
        mad = float(np.median(np.abs(smoothed - med))) + config.eps
        normalized = (smoothed - med) / (1.4826 * mad)
    else:
        raise ValueError(f"Unknown normalize method: {config.normalize}")

    aux = {"config": config}
    return ProcessedSignal(raw=raw, smoothed=smoothed, normalized=normalized, aux=aux)
    