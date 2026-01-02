# app/vision/color_signal.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import cv2

Channel = Literal["l", "a", "b"]
RowStat = Literal["median", "trimmed_mean", "percentile"]


@dataclass(frozen=True)
class ColorSignalConfig:
    """
    v4 Color signal config.

    channel:
      - 'b' recommended: yellow-ness (yellow reaction region).
      - 'a' optional auxiliary: red/pink tendency.
    stat:
      - trimmed_mean recommended (robust to reflection/printing).
    clip_l_range:
      - optional (low, high) in Lab L* space for suppressing highlights/shadows.
      - If set, pixels outside range are ignored in row stats.
    """
    channel: Channel = "b"
    stat: RowStat = "trimmed_mean"
    trim_ratio: float = 0.15           # used for trimmed_mean (0~0.49)
    percentile: float = 0.70           # used for percentile (0~1)
    clip_l_range: Optional[Tuple[int, int]] = None  # e.g. (20, 95) in OpenCV-Lab scale (0~255)


@dataclass(frozen=True)
class ColorSignalResult:
    signal: np.ndarray  # shape: (H,), float32
    aux: dict           # debug/extra arrays


def _channel_index(channel: Channel) -> int:
    if channel == "l":
        return 0
    if channel == "a":
        return 1
    if channel == "b":
        return 2
    raise ValueError(f"Unknown channel: {channel}")


def _rowwise_median(x2d: np.ndarray) -> np.ndarray:
    # x2d: (H, W)
    return np.median(x2d, axis=1).astype(np.float32)


def _rowwise_percentile(x2d: np.ndarray, q: float) -> np.ndarray:
    q = float(q)
    if not (0.0 <= q <= 1.0):
        raise ValueError("percentile must be in [0, 1]")
    return np.percentile(x2d, q * 100.0, axis=1).astype(np.float32)


def _rowwise_trimmed_mean(x2d: np.ndarray, trim_ratio: float) -> np.ndarray:
    """
    Compute row-wise trimmed mean.
    trim_ratio=0.15 means drop lowest 15% and highest 15% values per row.
    """
    r = float(trim_ratio)
    if not (0.0 <= r < 0.5):
        raise ValueError("trim_ratio must be in [0, 0.5)")
    H, W = x2d.shape
    if W == 0:
        raise ValueError("Empty width in ROI")
    if r == 0.0 or W < 4:
        return np.mean(x2d, axis=1).astype(np.float32)

    k = int(np.floor(W * r))
    # If k too large, fallback to mean
    if 2 * k >= W:
        return np.mean(x2d, axis=1).astype(np.float32)

    # sort each row and trim
    xs = np.sort(x2d, axis=1)
    trimmed = xs[:, k: W - k]
    return np.mean(trimmed, axis=1).astype(np.float32)


def _apply_l_clip_mask(
    lab: np.ndarray,
    clip_l_range: Tuple[int, int],
) -> np.ndarray:
    """
    Return a boolean mask (H, W) where True means keep pixel.
    OpenCV Lab: L in [0,255], a,b in [0,255] with 128 as ~0 offset.
    """
    lo, hi = int(clip_l_range[0]), int(clip_l_range[1])
    if lo >= hi:
        raise ValueError("clip_l_range must be (low, high) with low < high")

    L = lab[..., 0]
    return (L >= lo) & (L <= hi)


def extract_rowwise_color_signal(
    roi_bgr: np.ndarray,
    *,
    config: ColorSignalConfig = ColorSignalConfig(),
) -> ColorSignalResult:
    """
    Convert ROI to Lab, then summarize per-row channel values into a 1D signal.

    Returns:
      ColorSignalResult(
        signal=(H,), float32,
        aux={
          "lab": Lab image (H,W,3) uint8,
          "channel_map": chosen channel map (H,W) float32,
          "valid_mask": optional L-clip mask (H,W) bool,
          "config": config,
          "notes": ...
        }
      )

    Notes:
      - For change point detection, absolute scale is less important than stability.
      - We keep OpenCV Lab (uint8). If you want perceptual Lab float, we can convert later.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        raise ValueError("roi_bgr is empty")
    if roi_bgr.ndim != 3 or roi_bgr.shape[2] != 3:
        raise ValueError(f"roi_bgr must be HxWx3 BGR, got shape={roi_bgr.shape}")

    # 1) BGR -> Lab (OpenCV Lab: L,a,b in 0..255)
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)

    # 2) select channel map (float32)
    ch_idx = _channel_index(config.channel)
    ch = lab[..., ch_idx].astype(np.float32)

    valid_mask = None
    if config.clip_l_range is not None:
        valid_mask = _apply_l_clip_mask(lab, config.clip_l_range)
        # If masking, set invalid pixels to NaN so we can use nan-robust reducers
        ch_masked = ch.copy()
        ch_masked[~valid_mask] = np.nan
        ch_for_stat = ch_masked
    else:
        ch_for_stat = ch

    # 3) row-wise aggregation
    stat = config.stat
    if valid_mask is None:
        # fast path (no NaNs)
        if stat == "median":
            signal = _rowwise_median(ch_for_stat)
        elif stat == "percentile":
            signal = _rowwise_percentile(ch_for_stat, config.percentile)
        elif stat == "trimmed_mean":
            signal = _rowwise_trimmed_mean(ch_for_stat, config.trim_ratio)
        else:
            raise ValueError(f"Unknown stat: {stat}")
    else:
        # NaN-aware path
        if stat == "median":
            signal = np.nanmedian(ch_for_stat, axis=1).astype(np.float32)
        elif stat == "percentile":
            q = float(config.percentile)
            if not (0.0 <= q <= 1.0):
                raise ValueError("percentile must be in [0, 1]")
            signal = np.nanpercentile(ch_for_stat, q * 100.0, axis=1).astype(np.float32)
        elif stat == "trimmed_mean":
            # NaN-aware trimmed mean: per row, drop NaNs then trim
            H, W = ch_for_stat.shape
            out = np.empty((H,), dtype=np.float32)
            r = float(config.trim_ratio)
            if not (0.0 <= r < 0.5):
                raise ValueError("trim_ratio must be in [0, 0.5)")
            for y in range(H):
                row = ch_for_stat[y, :]
                row = row[~np.isnan(row)]
                if row.size == 0:
                    out[y] = np.nan
                    continue
                if r == 0.0 or row.size < 4:
                    out[y] = float(np.mean(row))
                    continue
                k = int(np.floor(row.size * r))
                if 2 * k >= row.size:
                    out[y] = float(np.mean(row))
                    continue
                row_sorted = np.sort(row)
                out[y] = float(np.mean(row_sorted[k: row.size - k]))
            signal = out
        else:
            raise ValueError(f"Unknown stat: {stat}")

        # If some rows are all-NaN, fill by nearest valid value (simple, stable)
        if np.any(np.isnan(signal)):
            signal = _fill_nan_1d_nearest(signal)

    aux = {
        "lab": lab,
        "channel_map": ch,
        "valid_mask": valid_mask,
        "config": config,
        "notes": {
            "opencv_lab_scale": "L,a,b in 0..255; a/b centered near 128",
            "recommendation": "Use channel='b' with stat='trimmed_mean' for v4 MVP",
        },
    }
    return ColorSignalResult(signal=signal.astype(np.float32), aux=aux)


def _fill_nan_1d_nearest(x: np.ndarray) -> np.ndarray:
    """
    Fill NaNs by nearest non-NaN value (forward/backward fill).
    """
    x = x.astype(np.float32, copy=True)
    n = x.size
    isn = np.isnan(x)
    if not np.any(isn):
        return x

    idx = np.arange(n)
    valid = ~isn
    if not np.any(valid):
        # all NaN: fallback to zeros
        return np.zeros_like(x)

    # forward fill
    last = -1
    for i in range(n):
        if valid[i]:
            last = i
        elif last != -1:
            x[i] = x[last]

    # backward fill for leading NaNs
    nextv = -1
    for i in range(n - 1, -1, -1):
        if valid[i]:
            nextv = i
        elif nextv != -1:
            x[i] = x[nextv]

    return x
