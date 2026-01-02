from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ProjectionConfig:
    # projection band (reduce edge noise): use center portion of ROI width
    use_center_band: bool = True
    band_left_ratio: float = 0.2   # ignore left 20%
    band_right_ratio: float = 0.2  # ignore right 20%

    # thresholds (for m-of-k fallback)
    min_row_ratio: float = 0.002   # e.g. 0.2% of band width
    min_pixels_per_row: int = 6

    # robust boundary rule: in a window of k rows, at least m rows meet threshold
    window_k: int = 8
    require_m: int = 3


@dataclass(frozen=True)
class PeakBoundaryConfig:
    # smoothing
    smooth_window: int = 11            # moving average window (odd recommended)

    # peak selection (search in lower part of ROI)
    peak_search_tail_ratio: float = 0.65  # search peaks in lower 65% of ROI
    peak_min_ratio: float = 0.012         # peak must exceed this ratio
    peak_prominence: float = 0.008        # peak must stand out from baseline

    # onset detection (boundary is onset before the main peak)
    onset_ratio: float = 0.010            # onset threshold on smoothed ratio
    onset_run: int = 4                    # require onset to hold for N rows

    # onset search range relative to peak (avoid picking early noise far above)
    onset_lookback: int = 200             # rows to look back from peak

    slope_window: int = 9
    slope_thr: float = 0.0015



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

    Return the FIRST satisfied row inside that window (not the window start),
    to avoid biasing boundary upward.
    """
    k = cfg.window_k
    m = cfg.require_m
    n = len(counts)
    if n < k:
        return None

    ok = (ratio >= cfg.min_row_ratio) | (counts >= cfg.min_pixels_per_row)

    for y in range(0, n - k + 1):
        window = ok[y:y+k]
        if int(window.sum()) >= m:
            first_idx = int(np.argmax(window))  # window has at least one True
            return y + first_idx
    return None


def smooth_1d(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.astype(np.float32)
    win = int(win)
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(x.astype(np.float32), kernel, mode="same")


def find_peak_onset_boundary(
    row_ratio: np.ndarray,
    cfg: PeakBoundaryConfig = PeakBoundaryConfig(),
) -> tuple[int | None, dict]:
    """
    Peak/onset strategy:
      - smooth row_ratio
      - find the strongest peak in the lower tail region
      - boundary = first sustained crossing of onset_ratio before that peak
    Returns (boundary_y, debug_dict)
    """
    n = len(row_ratio)
    if n == 0:
        return None, {"reason": "empty"}

    s = smooth_1d(row_ratio, cfg.smooth_window)

    # define tail region start (search in lower tail)
    start = int(max(0, min(n - 1, round(n * (1.0 - cfg.peak_search_tail_ratio)))))
    region = s[start:]
    if len(region) < 5:
        return None, {"reason": "region_too_small", "start": start}

    peak_rel = int(np.argmax(region))
    peak_y = start + peak_rel
    peak_val = float(s[peak_y])

    if peak_val < cfg.peak_min_ratio:
        return None, {"reason": "peak_too_small", "peak_y": peak_y, "peak_val": peak_val}

    # baseline before peak
    left0 = max(0, peak_y - 80)
    left1 = max(0, peak_y - 10)
    if left1 > left0:
        baseline = float(np.median(s[left0:left1]))
    else:
        baseline = float(np.median(s[:peak_y])) if peak_y > 0 else 0.0

    if (peak_val - baseline) < cfg.peak_prominence:
        return None, {
            "reason": "peak_not_prominent",
            "peak_y": peak_y,
            "peak_val": peak_val,
            "baseline": baseline,
        }

    # onset search
    onset_thr = cfg.onset_ratio
    run = cfg.onset_run
    y0 = max(0, peak_y - cfg.onset_lookback)
    y1 = peak_y

    # --- slope-based onset: look for "meaningful rise" before peak ---
    s2 = smooth_1d(s, cfg.slope_window)  # extra smoothing for slope stability
    ds = np.diff(s2, prepend=s2[0])  # first derivative (same length)

    # onset conditions:
    # 1) signal is above onset_thr (avoid pure noise)
    # 2) slope is above slope_thr (meaningful rise)
    # 3) (optional) relative-to-peak gate
    ok = (s >= onset_thr) & (ds >= cfg.slope_thr) & (s >= 0.30 * peak_val)

    consec = 0
    onset_y = None
    for y in range(y0, y1 + 1):
        consec = consec + 1 if ok[y] else 0
        if consec >= run:
            onset_y = y - run + 1
            break

    dbg = {
        "start": start,
        "peak_y": peak_y,
        "peak_val": peak_val,
        "baseline": baseline,
        "onset_thr": onset_thr,
        "onset_y": onset_y,
        "smooth_window": cfg.smooth_window,
        "onset_lookback": cfg.onset_lookback,
    }
    return onset_y, dbg
