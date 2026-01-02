from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time

import cv2
import numpy as np

# --- allow running as script from repo root ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.vision.contour_roi import extract_tube_roi  # noqa: E402
from app.vision.boundary_detection import detect_pink_boundary  # noqa: E402
from app.vision.color_mask import PinkMaskConfig  # noqa: E402
from app.vision.projection import ProjectionConfig  # noqa: E402


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_csv(path: Path, arr: np.ndarray, fmt: str = "%.6f") -> None:
    np.savetxt(str(path), arr, delimiter=",", fmt=fmt)


def _draw_hline(img_bgr: np.ndarray, y: int, thickness: int = 2) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]
    y = int(np.clip(y, 0, h - 1))
    cv2.line(out, (0, y), (w - 1, y), (0, 255, 0), thickness)  # green
    return out


def _draw_bbox(img_bgr: np.ndarray, bbox: tuple[int, int, int, int], thickness: int = 2) -> np.ndarray:
    x, y, w, h = bbox
    out = img_bgr.copy()
    cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), thickness)  # blue
    return out


def _stack_debug(roi_bgr: np.ndarray, mask: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    top = np.hstack([roi_bgr, mask3])
    bottom = np.hstack([overlay, np.zeros_like(overlay)])
    return np.vstack([top, bottom])


def _draw_row_ratio_signal(
    row_ratio: np.ndarray,
    *,
    row_ratio_smooth: np.ndarray | None,
    boundary_y: int | None,
    threshold: float,
    peak_y: int | None = None,
    onset_thr: float | None = None,
    height: int = 220,
    width: int = 700,
) -> np.ndarray:
    """
    Draw 1D row_ratio signal as an image.
    x-axis: row index (top -> bottom)
    y-axis: ratio magnitude
    - black: raw ratio
    - gray: smoothed ratio (if provided)
    - red: threshold line
    - green: boundary line
    - blue: peak line (optional)
    """
    n = len(row_ratio)
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    if n == 0:
        cv2.putText(canvas, "empty signal", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        return canvas

    xs = (np.arange(n) / max(1, n - 1) * (width - 1)).astype(int)

    # scaling
    max_val = float(max(row_ratio.max(), threshold * 1.5, 1e-6))
    if row_ratio_smooth is not None and len(row_ratio_smooth) == n:
        max_val = float(max(max_val, row_ratio_smooth.max()))

    def y_of(val: float) -> int:
        return int(height - 1 - (val / max_val) * (height - 1))

    # raw signal (black)
    for i in range(1, n):
        cv2.line(
            canvas,
            (xs[i - 1], y_of(float(row_ratio[i - 1]))),
            (xs[i], y_of(float(row_ratio[i]))),
            (0, 0, 0),
            1,
        )

    # smoothed signal (gray)
    if row_ratio_smooth is not None and len(row_ratio_smooth) == n:
        for i in range(1, n):
            cv2.line(
                canvas,
                (xs[i - 1], y_of(float(row_ratio_smooth[i - 1]))),
                (xs[i], y_of(float(row_ratio_smooth[i]))),
                (150, 150, 150),
                1,
            )

    # threshold line (red)
    y_thr = y_of(float(threshold))
    cv2.line(canvas, (0, y_thr), (width - 1, y_thr), (0, 0, 255), 1)
    cv2.putText(
        canvas,
        f"thr={threshold:.4f}",
        (5, max(15, y_thr - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )

    # onset threshold (orange-ish) if provided
    if onset_thr is not None:
        y_on = y_of(float(onset_thr))
        cv2.line(canvas, (0, y_on), (width - 1, y_on), (0, 165, 255), 1)
        cv2.putText(
            canvas,
            f"onset_thr={onset_thr:.4f}",
            (5, min(height - 5, y_on + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )

    # peak line (blue)
    if peak_y is not None and peak_y >= 0:
        x_p = xs[min(int(peak_y), n - 1)]
        cv2.line(canvas, (x_p, 0), (x_p, height - 1), (255, 0, 0), 1)
        cv2.putText(
            canvas,
            f"peak={peak_y}",
            (min(x_p + 5, width - 120), 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # boundary line (green)
    if boundary_y is not None and boundary_y >= 0:
        x_b = xs[min(int(boundary_y), n - 1)]
        cv2.line(canvas, (x_b, 0), (x_b, height - 1), (0, 200, 0), 1)
        cv2.putText(
            canvas,
            f"boundary={boundary_y}",
            (min(x_b + 5, width - 170), 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 200, 0),
            1,
            cv2.LINE_AA,
        )

    return canvas


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dev: v3 pink boundary detection (v2 ROI -> v3 boundary)")
    p.add_argument("image_path", type=str, help="Path to input image")

    # Output
    p.add_argument("--outdir", type=str, default="tmp/v3", help="Output directory")
    p.add_argument("--prefix", type=str, default="", help="Optional output filename prefix")
    p.add_argument("--save-panel", action="store_true", help="Save ROI+mask+overlay panel image")

    # ROI params (v2)
    p.add_argument("--long-edge", type=int, default=1080)
    p.add_argument("--canny-low", type=int, default=50)
    p.add_argument("--canny-high", type=int, default=150)
    p.add_argument("--pad-ratio", type=float, default=0.02)

    # Quick tuning knobs (override a few common cfg fields)
    p.add_argument("--s-min", type=int, default=None)
    p.add_argument("--v-min", type=int, default=None)

    p.add_argument("--min-row-ratio", type=float, default=None)
    p.add_argument("--min-pixels-per-row", type=int, default=None)
    p.add_argument("--window-k", type=int, default=None)
    p.add_argument("--require-m", type=int, default=None)
    p.add_argument("--band-left", type=float, default=None)
    p.add_argument("--band-right", type=float, default=None)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    img_path = Path(args.image_path)
    if not img_path.exists():
        print(f"[ERROR] not found: {img_path}")
        return 2

    outdir = Path(args.outdir)
    _ensure_dir(outdir)

    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"[ERROR] failed to read image: {img_path}")
        return 2

    # --- v2 ROI extraction ---
    try:
        tube = extract_tube_roi(
            img_bgr,
            long_edge=args.long_edge,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            pad_ratio=args.pad_ratio,
        )
    except ValueError as e:
        print(f"[ERROR] ROI extraction failed: {e}")
        return 1

    roi_bgr = tube.roi_bgr
    bbox = tube.bbox  # (x, y, w, h) in resized image coords

    # --- configs ---
    mask_cfg = PinkMaskConfig()
    proj_cfg = ProjectionConfig()

    if args.s_min is not None:
        mask_cfg = PinkMaskConfig(**{**asdict(mask_cfg), "s_min": int(args.s_min)})
    if args.v_min is not None:
        mask_cfg = PinkMaskConfig(**{**asdict(mask_cfg), "v_min": int(args.v_min)})

    proj_overrides = asdict(proj_cfg)
    if args.min_row_ratio is not None:
        proj_overrides["min_row_ratio"] = float(args.min_row_ratio)
    if args.min_pixels_per_row is not None:
        proj_overrides["min_pixels_per_row"] = int(args.min_pixels_per_row)
    if args.window_k is not None:
        proj_overrides["window_k"] = int(args.window_k)
    if args.require_m is not None:
        proj_overrides["require_m"] = int(args.require_m)
    if args.band_left is not None:
        proj_overrides["band_left_ratio"] = float(args.band_left)
    if args.band_right is not None:
        proj_overrides["band_right_ratio"] = float(args.band_right)
    proj_cfg = ProjectionConfig(**proj_overrides)

    # --- run v3 ---
    result = detect_pink_boundary(
        roi_bgr,
        mask_cfg=mask_cfg,
        proj_cfg=proj_cfg,
        return_mask=True,
    )

    # --- outputs ---
    ts = time.strftime("%Y%m%d_%H%M%S")
    stem = (args.prefix + img_path.stem).strip()
    base = outdir / f"{stem}_{ts}"

    # save resized + bbox overlay
    resized_bbox = _draw_bbox(tube.resized_bgr, bbox, thickness=2)
    cv2.imwrite(str(base.with_suffix(".resized_bbox.jpg")), resized_bbox)

    # save edges (v2 debug)
    cv2.imwrite(str(base.with_suffix(".edges.png")), tube.edges)

    # save ROI
    cv2.imwrite(str(base.with_suffix(".roi.jpg")), roi_bgr)

    # save mask
    if result.mask is not None:
        cv2.imwrite(str(base.with_suffix(".mask.png")), result.mask)

    # overlay boundary on ROI
    if result.boundary_y >= 0:
        overlay = _draw_hline(roi_bgr, result.boundary_y, thickness=2)
    else:
        overlay = roi_bgr.copy()
        cv2.putText(
            overlay,
            "BOUNDARY NOT FOUND",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(base.with_suffix(".overlay.jpg")), overlay)

    # save signals
    _save_csv(base.with_suffix(".row_counts.csv"), result.row_counts.astype(np.float32), fmt="%.1f")
    _save_csv(base.with_suffix(".row_ratio.csv"), result.row_ratio.astype(np.float32), fmt="%.6f")

    # --- save row_ratio signal image ---
    peak_y = None
    onset_thr = None
    if isinstance(getattr(result, "debug", None), dict):
        peak_y = result.debug.get("peak_y", None)
        onset_thr = result.debug.get("onset_thr", None)

    signal_img = _draw_row_ratio_signal(
        result.row_ratio,
        row_ratio_smooth=getattr(result, "row_ratio_smooth", None),
        boundary_y=result.boundary_y if result.boundary_y >= 0 else None,
        threshold=proj_cfg.min_row_ratio,  # fallback 기준선도 같이 보여주기
        peak_y=peak_y if peak_y is not None else None,
        onset_thr=onset_thr if onset_thr is not None else None,
    )
    cv2.imwrite(str(base.with_suffix(".signal.jpg")), signal_img)

    # meta
    meta_txt = base.with_suffix(".meta.txt")
    meta_txt.write_text(
        "\n".join(
            [
                f"image_path={img_path}",
                f"roi_bbox_resized={bbox}",  # bbox is in resized image coords
                f"boundary_y_roi={result.boundary_y}",
                f"confidence={result.confidence:.4f}",
                f"mask_cfg={asdict(mask_cfg)}",
                f"proj_cfg={asdict(proj_cfg)}",
                f"roi_shape={roi_bgr.shape}",
                f"resized_shape={tube.resized_bgr.shape}",
            ]
        ),
        encoding="utf-8",
    )

    # optional: combined panel
    if args.save_panel and result.mask is not None:
        panel = _stack_debug(roi_bgr, result.mask, overlay)
        cv2.imwrite(str(base.with_suffix(".panel.jpg")), panel)

    print("=== v3 boundary detection dev ===")
    print(f"image:        {img_path}")
    print(f"out base:     {base}.*")
    print(f"bbox(resized): {bbox}")
    print(f"boundary_y:   {result.boundary_y}")
    print(f"confidence:   {result.confidence:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
