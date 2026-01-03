# scripts/dev_opencv_boundary_v4.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import cv2

# --- PYTHONPATH for local run ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.vision.roi import RoiExtractor
from app.vision.color_signal import ColorSignalConfig, extract_rowwise_color_signal
from app.vision.signal_processing import SignalProcessingConfig, process_signal


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def normalize_to_uint8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-6:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - mn) / (mx - mn)
    return (y * 255.0).clip(0, 255).astype(np.uint8)


def draw_bbox(img_bgr: np.ndarray, bbox_xywh: tuple[int, int, int, int], label: str = "ROI") -> np.ndarray:
    x, y, w, h = bbox_xywh
    out = img_bgr.copy()
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if label:
        cv2.putText(
            out,
            label,
            (x, max(0, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return out


def draw_signal_plot(
    signal: np.ndarray,
    out_path: Path,
    title: str = "signal",
    width: int = 900,
    height: int = 420,
) -> None:
    """
    Minimal plotting without matplotlib:
    draw a polyline of signal values on a white canvas.
    y-axis is inverted (top=high) for readability.
    """
    sig = signal.astype(np.float32)
    H = height
    W = width
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    # If signal contains NaNs, ignore them for min/max
    finite = np.isfinite(sig)
    if not np.any(finite):
        cv2.putText(canvas, f"{title}: all NaN", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        save_image(out_path, canvas)
        return

    mn, mx = float(np.min(sig[finite])), float(np.max(sig[finite]))
    if mx - mn < 1e-6:
        mx = mn + 1.0

    n = sig.size
    pts = []
    for i in range(n):
        x = int(i * (W - 1) / max(1, n - 1))
        if not np.isfinite(sig[i]):
            # skip NaNs by holding previous value (visual continuity)
            v_norm = 0.5
        else:
            v_norm = (sig[i] - mn) / (mx - mn)
        y = int((1.0 - v_norm) * (H - 1))
        pts.append((x, y))
    pts = np.array(pts, dtype=np.int32)

    cv2.polylines(canvas, [pts], isClosed=False, color=(0, 0, 0), thickness=2)

    cv2.putText(
        canvas,
        f"{title}  (min={mn:.2f}, max={mx:.2f})",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    save_image(out_path, canvas)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/dev_opencv_boundary_v4.py <image_path> [out_dir]")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path("tmp/v4_debug")
    ensure_dir(out_dir)

    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    print("=== OpenCV boundary v4 debug ===")
    print(f"image: {image_path}")
    print(f"size: {img_bgr.shape[1]}x{img_bgr.shape[0]}")
    print(f"out:  {out_dir}")

    # Save input
    save_image(out_dir / "00_input.png", img_bgr)

    # 1) ROI extraction
    roi_extractor = RoiExtractor()
    roi_res = roi_extractor.extract(img_bgr, debug=True)
    roi = roi_res.roi_bgr

    # Save bbox on input (bbox is in ORIGINAL coords)
    annot = draw_bbox(img_bgr, roi_res.bbox_xywh, label="ROI bbox")
    save_image(out_dir / "01_input_with_bbox.png", annot)

    # Save cropped ROI
    save_image(out_dir / "02_roi.png", roi)

    # Save ROI debug artifacts (resized/edges/contour on resized)
    resized = roi_res.meta.get("resized_bgr", None)
    edges = roi_res.meta.get("edges", None)
    cnt = roi_res.meta.get("contour", None)

    if resized is not None:
        save_image(out_dir / "03_roi_resized.png", resized)

    if edges is not None:
        if edges.dtype != np.uint8:
            edges_u8 = (edges.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)
        else:
            edges_u8 = edges
        save_image(out_dir / "04_roi_edges.png", edges_u8)

    if resized is not None and cnt is not None:
        vis = resized.copy()
        cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
        save_image(out_dir / "05_roi_contour_on_resized.png", vis)

    # 2) Color signal (Lab b*)
    sig_cfg = ColorSignalConfig(
        channel="b",
        stat="trimmed_mean",
        trim_ratio=0.15,
        clip_l_range=None,  # try (20, 235) if reflections are severe
    )
    sig_res = extract_rowwise_color_signal(roi, config=sig_cfg)
    signal = sig_res.signal

    # Save b* channel map visualization
    ch_map = sig_res.aux["channel_map"]  # float32 HxW
    ch_u8 = normalize_to_uint8(ch_map)
    ch_vis = cv2.applyColorMap(ch_u8, cv2.COLORMAP_JET)
    save_image(out_dir / "06_bstar_map.png", ch_vis)

    # Save valid mask if present
    valid_mask = sig_res.aux.get("valid_mask", None)
    if valid_mask is not None:
        mask_u8 = valid_mask.astype(np.uint8) * 255
        save_image(out_dir / "07_valid_mask.png", mask_u8)

    draw_signal_plot(signal, out_dir / "08_signal_raw.png", title="row-wise b* (raw)")

    # 3) Signal processing
    proc_cfg = SignalProcessingConfig(
        smooth="moving_average",
        smooth_window=9,
        normalize="robust_z",
    )
    proc = process_signal(signal, config=proc_cfg)

    draw_signal_plot(proc.smoothed, out_dir / "09_signal_smoothed.png", title="row-wise b* (smoothed)")
    draw_signal_plot(proc.normalized, out_dir / "10_signal_normalized.png", title="row-wise b* (robust_z)")

    print("Saved:")
    for p in sorted(out_dir.glob("*.png")):
        print(f" - {p}")


if __name__ == "__main__":
    main()
