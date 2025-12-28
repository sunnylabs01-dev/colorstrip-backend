from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import cv2
import numpy as np

from app.vision.opencv_boundary_v1 import BoundaryV1Config, find_pink_boundary_y_v1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dev tool: OpenCV boundary v1 debug runner")
    p.add_argument("image_path", type=str, help="Path to an input image (jpg/png)")
    p.add_argument(
        "--out",
        type=str,
        default="tmp/opencv_boundary_v1_debug.jpg",
        help="Output debug image path",
    )

    # optional quick tuning (keep minimal)
    p.add_argument("--x1", type=float, default=None, help="ROI x_left_ratio override (0~1)")
    p.add_argument("--x2", type=float, default=None, help="ROI x_right_ratio override (0~1)")
    p.add_argument("--y1", type=float, default=None, help="ROI y_top_ratio override (0~1)")
    p.add_argument("--y2", type=float, default=None, help="ROI y_bottom_ratio override (0~1)")
    p.add_argument("--hmin", type=int, default=None, help="HSV h_min override (0~179)")
    p.add_argument("--hmax", type=int, default=None, help="HSV h_max override (0~179)")
    p.add_argument("--smin", type=int, default=None, help="HSV s_min override (0~255)")
    p.add_argument("--vmin", type=int, default=None, help="HSV v_min override (0~255)")
    p.add_argument("--row_ratio", type=float, default=None, help="row_ratio_threshold override")
    p.add_argument("--min_pixels", type=int, default=None, help="min_pink_pixels override")
    return p.parse_args()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _apply_overrides(cfg: BoundaryV1Config, args: argparse.Namespace) -> BoundaryV1Config:
    # dataclass(frozen=True)라서 replace 방식으로 새로 만든다
    d = cfg.__dict__.copy()
    if args.x1 is not None:
        d["x_left_ratio"] = args.x1
    if args.x2 is not None:
        d["x_right_ratio"] = args.x2
    if args.y1 is not None:
        d["y_top_ratio"] = args.y1
    if args.y2 is not None:
        d["y_bottom_ratio"] = args.y2
    if args.hmin is not None:
        d["h_min"] = args.hmin
    if args.hmax is not None:
        d["h_max"] = args.hmax
    if args.smin is not None:
        d["s_min"] = args.smin
    if args.vmin is not None:
        d["v_min"] = args.vmin
    if args.row_ratio is not None:
        d["row_ratio_threshold"] = args.row_ratio
    if args.min_pixels is not None:
        d["min_pink_pixels"] = args.min_pixels
    return BoundaryV1Config(**d)


def _compute_roi(h: int, w: int, cfg: BoundaryV1Config) -> tuple[int, int, int, int]:
    x1 = int(w * cfg.x_left_ratio)
    x2 = int(w * cfg.x_right_ratio)
    y1 = int(h * cfg.y_top_ratio)
    y2 = int(h * cfg.y_bottom_ratio)

    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))
    return x1, y1, x2, y2


def _pink_mask_hsv(bgr_roi: np.ndarray, cfg: BoundaryV1Config) -> np.ndarray:
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)

    lower = np.array([cfg.h_min, cfg.s_min, cfg.v_min], dtype=np.uint8)
    upper = np.array([cfg.h_max, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    k = max(1, int(cfg.morph_kernel))
    kernel = np.ones((k, k), dtype=np.uint8)

    if cfg.morph_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=cfg.morph_iter)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=cfg.morph_iter)

    return mask


def main() -> None:
    args = _parse_args()
    img_path = Path(args.image_path)
    if not img_path.exists():
        raise SystemExit(f"[ERROR] missing image: {img_path}")

    out_path = Path(args.out)
    _ensure_parent(out_path)

    cfg = _apply_overrides(BoundaryV1Config(), args)

    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        raise SystemExit("[ERROR] failed to read image via cv2.imread")

    res = find_pink_boundary_y_v1(bgr, cfg)

    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = _compute_roi(h, w, cfg)
    roi_bgr = bgr[y1:y2, x1:x2]
    mask = _pink_mask_hsv(roi_bgr, cfg)

    # --- create debug canvas ---
    debug = bgr.copy()

    # draw ROI
    cv2.rectangle(debug, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)

    # draw boundary line if found
    if res.found and res.y is not None:
        cv2.line(debug, (x1, res.y), (x2 - 1, res.y), (0, 0, 255), 2)
        cv2.putText(
            debug,
            f"boundary_y={res.y}",
            (x1, max(20, res.y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            debug,
            "boundary NOT found",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # render mask thumbnail and paste at top-left
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    thumb_w = min(240, mask_bgr.shape[1])
    thumb_h = int(mask_bgr.shape[0] * (thumb_w / max(1, mask_bgr.shape[1])))
    thumb = cv2.resize(mask_bgr, (thumb_w, max(1, thumb_h)), interpolation=cv2.INTER_NEAREST)

    # label for mask
    cv2.putText(
        thumb,
        "pink_mask",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # paste thumb onto debug image
    pad = 10
    y_end = min(h, pad + thumb.shape[0])
    x_end = min(w, pad + thumb.shape[1])
    debug[pad:y_end, pad:x_end] = thumb[: y_end - pad, : x_end - pad]

    cv2.imwrite(str(out_path), debug)

    # --- print summary ---
    print("=== OpenCV boundary v1 debug ===")
    print(f"image: {img_path}")
    print(f"size: {w}x{h}")
    print(f"roi: (x1={x1}, y1={y1}, x2={x2}, y2={y2})")
    print(f"cfg: h=[{cfg.h_min},{cfg.h_max}] s>={cfg.s_min} v>={cfg.v_min} row_ratio={cfg.row_ratio_threshold} min_pixels={cfg.min_pink_pixels}")
    print(f"found: {res.found}")
    print(f"y: {res.y}")
    print(f"debug: {res.debug}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
