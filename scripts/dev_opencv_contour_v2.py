#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


import cv2

from app.vision.contour_roi import extract_tube_roi
from app.vision.debug_draw import draw_bbox, draw_contour


def imwrite(path: Path, img) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenCV contour ROI v2 debug script")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--out-dir", type=str, default="tmp/contour_v2", help="Output directory")
    parser.add_argument("--long-edge", type=int, default=1080, help="Resize long edge")
    parser.add_argument("--canny-low", type=int, default=50, help="Canny low threshold")
    parser.add_argument("--canny-high", type=int, default=150, help="Canny high threshold")
    parser.add_argument("--pad-ratio", type=float, default=0.02, help="ROI padding ratio (bbox-based)")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    out_dir = Path(args.out_dir)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    print("=== OpenCV contour ROI v2 debug ===")
    print(f"image: {image_path}")
    print(f"orig_size: {img.shape[1]}x{img.shape[0]}")

    result = extract_tube_roi(
        img,
        long_edge=args.long_edge,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        pad_ratio=args.pad_ratio,
    )

    resized = result.resized_bgr
    edges = result.edges
    bbox = result.bbox
    cnt = result.contour
    roi = result.roi_bgr

    print(f"resized_size: {resized.shape[1]}x{resized.shape[0]}")
    print(f"bbox(x,y,w,h): {bbox}")
    print(f"roi_size: {roi.shape[1]}x{roi.shape[0]}")

    # 1) edges 저장 (단일 채널이라 그대로 저장)
    imwrite(out_dir / "01_edges.png", edges)

    # 2) contour + bbox overlay
    overlay = resized.copy()
    overlay = draw_contour(overlay, cnt, color=(0, 255, 0), thickness=2)
    overlay = draw_bbox(overlay, bbox, color=(255, 0, 0), thickness=2, label="tube_roi")
    imwrite(out_dir / "02_overlay_contour_bbox.png", overlay)

    # 3) roi crop 저장
    imwrite(out_dir / "03_roi.png", roi)

    # 4) (옵션) 원본도 함께 저장 (비교용)
    imwrite(out_dir / "00_input.png", img)

    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
