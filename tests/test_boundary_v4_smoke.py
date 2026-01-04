# tests/test_boundary_v4_smoke.py
from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from app.vision.boundary_detection_v4 import (
    BoundaryDetectorV4,
    BoundaryV4Config,
)


TEST_IMAGE = Path("tests/assets/visiontest.jpeg")


@pytest.mark.smoke
def test_boundary_v4_smoke():
    """
    Smoke test for v4 boundary detection.

    This test checks:
      - the pipeline runs end-to-end
      - boundary y is within image bounds
      - ROI bbox is valid
      - CP score is positive
    """
    assert TEST_IMAGE.exists(), f"Missing test image: {TEST_IMAGE}"

    img = cv2.imread(str(TEST_IMAGE))
    assert img is not None, "Failed to load test image"

    H, W = img.shape[:2]

    detector = BoundaryDetectorV4(config=BoundaryV4Config())
    result = detector.detect(img, debug=False)

    # --- basic existence ---
    assert result is not None

    # --- boundary sanity ---
    assert 0 <= result.boundary_y_in_image < H
    assert 0 <= result.boundary_y_in_roi < result.roi.roi_bgr.shape[0]

    # --- ROI bbox sanity ---
    x, y, w, h = result.roi_bbox_xywh
    assert w > 0 and h > 0
    assert 0 <= x < W
    assert 0 <= y < H
    assert x + w <= W
    assert y + h <= H

    # --- CP confidence sanity ---
    assert result.cp_score > 0.0

    # --- minimal structural checks ---
    assert result.roi.roi_bgr.size > 0
    assert result.processed_signal.normalized.size > 0
