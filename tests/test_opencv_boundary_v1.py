import numpy as np
import cv2

from app.vision.opencv_boundary_v1 import find_pink_boundary_y_v1, BoundaryV1Config


def _bgr_from_hsv(h: int, s: int, v: int) -> tuple[int, int, int]:
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def test_boundary_found_on_clean_band():
    h, w = 400, 200
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # background: yellow-ish (doesn't matter much)
    img[:] = (0, 255, 255)  # BGR

    # pink band from y=160 downward
    pink_bgr = _bgr_from_hsv(160, 200, 200)
    img[160:] = pink_bgr

    cfg = BoundaryV1Config(
        x_left_ratio=0.0, x_right_ratio=1.0, y_top_ratio=0.0, y_bottom_ratio=1.0,
        row_ratio_threshold=0.01,  # 1% width threshold
        min_pink_pixels=5,
        morph_iter=0,
    )
    res = find_pink_boundary_y_v1(img, cfg)

    assert res.found is True
    assert abs(res.y - 160) <= 1


def test_boundary_not_found_when_no_pink():
    h, w = 300, 200
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (0, 255, 255)  # yellow

    cfg = BoundaryV1Config(
        x_left_ratio=0.0, x_right_ratio=1.0, y_top_ratio=0.0, y_bottom_ratio=1.0,
        row_ratio_threshold=0.01,
        min_pink_pixels=5,
        morph_iter=0,
    )
    res = find_pink_boundary_y_v1(img, cfg)
    assert res.found is False


def test_noise_pink_dots_should_not_trigger_if_threshold_high_enough():
    h, w = 400, 200
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (0, 255, 255)

    pink_bgr = _bgr_from_hsv(160, 200, 200)

    # add a few pink dots near top (noise)
    rng = np.random.default_rng(0)
    ys = rng.integers(0, 80, size=20)
    xs = rng.integers(0, w, size=20)
    img[ys, xs] = pink_bgr

    # real band starts at y=220
    img[220:] = pink_bgr

    cfg = BoundaryV1Config(
        x_left_ratio=0.0, x_right_ratio=1.0, y_top_ratio=0.0, y_bottom_ratio=1.0,
        row_ratio_threshold=0.05,  # 5% of width => noise dots won't reach
        min_pink_pixels=20,
        morph_iter=0,
    )
    res = find_pink_boundary_y_v1(img, cfg)

    assert res.found is True
    assert abs(res.y - 220) <= 1
