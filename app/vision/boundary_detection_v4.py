# app/vision/boundary_detection_v4.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.vision.roi import RoiExtractor, RoiResult
from app.vision.color_signal import (
    ColorSignalConfig,
    ColorSignalResult,
    extract_rowwise_color_signal,
)
from app.vision.signal_processing import (
    SignalProcessingConfig,
    ProcessedSignal,
    process_signal,
)
from app.vision.change_point import (
    ChangePointConfig,
    ChangePointResult,
    detect_change_point,
)


from dataclasses import dataclass, field


@dataclass(frozen=True)
class BoundaryV4Config:
    roi: Optional[dict] = None
    color_signal: ColorSignalConfig = field(
        default_factory=lambda: ColorSignalConfig(channel="b", stat="trimmed_mean", trim_ratio=0.15)
    )
    signal_processing: SignalProcessingConfig = field(
        default_factory=lambda: SignalProcessingConfig(smooth="moving_average", smooth_window=9)
    )
    change_point: ChangePointConfig = field(
        default_factory=lambda: ChangePointConfig(model="piecewise_constant", min_segment_length=20)
    )


@dataclass(frozen=True)
class BoundaryV4Result:
    boundary_y_in_roi: int
    boundary_y_in_image: int
    roi_bbox_xywh: tuple[int, int, int, int]
    cp_score: float

    roi: RoiResult
    color_signal: ColorSignalResult
    processed_signal: ProcessedSignal
    change_point: ChangePointResult
    aux: dict



class BoundaryDetectorV4:
    """
    v4 boundary detector:
      image -> ROI -> Lab row-wise signal -> process -> change point -> boundary y
    """

    def __init__(self, config: BoundaryV4Config = BoundaryV4Config()) -> None:
        self.config = config
        self.roi_extractor = RoiExtractor(**(config.roi or {}))

    def detect(self, image_bgr: np.ndarray, *, debug: bool = False) -> BoundaryV4Result:
        roi_res = self.roi_extractor.extract(image_bgr, debug=debug)

        sig_res = extract_rowwise_color_signal(roi_res.roi_bgr, config=self.config.color_signal)
        proc_res = process_signal(sig_res.signal, config=self.config.signal_processing)

        cp_res = detect_change_point(proc_res.normalized, config=self.config.change_point)

        # boundary in image coordinates
        x, y, _w, _h = roi_res.bbox_xywh
        boundary_y_in_roi = int(cp_res.index)
        boundary_y_in_image = int(y + boundary_y_in_roi)

        aux = {"debug": debug}
        return BoundaryV4Result(
            boundary_y_in_roi=boundary_y_in_roi,
            boundary_y_in_image=boundary_y_in_image,
            roi_bbox_xywh=roi_res.bbox_xywh,
            cp_score=float(cp_res.score),

            roi=roi_res,
            color_signal=sig_res,
            processed_signal=proc_res,
            change_point=cp_res,
            aux=aux,
        )
