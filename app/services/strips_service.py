from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.schemas.strips import AnalyzeMeta, AnalyzeResult, AnalyzeResponse


@dataclass(frozen=True)
class StripAnalyzeInput:
    image_bytes: bytes
    filename: Optional[str] = None
    content_type: Optional[str] = None


class StripsService:
    @staticmethod
    def analyze(input_: StripAnalyzeInput, request_id: str) -> AnalyzeResponse:
        # NOTE: 지금은 dummy. 나중에 여기 안에서 Vision/OpenCV 호출로 교체.
        return AnalyzeResponse(
            ok=True,
            meta=AnalyzeMeta(request_id=request_id, model_version="dummy-v0"),
            result=AnalyzeResult(
                value_ppm=42.0,
                unit="ppm",
                lower_tick=40,
                upper_tick=50,
                relative_position=0.2,
            ),
        )
