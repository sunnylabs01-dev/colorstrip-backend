from __future__ import annotations

import uuid
from app.schemas.strips import AnalyzeMeta, AnalyzeResult, AnalyzeResponse


class StripsService:
    @staticmethod
    def analyze_dummy(filename: str, size_bytes: int) -> AnalyzeResponse:
        # NOTE: 오늘은 dummy. filename/size_bytes는 업로드가 제대로 들어왔는지만 확인용.
        request_id = str(uuid.uuid4())

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
