from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

from app.core.exceptions import RequestError, AnalysisError, UpstreamError
from app.schemas.strips import AnalyzeMeta, AnalyzeResult, AnalyzeResponse


@dataclass(frozen=True)
class StripAnalyzeInput:
    image_bytes: bytes
    filename: Optional[str] = None
    content_type: Optional[str] = None


@dataclass(frozen=True)
class Candidate:
    source: Literal["vision", "opencv", "fallback"]
    result: AnalyzeResult
    confidence: Optional[float] = None  # 0~1 (optional)


class StripsService:
    """
    Pipeline v1:
      1) validate input (bytes-level)
      2) run vision candidate
      3) run opencv candidate
      4) select/merge best candidate
      5) if no valid candidate -> fallback (or raise)
    """

    @staticmethod
    def analyze(input_: StripAnalyzeInput, request_id: str) -> AnalyzeResponse:
        StripsService._validate_input(input_)

        vision_cand = StripsService._analyze_with_vision(input_)
        opencv_cand = StripsService._analyze_with_opencv(input_)

        best = StripsService._select_best([vision_cand, opencv_cand])

        if best is None:
            # 선택지 1: 지금처럼 "unknown 결과"를 반환
            best = StripsService._fallback(input_)

            # 선택지 2(더 엄격): 완전 실패를 에러로 처리하고 싶다면 아래로 교체
            # raise AnalysisError(
            #     code="ANALYSIS_NO_VALID_CANDIDATE",
            #     message="Failed to produce a valid analysis result",
            #     details={"attempted": ["vision", "opencv"]},
            # )

        return AnalyzeResponse(
            ok=True,
            meta=AnalyzeMeta(request_id=request_id, model_version="pipeline-v1"),
            result=best.result,
        )

    # -----------------------
    # Pipeline building blocks
    # -----------------------

    @staticmethod
    def _validate_input(input_: StripAnalyzeInput) -> None:
        if not input_.image_bytes:
            raise RequestError(
                code="REQ_EMPTY_IMAGE_BYTES",
                message="Uploaded image file is empty",
            )

        if len(input_.image_bytes) < 10:
            raise RequestError(
                code="REQ_IMAGE_BYTES_TOO_SMALL",
                message="Uploaded image file is too small",
                details={"size_bytes": len(input_.image_bytes)},
            )

    @staticmethod
    def _analyze_with_vision(input_: StripAnalyzeInput) -> Optional[Candidate]:
        try:
            # STUB: later call Vision model/API and parse outputs
            dummy = AnalyzeResult(
                value_ppm=42.0,
                unit="ppm",
                lower_tick=40,
                upper_tick=50,
                relative_position=0.2,
            )
            return Candidate(source="vision", result=dummy, confidence=0.6)
        except TimeoutError as e:
            # 예: 나중에 Vision 호출 붙였을 때
            # retryable=True로 갈 만한 케이스
            raise UpstreamError(
                code="UPSTREAM_VISION_TIMEOUT",
                message="Vision inference timed out",
                details={"reason": str(e)},
                retryable=True,
            )
        except Exception:
            # Vision이 실패해도 OpenCV로 계속 가고 싶다면 "raise" 대신 None 반환도 가능
            return None

    @staticmethod
    def _analyze_with_opencv(input_: StripAnalyzeInput) -> Optional[Candidate]:
        try:
            # STUB: later decode image and run boundary detection
            dummy = AnalyzeResult(
                value_ppm=41.0,
                unit="ppm",
                lower_tick=40,
                upper_tick=50,
                relative_position=0.1,
            )
            return Candidate(source="opencv", result=dummy, confidence=0.4)
        except Exception:
            return None

    @staticmethod
    def _select_best(cands: list[Optional[Candidate]]) -> Optional[Candidate]:
        valid = [c for c in cands if c is not None and StripsService._is_valid(c.result)]
        if not valid:
            return None

        valid.sort(key=lambda c: (c.confidence is not None, c.confidence or 0.0), reverse=True)
        return valid[0]

    @staticmethod
    def _is_valid(result: AnalyzeResult) -> bool:
        if result.relative_position is not None and not (0.0 <= result.relative_position <= 1.0):
            return False
        return True

    @staticmethod
    def _fallback(input_: StripAnalyzeInput) -> Candidate:
        dummy = AnalyzeResult(
            value_ppm=None,
            unit="ppm",
            lower_tick=None,
            upper_tick=None,
            relative_position=None,
        )
        return Candidate(source="fallback", result=dummy, confidence=None)
