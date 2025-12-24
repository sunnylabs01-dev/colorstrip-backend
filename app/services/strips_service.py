from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal, Any

from app.core.exceptions import RequestError, AnalysisError, UpstreamError
from app.schemas.strips import AnalyzeMeta, AnalyzeResult, AnalyzeResponse


# -----------------------
# Input / Output structs
# -----------------------

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


# -----------------------
# Pipeline Context
# -----------------------

@dataclass
class PipelineContext:
    attempted: list[str] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)

    def add_attempt(self, stage: str) -> None:
        self.attempted.append(stage)

    def add_failure(self, stage: str, code: str, reason: str, extra: dict | None = None) -> None:
        item: dict[str, Any] = {"stage": stage, "code": code, "reason": reason}
        if extra:
            item["extra"] = extra
        self.failures.append(item)

    def to_details(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "failures": self.failures,
        }


class StripsService:
    """
    Pipeline v1 (refactored):
      1) validate input (bytes-level)
      2) run vision candidate (soft-fail -> record failure -> continue)
      3) run opencv candidate (soft-fail -> record failure -> continue)
      4) select best
      5) if no valid candidate -> fallback (or raise)
    """

    @staticmethod
    def analyze(input_: StripAnalyzeInput, request_id: str) -> AnalyzeResponse:
        ctx = PipelineContext()

        # 1) validate input
        StripsService._validate_input(input_)

        # 2) candidates
        vision_cand = StripsService._run_vision_candidate(input_, ctx)
        opencv_cand = StripsService._run_opencv_candidate(input_, ctx)

        # 3) select best
        best = StripsService._select_best([vision_cand, opencv_cand])

        # 4) fallback or raise
        if best is None:
            # 정책 A: fallback 유지
            best = StripsService._fallback()

            # 정책 B(대안): 완전 실패를 422로 올리고 싶으면 아래로 교체
            # raise AnalysisError(
            #     code="ANALYSIS_NO_VALID_CANDIDATE",
            #     message="Failed to produce a valid analysis result",
            #     details=ctx.to_details(),
            # )

        # 현재 AnalyzeResponse 스키마에 debug 필드가 없어서,
        # ctx는 "에러(details)"에만 쓰고 성공 응답에는 포함하지 않음.
        return AnalyzeResponse(
            ok=True,
            meta=AnalyzeMeta(request_id=request_id, model_version="pipeline-v1"),
            result=best.result,
        )

    # -----------------------
    # Stage 0: validation
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

        # content_type은 router에서 1차 검증해도 되고,
        # 여기서 방어적으로 한 번 더 해도 됨(정책 선택)
        # if input_.content_type and input_.content_type not in ("image/jpeg", "image/png"):
        #     raise RequestError(
        #         code="REQ_UNSUPPORTED_MEDIA_TYPE",
        #         message="Unsupported image type",
        #         details={"content_type": input_.content_type},
        #     )

    # -----------------------
    # Stage 1: vision
    # -----------------------

    @staticmethod
    def _run_vision_candidate(input_: StripAnalyzeInput, ctx: PipelineContext) -> Optional[Candidate]:
        stage = "vision"
        ctx.add_attempt(stage)

        try:
            # TODO: 실제 Vision 모델/API 호출 자리
            dummy = AnalyzeResult(
                value_ppm=42.0,
                unit="ppm",
                lower_tick=40,
                upper_tick=50,
                relative_position=0.2,
            )
            cand = Candidate(source="vision", result=dummy, confidence=0.6)

            if not StripsService._is_valid(cand.result):
                ctx.add_failure(stage, "ANALYSIS_INVALID_RESULT", "Vision produced invalid result")
                return None

            return cand

        except TimeoutError as e:
            # Vision timeout은 보통 retryable. 여기서는 "soft fail"로 기록하고 OpenCV로 넘어감.
            ctx.add_failure(stage, "UPSTREAM_VISION_TIMEOUT", "Vision inference timed out", {"reason": str(e)})
            return None

        except UpstreamError as e:
            # 이미 UpstreamError로 래핑된 경우(추후 코드에서 쓸 수 있음)
            ctx.add_failure(stage, e.code, e.message, e.details or {})
            return None

        except Exception as e:
            # 파싱 실패/예상치 못한 에러도 OpenCV fallback을 위해 soft fail 처리
            ctx.add_failure(stage, "UPSTREAM_VISION_FAILED", "Vision inference failed", {"reason": str(e)})
            return None

    # -----------------------
    # Stage 2: opencv
    # -----------------------

    @staticmethod
    def _run_opencv_candidate(input_: StripAnalyzeInput, ctx: PipelineContext) -> Optional[Candidate]:
        stage = "opencv"
        ctx.add_attempt(stage)

        try:
            # TODO: 실제 OpenCV decode + boundary/ticks detection 자리
            dummy = AnalyzeResult(
                value_ppm=41.0,
                unit="ppm",
                lower_tick=40,
                upper_tick=50,
                relative_position=0.1,
            )
            cand = Candidate(source="opencv", result=dummy, confidence=0.4)

            if not StripsService._is_valid(cand.result):
                ctx.add_failure(stage, "ANALYSIS_INVALID_RESULT", "OpenCV produced invalid result")
                return None

            return cand

        except Exception as e:
            ctx.add_failure(stage, "ANALYSIS_OPENCV_FAILED", "OpenCV analysis failed", {"reason": str(e)})
            return None

    # -----------------------
    # Stage 3: selection
    # -----------------------

    @staticmethod
    def _select_best(cands: list[Optional[Candidate]]) -> Optional[Candidate]:
        valid = [c for c in cands if c is not None and StripsService._is_valid(c.result)]
        if not valid:
            return None

        # Prefer higher confidence when available; otherwise keep first
        valid.sort(key=lambda c: (c.confidence is not None, c.confidence or 0.0), reverse=True)
        return valid[0]

    @staticmethod
    def _is_valid(result: AnalyzeResult) -> bool:
        # Minimal validity checks (can evolve)
        if result.relative_position is not None and not (0.0 <= result.relative_position <= 1.0):
            return False
        return True

    # -----------------------
    # Stage 4: fallback
    # -----------------------

    @staticmethod
    def _fallback() -> Candidate:
        dummy = AnalyzeResult(
            value_ppm=None,
            unit="ppm",
            lower_tick=None,
            upper_tick=None,
            relative_position=None,
        )
        return Candidate(source="fallback", result=dummy, confidence=None)
