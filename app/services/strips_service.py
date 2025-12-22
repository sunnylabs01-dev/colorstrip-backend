from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal
import uuid

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
      1) (optional) validate input (bytes-level)
      2) run vision candidate
      3) run opencv candidate
      4) select/merge best candidate
      5) if no valid candidate -> fallback
    """

    @staticmethod
    def analyze(input_: StripAnalyzeInput, request_id: str) -> AnalyzeResponse:
        # 1) validate (bytes-level only; decoding validation later if needed)
        StripsService._validate_input(input_)

        # 2) produce candidates (for now: stub/dummy)
        vision_cand = StripsService._analyze_with_vision(input_)
        opencv_cand = StripsService._analyze_with_opencv(input_)

        # 3) pick best
        best = StripsService._select_best([vision_cand, opencv_cand])

        # 4) fallback if needed
        if best is None:
            best = StripsService._fallback(input_)

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
        # Keep router validation for content-type; service can do minimal sanity checks.
        if not input_.image_bytes:
            # router already checks, but service protects itself too
            raise ValueError("Empty image bytes")

        # Optional: tiny size guard (helps catch accidental empty/garbage)
        if len(input_.image_bytes) < 10:
            raise ValueError("Image bytes too small")

    @staticmethod
    def _analyze_with_vision(input_: StripAnalyzeInput) -> Candidate:
        # STUB: later call Vision model/API and parse outputs
        dummy = AnalyzeResult(
            value_ppm=42.0,
            unit="ppm",
            lower_tick=40,
            upper_tick=50,
            relative_position=0.2,
        )
        return Candidate(source="vision", result=dummy, confidence=0.6)

    @staticmethod
    def _analyze_with_opencv(input_: StripAnalyzeInput) -> Candidate:
        # STUB: later decode image and run boundary detection
        dummy = AnalyzeResult(
            value_ppm=41.0,
            unit="ppm",
            lower_tick=40,
            upper_tick=50,
            relative_position=0.1,
        )
        return Candidate(source="opencv", result=dummy, confidence=0.4)

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
        # Example: if ticks exist, relative_position should be between 0~1
        if result.relative_position is not None and not (0.0 <= result.relative_position <= 1.0):
            return False
        return True

    @staticmethod
    def _fallback(input_: StripAnalyzeInput) -> Candidate:
        # STUB: safe empty/unknown response (or raise)
        dummy = AnalyzeResult(
            value_ppm=None,
            unit="ppm",
            lower_tick=None,
            upper_tick=None,
            relative_position=None,
        )
        return Candidate(source="fallback", result=dummy, confidence=None)
