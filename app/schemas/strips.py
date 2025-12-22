from typing import Optional, Literal
from pydantic import BaseModel, Field


class AnalyzeMeta(BaseModel):
    request_id: str
    model_version: str = "dummy-v0"


class AnalyzeResult(BaseModel):
    value_ppm: Optional[float] = None
    unit: Literal["ppm"] = "ppm"
    lower_tick: Optional[int] = None
    upper_tick: Optional[int] = None
    relative_position: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="0.0 at lower_tick, 1.0 at upper_tick"
    )


class AnalyzeResponse(BaseModel):
    ok: bool = True
    meta: AnalyzeMeta
    result: AnalyzeResult
