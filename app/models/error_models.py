from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    retryable: bool = False


class ErrorResponse(BaseModel):
    request_id: str
    ok: bool = False
    error: ErrorDetail
