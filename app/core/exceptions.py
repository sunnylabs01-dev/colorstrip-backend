from typing import Optional, Dict, Any


class AppError(Exception):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        http_status: int,
        details: Optional[Dict[str, Any]] = None,
        retryable: bool = False,
    ):
        self.code = code
        self.message = message
        self.http_status = http_status
        self.details = details
        self.retryable = retryable
        super().__init__(message)


class RequestError(AppError):
    def __init__(self, *, code: str, message: str, details=None):
        super().__init__(
            code=code,
            message=message,
            http_status=400,
            details=details,
            retryable=False,
        )


class AnalysisError(AppError):
    def __init__(self, *, code: str, message: str, details=None):
        super().__init__(
            code=code,
            message=message,
            http_status=422,
            details=details,
            retryable=False,
        )


class UpstreamError(AppError):
    def __init__(self, *, code: str, message: str, details=None, retryable=True):
        super().__init__(
            code=code,
            message=message,
            http_status=502,
            details=details,
            retryable=retryable,
        )


class InternalError(AppError):
    def __init__(self, *, message="Internal server error", details=None):
        super().__init__(
            code="INTERNAL_UNHANDLED",
            message=message,
            http_status=500,
            details=details,
            retryable=False,
        )
