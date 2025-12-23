# app/core/handlers.py

from fastapi import Request
from fastapi.exceptions import RequestValidationError

from fastapi.responses import JSONResponse

from app.core.exceptions import AppError
from app.models.error_models import ErrorResponse, ErrorDetail


async def app_error_handler(request: Request, exc: AppError):
    request_id = request.state.request_id

    error_response = ErrorResponse(
        request_id=request_id,
        ok=False,
        error=ErrorDetail(
            code=exc.code,
            message=exc.message,
            details=exc.details,
            retryable=exc.retryable,
        ),
    )

    return JSONResponse(
        status_code=exc.http_status,
        content=error_response.model_dump(),
    )


async def validation_error_handler(request: Request, exc: RequestValidationError):
    request_id = request.state.request_id

    field_errors = [
        {
            "field": ".".join(map(str, err["loc"])),
            "reason": err["msg"],
        }
        for err in exc.errors()
    ]

    error_response = ErrorResponse(
        request_id=request_id,
        ok=False,
        error=ErrorDetail(
            code="REQ_VALIDATION_FAILED",
            message="Request validation failed",
            details={"field_errors": field_errors},
            retryable=False,
        ),
    )

    return JSONResponse(
        status_code=422,
        content=error_response.model_dump(),
    )


async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = request.state.request_id

    error_response = ErrorResponse(
        request_id=request_id,
        ok=False,
        error=ErrorDetail(
            code="INTERNAL_UNHANDLED",
            message="Internal server error",
            retryable=False,
        ),
    )

    return JSONResponse(
        status_code=500,
        content=error_response.model_dump(),
    )
