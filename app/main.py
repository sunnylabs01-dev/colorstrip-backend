from fastapi import FastAPI
from app.routers.health import router as health_router
from app.routers.strips import router as strips_router
from app.core.middleware import RequestIdMiddleware

from fastapi.exceptions import RequestValidationError

from app.core.exceptions import AppError
from app.core.handlers import (
    app_error_handler,
    validation_error_handler,
    unhandled_exception_handler,
)

app = FastAPI(
    title="colorstrip-backend",
    description="FastAPI backend for colorimetric strip analysis.",
    version="0.1.0",
)


app.add_middleware(RequestIdMiddleware)


app.add_exception_handler(AppError, app_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

# Routers
app.include_router(health_router)
app.include_router(strips_router)

