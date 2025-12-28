from fastapi import APIRouter

from app.core.exceptions import RequestError

router = APIRouter(
    prefix="/debug",
    tags=["debug"],
)


@router.get("/error")
async def raise_error():
    """
    Debug endpoint to verify global error handling flow.
    """
    raise RequestError(
        code="REQ_DEBUG_ERROR",
        message="Debug error for testing global exception handler",
    )
