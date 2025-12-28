from fastapi import APIRouter, UploadFile, File, Request

from app.core.exceptions import RequestError
from app.core.error_codes import REQ_UNSUPPORTED_MEDIA_TYPE
from app.schemas.strips import AnalyzeResponse
from app.services.strips_service import StripsService, StripAnalyzeInput

router = APIRouter(prefix="/strips", tags=["strips"])

service = StripsService()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: Request, image: UploadFile = File(...)):
    if image.content_type not in ("image/jpeg", "image/png"):
        raise RequestError(
            code=REQ_UNSUPPORTED_MEDIA_TYPE,
            message="Only image uploads are supported.",
            details={"content_type": image.content_type},
        )
    image_bytes = await image.read()
    input_ = StripAnalyzeInput(
        image_bytes=image_bytes,
        filename=image.filename,
        content_type=image.content_type,
    )

    request_id = request.state.request_id
    return service.analyze(input_=input_, request_id=request_id)
