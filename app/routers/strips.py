from fastapi import APIRouter, UploadFile, File, Request

from app.schemas.strips import AnalyzeResponse
from app.services.strips_service import StripsService, StripAnalyzeInput

router = APIRouter(prefix="/strips", tags=["strips"])

service = StripsService()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: Request, image: UploadFile = File(...)):
    image_bytes = await image.read()

    input_ = StripAnalyzeInput(
        image_bytes=image_bytes,
        filename=image.filename,
        content_type=image.content_type,
    )

    request_id = request.state.request_id
    return service.analyze(input_=input_, request_id=request_id)
