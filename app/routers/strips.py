import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.schemas.strips import AnalyzeResponse
from app.services.strips_service import StripsService, StripAnalyzeInput

router = APIRouter(prefix="/v1/strips", tags=["strips"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_strip(image: UploadFile = File(...)) -> AnalyzeResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported.")

    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file.")

    request_id = str(uuid.uuid4())

    input_ = StripAnalyzeInput(
        image_bytes=content,
        filename=image.filename,
        content_type=image.content_type,
    )

    return StripsService.analyze(input_=input_, request_id=request_id)
