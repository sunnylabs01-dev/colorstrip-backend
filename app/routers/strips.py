from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas.strips import AnalyzeResponse
from app.services.strips_service import StripsService

router = APIRouter(prefix="/v1/strips", tags=["strips"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_strip(image: UploadFile = File(...)) -> AnalyzeResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported.")

    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file.")

    return StripsService.analyze_dummy(filename=image.filename or "unknown", size_bytes=len(content))
