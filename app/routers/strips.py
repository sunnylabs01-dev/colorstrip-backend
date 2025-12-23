from fastapi import APIRouter, UploadFile, File

from app.services.strips_service import StripsService

router = APIRouter(prefix="/strips", tags=["strips"])

service = StripsService()


@router.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    return await service.analyze(image=image)
