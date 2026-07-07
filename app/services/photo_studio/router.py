from fastapi import APIRouter
from app.services.photo_studio.engine import get_photo_studio_engine
from app.shared.schemas import PhotoEnhanceRequest, PhotoEnhanceResult

router = APIRouter(prefix="/photo-studio", tags=["Photo Studio"])


@router.post("/enhance", response_model=PhotoEnhanceResult)
def enhance_photo(req: PhotoEnhanceRequest):
    return get_photo_studio_engine().enhance(req)
