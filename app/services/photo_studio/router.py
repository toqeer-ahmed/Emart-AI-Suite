"""
Photo Studio API Router
=======================
Handles file uploads via multipart/form-data.
Supports single and batch processing endpoints with dynamic pipeline switches, EXIF tags, and AI background options.
"""
from __future__ import annotations
from typing import List
from fastapi import APIRouter, File, UploadFile, HTTPException

from app.config import PHOTO_MAX_UPLOAD_COUNT
from app.services.photo_studio.engine import get_photo_studio_engine
from app.services.photo_studio.schemas import PhotoProcessResult, PhotoBatchResult, PhotoFailedResult

router = APIRouter(prefix="/photo-studio", tags=["Photo Studio"])


@router.post("/process", response_model=PhotoProcessResult)
async def process_photo(
    file: UploadFile = File(...),
    format: str = "AUTO",
    sharpen: bool = True,
    background: bool = True,
    color: bool = True,
    contrast: bool = True,
    denoise: bool = True,
    brightness: bool = True,
    enable_exif: bool = True,
    background_mode: str = "white"
) -> PhotoProcessResult:
    """
    Uploads a product photo and returns the enhanced version.
    All enhancement steps can be dynamically toggled using request parameters.
    """
    file_bytes = await file.read()
    
    flags = {
        "brightness": brightness,
        "contrast": contrast,
        "denoise": denoise,
        "color": color,
        "sharpen": sharpen,
        "background": background,
    }
    
    engine = get_photo_studio_engine()
    result = engine.process_image(
        file_bytes=file_bytes,
        filename=file.filename or "uploaded_image.jpg",
        content_type=file.content_type or "image/jpeg",
        requested_format=format,
        enhancement_flags=flags,
        enable_exif=enable_exif,
        background_mode=background_mode
    )
    
    if result.status.startswith("failed"):
        error_msg = result.status.replace("failed: ", "")
        if any(keyword in error_msg.lower() for keyword in ["size", "format", "corrupt", "empty", "type"]):
            raise HTTPException(status_code=400, detail=error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
        
    return result


@router.post("/process-batch", response_model=PhotoBatchResult)
async def process_photo_batch(
    files: List[UploadFile] = File(...),
    format: str = "AUTO",
    sharpen: bool = True,
    background: bool = True,
    color: bool = True,
    contrast: bool = True,
    denoise: bool = True,
    brightness: bool = True,
    enable_exif: bool = True,
    background_mode: str = "white"
) -> PhotoBatchResult:
    """
    Batch enhancement endpoint. Accepts up to 10 images.
    Individual failures are isolated and do not crash the entire request.
    """
    if len(files) > PHOTO_MAX_UPLOAD_COUNT:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds maximum limit of {PHOTO_MAX_UPLOAD_COUNT} files."
        )

    processed_list = []
    failed_list = []
    
    flags = {
        "brightness": brightness,
        "contrast": contrast,
        "denoise": denoise,
        "color": color,
        "sharpen": sharpen,
        "background": background,
    }
    
    engine = get_photo_studio_engine()
    
    for f in files:
        file_bytes = await f.read()
        result = engine.process_image(
            file_bytes=file_bytes,
            filename=f.filename or "uploaded_image.jpg",
            content_type=f.content_type or "image/jpeg",
            requested_format=format,
            enhancement_flags=flags,
            enable_exif=enable_exif,
            background_mode=background_mode
        )
        
        if result.status.startswith("failed"):
            failed_list.append(PhotoFailedResult(
                filename=f.filename or "uploaded_image.jpg",
                error=result.status.replace("failed: ", "")
            ))
        else:
            processed_list.append(result)
            
    return PhotoBatchResult(
        processed=processed_list,
        failed=failed_list,
        total=len(files),
        successful=len(processed_list),
        failed_count=len(failed_list)
    )
