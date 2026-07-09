"""
Image Validation Processor
==========================
Validates file size, formats/MIME types, and detects corrupted images.
"""
from __future__ import annotations
import io
from PIL import Image
from app.config import PHOTO_MAX_UPLOAD_SIZE, PHOTO_SUPPORTED_FORMATS


class ValidationProcessor:
    def validate(self, file_bytes: bytes, filename: str, content_type: str) -> tuple[bool, str]:
        """
        Validates uploaded image payload.
        Returns (is_valid, error_message).
        """
        if len(file_bytes) == 0:
            return False, "Empty file uploaded."
            
        if len(file_bytes) > PHOTO_MAX_UPLOAD_SIZE:
            max_mb = PHOTO_MAX_UPLOAD_SIZE / (1024 * 1024)
            return False, f"File size exceeds maximum limit of {max_mb:.1f}MB."

        if content_type not in PHOTO_SUPPORTED_FORMATS:
            ext = filename.split(".")[-1].lower() if "." in filename else ""
            if ext not in ["jpg", "jpeg", "png", "webp"]:
                return False, f"Unsupported file type. Supported formats: {', '.join(PHOTO_SUPPORTED_FORMATS)}"

        try:
            img = Image.open(io.BytesIO(file_bytes))
            img.verify()
            
            img = Image.open(io.BytesIO(file_bytes))
            img.load()
        except Exception:
            return False, "Corrupted image file detected."

        return True, ""
