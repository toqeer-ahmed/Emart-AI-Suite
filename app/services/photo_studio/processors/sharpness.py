"""
Sharpness Enhancement Processor
==============================
Applies unsharp mask filter to enhance product edge details.
"""
from __future__ import annotations
from PIL import Image, ImageFilter
from app.config import PHOTO_SHARPEN_STRENGTH


class SharpnessProcessor:
    def process(self, img: Image.Image) -> tuple[Image.Image, dict]:
        """
        Enhances edge sharpness using UnsharpMask filter with configured strength.
        """
        percent_val = int(100 * PHOTO_SHARPEN_STRENGTH)
        
        processed_img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=percent_val, threshold=2))
        
        metadata = {
            "sharpen_radius": 1.5,
            "sharpen_percent": percent_val,
            "sharpen_threshold": 2
        }
        return processed_img, metadata
