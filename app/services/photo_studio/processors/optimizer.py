"""
Output Optimization Processor
=============================
Optimizes the output image compression, saving progressive JPEGs, lossless PNGs, or WebP formats.
"""
from __future__ import annotations
import os
from PIL import Image
from app.config import (
    PHOTO_JPEG_QUALITY,
    PHOTO_PNG_COMPRESSION,
    PHOTO_WEBP_QUALITY,
    PHOTO_OUTPUT_DIR,
)


class OutputOptimizer:
    def optimize_and_save(
        self,
        img: Image.Image,
        filename: str,
        original_has_alpha: bool = False,
        requested_format: str = "AUTO"
    ) -> tuple[str, str, dict]:
        """
        Saves the processed image based on dynamic transparency analysis or client requests.
        Returns: (output_path, actual_filename, metadata)
        """
        os.makedirs(PHOTO_OUTPUT_DIR, exist_ok=True)
        
        fmt = requested_format.upper()
        if fmt == "AUTO":
            fmt = "PNG" if original_has_alpha else "JPEG"

        base_name = os.path.splitext(filename)[0]

        if fmt == "PNG":
            ext = "png"
            actual_filename = f"{base_name}.{ext}"
            out_path = os.path.join(PHOTO_OUTPUT_DIR, actual_filename)
            save_img = img if img.mode in ("RGBA", "RGB") else img.convert("RGBA")
            save_img.save(out_path, "PNG", compress_level=PHOTO_PNG_COMPRESSION)
            params = {"compress_level": PHOTO_PNG_COMPRESSION}
            
        elif fmt == "WEBP":
            ext = "webp"
            actual_filename = f"{base_name}.{ext}"
            out_path = os.path.join(PHOTO_OUTPUT_DIR, actual_filename)
            img.save(out_path, "WEBP", quality=PHOTO_WEBP_QUALITY)
            params = {"quality": PHOTO_WEBP_QUALITY}
            
        else:
            ext = "jpg"
            actual_filename = f"{base_name}.{ext}"
            out_path = os.path.join(PHOTO_OUTPUT_DIR, actual_filename)
            save_img = img.convert("RGB") if img.mode == "RGBA" else img
            save_img.save(out_path, "JPEG", quality=PHOTO_JPEG_QUALITY, optimize=True, progressive=True)
            params = {"quality": PHOTO_JPEG_QUALITY, "optimize": True, "progressive": True}

        size_bytes = os.path.getsize(out_path)

        metadata = {
            "output_size_bytes": size_bytes,
            "output_format": fmt,
            "save_parameters": params
        }
        return out_path, actual_filename, metadata
