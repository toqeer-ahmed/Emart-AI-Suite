"""
Square Canvas Padding Processor
===============================
Pads the product image into a square canvas, centering the product and maintaining aspect ratio.
Supports transparency preservation.
"""
from __future__ import annotations
from PIL import Image
from app.config import PHOTO_CANVAS_SIZE, PHOTO_BACKGROUND_COLOR


class CanvasProcessor:
    def process(self, img: Image.Image) -> tuple[Image.Image, dict]:
        """
        Pads the enhanced image onto a configurable square canvas.
        Centers the product while preserving aspect ratio and alpha transparency if present.
        """
        target_size = int(PHOTO_CANVAS_SIZE * 0.95)
        
        width, height = img.size
        ratio = min(target_size / width, target_size / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
        mode = "RGBA" if has_alpha else "RGB"
        
        bg_color = PHOTO_BACKGROUND_COLOR if mode == "RGB" else (255, 255, 255, 0)
        canvas = Image.new(mode, (PHOTO_CANVAS_SIZE, PHOTO_CANVAS_SIZE), bg_color)
        
        offset_x = (PHOTO_CANVAS_SIZE - new_width) // 2
        offset_y = (PHOTO_CANVAS_SIZE - new_height) // 2
        
        if mode == "RGBA":
            canvas.paste(resized_img, (offset_x, offset_y), mask=resized_img)
        else:
            canvas.paste(resized_img, (offset_x, offset_y))
        
        metadata = {
            "canvas_size": PHOTO_CANVAS_SIZE,
            "background_color": list(bg_color) if isinstance(bg_color, tuple) else bg_color,
            "product_scaled_dimensions": [new_width, new_height],
            "transparency_preserved": has_alpha
        }
        return canvas, metadata
