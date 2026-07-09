"""
Color Balance and Saturation Correction Processor
==================================================
Analyzes color saturation statistics and applies dynamic color enhancement.
"""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image, ImageEnhance


class ColorProcessor:
    def process(self, img: Image.Image) -> tuple[Image.Image, dict]:
        """
        Analyzes HSV saturation channel mean and applies dynamic color balancing.
        Avoids over-saturation to preserve natural product appearance.
        """
        img_np = np.array(img.convert("RGB"))
        img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        
        mean_saturation = np.mean(img_hsv[:, :, 1])
        
        factor = 1.0
        status = "neutral"
        
        if mean_saturation < 40.0:
            factor = 1.12
            status = "color_boosted"
        elif mean_saturation > 150.0:
            factor = 0.95
            status = "color_damped"
            
        if factor != 1.0:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(factor)
            
        metadata = {
            "mean_saturation": round(mean_saturation, 2),
            "color_factor": round(factor, 2),
            "color_status": status
        }
        return img, metadata
