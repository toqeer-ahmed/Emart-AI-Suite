"""
Dynamic Brightness Optimization Processor
=========================================
Optionally adjusts brightness dynamically based on mean luminance.
"""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageEnhance, ImageStat


class BrightnessProcessor:
    def process(self, img: Image.Image) -> tuple[Image.Image, dict]:
        """
        Calculates average image luminance and applies dynamic enhancement.
        Dark images get brightened, while bright images remain mostly unchanged.
        """
        stat = ImageStat.Stat(img.convert("L"))
        mean_luminance = stat.mean[0]
        
        factor = 1.0
        status = "neutral"

        if mean_luminance < 110.0:
            factor = 1.0 + min(0.4, (110.0 - mean_luminance) / 150.0)
            status = "brightened"
        elif mean_luminance > 220.0:
            factor = max(0.95, 1.0 - (mean_luminance - 220.0) / 700.0)
            status = "dimmed"

        if factor != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)

        metadata = {
            "mean_luminance": round(mean_luminance, 2),
            "brightness_factor": round(factor, 2),
            "brightness_status": status
        }
        return img, metadata
