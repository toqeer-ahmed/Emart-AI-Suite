"""
Image Quality Assessment Processor
==================================
Determines brightness, contrast, sharpness, noise, and exposure statistics to govern adaptive pipeline skipping.
"""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image


class QualityAssessmentProcessor:
    def assess(self, img: Image.Image) -> dict:
        """
        Analyzes the image and returns deterministic quality metrics.
        """
        img_np = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        mean_brightness = np.mean(gray)
        brightness_score = 1.0 - abs(mean_brightness - 127.0) / 127.0
        
        if mean_brightness < 75.0:
            exposure = "poor_dark"
        elif mean_brightness > 200.0:
            exposure = "poor_bright"
        else:
            exposure = "good"

        std_dev = np.std(gray)
        contrast_score = min(1.0, std_dev / 75.0)

        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, laplacian_var / 800.0)

        diff = cv2.absdiff(gray, cv2.medianBlur(gray, 3))
        noise_level = min(1.0, np.mean(diff) / 15.0)

        from app.config import PHOTO_QUALITY_WEIGHTS
        w = PHOTO_QUALITY_WEIGHTS
        weighted_sum = (
            brightness_score * w.get("brightness", 0.25) +
            contrast_score * w.get("contrast", 0.25) +
            sharpness_score * w.get("sharpness", 0.30) +
            (1.0 - noise_level) * w.get("noise", 0.20)
        )
        quality_score = int(np.clip(weighted_sum * 100, 0, 100))

        return {
            "brightness": round(brightness_score, 4),
            "contrast": round(contrast_score, 4),
            "sharpness": round(sharpness_score, 4),
            "noise": round(noise_level, 4),
            "exposure": exposure,
            "quality_score": quality_score
        }
