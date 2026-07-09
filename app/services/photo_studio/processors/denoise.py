"""
Noise Reduction Processor
=========================
Applies edge-preserving bilateral filtering to reduce noise without blurring product details.
"""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image


class DenoiseProcessor:
    def process(self, img: Image.Image) -> tuple[Image.Image, dict]:
        """
        Applies a lightweight bilateral filter to smooth out noise while preserving sharp boundaries.
        """
        img_np = np.array(img.convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        denoised_bgr = cv2.bilateralFilter(img_bgr, d=5, sigmaColor=20, sigmaSpace=20)

        denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
        processed_img = Image.fromarray(denoised_rgb)

        metadata = {
            "noise_variance_estimate": round(laplacian_var, 2),
            "denoise_applied": True,
            "filter_method": "BilateralFilter",
            "filter_params": {"d": 5, "sigma_color": 20, "sigma_space": 20}
        }
        return processed_img, metadata
