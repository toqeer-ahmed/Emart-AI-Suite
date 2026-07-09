"""
Adaptive Contrast Optimization Processor
========================================
Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the luminance channel.
"""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image


class ContrastProcessor:
    def process(self, img: Image.Image) -> tuple[Image.Image, dict]:
        """
        Enhances image contrast dynamically using CLAHE in LAB color space.
        Improves local contrast without color shifting or clipping shadows/highlights.
        """
        img_np = np.array(img.convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl_l = clahe.apply(l_channel)

        enhanced_lab = cv2.merge((cl_l, a_channel, b_channel))

        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

        processed_img = Image.fromarray(enhanced_rgb)

        metadata = {
            "contrast_method": "CLAHE_LAB",
            "clip_limit": 2.0,
            "tile_grid_size": [8, 8]
        }
        return processed_img, metadata
