"""
Background Improvement Processor
================================
Dynamically cleans, brightens, and flattens light-gray or off-white product backgrounds.
"""
from __future__ import annotations
import numpy as np
from PIL import Image


class BackgroundProcessor:
    def process(self, img: Image.Image) -> tuple[Image.Image, dict]:
        """
        Brightens near-white and off-white backgrounds, reducing gray tint and noise
        using a soft threshold thresholding formula for anti-aliased edge blending.
        """
        img_np = np.array(img.convert("RGB")).astype(np.float32)
        threshold = 215.0
        
        min_channel = np.minimum(np.minimum(img_np[:, :, 0], img_np[:, :, 1]), img_np[:, :, 2])
        mask = np.clip((min_channel - threshold) / (255.0 - threshold), 0.0, 1.0)
        mask = np.expand_dims(mask, axis=-1)
        
        blended = img_np + (255.0 - img_np) * mask
        processed_img = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
        
        metadata = {
            "background_brightened_threshold": threshold,
            "background_clean_method": "SoftThresholdBlend"
        }
        return processed_img, metadata
