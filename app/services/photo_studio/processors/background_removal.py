"""
AI Background Removal Processor (Strategy Pattern)
==================================================
Handles background segmentation using RMBG, U²-Net, or SAM strategies, falling back to whitening.
"""
from __future__ import annotations
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger("photo_studio_bg_removal")


class BackgroundRemovalStrategy:
    def remove_background(self, img: Image.Image) -> tuple[Image.Image, bool, float]:
        """
        Removes image background.
        Returns (rgba_image, success_flag, confidence_score).
        """
        raise NotImplementedError


class RmbgStrategy(BackgroundRemovalStrategy):
    def remove_background(self, img: Image.Image) -> tuple[Image.Image, bool, float]:
        """
        Preconfigured preferred strategy using the rembg library.
        """
        try:
            from rembg import remove  # type: ignore
            
            img_rgba = img.convert("RGBA")
            out_img = remove(img_rgba)
            
            alpha = np.array(out_img)[:, :, 3]
            non_zero_ratio = np.count_nonzero(alpha) / alpha.size
            
            if non_zero_ratio < 0.05 or non_zero_ratio > 0.98:
                return img, False, 0.30
                
            return out_img, True, 0.95
        except ImportError:
            logger.debug("rembg library not installed. Skipping strategy.")
            return img, False, 0.0
        except Exception as e:
            logger.warning(f"RmbgStrategy execution failed: {e}")
            return img, False, 0.0


class UnetStrategy(BackgroundRemovalStrategy):
    def remove_background(self, img: Image.Image) -> tuple[Image.Image, bool, float]:
        """Future implementation placeholder for U²-Net models."""
        logger.debug("UnetStrategy not implemented yet.")
        return img, False, 0.0


class SamStrategy(BackgroundRemovalStrategy):
    def remove_background(self, img: Image.Image) -> tuple[Image.Image, bool, float]:
        """Future implementation placeholder for Segment Anything (SAM) models."""
        logger.debug("SamStrategy not implemented yet.")
        return img, False, 0.0


class BackgroundRemovalProcessor:
    def __init__(self, fallback_processor):
        self.fallback = fallback_processor
        self.strategy = RmbgStrategy()

    def process(self, img: Image.Image) -> tuple[Image.Image, dict]:
        """
        Orchestrates background removal strategy.
        Falls back to background whitening processor if segmentation fails or library is missing.
        """
        out_img, success, confidence = self.strategy.remove_background(img)
        
        metadata = {
            "bg_removal_success": success,
            "bg_removal_confidence": confidence,
            "bg_removal_fallback_triggered": not success
        }
        
        if not success or confidence < 0.5:
            logger.debug("AI background removal unavailable or poor confidence. Running fallback whitening.")
            img, fallback_meta = self.fallback.process(img)
            metadata.update(fallback_meta)
            metadata["bg_removal_fallback_method"] = "BackgroundWhitening"
            return img, metadata
            
        return out_img, metadata
