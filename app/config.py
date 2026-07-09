"""
Central configuration.

This is intentionally the ONE file you should need to touch to move from
"demo mode" to "connected to real E-Mart infrastructure":

  - DATA_LAYER: swap SQLiteDataLayer for your EmartDataLayer (see
    app/shared/data_layer.py for the interface to implement).
  - ANTHROPIC_API_KEY: set this env var to switch the Assistant and
    Listing Generator from template mode to live LLM mode.
  - CORS_ORIGINS: add E-Mart's actual frontend domain(s) here before
    going to production.
"""
import os

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CORS_ORIGINS = os.environ.get("EMART_AI_CORS_ORIGINS", "*").split(",")
API_KEY_HEADER_NAME = "X-EMart-AI-Key"
GATEWAY_API_KEY = os.environ.get("EMART_AI_GATEWAY_KEY", "")  # empty = auth disabled (dev mode)

# Photo Studio Configs
PHOTO_MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5MB
PHOTO_MAX_UPLOAD_COUNT = 10
PHOTO_SUPPORTED_FORMATS = {"image/jpeg", "image/png", "image/webp"}
PHOTO_JPEG_QUALITY = 90
PHOTO_PNG_COMPRESSION = 6
PHOTO_WEBP_QUALITY = 80
PHOTO_CANVAS_SIZE = 800
PHOTO_BACKGROUND_COLOR = (255, 255, 255)
PHOTO_SHARPEN_STRENGTH = 1.2
PHOTO_DEFAULT_OUTPUT_FORMAT = "AUTO"
PHOTO_ENABLE_EXIF_CORRECTION = True
PHOTO_DEFAULT_ENHANCEMENT_FLAGS = {
    "brightness": True,
    "contrast": True,
    "denoise": True,
    "color": True,
    "sharpen": True,
    "background": True
}
PHOTO_OUTPUT_DIR = os.environ.get("EMART_AI_PHOTO_OUTPUT_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "services", "photo_studio", "output")))

PHOTO_QUALITY_WEIGHTS = {
    "brightness": 0.25,
    "contrast": 0.25,
    "sharpness": 0.30,
    "noise": 0.20
}
PHOTO_QUALITY_BRIGHTNESS_SKIP = 0.85
PHOTO_QUALITY_CONTRAST_SKIP = 0.80
PHOTO_QUALITY_NOISE_SKIP = 0.05
PHOTO_QUALITY_SHARPNESS_SKIP = 0.75




