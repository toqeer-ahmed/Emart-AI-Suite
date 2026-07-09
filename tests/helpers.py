"""
Test Helper Utilities
=====================
Shared helpers for test mock generation and file cleanup.
"""
from __future__ import annotations
import io
import os
from PIL import Image
from app.config import PHOTO_OUTPUT_DIR


def create_dummy_image(width: int = 100, height: int = 100, color: tuple = (200, 200, 200), format_str: str = "JPEG") -> bytes:
    """Helper to generate dummy image bytes in memory."""
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color)
    img.save(buf, format=format_str)
    return buf.getvalue()


def cleanup_generated_file(url_or_path: str):
    """Safely cleans up enhanced images written to disk during test execution."""
    if not url_or_path:
        return
    filename = url_or_path.split("/")[-1]
    full_path = os.path.join(PHOTO_OUTPUT_DIR, filename)
    if os.path.exists(full_path):
        try:
            os.remove(full_path)
        except Exception:
            pass
