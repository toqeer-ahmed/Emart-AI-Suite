"""
Photo Studio Test Suite
======================
Covers image validation, dynamic processors (brightness, contrast, sharpness,
background, color, denoise, canvas, optimizer), engine orchestration,
and router API integration.
"""
from __future__ import annotations
import io
import os
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.services.photo_studio.engine import get_photo_studio_engine, PhotoStudioEngine
from app.services.photo_studio.processors import (
    ValidationProcessor,
    BrightnessProcessor,
    ContrastProcessor,
    SharpnessProcessor,
    BackgroundProcessor,
    ColorProcessor,
    DenoiseProcessor,
    CanvasProcessor,
    OutputOptimizer,
    QualityAssessmentProcessor,
    BackgroundRemovalProcessor,
)
from app.config import PHOTO_MAX_UPLOAD_SIZE, PHOTO_SUPPORTED_FORMATS
from tests.helpers import create_dummy_image, cleanup_generated_file

client = TestClient(app, raise_server_exceptions=False)



# ==========================================
# 1. PROCESSOR UNIT TESTS
# ==========================================

def test_validation_processor_valid():
    proc = ValidationProcessor()
    img_bytes = create_dummy_image()
    is_valid, err = proc.validate(img_bytes, "test.jpg", "image/jpeg")
    assert is_valid is True
    assert err == ""


def test_validation_processor_empty():
    proc = ValidationProcessor()
    is_valid, err = proc.validate(b"", "test.jpg", "image/jpeg")
    assert is_valid is False
    assert "Empty" in err


def test_validation_processor_too_large():
    proc = ValidationProcessor()
    img_bytes = create_dummy_image()
    # Patch maximum upload limit to trigger error
    with patch("app.services.photo_studio.processors.validation.PHOTO_MAX_UPLOAD_SIZE", 50):
        is_valid, err = proc.validate(img_bytes, "test.jpg", "image/jpeg")
        assert is_valid is False
        assert "exceeds" in err


def test_validation_processor_unsupported_format():
    proc = ValidationProcessor()
    is_valid, err = proc.validate(b"fakebytes", "test.exe", "application/octet-stream")
    assert is_valid is False
    assert "Unsupported" in err


def test_validation_processor_corrupted():
    proc = ValidationProcessor()
    # Sending invalid corrupted bytes with a supported format/extension
    is_valid, err = proc.validate(b"corrupted garbage bytes here", "test.jpg", "image/jpeg")
    assert is_valid is False
    assert "Corrupted" in err


def test_brightness_processor_dark():
    proc = BrightnessProcessor()
    # Create dark image (RGB=(30, 30, 30))
    img = Image.open(io.BytesIO(create_dummy_image(color=(30, 30, 30))))
    enhanced_img, meta = proc.process(img)
    assert meta["brightness_status"] == "brightened"
    assert meta["brightness_factor"] > 1.0


def test_brightness_processor_bright():
    proc = BrightnessProcessor()
    # Create bright image (RGB=(240, 240, 240))
    img = Image.open(io.BytesIO(create_dummy_image(color=(240, 240, 240))))
    enhanced_img, meta = proc.process(img)
    assert meta["brightness_status"] == "dimmed"
    assert meta["brightness_factor"] < 1.0


def test_contrast_processor():
    proc = ContrastProcessor()
    img = Image.open(io.BytesIO(create_dummy_image()))
    enhanced_img, meta = proc.process(img)
    assert enhanced_img.size == img.size
    assert meta["contrast_method"] == "CLAHE_LAB"


def test_sharpness_processor():
    proc = SharpnessProcessor()
    img = Image.open(io.BytesIO(create_dummy_image()))
    enhanced_img, meta = proc.process(img)
    assert enhanced_img.size == img.size
    assert "sharpen_percent" in meta


def test_background_processor_brightens():
    proc = BackgroundProcessor()
    # Off-white gray background (RGB=220, 220, 220) should move closer to 255
    img = Image.open(io.BytesIO(create_dummy_image(width=10, height=10, color=(220, 220, 220))))
    enhanced_img, meta = proc.process(img)
    enhanced_np = np.array(enhanced_img)
    # Average color of output should be brighter than original 220
    avg_color = np.mean(enhanced_np)
    assert avg_color > 220



def test_color_processor():
    proc = ColorProcessor()
    img = Image.open(io.BytesIO(create_dummy_image(color=(100, 150, 100))))
    enhanced_img, meta = proc.process(img)
    assert enhanced_img.size == img.size
    assert "mean_saturation" in meta


def test_denoise_processor():
    proc = DenoiseProcessor()
    img = Image.open(io.BytesIO(create_dummy_image()))
    enhanced_img, meta = proc.process(img)
    assert enhanced_img.size == img.size
    assert meta["denoise_applied"] is True


def test_canvas_processor():
    proc = CanvasProcessor()
    # Original aspect ratio (200 x 100)
    img = Image.open(io.BytesIO(create_dummy_image(width=200, height=100)))
    canvas_img, meta = proc.process(img)
    
    # Assert result is perfectly square
    assert canvas_img.width == canvas_img.height
    assert meta["canvas_size"] == canvas_img.width


def test_optimizer_processor():
    proc = OutputOptimizer()
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    out_path, actual_filename, meta = proc.optimize_and_save(img, "test_opt.jpg")
    
    assert os.path.exists(out_path)
    assert meta["output_format"] == "JPEG"
    assert meta["save_parameters"]["optimize"] is True
    
    # Cleanup file
    if os.path.exists(out_path):
        os.remove(out_path)



# ==========================================
# 2. ENGINE & API INTEGRATION TESTS
# ==========================================

def test_engine_orchestration_success():
    engine = PhotoStudioEngine()
    img_bytes = create_dummy_image()
    
    result = engine.process_image(img_bytes, "product.jpg", "image/jpeg")
    
    assert result.status == "success"
    assert result.processed_url.startswith("/static/photo-studio/")
    assert result.dimensions == [800, 800]  # square canvas size
    assert "output_size_bytes" in result.metadata
    
    # Cleanup physical file written during test
    filename = result.processed_url.split("/")[-1]
    from app.config import PHOTO_OUTPUT_DIR
    full_path = os.path.join(PHOTO_OUTPUT_DIR, filename)
    if os.path.exists(full_path):
        os.remove(full_path)


def test_router_upload_success():
    img_bytes = create_dummy_image()
    files = {"file": ("product.jpg", img_bytes, "image/jpeg")}
    
    response = client.post("/photo-studio/process", files=files)
    assert response.status_code == 200
    
    json_data = response.json()
    assert json_data["status"] == "success"
    assert json_data["processed_url"].startswith("/static/photo-studio/")
    assert len(json_data["dimensions"]) == 2
    
    # Cleanup file
    filename = json_data["processed_url"].split("/")[-1]
    from app.config import PHOTO_OUTPUT_DIR
    full_path = os.path.join(PHOTO_OUTPUT_DIR, filename)
    if os.path.exists(full_path):
        os.remove(full_path)


def test_router_upload_corrupted():
    files = {"file": ("corrupted.jpg", b"invalid_binary_stream_data", "image/jpeg")}
    response = client.post("/photo-studio/process", files=files)
    assert response.status_code == 400
    assert "Corrupted" in response.json()["detail"]


def test_png_transparency_preservation():
    # Generate transparent image (RGBA)
    rgba_bytes = create_dummy_image(format_str="PNG", color=(100, 100, 100))
    # Open with PIL to make it RGBA
    img = Image.new("RGBA", (100, 100), (100, 100, 100, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    rgba_bytes = buf.getvalue()
    
    engine = get_photo_studio_engine()
    result = engine.process_image(rgba_bytes, "transparent.png", "image/png", requested_format="AUTO")
    
    assert result.status == "success"
    assert result.output_format == "PNG"
    assert result.processed_url.endswith(".png")
    
    # Cleanup file
    filename = result.processed_url.split("/")[-1]
    from app.config import PHOTO_OUTPUT_DIR
    full_path = os.path.join(PHOTO_OUTPUT_DIR, filename)
    if os.path.exists(full_path):
        os.remove(full_path)


def test_webp_export():
    img_bytes = create_dummy_image()
    engine = get_photo_studio_engine()
    result = engine.process_image(img_bytes, "product.jpg", "image/jpeg", requested_format="WEBP")
    
    assert result.status == "success"
    assert result.output_format == "WEBP"
    assert result.processed_url.endswith(".webp")
    
    # Cleanup file
    filename = result.processed_url.split("/")[-1]
    from app.config import PHOTO_OUTPUT_DIR
    full_path = os.path.join(PHOTO_OUTPUT_DIR, filename)
    if os.path.exists(full_path):
        os.remove(full_path)


def test_batch_processing_success():
    img_bytes = create_dummy_image()
    # Upload 3 files in a single batch
    files = [
        ("files", ("p1.jpg", img_bytes, "image/jpeg")),
        ("files", ("p2.jpg", img_bytes, "image/jpeg")),
        ("files", ("p3.jpg", img_bytes, "image/jpeg"))
    ]
    
    response = client.post("/photo-studio/process-batch", files=files)
    assert response.status_code == 200
    
    json_data = response.json()
    assert json_data["total"] == 3
    assert json_data["successful"] == 3
    assert json_data["failed_count"] == 0
    assert len(json_data["processed"]) == 3
    assert len(json_data["failed"]) == 0
    
    # Verify rich metadata is populated
    p0 = json_data["processed"][0]
    assert p0["original_dimensions"] == [100, 100]
    assert p0["processed_dimensions"] == [800, 800]
    assert p0["original_size_bytes"] > 0
    assert p0["processed_size_bytes"] > 0
    assert p0["compression_ratio"] > 0.0
    assert "validation" in p0["stages_applied"]

    # Clean up physical files
    from app.config import PHOTO_OUTPUT_DIR
    for item in json_data["processed"]:
        fn = item["processed_url"].split("/")[-1]
        path = os.path.join(PHOTO_OUTPUT_DIR, fn)
        if os.path.exists(path):
            os.remove(path)


def test_batch_processing_mixed():
    img_bytes = create_dummy_image()
    # 1 valid file, 1 corrupted file
    files = [
        ("files", ("p1.jpg", img_bytes, "image/jpeg")),
        ("files", ("corrupted.jpg", b"bad_bytes", "image/jpeg"))
    ]
    
    response = client.post("/photo-studio/process-batch", files=files)
    assert response.status_code == 200
    
    json_data = response.json()
    assert json_data["total"] == 2
    assert json_data["successful"] == 1
    assert json_data["failed_count"] == 1
    assert len(json_data["processed"]) == 1
    assert len(json_data["failed"]) == 1
    assert json_data["failed"][0]["filename"] == "corrupted.jpg"
    assert "Corrupted" in json_data["failed"][0]["error"]
    
    # Clean up physical file
    from app.config import PHOTO_OUTPUT_DIR
    fn = json_data["processed"][0]["processed_url"].split("/")[-1]
    path = os.path.join(PHOTO_OUTPUT_DIR, fn)
    if os.path.exists(path):
        os.remove(path)


def test_optional_processors_skip():
    img_bytes = create_dummy_image()
    
    # Turn off contrast and sharpen stages
    params = {
        "contrast": False,
        "sharpen": False
    }
    
    response = client.post("/photo-studio/process", files={"file": ("p.jpg", img_bytes, "image/jpeg")}, params=params)
    assert response.status_code == 200
    
    json_data = response.json()
    stages = json_data["stages_applied"]
    
    # Assert that disabled stages were not executed
    assert "contrast" not in stages
    assert "sharpen" not in stages
    assert "brightness" in stages  # Still runs by default
    
    # Clean up file
    from app.config import PHOTO_OUTPUT_DIR
    fn = json_data["processed_url"].split("/")[-1]
    path = os.path.join(PHOTO_OUTPUT_DIR, fn)
    if os.path.exists(path):
        os.remove(path)


def test_exif_transpose_called():
    img_bytes = create_dummy_image()
    engine = get_photo_studio_engine()
    
    with patch("PIL.ImageOps.exif_transpose") as mock_transpose:
        # Mock transpose to return a new dummy image
        dummy_img = Image.new("RGB", (10, 10))
        mock_transpose.return_value = dummy_img
        
        result = engine.process_image(img_bytes, "exif.jpg", "image/jpeg", enable_exif=True)
        assert mock_transpose.called


def test_backend_cpu_detection():
    from app.services.photo_studio.backend import CPUBackend
    backend = CPUBackend()
    assert backend.get_name() == "CPU"
    assert backend.has_gpu() is False


def test_backend_cuda_detection():
    from app.services.photo_studio.backend import detect_backend
    mock_cuda = MagicMock()
    mock_cuda.getCudaEnabledDeviceCount.return_value = 1
    with patch("cv2.cuda", mock_cuda, create=True):
        backend = detect_backend()
        assert backend.get_name() == "CUDA"
        assert backend.has_gpu() is True


def test_quality_assessment_scoring():
    proc = QualityAssessmentProcessor()
    img = Image.new("RGB", (100, 100), (128, 128, 128))
    metrics = proc.assess(img)
    
    assert "brightness" in metrics
    assert "contrast" in metrics
    assert "sharpness" in metrics
    assert "noise" in metrics
    assert "exposure" in metrics
    assert "quality_score" in metrics
    assert 0 <= metrics["quality_score"] <= 100


def test_background_removal_rmbg_strategy_success():
    import sys
    from PIL import ImageDraw
    from app.services.photo_studio.processors.background_removal import RmbgStrategy
    
    mock_rembg = MagicMock()
    mock_remove = MagicMock()
    
    out_img = Image.new("RGBA", (100, 100), (128, 128, 128, 255))
    draw = ImageDraw.Draw(out_img)
    draw.rectangle([0, 0, 50, 100], fill=(0, 0, 0, 0))
    
    mock_remove.return_value = out_img
    mock_rembg.remove = mock_remove
    
    sys.modules["rembg"] = mock_rembg
    
    proc = RmbgStrategy()
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    
    res_img, success, confidence = proc.remove_background(img)
    assert success is True
    assert confidence > 0.5
    assert res_img.mode == "RGBA"
    
    sys.modules.pop("rembg", None)




def test_background_removal_fallback_to_whitening():
    engine = get_photo_studio_engine()
    img_bytes = create_dummy_image()
    
    result = engine.process_image(img_bytes, "p.jpg", "image/jpeg", background_mode="remove")
    assert result.status == "success"
    
    assert result.metadata["bg_removal_fallback_triggered"] is True
    assert result.metadata["bg_removal_fallback_method"] == "BackgroundWhitening"
    
    cleanup_generated_file(result.processed_url)


def test_adaptive_pipeline_skips():
    engine = get_photo_studio_engine()
    img_bytes = create_dummy_image(color=(127, 127, 127))
    
    result = engine.process_image(img_bytes, "p.jpg", "image/jpeg")
    assert result.status == "success"
    
    assert "brightness" in result.metadata["adaptive_skips"]
    assert "brightness" not in result.stages_applied
    
    cleanup_generated_file(result.processed_url)


def test_parallel_batch_processing():
    import concurrent.futures
    engine = get_photo_studio_engine()
    img_bytes = create_dummy_image()
    
    def run_task(background_mode):
        return engine.process_image(
            file_bytes=img_bytes,
            filename=f"test_{background_mode}.jpg",
            content_type="image/jpeg",
            background_mode=background_mode
        )
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(run_task, "remove"): "remove",
            executor.submit(run_task, "white"): "white"
        }
        results = {}
        for fut in concurrent.futures.as_completed(futures):
            mode = futures[fut]
            results[mode] = fut.result()
            
    assert results["remove"].status == "success"
    assert results["white"].status == "success"
    
    assert results["remove"].metadata["bg_removal_fallback_triggered"] is True
    assert "bg_removal_fallback_triggered" not in results["white"].metadata
    
    cleanup_generated_file(results["remove"].processed_url)
    cleanup_generated_file(results["white"].processed_url)
