"""
Photo Studio Core Processing Engine
===================================
Orchestrates the image enhancement pipeline using modular, dynamically assembled processors.
Supports EXIF orientation correction, Quality Assessment adaptive skipping, background removal strategies,
and hardware backend abstractions.
"""
from __future__ import annotations
import io
import time
import uuid
import logging
from PIL import Image, ImageOps

from app.shared.data_layer import get_data_layer
from app.services.photo_studio.schemas import PhotoProcessResult
from app.config import (
    PHOTO_DEFAULT_ENHANCEMENT_FLAGS,
    PHOTO_ENABLE_EXIF_CORRECTION,
)
from app.services.photo_studio.backend import detect_backend
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

logger = logging.getLogger("photo_studio")


class PhotoStudioEngine:
    def __init__(self):
        self.dl = get_data_layer()
        self.backend = detect_backend()
        self.validator = ValidationProcessor()
        self.quality_assessor = QualityAssessmentProcessor()
        
        self.brightness_proc = BrightnessProcessor()
        self.contrast_proc = ContrastProcessor()
        self.sharpness_proc = SharpnessProcessor()
        self.background_proc = BackgroundProcessor()
        self.color_proc = ColorProcessor()
        self.denoise_proc = DenoiseProcessor()
        
        self.background_removal_proc = BackgroundRemovalProcessor(fallback_processor=self.background_proc)
        
        self.canvas_proc = CanvasProcessor()
        self.optimizer = OutputOptimizer()

        self.processor_map = {
            "brightness": self.brightness_proc,
            "contrast": self.contrast_proc,
            "denoise": self.denoise_proc,
            "color": self.color_proc,
            "sharpen": self.sharpness_proc,
            "background": self.background_proc,
        }

    def process_image(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str,
        requested_format: str = "AUTO",
        enhancement_flags: dict[str, bool] = None,
        enable_exif: bool = True,
        background_mode: str = "white"
    ) -> PhotoProcessResult:
        """
        Runs the uploaded product image through the dynamically assembled, adaptive enhancement pipeline.
        """
        start_time = time.time()
        stages_executed = ["validation"]
        meta_summary = {}

        is_valid, err_msg = self.validator.validate(file_bytes, filename, content_type)
        if not is_valid:
            duration = time.time() - start_time
            self._log_telemetry(filename, [0, 0], [0, 0], len(file_bytes), 0, "AUTO", duration, "failed", err_msg, stages_executed, {})
            return PhotoProcessResult(
                processed_url="",
                dimensions=[0, 0],
                original_dimensions=[0, 0],
                processed_dimensions=[0, 0],
                original_size_bytes=len(file_bytes),
                processed_size_bytes=0,
                compression_ratio=0.0,
                output_format="AUTO",
                duration_seconds=round(duration, 4),
                status=f"failed: {err_msg}",
                stages_applied=stages_executed,
                metadata={"error": err_msg}
            )

        img = None
        try:
            img = Image.open(io.BytesIO(file_bytes))
            orig_width, orig_height = img.size
            original_has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
            stages_executed.append("loader")

            if enable_exif and PHOTO_ENABLE_EXIF_CORRECTION:
                img = ImageOps.exif_transpose(img)
                stages_executed.append("exif_transpose")

            quality_metrics = self.quality_assessor.assess(img)
            meta_summary["quality_assessment"] = quality_metrics
            meta_summary["hardware_backend"] = self.backend.get_name()
            stages_executed.append("quality_assessment")

            from app.config import (
                PHOTO_QUALITY_BRIGHTNESS_SKIP,
                PHOTO_QUALITY_CONTRAST_SKIP,
                PHOTO_QUALITY_NOISE_SKIP,
                PHOTO_QUALITY_SHARPNESS_SKIP,
            )
            skips = []
            if quality_metrics["brightness"] > PHOTO_QUALITY_BRIGHTNESS_SKIP:
                skips.append("brightness")
            if quality_metrics["contrast"] > PHOTO_QUALITY_CONTRAST_SKIP:
                skips.append("contrast")
            if quality_metrics["noise"] < PHOTO_QUALITY_NOISE_SKIP:
                skips.append("denoise")
            if quality_metrics["sharpness"] > PHOTO_QUALITY_SHARPNESS_SKIP:
                skips.append("sharpen")
            meta_summary["adaptive_skips"] = skips

            meta_summary["stages"] = {}

            flags = dict(PHOTO_DEFAULT_ENHANCEMENT_FLAGS)
            if enhancement_flags:
                flags.update({k.lower(): v for k, v in enhancement_flags.items()})

            stages_to_run = ["brightness", "contrast", "denoise", "color", "sharpen", "background"]
            for stage in stages_to_run:
                if stage in skips:
                    logger.debug(f"Adaptive Pipeline: skipping '{stage}' stage due to excellent quality metrics.")
                    continue
                    
                if flags.get(stage, False):
                    if stage == "background":
                        processor = self.background_removal_proc if background_mode == "remove" else self.background_proc
                    else:
                        processor = self.processor_map.get(stage)
                        
                    if processor:
                        img, stage_meta = processor.process(img)
                        meta_summary.update(stage_meta)
                        meta_summary["stages"][stage] = stage_meta
                        stages_executed.append(stage)

            img, canvas_meta = self.canvas_proc.process(img)
            meta_summary.update(canvas_meta)
            meta_summary["stages"]["canvas"] = canvas_meta
            stages_executed.append("canvas")

            safe_id = uuid.uuid4().hex[:12]
            out_filename = f"enhanced_{safe_id}.jpg"
            final_has_alpha = original_has_alpha or (img.mode == "RGBA")
            
            out_path, actual_filename, opt_meta = self.optimizer.optimize_and_save(
                img, out_filename, original_has_alpha=final_has_alpha, requested_format=requested_format
            )
            meta_summary.update(opt_meta)
            meta_summary["stages"]["optimizer"] = opt_meta
            stages_executed.append("optimizer")

            duration = time.time() - start_time
            processed_size = opt_meta.get("output_size_bytes", 0)
            ratio = round(processed_size / max(1, len(file_bytes)), 4)
            fmt = opt_meta.get("output_format", "JPEG")

            self._log_telemetry(
                filename=filename,
                orig_dims=[orig_width, orig_height],
                proc_dims=list(img.size),
                orig_size=len(file_bytes),
                proc_size=processed_size,
                out_format=fmt,
                duration=duration,
                status="success",
                error="",
                stages=stages_executed,
                metadata=meta_summary,
                background_mode=background_mode
            )

            return PhotoProcessResult(
                processed_url=f"/static/photo-studio/{actual_filename}",
                dimensions=list(img.size),
                original_dimensions=[orig_width, orig_height],
                processed_dimensions=list(img.size),
                original_size_bytes=len(file_bytes),
                processed_size_bytes=processed_size,
                compression_ratio=ratio,
                output_format=fmt,
                duration_seconds=round(duration, 4),
                status="success",
                stages_applied=stages_executed,
                metadata=meta_summary
            )

        except Exception as e:
            duration = time.time() - start_time
            err_str = str(e)
            logger.exception("Error executing photo studio processing pipeline")
            self._log_telemetry(
                filename=filename,
                orig_dims=[0, 0],
                proc_dims=[0, 0],
                orig_size=len(file_bytes),
                proc_size=0,
                out_format="AUTO",
                duration=duration,
                status="failed",
                error=err_str,
                stages=stages_executed,
                metadata=meta_summary,
                background_mode=background_mode
            )
            
            return PhotoProcessResult(
                processed_url="",
                dimensions=[0, 0],
                original_dimensions=[0, 0],
                processed_dimensions=[0, 0],
                original_size_bytes=len(file_bytes),
                processed_size_bytes=0,
                compression_ratio=0.0,
                output_format="AUTO",
                duration_seconds=round(duration, 4),
                status=f"failed: {err_str}",
                stages_applied=stages_executed,
                metadata={"error": err_str}
            )
        finally:
            if img is not None:
                img.close()

    def _log_telemetry(
        self, 
        filename: str, 
        orig_dims: list[int],
        proc_dims: list[int],
        orig_size: int,
        proc_size: int,
        out_format: str,
        duration: float, 
        status: str, 
        error: str, 
        stages: list[str], 
        metadata: dict,
        background_mode: str = "white"
    ):
        """
        Logs telemetry metrics cleanly to continuous learning data layer.
        Ensures privacy (never logs pixel buffers or image contents).
        """
        telemetry_payload = {
            "original_filename": filename,
            "original_dimensions": orig_dims,
            "processed_dimensions": proc_dims,
            "original_size_bytes": orig_size,
            "processed_size_bytes": proc_size,
            "compression_ratio": round(proc_size / max(1, orig_size), 4),
            "output_format": out_format,
            "duration_seconds": round(duration, 4),
            "status": status,
            "error_message": error,
            "stages_executed": stages,
            "hardware_backend": self.backend.get_name(),
            "brightness_factor": metadata.get("brightness_factor", 1.0),
            "color_factor": metadata.get("color_factor", 1.0),
            "quality_score": metadata.get("quality_assessment", {}).get("quality_score", 0),
            "adaptive_skips": metadata.get("adaptive_skips", []),
            "background_mode": background_mode,
        }
        try:
            self.dl.log_signal("photo_processed", telemetry_payload)
        except Exception as e:
            logger.warning(f"Failed logging telemetry signal for photo studio: {e}", exc_info=True)


_engine_singleton: PhotoStudioEngine = None


def get_photo_studio_engine() -> PhotoStudioEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = PhotoStudioEngine()
    return _engine_singleton
