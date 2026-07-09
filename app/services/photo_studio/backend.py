"""
GPU Acceleration Backend Abstraction Layer
=========================================
Dynamically detects and configures CPU vs. CUDA GPU hardware backends.
"""
from __future__ import annotations
import logging
import cv2

logger = logging.getLogger("photo_studio_backend")


class ProcessingBackend:
    def get_name(self) -> str:
        raise NotImplementedError

    def has_gpu(self) -> bool:
        raise NotImplementedError


class CPUBackend(ProcessingBackend):
    def get_name(self) -> str:
        return "CPU"

    def has_gpu(self) -> bool:
        return False


class CudaBackend(ProcessingBackend):
    def get_name(self) -> str:
        return "CUDA"

    def has_gpu(self) -> bool:
        return True


def detect_backend() -> ProcessingBackend:
    """
    Detects if CUDA capability is available in OpenCV or system drivers.
    Defaults to CPUBackend.
    """
    try:
        if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            logger.info("OpenCV CUDA device detected. Enabling GPU backend.")
            return CudaBackend()
    except Exception:
        pass
    
    logger.info("Defaulting to CPU backend execution.")
    return CPUBackend()
