"""
Photo Studio Schema Contracts
=============================
"""
from __future__ import annotations
from typing import Dict, Any, List
from pydantic import BaseModel, Field


class PhotoProcessResult(BaseModel):
    processed_url: str
    dimensions: List[int] = Field(default_factory=list, description="Processed image [width, height] (Backward Compatible)")
    original_dimensions: List[int] = Field(default_factory=list, description="[width, height] of source image")
    processed_dimensions: List[int] = Field(default_factory=list, description="[width, height] of processed image")
    original_size_bytes: int = 0
    processed_size_bytes: int = 0
    compression_ratio: float = 1.0
    output_format: str = "JPEG"
    duration_seconds: float
    status: str
    stages_applied: List[str] = Field(default_factory=list, description="Stages that executed successfully")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata from processing stages")


class PhotoFailedResult(BaseModel):
    filename: str
    error: str


class PhotoBatchResult(BaseModel):
    processed: List[PhotoProcessResult] = Field(default_factory=list)
    failed: List[PhotoFailedResult] = Field(default_factory=list)
    total: int = 0
    successful: int = 0
    failed_count: int = 0
