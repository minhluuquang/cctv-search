"""AI module for object detection and video analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class BoundingBox:
    """Bounding box for detected objects."""

    x: float
    y: float
    width: float
    height: float
    confidence: float


@dataclass
class DetectedObject:
    """Represents a detected object in video."""

    label: str
    bbox: BoundingBox
    confidence: float
    frame_timestamp: float


@dataclass
class VideoAnalysis:
    """Results of video analysis."""

    objects: list[DetectedObject]
    total_frames: int
    duration_seconds: float
    processed_at: float


class ObjectDetector(Protocol):
    """Protocol for object detection implementations."""

    async def load_model(self, model_path: str) -> None:
        """Load detection model."""
        ...

    async def detect(self, frame: bytes) -> list[DetectedObject]:
        """Detect objects in a video frame."""
        ...


class VideoSegmenter(Protocol):
    """Protocol for video segmentation implementations."""

    async def segment(self, video_path: str, prompt: str | None = None) -> list[dict]:
        """Segment video into meaningful clips."""
        ...
