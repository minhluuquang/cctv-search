"""AI module for object detection and video analysis.

This module provides real-time object detection using RF-DETR,
a transformer-based detection model optimized for CCTV footage.

Example:
    from cctv_search.ai import RFDetrDetector
    
    detector = RFDetrDetector()
    detector.load_model()
    
    detections = detector.detect(frame)
    for det in detections:
        print(f"{det.label}: {det.confidence:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


__all__ = [
    "BoundingBox",
    "DetectedObject",
    "RFDetrDetector",
    "ByteTrackTracker",
    "Track",
    "ObjectDetector",
]


@dataclass
class BoundingBox:
    """Bounding box for detected objects.
    
    Attributes:
        x: Left coordinate
        y: Top coordinate
        width: Box width
        height: Box height
        confidence: Detection confidence (0-1)
    """

    x: float
    y: float
    width: float
    height: float
    confidence: float


@dataclass
class DetectedObject:
    """Represents a detected object in video.
    
    Attributes:
        label: Object class label (e.g., "bicycle", "person")
        bbox: Bounding box coordinates
        confidence: Detection confidence score
        frame_timestamp: Frame timestamp in seconds
    """

    label: str
    bbox: BoundingBox
    confidence: float
    frame_timestamp: float


class ObjectDetector(Protocol):
    """Protocol for object detection implementations."""

    def load_model(self) -> None:
        """Load detection model."""
        ...

    def detect(self, frame: bytes) -> list[DetectedObject]:
        """Detect objects in a video frame."""
        ...


# Import RF-DETR detector after defining types to avoid circular imports
from cctv_search.ai.rf_detr import RFDetrDetector

# Import ByteTrack tracker
from cctv_search.ai.byte_tracker import ByteTrackTracker, Track
