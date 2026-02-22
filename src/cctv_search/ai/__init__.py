"""AI module for object detection and video analysis.

This module provides real-time object detection using RF-DETR,
a transformer-based detection model optimized for CCTV footage.

Features:
- Deep feature extraction for robust object matching
- Occlusion handling via feature similarity
- Feature-based tracking (replaces ByteTrack/BoT-SORT)

Usage:
    from cctv_search.ai import RFDetrDetector, FeatureTracker
    
    detector = RFDetrDetector()
    detector.load_model()
    
    # Detect with features
    detections = detector.detect_with_features(frame)
    
    # Match objects across frames
    tracker = FeatureTracker()
    tracks = tracker.update(detections, frame_idx=0)
"""

from __future__ import annotations

from typing import Protocol


__all__ = [
    "BoundingBox",
    "DetectedObject",
    "RFDetrDetector",
    "FeatureTracker",
    "Track",
    "ObjectDetector",
]


class ObjectDetector(Protocol):
    """Protocol for object detection implementations."""

    def load_model(self) -> None:
        """Load detection model."""
        ...

    def detect(self, frame: bytes) -> list["DetectedObject"]:
        """Detect objects in a video frame."""
        ...


# Import all classes from rf_detr
from cctv_search.ai.rf_detr import BoundingBox, DetectedObject, RFDetrDetector

# Import feature-based tracker
from cctv_search.ai.byte_tracker import FeatureTracker, Track
