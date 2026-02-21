"""Tests for AI module."""

from __future__ import annotations

import pytest

from cctv_search.ai import BoundingBox, DetectedObject, RFDetrDetector


@pytest.fixture
def detector():
    """Create detector fixture."""
    return RFDetrDetector()


def test_detector_creation():
    """Test detector can be created."""
    detector = RFDetrDetector()
    assert detector.confidence_threshold == 0.5
    assert detector._model_loaded is False


def test_detector_with_custom_threshold():
    """Test detector with custom confidence threshold."""
    detector = RFDetrDetector(confidence_threshold=0.7)
    assert detector.confidence_threshold == 0.7


def test_detect_raises_when_model_not_loaded(detector):
    """Test detect raises error when model not loaded."""
    with pytest.raises(RuntimeError, match="Model not loaded"):
        detector.detect(b"frame_data")


def test_detected_object_creation():
    """Test DetectedObject dataclass creation."""
    bbox = BoundingBox(x=10.0, y=20.0, width=50.0, height=100.0, confidence=0.95)
    obj = DetectedObject(
        label="person",
        bbox=bbox,
        confidence=0.95,
        frame_timestamp=1234567890.0,
    )
    assert obj.label == "person"
    assert obj.confidence == 0.95
    assert obj.bbox.x == 10.0


def test_bounding_box_properties():
    """Test BoundingBox properties."""
    bbox = BoundingBox(x=10.0, y=20.0, width=50.0, height=100.0, confidence=0.95)
    assert bbox.x == 10.0
    assert bbox.y == 20.0
    assert bbox.width == 50.0
    assert bbox.height == 100.0
    assert bbox.confidence == 0.95
