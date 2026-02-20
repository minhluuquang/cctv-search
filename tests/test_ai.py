"""Tests for AI module."""

from __future__ import annotations

import pytest

from cctv_search.ai import BoundingBox, DetectedObject
from cctv_search.ai.yolo import YOLODetector


@pytest.fixture
def detector():
    """Create detector fixture."""
    return YOLODetector()


@pytest.mark.asyncio
async def test_detector_loads_model(detector):
    """Test detector can load model."""
    await detector.load_model("/path/to/model.pt")
    assert detector._model_loaded is True


@pytest.mark.asyncio
async def test_detect_raises_when_model_not_loaded(detector):
    """Test detect raises error when model not loaded."""
    with pytest.raises(RuntimeError, match="Model not loaded"):
        await detector.detect(b"frame_data")


@pytest.mark.asyncio
async def test_detect_returns_objects_when_model_loaded(detector):
    """Test detect returns detected objects."""
    await detector.load_model("/path/to/model.pt")
    results = await detector.detect(b"frame_data")
    assert isinstance(results, list)
    assert len(results) > 0
    assert isinstance(results[0], DetectedObject)


@pytest.mark.asyncio
async def test_detect_batch_processes_multiple_frames(detector):
    """Test detect_batch processes multiple frames."""
    await detector.load_model("/path/to/model.pt")
    frames = [b"frame1", b"frame2", b"frame3"]
    results = await detector.detect_batch(frames)
    assert len(results) == 3
    for result in results:
        assert isinstance(result, list)


def test_yolo_detector_supported_labels():
    """Test YOLO detector has supported labels."""
    assert "person" in YOLODetector.SUPPORTED_LABELS
    assert "car" in YOLODetector.SUPPORTED_LABELS


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
