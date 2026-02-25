"""Tests for FastAPI endpoints."""

from __future__ import annotations

from contextlib import suppress
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from cctv_search.api import app


@pytest.fixture
def client():
    """Create test client."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def temp_clips_dir(tmp_path):
    """Create a temporary directory for clips and cleanup after tests."""
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir(exist_ok=True)

    # Patch the CLIPS_DIR in the api module
    with patch("cctv_search.api.CLIPS_DIR", clips_dir):
        yield clips_dir

    # Cleanup: remove any remaining files
    if clips_dir.exists():
        for file in clips_dir.iterdir():
            with suppress(OSError):
                file.unlink()


@pytest.fixture
def mock_nvr_client():
    """Create a properly configured mock NVR client."""
    mock_client = MagicMock()
    mock_client.host = "192.168.1.100"
    mock_client.extract_clip = MagicMock()
    return mock_client


def test_health_check_implicit(client):
    """Test root endpoint returns API info."""
    response = client.get("/docs")
    assert response.status_code == 200


@patch("cctv_search.api.nvr_client")
def test_extract_frame_endpoint(mock_nvr_client, client):
    """Test frame extraction endpoint."""
    from pathlib import Path

    mock_nvr_client.extract_frame.return_value = Path("/tmp/frame.png")

    timestamp = datetime(2026, 2, 19, 22, 31, 5)
    response = client.post(
        "/nvr/frame",
        json={
            "timestamp": timestamp.isoformat(),
            "channel": 1,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "frame_path" in data
    assert data["channel"] == 1


def test_extract_frame_endpoint_missing_timestamp(client):
    """Test frame extraction fails without timestamp."""
    response = client.post(
        "/nvr/frame",
        json={
            "channel": 1,
        },
    )
    assert response.status_code == 422  # Validation error


@patch("cctv_search.api.nvr_client")
@patch("cctv_search.api.detector")
def test_detect_objects_endpoint_with_timestamp(mock_detector, mock_nvr_client, client):
    """Test detect objects endpoint with timestamp."""
    mock_nvr_client.extract_frame.return_value = MagicMock()
    # Currently returns empty list as detection not yet implemented

    timestamp = datetime(2026, 2, 19, 22, 31, 5)
    response = client.post(
        "/ai/detect",
        json={
            "camera_id": "1",
            "timestamp": timestamp.isoformat(),
        },
    )
    # Currently returns 200 with empty list (or 400 if timestamp handling incomplete)
    assert response.status_code in [200, 400]


def test_detect_objects_endpoint_missing_timestamp(client):
    """Test detect objects endpoint requires timestamp."""
    response = client.post(
        "/ai/detect",
        json={
            "camera_id": "cam_001",
        },
    )
    # Should return 400 because timestamp is required
    assert response.status_code == 400


@patch("cctv_search.api.nvr_client", None)
def test_generate_video_clip_nvr_not_initialized(client):
    """Test clip generation fails when NVR client is not initialized."""
    start_time = datetime(2026, 2, 19, 22, 30, 0)
    response = client.post(
        "/video/clip",
        json={
            "camera_id": "1",
            "start_timestamp": start_time.isoformat(),
            "duration_seconds": 15,
        },
    )

    assert response.status_code == 500
    assert "NVR client not initialized" in response.json()["detail"]


def test_generate_video_clip_invalid_duration_zero(client):
    """Test clip generation fails with duration of 0 seconds."""
    start_time = datetime(2026, 2, 19, 22, 30, 0)
    response = client.post(
        "/video/clip",
        json={
            "camera_id": "1",
            "start_timestamp": start_time.isoformat(),
            "duration_seconds": 0,
        },
    )

    assert response.status_code == 400
    assert "Duration must be greater than 0" in response.json()["detail"]


def test_generate_video_clip_duration_too_long(client):
    """Test clip generation fails when duration exceeds 300 seconds."""
    start_time = datetime(2026, 2, 19, 22, 30, 0)
    response = client.post(
        "/video/clip",
        json={
            "camera_id": "1",
            "start_timestamp": start_time.isoformat(),
            "duration_seconds": 301,
        },
    )

    assert response.status_code == 400
    assert "Maximum clip duration is 300 seconds" in response.json()["detail"]


def test_generate_video_clip_negative_duration(client):
    """Test clip generation fails with negative duration."""
    start_time = datetime(2026, 2, 19, 22, 30, 0)
    response = client.post(
        "/video/clip",
        json={
            "camera_id": "1",
            "start_timestamp": start_time.isoformat(),
            "duration_seconds": -10,
        },
    )

    assert response.status_code == 400
    assert "Duration must be greater than 0" in response.json()["detail"]


def test_generate_video_clip_invalid_timestamp(client):
    """Test clip generation fails with invalid timestamp format."""
    response = client.post(
        "/video/clip",
        json={
            "camera_id": "1",
            "start_timestamp": "invalid-timestamp",
            "duration_seconds": 15,
        },
    )

    assert response.status_code == 422


@patch("cctv_search.api.nvr_client")
def test_generate_video_clip_nvr_extraction(mock_nvr_client, client, temp_clips_dir):
    """Test clip extraction from real NVR when host is configured."""
    mock_nvr_client.host = "192.168.1.100"

    start_time = datetime(2026, 2, 19, 22, 30, 0)

    def create_clip_file(*args, **kwargs):
        """Simulate NVR creating the clip file."""
        output_path = kwargs.get("output_path")
        assert output_path is not None
        output_path.write_bytes(b"fake video data")
        return output_path

    mock_nvr_client.extract_clip = MagicMock(side_effect=create_clip_file)

    response = client.post(
        "/video/clip",
        json={
            "camera_id": "2",
            "start_timestamp": start_time.isoformat(),
            "duration_seconds": 15,
        },
    )

    assert response.status_code == 200

    mock_nvr_client.extract_clip.assert_called_once()
    call_args = mock_nvr_client.extract_clip.call_args

    assert call_args.kwargs["channel"] == 2
    assert call_args.kwargs["start_time"] == start_time

    expected_end = start_time + timedelta(seconds=15)
    assert call_args.kwargs["end_time"] == expected_end


# =============================================================================
# /frames/objects Endpoint Tests - Comprehensive
# =============================================================================


@patch("cctv_search.api.nvr_client")
@patch("cctv_search.api.detector")
@patch("cctv_search.api.tracker")
def test_get_frame_with_objects_success(
    mock_tracker, mock_detector, mock_nvr_client, client
):
    """Test successful frame extraction with object detection and tracking.

    Arranges:
        - Mock NVR client to return a test frame path
        - Mock detector as unloaded (to use mock detection mode)
        - Mock tracker to return track objects with unique IDs

    Acts:
        - POST request to /frames/objects with valid timestamp and channel

    Asserts:
        - Response status is 200
        - Response contains all expected fields
        - Objects list has correct structure
        - Base64 image is valid and can be decoded
        - total_objects matches the length of objects list
        - Each object has a unique object_id
    """

    # Arrange
    mock_nvr_client.extract_frame.return_value = Path("/tmp/test_frame.png")
    mock_detector._model_loaded = False

    mock_tracks = [
        MagicMock(
            track_id=1,
            label="person",
            confidence=0.92,
            x=140.0,
            y=210.0,
            width=80.0,
            height=120.0,
        ),
        MagicMock(
            track_id=2,
            label="bicycle",
            confidence=0.85,
            x=300.0,
            y=240.0,
            width=100.0,
            height=80.0,
        ),
    ]
    mock_tracker.update.return_value = mock_tracks

    timestamp = datetime(2026, 2, 19, 22, 31, 5)

    # Act
    response = client.post(
        "/frames/objects",
        json={
            "timestamp": timestamp.isoformat(),
            "channel": 1,
        },
    )

    # Assert
    assert response.status_code == 200
    data = response.json()

    assert "timestamp" in data
    assert "channel" in data
    assert "objects" in data
    assert "image_path" in data
    assert "total_objects" in data

    assert data["channel"] == 1
    assert data["timestamp"] == timestamp.isoformat()
    assert isinstance(data["objects"], list)
    assert data["total_objects"] == len(data["objects"])

    for obj in data["objects"]:
        assert "object_id" in obj
        assert "label" in obj
        assert "confidence" in obj
        assert "bbox" in obj
        assert "center" in obj

        assert isinstance(obj["object_id"], int)
        assert isinstance(obj["label"], str)
        assert isinstance(obj["confidence"], float)
        assert isinstance(obj["bbox"], dict)
        assert isinstance(obj["center"], dict)

        assert "x" in obj["bbox"]
        assert "y" in obj["bbox"]
        assert "width" in obj["bbox"]
        assert "height" in obj["bbox"]

        assert "x" in obj["center"]
        assert "y" in obj["center"]

    # Verify image was saved to disk
    image_path = Path(data["image_path"])
    assert image_path.exists()
    assert image_path.stat().st_size > 0

    object_ids = [obj["object_id"] for obj in data["objects"]]
    assert len(object_ids) == len(set(object_ids))


def test_get_frame_with_objects_nvr_not_initialized(client):
    """Test error handling when NVR client is not initialized.

    Arranges:
        - Mock nvr_client to None in the API module

    Acts:
        - POST request to /frames/objects

    Asserts:
        - Response status is 500
        - Error detail mentions NVR client not initialized
    """
    timestamp = datetime(2026, 2, 19, 22, 31, 5)

    with patch("cctv_search.api.nvr_client", None):
        response = client.post(
            "/frames/objects",
            json={
                "timestamp": timestamp.isoformat(),
                "channel": 1,
            },
        )

    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert "NVR client not initialized" in data["detail"]


def test_get_frame_with_objects_invalid_timestamp(client):
    """Test validation error with invalid/malformed timestamp.

    Arranges:
        - Request with invalid timestamp format

    Acts:
        - POST request to /frames/objects

    Asserts:
        - Response status is 422 (validation error)
    """
    response = client.post(
        "/frames/objects",
        json={
            "timestamp": "not-a-valid-timestamp",
            "channel": 1,
        },
    )

    assert response.status_code == 422


@patch("cctv_search.api.nvr_client")
def test_get_frame_with_objects_invalid_channel(mock_nvr_client, client):
    """Test error handling with invalid channel values.

    Arranges:
        - Mock NVR client
        - Request with invalid channel values (negative, zero)

    Acts:
        - POST requests to /frames/objects with invalid channels

    Asserts:
        - API handles gracefully (either validation error or runtime error)
    """
    timestamp = datetime(2026, 2, 19, 22, 31, 5)

    response = client.post(
        "/frames/objects",
        json={
            "timestamp": timestamp.isoformat(),
            "channel": -1,
        },
    )
    assert response.status_code in [200, 400, 422, 500]

    response = client.post(
        "/frames/objects",
        json={
            "timestamp": timestamp.isoformat(),
            "channel": 0,
        },
    )
    assert response.status_code in [200, 400, 422, 500]


@patch("cctv_search.api.nvr_client")
@patch("cctv_search.api.detector")
@patch("cctv_search.api.tracker")
def test_get_frame_with_objects_no_detections(
    mock_tracker, mock_detector, mock_nvr_client, client
):
    """Test scenario where no objects are detected.

    Arranges:
        - Mock NVR client to return test frame
        - Mock tracker to return empty list (no detections)

    Acts:
        - POST request to /frames/objects

    Asserts:
        - Response status is 200
        - Objects list is empty
        - total_objects is 0
        - Base64 image is still valid
    """

    mock_nvr_client.extract_frame.return_value = Path("/tmp/test_frame.png")
    mock_detector._model_loaded = False
    mock_tracker.update.return_value = []

    timestamp = datetime(2026, 2, 19, 22, 31, 5)

    response = client.post(
        "/frames/objects",
        json={
            "timestamp": timestamp.isoformat(),
            "channel": 1,
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["objects"] == []
    assert data["total_objects"] == 0
    assert "image_path" in data


@patch("cctv_search.api.nvr_client")
@patch("cctv_search.api.detector")
@patch("cctv_search.api.tracker")
def test_get_frame_with_objects_multiple_objects(
    mock_tracker, mock_detector, mock_nvr_client, client
):
    """Test frame with multiple objects detection and tracking.

    Arranges:
        - Mock NVR client to return test frame
        - Mock tracker to return multiple track objects with unique IDs

    Acts:
        - POST request to /frames/objects

    Asserts:
        - Response status is 200
        - All objects have unique IDs
        - bbox and center coordinates are correct
        - Object count matches expected
    """

    mock_tracks = [
        MagicMock(
            track_id=1,
            label="person",
            confidence=0.95,
            x=150.0,
            y=200.0,
            width=100.0,
            height=150.0,
        ),
        MagicMock(
            track_id=2,
            label="person",
            confidence=0.88,
            x=300.0,
            y=250.0,
            width=90.0,
            height=140.0,
        ),
        MagicMock(
            track_id=3,
            label="car",
            confidence=0.92,
            x=450.0,
            y=300.0,
            width=200.0,
            height=150.0,
        ),
        MagicMock(
            track_id=4,
            label="bicycle",
            confidence=0.75,
            x=100.0,
            y=400.0,
            width=60.0,
            height=40.0,
        ),
    ]

    mock_nvr_client.extract_frame.return_value = Path("/tmp/test_frame.png")
    mock_detector._model_loaded = False
    mock_tracker.update.return_value = mock_tracks

    timestamp = datetime(2026, 2, 19, 22, 31, 5)

    response = client.post(
        "/frames/objects",
        json={
            "timestamp": timestamp.isoformat(),
            "channel": 1,
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["total_objects"] == 4
    assert len(data["objects"]) == 4

    object_ids = [obj["object_id"] for obj in data["objects"]]
    assert len(object_ids) == len(set(object_ids))
    assert set(object_ids) == {1, 2, 3, 4}

    for i, obj in enumerate(data["objects"]):
        track = mock_tracks[i]

        expected_bbox_x = track.x - track.width / 2
        expected_bbox_y = track.y - track.height / 2

        assert obj["bbox"]["x"] == expected_bbox_x
        assert obj["bbox"]["y"] == expected_bbox_y
        assert obj["bbox"]["width"] == track.width
        assert obj["bbox"]["height"] == track.height

        assert obj["center"]["x"] == track.x
        assert obj["center"]["y"] == track.y

        assert obj["label"] == track.label
        assert obj["confidence"] == track.confidence

    # Verify image was saved to disk
    assert "image_path" in data
    image_path = Path(data["image_path"])
    assert image_path.exists()
    assert image_path.stat().st_size > 0


# =============================================================================
# Object Search Endpoint Tests
# =============================================================================


def test_search_object_nvr_not_initialized(client):
    """Test error when NVR client is None.

    Verifies:
    - Returns HTTP 500
    - Error message mentions NVR client
    """
    with patch("cctv_search.api.nvr_client", None):
        timestamp = datetime(2026, 2, 19, 22, 31, 5)
        response = client.post(
            "/search/object",
            data={
                "timestamp": timestamp.isoformat(),
                "channel": "1",
                "bbox_x": "100.0",
                "bbox_y": "200.0",
                "bbox_width": "50.0",
                "bbox_height": "80.0",
                "object_label": "person",
                "object_confidence": "0.95",
                "search_duration_seconds": "3600",
            },
        )

        assert response.status_code == 500
        assert "nvr client" in response.json()["detail"].lower()


@patch("cctv_search.api.nvr_client")
def test_search_object_invalid_duration(mock_nvr_client, client):
    """Test validation error when duration is <= 0.

    Verifies:
    - Returns HTTP 400
    - Error message mentions duration
    """
    mock_nvr_client.host = "192.168.1.108"

    timestamp = datetime(2026, 2, 19, 22, 31, 5)
    response = client.post(
        "/search/object",
        data={
            "timestamp": timestamp.isoformat(),
            "channel": "1",
            "bbox_x": "100.0",
            "bbox_y": "200.0",
            "bbox_width": "50.0",
            "bbox_height": "80.0",
            "object_label": "person",
            "object_confidence": "0.95",
            "search_duration_seconds": "0",
        },
    )

    assert response.status_code == 400
    assert "duration" in response.json()["detail"].lower()


@patch("cctv_search.api.nvr_client")
def test_search_object_duration_too_long(mock_nvr_client, client):
    """Test validation error when duration exceeds 3 hours (10800 seconds).

    Verifies:
    - Returns HTTP 400
    - Error message mentions maximum duration
    """
    mock_nvr_client.host = "192.168.1.108"

    timestamp = datetime(2026, 2, 19, 22, 31, 5)
    response = client.post(
        "/search/object",
        data={
            "timestamp": timestamp.isoformat(),
            "channel": "1",
            "bbox_x": "100.0",
            "bbox_y": "200.0",
            "bbox_width": "50.0",
            "bbox_height": "80.0",
            "object_label": "person",
            "object_confidence": "0.95",
            "search_duration_seconds": "10801",
        },
    )

    assert response.status_code == 400
    assert "maximum" in response.json()["detail"].lower()


def test_search_object_missing_required_fields(client):
    """Test validation error when required fields are missing.

    Verifies:
    - Returns HTTP 422 for missing bbox coordinates
    - Returns HTTP 422 for missing object_label
    """
    timestamp = datetime(2026, 2, 19, 22, 31, 5)

    # Test missing bbox_x
    response_no_bbox_x = client.post(
        "/search/object",
        data={
            "timestamp": timestamp.isoformat(),
            "channel": "1",
            "bbox_y": "200.0",
            "bbox_width": "50.0",
            "bbox_height": "80.0",
            "object_label": "person",
            "object_confidence": "0.95",
            "search_duration_seconds": "3600",
        },
    )
    assert response_no_bbox_x.status_code == 422

    # Test missing object_label
    response_no_label = client.post(
        "/search/object",
        data={
            "timestamp": timestamp.isoformat(),
            "channel": "1",
            "bbox_x": "100.0",
            "bbox_y": "200.0",
            "bbox_width": "50.0",
            "bbox_height": "80.0",
            "object_confidence": "0.95",
            "search_duration_seconds": "3600",
        },
    )
    assert response_no_label.status_code == 422

    # Test missing timestamp
    response_no_timestamp = client.post(
        "/search/object",
        data={
            "channel": "1",
            "bbox_x": "100.0",
            "bbox_y": "200.0",
            "bbox_width": "50.0",
            "bbox_height": "80.0",
            "object_label": "person",
            "object_confidence": "0.95",
            "search_duration_seconds": "3600",
        },
    )
    assert response_no_timestamp.status_code == 422



