"""Integration tests for complete object search workflow.

This module tests the full object search workflow using all three
new endpoints together: /frames/objects, /search/object, and /video/clip.
"""

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
    mock_client.extract_frame.return_value = Path("/tmp/test_frame.png")
    return mock_client


class TestObjectSearchWorkflow:
    """Tests for the complete object search workflow."""

    @patch("cctv_search.api.nvr_client")
    @patch("cctv_search.api.detector")
    @patch("cctv_search.api.tracker")
    def test_workflow_object_not_found(
        self,
        mock_tracker,
        mock_detector,
        mock_nvr_client,
        client,
        temp_clips_dir,
    ):
        """Test workflow when object search fails due to frame extraction errors.

        With the 2-phase search algorithm, frame extraction failures during
        coarse sampling cause the search to abort with an error status.
        """
        mock_nvr_client.extract_frame.return_value = Path("/tmp/test_frame.png")
        mock_detector._model_loaded = False

        mock_tracks = [
            MagicMock(
                track_id=42,
                label="person",
                confidence=0.90,
                x=100.0,
                y=200.0,
                width=80.0,
                height=120.0,
            ),
        ]
        mock_tracker.update.return_value = mock_tracks

        timestamp = datetime(2026, 2, 19, 16, 30, 0)

        # Step 1: Get frame with objects
        response1 = client.post(
            "/frames/objects",
            json={
                "timestamp": timestamp.isoformat(),
                "channel": 1,
            },
        )

        assert response1.status_code == 200
        data1 = response1.json()
        obj = data1["objects"][0]

        # Step 2: Search for object - frame extraction fails
        response2 = client.post(
            "/search/object",
            json={
                "camera_id": "1",
                "start_timestamp": timestamp.isoformat(),
                "object_id": obj["object_id"],
                "search_duration_seconds": 3600,
                "object_label": obj["label"],
                "object_bbox": obj["bbox"],
                "object_confidence": obj["confidence"],
            },
        )

        # With 2-phase algorithm, frame extraction errors cause error response
        # The API returns 500 when the search encounters frame extraction failures
        assert response2.status_code == 500
        data2 = response2.json()
        assert "detail" in data2

        # In the new 2-phase algorithm, frame extraction failures during coarse
        # sampling cause the search to abort with an error rather than continuing

    @patch("cctv_search.api.nvr_client")
    @patch("cctv_search.api.detector")
    @patch("cctv_search.api.tracker")
    def test_workflow_error_handling(
        self,
        mock_tracker,
        mock_detector,
        mock_nvr_client,
        client,
    ):
        """Test error handling throughout the workflow.

        Verifies proper error handling when:
        - NVR is not initialized
        - Invalid parameters are provided
        - Search duration is invalid
        """
        # Test with NVR not initialized
        with patch("cctv_search.api.nvr_client", None):
            timestamp = datetime(2026, 2, 19, 10, 0, 0)

            response = client.post(
                "/frames/objects",
                json={
                    "timestamp": timestamp.isoformat(),
                    "channel": 1,
                },
            )
            assert response.status_code == 500
            assert "nvr client" in response.json()["detail"].lower()

        # Test invalid timestamp
        mock_nvr_client.extract_frame.return_value = Path("/tmp/test_frame.png")

        response = client.post(
            "/frames/objects",
            json={
                "timestamp": "invalid-timestamp",
                "channel": 1,
            },
        )
        assert response.status_code == 422

        # Test invalid search duration (zero)
        response = client.post(
            "/search/object",
            json={
                "camera_id": "1",
                "start_timestamp": datetime.now().isoformat(),
                "object_id": 1,
                "search_duration_seconds": 0,
                "object_label": "person",
                "object_bbox": {"x": 100, "y": 100, "width": 50, "height": 50},
                "object_confidence": 0.9,
            },
        )
        assert response.status_code == 400
        assert "duration" in response.json()["detail"].lower()

        # Test search duration too long
        response = client.post(
            "/search/object",
            json={
                "camera_id": "1",
                "start_timestamp": datetime.now().isoformat(),
                "object_id": 1,
                "search_duration_seconds": 20000,  # > 3 hours
                "object_label": "person",
                "object_bbox": {"x": 100, "y": 100, "width": 50, "height": 50},
                "object_confidence": 0.9,
            },
        )
        assert response.status_code == 400
        assert "maximum" in response.json()["detail"].lower()

        # Test invalid clip duration (zero)
        response = client.post(
            "/video/clip",
            json={
                "camera_id": "1",
                "start_timestamp": datetime.now().isoformat(),
                "duration_seconds": 0,
            },
        )
        assert response.status_code == 400


