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
    @patch("cctv_search.api.MOCK_SEARCH_MODE", True)
    def test_object_search_workflow_integration(
        self,
        mock_tracker,
        mock_detector,
        mock_nvr_client,
        client,
        temp_clips_dir,
    ):
        """Test the complete object search workflow.

        This test demonstrates the full use case:
        1. Extract frame and detect objects
        2. Select an object by ID
        3. Search backward for when object first appeared
        4. Generate video clip from first appearance
        """
        # Setup mocks
        from PIL import Image
        
        # Create temporary test image files
        test_frame_path = Path("/tmp/test_frame.png")
        test_search_frame_path = Path("/tmp/test_search_frame.png")
        
        # Create test images
        for img_path in [test_frame_path, test_search_frame_path]:
            img = Image.new("RGB", (640, 480), color=(100, 100, 100))
            img.save(img_path)
        
        # Make extract_frame return appropriate path based on timestamp
        def mock_extract_frame(timestamp, channel, output_path):
            if "search" in str(output_path):
                return test_search_frame_path
            return test_frame_path
        
        mock_nvr_client.extract_frame.side_effect = mock_extract_frame
        mock_detector._model_loaded = False

        # Create mock tracks for Step 1
        mock_tracks = [
            MagicMock(
                track_id=1,
                label="person",
                confidence=0.92,
                x=150.0,
                y=200.0,
                width=100.0,
                height=150.0,
            ),
            MagicMock(
                track_id=2,
                label="bicycle",
                confidence=0.85,
                x=350.0,
                y=250.0,
                width=120.0,
                height=100.0,
            ),
        ]
        mock_tracker.update.return_value = mock_tracks

        # Step 1: Get frame with objects at timestamp
        timestamp = datetime(2026, 2, 19, 10, 30, 0)

        response1 = client.post(
            "/frames/objects",
            json={
                "timestamp": timestamp.isoformat(),
                "channel": 1,
            },
        )

        assert response1.status_code == 200
        data1 = response1.json()
        assert len(data1["objects"]) > 0

        # Select first object
        selected_object = data1["objects"][0]
        object_id = selected_object["object_id"]
        object_label = selected_object["label"]
        object_bbox = selected_object["bbox"]
        object_confidence = selected_object["confidence"]

        # Verify object structure
        assert "object_id" in selected_object
        assert "label" in selected_object
        assert "confidence" in selected_object
        assert "bbox" in selected_object
        assert "center" in selected_object

        # Verify image was saved to disk
        assert "image_path" in data1
        image_path = Path(data1["image_path"])
        assert image_path.exists()
        assert image_path.stat().st_size > 0

        # Step 2: Search for this object backward in time
        with patch("cctv_search.api.random") as mock_random:
            # Force success case (object found)
            mock_random.random.return_value = 0.5  # < 0.75 triggers success
            mock_random.randint.return_value = 60  # Appeared 60s before
            mock_random.uniform.return_value = 0.92  # Confidence multiplier

            response2 = client.post(
                "/search/object",
                json={
                    "camera_id": "1",
                    "start_timestamp": timestamp.isoformat(),
                    "object_id": object_id,
                    "search_duration_seconds": 3600,
                    "object_label": object_label,
                    "object_bbox": object_bbox,
                    "object_confidence": object_confidence,
                },
            )

            assert response2.status_code == 200
            data2 = response2.json()
            assert data2["status"] == "success"
            assert data2["result"]["found"] is True

            first_seen = datetime.fromisoformat(data2["result"]["first_seen_timestamp"])

            # Verify search result structure
            assert data2["result"]["search_iterations"] > 0
            assert data2["result"]["track_duration_seconds"] is not None
            assert data2["result"]["confidence"] is not None
            assert 0 < data2["result"]["confidence"] < 1

            # Verify first_seen is before the search timestamp
            assert first_seen < timestamp

            # Verify clip was automatically generated and saved
            assert "clip_path" in data2["result"]
            assert data2["result"]["clip_path"] is not None
            clip_path = Path(data2["result"]["clip_path"])
            assert clip_path.exists()
            assert clip_path.stat().st_size > 0

            # Verify annotated image was automatically generated and saved
            assert "image_path" in data2["result"]
            assert data2["result"]["image_path"] is not None
            image_path = Path(data2["result"]["image_path"])
            assert image_path.exists()
            assert image_path.stat().st_size > 0

            # Cleanup
            clip_path.unlink()
            image_path.unlink()

    @patch("cctv_search.api.nvr_client")
    @patch("cctv_search.api.detector")
    @patch("cctv_search.api.tracker")
    def test_workflow_with_mock_nvr(
        self,
        mock_tracker,
        mock_detector,
        mock_nvr_client,
        client,
        temp_clips_dir,
    ):
        """Test the full workflow with mock NVR (no real NVR configured).

        Verifies all endpoints work correctly when NVR is not configured,
        using mock data generation for all operations.
        """
        # Setup: NVR with no host (mock mode)
        mock_nvr_client.host = ""
        mock_nvr_client.extract_frame.return_value = Path("/tmp/test_frame.png")
        mock_detector._model_loaded = False

        # Mock tracker to return objects
        mock_tracks = [
            MagicMock(
                track_id=1,
                label="car",
                confidence=0.88,
                x=200.0,
                y=150.0,
                width=180.0,
                height=120.0,
            ),
        ]
        mock_tracker.update.return_value = mock_tracks

        # Step 1: Extract frame with objects (mock mode)
        timestamp = datetime(2026, 2, 19, 14, 0, 0)

        response1 = client.post(
            "/frames/objects",
            json={
                "timestamp": timestamp.isoformat(),
                "channel": 2,
            },
        )

        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["channel"] == 2
        assert len(data1["objects"]) == 1

        obj = data1["objects"][0]
        assert obj["label"] == "car"
        assert obj["object_id"] == 1

        # Step 2: Search for object (mock mode)
        with patch("cctv_search.api.random") as mock_random:
            mock_random.random.return_value = 0.3  # Force success
            mock_random.randint.return_value = 120  # 2 minutes before

            response2 = client.post(
                "/search/object",
                json={
                    "camera_id": "2",
                    "start_timestamp": timestamp.isoformat(),
                    "object_id": obj["object_id"],
                    "search_duration_seconds": 3600,
                    "object_label": obj["label"],
                    "object_bbox": obj["bbox"],
                    "object_confidence": obj["confidence"],
                },
            )

            assert response2.status_code == 200
            data2 = response2.json()
            assert data2["status"] == "success"
            assert data2["result"]["found"] is True

            first_seen = datetime.fromisoformat(data2["result"]["first_seen_timestamp"])

        # Step 3: Generate clip (mock mode)
        response3 = client.post(
            "/video/clip",
            json={
                "camera_id": "2",
                "start_timestamp": first_seen.isoformat(),
                "duration_seconds": 15,
            },
        )

        assert response3.status_code == 200
        data3 = response3.json()
        assert data3["duration_seconds"] == 15
        assert data3["file_size_bytes"] > 0

        clip_path = Path(data3["clip_path"])
        assert clip_path.exists()
        clip_path.unlink()

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
        """Test workflow when object search doesn't find the object.

        Verifies graceful handling when object is not found,
        preventing clip generation in that case.
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

        # Step 2: Search for object - NOT FOUND case
        with patch("cctv_search.api.random") as mock_random:
            mock_random.random.return_value = 0.9  # > 0.75 triggers not found
            mock_random.randint.return_value = 50

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

            assert response2.status_code == 200
            data2 = response2.json()
            assert data2["status"] == "not_found"
            assert data2["result"]["found"] is False
            assert data2["result"]["first_seen_timestamp"] is None
            assert data2["result"]["last_seen_timestamp"] is None
            assert "not found" in data2["result"]["message"].lower()

        # In a real scenario, the application would handle the "not found"
        # case by not generating a clip. Here we verify the search endpoint
        # returns proper information to make that decision.

    @patch("cctv_search.api.nvr_client")
    @patch("cctv_search.api.detector")
    @patch("cctv_search.api.tracker")
    def test_workflow_multiple_objects(
        self,
        mock_tracker,
        mock_detector,
        mock_nvr_client,
        client,
        temp_clips_dir,
    ):
        """Test with frame containing multiple objects.

        Verifies each object can be searched independently and
        that object IDs are unique and stable.
        """
        mock_nvr_client.extract_frame.return_value = Path("/tmp/test_frame.png")
        mock_detector._model_loaded = False

        # Create multiple objects
        mock_tracks = [
            MagicMock(
                track_id=1,
                label="person",
                confidence=0.92,
                x=100.0,
                y=150.0,
                width=80.0,
                height=120.0,
            ),
            MagicMock(
                track_id=2,
                label="person",
                confidence=0.88,
                x=300.0,
                y=200.0,
                width=90.0,
                height=140.0,
            ),
            MagicMock(
                track_id=3,
                label="car",
                confidence=0.95,
                x=450.0,
                y=250.0,
                width=200.0,
                height=150.0,
            ),
            MagicMock(
                track_id=4,
                label="bicycle",
                confidence=0.82,
                x=150.0,
                y=350.0,
                width=70.0,
                height=50.0,
            ),
        ]
        mock_tracker.update.return_value = mock_tracks

        timestamp = datetime(2026, 2, 19, 12, 0, 0)

        # Step 1: Get frame with all objects
        response1 = client.post(
            "/frames/objects",
            json={
                "timestamp": timestamp.isoformat(),
                "channel": 1,
            },
        )

        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["total_objects"] == 4
        assert len(data1["objects"]) == 4

        # Verify unique IDs
        object_ids = [obj["object_id"] for obj in data1["objects"]]
        assert len(object_ids) == len(set(object_ids))
        assert set(object_ids) == {1, 2, 3, 4}

        # Step 2: Search for each object independently
        mock_nvr_client.host = ""

        for obj in data1["objects"]:
            with patch("cctv_search.api.random") as mock_random:
                # Different time offset for each object
                offset = obj["object_id"] * 30  # 30, 60, 90, 120 seconds

                mock_random.random.return_value = 0.5  # Force success
                mock_random.randint.return_value = offset
                mock_random.uniform.return_value = 0.95

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

                assert response2.status_code == 200
                data2 = response2.json()
                assert data2["status"] == "success"
                assert data2["result"]["found"] is True

                first_seen = datetime.fromisoformat(
                    data2["result"]["first_seen_timestamp"]
                )

                # Generate a clip for this object
                response3 = client.post(
                    "/video/clip",
                    json={
                        "camera_id": "1",
                        "start_timestamp": first_seen.isoformat(),
                        "duration_seconds": 15,
                        "object_id": obj["object_id"],
                    },
                )

                assert response3.status_code == 200
                data3 = response3.json()
                assert data3["duration_seconds"] == 15

                clip_path = Path(data3["clip_path"])
                assert clip_path.exists()
                clip_path.unlink()

    @patch("cctv_search.api.nvr_client")
    @patch("cctv_search.api.detector")
    @patch("cctv_search.api.tracker")
    def test_workflow_custom_duration(
        self,
        mock_tracker,
        mock_detector,
        mock_nvr_client,
        client,
        temp_clips_dir,
    ):
        """Test workflow with custom clip duration.

        Verifies that different clip durations work correctly
        throughout the workflow.
        """
        mock_nvr_client.extract_frame.return_value = Path("/tmp/test_frame.png")
        mock_nvr_client.host = ""
        mock_detector._model_loaded = False

        mock_tracks = [
            MagicMock(
                track_id=1,
                label="person",
                confidence=0.92,
                x=200.0,
                y=200.0,
                width=100.0,
                height=150.0,
            ),
        ]
        mock_tracker.update.return_value = mock_tracks

        timestamp = datetime(2026, 2, 19, 18, 0, 0)

        # Get frame
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

        # Search
        with patch("cctv_search.api.random") as mock_random:
            mock_random.random.return_value = 0.5
            mock_random.randint.return_value = 60

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
            assert response2.status_code == 200
            data2 = response2.json()
            first_seen = datetime.fromisoformat(data2["result"]["first_seen_timestamp"])

        # Test different durations
        for duration in [5, 30, 60]:
            response3 = client.post(
                "/video/clip",
                json={
                    "camera_id": "1",
                    "start_timestamp": first_seen.isoformat(),
                    "duration_seconds": duration,
                },
            )

            assert response3.status_code == 200
            data3 = response3.json()
            assert data3["duration_seconds"] == duration

            expected_end = first_seen + timedelta(seconds=duration)
            assert data3["end_timestamp"] == expected_end.isoformat()

            clip_path = Path(data3["clip_path"])
            assert clip_path.exists()
            clip_path.unlink()

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


class TestWorkflowPerformance:
    """Tests to verify workflow performance characteristics."""

    @patch("cctv_search.api.nvr_client")
    @patch("cctv_search.api.detector")
    @patch("cctv_search.api.tracker")
    def test_workflow_completes_quickly(
        self,
        mock_tracker,
        mock_detector,
        mock_nvr_client,
        client,
        temp_clips_dir,
    ):
        """Verify the full workflow completes quickly (under 5 seconds).

        This is a smoke test to ensure the workflow is efficient
        and doesn't have unnecessary delays.
        """
        import time

        mock_nvr_client.extract_frame.return_value = Path("/tmp/test_frame.png")
        mock_nvr_client.host = ""
        mock_detector._model_loaded = False

        mock_tracks = [
            MagicMock(
                track_id=1,
                label="person",
                confidence=0.90,
                x=200.0,
                y=200.0,
                width=100.0,
                height=150.0,
            ),
        ]
        mock_tracker.update.return_value = mock_tracks

        timestamp = datetime(2026, 2, 19, 10, 0, 0)

        start_time = time.time()

        # Step 1: Get frame
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

        # Step 2: Search
        with patch("cctv_search.api.random") as mock_random:
            mock_random.random.return_value = 0.5
            mock_random.randint.return_value = 60

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
            assert response2.status_code == 200
            data2 = response2.json()
            first_seen = datetime.fromisoformat(data2["result"]["first_seen_timestamp"])

        # Step 3: Generate clip
        response3 = client.post(
            "/video/clip",
            json={
                "camera_id": "1",
                "start_timestamp": first_seen.isoformat(),
                "duration_seconds": 15,
            },
        )
        assert response3.status_code == 200

        elapsed_time = time.time() - start_time

        # Cleanup
        clip_path = Path(response3.json()["clip_path"])
        if clip_path.exists():
            clip_path.unlink()

        # Verify workflow completes in under 5 seconds
        # (adjust threshold as needed based on mock clip generation time)
        assert elapsed_time < 5.0, (
            f"Workflow took {elapsed_time:.2f}s, expected under 5s"
        )
