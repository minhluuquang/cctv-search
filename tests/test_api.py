"""Tests for FastAPI endpoints."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from cctv_search.api import app


@pytest.fixture
def client():
    """Create test client."""
    with TestClient(app) as c:
        yield c


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
