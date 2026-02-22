"""Tests for NVR module."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cctv_search.nvr import DahuaNVRClient, FrameRequest


@pytest.fixture
def nvr_client():
    """Create Dahua NVR client fixture."""
    return DahuaNVRClient(
        host="192.168.1.100",
        port=554,
        username="admin",
        password="password123",
    )


def test_nvr_client_init_with_params():
    """Test NVR client initializes with explicit parameters."""
    client = DahuaNVRClient(
        host="192.168.1.50",
        port=554,
        username="admin",
        password="secret",
        rtsp_transport="udp",
    )
    assert client.host == "192.168.1.50"
    assert client.port == 554
    assert client.username == "admin"
    assert client.password == "secret"
    assert client.rtsp_transport == "udp"


@patch.dict(
    "os.environ",
    {
        "NVR_HOST": "192.168.1.200",
        "NVR_PORT": "554",
        "NVR_USERNAME": "envuser",
        "NVR_PASSWORD": "envpass",
    },
)
def test_nvr_client_init_with_env_vars():
    """Test NVR client initializes from environment variables."""
    client = DahuaNVRClient()
    assert client.host == "192.168.1.200"
    assert client.port == 554
    assert client.username == "envuser"
    assert client.password == "envpass"


def test_format_timestamp(nvr_client):
    """Test timestamp formatting for RTSP URL."""
    dt = datetime(2026, 2, 19, 22, 31, 5)
    result = nvr_client._format_timestamp(dt)
    assert result == "2026_02_19_22_31_05"


def test_build_rtsp_url(nvr_client):
    """Test RTSP URL construction."""
    start_time = datetime(2026, 2, 19, 22, 31, 5)
    end_time = datetime(2026, 2, 19, 22, 31, 6)

    url = nvr_client._build_rtsp_url(1, start_time, end_time)

    # Per Dahua spec, credentials are NOT in the URL
    expected = (
        "rtsp://192.168.1.100:554"
        "/cam/playback?channel=1&starttime=2026_02_19_22_31_05&endtime=2026_02_19_22_31_06"
    )
    assert url == expected


@patch("subprocess.run")
def test_extract_frame_success(mock_run, nvr_client, tmp_path):
    """Test successful frame extraction."""
    mock_run.return_value = MagicMock(returncode=0)

    timestamp = datetime(2026, 2, 19, 22, 31, 5)
    output_path = tmp_path / "test_frame.png"

    result = nvr_client.extract_frame(timestamp, channel=1, output_path=output_path)

    assert result == output_path
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]
    assert call_args[0] == "ffmpeg"
    assert "-rtsp_transport" in call_args
    assert "tcp" in call_args


@patch("subprocess.run")
def test_extract_frame_creates_directory(mock_run, nvr_client, tmp_path):
    """Test that extract_frame creates output directory if needed."""
    mock_run.return_value = MagicMock(returncode=0)

    timestamp = datetime(2026, 2, 19, 22, 31, 5)
    output_path = tmp_path / "subdir" / "frame.png"

    nvr_client.extract_frame(timestamp, output_path=output_path)

    assert output_path.parent.exists()


@patch("subprocess.run")
def test_extract_frame_failure_raises_runtime_error(mock_run, nvr_client, tmp_path):
    """Test that ffmpeg failure raises RuntimeError."""
    from subprocess import CalledProcessError

    mock_run.side_effect = CalledProcessError(1, "ffmpeg", stderr="Connection failed")

    timestamp = datetime(2026, 2, 19, 22, 31, 5)
    output_path = tmp_path / "frame.png"

    with pytest.raises(RuntimeError, match="Failed to extract frame"):
        nvr_client.extract_frame(timestamp, output_path=output_path)


def test_frame_request_dataclass():
    """Test FrameRequest dataclass."""
    timestamp = datetime(2026, 2, 19, 22, 31, 5)
    request = FrameRequest(
        timestamp=timestamp,
        channel=2,
        output_path=Path("/tmp/frame.png"),
    )

    assert request.timestamp == timestamp
    assert request.channel == 2
    assert request.output_path == Path("/tmp/frame.png")


def test_frame_request_defaults():
    """Test FrameRequest default values."""
    timestamp = datetime(2026, 2, 19, 22, 31, 5)
    request = FrameRequest(timestamp=timestamp)

    assert request.channel == 1
    assert request.output_path is None
