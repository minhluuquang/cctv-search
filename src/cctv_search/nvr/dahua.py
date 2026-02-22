"""Dahua NVR client using RTSP playback endpoint."""

from __future__ import annotations

import os
import subprocess
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class FrameRequest:
    """Request to extract a frame at a specific timestamp."""

    timestamp: datetime
    channel: int = 1
    output_path: Path | None = None


class DahuaNVRClient:
    """Dahua NVR client using RTSP playback endpoint for frame extraction."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        username: str | None = None,
        password: str | None = None,
        rtsp_transport: str = "tcp",
    ) -> None:
        """Initialize Dahua NVR client.

        Args:
            host: NVR IP address (defaults to NVR_HOST env var)
            port: NVR RTSP port (defaults to NVR_PORT env var or 554)
            username: NVR username (defaults to NVR_USERNAME env var)
            password: NVR password (defaults to NVR_PASSWORD env var)
            rtsp_transport: RTSP transport protocol (tcp/udp)
        """
        self.host = host or os.getenv("NVR_HOST", "")
        self.port = port or int(os.getenv("NVR_PORT", "554"))
        self.username = username or os.getenv("NVR_USERNAME", "")
        self.password = password or os.getenv("NVR_PASSWORD", "")
        self.rtsp_transport = rtsp_transport

    def _format_timestamp(self, dt: datetime) -> str:
        """Format datetime for Dahua RTSP endpoint (YYYY_MM_DD_HH_MM_SS)."""
        return dt.strftime("%Y_%m_%d_%H_%M_%S")

    def _build_rtsp_url(
        self, channel: int, start_time: datetime, end_time: datetime
    ) -> str:
        """Build RTSP playback URL.

        Per Dahua API spec, credentials are NOT in the URL:
        rtsp://<server>:[port]/cam/playback?channel=<channel>&starttime=<YYYY_MM_DD_HH_MM_SS>&endtime=<YYYY_MM_DD_HH_MM_SS>
        """
        start_str = self._format_timestamp(start_time)
        end_str = self._format_timestamp(end_time)
        return (
            f"rtsp://{self.host}:{self.port}"
            f"/cam/playback?channel={channel}&starttime={start_str}&endtime={end_str}"
        )

    def _build_rtsp_url_with_auth(
        self, channel: int, start_time: datetime, end_time: datetime
    ) -> str:
        """Build RTSP playback URL with authentication.

        Returns URL with credentials embedded for direct playback:
        rtsp://<user>:<pass>@<server>:[port]/cam/playback?channel=<channel>&starttime=<YYYY_MM_DD_HH_MM_SS>&endtime=<YYYY_MM_DD_HH_MM_SS>
        """
        from urllib.parse import quote
        encoded_username = quote(self.username, safe='')
        encoded_password = quote(self.password, safe='')
        start_str = self._format_timestamp(start_time)
        end_str = self._format_timestamp(end_time)
        return (
            f"rtsp://{encoded_username}:{encoded_password}@{self.host}:{self.port}"
            f"/cam/playback?channel={channel}&starttime={start_str}&endtime={end_str}"
        )

    def extract_frame(
        self,
        timestamp: datetime,
        channel: int = 1,
        output_path: str | Path = "frame.png",
    ) -> Path:
        """Extract a single frame at the exact timestamp using ffmpeg.

        Args:
            timestamp: Exact timestamp to extract frame from
            channel: Camera channel number (default: 1)
            output_path: Path to save the extracted frame

        Returns:
            Path to the extracted frame file

        Raises:
            RuntimeError: If ffmpeg fails to extract the frame
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build RTSP URL with 1-second window around the timestamp
        start_time = timestamp
        end_time = timestamp.replace(second=timestamp.second + 1)
        rtsp_url = self._build_rtsp_url(channel, start_time, end_time)

        # Extract frame using ffmpeg
        # Credentials are passed via URL according to ffmpeg RTSP implementation
        # URL-encode username and password to handle special characters like @
        encoded_username = urllib.parse.quote(self.username, safe="")
        encoded_password = urllib.parse.quote(self.password, safe="")
        auth_url = rtsp_url.replace(
            f"rtsp://{self.host}:{self.port}",
            f"rtsp://{encoded_username}:{encoded_password}@{self.host}:{self.port}",
        )

        cmd = [
            "ffmpeg",
            "-rtsp_transport",
            self.rtsp_transport,
            "-i",
            auth_url,
            "-frames:v",
            "1",
            "-y",  # Overwrite output file if exists
            str(output_path),
        ]

        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to extract frame: {e.stderr}") from e

        return output_path

    def extract_clip(
        self,
        start_time: datetime,
        end_time: datetime,
        channel: int = 1,
        output_path: str | Path = "clip.mp4",
    ) -> Path:
        """Extract a video clip from start_time to end_time.

        Uses ffmpeg to extract the clip from the RTSP playback stream.

        Args:
            start_time: Start timestamp for the clip
            end_time: End timestamp for the clip
            channel: Camera channel number (default: 1)
            output_path: Path to save the extracted clip

        Returns:
            Path to the extracted video file

        Raises:
            RuntimeError: If ffmpeg fails to extract the clip
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build RTSP URL for the time range
        rtsp_url = self._build_rtsp_url(channel, start_time, end_time)

        # URL-encode username and password to handle special characters
        encoded_username = urllib.parse.quote(self.username, safe="")
        encoded_password = urllib.parse.quote(self.password, safe="")
        auth_url = rtsp_url.replace(
            f"rtsp://{self.host}:{self.port}",
            f"rtsp://{encoded_username}:{encoded_password}@{self.host}:{self.port}",
        )

        # Calculate duration for ffmpeg -t option
        duration_seconds = int((end_time - start_time).total_seconds())

        cmd = [
            "ffmpeg",
            "-rtsp_transport",
            self.rtsp_transport,
            "-i",
            auth_url,
            "-t",
            str(duration_seconds),
            "-c:v",
            "copy",  # Copy video codec without re-encoding (faster)
            "-c:a",
            "copy",  # Copy audio codec without re-encoding
            "-y",  # Overwrite output file if exists
            str(output_path),
        ]

        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to extract clip: {e.stderr}") from e

        return output_path
