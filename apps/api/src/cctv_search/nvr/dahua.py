"""Dahua NVR client using RTSP playback endpoint."""

from __future__ import annotations

import os
import subprocess
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta
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

        print(f"[NVR] Initialized: host={self.host}, port={self.port}, "
              f"user={self.username}")
        if not self.host:
            print("[NVR] WARNING: NVR_HOST not set!")
        if not self.username:
            print("[NVR] WARNING: NVR_USERNAME not set!")
        if not self.password:
            print("[NVR] WARNING: NVR_PASSWORD not set!")

    def _format_timestamp(self, dt: datetime) -> str:
        """Format datetime for Dahua RTSP endpoint (YYYY_MM_DD_HH_MM_SS)."""
        formatted = dt.strftime("%Y_%m_%d_%H_%M_%S")
        print(f"[DEBUG] _format_timestamp: {dt} -> {formatted}")
        return formatted

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

        # DEBUG: Log timestamp details
        print(f"[DEBUG] extract_frame received timestamp: {timestamp}")
        print(f"[DEBUG] timestamp.isoformat(): {timestamp.isoformat()}")
        print(f"[DEBUG] timestamp.tzinfo: {timestamp.tzinfo}")

        # Build RTSP URL with 1-second window around the timestamp
        start_time = timestamp
        end_time = timestamp + timedelta(seconds=1)
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

    def _build_live_rtsp_url(self, channel: int) -> str:
        """Build RTSP live stream URL (real-time).

        Dahua live stream URL format:
        rtsp://<server>:[port]/cam/realmonitor?channel=<channel>&subtype=<0|1>
        subtype=0 for main stream (higher quality), 1 for sub stream
        """
        return (
            f"rtsp://{self.host}:{self.port}"
            f"/cam/realmonitor?channel={channel}&subtype=0"
        )

    def start_hls_stream(
        self,
        channel: int = 1,
        output_dir: str | Path = "/tmp/hls",
        start_time: datetime | None = None,
    ) -> tuple[subprocess.Popen, str]:
        """Start HLS streaming from RTSP source.

        Transcodes RTSP stream to HLS for browser playback.
        Supports both live streaming and playback from specific time.

        Args:
            channel: Camera channel number (default: 1)
            output_dir: Directory to save HLS files
            start_time: If provided, start playback from this time (playback mode)
                       If None, stream live (live mode)

        Returns:
            Tuple of (ffmpeg process, playlist path)
        """
        output_dir = Path(output_dir) / f"ch{channel}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build RTSP URL based on mode
        if start_time:
            # Playback mode - use playback URL with start time
            # End time is 1 hour after start for playback
            end_time = start_time + timedelta(hours=1)
            rtsp_url = self._build_rtsp_url(channel, start_time, end_time)
        else:
            # Live mode - use live stream URL
            rtsp_url = self._build_live_rtsp_url(channel)

        # Add authentication to URL
        encoded_username = urllib.parse.quote(self.username, safe="")
        encoded_password = urllib.parse.quote(self.password, safe="")
        auth_url = rtsp_url.replace(
            f"rtsp://{self.host}:{self.port}",
            f"rtsp://{encoded_username}:{encoded_password}@{self.host}:{self.port}",
        )

        playlist_path = output_dir / "playlist.m3u8"

        # Clean up old segments to avoid confusion
        for old_segment in output_dir.glob("segment_*.ts"):
            old_segment.unlink()
        if playlist_path.exists():
            playlist_path.unlink()

        # Calculate base timestamp for HLS (epoch seconds since 1970-01-01)
        # For playback mode, use the requested start_time
        # For live mode, use current time
        if start_time:
            # Convert datetime to epoch seconds
            hls_base_time = start_time.timestamp()
            print(f"[HLS] Playback mode - base time: {start_time.isoformat()} "
                  f"({hls_base_time})")
        else:
            hls_base_time = 0  # Live mode - use relative time
            print("[HLS] Live mode - using relative time")

        # FFmpeg command to transcode RTSP to HLS
        # Optimized for FAST START (low latency over stability)
        # Includes program date time for accurate timestamp tracking
        cmd = [
            "ffmpeg",
            "-rtsp_transport", self.rtsp_transport,
            # Buffer settings for input - smaller for faster start
            "-thread_queue_size", "1024",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            # No -re flag - process as fast as possible for faster start
            "-i", auth_url,
            # Video codec settings - faster preset for speed
            "-c:v", "libx264",
            "-preset", "ultrafast",  # Fastest encoding
            "-tune", "zerolatency",
            "-profile:v", "baseline",
            "-level", "3.0",
            # Keyframe settings for HLS - every 2 seconds
            "-g", "50",
            "-keyint_min", "50",
            "-sc_threshold", "0",
            "-r", "25",
            # HLS output settings - optimized for FAST START
            "-f", "hls",
            "-hls_time", "2",  # 2 second segments (faster start than 4)
            "-hls_list_size", "6",  # Keep 6 segments (12 seconds)
            # CRITICAL: append_list writes playlist after EACH segment
            "-hls_flags",
            "independent_segments+delete_segments+append_list",
            "-hls_segment_type", "mpegts",
            "-hls_segment_filename", str(output_dir / "segment_%03d.ts"),
            "-start_number", "0",
            "-hls_playlist_type", "event",
            "-y",
            str(playlist_path),
        ]

        # Start ffmpeg process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for playlist file to be created (with timeout)
        import time
        max_wait = 15  # 15 seconds max - should start in ~5-10s with optimized settings
        waited = 0
        first_segment = output_dir / "segment_000.ts"

        print(f"[DEBUG] Starting ffmpeg for channel {channel}")
        safe_url = auth_url.replace(self.password, '***')
        print(f"[DEBUG] RTSP URL: {safe_url}")
        print(f"[DEBUG] Playlist path: {playlist_path}")

        while waited < max_wait:
            time.sleep(0.5)
            waited += 0.5

            # Check if process is still running
            if process.poll() is not None:
                # Process exited early - there was an error
                stdout, stderr = process.communicate()
                error_msg = stderr.decode('utf-8', errors='ignore')[-1000:]
                print(f"[DEBUG] FFmpeg stderr: {error_msg}")
                raise RuntimeError(f"FFmpeg failed to start: {error_msg}")

            # Check if both playlist and first segment exist and have content
            if playlist_path.exists() and first_segment.exists():
                playlist_size = playlist_path.stat().st_size
                segment_size = first_segment.stat().st_size
                if playlist_size > 0 and segment_size > 100000:  # At least 100KB
                    print(f"[DEBUG] Stream ready after {waited}s")
                    print(f"[DEBUG] Playlist: {playlist_size} bytes")
                    print(f"[DEBUG] First segment: {segment_size} bytes")
                    break

            # Log progress every 5 seconds
            if waited % 5 == 0:
                # Check what files exist
                files = list(output_dir.glob("*"))
                print(f"[DEBUG] Waiting for stream... {waited}s elapsed")
                print(f"[DEBUG] Files in output dir: {[f.name for f in files]}")

        if not playlist_path.exists() or not first_segment.exists():
            # Capture any output before terminating
            try:
                stdout, stderr = process.communicate(timeout=2)
                error_msg = stderr.decode('utf-8', errors='ignore')[-1000:]
                print(f"[DEBUG] FFmpeg stderr on timeout: {error_msg}")
            except subprocess.TimeoutExpired:
                pass
            process.terminate()
            raise RuntimeError(
                f"Timeout waiting for HLS stream after {max_wait}s. "
                "Check NVR connection and credentials."
            )

        # Create timestamp mapping file for accurate frame extraction
        # This maps segment indices to actual video timestamps
        if start_time:
            timestamp_file = output_dir / "timestamps.json"
            self._create_timestamp_mapping(timestamp_file, start_time)
            print(f"[HLS] Created timestamp mapping: {timestamp_file}")

        return process, str(playlist_path)

    def _create_timestamp_mapping(
        self,
        timestamp_file: Path,
        start_time: datetime,
    ) -> None:
        """Create timestamp mapping file for accurate frame extraction.

        This creates a JSON file that maps segment indices to actual video
        timestamps, allowing the UI to get accurate timestamps without
        relying on HLS PROGRAM-DATE-TIME which uses wallclock time.

        Args:
            timestamp_file: Path to the timestamp mapping file
            start_time: The actual start time of the playback
        """
        import json

        mapping = {
            "start_time": start_time.isoformat(),
            "segment_duration": 2.0,
            "segments": {},
        }

        # Pre-populate first 10 segments
        for i in range(10):
            segment_time = start_time + timedelta(seconds=i * 2.0)
            mapping["segments"][f"segment_{i:03d}.ts"] = segment_time.isoformat()

        timestamp_file.write_text(json.dumps(mapping, indent=2))
        print(f"[HLS] Timestamp mapping: {start_time.isoformat()}")
