"""HTTP-based HLS streaming module with in-memory storage.

This module provides HLS streaming without serving files from disk.
FFmpeg outputs to temp files, which are read into memory and served via HTTP.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import HTTPException

logger = logging.getLogger(__name__)


@dataclass
class HLSSegment:
    """Represents an HLS segment in memory."""

    index: int
    data: bytes
    duration: float = 2.0
    timestamp: datetime | None = None


@dataclass
class HLSStream:
    """Manages an in-memory HLS stream for a channel."""

    channel: int
    process: subprocess.Popen | None = None
    segments: deque[HLSSegment] = field(default_factory=lambda: deque(maxlen=20))
    start_time: datetime | None = None
    is_live: bool = True
    lock: threading.Lock = field(default_factory=threading.Lock)
    is_running: bool = True
    last_activity: float = field(default_factory=time.time)
    temp_dir: Path | None = None

    def get_segment(self, index: int) -> HLSSegment | None:
        """Get a segment by index."""
        with self.lock:
            for seg in self.segments:
                if seg.index == index:
                    return seg
            return None

    def get_latest_segments(self, count: int = 6) -> list[HLSSegment]:
        """Get the latest N segments."""
        with self.lock:
            return list(self.segments)[-count:]

    def generate_playlist(self) -> str:
        """Generate HLS playlist content."""
        with self.lock:
            lines = ["#EXTM3U", "#EXT-X-VERSION:3", "#EXT-X-TARGETDURATION:3"]

            if self.segments:
                # For live streams, use sliding window
                # Last 6 segments (12 seconds)
                segments_to_include = list(self.segments)[-6:]
                media_sequence = (
                    segments_to_include[0].index if segments_to_include else 0
                )
                lines.append(f"#EXT-X-MEDIA-SEQUENCE:{media_sequence}")

                for seg in segments_to_include:
                    lines.append(f"#EXTINF:{seg.duration:.3f},")
                    if seg.timestamp:
                        lines.append(
                            f"#EXT-X-PROGRAM-DATE-TIME:{seg.timestamp.isoformat()}"
                        )
                    lines.append(f"segment_{seg.index:03d}.ts")
            else:
                lines.append("#EXT-X-MEDIA-SEQUENCE:0")

            return "\n".join(lines) + "\n"


class HTTPStreamManager:
    """Manages HTTP-based HLS streams for multiple channels."""

    def __init__(self):
        self.streams: dict[int, HLSStream] = {}
        self.lock = threading.Lock()
        self._cleanup_task: asyncio.Task | None = None
        self._base_temp_dir = Path(tempfile.gettempdir()) / "cctv_hls"
        self._base_temp_dir.mkdir(parents=True, exist_ok=True)

    async def start(self) -> None:
        """Start the cleanup task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop all streams and cleanup."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

        with self.lock:
            for stream in list(self.streams.values()):
                await self._stop_stream(stream)
            self.streams.clear()

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of inactive streams."""
        while True:
            try:
                await asyncio.sleep(30)
                await self._cleanup_inactive_streams()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[HLS] Cleanup error: {e}")

    async def _cleanup_inactive_streams(self) -> None:
        """Stop streams that have been inactive for too long."""
        inactive_threshold = 60
        current_time = time.time()

        with self.lock:
            to_remove = []
            for channel, stream in self.streams.items():
                inactive_time = current_time - stream.last_activity
                if inactive_time > inactive_threshold:
                    to_remove.append(channel)

            for channel in to_remove:
                logger.info(f"[HLS] Stopping inactive stream for channel {channel}")
                stream = self.streams.pop(channel)
                await self._stop_stream(stream)

    async def _stop_stream(self, stream: HLSStream) -> None:
        """Stop a stream and cleanup resources."""
        stream.is_running = False

        # Terminate FFmpeg process
        if stream.process and stream.process.poll() is None:
            stream.process.terminate()
            try:
                stream.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                stream.process.kill()
                stream.process.wait()

        # Remove temp directory
        if stream.temp_dir and stream.temp_dir.exists():
            with contextlib.suppress(Exception):
                shutil.rmtree(stream.temp_dir)

        logger.info(f"[HLS] Stream stopped for channel {stream.channel}")

    def get_stream(self, channel: int) -> HLSStream | None:
        """Get active stream for a channel."""
        with self.lock:
            return self.streams.get(channel)

    async def start_stream(
        self,
        channel: int,
        nvr_client: Any,
        start_time: datetime | None = None,
    ) -> HLSStream:
        """Start a new HTTP-based HLS stream."""
        # Stop existing stream if any
        await self.stop_stream(channel)

        # Build RTSP URL
        if start_time:
            end_time = start_time + timedelta(hours=1)
            rtsp_url = nvr_client._build_rtsp_url(channel, start_time, end_time)
        else:
            rtsp_url = nvr_client._build_live_rtsp_url(channel)

        # Add authentication
        import urllib.parse

        encoded_username = urllib.parse.quote(nvr_client.username, safe="")
        encoded_password = urllib.parse.quote(nvr_client.password, safe="")
        auth_url = rtsp_url.replace(
            f"rtsp://{nvr_client.host}:{nvr_client.port}",
            f"rtsp://{encoded_username}:{encoded_password}@{nvr_client.host}:{nvr_client.port}",
        )

        logger.info(f"[HLS] Starting HTTP stream for channel {channel}")

        # Create temp directory for FFmpeg output
        temp_dir = self._base_temp_dir / f"ch{channel}_{int(time.time())}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Create stream instance
        stream = HLSStream(
            channel=channel,
            start_time=start_time,
            is_live=start_time is None,
            temp_dir=temp_dir,
        )

        playlist_path = temp_dir / "playlist.m3u8"
        segment_pattern = temp_dir / "segment_%03d.ts"

        # Start FFmpeg with HLS output to temp directory
        cmd = [
            "ffmpeg",
            "-rtsp_transport", nvr_client.rtsp_transport,
            "-thread_queue_size", "1024",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-i", auth_url,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-profile:v", "baseline",
            "-level", "3.0",
            "-g", "50",
            "-keyint_min", "50",
            "-sc_threshold", "0",
            "-r", "25",
            # HLS output
            "-f", "hls",
            "-hls_time", "2",
            "-hls_list_size", "6",
            "-hls_flags", "delete_segments+omit_endlist",
            "-hls_segment_type", "mpegts",
            "-hls_segment_filename", str(segment_pattern),
            str(playlist_path),
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        stream.process = process

        with self.lock:
            self.streams[channel] = stream

        # Start segment reader thread
        reader_thread = threading.Thread(
            target=self._read_segments,
            args=(stream, temp_dir),
            daemon=True,
        )
        reader_thread.start()

        logger.info(f"[HLS] Stream started for channel {channel}")
        return stream

    async def stop_stream(self, channel: int) -> None:
        """Stop a stream for a specific channel."""
        with self.lock:
            stream = self.streams.pop(channel, None)

        if stream:
            await self._stop_stream(stream)

    def _read_segments(self, stream: HLSStream, temp_dir: Path) -> None:
        """Read segments from FFmpeg output directory."""
        import glob

        segment_pattern = temp_dir / "segment_*.ts"

        try:
            while stream.is_running:
                # Look for new segment files
                segment_files = sorted(glob.glob(str(segment_pattern)))

                for seg_path in segment_files:
                    seg_file = Path(seg_path)
                    try:
                        # Check if file has been fully written (modified > 100ms ago)
                        mtime = seg_file.stat().st_mtime
                        if time.time() - mtime < 0.1:
                            # File still being written, skip for now
                            continue

                        # Read segment data
                        data = seg_file.read_bytes()
                        if len(data) == 0:
                            continue

                        # Extract index from filename
                        index = int(seg_file.stem.split("_")[-1])

                        # Check if we already have this segment
                        if stream.get_segment(index):
                            # Delete file after reading if already in memory
                            seg_file.unlink()
                            continue

                        # Create segment
                        timestamp = None
                        if stream.start_time:
                            timestamp = stream.start_time + timedelta(
                                seconds=index * 2
                            )

                        segment = HLSSegment(
                            index=index,
                            data=data,
                            duration=2.0,
                            timestamp=timestamp,
                        )

                        with stream.lock:
                            stream.segments.append(segment)
                            stream.last_activity = time.time()

                        logger.debug(
                            f"[HLS] Channel {stream.channel}: "
                            f"Added segment {index} ({len(data)} bytes)"
                        )

                        # Delete file after reading into memory
                        seg_file.unlink()

                    except Exception as e:
                        logger.error(
                            f"[HLS] Error reading segment {seg_path}: {e}"
                        )

                # Check if FFmpeg is still running
                if stream.process and stream.process.poll() is not None:
                    # FFmpeg exited
                    if stream.process.stderr:
                        stderr = stream.process.stderr.read()
                        if stderr:
                            err_msg = stderr.decode(
                                'utf-8', errors='ignore'
                            )[-500:]
                            logger.error(f"[HLS] FFmpeg error: {err_msg}")
                    break

                time.sleep(0.2)  # Check every 200ms

        except Exception as e:
            channel = stream.channel
            logger.error(f"[HLS] Error in segment reader for channel {channel}: {e}")
        finally:
            stream.is_running = False

    async def get_playlist(self, channel: int) -> str:
        """Get HLS playlist for a channel."""
        stream = self.get_stream(channel)
        if not stream:
            raise HTTPException(
                status_code=404,
                detail=f"Stream not found for channel {channel}",
            )

        stream.last_activity = time.time()
        return stream.generate_playlist()

    async def get_segment(self, channel: int, segment_index: int) -> bytes:
        """Get a specific segment for a channel."""
        stream = self.get_stream(channel)
        if not stream:
            raise HTTPException(
                status_code=404,
                detail=f"Stream not found for channel {channel}",
            )

        segment = stream.get_segment(segment_index)
        if not segment:
            raise HTTPException(
                status_code=404,
                detail=f"Segment {segment_index} not found",
            )

        stream.last_activity = time.time()
        return segment.data


# Global instance
http_stream_manager = HTTPStreamManager()
