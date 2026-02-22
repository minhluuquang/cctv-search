#!/usr/bin/env python3
"""NVR Video Extraction, Detection and Tracking Script.

This script extracts a video clip from the Dahua NVR for a time range,
runs RF-DETR object detection and feature-based multi-object tracking on each frame,
and outputs an annotated video with bounding boxes and track IDs.

Usage:
    # Extract and track video clip
    python scripts/extract_and_track.py \
        --start "2026-02-19 10:00:00" \
        --duration 30 \
        --output tracked_video.mp4
    
    # Track specific object class only
    python scripts/extract_and_track.py \
        --start "2026-02-19 10:00:00" \
        --duration 30 \
        --class person \
        --output people_tracking.mp4
    
    # Adjust detection confidence
    python scripts/extract_and_track.py \
        --start "2026-02-19 10:00:00" \
        --duration 60 \
        --confidence 0.7 \
        --output high_conf_tracking.mp4
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


def extract_video_from_nvr(
    start_time: datetime,
    duration_seconds: int,
    channel: int = 1,
    output_path: Path | None = None
) -> Path:
    """Extract video clip from NVR using ffmpeg.
    
    Args:
        start_time: Start timestamp
        duration_seconds: Duration in seconds
        channel: Camera channel (default: 1)
        output_path: Where to save the video (optional)
        
    Returns:
        Path to extracted video
    """
    from cctv_search.nvr import DahuaNVRClient
    
    logger.info(f"Connecting to NVR...")
    client = DahuaNVRClient()
    
    if output_path is None:
        timestamp_str = start_time.strftime("%Y_%m_%d_%H%M%S")
        output_path = Path(f"video_{timestamp_str}_ch{channel}_{duration_seconds}s.mp4")
    
    # Build RTSP URL
    end_time = start_time + timedelta(seconds=duration_seconds)
    
    # Format timestamps for Dahua
    start_str = start_time.strftime("%Y_%m_%d_%H_%M_%S")
    end_str = end_time.strftime("%Y_%m_%d_%H_%M_%S")
    
    rtsp_url = (
        f"rtsp://{client.username}:{client.password}@{client.host}:{client.port}"
        f"/cam/playback?channel={channel}"
        f"&starttime={start_str}&endtime={end_str}"
    )
    
    logger.info(f"Extracting {duration_seconds}s video from {start_time}")
    logger.info(f"NVR Host: {client.host}")
    logger.info(f"RTSP URL: {rtsp_url[:50]}...")
    
    # Use ffmpeg to extract video
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-t", str(duration_seconds),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    
    try:
        logger.info("Running ffmpeg...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=duration_seconds + 60
        )
        
        if result.returncode != 0:
            logger.error(f"ffmpeg stderr: {result.stderr}")
            raise RuntimeError(f"ffmpeg failed: {result.returncode}")
        
        logger.info(f"✓ Video extracted: {output_path}")
        logger.info(f"  File size: {output_path.stat().st_size:,} bytes")
        return output_path
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("Video extraction timed out")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg")


def process_video_with_tracking(
    video_path: Path,
    output_path: Path,
    confidence_threshold: float = 0.5,
    target_class: str | None = None,
    fps: int = 20,
) -> dict:
    """Process video with detection and feature-based tracking.
    
    Args:
        video_path: Path to input video
        output_path: Path for output video
        confidence_threshold: Detection confidence threshold
        target_class: Only track this class (optional)
        fps: Video frame rate
        
    Returns:
        Statistics dict with tracking info
    """
    import cv2
    import numpy as np
    
    from cctv_search.ai import RFDetrDetector, FeatureTracker, BoundingBox, DetectedObject
    
    # Initialize detector and tracker
    logger.info(f"Initializing detector (confidence: {confidence_threshold})...")
    detector = RFDetrDetector(confidence_threshold=confidence_threshold)
    detector.load_model()
    
    logger.info("Initializing FeatureTracker...")
    tracker = FeatureTracker(
        feature_threshold=0.75,
        iou_threshold=0.8,
        max_age=30,
    )
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"Video: {width}x{height} @ {video_fps:.1f}fps, {total_frames} frames")
    
    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, video_fps, (width, height))
    
    # Color map for different track IDs
    def get_track_color(track_id: int) -> tuple:
        """Generate consistent color for track ID."""
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (0, 128, 128),  # Teal
            (128, 128, 0),  # Olive
        ]
        return colors[track_id % len(colors)]
    
    # Process frames
    frame_idx = 0
    all_tracks = {}  # track_id -> list of positions
    
    logger.info("Processing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to bytes for detector
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Detect objects
        detections = detector.detect(frame_bytes)
        
        # Filter by class if specified
        if target_class:
            detections = [d for d in detections if d.label.lower() == target_class.lower()]
        
        # Update tracker
        tracks = tracker.update(detections, frame_idx)
        
        # Draw tracks on frame
        for track in tracks:
            if not track.is_active:
                continue
            
            # Get color for this track
            color = get_track_color(track.track_id)
            
            # Calculate box coordinates (center -> top-left)
            x1 = int(track.x - track.width / 2)
            y1 = int(track.y - track.height / 2)
            x2 = int(track.x + track.width / 2)
            y2 = int(track.y + track.height / 2)
            
            # Ensure within bounds
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID and label
            label = f"ID:{track.track_id} {track.label}"
            if track.is_activated:
                label += f" ({track.confidence:.0%})"
            
            # Text background
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame,
                (x1, y1 - text_h - 10),
                (x1 + text_w, y1),
                color,
                -1
            )
            
            # Text
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            # Store track info
            if track.track_id not in all_tracks:
                all_tracks[track.track_id] = {
                    'label': track.label,
                    'positions': [],
                    'frames': [],
                }
            all_tracks[track.track_id]['positions'].append((track.x, track.y))
            all_tracks[track.track_id]['frames'].append(frame_idx)
        
        # Draw frame number
        cv2.putText(
            frame,
            f"Frame: {frame_idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Draw active track count
        active_count = sum(1 for t in tracks if t.is_active)
        cv2.putText(
            frame,
            f"Active Tracks: {active_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Write frame
        out.write(frame)
        
        # Progress
        frame_idx += 1
        if frame_idx % 30 == 0:
            progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
            logger.info(f"  Frame {frame_idx}/{total_frames} ({progress:.1f}%) - {len(tracks)} tracks")
    
    # Cleanup
    cap.release()
    out.release()
    
    # Calculate statistics
    stats = {
        'total_frames': frame_idx,
        'unique_tracks': len(all_tracks),
        'track_details': {},
    }
    
    for track_id, info in all_tracks.items():
        frames_tracked = len(info['frames'])
        duration_frames = max(info['frames']) - min(info['frames']) + 1 if info['frames'] else 0
        stats['track_details'][track_id] = {
            'label': info['label'],
            'frames_tracked': frames_tracked,
            'duration_frames': duration_frames,
            'first_frame': min(info['frames']) if info['frames'] else 0,
            'last_frame': max(info['frames']) if info['frames'] else 0,
        }
    
    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract video from NVR and run object tracking"
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start timestamp (format: 'YYYY-MM-DD HH:MM:SS')"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Duration in seconds (default: 30)"
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=1,
        help="Camera channel (default: 1)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output video path (default: tracked_<timestamp>.mp4)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--class",
        dest="target_class",
        type=str,
        help="Only track this class (e.g., 'person', 'car')"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Video frame rate (default: 20)"
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep the raw video file (not just annotated)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NVR Video Extraction & Object Tracking")
    print("=" * 60)
    
    try:
        # Parse start time
        start_time = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
        
        # Step 1: Extract video from NVR
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_video_path = Path(tmpdir) / "raw_video.mp4"
            
            extracted_path = extract_video_from_nvr(
                start_time=start_time,
                duration_seconds=args.duration,
                channel=args.channel,
                output_path=raw_video_path
            )
            
            # Step 2: Process with tracking
            if args.output:
                output_path = Path(args.output)
            else:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = Path(f"tracked_{timestamp_str}.mp4")
            
            logger.info(f"\nProcessing video with detection + tracking...")
            stats = process_video_with_tracking(
                video_path=extracted_path,
                output_path=output_path,
                confidence_threshold=args.confidence,
                target_class=args.target_class,
                fps=args.fps,
            )
            
            # Step 3: Copy raw video if requested
            if args.keep_raw:
                raw_output = output_path.parent / f"{output_path.stem}_raw.mp4"
                import shutil
                shutil.copy(extracted_path, raw_output)
                logger.info(f"✓ Raw video saved: {raw_output}")
        
        # Print summary
        print()
        print("=" * 60)
        print("✅ Processing complete!")
        print("=" * 60)
        print(f"\nInput:  {args.start} (+{args.duration}s)")
        print(f"Output: {output_path}")
        print(f"\nStatistics:")
        print(f"  Frames processed: {stats['total_frames']}")
        print(f"  Unique tracks: {stats['unique_tracks']}")
        print(f"\nTrack Details:")
        for track_id, info in stats['track_details'].items():
            print(f"  Track {track_id}: {info['label']}")
            print(f"    Frames: {info['frames_tracked']}")
            print(f"    Duration: {info['duration_frames']} frames")
            print(f"    First seen: frame {info['first_frame']}")
            print(f"    Last seen: frame {info['last_frame']}")
        
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
