#!/usr/bin/env python3
"""NVR Frame Extraction and Object Detection Script.

This script extracts a frame from the Dahua NVR at a specific timestamp,
runs RF-DETR object detection on it, and outputs an annotated image with
bounding boxes.

Usage:
    # Extract and detect at specific time
    python scripts/extract_and_detect.py --timestamp "2026-02-19 10:00:00"
    
    # Use custom output path
    python scripts/extract_and_detect.py --timestamp "2026-02-19 10:00:00" \
                                         --output detected_frame.jpg
    
    # Adjust confidence threshold
    python scripts/extract_and_detect.py --timestamp "2026-02-19 10:00:00" \
                                         --confidence 0.7
    
    # Test with local image (no NVR required)
    python scripts/extract_and_detect.py --image /path/to/image.jpg
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from cctv_search.ai import DetectedObject


def load_image(image_path: str) -> bytes:
    """Load image from file."""
    with open(image_path, 'rb') as f:
        return f.read()


def extract_frame_from_nvr(
    timestamp: datetime,
    channel: int = 1,
    output_path: Path | None = None
) -> Path:
    """Extract frame from NVR at given timestamp.
    
    Args:
        timestamp: Target timestamp
        channel: Camera channel (default: 1)
        output_path: Where to save the frame (optional)
        
    Returns:
        Path to extracted frame
    """
    from cctv_search.nvr import DahuaNVRClient
    
    logger.info(f"Connecting to NVR...")
    client = DahuaNVRClient()
    
    if output_path is None:
        timestamp_str = timestamp.strftime("%Y_%m_%d_%H%M%S")
        output_path = Path(f"frame_{timestamp_str}_ch{channel}.png")
    
    logger.info(f"Extracting frame at {timestamp} from channel {channel}")
    logger.info(f"NVR Host: {client.host}")
    
    try:
        result = client.extract_frame(
            timestamp=timestamp,
            channel=channel,
            output_path=output_path
        )
        logger.info(f"✓ Frame extracted: {result}")
        logger.info(f"  File size: {result.stat().st_size:,} bytes")
        return result
    except Exception as e:
        logger.error(f"✗ Failed to extract frame: {e}")
        raise


def load_detector(confidence_threshold: float = 0.5) -> "RFDetrDetector":
    """Load RF-DETR detector.
    
    Args:
        confidence_threshold: Minimum confidence for detections
        
    Returns:
        Loaded detector instance
    """
    from cctv_search.ai import RFDetrDetector
    
    logger.info(f"Initializing RF-DETR detector (confidence: {confidence_threshold})...")
    detector = RFDetrDetector(confidence_threshold=confidence_threshold)
    
    try:
        logger.info("Loading model (may download on first run)...")
        detector.load_model()
        logger.info("✓ Model loaded successfully")
        return detector
    except RuntimeError as e:
        logger.error(f"✗ Failed to load model: {e}")
        logger.error("Install RF-DETR: pip install rfdetr")
        raise


def detect_objects(
    detector: "RFDetrDetector",
    image_path: Path
) -> list["DetectedObject"]:
    """Run object detection on image.
    
    Args:
        detector: Loaded detector
        image_path: Path to image file
        
    Returns:
        List of detected objects
    """
    logger.info(f"Running detection on {image_path}...")
    
    # Load image
    frame_bytes = load_image(str(image_path))
    logger.info(f"  Image size: {len(frame_bytes):,} bytes")
    
    # Run detection
    start_time = datetime.now()
    detections = detector.detect(frame_bytes)
    elapsed = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"✓ Detection complete in {elapsed*1000:.1f}ms")
    logger.info(f"✓ Found {len(detections)} objects:")
    
    for i, det in enumerate(detections, 1):
        logger.info(f"  {i}. {det.label.upper()}")
        logger.info(f"     Confidence: {det.confidence:.1%}")
        logger.info(f"     Location: ({det.bbox.x:.0f}, {det.bbox.y:.0f})")
        logger.info(f"     Size: {det.bbox.width:.0f}x{det.bbox.height:.0f}")
    
    return detections


def draw_bounding_boxes(
    image_path: Path,
    detections: list["DetectedObject"],
    output_path: Path
) -> Path:
    """Draw bounding boxes on image and save.
    
    Args:
        image_path: Path to original image
        detections: List of detections to draw
        output_path: Where to save annotated image
        
    Returns:
        Path to annotated image
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.error("OpenCV not installed. Install with: pip install opencv-python")
        raise
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    height, width = image.shape[:2]
    logger.info(f"Drawing boxes on {width}x{height} image...")
    
    # Color map for different classes
    color_map = {
        'person': (0, 255, 0),      # Green
        'bicycle': (255, 0, 0),     # Blue
        'car': (0, 0, 255),         # Red
        'motorcycle': (255, 255, 0), # Cyan
        'truck': (255, 0, 255),     # Magenta
        'bus': (0, 255, 255),       # Yellow
    }
    
    # Draw each detection
    for det in detections:
        # Get color (default to white if class not in map)
        color = color_map.get(det.label.lower(), (255, 255, 255))
        
        # Calculate box coordinates
        x1 = int(det.bbox.x)
        y1 = int(det.bbox.y)
        x2 = int(det.bbox.x + det.bbox.width)
        y2 = int(det.bbox.y + det.bbox.height)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label text
        label = f"{det.label}: {det.confidence:.0%}"
        
        # Calculate text size
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Draw label background
        cv2.rectangle(
            image,
            (x1, y1 - text_height - 10),
            (x1 + text_width, y1),
            color,
            -1  # Filled
        )
        
        # Draw label text
        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # White text
            2
        )
    
    # Save annotated image
    cv2.imwrite(str(output_path), image)
    logger.info(f"✓ Annotated image saved: {output_path}")
    logger.info(f"  File size: {output_path.stat().st_size:,} bytes")
    
    return output_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract frame from NVR and run object detection"
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        help="Timestamp to extract (format: 'YYYY-MM-DD HH:MM:SS')"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Use local image instead of NVR (for testing)"
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
        help="Output path for annotated image"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Also save the raw (non-annotated) frame"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.timestamp and not args.image:
        parser.error("Either --timestamp or --image must be specified")
    
    print("=" * 60)
    print("NVR Frame Extraction & Object Detection")
    print("=" * 60)
    
    try:
        # Step 1: Get image
        if args.image:
            # Use local image
            image_path = Path(args.image)
            if not image_path.exists():
                logger.error(f"Image not found: {image_path}")
                sys.exit(1)
            logger.info(f"Using local image: {image_path}")
        else:
            # Extract from NVR
            timestamp = datetime.strptime(args.timestamp, "%Y-%m-%d %H:%M:%S")
            image_path = extract_frame_from_nvr(
                timestamp=timestamp,
                channel=args.channel
            )
        
        # Step 2: Load detector
        detector = load_detector(confidence_threshold=args.confidence)
        
        # Step 3: Run detection
        detections = detect_objects(detector, image_path)
        
        # Step 4: Draw bounding boxes
        if args.output:
            output_path = Path(args.output)
        else:
            # Generate default output name
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"detected_{timestamp_str}.jpg")
        
        annotated_path = draw_bounding_boxes(image_path, detections, output_path)
        
        # Step 5: Optionally save raw frame
        if args.save_raw and not args.image:
            raw_output = image_path.parent / f"{image_path.stem}_raw{image_path.suffix}"
            import shutil
            shutil.copy(image_path, raw_output)
            logger.info(f"✓ Raw frame saved: {raw_output}")
        
        print()
        print("=" * 60)
        print("✅ Processing complete!")
        print("=" * 60)
        print(f"\nInput:  {image_path}")
        print(f"Output: {annotated_path}")
        print(f"Objects detected: {len(detections)}")
        
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
