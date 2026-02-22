#!/usr/bin/env python3
"""Test script for object matching function.

This script tests the is_same_object matching logic by:
1. Loading test images from test_images/
2. Running detection on each image
3. Checking if the target object (motorcycle) matches any detected objects
4. Reporting results

Usage:
    uv run python test_matching.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Target object data from user
TARGET_OBJECT = {
    "camera_id": "1",
    "start_timestamp": "2026-02-19T08:30:00",
    "object_id": 2,
    "search_duration_seconds": 3600,
    "object_label": "motorcycle",
    "object_bbox": {
        "x": 380.8313903808594,
        "y": 371.013671875,
        "width": 245.56320190429688,
        "height": 196.68145751953125,
    },
    "object_confidence": 0.8423992991447449,
}


def test_matching():
    """Test object matching against test images."""
    from cctv_search.ai import ByteTrackTracker, RFDetrDetector
    from cctv_search.api import SearchObjectTracker

    # Initialize detector and tracker
    logger.info("Loading RF-DETR detector...")
    detector = RFDetrDetector()
    try:
        detector.load_model()
        logger.info("Detector loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load detector: {e}")
        return

    logger.info("Initializing ByteTrack tracker...")
    tracker = ByteTrackTracker(track_thresh=0.5, track_buffer=30, frame_rate=20)

    # Create search tracker wrapper with strict matching
    search_tracker = SearchObjectTracker(
        tracker=tracker,
        target_bbox=TARGET_OBJECT["object_bbox"],
        target_label=TARGET_OBJECT["object_label"],
    )

    # Get test images
    test_images_dir = Path("test_images")
    if not test_images_dir.exists():
        logger.error(f"Test images directory not found: {test_images_dir}")
        return

    image_files = sorted(test_images_dir.glob("*.png"))
    if not image_files:
        logger.error(f"No PNG images found in {test_images_dir}")
        return

    logger.info(f"Found {len(image_files)} test images")
    logger.info("=" * 70)

    # Create target detection object for matching
    from cctv_search.search.algorithm import BoundingBox, ObjectDetection

    target_detection = ObjectDetection(
        label=TARGET_OBJECT["object_label"],
        bbox=BoundingBox(
            x=TARGET_OBJECT["object_bbox"]["x"],
            y=TARGET_OBJECT["object_bbox"]["y"],
            width=TARGET_OBJECT["object_bbox"]["width"],
            height=TARGET_OBJECT["object_bbox"]["height"],
            confidence=TARGET_OBJECT["object_confidence"],
        ),
        confidence=TARGET_OBJECT["object_confidence"],
    )

    # Test each image
    results = []
    for img_path in image_files:
        logger.info(f"\nTesting image: {img_path.name}")
        logger.info("-" * 70)

        # Load image
        with open(img_path, "rb") as f:
            image_bytes = f.read()

        # Run detection
        logger.info("Running detection...")
        detections = detector.detect(image_bytes)
        logger.info(f"Found {len(detections)} objects")

        # Filter by label
        matching_label = [
            d for d in detections
            if d.label.lower() == TARGET_OBJECT["object_label"].lower()
        ]
        logger.info(f"  {len(matching_label)} with label '{TARGET_OBJECT['object_label']}'")

        # Check for match using is_same_object
        best_match = None
        best_score = None
        match_found = False

        if matching_label:
            logger.info("\nChecking matches with is_same_object()...")
            for i, det in enumerate(matching_label):
                # Convert to search algorithm format
                det_obj = ObjectDetection(
                    label=det.label,
                    bbox=BoundingBox(
                        x=det.bbox.x,
                        y=det.bbox.y,
                        width=det.bbox.width,
                        height=det.bbox.height,
                        confidence=det.confidence,
                    ),
                    confidence=det.confidence,
                )

                # Check if same object
                is_match = search_tracker.is_same_object(target_detection, det_obj)

                # Calculate IoU and distance for reporting
                iou = target_detection.bbox.iou_with(det_obj.bbox)
                center1 = target_detection.bbox.center
                center2 = det_obj.bbox.center
                distance = ((center1.x - center2.x) ** 2 +
                           (center1.y - center2.y) ** 2) ** 0.5

                logger.info(f"  Candidate {i+1}:")
                logger.info(f"    BBox: x={det.bbox.x:.1f}, y={det.bbox.y:.1f}, "
                           f"w={det.bbox.width:.1f}, h={det.bbox.height:.1f}")
                logger.info(f"    IoU: {iou:.3f}")
                logger.info(f"    Center distance: {distance:.1f}px")
                logger.info(f"    is_same_object: {is_match}")

                if is_match:
                    match_found = True
                    best_match = det
                    best_score = {"iou": iou, "distance": distance}

        # Record result
        result = {
            "image": img_path.name,
            "total_detections": len(detections),
            "matching_label_count": len(matching_label),
            "match_found": match_found,
            "best_match": best_score if best_match else None,
        }
        results.append(result)

        # Summary for this image
        if match_found:
            logger.info(f"\n✓ MATCH FOUND in {img_path.name}")
            logger.info(f"  IoU: {best_score['iou']:.3f}")
            logger.info(f"  Distance: {best_score['distance']:.1f}px")
        else:
            logger.info(f"\n✗ No match found in {img_path.name}")

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Target object: {TARGET_OBJECT['object_label']} "
                f"(ID: {TARGET_OBJECT['object_id']})")
    logger.info(f"Target bbox: {json.dumps(TARGET_OBJECT['object_bbox'], indent=2)}")
    logger.info("\nResults by image:")

    for r in results:
        status = "✓ MATCH" if r["match_found"] else "✗ No match"
        logger.info(f"  {r['image']}: {status}")
        logger.info(f"    Detections: {r['total_detections']}, "
                   f"Label matches: {r['matching_label_count']}")
        if r["best_match"]:
            logger.info(f"    Best IoU: {r['best_match']['iou']:.3f}, "
                       f"Distance: {r['best_match']['distance']:.1f}px")

    # Overall result
    matches_found = sum(1 for r in results if r["match_found"])
    logger.info(f"\nOverall: {matches_found}/{len(results)} images contain matching object")


if __name__ == "__main__":
    test_matching()
