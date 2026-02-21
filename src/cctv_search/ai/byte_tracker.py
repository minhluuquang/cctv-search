"""ByteTrack-style object tracker implementation using roboflow/trackers.

This module provides a wrapper around the official roboflow/trackers ByteTrack
implementation, adapting it to work with the cctv_search DetectedObject format.

The algorithm uses:
- Kalman filter for motion prediction
- Hungarian algorithm for data association
- Two-stage matching (high/low confidence)

Usage:
    from cctv_search.ai.byte_tracker import ByteTrackTracker, Track

    tracker = ByteTrackTracker()
    tracks = tracker.update(detections, frame_idx=100)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker as _ByteTrackTracker

if TYPE_CHECKING:
    from cctv_search.ai import DetectedObject

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Represents a tracked object.

    Attributes:
        track_id: Unique track identifier
        label: Object class label
        x: Center X coordinate
        y: Center Y coordinate
        width: Bounding box width
        height: Bounding box height
        confidence: Detection confidence
        frame_idx: Frame where track was updated
        is_active: Whether track is currently active
        is_activated: Whether track has been confirmed
        state: Track state ('tracked', 'lost', or 'removed')
    """

    track_id: int
    label: str
    x: float
    y: float
    width: float
    height: float
    confidence: float
    frame_idx: int
    is_active: bool = True
    is_activated: bool = False
    state: str = "tracked"
    age: int = 0
    hits: int = 0

    def update(self, detection: DetectedObject, frame_idx: int) -> None:
        """Update track with new detection."""
        self.x = detection.bbox.x + detection.bbox.width / 2
        self.y = detection.bbox.y + detection.bbox.height / 2
        self.width = detection.bbox.width
        self.height = detection.bbox.height
        self.confidence = detection.confidence
        self.frame_idx = frame_idx
        self.hits += 1
        self.age = 0
        self.is_active = True
        if self.hits >= 3:  # Activate after 3 hits
            self.is_activated = True

    def predict(self) -> None:
        """Predict next position (simple motion model)."""
        self.age += 1
        if self.age > 30:  # Mark as lost after 30 frames
            self.state = "lost"
            self.is_active = False

    def mark_removed(self) -> None:
        """Mark track as removed."""
        self.state = "removed"
        self.is_active = False


class ByteTrackTracker:
    """ByteTrack multi-object tracker using roboflow/trackers.

    Wrapper around the official ByteTrack implementation that adapts it
    to work with cctv_search's DetectedObject format.

    Example:
        >>> tracker = ByteTrackTracker(track_thresh=0.5)
        >>> detections = [
        ...     DetectedObject(label="person", bbox=bbox, confidence=0.9, ...)
        ... ]
        >>> tracks = tracker.update(detections, frame_idx=100)
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        frame_rate: int = 20,
    ):
        """Initialize ByteTrack tracker.

        Args:
            track_thresh: Detection confidence threshold for first matching
            match_thresh: IoU threshold for matching (kept for API compatibility)
            track_buffer: Maximum frames to keep lost tracks
            frame_rate: Video frame rate
        """
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.frame_rate = frame_rate

        # Initialize the official ByteTrack tracker
        # Note: track_activation_threshold corresponds to track_thresh
        self._tracker = _ByteTrackTracker(
            lost_track_buffer=track_buffer,
            frame_rate=float(frame_rate),
            track_activation_threshold=track_thresh,
            minimum_consecutive_frames=2,
            minimum_iou_threshold=0.1,
            high_conf_det_threshold=0.6,
        )

        self._frame_count: int = 0

        # Keep track of detection-to-track mappings
        self._label_map: dict[int, str] = {}

        logger.info(f"ByteTrack tracker initialized (thresh={track_thresh})")

    def update(
        self,
        detections: list[DetectedObject],
        frame_idx: int,
    ) -> list[Track]:
        """Update tracker with new detections.

        Args:
            detections: List of detections from current frame
            frame_idx: Current frame index

        Returns:
            List of active tracks
        """
        self._frame_count = frame_idx

        if not detections:
            # Process empty frame to let tracker handle lost tracks
            empty_detections = sv.Detections(
                xyxy=np.empty((0, 4)),
                confidence=np.empty(0),
                class_id=np.empty(0, dtype=int),
            )
            _ = self._tracker.update(empty_detections)
            return []

        # Convert DetectedObject list to supervision.Detections
        sv_detections = self._to_sv_detections(detections)

        # Run the official tracker
        tracked_detections = self._tracker.update(sv_detections)

        # Convert back to Track objects
        tracks = self._to_tracks(tracked_detections, frame_idx)

        return tracks

    def _to_sv_detections(self, detections: list[DetectedObject]) -> sv.Detections:
        """Convert DetectedObject list to supervision.Detections.

        Args:
            detections: List of DetectedObject

        Returns:
            supervision.Detections object
        """
        if not detections:
            return sv.Detections(
                xyxy=np.empty((0, 4)),
                confidence=np.empty(0),
                class_id=np.empty(0, dtype=int),
            )

        xyxy_list = []
        confidence_list = []
        class_id_list = []
        label_map = {}

        for _i, det in enumerate(detections):
            # Convert bbox (x, y, width, height) to xyxy (x1, y1, x2, y2)
            x1 = det.bbox.x
            y1 = det.bbox.y
            x2 = det.bbox.x + det.bbox.width
            y2 = det.bbox.y + det.bbox.height
            xyxy_list.append([x1, y1, x2, y2])

            confidence_list.append(det.confidence)

            # Map labels to class IDs
            # Use a simple hash-based mapping for consistency
            label_hash = hash(det.label) % 10000
            class_id_list.append(label_hash)
            label_map[label_hash] = det.label

        self._label_map = label_map

        return sv.Detections(
            xyxy=np.array(xyxy_list),
            confidence=np.array(confidence_list),
            class_id=np.array(class_id_list, dtype=int),
        )

    def _to_tracks(
        self, detections: sv.Detections, frame_idx: int
    ) -> list[Track]:
        """Convert supervision.Detections to Track objects.

        Args:
            detections: Tracked detections with tracker_id
            frame_idx: Current frame index

        Returns:
            List of Track objects
        """
        tracks: list[Track] = []

        if detections.is_empty():
            return tracks

        tracker_ids = detections.tracker_id
        if tracker_ids is None:
            return tracks

        confidences = detections.confidence
        class_ids = detections.class_id
        if confidences is None or class_ids is None:
            return tracks

        for i in range(len(detections)):
            xyxy = detections.xyxy[i]
            confidence = confidences[i]
            class_id = class_ids[i]
            tracker_id = tracker_ids[i]

            # Skip tracks that haven't been activated yet (tracker_id < 0)
            if tracker_id < 0:
                continue

            # Convert xyxy back to center x, y, width, height
            x1, y1, x2, y2 = xyxy
            width = x2 - x1
            height = y2 - y1
            x = x1 + width / 2
            y = y1 + height / 2

            # Get label from mapping or use class_id as string
            label = self._label_map.get(int(class_id), f"class_{class_id}")

            track = Track(
                track_id=int(tracker_id),
                label=label,
                x=float(x),
                y=float(y),
                width=float(width),
                height=float(height),
                confidence=float(confidence),
                frame_idx=frame_idx,
                is_active=True,
                is_activated=True,  # Official tracker handles activation internally
                state="tracked",
            )
            tracks.append(track)

        return tracks

    def is_same_object(
        self,
        detection1: Any,
        detection2: Any,
    ) -> bool:
        """Check if two detections represent the same object instance.

        This is the core "SameBike" predicate using IoU and motion consistency.

        Args:
            detection1: First detection
            detection2: Second detection

        Returns:
            True if detections likely represent the same physical object
        """
        from cctv_search.ai import BoundingBox

        # Handle both Detection objects and DetectedObject objects
        def get_bbox(det):
            if hasattr(det, "bbox"):
                return det.bbox
            return det

        def get_label(det):
            if hasattr(det, "label"):
                return det.label
            if hasattr(det, "class_label"):
                return det.class_label
            return None

        bbox1 = get_bbox(detection1)
        bbox2 = get_bbox(detection2)
        label1 = get_label(detection1)
        label2 = get_label(detection2)

        # Different classes can never be the same object
        if label1 is not None and label2 is not None and label1 != label2:
            return False

        # Convert to BoundingBox if needed
        def get_coords(bbox):
            if isinstance(bbox, BoundingBox):
                x1 = bbox.x
                y1 = bbox.y
                x2 = bbox.x + bbox.width
                y2 = bbox.y + bbox.height
            else:
                x1 = bbox.x1 if hasattr(bbox, "x1") else bbox.x
                y1 = bbox.y1 if hasattr(bbox, "y1") else bbox.y
                x2 = bbox.x2 if hasattr(bbox, "x2") else bbox.x + bbox.width
                y2 = bbox.y2 if hasattr(bbox, "y2") else bbox.y + bbox.height
            return x1, y1, x2, y2

        x1_1, y1_1, x2_1, y2_1 = get_coords(bbox1)
        x1_2, y1_2, x2_2, y2_2 = get_coords(bbox2)

        # Compute bounding box IoU
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        if xi2 <= xi1 or yi2 <= yi1:
            iou = 0.0
        else:
            intersection = (xi2 - xi1) * (yi2 - yi1)
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection
            iou = intersection / union if union > 0 else 0.0

        # Compute center distance
        c1x = (x1_1 + x2_1) / 2
        c1y = (y1_1 + y2_1) / 2
        c2x = (x1_2 + x2_2) / 2
        c2y = (y1_2 + y2_2) / 2
        distance = ((c1x - c2x) ** 2 + (c1y - c2y) ** 2) ** 0.5

        # Check thresholds
        iou_match = iou >= self.match_thresh
        motion_match = distance <= 50.0  # 50 pixel threshold

        # Both must pass (AND logic)
        return iou_match and motion_match

    def reset(self) -> None:
        """Reset tracker state."""
        # Re-initialize the official tracker
        self._tracker = _ByteTrackTracker(
            lost_track_buffer=self.track_buffer,
            frame_rate=float(self.frame_rate),
            track_activation_threshold=self.track_thresh,
            minimum_consecutive_frames=2,
            minimum_iou_threshold=0.1,
            high_conf_det_threshold=0.6,
        )
        self._frame_count = 0
        self._label_map = {}
        logger.info("Tracker reset")


# Backward compatibility alias
ByteTrack = ByteTrackTracker
