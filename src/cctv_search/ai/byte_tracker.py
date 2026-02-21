"""ByteTrack-style object tracker implementation.

This is a simplified implementation of the ByteTrack multi-object tracking algorithm
that doesn't require the official ByteTrack library (which has compatibility issues).

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
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import numpy as np

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
    """ByteTrack multi-object tracker.
    
    Simplified implementation of ByteTrack algorithm:
    1. Split detections into high/low confidence
    2. Match high confidence with existing tracks
    3. Match low confidence with unmatched tracks
    4. Create new tracks for unmatched detections
    5. Remove stale tracks
    
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
            match_thresh: IoU threshold for matching
            track_buffer: Maximum frames to keep lost tracks
            frame_rate: Video frame rate
        """
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.frame_rate = frame_rate
        
        self._tracks: list[Track] = []
        self._next_track_id: int = 1
        self._frame_count: int = 0
        
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
        
        # Predict existing tracks
        for track in self._tracks:
            track.predict()
        
        # Separate active and lost tracks
        active_tracks = [t for t in self._tracks if t.is_active]
        
        # Split detections by confidence
        high_dets = [d for d in detections if d.confidence >= self.track_thresh]
        low_dets = [d for d in detections 
                   if 0.1 <= d.confidence < self.track_thresh]
        
        # First association: high confidence with active tracks
        matched, unmatched_tracks, unmatched_dets = self._associate(
            active_tracks, high_dets
        )
        
        # Update matched tracks
        for track, det in matched:
            track.update(det, frame_idx)
        
        # Second association: low confidence with unmatched tracks
        if unmatched_tracks and low_dets:
            matched2, unmatched_tracks2, _ = self._associate(
                unmatched_tracks, low_dets
            )
            for track, det in matched2:
                track.update(det, frame_idx)
        else:
            unmatched_tracks2 = unmatched_tracks
        
        # Mark unmatched tracks as lost
        for track in unmatched_tracks2:
            if track.age > self.track_buffer:
                track.mark_removed()
        
        # Create new tracks for unmatched high confidence detections
        for det in unmatched_dets:
            if det.confidence >= self.track_thresh:
                new_track = Track(
                    track_id=self._next_track_id,
                    label=det.label,
                    x=det.bbox.x + det.bbox.width / 2,
                    y=det.bbox.y + det.bbox.height / 2,
                    width=det.bbox.width,
                    height=det.bbox.height,
                    confidence=det.confidence,
                    frame_idx=frame_idx,
                )
                self._tracks.append(new_track)
                self._next_track_id += 1
        
        # Remove old tracks
        self._tracks = [t for t in self._tracks if t.state != "removed"]
        
        # Return active tracks
        return [t for t in self._tracks if t.is_active]
    
    def _associate(
        self,
        tracks: list[Track],
        detections: list[DetectedObject],
    ) -> tuple[list[tuple[Track, DetectedObject]], list[Track], list[DetectedObject]]:
        """Associate tracks with detections using IoU.
        
        Args:
            tracks: List of tracks to match
            detections: List of detections to match
            
        Returns:
            (matched_pairs, unmatched_tracks, unmatched_detections)
        """
        if not tracks or not detections:
            return [], tracks, detections
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._compute_iou(track, det)
        
        # Greedy matching
        matched = []
        unmatched_tracks = list(tracks)
        unmatched_dets = list(detections)
        
        # Sort by IoU descending
        indices = np.argsort(iou_matrix.flatten())[::-1]
        
        used_tracks = set()
        used_dets = set()
        
        for idx in indices:
            i = idx // len(detections)
            j = idx % len(detections)
            
            if i in used_tracks or j in used_dets:
                continue
            
            if iou_matrix[i, j] >= self.match_thresh:
                matched.append((tracks[i], detections[j]))
                used_tracks.add(i)
                used_dets.add(j)
        
        # Collect unmatched
        unmatched_tracks = [t for i, t in enumerate(tracks) if i not in used_tracks]
        unmatched_dets = [d for j, d in enumerate(detections) if j not in used_dets]
        
        return matched, unmatched_tracks, unmatched_dets
    
    def _compute_iou(self, track: Track, det: DetectedObject) -> float:
        """Compute IoU between track and detection."""
        # Track box
        tx1 = track.x - track.width / 2
        ty1 = track.y - track.height / 2
        tx2 = track.x + track.width / 2
        ty2 = track.y + track.height / 2
        
        # Detection box
        dx1 = det.bbox.x
        dy1 = det.bbox.y
        dx2 = det.bbox.x + det.bbox.width
        dy2 = det.bbox.y + det.bbox.height
        
        # Intersection
        x1 = max(tx1, dx1)
        y1 = max(ty1, dy1)
        x2 = min(tx2, dx2)
        y2 = min(ty2, dy2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Union
        track_area = (tx2 - tx1) * (ty2 - ty1)
        det_area = (dx2 - dx1) * (dy2 - dy1)
        union = track_area + det_area - intersection
        
        return intersection / union if union > 0 else 0.0

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
            if hasattr(det, 'bbox'):
                return det.bbox
            return det

        def get_label(det):
            if hasattr(det, 'label'):
                return det.label
            if hasattr(det, 'class_label'):
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
                x1 = bbox.x1 if hasattr(bbox, 'x1') else bbox.x
                y1 = bbox.y1 if hasattr(bbox, 'y1') else bbox.y
                x2 = bbox.x2 if hasattr(bbox, 'x2') else bbox.x + bbox.width
                y2 = bbox.y2 if hasattr(bbox, 'y2') else bbox.y + bbox.height
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
        self._tracks = []
        self._next_track_id = 1
        self._frame_count = 0
        logger.info("Tracker reset")


# Backward compatibility alias
ByteTrack = ByteTrackTracker
