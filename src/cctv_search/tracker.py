"""ByteTrack object tracker integration for CCTV search.

This module provides the integration layer between ByteTrack (from cctv_search.ai)
and the search algorithm. It converts ByteTrack tracks to the format expected
by the search algorithm.

ByteTrack Installation:
    pip install git+https://github.com/ifzhang/ByteTrack.git

Usage:
    from cctv_search.tracker import ByteTrackTracker, Detection
    
    tracker = ByteTrackTracker()
    
    # Update with detections
    result = tracker.update(detections, frame_idx=100, timestamp=5.0)
    
    # Check if same object
    is_same = tracker.is_same_object(det1, det2)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cctv_search.detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Represents an object track across multiple frames.
    
    This is a compatibility wrapper around ByteTrack's internal track format
    for use with the search algorithm.
    """
    
    track_id: int
    detections: list
    last_seen_frame: int = 0
    last_seen_timestamp: float = 0.0
    is_active: bool = True
    motion_vector: tuple[float, float] = (0.0, 0.0)
    
    @property
    def latest_detection(self):
        """Return the most recent detection in this track."""
        return self.detections[-1] if self.detections else None


@dataclass
class TrackerConfig:
    """Configuration for ByteTrack tracker.
    
    Attributes:
        track_thresh: Detection confidence threshold for tracking
        match_thresh: IoU threshold for matching
        track_buffer: Maximum frames to keep lost tracks
        frame_rate: Video frame rate
    """
    
    track_thresh: float = 0.5
    match_thresh: float = 0.8
    track_buffer: int = 30
    frame_rate: int = 20


class ByteTrackTracker:
    """ByteTrack tracker wrapper for CCTV search.
    
    This class wraps the ByteTrack multi-object tracking algorithm from
    cctv_search.ai and provides the interface expected by the search algorithm.
    
    ByteTrack uses Kalman filter and Hungarian algorithm for robust tracking
    of multiple objects across frames.
    
    Example:
        >>> tracker = ByteTrackTracker()
        >>> result = tracker.update(detections, frame_idx=100, timestamp=5.0)
        >>> for track in result.matched:
        ...     print(f"Track {track.track_id}: {len(track.detections)} detections")
    """
    
    def __init__(self, config: TrackerConfig | None = None):
        """Initialize ByteTrack tracker.
        
        Args:
            config: Tracker configuration
        """
        self.config = config or TrackerConfig()
        self._tracker = None
        self._tracks: dict[int, Track] = {}
        self._frame_count = 0
        
        self._load_tracker()
    
    def _load_tracker(self) -> None:
        """Load ByteTrack tracker from ai module."""
        try:
            from cctv_search.ai import ByteTrackTracker as AiByteTracker
            
            logger.info("Loading ByteTrack tracker...")
            self._tracker = AiByteTracker(
                track_thresh=self.config.track_thresh,
                match_thresh=self.config.match_thresh,
                track_buffer=self.config.track_buffer,
                frame_rate=self.config.frame_rate,
            )
            logger.info("ByteTrack tracker loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load ByteTrack: {e}")
            raise RuntimeError(f"Failed to load ByteTrack: {e}") from e
    
    def update(
        self,
        detections: list[Detection],
        frame_idx: int,
        timestamp: float,
    ) -> "AssociationResult":
        """Update tracker with new detections.
        
        Args:
            detections: List of detections in current frame
            frame_idx: Current frame index
            timestamp: Current timestamp in seconds
            
        Returns:
            AssociationResult with matched and unmatched tracks/detections
        """
        from cctv_search.ai import DetectedObject, BoundingBox
        
        self._frame_count = frame_idx
        
        # Convert detections to ai.DetectedObject format
        ai_detections = []
        for det in detections:
            ai_det = DetectedObject(
                label=det.class_label,
                bbox=BoundingBox(
                    x=det.bbox.x1,
                    y=det.bbox.y1,
                    width=det.bbox.x2 - det.bbox.x1,
                    height=det.bbox.y2 - det.bbox.y1,
                    confidence=det.confidence,
                ),
                confidence=det.confidence,
                frame_timestamp=timestamp,
            )
            ai_detections.append(ai_det)
        
        # Update ByteTrack
        tracks = self._tracker.update(ai_detections, frame_idx)
        
        # Convert to our Track format and build AssociationResult
        matched = []
        unmatched_tracks = []
        unmatched_detections = []
        
        # Build result
        for track in tracks:
            if track.is_active:
                # Find matching detection
                matching_det = None
                for det in detections:
                    det_x = det.bbox.x1 + (det.bbox.x2 - det.bbox.x1) / 2
                    det_y = det.bbox.y1 + (det.bbox.y2 - det.bbox.y1) / 2
                    
                    # Check if centers match (approximately)
                    if abs(det_x - track.x) < 5 and abs(det_y - track.y) < 5:
                        matching_det = det
                        break
                
                if matching_det:
                    # Update or create Track
                    if track.track_id not in self._tracks:
                        self._tracks[track.track_id] = Track(
                            track_id=track.track_id,
                            detections=[],
                        )
                    
                    self._tracks[track.track_id].detections.append(matching_det)
                    self._tracks[track.track_id].last_seen_frame = frame_idx
                    self._tracks[track.track_id].last_seen_timestamp = timestamp
                    self._tracks[track.track_id].is_active = True
                    
                    matched.append((self._tracks[track.track_id], matching_det))
        
        # Find unmatched detections
        matched_dets = {id(m[1]) for m in matched}
        for det in detections:
            if id(det) not in matched_dets:
                unmatched_detections.append(det)
        
        return AssociationResult(
            matched=matched,
            unmatched_tracks=unmatched_tracks,
            unmatched_detections=unmatched_detections,
        )
    
    def is_same_object(
        self,
        detection1: Detection,
        detection2: Detection,
    ) -> bool:
        """Check if two detections represent the same object instance.
        
        This is the core "SameBike" predicate using ByteTrack's association logic.
        Uses IoU and motion consistency for matching.
        
        Args:
            detection1: First detection
            detection2: Second detection
            
        Returns:
            True if detections likely represent the same physical object
        """
        # Different classes can never be the same object
        if detection1.class_label != detection2.class_label:
            return False
        
        # Compute bounding box IoU
        def compute_iou(box1, box2):
            x1 = max(box1.x1, box2.x1)
            y1 = max(box1.y1, box2.y1)
            x2 = min(box1.x2, box2.x2)
            y2 = min(box1.y2, box2.y2)
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
            
            intersection = (x2 - x1) * (y2 - y1)
            area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
            area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        # Compute center distance
        def center_distance(box1, box2):
            c1x = (box1.x1 + box1.x2) / 2
            c1y = (box1.y1 + box1.y2) / 2
            c2x = (box2.x1 + box2.x2) / 2
            c2y = (box2.y1 + box2.y2) / 2
            return ((c1x - c2x) ** 2 + (c1y - c2y) ** 2) ** 0.5
        
        # Check IoU
        iou = compute_iou(detection1.bbox, detection2.bbox)
        iou_match = iou >= self.config.match_thresh
        
        # Check motion consistency (distance)
        distance = center_distance(detection1.bbox, detection2.bbox)
        motion_match = distance <= 50.0  # 50 pixel threshold
        
        # Both must pass (AND logic)
        return iou_match and motion_match
    
    def reset(self) -> None:
        """Reset tracker state."""
        self._load_tracker()
        self._tracks = {}
        self._frame_count = 0
        logger.info("Tracker reset")


@dataclass
class AssociationResult:
    """Result of associating detections with existing tracks."""
    
    matched: list[tuple[Track, Detection]]
    unmatched_tracks: list[Track]
    unmatched_detections: list[Detection]
    
    def __init__(
        self,
        matched: list | None = None,
        unmatched_tracks: list | None = None,
        unmatched_detections: list | None = None,
    ):
        self.matched = matched or []
        self.unmatched_tracks = unmatched_tracks or []
        self.unmatched_detections = unmatched_detections or []


# Backward compatibility
MockByteTrackTracker = ByteTrackTracker
