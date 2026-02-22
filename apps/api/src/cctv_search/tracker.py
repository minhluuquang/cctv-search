"""Feature-based object tracker integration for CCTV search.

This module provides the integration layer between the feature-based tracker
(from cctv_search.ai) and the search algorithm.

The tracker uses RF-DETR deep features for robust object matching,
especially during occlusion scenarios.

Usage:
    from cctv_search.tracker import FeatureTracker, Detection
    
    tracker = FeatureTracker()
    
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
    
    This is a compatibility wrapper for use with the search algorithm.
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
    """Configuration for feature-based tracker.
    
    Attributes:
        feature_threshold: Minimum cosine similarity for feature matching
        iou_threshold: Minimum IoU for spatial matching
        max_age: Maximum frames to keep lost tracks
    """
    
    feature_threshold: float = 0.75
    iou_threshold: float = 0.5
    max_age: int = 30


class FeatureTracker:
    """Feature-based tracker wrapper for CCTV search.
    
    This class wraps the feature-based tracker from cctv_search.ai
    and provides the interface expected by the search algorithm.
    
    Uses RF-DETR deep features for robust object matching across frames,
    especially during occlusion.
    
    Example:
        >>> tracker = FeatureTracker()
        >>> result = tracker.update(detections, frame_idx=100, timestamp=5.0)
        >>> for track in result.matched:
        ...     print(f"Track {track.track_id}: {len(track.detections)} detections")
    """
    
    def __init__(self, config: TrackerConfig | None = None):
        """Initialize feature-based tracker.
        
        Args:
            config: Tracker configuration
        """
        self.config = config or TrackerConfig()
        self._tracker = None
        self._tracks: dict[int, Track] = {}
        self._frame_count = 0
        
        self._load_tracker()
    
    def _load_tracker(self) -> None:
        """Load feature-based tracker from ai module."""
        try:
            from cctv_search.ai import FeatureTracker as AiFeatureTracker
            
            logger.info("Loading FeatureTracker...")
            self._tracker = AiFeatureTracker(
                feature_threshold=self.config.feature_threshold,
                iou_threshold=self.config.iou_threshold,
                max_age=self.config.max_age,
            )
            logger.info("FeatureTracker loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load FeatureTracker: {e}")
            raise RuntimeError(f"Failed to load FeatureTracker: {e}") from e
    
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
        from cctv_search.ai import BoundingBox, DetectedObject
        
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
                features=getattr(det, "features", None),
            )
            ai_detections.append(ai_det)
        
        # Update tracker
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
        
        Uses feature similarity as primary criterion, falling back to
        IoU and distance if features unavailable.
        
        Args:
            detection1: First detection
            detection2: Second detection
            
        Returns:
            True if detections likely represent the same physical object
        """
        # Delegate to the ai tracker
        return self._tracker.is_same_object(detection1, detection2)
    
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



