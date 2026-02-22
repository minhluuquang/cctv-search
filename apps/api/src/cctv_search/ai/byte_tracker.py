"""Feature-based object tracker using RF-DETR deep embeddings.

This module provides tracking using deep feature embeddings from RF-DETR
to match objects across frames.

This approach is:
- More robust to occlusion (features persist when bbox changes)
- Simpler (no Kalman filters or motion models)
- Consistent with detection (uses same features)

Usage:
    from cctv_search.ai.tracker import FeatureTracker, Track
    
    tracker = FeatureTracker(detector=rf_detr_detector)
    tracks = tracker.update(detections, frame_idx=100)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from cctv_search.ai import DetectedObject, RFDetrDetector

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Represents a tracked object across frames.

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
        features: Deep feature embedding for matching
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
    features: np.ndarray | None = field(default=None, repr=False)

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
        self.features = detection.features
        if self.hits >= 2:  # Activate after 2 hits
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


class FeatureTracker:
    """Feature-based object tracker using RF-DETR embeddings.

    Matches objects across frames using deep feature similarity from RF-DETR
    instead of motion-based tracking. This handles occlusion better because
    features persist even when objects overlap or are partially hidden.

    Example:
        >>> tracker = FeatureTracker(detector=rf_detr_detector)
        >>> detections = detector.detect_with_features(frame)
        >>> tracks = tracker.update(detections, frame_idx=100)
    """

    def __init__(
        self,
        detector: RFDetrDetector | None = None,
        feature_threshold: float = 0.75,
        iou_threshold: float = 0.5,
        distance_threshold: float = 100.0,
        max_age: int = 30,
    ):
        """Initialize feature-based tracker.

        Args:
            detector: RF-DETR detector for feature extraction (optional)
            feature_threshold: Minimum cosine similarity for feature matching
            iou_threshold: Minimum IoU for spatial matching
            distance_threshold: Maximum center distance in pixels
            max_age: Maximum frames to keep lost tracks
        """
        self.detector = detector
        self.feature_threshold = feature_threshold
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.max_age = max_age

        self._tracks: dict[int, Track] = {}
        self._next_track_id: int = 1
        self._frame_count: int = 0

        logger.info(
            f"FeatureTracker initialized "
            f"(feature_thresh={feature_threshold}, iou_thresh={iou_threshold})"
        )

    def update(
        self,
        detections: list[DetectedObject],
        frame_idx: int,
    ) -> list[Track]:
        """Update tracker with new detections.

        Matches detections to existing tracks using:
        1. Feature similarity (primary, handles occlusion)
        2. IoU overlap (secondary, spatial consistency)

        Args:
            detections: List of detections with features from RF-DETR
            frame_idx: Current frame index

        Returns:
            List of active tracks
        """
        self._frame_count = frame_idx

        # Age all existing tracks
        for track in self._tracks.values():
            if track.is_active:
                track.predict()

        # Match detections to tracks
        matched_tracks: list[tuple[Track, DetectedObject]] = []
        unmatched_detections: list[DetectedObject] = []

        # Get active tracks
        active_tracks = [t for t in self._tracks.values() if t.is_active]

        # Match each detection to best track
        detection_matched = [False] * len(detections)

        for det_idx, detection in enumerate(detections):
            best_track = None
            best_score = -1.0

            for track in active_tracks:
                # Skip if already matched this iteration
                if any(t.track_id == track.track_id for t, _ in matched_tracks):
                    continue

                # Check class consistency
                if track.label != detection.label:
                    continue

                # Compute matching score
                score = self._compute_match_score(track, detection)

                if score > best_score and score > 0.5:  # Minimum threshold
                    best_score = score
                    best_track = track

            if best_track is not None:
                matched_tracks.append((best_track, detection))
                detection_matched[det_idx] = True
            else:
                unmatched_detections.append(detection)

        # Update matched tracks
        for track, detection in matched_tracks:
            track.update(detection, frame_idx)

        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            new_track = Track(
                track_id=self._next_track_id,
                label=detection.label,
                x=detection.bbox.x + detection.bbox.width / 2,
                y=detection.bbox.y + detection.bbox.height / 2,
                width=detection.bbox.width,
                height=detection.bbox.height,
                confidence=detection.confidence,
                frame_idx=frame_idx,
                is_active=True,
                is_activated=False,
                features=detection.features,
            )
            self._tracks[self._next_track_id] = new_track
            self._next_track_id += 1

        # Remove old tracks
        tracks_to_remove = []
        for track_id, track in self._tracks.items():
            if track.age > self.max_age:
                track.mark_removed()
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self._tracks[track_id]

        # Return active tracks
        return [t for t in self._tracks.values() if t.is_active]

    def _compute_match_score(self, track: Track, detection: DetectedObject) -> float:
        """Compute matching score between track and detection.

        Combines feature similarity, IoU, and distance into single score.

        Returns:
            Score between 0 and 1, higher = better match
        """
        scores = []

        # Feature similarity (most important for occlusion)
        if track.features is not None and detection.has_features():
            feat_sim = self._compute_feature_similarity(track.features, detection.features)
            scores.append(feat_sim * 0.6)  # 60% weight

        # IoU
        iou = self._compute_iou(track, detection)
        scores.append(iou * 0.25)  # 25% weight

        # Distance
        distance = self._compute_distance(track, detection)
        distance_score = max(0, 1 - distance / self.distance_threshold)
        scores.append(distance_score * 0.15)  # 15% weight

        return sum(scores) if scores else 0.0

    def _compute_feature_similarity(
        self, feat1: np.ndarray, feat2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two feature vectors."""
        feat1 = feat1.flatten()
        feat2 = feat2.flatten()
        return float(
            np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-8)
        )

    def _compute_iou(self, track: Track, detection: DetectedObject) -> float:
        """Compute IoU between track and detection bounding boxes."""
        x1_t = track.x - track.width / 2
        y1_t = track.y - track.height / 2
        x2_t = track.x + track.width / 2
        y2_t = track.y + track.height / 2

        x1_d = detection.bbox.x
        y1_d = detection.bbox.y
        x2_d = detection.bbox.x + detection.bbox.width
        y2_d = detection.bbox.y + detection.bbox.height

        xi1 = max(x1_t, x1_d)
        yi1 = max(y1_t, y1_d)
        xi2 = min(x2_t, x2_d)
        yi2 = min(y2_t, y2_d)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        intersection = (xi2 - xi1) * (yi2 - yi1)
        area_t = track.width * track.height
        area_d = detection.bbox.width * detection.bbox.height
        union = area_t + area_d - intersection

        return intersection / union if union > 0 else 0.0

    def _compute_distance(self, track: Track, detection: DetectedObject) -> float:
        """Compute center distance between track and detection."""
        det_x = detection.bbox.x + detection.bbox.width / 2
        det_y = detection.bbox.y + detection.bbox.height / 2
        return ((track.x - det_x) ** 2 + (track.y - det_y) ** 2) ** 0.5

    def is_same_object(
        self,
        detection1: Any,
        detection2: Any,
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
        from cctv_search.ai import DetectedObject

        # Handle different input types
        if not isinstance(detection1, DetectedObject):
            detection1 = self._convert_to_detected_object(detection1)
        if not isinstance(detection2, DetectedObject):
            detection2 = self._convert_to_detected_object(detection2)

        # Different classes can never be the same object
        if detection1.label != detection2.label:
            return False

        # Try feature matching first (best for occlusion)
        if detection1.has_features() and detection2.has_features():
            similarity = self._compute_feature_similarity(
                detection1.features, detection2.features
            )
            if similarity >= self.feature_threshold:
                return True

        # Fall back to IoU + distance
        iou = self._compute_iou_between_detections(detection1, detection2)
        distance = self._compute_distance_between_detections(detection1, detection2)

        return iou >= self.iou_threshold and distance <= self.distance_threshold

    def _convert_to_detected_object(self, obj: Any) -> DetectedObject:
        """Convert various detection formats to DetectedObject."""
        from cctv_search.ai import BoundingBox, DetectedObject

        if hasattr(obj, "bbox"):
            bbox = obj.bbox
            if not hasattr(bbox, "width"):
                # Convert x1,y1,x2,y2 to x,y,width,height
                bbox = BoundingBox(
                    x=getattr(bbox, "x", getattr(bbox, "x1", 0)),
                    y=getattr(bbox, "y", getattr(bbox, "y1", 0)),
                    width=getattr(bbox, "width", getattr(bbox, "x2", 0) - getattr(bbox, "x1", 0)),
                    height=getattr(bbox, "height", getattr(bbox, "y2", 0) - getattr(bbox, "y1", 0)),
                    confidence=getattr(bbox, "confidence", 0.5),
                )
            return DetectedObject(
                label=getattr(obj, "label", getattr(obj, "class_label", "unknown")),
                bbox=bbox,
                confidence=getattr(obj, "confidence", 0.5),
                frame_timestamp=getattr(obj, "frame_timestamp", 0.0),
                features=getattr(obj, "features", None),
            )
        else:
            raise ValueError(f"Cannot convert {type(obj)} to DetectedObject")

    def _compute_iou_between_detections(
        self, det1: DetectedObject, det2: DetectedObject
    ) -> float:
        """Compute IoU between two detections (handles both xywh and xyxy formats)."""
        # Handle different bbox formats
        def get_coords(bbox):
            if hasattr(bbox, 'x1') and hasattr(bbox, 'x2'):
                # xyxy format (x1, y1, x2, y2) - from detector module
                return bbox.x1, bbox.y1, bbox.x2, bbox.y2
            elif hasattr(bbox, 'x') and hasattr(bbox, 'width'):
                # xywh format - from ai module
                return bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height
            else:
                raise ValueError(f"Unknown bbox format: {type(bbox)}")
        
        x1_1, y1_1, x2_1, y2_1 = get_coords(det1.bbox)
        x1_2, y1_2, x2_2, y2_2 = get_coords(det2.bbox)

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        intersection = (xi2 - xi1) * (yi2 - yi1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _compute_distance_between_detections(
        self, det1: DetectedObject, det2: DetectedObject
    ) -> float:
        """Compute center distance between two detections (handles both bbox formats)."""
        def get_center(bbox):
            if hasattr(bbox, 'x1') and hasattr(bbox, 'x2'):
                # xyxy format (x1, y1, x2, y2) - from detector module
                return (bbox.x1 + bbox.x2) / 2, (bbox.y1 + bbox.y2) / 2
            elif hasattr(bbox, 'x') and hasattr(bbox, 'width'):
                # xywh format - from ai module
                return bbox.x + bbox.width / 2, bbox.y + bbox.height / 2
            else:
                raise ValueError(f"Unknown bbox format: {type(bbox)}")
        
        c1x, c1y = get_center(det1.bbox)
        c2x, c2y = get_center(det2.bbox)
        return ((c1x - c2x) ** 2 + (c1y - c2y) ** 2) ** 0.5

    def reset(self) -> None:
        """Reset tracker state."""
        self._tracks.clear()
        self._next_track_id = 1
        self._frame_count = 0
        logger.info("FeatureTracker reset")



Track = Track
