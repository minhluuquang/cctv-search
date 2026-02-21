"""Mock ByteTrack-style tracker for CCTV search.

This module provides a mock/stub implementation of ByteTrack object tracking
for testing and development purposes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from cctv_search.detector import BoundingBox, Detection


@dataclass
class Track:
    """Represents an object track across multiple frames.

    Attributes:
        track_id: Unique identifier for this track
        detections: List of detections belonging to this track
        last_seen_frame: Frame index where track was last updated
        last_seen_timestamp: Timestamp where track was last updated
        is_active: Whether the track is currently active
        motion_vector: Estimated motion (dx, dy) between frames
    """

    track_id: int
    detections: list[Detection] = field(default_factory=list)
    last_seen_frame: int = 0
    last_seen_timestamp: float = 0.0
    is_active: bool = True
    motion_vector: tuple[float, float] = (0.0, 0.0)

    @property
    def latest_detection(self) -> Detection | None:
        """Return the most recent detection in this track."""
        return self.detections[-1] if self.detections else None

    @property
    def first_detection(self) -> Detection | None:
        """Return the first detection in this track."""
        return self.detections[0] if self.detections else None

    @property
    def duration_frames(self) -> int:
        """Return track duration in frames."""
        if len(self.detections) < 2:
            return 1
        return self.detections[-1].frame_idx - self.detections[0].frame_idx + 1

    @property
    def duration_seconds(self) -> float:
        """Return track duration in seconds."""
        if len(self.detections) < 2:
            return 0.0
        return self.detections[-1].timestamp - self.detections[0].timestamp

    def predict_position(self, frame_idx: int) -> BoundingBox:
        """Predict bounding box position at a future frame.

        Uses linear motion model to extrapolate position.

        Args:
            frame_idx: Target frame index

        Returns:
            Predicted bounding box
        """
        if not self.latest_detection:
            raise ValueError("Cannot predict position for empty track")

        frames_diff = frame_idx - self.last_seen_frame
        dx = self.motion_vector[0] * frames_diff
        dy = self.motion_vector[1] * frames_diff

        last_bbox = self.latest_detection.bbox
        return BoundingBox(
            x1=last_bbox.x1 + dx,
            y1=last_bbox.y1 + dy,
            x2=last_bbox.x2 + dx,
            y2=last_bbox.y2 + dy,
        )

    def update(self, detection: Detection) -> None:
        """Update track with a new detection.

        Args:
            detection: New detection to add to track
        """
        self.detections.append(detection)

        # Update motion vector
        if len(self.detections) >= 2:
            prev = self.detections[-2].bbox.center
            curr = detection.bbox.center
            frames_diff = detection.frame_idx - self.last_seen_frame

            if frames_diff > 0:
                self.motion_vector = (
                    (curr[0] - prev[0]) / frames_diff,
                    (curr[1] - prev[1]) / frames_diff,
                )

        self.last_seen_frame = detection.frame_idx
        self.last_seen_timestamp = detection.timestamp
        self.is_active = True

    def mark_lost(self) -> None:
        """Mark track as lost/inactive."""
        self.is_active = False


@dataclass
class AssociationResult:
    """Result of associating detections with existing tracks.

    Attributes:
        matched: List of (track, detection) pairs that were matched
        unmatched_tracks: Tracks that were not matched to any detection
        unmatched_detections: Detections that were not matched to any track
    """

    matched: list[tuple[Track, Detection]] = field(default_factory=list)
    unmatched_tracks: list[Track] = field(default_factory=list)
    unmatched_detections: list[Detection] = field(default_factory=list)


@dataclass
class TrackerConfig:
    """Configuration for tracker behavior.

    Attributes:
        mask_iou_threshold: Minimum IoU for mask-based matching
        motion_threshold: Maximum distance (pixels) for motion consistency
        max_lost_frames: Maximum frames a track can be lost before deletion
        min_hits: Minimum detections required to confirm a track
        custom_association: Optional callback for custom association logic
    """

    mask_iou_threshold: float = 0.5
    motion_threshold: float = 50.0
    max_lost_frames: int = 30
    min_hits: int = 3
    custom_association: (
        Callable[[list[Track], list[Detection]], AssociationResult] | None
    ) = None


class MockByteTrackTracker:
    """Mock ByteTrack-style tracker for testing and development.

    This class simulates the ByteTrack multi-object tracking algorithm
    with configurable behavior for testing purposes.

    The tracker maintains object identity across frames using:
    - Mask IoU for spatial matching
    - Motion consistency checks
    - Track lifecycle management (active/lost/deleted)

    Example:
        >>> config = TrackerConfig(mask_iou_threshold=0.5)
        >>> tracker = MockByteTrackTracker(config)
        >>> result = tracker.update(detections, frame_idx=100, timestamp=5.0)
    """

    def __init__(self, config: TrackerConfig | None = None):
        """Initialize the mock tracker.

        Args:
            config: Configuration for tracker behavior
        """
        self.config = config or TrackerConfig()
        self._tracks: list[Track] = []
        self._next_track_id: int = 1
        self._frame_count: int = 0

    @property
    def active_tracks(self) -> list[Track]:
        """Return list of currently active tracks."""
        return [t for t in self._tracks if t.is_active]

    @property
    def all_tracks(self) -> list[Track]:
        """Return all tracks (active and inactive)."""
        return self._tracks.copy()

    def reset(self) -> None:
        """Reset tracker state (clear all tracks)."""
        self._tracks = []
        self._next_track_id = 1
        self._frame_count = 0

    def update(
        self,
        detections: list[Detection],
        frame_idx: int,
        timestamp: float,
    ) -> AssociationResult:
        """Update tracker with new detections.

        Args:
            detections: List of detections in current frame
            frame_idx: Current frame index
            timestamp: Current timestamp in seconds

        Returns:
            AssociationResult with matched and unmatched tracks/detections
        """
        self._frame_count = frame_idx

        if self.config.custom_association:
            result = self.config.custom_association(self.active_tracks, detections)
        else:
            result = self._associate_detections(self.active_tracks, detections)

        # Update matched tracks
        for track, detection in result.matched:
            track.update(detection)

        # Mark unmatched tracks as lost
        for track in result.unmatched_tracks:
            track.mark_lost()

        # Create new tracks for unmatched detections
        for detection in result.unmatched_detections:
            new_track = Track(
                track_id=self._next_track_id,
                detections=[detection],
                last_seen_frame=frame_idx,
                last_seen_timestamp=timestamp,
                is_active=True,
            )
            self._tracks.append(new_track)
            self._next_track_id += 1

        # Remove stale tracks
        self._cleanup_tracks()

        return result

    def _associate_detections(
        self,
        tracks: list[Track],
        detections: list[Detection],
    ) -> AssociationResult:
        """Associate detections with existing tracks.

        Uses mask IoU and motion consistency for matching.

        Args:
            tracks: Active tracks to match against
            detections: New detections to associate

        Returns:
            AssociationResult with matching results
        """
        matched: list[tuple[Track, Detection]] = []
        unmatched_tracks: list[Track] = []
        unmatched_detections: list[Detection] = list(detections)

        # Build cost matrix based on mask IoU and motion
        track_indices: list[int] = []
        detection_indices: list[int] = []
        scores: list[float] = []

        for i, track in enumerate(tracks):
            if not track.latest_detection:
                continue

            for j, detection in enumerate(detections):
                score = self._compute_association_score(track, detection)
                if score > 0:
                    track_indices.append(i)
                    detection_indices.append(j)
                    scores.append(score)

        # Greedy matching (simplified ByteTrack logic)
        used_tracks: set[int] = set()
        used_detections: set[int] = set()

        # Sort by score descending
        sorted_pairs = sorted(
            zip(track_indices, detection_indices, scores, strict=False),
            key=lambda x: x[2],
            reverse=True,
        )

        for track_idx, det_idx, score in sorted_pairs:
            if track_idx in used_tracks or det_idx in used_detections:
                continue

            if score >= self.config.mask_iou_threshold:
                matched.append((tracks[track_idx], detections[det_idx]))
                used_tracks.add(track_idx)
                used_detections.add(det_idx)

        # Collect unmatched
        for i, track in enumerate(tracks):
            if i not in used_tracks:
                unmatched_tracks.append(track)

        unmatched_detections = [
            d for j, d in enumerate(detections) if j not in used_detections
        ]

        return AssociationResult(
            matched=matched,
            unmatched_tracks=unmatched_tracks,
            unmatched_detections=unmatched_detections,
        )

    def _compute_association_score(
        self,
        track: Track,
        detection: Detection,
    ) -> float:
        """Compute association score between track and detection.

        Combines mask IoU and motion consistency.

        Args:
            track: Existing track
            detection: New detection to score

        Returns:
            Association score (higher is better)
        """
        if not track.latest_detection:
            return 0.0

        # Check motion consistency first
        predicted_bbox = track.predict_position(detection.frame_idx)
        pred_center = predicted_bbox.center
        det_center = detection.bbox.center
        distance = (
            (pred_center[0] - det_center[0]) ** 2
            + (pred_center[1] - det_center[1]) ** 2
        ) ** 0.5

        if distance > self.config.motion_threshold:
            return 0.0

        # Compute mask IoU
        mask_iou = track.latest_detection.mask_iou(detection)

        # Combined score (weighted average)
        motion_score = max(0, 1 - distance / self.config.motion_threshold)
        combined_score = 0.7 * mask_iou + 0.3 * motion_score

        return combined_score

    def _cleanup_tracks(self) -> None:
        """Remove tracks that have been lost for too long."""
        self._tracks = [
            t
            for t in self._tracks
            if t.is_active
            or (self._frame_count - t.last_seen_frame) <= self.config.max_lost_frames
        ]

    def find_track_by_id(self, track_id: int) -> Track | None:
        """Find a track by its ID.

        Args:
            track_id: Track ID to search for

        Returns:
            Track if found, None otherwise
        """
        for track in self._tracks:
            if track.track_id == track_id:
                return track
        return None

    def get_track_history(
        self,
        track_id: int,
        start_frame: int | None = None,
        end_frame: int | None = None,
    ) -> list[Detection]:
        """Get detection history for a specific track.

        Args:
            track_id: Track ID
            start_frame: Optional start frame (inclusive)
            end_frame: Optional end frame (inclusive)

        Returns:
            List of detections within the specified range
        """
        track = self.find_track_by_id(track_id)
        if not track:
            return []

        detections = track.detections
        if start_frame is not None:
            detections = [d for d in detections if d.frame_idx >= start_frame]
        if end_frame is not None:
            detections = [d for d in detections if d.frame_idx <= end_frame]

        return detections

    def is_same_object(
        self,
        detection1: Detection,
        detection2: Detection,
    ) -> bool:
        """Check if two detections represent the same object instance.

        This is the core "SameBike" predicate from the problem description.
        Both mask IoU AND motion consistency must pass for a match.

        Args:
            detection1: First detection
            detection2: Second detection

        Returns:
            True if detections likely represent the same physical object
        """
        # Different classes can never be the same object
        if detection1.class_label != detection2.class_label:
            return False

        # Check mask IoU (must pass)
        mask_iou = detection1.mask_iou(detection2)
        iou_match = mask_iou >= self.config.mask_iou_threshold

        # Check motion consistency (must pass)
        distance = detection1.center_distance(detection2)
        motion_match = distance <= self.config.motion_threshold

        # Both must pass for a match (AND logic)
        return iou_match and motion_match
