"""Backward coarse-to-fine temporal search for fixed CCTV cameras.

This module implements the optimized frame-based search algorithm
for finding object instances in fixed CCTV camera footage.

Key optimizations:
- Frame-level binary search (not time-based)
- Adaptive sampling (coarse → medium → fine)
- ~20 detector calls vs ~2160 for naive approach
- Works with 20 FPS cameras efficiently
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Default search parameters
DEFAULT_FPS = 20.0
DEFAULT_MIN_WINDOW = 5  # 5 seconds precision
DEFAULT_MAX_LOOKBACK = 3 * 60 * 60  # 3 hours


class SearchStatus(Enum):
    """Status of the search operation."""

    SUCCESS = auto()
    NOT_FOUND = auto()
    IN_PROGRESS = auto()
    ERROR = auto()


@dataclass(frozen=True)
class Point:
    """2D point representing a position in image space."""

    x: float
    y: float

    def distance_to(self, other: Point) -> float:
        """Calculate Euclidean distance to another point."""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box for detected objects."""

    x: float
    y: float
    width: float
    height: float
    confidence: float

    @property
    def center(self) -> Point:
        """Get the center point of the bounding box."""
        return Point(
            x=self.x + self.width / 2,
            y=self.y + self.height / 2,
        )

    @property
    def area(self) -> float:
        """Get the area of the bounding box."""
        return self.width * self.height

    def iou_with(self, other: BoundingBox) -> float:
        """Calculate IoU (Intersection over Union) with another box."""
        x_left = max(self.x, other.x)
        y_top = max(self.y, other.y)
        x_right = min(self.x + self.width, other.x + other.width)
        y_bottom = min(self.y + self.height, other.y + other.height)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = self.area + other.area - intersection

        return intersection / union if union > 0 else 0.0


@dataclass
class SegmentationMask:
    """Binary segmentation mask for an object."""

    mask: list[list[bool]]  # 2D boolean array
    width: int
    height: int

    def iou_with(self, other: SegmentationMask) -> float:
        """Calculate mask IoU with another mask."""
        if self.width != other.width or self.height != other.height:
            logger.warning("Mask dimensions mismatch, returning 0 IoU")
            return 0.0

        intersection = 0
        union = 0

        for y in range(self.height):
            for x in range(self.width):
                if self.mask[y][x] and other.mask[y][x]:
                    intersection += 1
                if self.mask[y][x] or other.mask[y][x]:
                    union += 1

        return intersection / union if union > 0 else 0.0


@dataclass
class ObjectDetection:
    """A detected object with bbox and optional mask."""

    label: str
    bbox: BoundingBox
    mask: SegmentationMask | None = None
    confidence: float = 0.0

    @property
    def center(self) -> Point:
        """Get center point of the detection."""
        return self.bbox.center


@dataclass
class ObjectTrack:
    """Track of an object across frames."""

    track_id: str
    label: str
    detections: list[ObjectDetection]
    first_seen: datetime
    last_seen: datetime


@dataclass
class SearchResult:
    """Result of a search operation."""

    status: SearchStatus
    timestamp: float | None = None
    precision_seconds: float | None = None
    confidence: float | None = None
    track: ObjectTrack | None = None
    iterations: int = 0
    message: str = ""

    @property
    def found(self) -> bool:
        """Return True if search was successful."""
        return self.status == SearchStatus.SUCCESS


class VideoDecoder(Protocol):
    """Protocol for video decoder."""

    def get_frame(self, timestamp: float) -> bytes | None:
        """Get frame at timestamp.

        Returns frame data or None if unavailable.
        """
        ...

    def get_frame_by_index(self, frame_index: int) -> bytes | None:
        """Get frame by index (for frame-level seeking).

        Args:
            frame_index: Frame number (0 = first frame)

        Returns:
            Frame data or None if unavailable.
        """
        ...

    def timestamp_to_frame(self, timestamp: float) -> int:
        """Convert timestamp to frame index."""
        ...

    def frame_to_timestamp(self, frame_index: int) -> float:
        """Convert frame index to timestamp."""
        ...


class ObjectDetector(Protocol):
    """Protocol for object detector (e.g., RF-DETR)."""

    def detect(self, frame: bytes) -> list[ObjectDetection]:
        """Run detection on frame.

        Returns list of detections with bounding boxes and masks.
        """
        ...


class ObjectTracker(Protocol):
    """Protocol for multi-object tracking (e.g., ByteTrack)."""

    def update(self, detections: list[ObjectDetection]) -> list[ObjectTrack]:
        """Update tracks with new detections.

        Returns updated track list with consistent track IDs.
        """
        ...

    def is_same_object(
        self,
        detection1: ObjectDetection,
        detection2: ObjectDetection,
    ) -> bool:
        """Check if two detections represent the same object instance."""
        ...

    def reset(self) -> None:
        """Reset tracker state."""
        ...


class SearchConfigError(Exception):
    """Error raised when search configuration is invalid."""

    pass


def backward_search(
    target_time: float,
    target_object: Any,
    video_source: Any,
    detector: Any,
    tracker: Any,
    config: dict,
    fps: float = 20.0,
) -> SearchResult:
    """Backward frame-based search with adaptive sampling.

    Optimized for 20 FPS cameras using:
    1. Phase 1: Coarse sampling (30 second steps)
    2. Phase 2: Medium sampling (5 second steps)
    3. Phase 3: Frame-level binary search (exact frame)

    Args:
        target_time: Unix timestamp where target object is visible
        target_object: The target object to search for
        video_source: Video source protocol implementation
        detector: Object detector protocol implementation
        tracker: Object tracker protocol implementation
        config: Search configuration dictionary
        fps: Camera frame rate (default 20 FPS)

    Returns:
        SearchResult with status and timestamp if found
    """
    # Validate configuration
    min_window = config.get("min_window", DEFAULT_MIN_WINDOW)
    max_lookback = config.get("max_lookback", DEFAULT_MAX_LOOKBACK)

    if min_window <= 0:
        raise SearchConfigError("min_window must be positive")
    if max_lookback <= 0:
        raise SearchConfigError("max_lookback must be positive")

    # Convert to frames
    fps_float = float(fps)
    seconds_to_frames = int(fps_float)

    # Step sizes in frames
    coarse_step = 30 * seconds_to_frames  # 30 seconds = 600 frames @ 20fps
    medium_step = 5 * seconds_to_frames   # 5 seconds = 100 frames @ 20fps

    # Calculate frame indices
    frame_at_target = video_source.timestamp_to_frame(target_time)
    max_frames_back = int(max_lookback * fps_float)
    frame_limit = max(0, frame_at_target - max_frames_back)

    iterations = 0
    target_label = getattr(target_object, "label", None)

    logger.info(
        f"Starting frame-based search from frame {frame_at_target} "
        f"(T={target_time}) with target: {target_label}"
    )

    def check_frame(frame_idx: int) -> bool:
        """Check if object exists at specific frame index."""
        nonlocal iterations
        iterations += 1

        # Get frame
        frame = video_source.get_frame_by_index(frame_idx)
        if frame is None:
            return False

        # Run detection
        detections = detector.detect(frame)
        matching = [d for d in detections if getattr(d, "label", None) == target_label]

        # Check if it's the same object
        if matching and tracker is not None:
            for candidate in matching:
                if tracker.is_same_object(target_object, candidate):
                    return True

        return False

    # === PHASE 1: Coarse Sampling (30 second steps) ===
    logger.debug("Phase 1: Coarse sampling (30 second steps)")

    step = coarse_step
    current_frame = frame_at_target
    last_known_frame = frame_at_target
    found_any = False

    # Check if object at T
    if check_frame(frame_at_target):
        found_any = True
        last_known_frame = frame_at_target

    # Coarse search backward - always search full range even if not at T
    while step > 0 and current_frame - step >= frame_limit:
        check_idx = current_frame - step

        if check_frame(check_idx):
            # Object found here
            if not found_any:
                # First time finding the object - this is our starting point
                found_any = True
                last_known_frame = check_idx
                current_frame = check_idx
                step = coarse_step  # Reset to standard step for fine search
            else:
                # Object still here from previous search
                last_known_frame = check_idx
                current_frame = check_idx
                step *= 2  # Accelerate backward
        else:
            if found_any:
                # Found gap after having found object!
                logger.debug(
                    f"Phase 1: Gap found between frames {check_idx} "
                    f"and {last_known_frame}"
                )
                break
            else:
                # Haven't found object yet, continue searching
                current_frame = check_idx
                # Keep same step size to continue searching

        # Safety check
        if step > max_frames_back:
            step = max_frames_back

    # If we never found the object, return NOT_FOUND
    if not found_any:
        logger.info("Object not found in coarse search")
        return SearchResult(
            status=SearchStatus.NOT_FOUND,
            iterations=iterations,
            message="Object not found",
        )

    # If we reached the limit without finding a gap
    if current_frame - step < frame_limit:
        # Object exists all the way to the limit
        result_time = video_source.frame_to_timestamp(frame_limit)
        logger.info(f"Object exists to limit, returning frame {frame_limit}")
        return SearchResult(
            status=SearchStatus.SUCCESS,
            timestamp=result_time,
            precision_seconds=min_window,
            confidence=getattr(target_object, "confidence", None),
            iterations=iterations,
            message=f"Object found at frame {frame_limit}",
        )

    # === PHASE 2: Medium Sampling (5 second steps) ===
    logger.debug("Phase 2: Medium sampling (5 second steps)")

    # Search range from Phase 1
    gap_start = current_frame - step  # No object here
    gap_end = last_known_frame         # Object here

    # Scan backward with 5-second steps
    check_idx = gap_end
    while check_idx > gap_start:
        check_idx -= medium_step

        if check_idx < gap_start:
            check_idx = gap_start

        if check_frame(check_idx):
            # Object still here, continue
            pass
        else:
            # Found the gap at this level
            gap_start = check_idx
            gap_end = check_idx + medium_step
            logger.debug(f"Phase 2: Gap narrowed to frames {gap_start}-{gap_end}")
            break

    # === PHASE 3: Frame-Level Binary Search ===
    logger.debug("Phase 3: Frame-level binary search")

    left = gap_start   # No object
    right = gap_end    # Has object

    # Binary search to find exact boundary (within 1 frame)
    while right - left > 1:
        mid = (left + right) // 2

        if check_frame(mid):
            # Object exists, look earlier
            right = mid
        else:
            # No object, look later
            left = mid

    # 'right' is the first frame with object
    result_frame = right
    result_time = video_source.frame_to_timestamp(result_frame)

    frames_before = frame_at_target - result_frame
    logger.info(
        f"Search complete. Found at frame {result_frame} (T-{frames_before} frames)"
    )

    return SearchResult(
        status=SearchStatus.SUCCESS,
        timestamp=result_time,
        precision_seconds=min_window,
        confidence=getattr(target_object, "confidence", None),
        iterations=iterations,
        message=f"Object found at frame {result_frame}",
    )


class BackwardTemporalSearch:
    """Backward coarse-to-fine temporal search for object instances."""

    # Search parameters
    INITIAL_WINDOW_MINUTES = 30.0
    MIN_WINDOW_SECONDS = 5.0
    MAX_LOOKBACK_HOURS = 3.0
    DEFAULT_FPS = 20.0

    def __init__(
        self,
        video_decoder: VideoDecoder,
        detector: ObjectDetector,
        tracker: ObjectTracker | None = None,
        fps: float = DEFAULT_FPS,
    ) -> None:
        self.video_decoder = video_decoder
        self.detector = detector
        self.tracker = tracker
        self.fps = fps

    def search(
        self,
        start_time: datetime,
        target_detection: ObjectDetection,
    ) -> SearchResult:
        """Execute backward frame-based search."""
        config = {
            "initial_window": int(self.INITIAL_WINDOW_MINUTES * 60),
            "min_window": self.MIN_WINDOW_SECONDS,
            "max_lookback": int(self.MAX_LOOKBACK_HOURS * 60 * 60),
            "precision": self.MIN_WINDOW_SECONDS,
        }

        return backward_search(
            target_time=start_time.timestamp(),
            target_object=target_detection,
            video_source=self.video_decoder,
            detector=self.detector,
            tracker=self.tracker,
            config=config,
            fps=self.fps,
        )


class MockVideoDecoder:
    """Mock video decoder for testing with frame-based API."""

    def __init__(
        self,
        fps: float = 20.0,
        available_ranges: list[tuple[int, int]] | None = None,
    ) -> None:
        self.fps = fps
        self.available_ranges = available_ranges or []
        self.frame_data = b"mock_frame_data"

    def timestamp_to_frame(self, timestamp: float) -> int:
        """Convert timestamp to frame index."""
        # Assume frame 0 is at timestamp 0 for simplicity
        return int(timestamp * self.fps)

    def frame_to_timestamp(self, frame_index: int) -> float:
        """Convert frame index to timestamp."""
        return frame_index / self.fps

    def get_frame(self, timestamp: float) -> bytes | None:
        """Get frame at timestamp."""
        frame_idx = self.timestamp_to_frame(timestamp)
        return self.get_frame_by_index(frame_idx)

    def get_frame_by_index(self, frame_index: int) -> bytes | None:
        """Get frame by index."""
        # Check if frame is in available ranges
        for start, end in self.available_ranges:
            if start <= frame_index <= end:
                return self.frame_data
        return self.frame_data  # Default to available for testing


class MockObjectDetector:
    """Mock object detector for testing."""

    def __init__(
        self,
        mock_detections: dict[int, list[ObjectDetection]] | None = None,
    ) -> None:
        self.mock_detections = mock_detections or {}
        self.call_count = 0

    def detect(self, frame: bytes) -> list[ObjectDetection]:
        """Mock: Return predefined detections or empty list."""
        self.call_count += 1
        return []


class MockObjectTracker:
    """Mock object tracker for testing."""

    def __init__(self) -> None:
        self.tracks: dict[str, ObjectTrack] = {}
        self._next_id = 1

    def update(self, detections: list[ObjectDetection]) -> list[ObjectTrack]:
        """Mock: Simple tracking by temporal proximity."""
        for det in detections:
            if getattr(det, "detection_id", None) is None:
                det.detection_id = f"track_{self._next_id}"
                self._next_id += 1
        return []

    def is_same_object(
        self,
        detection1: ObjectDetection,
        detection2: ObjectDetection,
    ) -> bool:
        """Mock: Check if same object."""
        return getattr(detection1, "label", None) == getattr(
            detection2, "label", None
        )

    def reset(self) -> None:
        """Reset tracker state."""
        self.tracks.clear()
        self._next_id = 1


# Type alias for backward compatibility
SearchResultLegacy = SearchResult
