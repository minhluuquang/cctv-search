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

# Configure logging to show INFO level logs
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class FrameExtractionError(Exception):
    """Raised when frame extraction from video source fails.

    This exception indicates that the NVR or video source is unavailable
    or the requested timestamp does not have recorded footage.
    """
    pass


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
    1. Phase 1: Coarse sampling (300 second / 5 minute steps)
    2. Phase 2: Binary search (finds exact boundary within 5 seconds)

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
    logger.info(f"backward_search STARTED target_time={target_time}")
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
    coarse_step = 300 * seconds_to_frames  # 300 seconds = 6000 frames @ 20fps

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

        # Get frame with timestamp for logging
        frame_timestamp = video_source.frame_to_timestamp(frame_idx)
        frame_dt = datetime.fromtimestamp(frame_timestamp)
        logger.info(
            f"[check_frame] Frame {frame_idx} ({frame_dt.isoformat()}) - "
            f"iteration {iterations}"
        )

        frame = video_source.get_frame_by_index(frame_idx)
        if frame is None:
            timestamp = video_source.frame_to_timestamp(frame_idx)
            dt = datetime.fromtimestamp(timestamp)
            raise FrameExtractionError(
                f"Frame extraction failed at frame {frame_idx} "
                f"(timestamp: {dt.isoformat()}). "
                f"Check NVR connectivity and ensure recordings exist for "
                f"this time period."
            )

        # Run detection
        detections = detector.detect(frame)
        matching = [d for d in detections if getattr(d, "label", None) == target_label]

        # Check if it's the same object
        if matching and tracker is not None:
            for candidate in matching:
                if tracker.is_same_object(target_object, candidate):
                    return True

        return False

    try:
        # === PHASE 1: Coarse Sampling (300 second / 5 minute steps) ===
        logger.info("=" * 70)
        logger.info("PHASE 1: Coarse sampling (300 second / 5 minute steps)")
        logger.info("=" * 70)

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
                    gap_start_ts = video_source.frame_to_timestamp(check_idx)
                    gap_end_ts = video_source.frame_to_timestamp(last_known_frame)
                    logger.info(
                        f"Phase 1: Gap found! Object disappeared between "
                        f"{datetime.fromtimestamp(gap_start_ts).isoformat()} and "
                        f"{datetime.fromtimestamp(gap_end_ts).isoformat()}"
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

        # === PHASE 2: Binary Search (find exact boundary) ===
        logger.info("=" * 70)
        logger.info("PHASE 2: Binary search (finding exact boundary)")
        logger.info("=" * 70)

        # Search range from Phase 1
        left = current_frame - step   # No object here (gap_start)
        right = last_known_frame      # Object here (gap_end)
        
        # Target precision: 5 seconds in frames
        precision_frames = int(5 * fps_float)
        
        left_ts = video_source.frame_to_timestamp(left)
        right_ts = video_source.frame_to_timestamp(right)
        logger.info(
            f"Binary search range: {datetime.fromtimestamp(left_ts).isoformat()} to "
            f"{datetime.fromtimestamp(right_ts).isoformat()} ({right - left} frames, "
            f"target precision: {precision_frames} frames)"
        )
        
        # Binary search to find exact boundary within 5-second precision
        phase2_iterations = 0
        while right - left > precision_frames:
            mid = (left + right) // 2
            phase2_iterations += 1
            
            mid_ts = video_source.frame_to_timestamp(mid)
            logger.info(f"Phase 2 iteration {phase2_iterations}: checking frame {mid} ({datetime.fromtimestamp(mid_ts).isoformat()})")
            
            if check_frame(mid):
                # Object exists, move left boundary
                right = mid
                logger.info(f"  -> Object found, narrowing to earlier time")
            else:
                # No object, move right boundary
                left = mid
                logger.info(f"  -> Object not found, narrowing to later time")
        
        # 'left' is now within 5 seconds of the first appearance boundary
        result_frame = left
        result_time = video_source.frame_to_timestamp(result_frame)
        frames_before = frame_at_target - result_frame
        
        logger.info(
            f"Search complete. Found at frame {result_frame} (T-{frames_before} frames) "
            f"with {right - left} frame precision"
        )

        return SearchResult(
            status=SearchStatus.SUCCESS,
            timestamp=result_time,
            precision_seconds=5.0,
            confidence=getattr(target_object, "confidence", None),
            iterations=iterations,
            message=f"Object found at frame {result_frame}",
        )

    except FrameExtractionError as e:
        logger.error(f"[ERROR] Search aborted due to frame extraction failure: {e}")
        return SearchResult(
            status=SearchStatus.ERROR,
            message=str(e),
            iterations=iterations,
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
        logger.info("BackwardTemporalSearch.search() CALLED")
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
