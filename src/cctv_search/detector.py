"""Mock RF-DETR object detector for CCTV search.

This module provides a mock/stub implementation of RF-DETR detection and
segmentation for testing and development purposes.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class BoundingBox:
    """Bounding box coordinates for detected objects.

    Attributes:
        x1: Left coordinate
        y1: Top coordinate
        x2: Right coordinate
        y2: Bottom coordinate
    """

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Return width of the box."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Return height of the box."""
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        """Return center point (x, y)."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        """Return area of the box."""
        return self.width * self.height

    def iou(self, other: BoundingBox) -> float:
        """Calculate IoU with another bounding box."""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection

        return intersection / union if union > 0 else 0.0


@dataclass
class Detection:
    """A single object detection result.

    Attributes:
        bbox: Bounding box coordinates
        mask: Binary segmentation mask (2D boolean array)
        confidence: Detection confidence score (0-1)
        class_label: Object class label (e.g., "bicycle", "person")
        frame_idx: Frame index where detection occurred
        timestamp: Timestamp in seconds
    """

    bbox: BoundingBox
    mask: NDArray[np.bool_]
    confidence: float
    class_label: str
    frame_idx: int
    timestamp: float

    def __post_init__(self):
        """Validate detection attributes."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")

    def mask_iou(self, other: Detection) -> float:
        """Calculate IoU between this detection's mask and another.

        Args:
            other: Another detection to compare masks with

        Returns:
            Mask IoU value between 0 and 1
        """
        if self.mask.shape != other.mask.shape:
            raise ValueError(
                f"Mask shapes must match: {self.mask.shape} vs {other.mask.shape}"
            )

        intersection = np.logical_and(self.mask, other.mask).sum()
        union = np.logical_or(self.mask, other.mask).sum()

        return float(intersection / union) if union > 0 else 0.0

    def center_distance(self, other: Detection) -> float:
        """Calculate Euclidean distance between centers.

        Args:
            other: Another detection to calculate distance to

        Returns:
            Euclidean distance in pixels
        """
        cx1, cy1 = self.bbox.center
        cx2, cy2 = other.bbox.center
        return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5


@dataclass
class DetectorConfig:
    """Configuration for mock detector behavior.

    Attributes:
        default_classes: List of class labels to detect
        confidence_range: Min/max confidence values to generate
        default_frame_size: Default frame dimensions (width, height)
        detection_probability: Probability of detecting an object per frame
        custom_behavior: Optional callback to customize detection generation
    """

    default_classes: list[str] = field(
        default_factory=lambda: ["bicycle", "person", "car", "motorcycle"]
    )
    confidence_range: tuple[float, float] = (0.7, 0.99)
    default_frame_size: tuple[int, int] = (1920, 1080)
    detection_probability: float = 0.8
    custom_behavior: Callable[[int, float], list[Detection]] | None = None


class MockRFDetrDetector:
    """Mock RF-DETR detector for testing and development.

    This class simulates the RF-DETR object detection and segmentation model
    with configurable behavior for testing purposes.

    Example:
        >>> config = DetectorConfig(detection_probability=0.9)
        >>> detector = MockRFDetrDetector(config)
        >>> detections = detector.detect(frame_idx=100, timestamp=5.0)
    """

    def __init__(self, config: DetectorConfig | None = None):
        """Initialize the mock detector.

        Args:
            config: Configuration for detector behavior
        """
        self.config = config or DetectorConfig()
        self._rng = random.Random()
        self._rng.seed(42)

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducible detection results.

        Args:
            seed: Random seed value
        """
        self._rng.seed(seed)

    def detect(
        self,
        frame_idx: int,
        timestamp: float,
        frame: NDArray[np.uint8] | None = None,
    ) -> list[Detection]:
        """Run detection on a frame.

        Args:
            frame_idx: Frame index in the video
            timestamp: Timestamp in seconds
            frame: Optional frame data (not used in mock, for API compatibility)

        Returns:
            List of Detection objects
        """
        if self.config.custom_behavior:
            return self.config.custom_behavior(frame_idx, timestamp)

        if self._rng.random() > self.config.detection_probability:
            return []

        return self._generate_random_detections(frame_idx, timestamp)

    def _generate_random_detections(
        self, frame_idx: int, timestamp: float
    ) -> list[Detection]:
        """Generate random detections for testing.

        Args:
            frame_idx: Frame index
            timestamp: Timestamp in seconds

        Returns:
            List of generated Detection objects
        """
        width, height = self.config.default_frame_size

        num_objects = self._rng.randint(1, 4)
        detections = []

        for _ in range(num_objects):
            # Generate random bounding box
            box_width = self._rng.uniform(50, 200)
            box_height = self._rng.uniform(50, 200)
            x1 = self._rng.uniform(0, width - box_width)
            y1 = self._rng.uniform(0, height - box_height)
            x2 = x1 + box_width
            y2 = y1 + box_height

            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

            # Generate mask (simplified as rectangle for mock)
            mask = np.zeros((height, width), dtype=np.bool_)
            mask[int(y1) : int(y2), int(x1) : int(x2)] = True

            confidence = self._rng.uniform(*self.config.confidence_range)
            class_label = self._rng.choice(self.config.default_classes)

            detection = Detection(
                bbox=bbox,
                mask=mask,
                confidence=confidence,
                class_label=class_label,
                frame_idx=frame_idx,
                timestamp=timestamp,
            )
            detections.append(detection)

        return detections

    def detect_with_target(
        self,
        frame_idx: int,
        timestamp: float,
        target_detection: Detection,
        appear_probability: float = 0.7,
        similarity_noise: float = 0.1,
    ) -> Detection | None:
        """Generate a detection matching a target (for simulating tracking).

        Args:
            frame_idx: Frame index
            timestamp: Timestamp in seconds
            target_detection: Target detection to simulate
            appear_probability: Probability that target appears in this frame
            similarity_noise: Amount of position variation to add

        Returns:
            Detection matching target if it appears, None otherwise
        """
        if self._rng.random() > appear_probability:
            return None

        # Add some noise to position
        noise_x = self._rng.uniform(-similarity_noise, similarity_noise) * 50
        noise_y = self._rng.uniform(-similarity_noise, similarity_noise) * 50

        bbox = BoundingBox(
            x1=target_detection.bbox.x1 + noise_x,
            y1=target_detection.bbox.y1 + noise_y,
            x2=target_detection.bbox.x2 + noise_x,
            y2=target_detection.bbox.y2 + noise_y,
        )

        # Copy mask with slight modifications
        mask = target_detection.mask.copy()

        confidence = min(
            1.0,
            target_detection.confidence + self._rng.uniform(-0.05, 0.05),
        )

        return Detection(
            bbox=bbox,
            mask=mask,
            confidence=confidence,
            class_label=target_detection.class_label,
            frame_idx=frame_idx,
            timestamp=timestamp,
        )
