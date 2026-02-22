"""RF-DETR object detector integration for CCTV search.

This module provides the integration layer between RF-DETR (from cctv_search.ai)
and the search algorithm. It converts RF-DETR detections to the format expected
by the search algorithm.

RF-DETR Installation:
    pip install rfdetr

Usage:
    from cctv_search.detector import RFDetrDetector, Detection, BoundingBox
    
    detector = RFDetrDetector()
    detector.load_model()
    
    detections = detector.detect(frame_bytes)
    for det in detections:
        print(f"{det.class_label}: {det.confidence:.2f}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


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


class RFDetrDetector:
    """RF-DETR detector wrapper for CCTV search.
    
    This class wraps the RF-DETR model from cctv_search.ai and provides
    the Detection interface expected by the search algorithm.
    
    Example:
        >>> detector = RFDetrDetector()
        >>> detector.load_model()
        >>> detections = detector.detect(frame_bytes, frame_idx=100)
        >>> for det in detections:
        ...     print(f"{det.class_label}: {det.confidence:.2f}")
    """
    
    DEFAULT_CONFIDENCE = 0.5
    DEFAULT_FRAME_SIZE = (1920, 1080)
    
    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE,
        frame_size: tuple[int, int] = DEFAULT_FRAME_SIZE,
    ):
        """Initialize RF-DETR detector.
        
        Args:
            confidence_threshold: Minimum confidence for detections
            frame_size: Frame dimensions (width, height)
        """
        self.confidence_threshold = confidence_threshold
        self.frame_size = frame_size
        self._detector = None
        self._model_loaded = False
        
    def load_model(self) -> None:
        """Load RF-DETR model.
        
        Downloads and initializes the pre-trained RF-DETR model.
        """
        try:
            from cctv_search.ai import RFDetrDetector as AiRFDetrDetector
            
            logger.info("Loading RF-DETR model...")
            self._detector = AiRFDetrDetector(
                confidence_threshold=self.confidence_threshold
            )
            self._detector.load_model()
            self._model_loaded = True
            logger.info("RF-DETR model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load RF-DETR: {e}")
            raise RuntimeError(f"Failed to load RF-DETR: {e}") from e
    
    def detect(
        self,
        frame: bytes,
        frame_idx: int = 0,
        timestamp: float = 0.0,
    ) -> list[Detection]:
        """Run detection on a frame.
        
        Args:
            frame: Frame data as bytes
            frame_idx: Frame index in video
            timestamp: Timestamp in seconds
            
        Returns:
            List of Detection objects
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Run detection with RF-DETR
            ai_detections = self._detector.detect(frame)
            
            # Convert to search algorithm format
            detections = []
            width, height = self.frame_size
            
            for ai_det in ai_detections:
                # Create bounding box
                bbox = BoundingBox(
                    x1=ai_det.bbox.x,
                    y1=ai_det.bbox.y,
                    x2=ai_det.bbox.x + ai_det.bbox.width,
                    y2=ai_det.bbox.y + ai_det.bbox.height,
                )
                
                # Create mask (rectangle for now, can be improved with segmentation)
                mask = np.zeros((height, width), dtype=np.bool_)
                x1_int = max(0, int(bbox.x1))
                y1_int = max(0, int(bbox.y1))
                x2_int = min(width, int(bbox.x2))
                y2_int = min(height, int(bbox.y2))
                mask[y1_int:y2_int, x1_int:x2_int] = True
                
                # Create detection
                detection = Detection(
                    bbox=bbox,
                    mask=mask,
                    confidence=ai_det.confidence,
                    class_label=ai_det.label,
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                )
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []


# Backward compatibility
MockRFDetrDetector = RFDetrDetector
