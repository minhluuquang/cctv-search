"""RF-DETR object detector implementation.

RF-DETR (Robust and Flexible DETR) is a transformer-based real-time object 
detection and instance segmentation model.

Installation:
    pip install rfdetr

Usage:
    from cctv_search.ai.rf_detr import RFDetrDetector
    
    detector = RFDetrDetector()
    detector.load_model()
    
    detections = detector.detect(frame_bytes)
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
    """Bounding box for detected objects."""

    x: float
    y: float
    width: float
    height: float
    confidence: float


@dataclass
class DetectedObject:
    """Represents a detected object in video."""

    label: str
    bbox: BoundingBox
    confidence: float
    frame_timestamp: float

logger = logging.getLogger(__name__)


class RFDetrDetector:
    """RF-DETR object detector for CCTV footage.
    
    RF-DETR is a transformer-based detection model that provides:
    - Real-time object detection
    - Instance segmentation
    - High accuracy on small objects
    
    Example:
        >>> detector = RFDetrDetector()
        >>> detector.load_model()
        >>> detections = detector.detect(frame_bytes)
        >>> for det in detections:
        ...     print(f"{det.label}: {det.bbox.confidence:.2f}")
    """
    
    # Default confidence threshold
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5
    
    def __init__(self, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> None:
        """Initialize RF-DETR detector.
        
        Args:
            confidence_threshold: Minimum confidence for detections (0-1)
        """
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._model_loaded = False
        
    def load_model(self) -> None:
        """Load RF-DETR model.
        
        Downloads and initializes the pre-trained RF-DETR model.
        Model is cached after first download.
        """
        try:
            from rfdetr import RFDETRBase
            
            logger.info("Loading RF-DETR model...")
            self._model = RFDETRBase()
            self._model_loaded = True
            logger.info("RF-DETR model loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to import rfdetr: {e}")
            logger.error("Please install RF-DETR: pip install rfdetr")
            raise RuntimeError("RF-DETR not installed. Run: pip install rfdetr") from e
        except Exception as e:
            logger.error(f"Failed to load RF-DETR model: {e}")
            raise RuntimeError(f"Failed to load model: {e}") from e
    
    def detect(self, frame: bytes | NDArray[np.uint8]) -> list[DetectedObject]:
        """Detect objects in a frame.
        
        Args:
            frame: Video frame as bytes or numpy array
            
        Returns:
            List of detected objects with bounding boxes and confidence scores
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Convert bytes to numpy array if needed
            if isinstance(frame, bytes):
                frame = self._bytes_to_numpy(frame)
            
            # Run inference
            results = self._model.predict(frame)
            
            # Parse supervision.Detections object
            # results.xyxy: array of shape (N, 4) with [x1, y1, x2, y2]
            # results.confidence: array of shape (N,) with confidence scores
            # results.class_id: array of shape (N,) with class IDs
            detections = []
            
            # Get class names from model if available
            class_names = getattr(self._model, 'class_names', None)
            
            for i in range(len(results.xyxy)):
                confidence = float(results.confidence[i])
                
                # Filter by confidence threshold
                if confidence < self.confidence_threshold:
                    continue
                
                # Extract bounding box
                x1, y1, x2, y2 = results.xyxy[i]
                
                bbox = BoundingBox(
                    x=float(x1),
                    y=float(y1),
                    width=float(x2 - x1),
                    height=float(y2 - y1),
                    confidence=confidence,
                )
                
                # Get class label
                class_id = int(results.class_id[i]) if results.class_id is not None else 0
                if class_names and class_id < len(class_names):
                    label = class_names[class_id]
                else:
                    label = f"class_{class_id}"
                
                # Create detected object
                detected_obj = DetectedObject(
                    label=label,
                    bbox=bbox,
                    confidence=confidence,
                    frame_timestamp=0.0,  # Will be set by caller
                )
                detections.append(detected_obj)
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def detect_batch(
        self, frames: list[bytes | NDArray[np.uint8]]
    ) -> list[list[DetectedObject]]:
        """Detect objects in multiple frames.
        
        Args:
            frames: List of video frames
            
        Returns:
            List of detection results for each frame
        """
        results = []
        for frame in frames:
            result = self.detect(frame)
            results.append(result)
        return results
    
    def _bytes_to_numpy(self, frame_bytes: bytes) -> NDArray[np.uint8]:
        """Convert frame bytes to numpy array.
        
        Args:
            frame_bytes: Raw frame data
            
        Returns:
            Frame as numpy array
        """
        # Decode bytes to image using OpenCV
        import cv2
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(frame_bytes, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode frame bytes to image")
        
        # Convert BGR to RGB (RF-DETR expects RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image


# Backward compatibility alias
RFDetrObjectDetector = RFDetrDetector
