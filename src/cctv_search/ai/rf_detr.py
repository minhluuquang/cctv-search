"""RF-DETR object detector implementation with deep feature extraction.

RF-DETR (Robust and Flexible DETR) is a transformer-based real-time object 
detection and instance segmentation model.

This implementation extracts deep features from the transformer's encoder
for robust object matching, especially during occlusion scenarios.

Installation:
    pip install rfdetr

Usage:
    from cctv_search.ai.rf_detr import RFDetrDetector, DetectedObject
    
    detector = RFDetrDetector()
    detector.load_model()
    
    # Detect with features
    detections = detector.detect_with_features(frame)
    
    # Match objects by feature similarity
    similarity = detector.compute_feature_similarity(det1, det2)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

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
    """Represents a detected object in video with deep features."""

    label: str
    bbox: BoundingBox
    confidence: float
    frame_timestamp: float
    features: np.ndarray | None = field(default=None, repr=False)
    """Deep feature embedding from RF-DETR (shape: [feature_dim])."""

    def has_features(self) -> bool:
        """Check if this detection has feature embedding."""
        return self.features is not None


class RFDetrDetector:
    """RF-DETR object detector with deep feature extraction for CCTV footage.
    
    RF-DETR is a transformer-based detection model that provides:
    - Real-time object detection
    - Deep feature embeddings for robust matching
    - High accuracy on small objects
    - Occlusion handling via feature similarity
    
    Example:
        >>> detector = RFDetrDetector()
        >>> detector.load_model()
        >>> detections = detector.detect_with_features(frame)
        >>> for det in detections:
        ...     print(f"{det.label}: {det.confidence:.2f}, features: {det.features.shape}")
    """

    # Default confidence threshold
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5

    # Feature similarity threshold for matching (cosine similarity)
    DEFAULT_FEATURE_THRESHOLD = 0.75

    # Default NMS IoU threshold (higher = less suppression)
    DEFAULT_NMS_THRESHOLD = 0.5

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD,
    ) -> None:
        """Initialize RF-DETR detector.

        Args:
            confidence_threshold: Minimum confidence for detections (0-1)
            nms_threshold: IoU threshold for NMS suppression (0-1, higher = less suppression)
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self._model = None
        self._model_loaded = False
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self) -> None:
        """Load RF-DETR model.
        
        Downloads and initializes the pre-trained RF-DETR model.
        Model is cached after first download.
        """
        try:
            from rfdetr import RFDETRLarge

            logger.info("Loading RF-DETR Large model...")
            self._model = RFDETRLarge()

            # Optimize model for inference to reduce latency
            logger.info("Optimizing RF-DETR model for inference...")
            # Use compile=False to avoid PyTorch JIT tracing warnings
            # The model is still optimized, just without torch.compile()
            self._model.optimize_for_inference(compile=False)

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
        """Detect objects in a frame (without features).
        
        Args:
            frame: Video frame as bytes or numpy array
            
        Returns:
            List of detected objects with bounding boxes and confidence scores
        """
        return self.detect_with_features(frame, extract_features=False)

    def detect_with_features(
        self, 
        frame: bytes | NDArray[np.uint8],
        extract_features: bool = True
    ) -> list[DetectedObject]:
        """Detect objects in a frame with optional deep feature extraction.
        
        Extracts features from the transformer's encoder output using
        RoI pooling for each detected bounding box.
        
        Args:
            frame: Video frame as bytes or numpy array
            extract_features: Whether to extract deep features (slower but more accurate)
            
        Returns:
            List of detected objects with bounding boxes, confidence, and features
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Convert bytes to numpy array if needed
            if isinstance(frame, bytes):
                frame = self._bytes_to_numpy(frame)
            
            # Store original frame for feature extraction
            original_frame = frame.copy()

            # Run inference with NMS threshold
            results = self._model.predict(frame, iou_threshold=self.nms_threshold)

            # Get class names from model if available
            class_names = getattr(self._model, 'class_names', None)
            
            detections = []

            for i in range(len(results.xyxy)):
                confidence = float(results.confidence[i])

                # Filter by confidence threshold
                if confidence < self.confidence_threshold:
                    continue

                # Extract bounding box
                x1, y1, x2, y2 = results.xyxy[i]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

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

                # Extract features if requested
                features = None
                if extract_features:
                    features = self._extract_features_for_bbox(
                        original_frame, x1, y1, x2, y2
                    )

                # Create detected object
                detected_obj = DetectedObject(
                    label=label,
                    bbox=bbox,
                    confidence=confidence,
                    frame_timestamp=0.0,  # Will be set by caller
                    features=features,
                )
                detections.append(detected_obj)

            return detections

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _extract_features_for_bbox(
        self, 
        frame: NDArray[np.uint8], 
        x1: int, y1: int, x2: int, y2: int
    ) -> np.ndarray | None:
        """Extract deep features for a specific bounding box region.
        
        Uses a simple but effective approach: crop the region and run
        through the backbone to get features.
        
        Args:
            frame: Full frame image
            x1, y1, x2, y2: Bounding box coordinates
            
        Returns:
            Feature vector as numpy array or None if extraction fails
        """
        try:
            import cv2
            
            # Ensure valid coordinates
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            # Crop the region
            crop = frame[y1:y2, x1:x2]
            
            # Resize to standard size for consistent features
            crop_resized = cv2.resize(crop, (224, 224))
            
            # Convert to tensor
            crop_tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).float() / 255.0
            crop_tensor = crop_tensor.unsqueeze(0).to(self._device)
            
            # Extract features using the backbone
            with torch.no_grad():
                # Get features from the encoder
                if hasattr(self._model.model, 'backbone'):
                    features = self._model.model.backbone(crop_tensor)
                    if isinstance(features, (list, tuple)):
                        features = features[-1]  # Use last layer features
                elif hasattr(self._model.model, 'encoder'):
                    features = self._model.model.encoder(crop_tensor)
                else:
                    # Fallback: use the model's feature extraction
                    features = self._extract_features_simple(crop_tensor)
                
                # Global average pooling to get feature vector
                if features.dim() == 4:  # [B, C, H, W]
                    features = F.adaptive_avg_pool2d(features, (1, 1))
                    features = features.view(features.size(0), -1)
                
                # L2 normalize for cosine similarity
                features = F.normalize(features, p=2, dim=1)
                
            return features.cpu().numpy().squeeze()
            
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return None

    def _extract_features_simple(self, tensor: torch.Tensor) -> torch.Tensor:
        """Simple feature extraction as fallback.
        
        Runs partial forward pass through the model to get features.
        """
        # Hook to capture intermediate features
        features = []
        
        def hook_fn(module, input, output):
            features.append(output)
        
        # Register hook on the first transformer layer or backbone output
        if hasattr(self._model.model, 'transformer'):
            handle = self._model.model.transformer.encoder.layers[0].register_forward_hook(hook_fn)
        else:
            # Use input as features (fallback)
            return tensor
        
        try:
            # Forward pass to trigger hook
            _ = self._model.model(tensor)
            if features:
                return features[0]
            else:
                return tensor
        finally:
            handle.remove()

    def compute_feature_similarity(
        self, 
        det1: DetectedObject, 
        det2: DetectedObject
    ) -> float:
        """Compute cosine similarity between two detections' features.
        
        Args:
            det1: First detected object with features
            det2: Second detected object with features
            
        Returns:
            Cosine similarity score (0-1), higher = more similar
        """
        if not det1.has_features() or not det2.has_features():
            return 0.0
        
        feat1 = det1.features
        feat2 = det2.features
        
        # Ensure 1D vectors
        if feat1.ndim > 1:
            feat1 = feat1.flatten()
        if feat2.ndim > 1:
            feat2 = feat2.flatten()
        
        # Cosine similarity
        similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-8)
        
        return float(similarity)

    def is_same_object(
        self,
        det1: DetectedObject,
        det2: DetectedObject,
        iou_threshold: float = 0.5,
        feature_threshold: float = DEFAULT_FEATURE_THRESHOLD,
        distance_threshold: float = 100.0,
    ) -> bool:
        """Check if two detections represent the same object instance.
        
        Uses a multi-criteria approach:
        1. IoU (bbox overlap) for spatial matching
        2. Deep feature similarity for appearance matching
        3. Center distance for motion consistency
        
        This handles occlusion by falling back to feature matching when
        bboxes don't overlap well.
        
        Args:
            det1: First detection
            det2: Second detection
            iou_threshold: Minimum IoU for spatial match
            feature_threshold: Minimum feature similarity for appearance match
            distance_threshold: Maximum center distance in pixels
            
        Returns:
            True if detections likely represent the same physical object
        """
        # Different classes can never be the same object
        if det1.label != det2.label:
            return False
        
        # Compute center distance
        c1x = det1.bbox.x + det1.bbox.width / 2
        c1y = det1.bbox.y + det1.bbox.height / 2
        c2x = det2.bbox.x + det2.bbox.width / 2
        c2y = det2.bbox.y + det2.bbox.height / 2
        distance = ((c1x - c2x) ** 2 + (c1y - c2y) ** 2) ** 0.5
        
        # Compute IoU
        x1_1, y1_1 = det1.bbox.x, det1.bbox.y
        x2_1, y2_1 = det1.bbox.x + det1.bbox.width, det1.bbox.y + det1.bbox.height
        x1_2, y2_1 = det2.bbox.x, det2.bbox.y
        x2_2, y2_2 = det2.bbox.x + det2.bbox.width, det2.bbox.y + det2.bbox.height
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y2_1)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            iou = 0.0
        else:
            intersection = (xi2 - xi1) * (yi2 - yi1)
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y2_1)
            union = area1 + area2 - intersection
            iou = intersection / union if union > 0 else 0.0
        
        # Criteria 1: Strong IoU match
        if iou >= iou_threshold and distance <= distance_threshold:
            return True
        
        # Criteria 2: Feature similarity (handles occlusion)
        if det1.has_features() and det2.has_features():
            feature_sim = self.compute_feature_similarity(det1, det2)
            if feature_sim >= feature_threshold and distance <= distance_threshold * 1.5:
                logger.debug(f"Matched by features: similarity={feature_sim:.3f}")
                return True
        
        return False

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
            result = self.detect_with_features(frame, extract_features=True)
            results.append(result)
        return results

    def _bytes_to_numpy(self, frame_bytes: bytes) -> NDArray[np.uint8]:
        """Convert frame bytes to numpy array.
        
        Args:
            frame_bytes: Raw frame data
            
        Returns:
            Frame as numpy array
        """
        import cv2

        nparr = np.frombuffer(frame_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode frame bytes to image")

        # Convert BGR to RGB (RF-DETR expects RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image


# Backward compatibility alias
RFDetrObjectDetector = RFDetrDetector
