"""YOLO-based object detector implementation."""

from __future__ import annotations

import time

from cctv_search.ai import BoundingBox, DetectedObject


class YOLODetector:
    """YOLO-based object detector."""

    SUPPORTED_LABELS = ["person", "car", "truck", "bicycle", "dog", "cat"]

    def __init__(self) -> None:
        self._model_loaded = False
        self._model_path: str | None = None

    async def load_model(self, model_path: str) -> None:
        """Load YOLO model."""
        # TODO: Implement actual YOLO model loading
        self._model_path = model_path
        self._model_loaded = True

    async def detect(self, frame: bytes) -> list[DetectedObject]:
        """Detect objects in frame."""
        if not self._model_loaded:
            raise RuntimeError("Model not loaded")
        # TODO: Implement actual detection with YOLO
        # Return dummy detection for now
        return [
            DetectedObject(
                label="person",
                bbox=BoundingBox(
                    x=100.0, y=100.0, width=50.0, height=100.0, confidence=0.95
                ),
                confidence=0.95,
                frame_timestamp=time.time(),
            )
        ]

    async def detect_batch(self, frames: list[bytes]) -> list[list[DetectedObject]]:
        """Detect objects in multiple frames."""
        results = []
        for frame in frames:
            result = await self.detect(frame)
            results.append(result)
        return results
