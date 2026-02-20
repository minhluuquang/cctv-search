"""FastAPI application and routes."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from cctv_search.ai.yolo import YOLODetector
from cctv_search.nvr import DahuaNVRClient

# Global state
nvr_client: DahuaNVRClient | None = None
detector: YOLODetector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    global nvr_client, detector
    nvr_client = DahuaNVRClient()
    detector = YOLODetector()
    # Load model on startup (TODO: make path configurable)
    await detector.load_model("/models/yolov8n.pt")
    yield
    # Shutdown
    # No disconnect needed for Dahua client


app = FastAPI(
    title="CCTV Search API",
    description="API for searching CCTV footage using AI",
    version="0.1.0",
    lifespan=lifespan,
)


# Pydantic models
class FrameExtractRequest(BaseModel):
    """Request to extract a frame at a specific timestamp."""

    timestamp: datetime
    channel: int = 1


class FrameExtractResponse(BaseModel):
    """Frame extraction response."""

    frame_path: str
    timestamp: datetime
    channel: int


class DetectionRequest(BaseModel):
    """Request for object detection."""

    camera_id: str
    timestamp: datetime | None = None


class DetectedObjectResponse(BaseModel):
    """Detected object response."""

    label: str
    confidence: float
    bbox: dict[str, float]
    timestamp: float


@app.post("/nvr/frame", response_model=FrameExtractResponse)
async def extract_frame(request: FrameExtractRequest) -> FrameExtractResponse:
    """Extract a frame at the specified timestamp."""
    if not nvr_client:
        raise HTTPException(status_code=500, detail="NVR client not initialized")

    try:
        frame_path = nvr_client.extract_frame(
            timestamp=request.timestamp,
            channel=request.channel,
        )
        return FrameExtractResponse(
            frame_path=str(frame_path),
            timestamp=request.timestamp,
            channel=request.channel,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/ai/detect")
async def detect_objects(request: DetectionRequest) -> list[DetectedObjectResponse]:
    """Detect objects in a video frame."""
    if not detector:
        raise HTTPException(status_code=500, detail="Detector not initialized")
    if not nvr_client:
        raise HTTPException(status_code=500, detail="NVR client not initialized")

    try:
        # Extract frame if timestamp specified
        if request.timestamp:
            nvr_client.extract_frame(
                timestamp=request.timestamp,
                channel=int(request.camera_id) if request.camera_id.isdigit() else 1,
            )
            # TODO: Load frame from file and run detection
            # For now, return empty list
            return []
        raise HTTPException(
            status_code=400, detail="Timestamp required for frame extraction"
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
