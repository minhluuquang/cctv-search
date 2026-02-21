"""FastAPI application and routes."""

from __future__ import annotations

import io
import logging
import os
import random
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from cctv_search.ai import BoundingBox, ByteTrackTracker, DetectedObject, RFDetrDetector
from cctv_search.nvr import DahuaNVRClient

logger = logging.getLogger(__name__)

# Configuration for mock mode - can be set via environment variable
# Set MOCK_SEARCH_MODE=false to use real NVR integration
MOCK_SEARCH_MODE = os.getenv("MOCK_SEARCH_MODE", "true").lower() in ("true", "1", "yes")

# Global state
nvr_client: DahuaNVRClient | None = None
detector: RFDetrDetector | None = None
tracker: ByteTrackTracker | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    global nvr_client, detector, tracker
    nvr_client = DahuaNVRClient()
    detector = RFDetrDetector()
    tracker = ByteTrackTracker(track_thresh=0.5, track_buffer=30, frame_rate=20)
    # Load RF-DETR model on startup (optional - may fail if not installed)
    try:
        detector.load_model()
    except RuntimeError as e:
        logger.warning(f"Failed to load RF-DETR model: {e}")
        detector = None
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


class FrameObjectsRequest(BaseModel):
    """Request to extract a frame and detect/track objects."""

    timestamp: datetime
    channel: int = 1


class DetectedObjectWithId(BaseModel):
    """Detected object with track ID."""

    object_id: int
    label: str
    confidence: float
    bbox: dict[str, float]
    center: dict[str, float]


class FrameObjectsResponse(BaseModel):
    """Response with detected objects and annotated image path."""

    timestamp: datetime
    channel: int
    objects: list[DetectedObjectWithId]
    image_path: str  # Path where annotated image is saved
    total_objects: int


class VideoClipRequest(BaseModel):
    """Request to generate a video clip."""

    camera_id: str  # Channel number as string
    start_timestamp: datetime  # When to start the clip
    duration_seconds: int = 15  # Clip duration (default 15s)
    object_id: int | None = None  # Optional: track specific object in clip
    annotate_objects: bool = True  # Whether to draw bounding boxes on video


class VideoClipResponse(BaseModel):
    """Response with generated clip metadata."""

    clip_path: str  # Path to generated video file
    start_timestamp: datetime
    end_timestamp: datetime
    duration_seconds: int
    file_size_bytes: int
    download_url: str  # URL to download the clip
    objects_tracked: list[dict] | None  # List of tracked objects in clip


class ObjectSearchRequest(BaseModel):
    """Request to search for an object backward in time."""

    camera_id: str  # Channel number as string
    start_timestamp: datetime  # Starting point for backward search
    object_id: int  # Track ID from frame detection
    search_duration_seconds: int = 3600  # How far back to search (default 1 hour)
    # Object details for matching (from frame detection)
    object_label: str  # e.g., "person", "bicycle"
    object_bbox: dict[str, float]  # {x, y, width, height}
    object_confidence: float


class ObjectSearchResult(BaseModel):
    """Result of searching for an object."""

    found: bool
    first_seen_timestamp: datetime | None
    last_seen_timestamp: datetime | None
    search_iterations: int
    confidence: float | None
    message: str
    track_duration_seconds: float | None
    clip_path: str | None  # Path to generated video clip (if found)
    image_path: str | None  # Path to annotated frame image at first appearance


class ObjectSearchResponse(BaseModel):
    """Response for object search endpoint."""

    status: str  # "success", "not_found", "error"
    result: ObjectSearchResult | None


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
    if not nvr_client:
        raise HTTPException(status_code=500, detail="NVR client not initialized")

    try:
        # Check timestamp first
        if not request.timestamp:
            raise HTTPException(
                status_code=400, detail="Timestamp required for frame extraction"
            )

        # Check detector availability
        if not detector:
            raise HTTPException(status_code=500, detail="Detector not initialized")

        # Extract frame
        nvr_client.extract_frame(
            timestamp=request.timestamp,
            channel=int(request.camera_id) if request.camera_id.isdigit() else 1,
        )
        # TODO: Load frame from file and run detection
        # For now, return empty list
        return []

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def annotate_frame_with_objects(
    frame_path: Path,
    objects: list[DetectedObjectWithId],
) -> bytes:
    """Draw bounding boxes and object IDs on the frame image.

    Args:
        frame_path: Path to the frame image file.
        objects: List of detected objects with track IDs.

    Returns:
        Annotated image as PNG bytes.

    Raises:
        RuntimeError: If image cannot be loaded or annotated.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont

        # Load the image
        with Image.open(frame_path) as img:
            draw = ImageDraw.Draw(img)

            # Try to get a font, fall back to default if not available
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
            except OSError:
                font = ImageFont.load_default()

            # Color palette for different object classes
            colors = {
                "person": (0, 255, 0),  # Green
                "bicycle": (255, 0, 0),  # Blue
                "car": (0, 0, 255),  # Red
                "motorcycle": (255, 255, 0),  # Cyan
                "bus": (255, 0, 255),  # Magenta
                "truck": (0, 255, 255),  # Yellow
            }

            for obj in objects:
                # Get bounding box coordinates
                x = obj.bbox["x"]
                y = obj.bbox["y"]
                width = obj.bbox["width"]
                height = obj.bbox["height"]

                # Choose color based on label, default to green
                color = colors.get(obj.label.lower(), (0, 255, 0))

                # Draw bounding box
                draw.rectangle(
                    [x, y, x + width, y + height],
                    outline=color,
                    width=2,
                )

                # Prepare label text
                label_text = f"ID:{obj.object_id} {obj.label}"

                # Get text size for background rectangle
                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # Draw label background
                draw.rectangle(
                    [x, y - text_height - 4, x + text_width + 4, y],
                    fill=color,
                )

                # Draw label text (white on colored background)
                draw.text(
                    (x + 2, y - text_height - 2),
                    label_text,
                    fill=(255, 255, 255),
                    font=font,
                )

            # Save to bytes buffer
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return buffer.getvalue()

    except Exception as e:
        raise RuntimeError(f"Failed to annotate frame: {e}") from e


def _create_mock_annotated_image(objects: list[DetectedObjectWithId]) -> bytes:
    """Create a mock annotated image for testing without AI models.

    Args:
        objects: List of mock objects to display.

    Returns:
        PNG image bytes with mock annotations.
    """
    from PIL import Image, ImageDraw, ImageFont

    # Create a blank test image
    img = Image.new("RGB", (640, 480), color=(50, 50, 50))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except OSError:
        font = ImageFont.load_default()
        title_font = font

    # Draw title
    draw.text(
        (20, 20),
        "Mock Annotated Frame (AI Model Not Loaded)",
        fill=(255, 255, 255),
        font=title_font,
    )

    # Draw mock objects
    for i, obj in enumerate(objects):
        y_offset = 80 + i * 60
        x = 50 + (i % 3) * 180
        y = y_offset
        width = 120
        height = 80

        # Draw bounding box
        draw.rectangle([x, y, x + width, y + height], outline=(0, 255, 0), width=2)

        # Draw label
        label_text = f"ID:{obj.object_id} {obj.label}"
        draw.text((x, y - 20), label_text, fill=(0, 255, 0), font=font)

        # Draw mock object representation
        draw.text(
            (x + 10, y + 30),
            f"{obj.confidence:.2f}",
            fill=(0, 255, 0),
            font=font,
        )

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@app.post("/frames/objects", response_model=FrameObjectsResponse)
async def get_frame_with_objects(
    request: FrameObjectsRequest,
) -> FrameObjectsResponse:
    """Extract frame at timestamp, detect and track objects, save annotated image.

    Flow:
    1. Extract frame from NVR at specified timestamp and channel
    2. Run detection using detector.detect(frame_bytes)
    3. Run tracking using tracker.update(detections, frame_idx=0) to get track IDs
    4. Draw bounding boxes with track IDs on the image
    5. Save annotated image to ./frames/ directory and return path

    Args:
        request: Frame extraction and object detection request.

    Returns:
        FrameObjectsResponse with detected objects and image file path.

    Raises:
        HTTPException: If NVR client is not initialized or frame extraction fails.
    """
    global tracker

    if not nvr_client:
        raise HTTPException(status_code=500, detail="NVR client not initialized")

    # Check if detector is available - if not, use mock data
    use_mock = detector is None or not getattr(detector, "_model_loaded", False)

    try:
        # Step 1: Extract frame from NVR
        frame_path = nvr_client.extract_frame(
            timestamp=request.timestamp,
            channel=request.channel,
            output_path=f"/tmp/frame_{request.timestamp.isoformat()}.png",
        )

        objects_with_id: list[DetectedObjectWithId] = []

        if use_mock:
            # Generate mock objects for testing
            mock_objects = [
                DetectedObject(
                    label="person",
                    bbox=BoundingBox(
                        x=100.0, y=150.0, width=80.0, height=120.0, confidence=0.92
                    ),
                    confidence=0.92,
                    frame_timestamp=request.timestamp.timestamp(),
                ),
                DetectedObject(
                    label="bicycle",
                    bbox=BoundingBox(
                        x=250.0, y=200.0, width=100.0, height=80.0, confidence=0.85
                    ),
                    confidence=0.85,
                    frame_timestamp=request.timestamp.timestamp(),
                ),
                DetectedObject(
                    label="person",
                    bbox=BoundingBox(
                        x=400.0, y=100.0, width=70.0, height=110.0, confidence=0.78
                    ),
                    confidence=0.78,
                    frame_timestamp=request.timestamp.timestamp(),
                ),
            ]

            # Run through tracker to get track IDs
            tracks = tracker.update(mock_objects, frame_idx=0)

            # Create DetectedObjectWithId from tracks
            for track in tracks:
                obj_with_id = DetectedObjectWithId(
                    object_id=track.track_id,
                    label=track.label,
                    confidence=track.confidence,
                    bbox={
                        "x": track.x - track.width / 2,
                        "y": track.y - track.height / 2,
                        "width": track.width,
                        "height": track.height,
                    },
                    center={"x": track.x, "y": track.y},
                )
                objects_with_id.append(obj_with_id)

            # Create mock annotated image
            annotated_bytes = _create_mock_annotated_image(objects_with_id)

        else:
            # Real detection flow
            # Step 2: Read frame bytes
            with open(frame_path, "rb") as f:
                frame_bytes = f.read()

            # Step 3: Run detection
            detections: list[DetectedObject] = detector.detect(frame_bytes)

            # Set frame timestamp
            for det in detections:
                det.frame_timestamp = request.timestamp.timestamp()

            # Step 4: Run tracking
            tracks = tracker.update(detections, frame_idx=0)

            # Convert tracks to DetectedObjectWithId
            for track in tracks:
                obj_with_id = DetectedObjectWithId(
                    object_id=track.track_id,
                    label=track.label,
                    confidence=track.confidence,
                    bbox={
                        "x": track.x - track.width / 2,
                        "y": track.y - track.height / 2,
                        "width": track.width,
                        "height": track.height,
                    },
                    center={"x": track.x, "y": track.y},
                )
                objects_with_id.append(obj_with_id)

            # Step 5: Annotate image
            annotated_bytes = annotate_frame_with_objects(frame_path, objects_with_id)

        # Save annotated image to frames directory
        FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        ts_str = request.timestamp.strftime("%Y%m%d_%H%M%S")
        image_filename = f"frame_{ts_str}_ch{request.channel}.png"
        image_path = FRAMES_DIR / image_filename
        image_path.write_bytes(annotated_bytes)

        return FrameObjectsResponse(
            timestamp=request.timestamp,
            channel=request.channel,
            objects=objects_with_id,
            image_path=str(image_path),
            total_objects=len(objects_with_id),
        )

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Failed to process frame with object detection")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}") from e


# Storage configuration
CLIPS_DIR = Path("./clips")
FRAMES_DIR = Path("./frames")


def _generate_clip_filename(camera_id: str, timestamp: datetime) -> str:
    """Generate a unique filename for a video clip.

    Args:
        camera_id: Camera/channel identifier
        timestamp: Start timestamp of the clip

    Returns:
        Filename string with timestamp and camera ID
    """
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    return f"clip_cam{camera_id}_{timestamp_str}.mp4"


def _generate_mock_clip(
    output_path: Path,
    duration_seconds: int,
    resolution: tuple[int, int] = (640, 480),
) -> None:
    """Generate a mock video clip using ffmpeg testsrc.

    Args:
        output_path: Path where the clip will be saved
        duration_seconds: Duration of the clip in seconds
        resolution: Video resolution as (width, height)

    Raises:
        RuntimeError: If ffmpeg fails to generate the clip
    """
    import subprocess

    width, height = resolution

    cmd = [
        "ffmpeg",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration_seconds}:size={width}x{height}:rate=25",
        "-pix_fmt",
        "yuv420p",
        "-y",  # Overwrite output file if exists
        str(output_path),
    ]

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to generate mock clip: {e.stderr}") from e


@app.post("/video/clip", response_model=VideoClipResponse)
async def generate_video_clip(request: VideoClipRequest) -> VideoClipResponse:
    """Generate a video clip from the specified time range.

    Flow:
    1. Extract video segment from NVR using ffmpeg
    2. Store clip in the clips directory
    3. Return clip metadata with download URL

    Args:
        request: Video clip generation request

    Returns:
        VideoClipResponse with clip metadata and download URL

    Raises:
        HTTPException: If NVR client is not initialized or clip extraction fails
    """
    if not nvr_client:
        raise HTTPException(status_code=500, detail="NVR client not initialized")

    # Validate request parameters
    if request.duration_seconds <= 0:
        raise HTTPException(
            status_code=400, detail="Duration must be greater than 0 seconds"
        )

    if request.duration_seconds > 300:  # 5 minutes max
        raise HTTPException(
            status_code=400, detail="Maximum clip duration is 300 seconds (5 minutes)"
        )

    # Calculate end time
    end_timestamp = request.start_timestamp + timedelta(
        seconds=request.duration_seconds
    )

    # Ensure clips directory exists
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate unique filename
    clip_filename = _generate_clip_filename(request.camera_id, request.start_timestamp)
    clip_path = CLIPS_DIR / clip_filename

    try:
        # Determine if we should use mock clip (for testing without NVR)
        use_mock = not nvr_client.host or nvr_client.host in ("", "localhost")

        if use_mock:
            # Generate mock clip using ffmpeg testsrc
            logger.info(
                f"Generating mock clip: {clip_path} "
                f"(duration: {request.duration_seconds}s)"
            )
            _generate_mock_clip(clip_path, request.duration_seconds)
        else:
            # Extract clip from NVR
            logger.info(
                f"Extracting clip from NVR: channel={request.camera_id}, "
                f"start={request.start_timestamp}, duration={request.duration_seconds}s"
            )

            channel = int(request.camera_id) if request.camera_id.isdigit() else 1

            nvr_client.extract_clip(
                start_time=request.start_timestamp,
                end_time=end_timestamp,
                channel=channel,
                output_path=clip_path,
            )

        # Get file size
        file_size_bytes = clip_path.stat().st_size

        # Build download URL
        download_url = f"/clips/{clip_filename}"

        return VideoClipResponse(
            clip_path=str(clip_path),
            start_timestamp=request.start_timestamp,
            end_timestamp=end_timestamp,
            duration_seconds=request.duration_seconds,
            file_size_bytes=file_size_bytes,
            download_url=download_url,
            objects_tracked=None,  # TODO: Implement object tracking across clip
        )

    except RuntimeError as e:
        # Clean up partial file if it exists
        if clip_path.exists():
            clip_path.unlink()
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        # Clean up partial file if it exists
        if clip_path.exists():
            clip_path.unlink()
        logger.exception("Failed to generate video clip")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate clip: {e}"
        ) from e


# Import search algorithm types
from cctv_search.search.algorithm import BackwardTemporalSearch
from cctv_search.search.algorithm import BoundingBox as SearchBBox
from cctv_search.search.algorithm import ObjectDetection


class NVRVideoDecoder:
    """Video decoder adapter for NVR client.
    
    Adapts DahuaNVRClient to the VideoDecoder protocol expected by
    the search algorithm.
    """

    def __init__(self, nvr_client: DahuaNVRClient, channel: int, fps: float = 20.0):
        self.nvr_client = nvr_client
        self.channel = channel
        self.fps = fps
        self._temp_dir = Path("/tmp/cctv_search")
        self._temp_dir.mkdir(parents=True, exist_ok=True)

    def timestamp_to_frame(self, timestamp: float) -> int:
        """Convert timestamp to frame index."""
        return int(timestamp * self.fps)

    def frame_to_timestamp(self, frame_index: int) -> float:
        """Convert frame index to timestamp."""
        return frame_index / self.fps

    def get_frame(self, timestamp: float) -> bytes | None:
        """Get frame at timestamp."""
        frame_idx = self.timestamp_to_frame(timestamp)
        return self.get_frame_by_index(frame_idx)

    def get_frame_by_index(self, frame_index: int) -> bytes | None:
        """Get frame by index from NVR."""
        timestamp = self.frame_to_timestamp(frame_index)
        dt = datetime.fromtimestamp(timestamp)
        frame_path = self._temp_dir / f"frame_{frame_index}.png"

        try:
            self.nvr_client.extract_frame(dt, self.channel, frame_path)
            return frame_path.read_bytes()
        except Exception as e:
            logger.warning(f"Failed to get frame {frame_index}: {e}")
            return None


class SearchObjectDetector:
    """Detector wrapper for search algorithm.
    
    Wraps RFDetrDetector to match the ObjectDetector protocol expected
    by the search algorithm.
    """

    def __init__(self, detector, fps: float = 20.0):
        self.detector = detector
        self.fps = fps
        self.frame_size = (1920, 1080)  # Default frame size

    def detect(self, frame: bytes) -> list:
        """Run detection on frame and return ObjectDetection list."""
        # Run detection using the RF-DETR detector
        from cctv_search.ai import BoundingBox as AiBoundingBox

        ai_detections = self.detector.detect(frame)

        # Convert to search algorithm format
        object_detections = []
        for det in ai_detections:
            # Create segmentation mask (rectangle for now)
            width, height = self.frame_size
            mask_data = [[False] * width for _ in range(height)]
            x1 = int(det.bbox.x)
            y1 = int(det.bbox.y)
            x2 = int(det.bbox.x + det.bbox.width)
            y2 = int(det.bbox.y + det.bbox.height)
            for y in range(max(0, y1), min(height, y2)):
                for x in range(max(0, x1), min(width, x2)):
                    mask_data[y][x] = True

            from cctv_search.search.algorithm import SegmentationMask

            obj_det = ObjectDetection(
                label=det.label,
                bbox=SearchBBox(
                    x=det.bbox.x,
                    y=det.bbox.y,
                    width=det.bbox.width,
                    height=det.bbox.height,
                    confidence=det.confidence,
                ),
                mask=SegmentationMask(
                    mask=mask_data,
                    width=width,
                    height=height,
                ),
                confidence=det.confidence,
            )
            object_detections.append(obj_det)

        return object_detections


class SearchObjectTracker:
    """Tracker wrapper for search algorithm.

    Wraps ByteTrackTracker to match the ObjectTracker protocol expected
    by the search algorithm, specifically for is_same_object matching.
    """

    def __init__(
        self,
        tracker,
        target_bbox: dict[str, float],
        target_label: str,
    ):
        self.tracker = tracker
        self.target_bbox = target_bbox
        self.target_label = target_label

    def update(self, detections: list[ObjectDetection]) -> list:
        """Update tracker with detections."""
        # Convert ObjectDetection to tracker format if needed
        # For now, return empty list (tracker state managed internally)
        return []

    def is_same_object(
        self,
        detection1: ObjectDetection,
        detection2: ObjectDetection,
    ) -> bool:
        """Check if two detections represent the same object.
        
        Uses IoU and spatial proximity matching.
        """
        # Check label match
        if detection1.label != detection2.label:
            return False

        # Calculate IoU
        iou = detection1.bbox.iou_with(detection2.bbox)

        # Calculate center distance
        center1 = detection1.bbox.center
        center2 = detection2.bbox.center
        distance = ((center1.x - center2.x) ** 2 + (center1.y - center2.y) ** 2) ** 0.5

        # Match criteria: IoU >= 0.5 OR distance <= 100 pixels
        return iou >= 0.5 or distance <= 100.0

    def reset(self) -> None:
        """Reset tracker state."""
        pass


@app.post("/search/object", response_model=ObjectSearchResponse)
async def search_object(request: ObjectSearchRequest) -> ObjectSearchResponse:
    """Search backward in time to find when object first appeared.

    Implements backward coarse-to-fine search:
    1. Start from given timestamp (object is visible here)
    2. Search backward using coarse sampling (30 sec steps)
    3. Refine with medium sampling (5 sec steps)
    4. Pinpoint with frame-level binary search

    Args:
        request: Object search request with object ID and search parameters

    Returns:
        ObjectSearchResponse with search results

    Raises:
        HTTPException: If search fails or invalid parameters
    """
    if not nvr_client:
        raise HTTPException(status_code=500, detail="NVR client not initialized")

    # Validate request
    if request.search_duration_seconds <= 0:
        raise HTTPException(
            status_code=400, detail="Search duration must be greater than 0"
        )

    if request.search_duration_seconds > 10800:  # 3 hours max
        raise HTTPException(
            status_code=400, detail="Maximum search duration is 3 hours (10800 seconds)"
        )

    logger.info(
        f"Starting object search: object_id={request.object_id}, "
        f"label={request.object_label}, from={request.start_timestamp}, "
        f"searching_back={request.search_duration_seconds}s"
    )

    try:
        # Determine if we should use mock results (for testing without real NVR)
        use_mock = MOCK_SEARCH_MODE or not nvr_client.host

        if use_mock:
            # Generate realistic mock search results
            logger.info("Using mock search mode")

            # Simulate search iterations (coarse + medium + fine phases)
            mock_iterations = random.randint(15, 45)

            # Simulate finding object 60-90% of the time
            if random.random() < 0.75:
                # Object found - appeared somewhere in the search window
                appear_offset_seconds = random.randint(
                    30, int(request.search_duration_seconds * 0.8)
                )
                first_seen = request.start_timestamp - timedelta(
                    seconds=appear_offset_seconds
                )
                track_duration = (request.start_timestamp - first_seen).total_seconds()

                # Generate 15-second video clip from first appearance
                clip_duration = 15
                clip_filename = _generate_clip_filename(
                    request.camera_id, first_seen
                )
                clip_path = CLIPS_DIR / clip_filename

                try:
                    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

                    if use_mock or not nvr_client.host:
                        # Generate mock clip using ffmpeg testsrc
                        logger.info(
                            f"Generating mock clip for search result: {clip_path}"
                        )
                        _generate_mock_clip(clip_path, clip_duration)
                    else:
                        # Extract clip from NVR
                        logger.info(
                            f"Extracting clip from NVR: channel={request.camera_id}, "
                            f"start={first_seen}, duration={clip_duration}s"
                        )
                        channel = (
                            int(request.camera_id)
                            if request.camera_id.isdigit()
                            else 1
                        )
                        end_time = first_seen + timedelta(seconds=clip_duration)
                        nvr_client.extract_clip(
                            start_time=first_seen,
                            end_time=end_time,
                            channel=channel,
                            output_path=clip_path,
                        )

                    clip_path_str = str(clip_path)
                    logger.info(f"Clip saved to: {clip_path_str}")

                except Exception as e:
                    logger.warning(f"Failed to generate clip: {e}")
                    clip_path_str = None

                # Generate annotated image at first_seen timestamp
                image_path_str = None
                try:
                    channel = int(request.camera_id) if request.camera_id.isdigit() else 1
                    frame_path = nvr_client.extract_frame(
                        timestamp=first_seen,
                        channel=channel,
                        output_path=f"/tmp/search_frame_{first_seen.isoformat()}.png",
                    )
                    
                    # Create mock detected object for annotation
                    mock_obj = DetectedObjectWithId(
                        object_id=request.object_id,
                        label=request.object_label,
                        confidence=request.object_confidence,
                        bbox=request.object_bbox,
                        center={
                            "x": request.object_bbox["x"] + request.object_bbox["width"] / 2,
                            "y": request.object_bbox["y"] + request.object_bbox["height"] / 2,
                        },
                    )
                    
                    # Annotate and save image
                    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
                    ts_str = first_seen.strftime("%Y%m%d_%H%M%S")
                    image_filename = f"search_frame_{ts_str}_ch{channel}_obj{request.object_id}.png"
                    image_path = FRAMES_DIR / image_filename
                    
                    annotated_bytes = annotate_frame_with_objects(frame_path, [mock_obj])
                    image_path.write_bytes(annotated_bytes)
                    image_path_str = str(image_path)
                    logger.info(f"Search annotated image saved to: {image_path_str}")
                except Exception as e:
                    logger.warning(f"Failed to generate annotated image: {e}")
                    image_path_str = None

                result = ObjectSearchResult(
                    found=True,
                    first_seen_timestamp=first_seen,
                    last_seen_timestamp=request.start_timestamp,
                    search_iterations=mock_iterations,
                    confidence=request.object_confidence * random.uniform(0.85, 0.98),
                    message=f"Object found after {mock_iterations} search iterations. "
                    f"Track duration: {track_duration:.1f}s",
                    track_duration_seconds=track_duration,
                    clip_path=clip_path_str,
                    image_path=image_path_str,
                )

                return ObjectSearchResponse(status="success", result=result)
            else:
                # Object not found in search window
                result = ObjectSearchResult(
                    found=False,
                    first_seen_timestamp=None,
                    last_seen_timestamp=None,
                    search_iterations=mock_iterations,
                    confidence=None,
                    message="Object not found in specified search window. "
                    "It may have appeared earlier than the search range.",
                    track_duration_seconds=None,
                    clip_path=None,
                    image_path=None,
                )

                return ObjectSearchResponse(status="not_found", result=result)

        else:
            # Real implementation using NVR and search algorithm
            logger.info("Running real object search with NVR")

            # Check if detector and tracker are available
            if not detector:
                raise HTTPException(
                    status_code=500, detail="Detector not initialized"
                )
            if not tracker:
                raise HTTPException(
                    status_code=500, detail="Tracker not initialized"
                )

            try:
                # Create video decoder adapter
                video_decoder = NVRVideoDecoder(
                    nvr_client=nvr_client,
                    channel=int(request.camera_id)
                    if request.camera_id.isdigit()
                    else 1,
                    fps=20.0,
                )

                # Create search detector wrapper
                search_detector = SearchObjectDetector(
                    detector=detector, fps=20.0
                )

                # Create search tracker wrapper
                search_tracker = SearchObjectTracker(
                    tracker=tracker,
                    target_bbox=request.object_bbox,
                    target_label=request.object_label,
                )

                # Create backward search
                search = BackwardTemporalSearch(
                    video_decoder=video_decoder,
                    detector=search_detector,
                    tracker=search_tracker,
                    fps=20.0,
                )

                # Create target detection from request using search algorithm
                target_detection = ObjectDetection(
                    label=request.object_label,
                    bbox=SearchBBox(
                        x=request.object_bbox["x"],
                        y=request.object_bbox["y"],
                        width=request.object_bbox["width"],
                        height=request.object_bbox["height"],
                        confidence=request.object_confidence,
                    ),
                    confidence=request.object_confidence,
                )

                # Run search
                search_result = search.search(
                    start_time=request.start_timestamp,
                    target_detection=target_detection,
                )

                if search_result.found:
                    first_seen = datetime.fromtimestamp(
                        search_result.timestamp or 0
                    )
                    track_duration = (
                        request.start_timestamp - first_seen
                    ).total_seconds()

                    # Generate 15-second clip from first appearance
                    clip_duration = 15
                    clip_filename = _generate_clip_filename(
                        request.camera_id, first_seen
                    )
                    clip_path = CLIPS_DIR / clip_filename

                    try:
                        CLIPS_DIR.mkdir(parents=True, exist_ok=True)
                        logger.info(
                            f"Extracting clip from NVR: channel={request.camera_id}, "
                            f"start={first_seen}, duration={clip_duration}s"
                        )
                        channel = (
                            int(request.camera_id)
                            if request.camera_id.isdigit()
                            else 1
                        )
                        end_time = first_seen + timedelta(
                            seconds=clip_duration
                        )
                        nvr_client.extract_clip(
                            start_time=first_seen,
                            end_time=end_time,
                            channel=channel,
                            output_path=clip_path,
                        )
                        clip_path_str = str(clip_path)
                        logger.info(f"Clip saved to: {clip_path_str}")
                    except Exception as e:
                        logger.warning(f"Failed to generate clip: {e}")
                        clip_path_str = None

                    # Generate annotated image at first_seen timestamp
                    image_path_str = None
                    try:
                        channel = int(request.camera_id) if request.camera_id.isdigit() else 1
                        frame_path = nvr_client.extract_frame(
                            timestamp=first_seen,
                            channel=channel,
                            output_path=f"/tmp/search_frame_{first_seen.isoformat()}.png",
                        )
                        
                        # Run detection on the frame
                        with open(frame_path, "rb") as f:
                            frame_bytes = f.read()
                        
                        detections = detector.detect(frame_bytes)
                        
                        # Convert detections to DetectedObjectWithId format
                        objects_with_id = []
                        for i, det in enumerate(detections):
                            obj = DetectedObjectWithId(
                                object_id=i,
                                label=det.label,
                                confidence=det.confidence,
                                bbox={
                                    "x": det.bbox.x,
                                    "y": det.bbox.y,
                                    "width": det.bbox.width,
                                    "height": det.bbox.height,
                                },
                                center={
                                    "x": det.bbox.x + det.bbox.width / 2,
                                    "y": det.bbox.y + det.bbox.height / 2,
                                },
                            )
                            objects_with_id.append(obj)
                        
                        # Annotate and save image
                        FRAMES_DIR.mkdir(parents=True, exist_ok=True)
                        ts_str = first_seen.strftime("%Y%m%d_%H%M%S")
                        image_filename = f"search_frame_{ts_str}_ch{channel}_obj{request.object_id}.png"
                        image_path = FRAMES_DIR / image_filename
                        
                        annotated_bytes = annotate_frame_with_objects(frame_path, objects_with_id)
                        image_path.write_bytes(annotated_bytes)
                        image_path_str = str(image_path)
                        logger.info(f"Search annotated image saved to: {image_path_str}")
                    except Exception as e:
                        logger.warning(f"Failed to generate annotated image: {e}")
                        image_path_str = None

                    result = ObjectSearchResult(
                        found=True,
                        first_seen_timestamp=first_seen,
                        last_seen_timestamp=request.start_timestamp,
                        search_iterations=search_result.iterations,
                        confidence=search_result.confidence,
                        message=f"Object found after {search_result.iterations} "
                        f"search iterations. Track duration: {track_duration:.1f}s",
                        track_duration_seconds=track_duration,
                        clip_path=clip_path_str,
                        image_path=image_path_str,
                    )
                    return ObjectSearchResponse(status="success", result=result)
                else:
                    result = ObjectSearchResult(
                        found=False,
                        first_seen_timestamp=None,
                        last_seen_timestamp=None,
                        search_iterations=search_result.iterations,
                        confidence=None,
                        message="Object not found in specified search window. "
                        "It may have appeared earlier than the search range.",
                        track_duration_seconds=None,
                        clip_path=None,
                        image_path=None,
                    )
                    return ObjectSearchResponse(
                        status="not_found", result=result
                    )

            except Exception as e:
                logger.exception("Real object search failed")
                raise HTTPException(
                    status_code=500, detail=f"Search failed: {e}"
                ) from e

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Object search failed")
        raise HTTPException(status_code=500, detail=f"Search failed: {e}") from e
