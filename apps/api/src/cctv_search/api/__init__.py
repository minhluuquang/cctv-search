"""FastAPI application and routes."""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from cctv_search.ai import DetectedObject, FeatureTracker, RFDetrDetector
from cctv_search.nvr import DahuaNVRClient
from cctv_search.search.algorithm import BackwardTemporalSearch, ObjectDetection
from cctv_search.search.algorithm import BoundingBox as SearchBBox
from cctv_search.streaming import http_stream_manager

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

# Global state - connection managers (thread-safe)
nvr_client: DahuaNVRClient | None = None
detector: RFDetrDetector | None = None  # Singleton - expensive to recreate

# Clips directory (for video clip downloads)
CLIPS_DIR = Path("./clips")
CLIPS_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    global nvr_client, detector
    nvr_client = DahuaNVRClient()
    detector = RFDetrDetector()
    # Start HTTP stream manager
    await http_stream_manager.start()
    # Load RF-DETR model on startup (optional - may fail if not installed)
    try:
        detector.load_model()
    except RuntimeError as e:
        logger.warning(f"Failed to load RF-DETR model: {e}")
        detector = None
    yield
    # Shutdown
    await http_stream_manager.stop()
    # No disconnect needed for Dahua client


app = FastAPI(
    title="CCTV Search API",
    description="API for searching CCTV footage using AI",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount clips directory for downloads
app.mount("/clips", StaticFiles(directory=str(CLIPS_DIR)), name="clips")


# Pydantic models for streaming
class StreamStartRequest(BaseModel):
    """Request to start HLS stream."""
    channel: int = 1
    start_time: datetime | None = None  # If provided, playback from this time


class StreamStartResponse(BaseModel):
    """Response with HLS stream URL."""
    playlist_url: str
    channel: int
    message: str


class StreamStopRequest(BaseModel):
    """Request to stop HLS stream."""
    channel: int = 1


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
    """Request to extract a frame and detect/track objects.

    Can provide either:
    - timestamp: Extract frame from NVR at specified time
    - frame_image: Base64-encoded image to use directly (more accurate)

    Note: timestamp is a string in format YYYY-MM-DDTHH:MM:SS to preserve
    local time without timezone conversion issues.
    """

    timestamp: str  # Format: "YYYY-MM-DDTHH:MM:SS" - local time, no timezone
    channel: int = 1
    frame_image: str | None = None  # Base64-encoded image (optional)


class DetectedObjectWithId(BaseModel):
    """Detected object with track ID."""

    object_id: int
    label: str
    confidence: float
    bbox: dict[str, float]
    center: dict[str, float]


class FrameObjectsResponse(BaseModel):
    """Response with detected objects."""

    timestamp: datetime
    channel: int
    objects: list[DetectedObjectWithId]
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
    # How far back to search (default 1 hour)
    search_duration_seconds: int = 3600
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
    play_command: str | None  # Full ffplay command to view the 15s clip


class ObjectSearchResponse(BaseModel):
    """Response for object search endpoint."""

    status: str  # "success", "not_found", "error"
    result: ObjectSearchResult | None


@app.post("/nvr/frame", response_model=FrameExtractResponse)
async def extract_frame(request: FrameExtractRequest) -> FrameExtractResponse:
    """Extract a frame at the specified timestamp."""
    if not nvr_client:
        raise HTTPException(
            status_code=500, detail="NVR client not initialized")

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
                font = ImageFont.truetype(
                    "/System/Library/Fonts/Helvetica.ttc", 16)
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


def annotate_video_clip(
    video_path: Path,
    output_path: Path,
    detector,
    tracker,
    target_label: str,
    target_bbox: dict[str, float],
    fps: float = 20.0,
) -> None:
    """Annotate video clip with bounding boxes around the searched object.

    Args:
        video_path: Path to input video file.
        output_path: Path to save annotated video.
        detector: Object detector instance.
        tracker: Object tracker instance.
        target_label: Label of the object to highlight (e.g., "person").
        target_bbox: Initial bounding box of the target object to match.
        fps: Video frame rate.
    """
    try:
        import cv2

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or fps

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc,
                              video_fps, (width, height))

        # Color palette
        colors = {
            "person": (0, 255, 0),      # Green
            "bicycle": (255, 0, 0),     # Blue
            "car": (0, 0, 255),         # Red
            "motorcycle": (255, 255, 0),  # Cyan
            "bus": (255, 0, 255),       # Magenta
            "truck": (0, 255, 255),     # Yellow
        }

        frame_idx = 0
        target_track_id = None
        logger.info(f"Annotating video: {total_frames} frames to process")

        # Create local tracker for this request (no global state)
        from cctv_search.ai import FeatureTracker
        local_tracker = FeatureTracker()
        logger.info("Local tracker created for video annotation")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Encode frame for detector
            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()

            # Detect objects
            detections = detector.detect(frame_bytes)

            # Update local tracker
            tracks = local_tracker.update(detections, frame_idx)

            # Find target track on first frame using bbox matching
            if frame_idx == 0 and tracks:
                # Calculate target center
                target_cx = target_bbox["x"] + target_bbox["width"] / 2
                target_cy = target_bbox["y"] + target_bbox["height"] / 2

                # Find track with closest center to target bbox
                best_track = None
                best_distance = float("inf")

                for track in tracks:
                    if track.is_active and track.label == target_label:
                        cx = track.x
                        cy = track.y
                        distance = ((cx - target_cx) ** 2 +
                                    (cy - target_cy) ** 2) ** 0.5

                        if distance < best_distance:
                            best_distance = distance
                            best_track = track

                if best_track and best_distance < 100:  # Within 100 pixels
                    target_track_id = best_track.track_id
                    logger.info(f"Target track identified: ID {target_track_id} "
                                f"(distance: {best_distance:.1f}px)")

            # Find target track in current frame
            target_track = None
            if target_track_id is not None:
                for track in tracks:
                    if track.is_active and track.track_id == target_track_id:
                        target_track = track
                        break

            # Draw bounding boxes
            if target_track:
                # Get target bounding box (Track stores x, y, width, height directly)
                x1 = int(target_track.x - target_track.width / 2)
                y1 = int(target_track.y - target_track.height / 2)
                x2 = int(target_track.x + target_track.width / 2)
                y2 = int(target_track.y + target_track.height / 2)

                # Get color for target
                color = colors.get(target_label.lower(), (0, 255, 0))

                # Draw thick bounding box for target
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                # Draw label with background
                label = f"ID:{target_track.track_id} {target_label}"
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )

                # Label background
                cv2.rectangle(
                    frame,
                    (x1, y1 - text_h - 8),
                    (x1 + text_w + 8, y1),
                    color,
                    -1,
                )

                # Label text
                cv2.putText(
                    frame,
                    label,
                    (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            # Add timestamp overlay
            timestamp_text = f"Frame: {frame_idx}"
            cv2.putText(
                frame,
                timestamp_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # Write frame
            out.write(frame)
            frame_idx += 1

            # Progress logging every 50 frames
            if frame_idx % 50 == 0:
                progress = (frame_idx / total_frames) * 100
                logger.info(
                    f"  Annotating: {frame_idx}/{total_frames} frames ({progress:.1f}%)")

        # Cleanup
        cap.release()
        out.release()

        logger.info(f"Annotated video saved: {output_path}")

    except ImportError:
        logger.warning("OpenCV not available, skipping video annotation")
    except Exception as e:
        logger.warning(f"Failed to annotate video: {e}")


@app.post("/frames/objects", response_model=FrameObjectsResponse)
async def get_frame_with_objects(
    timestamp: str = Form(..., description="Timestamp in format YYYY-MM-DDTHH:MM:SS"),
    channel: int = Form(1, description="Camera channel number"),
    frame_image: UploadFile | None = File(
        None, description="Frame image file (optional)"
    ),
) -> FrameObjectsResponse:
    """Extract frame and detect objects.

    Flow:
    1. Use provided frame image (multipart) OR extract from NVR at timestamp
    2. Run detection using detector.detect(frame_bytes)
    3. Run tracking using tracker.update(detections, frame_idx=0) to get track IDs
    4. Return detected objects with bounding boxes

    Args:
        timestamp: Timestamp string in format "YYYY-MM-DDTHH:MM:SS"
        channel: Camera channel number (default: 1)
        frame_image: Optional uploaded frame image file (more accurate than NVR)

    Returns:
        FrameObjectsResponse with detected objects.

    Raises:
        HTTPException: If NVR client is not initialized or frame extraction fails.
    """
    # Check if detector is available
    if not detector:
        raise HTTPException(status_code=500, detail="Detector not initialized")

    try:
        # Parse timestamp string to datetime
        from datetime import datetime
        timestamp_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

        # DEBUG: Log received timestamp details
        logger.info(f"[DEBUG] Received timestamp string: {timestamp}")
        logger.info(f"[DEBUG] Parsed timestamp: {timestamp_dt}")

        objects_with_id: list[DetectedObjectWithId] = []
        frame_bytes: bytes

        # Step 1: Get frame bytes - either from provided image or extract from NVR
        if frame_image:
            # Use provided uploaded image (more accurate)
            logger.info(f"[DEBUG] Using provided frame image: {frame_image.filename}")
            frame_bytes = await frame_image.read()
            logger.info(f"[DEBUG] Read frame image: {len(frame_bytes)} bytes")
        else:
            # Fall back to extracting frame from NVR (less accurate)
            logger.info("[DEBUG] No frame image provided, extracting from NVR")
            if not nvr_client:
                raise HTTPException(
                    status_code=500, detail="NVR client not initialized and no frame image provided")

            frame_path = nvr_client.extract_frame(
                timestamp=timestamp_dt,
                channel=channel,
                output_path=f"/tmp/frame_{timestamp.replace(':', '-')}.png",
            )
            with open(frame_path, "rb") as f:
                frame_bytes = f.read()

        # Step 2: Run detection
        detections: list[DetectedObject] = detector.detect(frame_bytes)

        # Set frame timestamp
        for det in detections:
            det.frame_timestamp = timestamp_dt.timestamp()

        # Step 3: Create fresh tracker for each request to avoid ID accumulation
        from cctv_search.ai import FeatureTracker
        fresh_tracker = FeatureTracker()
        tracks = fresh_tracker.update(detections, frame_idx=0)

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

        return FrameObjectsResponse(
            timestamp=timestamp_dt,
            channel=channel,
            objects=objects_with_id,
            total_objects=len(objects_with_id),
        )

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Failed to process frame with object detection")
        raise HTTPException(
            status_code=500, detail=f"Processing failed: {e}") from e


# Storage configuration
CLIPS_DIR = Path("./clips")


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
        raise HTTPException(
            status_code=500, detail="NVR client not initialized")

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
    clip_filename = _generate_clip_filename(
        request.camera_id, request.start_timestamp)
    clip_path = CLIPS_DIR / clip_filename

    try:
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
        """Get frame by index from NVR with exponential backoff retry logic.

        Retries up to 5 times with exponential backoff:
        Retry delays: 1s, 2s, 4s, 8s, 16s
        """
        timestamp = self.frame_to_timestamp(frame_index)
        dt = datetime.fromtimestamp(timestamp)
        frame_path = self._temp_dir / f"frame_{frame_index}.png"

        max_retries = 5
        # Exponential backoff: 2^0, 2^1, 2^2, 2^3, 2^4
        retry_delays = [1, 2, 4, 8, 16]

        for attempt, delay in enumerate(retry_delays, 1):
            try:
                self.nvr_client.extract_frame(dt, self.channel, frame_path)
                if attempt > 1:
                    logger.info(
                        f"Successfully extracted frame {frame_index} on retry {attempt}"
                    )
                return frame_path.read_bytes()
            except Exception as e:
                if attempt < max_retries:
                    retry_dt = dt + timedelta(seconds=delay)
                    logger.warning(
                        f"Failed to get frame {frame_index} at {dt.isoformat()} (attempt {attempt}/{max_retries}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    try:
                        self.nvr_client.extract_frame(
                            retry_dt, self.channel, frame_path)
                        logger.info(
                            f"Successfully extracted frame {frame_index} at retry time {retry_dt.isoformat()} "
                            f"(attempt {attempt})"
                        )
                        return frame_path.read_bytes()
                    except Exception:
                        # Retry failed, continue to next backoff delay
                        dt = retry_dt  # Update dt for next retry
                        continue
                else:
                    logger.error(
                        f"Failed to get frame {frame_index} after {max_retries} attempts. "
                        f"Last attempt at {dt.isoformat()} failed: {e}"
                    )
                    return None

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

    Wraps FeatureTracker to match the ObjectTracker protocol expected
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

        Delegates to FeatureTracker's matching (combines feature similarity + IoU + distance).
        """
        # Delegate to FeatureTracker's matching logic
        return self.tracker.is_same_object(detection1, detection2)

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
        raise HTTPException(
            status_code=500, detail="NVR client not initialized")

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
        # Real implementation using NVR and search algorithm
        logger.info("Running real object search with NVR")

        # Check if detector is available
        if not detector:
            raise HTTPException(
                status_code=500, detail="Detector not initialized"
            )

        try:
            # Create local tracker for this search request
            from cctv_search.ai import FeatureTracker
            local_tracker = FeatureTracker()

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

            # Create search tracker wrapper using local FeatureTracker
            search_tracker = SearchObjectTracker(
                tracker=local_tracker,
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

            # Log search configuration
            logger.info("\n" + "=" * 70)
            logger.info("OBJECT SEARCH STARTED")
            logger.info("=" * 70)
            logger.info(f"Camera ID: {request.camera_id}")
            logger.info(f"Start timestamp: {request.start_timestamp}")
            logger.info(
                f"Object: {request.object_label} (ID: {request.object_id})")
            logger.info(f"Search duration: {request.search_duration_seconds}s")
            logger.info(f"Bounding box: x={request.object_bbox['x']:.1f}, y={request.object_bbox['y']:.1f}, "
                        f"w={request.object_bbox['width']:.1f}, h={request.object_bbox['height']:.1f}")
            logger.info(f"Confidence: {request.object_confidence:.3f}")
            logger.info("=" * 70)

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
            logger.info("\nStarting backward temporal search...")
            search_result = search.search(
                start_time=request.start_timestamp,
                target_detection=target_detection,
            )

            # Check for error status first
            from cctv_search.search.algorithm import SearchStatus

            if search_result.status == SearchStatus.ERROR:
                logger.error("\n" + "=" * 70)
                logger.error("SEARCH RESULT: ERROR")
                logger.error("=" * 70)
                logger.error(f"Error: {search_result.message}")
                logger.error("=" * 70)
                raise HTTPException(
                    status_code=503,
                    detail=f"Search failed: {search_result.message}"
                )

            if search_result.found:
                first_seen = datetime.fromtimestamp(
                    search_result.timestamp or 0
                )
                track_duration = (
                    request.start_timestamp - first_seen
                ).total_seconds()

                # Skip clip and image generation - only search results returned
                clip_path_str = None
                image_path_str = None

                # Build RTSP playback URL for 15 seconds before first appearance
                playback_start = first_seen - timedelta(seconds=15)
                playback_end = first_seen  # End at the found timestamp
                channel = int(
                    request.camera_id) if request.camera_id.isdigit() else 1
                rtsp_url = nvr_client._build_rtsp_url_with_auth(
                    channel=channel,
                    start_time=playback_start,
                    end_time=playback_end,
                )
                # Return full ffplay command (no quotes needed, URL is already encoded)
                rtsp_url = f"ffplay -rtsp_transport tcp {rtsp_url}"

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
                    play_command=rtsp_url,
                )

                logger.info("\n" + "=" * 70)
                logger.info("SEARCH RESULT: SUCCESS")
                logger.info("=" * 70)
                logger.info(f"First seen: {first_seen}")
                logger.info(f"Track duration: {track_duration:.1f} seconds")
                logger.info(f"Search iterations: {search_result.iterations}")
                logger.info(f"Confidence: {search_result.confidence}")
                logger.info("=" * 70)

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
                    play_command=None,
                )

                logger.info("\n" + "=" * 70)
                logger.info("SEARCH RESULT: NOT FOUND")
                logger.info("=" * 70)
                logger.info(f"Search iterations: {search_result.iterations}")
                logger.info(
                    "Reason: Object not found in specified search window")
                logger.info(
                    "Suggestion: The object may have appeared earlier than the search range")
                logger.info("=" * 70)

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
        raise HTTPException(
            status_code=500, detail=f"Search failed: {e}") from e


@app.post("/stream/start", response_model=StreamStartResponse)
async def start_stream(request: StreamStartRequest) -> StreamStartResponse:
    """Start HLS stream for a channel.

    Transcodes RTSP stream to HLS for browser playback.
    Supports both live streaming and playback from specific time.
    Credentials are kept backend-only for security.

    Args:
        request: Stream start request with channel number and optional start time.

    Returns:
        StreamStartResponse with HLS playlist URL.

    Raises:
        HTTPException: If NVR client not initialized.
    """
    if not nvr_client:
        raise HTTPException(
            status_code=500, detail="NVR client not initialized")

    try:
        # Start HTTP-based stream
        await http_stream_manager.start_stream(
            channel=request.channel,
            nvr_client=nvr_client,
            start_time=request.start_time,
        )

        mode = "playback" if request.start_time else "live"
        return StreamStartResponse(
            playlist_url=f"/hls/{request.channel}/playlist.m3u8",
            channel=request.channel,
            message=f"HLS {mode} stream started for channel {request.channel}",
        )
    except Exception as e:
        logger.exception("Failed to start stream")
        raise HTTPException(
            status_code=500, detail=f"Failed to start stream: {e}") from e


@app.post("/stream/stop")
async def stop_stream(request: StreamStopRequest) -> dict:
    """Stop HLS stream for a channel.

    Args:
        request: Stream stop request with channel number.

    Returns:
        Success message.
    """
    try:
        await http_stream_manager.stop_stream(request.channel)
        return {
            "message": f"Stream stopped for channel {request.channel}",
            "channel": request.channel,
        }
    except Exception as e:
        logger.exception("Failed to stop stream")
        raise HTTPException(
            status_code=500, detail=f"Failed to stop stream: {e}") from e


@app.get("/stream/status")
async def get_stream_status() -> dict:
    """Get status of active streams.

    Returns:
        Dictionary with active channel list.
    """
    active_channels = list(http_stream_manager.streams.keys())
    return {
        "active_channels": active_channels,
        "count": len(active_channels),
    }


@app.get("/stream/ready/{channel}")
async def check_stream_ready(channel: int) -> dict:
    """Check if a stream is ready to play.

    Returns:
        Dictionary with ready status and playlist URL if ready.
    """
    stream = http_stream_manager.get_stream(channel)

    if stream and stream.is_running and len(stream.segments) > 0:
        return {
            "ready": True,
            "playlist_url": f"/hls/{channel}/playlist.m3u8",
            "channel": channel,
            "segments_available": len(stream.segments),
        }

    return {
        "ready": False,
        "playlist_url": None,
        "channel": channel,
    }


@app.get("/hls/{channel}/playlist.m3u8")
async def get_hls_playlist(channel: int) -> Response:
    """Get HLS playlist for a channel.

    Args:
        channel: Camera channel number.

    Returns:
        HLS playlist content.
    """
    playlist = await http_stream_manager.get_playlist(channel)
    return Response(
        content=playlist,
        media_type="application/vnd.apple.mpegurl",
    )


@app.get("/hls/{channel}/segment_{segment_index}.ts")
async def get_hls_segment(channel: int, segment_index: int) -> Response:
    """Get HLS segment for a channel.

    Args:
        channel: Camera channel number.
        segment_index: Segment index.

    Returns:
        MPEG-TS segment data.
    """
    data = await http_stream_manager.get_segment(channel, segment_index)
    return Response(
        content=data,
        media_type="video/mp2t",
    )


@app.get("/nvr/status")
async def get_nvr_status() -> dict:
    """Get NVR connection status.

    Returns:
        Dictionary with NVR configuration and connection status.
    """
    if not nvr_client:
        return {
            "connected": False,
            "error": "NVR client not initialized",
            "config": {
                "host": None,
                "port": None,
                "username": None,
                "has_password": False,
            }
        }

    return {
        "connected": True,
        "config": {
            "host": nvr_client.host if nvr_client.host else None,
            "port": nvr_client.port,
            "username": nvr_client.username if nvr_client.username else None,
            "has_password": bool(nvr_client.password),
        },
        "message": "NVR client configured" if nvr_client.host else "NVR not configured - set NVR_HOST env var"
    }


@app.get("/stream/{channel}/timestamps")
async def get_hls_timestamps(channel: int) -> dict:
    """Get timestamp mapping for HLS stream segments.

    Returns a mapping of segment filenames to actual video timestamps,
    allowing the UI to extract accurate timestamps from the HLS stream.

    Args:
        channel: Camera channel number.

    Returns:
        Dictionary with start_time, segment_duration, and segment mappings.
    """
    stream = http_stream_manager.get_stream(channel)

    if not stream or not stream.start_time:
        raise HTTPException(
            status_code=404,
            detail=f"Timestamp mapping not found for channel {channel}. "
                   "Stream may not be started or is in live mode."
        )

    try:
        mapping = {
            "start_time": stream.start_time.isoformat(),
            "segment_duration": 2.0,
            "segments": {},
        }

        for seg in stream.segments:
            if seg.timestamp:
                mapping["segments"][f"segment_{seg.index:03d}.ts"] = seg.timestamp.isoformat()

        return mapping
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read timestamp mapping: {e}"
        ) from e
