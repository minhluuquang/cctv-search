# API Reference

## REST API Endpoints

### NVR Endpoints

#### POST /nvr/frame

Extract a single frame from the NVR at a specific timestamp.

**Request Body**:
```json
{
  "timestamp": "2024-01-15T14:30:00",
  "channel": 1
}
```

**Response**:
```json
{
  "frame_path": "frame.png",
  "timestamp": "2024-01-15T14:30:00",
  "channel": 1
}
```

**Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| timestamp | ISO 8601 datetime | Yes | Exact time to extract frame |
| channel | integer | No | Camera channel number (default: 1) |

**Errors**:
- `500`: NVR client not initialized
- `400`: Invalid timestamp or extraction failed

---

### AI Endpoints

#### POST /ai/detect

Detect objects in a video frame.

**Request Body**:
```json
{
  "camera_id": "1",
  "timestamp": "2024-01-15T14:30:00"
}
```

**Response**:
```json
[
  {
    "label": "person",
    "confidence": 0.95,
    "bbox": {
      "x": 100.5,
      "y": 200.0,
      "width": 50.0,
      "height": 100.0
    },
    "timestamp": 14.5
  }
]
```

**Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| camera_id | string | Yes | Camera identifier (maps to channel) |
| timestamp | ISO 8601 datetime | Yes | Frame timestamp |

**Errors**:
- `500`: NVR client or detector not initialized
- `400`: Missing timestamp or detection failed

---

## Python SDK

### NVR Client

#### DahuaNVRClient

```python
from cctv_search.nvr import DahuaNVRClient, FrameRequest

# Initialize client
client = DahuaNVRClient(
    host="192.168.1.100",
    port=554,
    username="admin",
    password="password",
    rtsp_transport="tcp"
)

# Or use environment variables
client = DahuaNVRClient()  # Reads from env

# Extract frame
from datetime import datetime
frame_path = client.extract_frame(
    timestamp=datetime.now(),
    channel=1,
    output_path="frame.png"
)
```

**Methods**:

##### extract_frame()
```python
def extract_frame(
    self,
    timestamp: datetime,
    channel: int = 1,
    output_path: str | Path = "frame.png"
) -> Path
```

Extract a single frame at the exact timestamp using FFmpeg.

**Args**:
- `timestamp`: Exact timestamp to extract frame from
- `channel`: Camera channel number (default: 1)
- `output_path`: Path to save the extracted frame

**Returns**: Path to the extracted frame file

**Raises**: `RuntimeError` if FFmpeg fails

---

### AI Detector

#### RFDetrDetector

```python
from cctv_search.ai import RFDetrDetector

# Initialize detector
detector = RFDetrDetector(confidence_threshold=0.5)

# Load model
detector.load_model()

# Detect objects in frame
import cv2
frame = cv2.imread("frame.png")
detections = detector.detect(frame)

for det in detections:
    print(f"{det.label}: {det.confidence:.2f} at ({det.bbox.x}, {det.bbox.y})")
```

**Methods**:

##### load_model()
```python
def load_model(self) -> None
```

Download and initialize the pre-trained RF-DETR model. Model is cached after first download.

**Raises**: `RuntimeError` if RF-DETR is not installed

##### detect()
```python
def detect(
    self, 
    frame: bytes | NDArray[np.uint8]
) -> list[DetectedObject]
```

Detect objects in a video frame.

**Args**:
- `frame`: Video frame as bytes or numpy array (BGR format)

**Returns**: List of detected objects with bounding boxes and confidence scores

**Raises**: `RuntimeError` if model not loaded

##### detect_batch()
```python
def detect_batch(
    self,
    frames: list[bytes | NDArray[np.uint8]]
) -> list[list[DetectedObject]]
```

Detect objects in multiple frames.

---

### Object Tracker

#### ByteTrackTracker

```python
from cctv_search.ai import ByteTrackTracker

# Initialize tracker
tracker = ByteTrackTracker(
    track_thresh=0.5,
    match_thresh=0.8,
    track_buffer=30,
    frame_rate=20
)

# Update with detections
tracks = tracker.update(detections, frame_idx=100)

# Check if same object
is_same = tracker.is_same_object(det1, det2)
```

**Methods**:

##### update()
```python
def update(
    self,
    detections: list[DetectedObject],
    frame_idx: int
) -> list[Track]
```

Update tracker with new detections.

**Args**:
- `detections`: List of detections from current frame
- `frame_idx`: Current frame index

**Returns**: List of active tracks

##### is_same_object()
```python
def is_same_object(
    self,
    detection1: Any,
    detection2: Any
) -> bool
```

Check if two detections represent the same physical object using IoU and motion consistency.

**Args**:
- `detection1`: First detection
- `detection2`: Second detection

**Returns**: True if detections likely represent the same object

##### reset()
```python
def reset(self) -> None
```

Reset tracker state.

---

### Search Algorithm

#### BackwardTemporalSearch

```python
from cctv_search.search import BackwardTemporalSearch

# Initialize search
search = BackwardTemporalSearch(
    detector=detector,
    video_decoder=decoder,
    fps=20.0,
    min_window=5,
    max_lookback=10800  # 3 hours
)

# Search for object
result = search.search(
    target_detection=target,
    start_time=start,
    end_time=end
)
```

**Methods**:

##### search()
```python
def search(
    self,
    target_detection: ObjectDetection,
    start_time: float,
    end_time: float
) -> SearchResult
```

Execute backward coarse-to-fine search for target object.

**Args**:
- `target_detection`: Reference object to search for
- `start_time`: Search start timestamp (seconds)
- `end_time`: Search end timestamp (seconds)

**Returns**: `SearchResult` with status and found tracks

---

## Data Types

### BoundingBox

```python
@dataclass
class BoundingBox:
    x: float          # Top-left x coordinate
    y: float          # Top-left y coordinate
    width: float      # Box width
    height: float     # Box height
    confidence: float # Detection confidence (0-1)
```

**Methods**:
- `center: Point` - Get center point
- `area: float` - Get box area
- `iou_with(other: BoundingBox) -> float` - Calculate IoU

### DetectedObject

```python
@dataclass
class DetectedObject:
    label: str           # Object class label
    bbox: BoundingBox    # Bounding box
    confidence: float    # Detection confidence
    frame_timestamp: float  # Frame timestamp in seconds
```

### Track

```python
@dataclass
class Track:
    track_id: int        # Unique track ID
    label: str          # Object class label
    x: float            # Center x coordinate
    y: float            # Center y coordinate
    width: float        # Bounding box width
    height: float       # Bounding box height
    confidence: float   # Detection confidence
    frame_idx: int      # Frame index
    is_active: bool     # Whether track is active
    is_activated: bool  # Whether track is confirmed
    state: str          # 'tracked', 'lost', or 'removed'
    age: int            # Frames since last update
    hits: int           # Total detections in track
```

**Methods**:
- `update(detection, frame_idx)` - Update track with new detection
- `predict()` - Predict next position
- `mark_removed()` - Mark track as removed

### SearchResult

```python
@dataclass
class SearchResult:
    status: SearchStatus    # SUCCESS, NOT_FOUND, ERROR
    track: ObjectTrack | None  # Found object track
    search_time: float      # Search duration in seconds
    frames_checked: int     # Number of frames analyzed
    confidence: float       # Match confidence (0-1)
```

### SearchStatus (Enum)

```python
class SearchStatus:
    SUCCESS      # Object found
    NOT_FOUND    # Object not in search window
    IN_PROGRESS  # Search ongoing
    ERROR        # Search error occurred
```

---

## Error Handling

### HTTP Status Codes

| Code | Meaning | Common Causes |
|------|---------|---------------|
| 200 | Success | Request completed successfully |
| 400 | Bad Request | Invalid timestamp, missing parameters |
| 500 | Server Error | NVR not connected, model not loaded |

### Python Exceptions

| Exception | Raised When | Handling |
|-----------|-------------|----------|
| `RuntimeError` | Model not loaded, FFmpeg failed | Check setup, retry |
| `ValueError` | Invalid frame data | Verify input format |
| `HTTPException` | API validation failed | Check request format |
