# API Reference

## REST API Endpoints

### Frame Operations

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

#### POST /frames/objects

Extract a frame at timestamp, detect and track objects, save annotated image.

**Flow**:
1. Extract frame from NVR at specified timestamp and channel
2. Run object detection using RF-DETR
3. Run tracking using FeatureTracker to get track IDs
4. Draw bounding boxes with track IDs on the image
5. Save annotated image to `./frames/` directory

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
  "timestamp": "2024-01-15T14:30:00",
  "channel": 1,
  "objects": [
    {
      "object_id": 1,
      "label": "person",
      "confidence": 0.95,
      "bbox": {
        "x": 100.5,
        "y": 200.0,
        "width": 50.0,
        "height": 100.0
      },
      "center": {
        "x": 125.5,
        "y": 250.0
      }
    }
  ],
  "image_path": "./frames/frame_20240115_143000_ch1.png",
  "total_objects": 1
}
```

**Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| timestamp | ISO 8601 datetime | Yes | Frame timestamp |
| channel | integer | No | Camera channel number (default: 1) |

**Errors**:
- `500`: NVR client or detector not initialized
- `400`: Frame extraction failed

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
| timestamp | ISO 8601 datetime | No | Frame timestamp (defaults to now) |

**Errors**:
- `500`: NVR client or detector not initialized
- `400`: Missing timestamp or detection failed

---

### Video Clips

#### POST /video/clip

Generate a video clip from the specified time range.

**Request Body**:
```json
{
  "camera_id": "1",
  "start_timestamp": "2024-01-15T14:30:00",
  "duration_seconds": 15,
  "object_id": null,
  "annotate_objects": true
}
```

**Response**:
```json
{
  "clip_path": "./clips/clip_cam1_20240115_143000.mp4",
  "start_timestamp": "2024-01-15T14:30:00",
  "end_timestamp": "2024-01-15T14:30:15",
  "duration_seconds": 15,
  "file_size_bytes": 2048576,
  "download_url": "/clips/clip_cam1_20240115_143000.mp4",
  "objects_tracked": null
}
```

**Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| camera_id | string | Yes | Camera/channel identifier |
| start_timestamp | ISO 8601 datetime | Yes | Clip start time |
| duration_seconds | integer | No | Clip duration (default: 15, max: 300) |
| object_id | integer | No | Optional: track specific object |
| annotate_objects | boolean | No | Whether to draw bounding boxes |

**Errors**:
- `400`: Invalid duration (must be 1-300 seconds)
- `500`: NVR client not initialized or extraction failed

---

### Object Search

#### POST /search/object

Search backward in time to find when an object first appeared.

Implements backward coarse-to-fine search algorithm:
1. **Coarse Phase**: Sample video at 30-second intervals to find candidate windows
2. **Medium Phase**: Refine search within promising windows at 5-second intervals
3. **Fine Phase**: Binary search at frame level to pinpoint exact appearance time

**Request Body**:
```json
{
  "camera_id": "1",
  "start_timestamp": "2024-01-15T14:30:00",
  "object_id": 5,
  "search_duration_seconds": 3600,
  "object_label": "person",
  "object_bbox": {
    "x": 100.5,
    "y": 200.0,
    "width": 50.0,
    "height": 100.0
  },
  "object_confidence": 0.95
}
```

**Response** (success):
```json
{
  "status": "success",
  "result": {
    "found": true,
    "first_seen_timestamp": "2024-01-15T13:45:23",
    "last_seen_timestamp": "2024-01-15T14:30:00",
    "search_iterations": 47,
    "confidence": 0.92,
    "message": "Object found after 47 search iterations. Track duration: 2677.0s",
    "track_duration_seconds": 2677.0,
    "clip_path": null,
    "image_path": null,
    "play_command": "ffplay -rtsp_transport tcp rtsp://admin:pass@192.168.1.100:554/cam/playback?channel=1&starttime=2024_01_15_13_45_08&endtime=2024_01_15_13_45_23"
  }
}
```

**Response** (not found):
```json
{
  "status": "not_found",
  "result": {
    "found": false,
    "first_seen_timestamp": null,
    "last_seen_timestamp": null,
    "search_iterations": 50,
    "confidence": null,
    "message": "Object not found in specified search window. It may have appeared earlier than the search range.",
    "track_duration_seconds": null,
    "clip_path": null,
    "image_path": null,
    "play_command": null
  }
}
```

**Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| camera_id | string | Yes | Camera/channel identifier |
| start_timestamp | ISO 8601 datetime | Yes | Starting point for backward search |
| object_id | integer | Yes | Track ID from frame detection |
| search_duration_seconds | integer | No | How far back to search (default: 3600, max: 10800) |
| object_label | string | Yes | Object class label (e.g., "person", "bicycle") |
| object_bbox | object | Yes | Bounding box: {x, y, width, height} |
| object_confidence | float | Yes | Detection confidence (0-1) |

**Response Fields**:
| Field | Description |
|-------|-------------|
| status | "success", "not_found", or "error" |
| found | Whether the object was found |
| first_seen_timestamp | When object first appeared in search window |
| search_iterations | Number of frames analyzed during search |
| confidence | Match confidence score (0-1) |
| track_duration_seconds | How long the object was visible |
| play_command | Full ffplay command to view the 15s clip before first appearance |

**Errors**:
- `400`: Invalid search parameters (duration <= 0 or > 10800)
- `500`: NVR client, detector, or tracker not initialized
- `503`: Search failed with error

---

## Python SDK

### NVR Client

#### DahuaNVRClient

```python
from cctv_search.nvr import DahuaNVRClient, FrameRequest
from datetime import datetime

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
frame_path = client.extract_frame(
    timestamp=datetime.now(),
    channel=1,
    output_path="frame.png"
)

# Extract video clip
clip_path = client.extract_clip(
    start_time=datetime(2024, 1, 15, 14, 0, 0),
    end_time=datetime(2024, 1, 15, 14, 0, 15),
    channel=1,
    output_path="clip.mp4"
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

##### extract_clip()
```python
def extract_clip(
    self,
    start_time: datetime,
    end_time: datetime,
    channel: int = 1,
    output_path: str | Path = "clip.mp4"
) -> Path
```

Extract a video clip from start_time to end_time.

**Args**:
- `start_time`: Start timestamp for the clip
- `end_time`: End timestamp for the clip
- `channel`: Camera channel number (default: 1)
- `output_path`: Path to save the extracted clip

**Returns**: Path to the extracted video file

**Raises**: `RuntimeError` if FFmpeg fails

---

##### _build_rtsp_url()
```python
def _build_rtsp_url(
    self,
    channel: int,
    start_time: datetime,
    end_time: datetime
) -> str
```

Build RTSP playback URL for Dahua NVR.

**URL Format**: `rtsp://host:port/cam/playback?channel=N&starttime=YYYY_MM_DD_HH_MM_SS&endtime=YYYY_MM_DD_HH_MM_SS`

---

### AI Detector

#### RFDetrDetector

```python
from cctv_search.ai import RFDetrDetector, DetectedObject

# Initialize detector
detector = RFDetrDetector(confidence_threshold=0.5)

# Load model
detector.load_model(resolution=560)

# Detect objects in frame
import cv2
frame = cv2.imread("frame.png")
detections = detector.detect(frame)

for det in detections:
    print(f"{det.label}: {det.confidence:.2f} at ({det.bbox.x}, {det.bbox.y})")
    if det.has_features():
        print(f"  Features: {det.features.shape}")
```

**Methods**:

##### load_model()
```python
def load_model(
    self,
    model_id: str = "rfdetr_base",
    resolution: int = 560
) -> None
```

Download and initialize the pre-trained RF-DETR model.

**Args**:
- `model_id`: Model identifier (default: "rfdetr_base")
- `resolution`: Input resolution (default: 560, options: 560, 864)

**Raises**: `RuntimeError` if RF-DETR is not installed

---

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

---

##### detect_with_features()
```python
def detect_with_features(
    self,
    frame: bytes | NDArray[np.uint8]
) -> list[DetectedObject]
```

Detect objects and extract deep feature embeddings.

**Returns**: List of detected objects with features for matching

---

##### compute_feature_similarity()
```python
def compute_feature_similarity(
    self,
    det1: DetectedObject,
    det2: DetectedObject
) -> float
```

Compute cosine similarity between two detections using deep features.

**Returns**: Similarity score (0-1, higher = more similar)

---

### Object Tracker

#### FeatureTracker

```python
from cctv_search.ai import FeatureTracker

# Initialize tracker
tracker = FeatureTracker(
    track_thresh=0.5,
    match_thresh=0.8,
    track_buffer=30,
    frame_rate=20,
    feature_weight=0.3,
    feature_threshold=0.75
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
- `detections`: List of detections from current frame (may include features)
- `frame_idx`: Current frame index

**Returns**: List of active tracks

---

##### is_same_object()
```python
def is_same_object(
    self,
    detection1: DetectedObject | Track | ObjectDetection,
    detection2: DetectedObject | Track | ObjectDetection
) -> bool
```

Check if two detections represent the same physical object.

Uses combined IoU + feature similarity matching:
- IoU threshold: 0.8 (bounding box overlap)
- Feature threshold: 0.75 (cosine similarity)
- Both must pass (AND logic)

**Args**:
- `detection1`: First detection or track
- `detection2`: Second detection or track

**Returns**: True if detections likely represent the same object

---

##### reset()
```python
def reset(self) -> None
```

Reset tracker state for new tracking session.

---

### Search Algorithm

#### BackwardTemporalSearch

```python
from cctv_search.search import BackwardTemporalSearch

# Initialize search
search = BackwardTemporalSearch(
    detector=detector,
    video_decoder=decoder,
    tracker=tracker,
    fps=20.0
)

# Search for object
result = search.search(
    target_detection=target,
    start_time=start_timestamp
)
```

**Methods**:

##### search()
```python
def search(
    self,
    target_detection: ObjectDetection,
    start_time: datetime
) -> SearchResult
```

Execute backward coarse-to-fine search for target object.

**Args**:
- `target_detection`: Reference object to search for
- `start_time`: Search starting timestamp (search goes backward from here)

**Returns**: `SearchResult` with status and found information

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

---

### DetectedObject

```python
@dataclass
class DetectedObject:
    label: str           # Object class label
    bbox: BoundingBox    # Bounding box
    confidence: float    # Detection confidence
    frame_timestamp: float  # Frame timestamp in seconds
    features: np.ndarray | None  # Deep feature embedding
```

**Methods**:
- `has_features() -> bool` - Check if features are available

---

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

---

### ObjectDetection (Search Algorithm)

```python
@dataclass
class ObjectDetection:
    label: str           # Object class label
    bbox: BoundingBox    # Bounding box
    mask: SegmentationMask  # Segmentation mask
    confidence: float    # Detection confidence
```

---

### SearchResult

```python
@dataclass
class SearchResult:
    found: bool           # Whether object was found
    timestamp: float | None  # First appearance timestamp
    iterations: int       # Number of search iterations
    confidence: float     # Match confidence (0-1)
    message: str         # Status message
    status: SearchStatus # SUCCESS, NOT_FOUND, or ERROR
```

### SearchStatus (Enum)

```python
class SearchStatus:
    SUCCESS      # Object found
    NOT_FOUND    # Object not in search window
    ERROR        # Search error occurred
```

---

## Error Handling

### HTTP Status Codes

| Code | Meaning | Common Causes |
|------|---------|---------------|
| 200 | Success | Request completed successfully |
| 400 | Bad Request | Invalid timestamp, missing parameters, invalid duration |
| 500 | Server Error | NVR not connected, model not loaded, detector/tracker not initialized |
| 503 | Service Unavailable | Search algorithm error |

### Python Exceptions

| Exception | Raised When | Handling |
|-----------|-------------|----------|
| `RuntimeError` | Model not loaded, FFmpeg failed, frame extraction failed | Check setup, verify NVR connection, retry |
| `ValueError` | Invalid frame data or parameters | Verify input format and values |
| `HTTPException` | API validation failed | Check request format and parameters |

---

## Rate Limiting and Performance

### Search Algorithm Performance
- **Coarse search**: ~30x fewer frames than naive approach
- **Medium search**: ~6x fewer frames than coarse
- **Binary search**: O(log n) complexity

### Typical Frame Extraction Times
- Single frame: ~500-1000ms (network dependent)
- 15-second clip: ~2-5 seconds
- RTSP connection setup: ~200-500ms

### Model Inference
- RF-DETR detection: ~50ms on GPU, ~500ms on CPU
- Feature extraction: ~10ms additional overhead
