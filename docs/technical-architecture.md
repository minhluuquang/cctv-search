# Technical Architecture

## System Overview

CCTV Search is built on a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Application                    │
│                         (cctv_search.api)                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
       ┌───────────────┼───────────────┐
       │               │               │
       ▼               ▼               ▼
┌─────────────┐ ┌──────────────┐ ┌──────────────┐
│ NVR Module  │ │ AI Module    │ │ Search Module│
│ (nvr/)      │ │ (ai/)        │ │ (search/)    │
├─────────────┤ ├──────────────┤ ├──────────────┤
│ Dahua Client│ │ RF-DETR      │ │ Backward     │
│ RTSP/FFmpeg │ │ Detector     │ │ Search Alg   │
└─────────────┘ ├──────────────┤ └──────────────┘
                │ ByteTrack    │
                │ Tracker      │
                └──────────────┘
```

## Module Breakdown

### 1. API Layer (`cctv_search.api`)

**File**: `src/cctv_search/api/__init__.py`

FastAPI application providing REST endpoints:

```python
@app.post("/nvr/frame")        # Extract frame at timestamp
@app.post("/ai/detect")         # Detect objects in frame
```

**Key Components**:
- `lifespan()`: Manages application state (NVR client, detector initialization)
- Pydantic models for request/response validation
- Global state for NVR client and detector instances

### 2. NVR Module (`cctv_search.nvr`)

**File**: `src/cctv_search/nvr/dahua.py`

Handles communication with Network Video Recorders:

#### DahuaNVRClient
```python
class DahuaNVRClient:
    def extract_frame(timestamp, channel) -> Path
    def _build_rtsp_url(channel, start_time, end_time) -> str
```

**Implementation Details**:
- Uses RTSP protocol for video streaming
- Leverages FFmpeg for frame extraction
- Supports Dahua RTSP playback endpoint format
- Credentials URL-encoded for special characters
- RTSP URL pattern: `rtsp://host:port/cam/playback?channel=N&starttime=YYYY_MM_DD_HH_MM_SS`

### 3. AI Module (`cctv_search.ai`)

#### RF-DETR Detector (`ai/rf_detr.py`)

Transformer-based object detection using RF-DETR model:

```python
class RFDetrDetector:
    def load_model() -> None           # Initialize pre-trained model
    def detect(frame) -> list[DetectedObject]
    def detect_batch(frames) -> list[list[DetectedObject]]
    def _bytes_to_numpy(frame_bytes) -> NDArray
```

**Key Features**:
- Real-time detection at 20+ FPS
- Supports both bytes and numpy array inputs
- Confidence threshold filtering (default: 0.5)
- BGR to RGB conversion for OpenCV compatibility

#### ByteTrack Tracker (`ai/byte_tracker.py`)

Multi-object tracking using ByteTrack algorithm:

```python
class ByteTrackTracker:
    def update(detections, frame_idx) -> list[Track]
    def is_same_object(det1, det2) -> bool
    def reset() -> None
```

**Algorithm**:
1. Two-stage matching (high/low confidence)
2. Kalman filter for motion prediction
3. Hungarian algorithm for optimal assignment
4. Track activation after 3 consecutive hits

**Parameters**:
- `track_thresh`: 0.5 (confidence threshold)
- `match_thresh`: 0.8 (IoU threshold)
- `track_buffer`: 30 frames (lost track retention)
- `frame_rate`: 20 FPS

### 4. Search Module (`cctv_search.search`)

#### Backward Coarse-to-Fine Search (`search/algorithm.py`)

Efficient temporal search algorithm:

```python
class BackwardTemporalSearch:
    def search(target_detection, start_time, end_time) -> SearchResult
    def _coarse_search(window) -> list[CandidateWindow]
    def _medium_search(window) -> list[CandidateWindow]
    def _fine_search(window) -> ObjectTrack
```

**Phases**:
1. **Coarse**: Every 30 seconds (120 frames @ 20 FPS)
2. **Medium**: Every 5 seconds (20 frames)
3. **Fine**: Every frame (1 frame)

**Complexity**: O(log n) frame-level binary search

### 5. Tracker Integration (`cctv_search.tracker`)

**File**: `src/cctv_search/tracker.py`

Bridge between ByteTrack and search algorithm:

```python
class ByteTrackTracker:
    def update(detections, frame_idx, timestamp) -> AssociationResult
    def is_same_object(det1, det2) -> bool
```

**Responsibilities**:
- Format conversion between detector and tracker
- Track state management
- Motion vector calculation

## Data Models

### Core Types

```python
# Detection
@dataclass
class DetectedObject:
    label: str
    bbox: BoundingBox
    confidence: float
    frame_timestamp: float

# Bounding Box
@dataclass  
class BoundingBox:
    x: float          # Top-left x
    y: float          # Top-left y  
    width: float
    height: float
    confidence: float

# Track
@dataclass
class Track:
    track_id: int
    label: str
    x: float          # Center x
    y: float          # Center y
    width: float
    height: float
    is_active: bool
    state: str        # 'tracked', 'lost', 'removed'
```

## Dependencies

### External Libraries

| Library | Purpose | Version |
|---------|---------|---------|
| FastAPI | Web framework | >=0.129.0 |
| uvicorn | ASGI server | >=0.41.0 |
| rfdetr | Object detection | >=1.4.3 |
| trackers | ByteTrack implementation | >=2.2.0 |
| supervision | Detection utilities | (via trackers) |
| opencv-python | Image processing | >=4.13.0 |
| numpy | Numerical computing | >=2.4.2 |
| ffmpeg | Frame extraction | External binary |

### Protocols

- **RTSP**: Real-time Streaming Protocol for NVR communication
- **HTTP/REST**: API communication
- **JPEG/PNG**: Frame image format

## Configuration

### Environment Variables

```bash
NVR_HOST=192.168.1.100        # NVR IP address
NVR_PORT=554                  # RTSP port
NVR_USERNAME=admin            # NVR credentials
NVR_PASSWORD=password         # NVR credentials
```

### Runtime Configuration

- Model confidence threshold: 0.5
- Tracker IoU threshold: 0.8
- Max lookback time: 3 hours
- Min search window: 5 seconds

## Error Handling

### Exception Hierarchy

```
RuntimeError
├── Model not loaded
├── Frame extraction failed
└── RTSP connection error

HTTPException (FastAPI)
├── 400: Bad request (invalid timestamp)
├── 500: Server error (client not initialized)
```

### Logging

Uses Python's standard logging with loguru integration:
- INFO: Normal operations
- WARNING: Recoverable issues (model load failure)
- ERROR: Fatal errors with stack traces

## Performance Considerations

### Bottlenecks

1. **FFmpeg frame extraction**: ~500ms per frame (network dependent)
2. **Model inference**: ~50ms on GPU, ~500ms on CPU
3. **RTSP connection setup**: ~200-500ms

### Optimizations

- Frame-level binary search (99% reduction in calls)
- Batch detection for multiple frames
- Connection reuse where possible
- Confidence threshold filtering before tracking

## Security

- Credentials stored in environment variables (not code)
- URL-encoded credentials for special characters
- No persistent storage of video data
- Local processing (no cloud dependencies)
