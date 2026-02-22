# Technical Architecture

## System Overview

CCTV Search is built on a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Application                    │
│                         (cctv_search.api)                   │
├──────────────────────┬──────────────────────────────────────┤
│  Frame Endpoints     │  Search Endpoints                    │
│  /nvr/frame          │  /search/object                      │
│  /frames/objects     │                                      │
│  /video/clip         │  AI Endpoints                        │
│                      │  /ai/detect                          │
└──────────────────────┴──────────────────────────────────────┘
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
│ Clip Extract│ ├──────────────┤ │ Video Decoder│
└─────────────┘ │ Feature      │ └──────────────┘
                │ Tracker      │
                │ (Deep Match) │
                └──────────────┘
```

## Module Breakdown

### 1. API Layer (`cctv_search.api`)

**File**: `src/cctv_search/api/__init__.py`

FastAPI application providing REST endpoints and coordinating between modules.

**Endpoints**:
```python
@app.post("/nvr/frame")           # Extract frame at timestamp
@app.post("/frames/objects")      # Extract + detect + track + annotate
@app.post("/ai/detect")            # Detect objects in frame
@app.post("/video/clip")          # Generate video clip
@app.post("/search/object")       # Backward temporal search
```

**Key Components**:
- `lifespan()`: Manages application state (NVR client, detector, tracker initialization)
- `NVRVideoDecoder`: Adapter between Dahua client and search algorithm
- `SearchObjectDetector`: Detector wrapper for search algorithm
- `SearchObjectTracker`: Tracker wrapper for search algorithm
- Pydantic models for request/response validation
- Frame and video annotation functions

**Global State**:
- `nvr_client`: DahuaNVRClient instance
- `detector`: RFDetrDetector instance
- `tracker`: FeatureTracker instance

---

### 2. NVR Module (`cctv_search.nvr`)

**File**: `src/cctv_search/nvr/dahua.py`

Handles communication with Network Video Recorders via RTSP protocol.

#### DahuaNVRClient
```python
class DahuaNVRClient:
    def extract_frame(timestamp, channel, output_path) -> Path
    def extract_clip(start_time, end_time, channel, output_path) -> Path
    def _build_rtsp_url(channel, start_time, end_time) -> str
    def _build_rtsp_url_with_auth(channel, start_time, end_time) -> str
```

**Implementation Details**:
- Uses RTSP protocol for video streaming from Dahua NVRs
- Leverages FFmpeg for frame and clip extraction
- Supports Dahua RTSP playback endpoint format
- Credentials URL-encoded for special characters (@, :, etc.)
- RTSP URL pattern: `rtsp://host:port/cam/playback?channel=N&starttime=YYYY_MM_DD_HH_MM_SS&endtime=YYYY_MM_DD_HH_MM_SS`
- 1-second window for frame extraction
- Direct codec copy for clip extraction (faster, no re-encoding)

**Frame Extraction**:
- Builds 1-second RTSP window around target timestamp
- Uses FFmpeg with RTSP transport (TCP/UDP)
- Extracts single frame with `-frames:v 1`

**Clip Extraction**:
- Calculates duration from start/end times
- Uses FFmpeg with `-c:v copy` and `-c:a copy`
- Maintains original codec (faster than re-encoding)

---

### 3. AI Module (`cctv_search.ai`)

#### RF-DETR Detector (`ai/rf_detr.py`)

Transformer-based object detection using RF-DETR model with deep feature extraction:

```python
class RFDetrDetector:
    def load_model(model_id, resolution) -> None
    def detect(frame) -> list[DetectedObject]
    def detect_with_features(frame) -> list[DetectedObject]
    def compute_feature_similarity(det1, det2) -> float
    def _bytes_to_numpy(frame_bytes) -> NDArray
    def _extract_features(frame_tensor, boxes) -> Tensor
```

**Key Features**:
- Real-time detection at 20+ FPS on GPU
- Supports both bytes and numpy array inputs (BGR format)
- Confidence threshold filtering (default: 0.5)
- BGR to RGB conversion for OpenCV compatibility
- Deep feature extraction from transformer encoder
- Feature similarity matching for occlusion handling

**Model Resolution**:
- Default: 560x560
- Alternative: 864x864 (higher accuracy, slower)

**Feature Extraction**:
- Hooks into transformer encoder output
- Extracts features at object locations
- Uses bilinear interpolation for ROI pooling
- L2 normalization for cosine similarity

#### FeatureTracker (`ai/byte_tracker.py`)

Custom multi-object tracker using deep feature embeddings from RF-DETR:

```python
class FeatureTracker:
    def update(detections, frame_idx) -> list[Track]
    def is_same_object(det1, det2) -> bool
    def reset() -> None
```

**Algorithm**:
1. Match detections to existing tracks using combined scoring
2. Simple motion prediction (age-based removal)
3. Track activation after 2 consecutive hits
4. Track deletion after max_age frames (default: 30)

**Matching Strategy** (combined scoring):
- Feature similarity: 60% weight (primary, handles occlusion)
- IoU overlap: 20% weight (spatial consistency)
- Center distance: 20% weight (proximity check)
- Label consistency required

**Track States**:
- `new`: Recently created, not yet activated
- `tracked`: Active and confirmed
- `lost`: Temporarily not detected (occlusion)
- `removed`: Deleted track

**Key Features**:
- Uses deep features from RF-DETR transformer encoder
- No external tracking library dependency
- Cosine similarity for feature matching (threshold: 0.75)
- Robust to occlusion via feature persistence

---

### 4. Search Module (`cctv_search.search`)

#### Backward Coarse-to-Fine Search (`search/algorithm.py`)

Efficient temporal search algorithm that minimizes AI model calls:

```python
class BackwardTemporalSearch:
    def search(target_detection, start_time) -> SearchResult
    def _coarse_search(target, video_decoder, tracker, fps) -> list[CandidateWindow]
    def _medium_search(target, windows, video_decoder, tracker, fps) -> list[CandidateWindow]
    def _fine_search(target, windows, video_decoder, tracker, fps) -> float | None
```

**Search Phases**:

1. **Coarse Phase** (30-second steps):
   - Sample every 30 seconds going backward
   - Find windows where target object appears
   - ~99% reduction in model calls

2. **Medium Phase** (5-second steps):
   - Refine within promising coarse windows
   - Sample every 5 seconds
   - ~6x reduction from coarse

3. **Fine Phase** (frame-level binary search):
   - Binary search within 5-second windows
   - Find exact first appearance frame
   - O(log n) complexity

**Result**: ~20 detector calls vs ~2,160 for naive approach (99% reduction)

**VideoDecoder Protocol**:
```python
class VideoDecoder:
    def timestamp_to_frame(timestamp: float) -> int
    def frame_to_timestamp(frame_index: int) -> float
    def get_frame(timestamp: float) -> bytes | None
    def get_frame_by_index(frame_index: int) -> bytes | None
```

**NVRVideoDecoder Implementation**:
- Adapter for DahuaNVRClient
- Exponential backoff retry logic (5 attempts)
- Retry delays: 1s, 2s, 4s, 8s, 16s
- Converts between timestamps and frame indices

---

### 5. Tracker Integration (`cctv_search.tracker`)

**File**: `src/cctv_search/tracker.py`

Bridge for tracking integration. Provides unified interface.

---

### 6. Detector Integration (`cctv_search.detector`)

**File**: `src/cctv_search/detector.py`

Detector utilities and integration layer.

---

## Data Models

### Core Types

```python
# Detection with Features
@dataclass
class DetectedObject:
    label: str
    bbox: BoundingBox
    confidence: float
    frame_timestamp: float
    features: np.ndarray | None  # Deep feature embedding

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
    age: int
    hits: int

# Search Result
@dataclass
class SearchResult:
    found: bool
    timestamp: float | None
    iterations: int
    confidence: float
    message: str
    status: SearchStatus
```

## Dependencies

### External Libraries

| Library | Purpose | Version |
|---------|---------|---------|
| FastAPI | Web framework | >=0.129.0 |
| uvicorn | ASGI server | >=0.41.0 |
| rfdetr | Object detection | >=1.4.3 |

| supervision | Detection utilities | (via trackers) |
| opencv-python | Image processing | >=4.13.0 |
| numpy | Numerical computing | >=2.4.2 |
| torch | Deep learning | >=2.0 |
| Pillow | Image annotation | >=10.0 |
| python-dotenv | Environment config | >=1.0 |
| ffmpeg | Frame extraction | External binary |

### Protocols

- **RTSP**: Real-time Streaming Protocol for NVR communication
- **HTTP/REST**: API communication
- **JPEG/PNG**: Frame image format
- **H.264/H.265**: Video codec (via FFmpeg)

## Configuration

### Environment Variables

```bash
# Required
NVR_HOST=192.168.1.100        # NVR IP address
NVR_USERNAME=admin            # NVR credentials
NVR_PASSWORD=password         # NVR credentials

# Optional
NVR_PORT=554                  # RTSP port (default: 554)
NVR_CHANNEL=1                 # Default camera channel
RTSP_TRANSPORT=tcp            # tcp or udp (default: tcp)
```

### Runtime Configuration

- Model confidence threshold: 0.5
- Feature similarity threshold: 0.75
- Tracker IoU threshold: 0.8
- Tracker feature weight: 0.3
- Max lookback time: 3 hours (10800 seconds)
- Min search window: 5 seconds
- Frame rate: 20 FPS
- Track buffer: 30 frames (1.5s at 20 FPS)

## Error Handling

### Exception Hierarchy

```
RuntimeError
├── Model not loaded
├── Frame extraction failed
├── RTSP connection error
└── FFmpeg execution failed

HTTPException (FastAPI)
├── 400: Bad request (invalid timestamp, invalid duration)
├── 500: Server error (client not initialized)
└── 503: Search error (algorithm failure)
```

### Logging

Uses Python's standard logging:
- INFO: Normal operations (frame extraction, detections)
- WARNING: Recoverable issues (model load warning)
- ERROR: Fatal errors with stack traces

Log format:
```
%(asctime)s [%(levelname)s] %(name)s - %(message)s
```

## Performance Considerations

### Bottlenecks

1. **FFmpeg frame extraction**: ~500-1000ms per frame (network dependent)
2. **RTSP connection setup**: ~200-500ms per extraction
3. **Model inference**: ~50ms on GPU, ~500ms on CPU
4. **Retry logic**: Exponential backoff adds delay

### Optimizations

- Frame-level binary search (99% reduction in detector calls)
- Coarse-to-fine sampling strategy
- Feature caching during search
- Direct codec copy for video clips (no re-encoding)
- Connection reuse within extraction operations
- Confidence threshold filtering before tracking

### Memory Usage

- Model: ~500MB-1GB (depending on resolution)
- Features: ~4KB per detection (1024-dim float32)
- Frame buffer: ~2MB per frame (1920x1080)

## Security

- Credentials stored in environment variables (not code)
- URL-encoded credentials for special characters
- No persistent storage of video data (temporary files only)
- Local processing (no cloud dependencies)
- RTSP transport configurable (TCP recommended for firewall traversal)

## Directory Structure

```
cctv-search/
├── src/cctv_search/
│   ├── api/__init__.py          # FastAPI application
│   ├── nvr/
│   │   └── dahua.py            # Dahua NVR client
│   ├── ai/
│   │   ├── __init__.py         # AI module exports
│   │   ├── rf_detr.py          # RF-DETR detector
│   │   └── byte_tracker.py     # FeatureTracker
│   ├── search/
│   │   ├── __init__.py         # Search module exports
│   │   └── algorithm.py        # Backward temporal search
│   ├── tracker.py              # Tracker integration
│   └── detector.py             # Detector integration
├── scripts/                    # Utility scripts
├── tests/                      # Test suite
├── clips/                      # Generated video clips
├── frames/                     # Extracted/annotated frames
└── docs/                       # Documentation
```
