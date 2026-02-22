# Business Logic Overview

## What is CCTV Search?

CCTV Search is an intelligent video analysis system that helps security operators find specific objects or events in recorded CCTV footage from Network Video Recorders (NVR). Instead of manually reviewing hours of video, the system uses AI to automatically detect, track, and search for objects.

## Core Use Cases

### 1. Object Search
Find when a specific object (person, vehicle, bicycle) appeared in camera footage. The system supports:
- **Targeted search**: Find a specific instance of an object across time
- **Appearance-based matching**: Identify the same object by visual and spatial features
- **Temporal tracking**: Trace an object's path through the camera's field of view
- **First appearance detection**: Find exactly when an object entered the camera view

### 2. Real-time Detection
Perform AI-powered object detection on live camera streams or archived footage to identify:
- People
- Vehicles (cars, trucks, bicycles, motorcycles)
- Animals
- Custom object classes (depending on model)

### 3. Multi-Object Tracking
Track multiple objects simultaneously across video frames using feature-based tracking with deep embeddings:
- Assign unique IDs to detected objects
- Maintain identity across occlusions and temporary disappearances
- Associate detections with consistent object tracks
- Handle complex scenarios like crossing paths and partial occlusion

### 4. Video Clip Generation
Extract video segments from recorded footage:
- Generate clips of specific durations (up to 5 minutes)
- Annotate clips with bounding boxes around tracked objects
- Download clips for evidence or further analysis
- Direct RTSP playback integration for live viewing

## Key Business Concepts

### Backward Coarse-to-Fine Search
The system implements an efficient search algorithm that minimizes AI model calls:

1. **Coarse Phase**: Sample video at low frequency (every 30 seconds) to find candidate windows
2. **Medium Phase**: Refine search within promising windows (every 5 seconds)
3. **Fine Phase**: Binary search at frame level to pinpoint exact appearance times

**Why it works**: Objects typically remain visible for seconds or minutes, so sampling every 30 seconds catches most appearances. The subsequent refinement phases only run when objects are likely present.

**Result**: ~20 detector calls vs ~2,160 for naive frame-by-frame approach (99% reduction)

### Same Object Detection (Multi-Criteria Matching)
The core business logic for determining if two detections represent the same physical object combines multiple criteria:

**Spatial Matching**:
- **IoU (Intersection over Union)**: Bounding box overlap ≥ 80%
- **Center Distance**: Object centers within 50 pixels

**Visual Matching** (using deep features):
- **Feature Similarity**: Cosine similarity ≥ 0.75
- Extracted from RF-DETR transformer encoder
- Robust to appearance changes and partial occlusion

**Combined Logic**:
- Both IoU AND distance must pass (strict spatial matching)
- Features used for re-identification during occlusion
- Label consistency required (object class must match)

### Frame-Level Binary Search
Unlike time-based searches, the system operates at the frame level for precision:
- More accurate for detecting brief appearances
- Handles variable frame rates correctly
- Prevents missing objects between sample points
- Binary search converges in O(log n) time

Example: Searching a 5-second window (100 frames @ 20 FPS):
- Frame 50: Check if object present
- If yes: Check frame 25; If no: Check frame 75
- Continue until exact first appearance frame found

### Feature-Based Re-identification
The system uses deep learning features for robust object matching:

**Feature Extraction**:
- Extracted from transformer encoder output
- 1024-dimensional feature vectors
- Captures object appearance (color, shape, texture)

**Feature Matching**:
- Cosine similarity for comparison
- Threshold: 0.75 (75% similarity)
- Useful when objects are temporarily occluded
- Maintains identity across gaps in detection

**Use Case**: A person walks behind a column and reappears on the other side. Spatial matching might fail due to position change, but feature matching maintains the track ID.

## Data Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   NVR       │────▶│  Frame      │────▶│  AI Model   │
│   Device    │     │  Extraction │     │  (RF-DETR)  │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                                │
                                                ▼
                                         ┌─────────────┐
                                         │  Tracker    │
                                         │(FeatureTrack)│
                                         └──────┬──────┘
                                                │
                                                ▼
                                         ┌─────────────┐
                                         │  Search     │
                                         │  Algorithm  │
                                         └──────┬──────┘
                                                │
                        ┌───────────────────────┼───────────────────────┐
                        ▼                       ▼                       ▼
                 ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
                 │   Results   │        │  Annotated  │        │   Video     │
                 │  (Tracks)   │        │   Frames    │        │   Clips     │
                 └─────────────┘        └─────────────┘        └─────────────┘
```

## Workflows

### Workflow 1: Find Object First Appearance

```
1. Select a frame where object is visible
2. System extracts frame and detects objects
3. User selects target object from detection list
4. System runs backward search:
   a. Coarse phase (30s intervals)
   b. Medium phase (5s intervals)  
   c. Binary search (frame level)
5. Returns first appearance timestamp
6. Optionally generate 15s clip before first appearance
```

**Use Case**: Security needs to know when a suspicious person first appeared on camera.

### Workflow 2: Generate Annotated Video Clip

```
1. Specify start timestamp and duration
2. System extracts video segment from NVR
3. System processes each frame:
   a. Detect objects
   b. Update tracks
   c. Draw bounding boxes
4. Saves annotated video
5. Returns download URL
```

**Use Case**: Generate evidence video of an incident with object tracking overlay.

### Workflow 3: Real-time Object Detection

```
1. Specify timestamp and camera channel
2. System extracts frame
3. Runs object detection
4. Assigns track IDs
5. Saves annotated frame
6. Returns detection list and image path
```

**Use Case**: Quick check of who/what is visible at a specific moment.

## Input/Output

### Inputs
- **Video Source**: RTSP streams from Dahua/Hikvision NVRs
- **Search Parameters**: Object class, time range, confidence threshold
- **Reference Object**: Sample detection with bounding box and features
- **Camera Configuration**: Channel number, credentials, transport settings

### Outputs
- **Object Tracks**: Timestamped path of object movement
- **Bounding Boxes**: Coordinates with confidence scores
- **Feature Embeddings**: Deep learning features for re-identification
- **Annotated Images**: Frames with bounding box overlays
- **Video Clips**: Extracted segments with optional annotations
- **Search Results**: First/last appearance times, confidence scores

## Performance Characteristics

- **Detection Speed**: Real-time capable (20+ FPS on GPU, 2+ FPS on CPU)
- **Search Efficiency**: 99% reduction in model calls vs naive approach
- **Tracking Accuracy**: 
  - Handles occlusions up to 30 frames (1.5s at 20 FPS)
  - Re-identification via features during longer occlusions
- **Scalability**: Supports multiple cameras simultaneously
- **Memory Usage**: ~500MB-1GB for model + ~2MB per frame

## Limitations

1. **Fixed Cameras Only**: Algorithm assumes static camera position
2. **Class-Dependent Accuracy**: Detection quality varies by object type
3. **Processing Delay**: Frame extraction requires RTSP connection setup (~500ms)
4. **Storage Requirements**: Temporary frame storage during processing
5. **Feature Extraction Overhead**: ~10ms additional per detection
6. **Network Dependency**: RTSP stream quality affects extraction success
7. **Time Synchronization**: Requires NVR time to be accurate

## Best Practices

### Search Configuration
- Start with shorter search windows (1-2 hours) for faster results
- Use higher confidence thresholds (0.7+) for critical searches
- Ensure reference object has clear features and good detection confidence

### Performance Optimization
- Use GPU for real-time processing
- Pre-warm model to avoid cold-start delays
- Adjust RTSP transport (TCP vs UDP) based on network stability
- Increase track buffer for cameras with lower frame rates

### Accuracy Improvement
- Use feature matching for scenarios with frequent occlusions
- Adjust IoU threshold based on object movement speed
- Enable higher resolution (864x864) for small object detection

## Business Value

### Time Savings
- Manual review: 3 hours to review 1 hour of footage
- Automated search: 2-5 minutes to find target object
- **ROI**: 36-90x time reduction

### Accuracy Benefits
- Human error rate: ~20% miss rate for brief appearances
- Automated detection: <1% miss rate for configured thresholds
- Consistent 24/7 operation without fatigue

### Use Case Examples

**Retail Security**:
- Find when a shoplifter first entered the store
- Track suspicious behavior across multiple camera angles
- Generate evidence clips for law enforcement

**Facility Management**:
- Track vehicle entry/exit times
- Monitor restricted area access
- Audit security guard patrol routes

**Traffic Management**:
- Track vehicle movements through intersections
- Identify parking violations
- Monitor traffic flow patterns
