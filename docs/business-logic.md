# Business Logic Overview

## What is CCTV Search?

CCTV Search is an intelligent video analysis system that helps security operators find specific objects or events in recorded CCTV footage from Network Video Recorders (NVR). Instead of manually reviewing hours of video, the system uses AI to automatically detect, track, and search for objects.

## Core Use Cases

### 1. Object Search
Find when a specific object (person, vehicle, bicycle) appeared in camera footage. The system supports:
- **Targeted search**: Find a specific instance of an object across time
- **Appearance-based matching**: Identify the same object by visual and spatial features
- **Temporal tracking**: Trace an object's path through the camera's field of view

### 2. Real-time Detection
Perform AI-powered object detection on live camera streams or archived footage to identify:
- People
- Vehicles (cars, trucks, bicycles, motorcycles)
- Animals
- Custom object classes (depending on model)

### 3. Multi-Object Tracking
Track multiple objects simultaneously across video frames using ByteTrack algorithm:
- Assign unique IDs to detected objects
- Maintain identity across occlusions and temporary disappearances
- Associate detections with consistent object tracks

## Key Business Concepts

### Backward Coarse-to-Fine Search
The system implements an efficient search algorithm that minimizes AI model calls:

1. **Coarse Phase**: Sample video at low frequency (every 30 seconds) to find candidate windows
2. **Medium Phase**: Refine search within promising windows (every 5 seconds)
3. **Fine Phase**: Examine individual frames to pinpoint exact appearance times

**Result**: ~20 detector calls vs ~2,160 for naive frame-by-frame approach (99% reduction)

### Same Object Detection ("SameBike" Predicate)
The core business logic for determining if two detections represent the same physical object:

- **IoU (Intersection over Union)**: Bounding box overlap ≥ 80%
- **Spatial proximity**: Center distance ≤ 50 pixels
- **Label consistency**: Object class must match

Both conditions must be met (AND logic) for positive identification.

### Frame-Level Binary Search
Unlike time-based searches, the system operates at the frame level:
- More accurate for detecting brief appearances
- Handles variable frame rates correctly
- Prevents missing objects between sample points

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
                                        │  (ByteTrack)│
                                        └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │  Search     │
                                        │  Algorithm  │
                                        └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │  Results    │
                                        │  (Tracks)   │
                                        └─────────────┘
```

## Input/Output

### Inputs
- **Video Source**: RTSP streams from Dahua/Hikvision NVRs
- **Search Parameters**: Object class, time range, confidence threshold
- **Reference Object**: Optional sample detection for matching

### Outputs
- **Object Tracks**: Timestamped path of object movement
- **Bounding Boxes**: Coordinates with confidence scores
- **Video Metadata**: Frame indices, timestamps, camera IDs

## Performance Characteristics

- **Detection Speed**: Real-time capable (20+ FPS on GPU)
- **Search Efficiency**: 100x reduction in model calls vs naive approach
- **Tracking Accuracy**: Handles occlusions up to 30 frames (1.5s at 20 FPS)
- **Scalability**: Supports multiple cameras simultaneously

## Limitations

1. **Fixed Cameras Only**: Algorithm assumes static camera position
2. **Class-Dependent Accuracy**: Detection quality varies by object type
3. **Processing Delay**: Frame extraction requires RTSP connection setup
4. **Storage Requirements**: Temporary frame storage during processing
