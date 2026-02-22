# Video Extraction and Tracking Script

Script for extracting video clips from NVR, running RF-DETR detection and feature-based multi-object tracking, and outputting annotated videos.

## Features

- 📹 **Extract video clips** from Dahua NVR via RTSP
- 🤖 **RF-DETR object detection** on every frame
- 🎯 **Feature-based multi-object tracking** with persistent IDs
- 🎬 **Output annotated video** with bounding boxes and track IDs
- 📊 **Track statistics** (duration, frame count, etc.)

## Prerequisites

```bash
# Install ffmpeg (required for video extraction)
brew install ffmpeg  # macOS
# or
apt-get install ffmpeg  # Ubuntu/Debian

# Install Python dependencies
pip install opencv-python numpy
```

## Usage

### Basic Usage

```bash
# Extract 30 seconds and track all objects
python scripts/extract_and_track.py \
    --start "2026-02-19 10:00:00" \
    --duration 30
```

### Track Specific Class

```bash
# Track only people
python scripts/extract_and_track.py \
    --start "2026-02-19 10:00:00" \
    --duration 30 \
    --class person \
    --output people_tracking.mp4

# Track only cars
python scripts/extract_and_track.py \
    --start "2026-02-19 10:00:00" \
    --duration 60 \
    --class car \
    --output car_tracking.mp4
```

### Adjust Detection Settings

```bash
# Higher confidence threshold (fewer false positives)
python scripts/extract_and_track.py \
    --start "2026-02-19 10:00:00" \
    --duration 30 \
    --confidence 0.7 \
    --output high_conf_tracking.mp4

# Lower confidence (more detections)
python scripts/extract_and_track.py \
    --start "2026-02-19 10:00:00" \
    --duration 30 \
    --confidence 0.3 \
    --output sensitive_tracking.mp4
```

### Longer Clips

```bash
# Extract and track 5 minutes
python scripts/extract_and_track.py \
    --start "2026-02-19 10:00:00" \
    --duration 300 \
    --output long_tracking.mp4
```

### Keep Raw Video

```bash
# Save both raw and annotated videos
python scripts/extract_and_track.py \
    --start "2026-02-19 10:00:00" \
    --duration 30 \
    --keep-raw \
    --output annotated.mp4
# Creates: annotated.mp4 and annotated_raw.mp4
```

## Command Line Options

```
usage: extract_and_track.py [-h] --start START [--duration DURATION]
                            [--channel CHANNEL] [--output OUTPUT]
                            [--confidence CONFIDENCE] [--class CLASS]
                            [--fps FPS] [--keep-raw]

Extract video from NVR and run object tracking

optional arguments:
  -h, --help            Show this help message and exit
  --start START         Start timestamp (format: 'YYYY-MM-DD HH:MM:SS')
  --duration DURATION   Duration in seconds (default: 30)
  --channel CHANNEL     Camera channel (default: 1)
  --output OUTPUT       Output video path
  --confidence CONFIDENCE
                        Detection confidence threshold (default: 0.5)
  --class CLASS         Only track this class (e.g., 'person', 'car')
  --fps FPS             Video frame rate (default: 20)
  --keep-raw            Keep the raw video file (not just annotated)
```

## Example Output

```bash
$ python scripts/extract_and_track.py --start "2026-02-19 10:00:00" --duration 30 --class person
============================================================
NVR Video Extraction & Object Tracking
============================================================

Connecting to NVR...
Extracting 30s video from 2026-02-19 10:00:00
NVR Host: 192.168.1.100
RTSP URL: rtsp://admin:****@192.168.1.100:554/cam/playback?...
Running ffmpeg...
✓ Video extracted: /tmp/.../raw_video.mp4
  File size: 15,245,678 bytes

Initializing detector (confidence: 0.5)...
Loading model (may download on first run)...
✓ Model loaded successfully

Initializing feature tracker...
✓ Tracker initialized

Video: 1920x1080 @ 20.0fps, 600 frames
Processing frames...
  Frame 30/600 (5.0%) - 3 tracks
  Frame 60/600 (10.0%) - 4 tracks
  Frame 90/600 (15.0%) - 5 tracks
  ...
  Frame 600/600 (100.0%) - 3 tracks

✓ Annotated video saved: tracked_20260221_155301.mp4

============================================================
✅ Processing complete!
============================================================

Input:  2026-02-19 10:00:00 (+30s)
Output: tracked_20260221_155301.mp4

Statistics:
  Frames processed: 600
  Unique tracks: 8

Track Details:
  Track 1: person
    Frames: 45
    Duration: 45 frames
    First seen: frame 0
    Last seen: frame 44
  Track 2: person
    Frames: 120
    Duration: 120 frames
    First seen: frame 10
    Last seen: frame 129
  ...
```

## Visual Output

The output video includes:

### Bounding Boxes
- **Colored by track ID** - Each object gets a consistent color
- **Track ID label** - Shows "ID:X" for each tracked object
- **Class name** - Person, car, bicycle, etc.
- **Confidence score** - Detection confidence percentage

### Info Overlay
- **Frame number** - Current frame in top-left
- **Active track count** - Number of currently tracked objects

### Color Coding
Track IDs cycle through colors:
- 🟢 Green (ID: 1, 11, 21...)
- 🔵 Blue (ID: 2, 12, 22...)
- 🔴 Red (ID: 3, 13, 23...)
- 🟡 Cyan (ID: 4, 14, 24...)
- 🟣 Magenta (ID: 5, 15, 25...)
- 🟡 Yellow (ID: 6, 16, 26...)
- 🟣 Purple (ID: 7, 17, 27...)
- 🟠 Orange (ID: 8, 18, 28...)
- 🔵 Teal (ID: 9, 19, 29...)
- 🟢 Olive (ID: 10, 20, 30...)

## How It Works

1. **Video Extraction** - Uses ffmpeg to download RTSP stream from NVR
2. **Frame-by-Frame Processing** - Reads video and processes each frame
3. **Object Detection** - RF-DETR detects objects in each frame
4. **Multi-Object Tracking** - FeatureTracker maintains object identities across frames using deep embeddings
5. **Annotation** - Draws bounding boxes with track IDs on each frame
6. **Video Encoding** - Writes annotated frames to output video file

## Performance

Typical processing speed:
- **20 FPS video**: ~2-5 seconds per second of video (depending on hardware)
- **Detection**: ~50ms per frame on GPU, ~200ms on CPU
- **Tracking**: ~5ms per frame

**Total time for 30-second clip**: ~2-5 minutes

## Troubleshooting

### Error: "ffmpeg not found"
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html and add to PATH
```

### Error: "Could not open video"
- Check NVR credentials in environment variables
- Verify camera channel exists
- Ensure RTSP port (554) is accessible

### Slow Processing
- Use GPU if available (RF-DETR supports CUDA)
- Reduce video resolution with `--resolution 1280x720`
- Use higher confidence threshold to filter detections

### No Tracks Found
- Lower confidence threshold: `--confidence 0.3`
- Check if objects are visible in the video
- Verify the `--class` filter isn't too restrictive

### Video Quality Issues
- The script uses H.264 encoding with reasonable quality settings
- For higher quality, modify the ffmpeg command in the script

## Advanced Usage

### Batch Processing Multiple Time Ranges

```bash
#!/bin/bash

TIMES=("09:00:00" "10:00:00" "11:00:00")

for time in "${TIMES[@]}"; do
    python scripts/extract_and_track.py \
        --start "2026-02-19 $time" \
        --duration 60 \
        --class person \
        --output "tracking_${time//:/}.mp4"
done
```

### Extract Specific Event

```bash
# Extract 10 seconds around an event at 10:05:30
python scripts/extract_and_track.py \
    --start "2026-02-19 10:05:25" \
    --duration 10 \
    --confidence 0.6 \
    --output event_tracking.mp4
```

### Compare Different Settings

```bash
# Test different confidence thresholds
for conf in 0.3 0.5 0.7; do
    python scripts/extract_and_track.py \
        --start "2026-02-19 10:00:00" \
        --duration 30 \
        --confidence $conf \
        --output "test_conf_${conf}.mp4"
done
```

## Integration with Search Algorithm

This script demonstrates the full tracking pipeline used by the backward search algorithm:

1. **Frame Extraction** - Same NVR client used by search
2. **Detection** - Same RF-DETR detector
3. **Tracking** - Same FeatureTracker
4. **Visualization** - Helps debug and validate tracking results

The search algorithm (`backward_search`) uses the same components to find when objects appeared in the past.

## Next Steps

1. Test with a short clip first:
   ```bash
   python scripts/extract_and_track.py --start "2026-02-19 10:00:00" --duration 10
   ```

2. Check the output video quality and tracking accuracy

3. Adjust `--confidence` and `--class` settings as needed

4. For longer clips, be patient - processing takes time!
