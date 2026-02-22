# NVR Frame Extraction and Detection Script

Combined script that extracts frames from Dahua NVR, runs RF-DETR object detection, and outputs annotated images with bounding boxes.

## Features

- 📹 **Extract frames** from Dahua NVR at specific timestamps
- 🤖 **Run RF-DETR** object detection
- 🎨 **Draw bounding boxes** with labels and confidence scores
- 💾 **Save annotated images** for review

## Prerequisites

```bash
# Install required packages
pip install rfdetr opencv-python numpy

# Or use the project's package manager
uv add rfdetr opencv-python
```

## Usage

### Extract from NVR and Detect

```bash
# Extract at specific time and detect objects
python scripts/extract_and_detect.py --timestamp "2026-02-19 10:00:00"

# Specify channel and confidence threshold
python scripts/extract_and_detect.py \
    --timestamp "2026-02-19 10:00:00" \
    --channel 2 \
    --confidence 0.7

# Custom output path
python scripts/extract_and_detect.py \
    --timestamp "2026-02-19 10:00:00" \
    --output my_detection.jpg

# Also save raw frame (without boxes)
python scripts/extract_and_detect.py \
    --timestamp "2026-02-19 10:00:00" \
    --save-raw
```

### Test with Local Image (No NVR Required)

```bash
# Test with your own image
python scripts/extract_and_detect.py --image /path/to/your/image.jpg

# Adjust confidence threshold
python scripts/extract_and_detect.py \
    --image /path/to/your/image.jpg \
    --confidence 0.6
```

## Command Line Options

```
usage: extract_and_detect.py [-h] [--timestamp TIMESTAMP] [--image IMAGE]
                             [--channel CHANNEL] [--output OUTPUT]
                             [--confidence CONFIDENCE] [--save-raw]

Extract frame from NVR and run object detection

optional arguments:
  -h, --help            Show this help message and exit
  --timestamp TIMESTAMP
                        Timestamp to extract (format: 'YYYY-MM-DD HH:MM:SS')
  --image IMAGE         Use local image instead of NVR (for testing)
  --channel CHANNEL     Camera channel (default: 1)
  --output OUTPUT       Output path for annotated image
  --confidence CONFIDENCE
                        Detection confidence threshold (default: 0.5)
  --save-raw            Also save the raw (non-annotated) frame
```

## Example Output

```bash
$ python scripts/extract_and_detect.py --timestamp "2026-02-19 10:00:00"
============================================================
NVR Frame Extraction & Object Detection
============================================================

Connecting to NVR...
Extracting frame at 2026-02-19 10:00:00 from channel 1
NVR Host: 192.168.1.100
✓ Frame extracted: frame_2026_02_19_100000_ch1.png
  File size: 1,245,678 bytes

Initializing RF-DETR detector (confidence: 0.5)...
Loading model (may download on first run)...
✓ Model loaded successfully

Running detection on frame_2026_02_19_100000_ch1.png...
  Image size: 1,245,678 bytes
✓ Detection complete in 45.2ms
✓ Found 3 objects:
  1. BICYCLE
     Confidence: 94.5%
     Location: (523, 891)
     Size: 267x412
  2. PERSON
     Confidence: 87.3%
     Location: (1241, 623)
     Size: 89x156
  3. CAR
     Confidence: 76.2%
     Location: (89, 445)
     Size: 445x223

Drawing boxes on 1920x1080 image...
✓ Annotated image saved: detected_20260221_154230.jpg
  File size: 892,456 bytes

============================================================
✅ Processing complete!
============================================================

Input:  frame_2026_02_19_100000_ch1.png
Output: detected_20260221_154230.jpg
Objects detected: 3
```

## Color Coding

Bounding boxes are color-coded by class:

- 🟢 **Green** - Person
- 🔵 **Blue** - Bicycle
- 🔴 **Red** - Car
- 🟡 **Cyan** - Motorcycle
- 🟣 **Magenta** - Truck
- 🟡 **Yellow** - Bus
- ⚪ **White** - Other classes

## Environment Variables

The script uses the same environment variables as the NVR client:

```bash
export NVR_HOST=192.168.1.100
export NVR_USERNAME=admin
export NVR_PASSWORD=your_password
export NVR_PORT=554
```

Or create a `.env` file in the project root.

## Troubleshooting

### Error: "No module named 'rfdetr'"
```bash
pip install rfdetr
```

### Error: "No module named 'cv2'"
```bash
pip install opencv-python
```

### Error: "Could not load image"
- Check the image path is correct
- Ensure image format is supported (PNG, JPG)
- Try a different image

### Error: "Failed to connect to NVR"
- Verify NVR_HOST environment variable
- Check network connectivity
- Verify username and password
- Ensure RTSP port is open (default: 554)

### No objects detected
- Try lowering confidence threshold: `--confidence 0.3`
- Check if objects are clearly visible in frame
- Verify RF-DETR model loaded correctly

## Integration with Search Algorithm

This script demonstrates the full pipeline that the search algorithm uses:

1. **Frame Extraction** - NVR client extracts frame
2. **Object Detection** - RF-DETR detects objects
3. **Annotation** - Bounding boxes drawn for visualization

The search algorithm (`backward_search`) does steps 1-2 internally, iterating through frames to find when objects appeared.

## Performance Tips

- **First run** downloads model (~100MB), subsequent runs are faster
- **GPU acceleration** available if CUDA installed
- Typical detection speed: 20-50ms per frame on GPU, 100-200ms on CPU
- Use `--confidence 0.7` to reduce false positives
- Use `--confidence 0.3` to catch more objects

## Examples

### Find all people in a frame
```bash
python scripts/extract_and_detect.py \
    --timestamp "2026-02-19 10:00:00" \
    --confidence 0.6 \
    --output people_detected.jpg
```

### Batch process multiple timestamps
```bash
for time in "09:00:00" "10:00:00" "11:00:00"; do
    python scripts/extract_and_detect.py \
        --timestamp "2026-02-19 $time" \
        --output "detection_$time.jpg"
done
```

### Test detection on your own images
```bash
python scripts/extract_and_detect.py \
    --image ~/Pictures/my_photo.jpg \
    --confidence 0.5 \
    --output my_photo_detected.jpg
```
