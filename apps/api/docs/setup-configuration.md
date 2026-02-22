# Setup and Configuration Guide

## Prerequisites

### System Requirements

- **OS**: Linux, macOS, or Windows with WSL2
- **Python**: 3.12 or higher
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 2GB free space for models and dependencies
- **GPU**: Optional but recommended (CUDA-capable for GPU acceleration)
- **Network**: Access to NVR via RTSP (port 554)

### Required Software

1. **uv** - Modern Python package manager
   ```bash
   # Install uv
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **FFmpeg** - Frame extraction from RTSP streams
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   
   # Windows (via chocolatey)
   choco install ffmpeg
   ```

3. **Git** - Version control

4. **RTSP-capable NVR** - Dahua or Hikvision NVR with network access

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd cctv-search
```

### 2. Install Dependencies

```bash
# Install all dependencies (production + dev)
uv sync --dev

# Or just production dependencies
uv sync
```

This will:
- Create a virtual environment in `.venv/`
- Install Python 3.12 (if not present)
- Install all packages from `pyproject.toml`
- Lock dependencies in `uv.lock`

### 3. Verify Installation

```bash
# Run tests
uv run pytest -v

# Check code style
uv run ruff check .

# Format code
uv run ruff format .
```

## Configuration

### Environment Variables

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` with your NVR credentials:

```bash
# Dahua NVR Configuration
NVR_HOST=192.168.1.220        # Your NVR IP address
NVR_PORT=554                  # RTSP port (default: 554)
NVR_USERNAME=admin            # NVR username
NVR_PASSWORD=your_password    # NVR password
NVR_CHANNEL=1                 # Default camera channel

# Optional: RTSP transport protocol
RTSP_TRANSPORT=tcp            # tcp or udp (recommended: tcp)
```

### Configuration Options

#### NVR Settings

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `NVR_HOST` | NVR IP address or hostname | - | Yes |
| `NVR_PORT` | RTSP port number | 554 | No |
| `NVR_USERNAME` | NVR login username | - | Yes |
| `NVR_PASSWORD` | NVR login password | - | Yes |
| `NVR_CHANNEL` | Default camera channel | 1 | No |
| `RTSP_TRANSPORT` | Transport protocol | tcp | No |

**RTSP Transport Notes**:
- `tcp`: More reliable, works through firewalls (recommended)
- `udp`: Lower latency but may drop packets

#### Model Settings (Runtime)

Configure detection parameters in code:

```python
from cctv_search.ai import RFDetrDetector

# Adjust confidence threshold
detector = RFDetrDetector(confidence_threshold=0.6)

# Load model with specific resolution
detector.load_model(model_id="rfdetr_base", resolution=560)
# Options: 560 (faster), 864 (more accurate)

# Adjust tracker parameters
from cctv_search.ai import FeatureTracker

tracker = FeatureTracker(
    track_thresh=0.5,        # Detection confidence threshold
    match_thresh=0.8,        # IoU threshold for matching
    track_buffer=30,         # Frames to keep lost tracks
    frame_rate=20,           # Video FPS
    feature_weight=0.3,      # Feature matching weight
    feature_threshold=0.75   # Feature similarity threshold
)
```

#### Search Algorithm Settings

```python
from cctv_search.search import BackwardTemporalSearch

search = BackwardTemporalSearch(
    video_decoder=decoder,
    detector=detector,
    tracker=tracker,
    fps=20.0
)
```

## Running the Application

### Development Mode

```bash
# Run with auto-reload (development)
uv run uvicorn cctv_search.api:app --reload --host 0.0.0.0 --port 8000

# Or using the CLI
uv run cctv-search
```

### Production Mode

```bash
# Run without reload (production)
uv run uvicorn cctv_search.api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Access API Documentation

Once running, open:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Testing

### Run All Tests

```bash
uv run pytest
```

### Run Specific Test

```bash
# Run single test file
uv run pytest tests/test_nvr.py -v

# Run specific test function
uv run pytest tests/test_nvr.py::test_nvr_client_init_with_params -v

# Run with coverage
uv run pytest --cov=src/cctv_search --cov-report=term-missing
```

### Run Integration Tests

```bash
# Test NVR connection
uv run python scripts/test_connection.py

# Test frame extraction
uv run python scripts/test_extract_frame.py

# Test NVR diagnostics
uv run python scripts/test_nvr_diagnose.py

# Test RF-DETR detector
uv run python scripts/test_detector.py

# Test RF-DETR model loading
uv run python scripts/test_rfdetr.py

# Test object matching
uv run python scripts/test_matching.py
```

## Utility Scripts

The `scripts/` directory contains utility scripts for development and testing:

### Frame and Video Extraction
- **`extract_and_track.py`** - Extract frames and track objects across video
- **`extract_and_detect.py`** - Extract frames and detect objects

### Testing and Diagnostics
- **`test_connection.py`** - Test basic NVR connection
- **`test_extract_frame.py`** - Test single frame extraction
- **`test_nvr_diagnose.py`** - Comprehensive NVR diagnostics
- **`test_detector.py`** - Test RF-DETR detection on images
- **`test_rfdetr.py`** - Test RF-DETR model loading
- **`test_matching.py`** - Test object matching algorithms

### Usage Examples

```bash
# Test NVR connection
uv run python scripts/test_connection.py

# Extract and track objects in a video segment
uv run python scripts/extract_and_track.py

# Test detection on a test image
uv run python scripts/test_detector.py
```

## Troubleshooting

### Common Issues

#### FFmpeg Not Found

```
RuntimeError: Failed to extract frame: ffmpeg not found
```

**Solution**: Install FFmpeg and ensure it's in PATH:
```bash
which ffmpeg  # macOS/Linux
where ffmpeg  # Windows
```

#### Model Loading Fails

```
RuntimeError: RF-DETR not installed
```

**Solution**: Reinstall dependencies:
```bash
uv sync --dev
```

Or install manually:
```bash
uv pip install rfdetr
```

#### RTSP Connection Timeout

```
RuntimeError: Failed to extract frame: Connection timed out
```

**Solutions**:
1. Check NVR IP address and port
2. Verify RTSP is enabled on NVR (usually in Network settings)
3. Try switching transport protocol (TCP/UDP):
   ```bash
   export RTSP_TRANSPORT=udp
   ```
4. Check firewall rules (port 554 must be open)
5. Verify NVR credentials are correct
6. Check NVR network connection: `ping <NVR_HOST>`

#### GPU Not Available

```
RuntimeError: CUDA out of memory
# or
WARNING: Running on CPU (GPU not available)
```

**Solutions**:
1. Check CUDA installation:
   ```bash
   uv run python -c "import torch; print(torch.cuda.is_available())"
   ```
2. Reduce batch size or model resolution:
   ```python
   detector.load_model(resolution=560)  # Instead of 864
   ```
3. Use CPU inference (slower but no GPU needed):
   ```bash
   export CUDA_VISIBLE_DEVICES=""
   ```
4. Close other GPU applications
5. Add more GPU memory or use a GPU with more VRAM

#### Out of Memory

```
RuntimeError: CUDA out of memory
# or
Killed (process terminated)
```

**Solutions**:
1. Reduce model resolution (560 instead of 864)
2. Process shorter video clips
3. Close other applications
4. Add more RAM (16GB recommended)
5. Use swap space for temporary files

#### Search Not Finding Objects

```
Object not found in specified search window
```

**Solutions**:
1. Increase search duration:
   ```json
   {
     "search_duration_seconds": 7200  // 2 hours instead of 1
   }
   ```
2. Lower confidence threshold in tracker
3. Check that reference object bbox is accurate
4. Verify NVR has footage for that time period
5. Check for time zone differences between system and NVR

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or set environment variable:
```bash
export LOG_LEVEL=DEBUG
uv run cctv-search
```

### Log Files

Logs are output to stdout/stderr. To save to file:
```bash
uv run cctv-search 2>&1 | tee cctv_search.log
```

## Advanced Configuration

### Custom Model Path

Download and use a custom RF-DETR model:

```python
from cctv_search.ai import RFDetrDetector

detector = RFDetrDetector()
detector.load_model()  # Downloads to ~/.cache/rfdetr/
```

To use a specific cached model:
```python
import os
os.environ["RFDETR_CACHE_DIR"] = "/path/to/models"
detector.load_model()
```

### Proxy Configuration

If behind a corporate proxy:

```bash
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
uv sync
```

### GPU Acceleration

Ensure CUDA is properly installed:

```bash
# Check CUDA availability
uv run python -c "import torch; print(torch.cuda.is_available())"
uv run python -c "import torch; print(torch.cuda.get_device_name(0))"

# If False, install CUDA-enabled PyTorch
uv pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu118
```

### Multiple Cameras

Configure multiple cameras by passing channel parameter in API calls:

```python
# Channel 1
requests.post("/frames/objects", json={
    "timestamp": "2024-01-15T14:30:00",
    "channel": 1
})

# Channel 2
requests.post("/frames/objects", json={
    "timestamp": "2024-01-15T14:30:00",
    "channel": 2
})
```

### Directory Structure

```
cctv-search/
├── .env                    # Environment configuration
├── pyproject.toml          # Project dependencies
├── uv.lock                # Locked dependency versions
├── src/
│   └── cctv_search/
│       ├── __init__.py     # Main entry point
│       ├── api/            # FastAPI application
│       ├── nvr/            # NVR client modules
│       ├── ai/             # AI/ML modules
│       ├── search/         # Search algorithms
│       ├── detector.py     # Detector integration
│       └── tracker.py      # Object tracker integration
├── tests/                  # Test suite
├── scripts/                # Utility scripts
├── clips/                  # Generated video clips
├── frames/                 # Extracted/annotated frames
└── docs/                   # Documentation
```

## Performance Tuning

### Optimize for Speed

1. Use GPU (10x faster than CPU)
2. Use lower model resolution (560 vs 864)
3. Reduce track_buffer for faster processing
4. Use TCP transport (more reliable, less retry overhead)

### Optimize for Accuracy

1. Use higher model resolution (864 vs 560)
2. Increase confidence threshold (0.7 vs 0.5)
3. Enable feature matching (feature_weight=0.3)
4. Lower IoU threshold for better recall (0.7 vs 0.8)

### Optimize for Memory

1. Use CPU instead of GPU
2. Reduce batch size
3. Process shorter clips
4. Clear frames/clips directory regularly

## Next Steps

1. Read [Business Logic](business-logic.md) for understanding use cases
2. Review [Technical Architecture](technical-architecture.md) for system design
3. Check [API Reference](api-reference.md) for programming interface
4. See `AGENTS.md` for development guidelines

## Getting Help

- Check logs for error messages
- Review troubleshooting section above
- Verify NVR configuration with `scripts/test_nvr_diagnose.py`
- Test basic connectivity with `scripts/test_connection.py`
