# Setup and Configuration Guide

## Prerequisites

### System Requirements

- **OS**: Linux, macOS, or Windows with WSL2
- **Python**: 3.12 or higher
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 2GB free space for models and dependencies
- **GPU**: Optional but recommended (CUDA-capable for GPU acceleration)

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
RTSP_TRANSPORT=tcp            # tcp or udp
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

# Adjust tracker parameters
from cctv_search.ai import ByteTrackTracker

tracker = ByteTrackTracker(
    track_thresh=0.5,      # Detection confidence threshold
    match_thresh=0.8,      # IoU threshold for matching
    track_buffer=30,       # Frames to keep lost tracks
    frame_rate=20          # Video FPS
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
uv run python test_connection.py

# Test frame extraction
uv run python test_extract_frame.py

# Test NVR diagnostics
uv run python test_nvr_diagnose.py
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

#### RTSP Connection Timeout

```
RuntimeError: Failed to extract frame: Connection timed out
```

**Solutions**:
1. Check NVR IP address and port
2. Verify RTSP is enabled on NVR
3. Try switching transport protocol (TCP/UDP)
4. Check firewall rules

#### Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Reduce batch size
2. Use CPU inference (slower)
3. Close other applications
4. Add more RAM or GPU memory

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

## Advanced Configuration

### Custom Model Path

Download and use a custom RF-DETR model:

```python
from cctv_search.ai import RFDetrDetector

detector = RFDetrDetector()
detector.load_model()  # Downloads to ~/.cache/rfdetr/
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

# If False, install CUDA-enabled PyTorch
uv pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu118
```

## Directory Structure

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
│       └── tracker.py      # Object tracker integration
├── tests/                  # Test suite
├── docs/                   # Documentation
└── scripts/                # Utility scripts
```

## Next Steps

1. Read [Business Logic](business-logic.md) for understanding use cases
2. Review [Technical Architecture](technical-architecture.md) for system design
3. Check [API Reference](api-reference.md) for programming interface
4. See `AGENTS.md` for development guidelines
