# CCTV Search

A FastAPI-based server for searching CCTV footage using AI object detection and tracking.

## Architecture

- **FastAPI Server**: HTTP API for video management and AI analysis
- **NVR Module**: Connect to Network Video Recorders (Dahua/Hikvision support)
- **AI Module**: Object detection using RF-DETR models with deep feature extraction
- **Tracker Module**: Multi-object tracking with feature-based matching using RF-DETR embeddings
- **Search Module**: Backward temporal search for finding objects in video

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) - Modern Python package manager
- FFmpeg (for frame and video extraction from RTSP streams)

## Setup

```bash
# Install dependencies
uv sync --dev

# Activate virtual environment
source .venv/bin/activate
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Dahua NVR Configuration
NVR_HOST=192.168.1.220        # Your NVR IP address
NVR_PORT=554                  # RTSP port (default: 554)
NVR_USERNAME=admin            # NVR username
NVR_PASSWORD=your_password    # NVR password
NVR_CHANNEL=1                 # Default camera channel
RTSP_TRANSPORT=tcp            # tcp or udp
```

## Usage

```bash
# Run the server
uv run cctv-search

# Or with uvicorn directly (with reload)
uv run uvicorn cctv_search.api:app --reload
```

## API Endpoints

### Frame Operations
- `POST /nvr/frame` - Extract a single frame at a specific timestamp
- `POST /frames/objects` - Extract frame, detect/track objects, save annotated image

### AI Analysis
- `POST /ai/detect` - Detect objects in video/live stream

### Video Clips
- `POST /video/clip` - Generate a video clip from a time range

### Object Search
- `POST /search/object` - Search backward in time to find when an object first appeared

## Key Features

### Object Detection with RF-DETR
Real-time object detection using RF-DETR transformer model with:
- Deep feature extraction for robust object matching
- Support for multiple object classes (person, bicycle, car, etc.)
- Configurable confidence thresholds

### Multi-Object Tracking
Feature-based tracking with:
- Deep feature embeddings from RF-DETR for matching
- Track ID assignment and maintenance
- Robust to occlusion using cosine similarity

### Backward Temporal Search
Efficient coarse-to-fine search algorithm:
1. **Coarse Phase**: Sample at 30-second intervals
2. **Medium Phase**: Refine to 5-second intervals  
3. **Fine Phase**: Binary search at frame level

Result: ~99% reduction in AI model calls vs naive frame-by-frame search

### Video Clip Generation
Extract video segments from NVR using FFmpeg with:
- Configurable duration (up to 5 minutes)
- Direct RTSP playback integration
- Automatic annotation of tracked objects

## Development

```bash
# Run tests
uv run pytest

# Run a single test
uv run pytest tests/test_nvr.py::test_nvr_client_init_with_params -v

# Lint code
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check . --fix

# Format code
uv run ruff format .

# Add dependencies
uv add <package>

# Add dev dependencies
uv add --dev <package>
```

## Project Structure

```
.
├── pyproject.toml         # Project configuration
├── uv.lock               # Dependency lock file
├── README.md
├── AGENTS.md             # AI coding agent guidelines
├── src/
│   └── cctv_search/
│       ├── __init__.py      # Main entry point
│       ├── api/             # FastAPI routes and endpoints
│       ├── nvr/             # NVR client modules (Dahua)
│       ├── ai/              # AI/ML modules (RF-DETR, FeatureTracker)
│       ├── search/          # Video search algorithms
│       ├── detector.py      # Detector integration
│       └── tracker.py       # Tracker integration
├── tests/                   # Test suite
├── scripts/                 # Utility scripts
│   ├── extract_and_track.py
│   ├── extract_and_detect.py
│   ├── test_connection.py
│   ├── test_detector.py
│   └── ...
└── docs/                    # Documentation
    ├── api-reference.md
    ├── business-logic.md
    ├── setup-configuration.md
    └── technical-architecture.md
```

## Scripts

The `scripts/` directory contains utility scripts for testing and development:

- `extract_and_track.py` - Extract frames and track objects across video
- `extract_and_detect.py` - Extract frames and detect objects
- `test_connection.py` - Test NVR connection
- `test_detector.py` - Test RF-DETR detection
- `test_rfdetr.py` - Test RF-DETR model loading
- `test_matching.py` - Test object matching logic
- `test_extract_frame.py` - Test frame extraction
- `test_nvr_diagnose.py` - NVR diagnostics

## Documentation

- [API Reference](docs/api-reference.md) - Complete API documentation
- [Business Logic](docs/business-logic.md) - Use cases and concepts
- [Technical Architecture](docs/technical-architecture.md) - System design
- [Setup & Configuration](docs/setup-configuration.md) - Installation guide

See `AGENTS.md` for detailed coding guidelines.
