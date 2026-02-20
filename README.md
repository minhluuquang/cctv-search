# CCTV Search

A FastAPI-based server for searching CCTV footage using AI object detection.

## Architecture

- **FastAPI Server**: HTTP API for video management and AI analysis
- **NVR Module**: Connect to Network Video Recorders (Hikvision support)
- **AI Module**: Object detection using YOLO models

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) - Modern Python package manager

## Setup

```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

## Usage

```bash
# Run the server
uv run cctv-search

# Or with uvicorn directly
uv run uvicorn cctv_search.api:app --reload
```

## API Endpoints

### NVR Management
- `POST /nvr/connect` - Connect to NVR device
- `POST /nvr/disconnect` - Disconnect from NVR
- `GET /nvr/cameras` - List available cameras
- `GET /nvr/cameras/{camera_id}/stream` - Get live stream URL

### AI Analysis
- `POST /ai/detect` - Detect objects in video/live stream

## Development

```bash
# Run tests
uv run pytest

# Run a single test
uv run pytest tests/test_nvr.py::test_nvr_client_connects_successfully -v

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
├── src/
│   └── cctv_search/
│       ├── __init__.py      # Main entry point
│       ├── api/             # FastAPI routes
│       │   └── __init__.py
│       ├── nvr/             # NVR client modules
│       │   ├── __init__.py
│       │   └── hikvision.py
│       └── ai/              # AI/ML modules
│           ├── __init__.py
│           └── yolo.py
└── tests/
    ├── test_placeholder.py
    ├── test_nvr.py
    ├── test_ai.py
    └── test_api.py
```
