# CCTV Search API

FastAPI backend for CCTV Search - AI-powered video analysis and object tracking.

## Features

- Frame extraction from NVR via RTSP
- Object detection using RF-DETR
- Feature-based multi-object tracking
- Backward temporal search algorithm
- Video clip generation

## Development

```bash
# Install dependencies
uv sync --dev

# Run dev server
uv run uvicorn cctv_search.api:app --reload

# Run tests
uv run pytest

# Lint code
uv run ruff check .
```

## API Endpoints

- `POST /nvr/frame` - Extract frame at timestamp
- `POST /frames/objects` - Extract frame with object detection
- `POST /ai/detect` - Detect objects
- `POST /video/clip` - Generate video clip
- `POST /search/object` - Search for objects

## Configuration

Copy `.env.example` to `.env` and configure NVR settings:

```bash
NVR_HOST=192.168.1.100
NVR_PORT=554
NVR_USERNAME=admin
NVR_PASSWORD=password
```
