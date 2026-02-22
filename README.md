# CCTV Search

A FastAPI-based server for searching CCTV footage using AI object detection and tracking, with a modern Next.js frontend.

## Architecture

This is a **monorepo** containing:

- **API** (`apps/api/`): FastAPI backend for video management and AI analysis
- **Web** (`apps/web/`): Next.js 15 frontend with shadcn/ui

### Tech Stack

**Backend:**
- **FastAPI Server**: HTTP API for video management and AI analysis
- **NVR Module**: Connect to Network Video Recorders (Dahua/Hikvision support)
- **AI Module**: Object detection using RF-DETR models
- **Tracker Module**: Feature-based tracking using deep embeddings
- **Search Module**: Backward temporal search algorithm

**Frontend:**
- **Next.js 15**: React framework with App Router
- **TypeScript**: Type-safe development
- **shadcn/ui**: Modern UI components
- **TanStack Query**: Server state management
- **Tailwind CSS**: Utility-first styling

## Requirements

- **Node.js**: 20+ (with pnpm 9+)
- **Python**: 3.12+
- **Package Manager**: [uv](https://docs.astral.sh/uv/) for Python, pnpm for Node.js

## Setup

### 1. Clone and Install

```bash
# Clone repository
git clone <repository-url>
cd cctv-search

# Install Node.js dependencies
pnpm install

# Install Python dependencies
cd apps/api
uv sync --dev
cd ../..
```

### 2. Configuration

Create a `.env` file in the root:

```bash
# NVR Configuration
NVR_HOST=192.168.1.220
NVR_PORT=554
NVR_USERNAME=admin
NVR_PASSWORD=your_password
NVR_CHANNEL=1
RTSP_TRANSPORT=tcp

# Frontend
API_URL=http://localhost:8000
```

Copy to API app:
```bash
cp .env apps/api/.env
```

## Development

### Option 1: Using Turbo (Recommended)

```bash
# Start all services
pnpm dev

# Or start individually
pnpm api:dev      # Backend only
pnpm --filter @cctv-search/web dev  # Frontend only
```

### Option 2: Using Docker Compose

```bash
# Build and start all services
pnpm docker:up

# View logs
pnpm docker:logs

# Stop services
pnpm docker:down
```

### Option 3: Manual

**Terminal 1 - API:**
```bash
cd apps/api
uv run uvicorn cctv_search.api:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Web:**
```bash
cd apps/web
pnpm dev
```

## Project Structure

```
cctv-search/
├── apps/
│   ├── api/                    # FastAPI backend
│   │   ├── src/cctv_search/    # Source code
│   │   ├── tests/              # Test suite
│   │   ├── docs/               # Documentation
│   │   ├── scripts/            # Utility scripts
│   │   ├── pyproject.toml      # Python dependencies
│   │   └── Dockerfile
│   └── web/                    # Next.js frontend
│       ├── app/                # App Router pages
│       ├── components/         # React components
│       ├── lib/                # Utilities
│       ├── package.json
│       └── Dockerfile
├── packages/
│   └── shared-types/           # Shared TypeScript types
│       ├── src/
│       └── package.json
├── package.json                # Root workspace config
├── pnpm-workspace.yaml         # pnpm workspace definition
├── turbo.json                  # Turbo pipeline config
├── docker-compose.yml          # Local development
└── README.md
```

## Available Scripts

**Root:**
```bash
pnpm dev          # Start all services with Turbo
pnpm build        # Build all apps
pnpm lint         # Lint all apps
pnpm test         # Run all tests
pnpm clean        # Clean build artifacts
```

**API:**
```bash
pnpm api:dev      # Start FastAPI dev server
pnpm api:lint     # Run ruff linter
pnpm api:format   # Run ruff formatter
pnpm api:test     # Run pytest
```

**Docker:**
```bash
pnpm docker:up    # Start all services
pnpm docker:down  # Stop services
pnpm docker:build # Rebuild images
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

### Feature-Based Tracking
Custom tracking implementation using:
- Deep feature embeddings from RF-DETR
- Track ID assignment and maintenance
- Robust to occlusion using cosine similarity

### Backward Temporal Search
Efficient coarse-to-fine search algorithm:
1. **Coarse Phase**: Sample at 30-second intervals
2. **Medium Phase**: Refine to 5-second intervals  
3. **Fine Phase**: Binary search at frame level

Result: ~99% reduction in AI model calls vs naive frame-by-frame search

## Development Guidelines

See `apps/api/AGENTS.md` for:
- Python code style guidelines
- Testing procedures
- Linting and formatting commands

### Adding New Components

```bash
cd apps/web
pnpm dlx shadcn@latest add <component-name>
```

### Adding Shared Types

```bash
cd packages/shared-types
# Edit src/index.ts
pnpm build
```

## Documentation

- [API Documentation](apps/api/docs/api-reference.md)
- [Business Logic](apps/api/docs/business-logic.md)
- [Technical Architecture](apps/api/docs/technical-architecture.md)
- [Setup Guide](apps/api/docs/setup-configuration.md)

## License

[Add license information here]
