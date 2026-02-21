# CCTV Search Documentation

Welcome to the CCTV Search documentation. This documentation covers business logic, technical architecture, API reference, and setup instructions.

## Quick Start

New to the project? Start here:

1. **[Setup and Configuration](setup-configuration.md)** - Installation and configuration guide
2. **[Business Logic](business-logic.md)** - Understanding the system's purpose and use cases
3. **[API Reference](api-reference.md)** - Programming interface and examples
4. **[Technical Architecture](technical-architecture.md)** - System design and implementation details

## Documentation Index

### For Users

- **[Setup and Configuration](setup-configuration.md)**
  - Installation instructions
  - Environment configuration
  - Running the application
  - Troubleshooting guide

- **[Business Logic](business-logic.md)**
  - Use cases and workflows
  - Backward coarse-to-fine search algorithm
  - Same object detection logic
  - Performance characteristics

### For Developers

- **[API Reference](api-reference.md)**
  - REST API endpoints
  - Python SDK documentation
  - Data types and schemas
  - Error handling

- **[Technical Architecture](technical-architecture.md)**
  - System overview
  - Module breakdown
  - Data models
  - Dependencies and protocols
  - Performance considerations

## Project Overview

CCTV Search is an AI-powered video analysis system for searching CCTV footage from Network Video Recorders (NVR). Key features:

- **Object Detection**: Real-time detection using RF-DETR transformer models
- **Multi-Object Tracking**: ByteTrack algorithm for consistent object IDs
- **Efficient Search**: Backward coarse-to-fine algorithm (99% reduction in model calls)
- **NVR Integration**: Direct RTSP connection to Dahua/Hikvision NVRs

## Architecture Summary

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   NVR       │────▶│  Frame      │────▶│  AI Model   │
│   (RTSP)    │     │  Extraction │     │  (RF-DETR)  │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                                │
                                                ▼
                                        ┌─────────────┐
                                        │  Tracker    │
                                        │ (ByteTrack) │
                                        └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │  Search     │
                                        │  Algorithm  │
                                        └─────────────┘
```

## Key Technologies

- **FastAPI**: Modern web framework for the REST API
- **RF-DETR**: Transformer-based object detection
- **ByteTrack**: Multi-object tracking algorithm
- **FFmpeg**: Video frame extraction
- **OpenCV**: Image processing
- **Python 3.12+**: Programming language

## Additional Resources

- **AGENTS.md**: AI coding agent guidelines
- **README.md**: Project overview and quick start
- **pyproject.toml**: Dependency configuration
- **tests/**: Test suite and examples

## Contributing

See `AGENTS.md` for:
- Code style guidelines
- Testing procedures
- Git workflow
- Linting and formatting commands

## License

[Add license information here]
