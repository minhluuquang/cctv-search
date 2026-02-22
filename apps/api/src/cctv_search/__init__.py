"""CCTV Search - Main module."""

from __future__ import annotations

try:
    import uvicorn
    from cctv_search.api import app

    def main():
        """Run the FastAPI server."""
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
        )

    if __name__ == "__main__":
        main()
except ImportError:
    # uvicorn not installed, skip main entry point
    pass
