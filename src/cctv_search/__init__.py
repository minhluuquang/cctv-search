"""CCTV Search - Main module."""

import uvicorn

from cctv_search.api import app


def main():
    """Run the FastAPI server."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
