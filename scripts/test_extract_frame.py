#!/usr/bin/env python3
"""Test script to extract a frame from NVR at a specific timestamp."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from cctv_search.nvr import DahuaNVRClient


def main():
    """Extract frame at 2026-02-19 9:00 AM."""
    # Initialize client (loads from .env automatically)
    client = DahuaNVRClient()

    # Target timestamp: 2026-02-19 9:00 AM
    timestamp = datetime(2026, 2, 19, 10, 0, 0)

    output_path = Path("frame_2026_02_19_090000.png")

    print(f"Extracting frame from NVR at {timestamp}")
    print(f"NVR Host: {client.host}")
    print(f"Channel: 1")
    print(f"Output: {output_path}")
    print()

    try:
        result = client.extract_frame(
            timestamp=timestamp,
            channel=1,
            output_path=output_path
        )
        print(f"✓ Frame extracted successfully: {result}")
        print(f"  File size: {result.stat().st_size} bytes")
    except Exception as e:
        print(f"✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()
