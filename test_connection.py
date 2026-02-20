#!/usr/bin/env python3
"""Test RTSP live stream to verify connection works."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from cctv_search.nvr import DahuaNVRClient


def test_live_stream():
    """Test live stream endpoint."""
    client = DahuaNVRClient()
    
    # Dahua live stream URL format
    live_url = f"rtsp://{client.username}:{client.password}@{client.host}:{client.port}/cam/realmonitor?channel=1&subtype=0"
    
    print(f"Testing LIVE STREAM:")
    print(f"URL: rtsp://{client.username}:****@{client.host}:{client.port}/cam/realmonitor?channel=1&subtype=0")
    
    import subprocess
    
    cmd = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-timeout", "5000000",
        "-i", live_url,
        "-frames:v", "1",
        "-y",
        "test_live_frame.png",
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ LIVE STREAM WORKS!")
            print(f"  Frame saved: test_live_frame.png")
            return True
        else:
            print(f"✗ Live stream failed:")
            if "404" in result.stderr:
                print("  - 404: Wrong endpoint")
            elif "401" in result.stderr:
                print("  - 401: Wrong password")
            else:
                print(f"  - {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_playback_now():
    """Test playback with current time."""
    client = DahuaNVRClient()
    
    # Use current time
    now = datetime.now()
    print(f"\nTesting PLAYBACK with current time: {now}")
    
    try:
        result = client.extract_frame(
            timestamp=now,
            channel=1,
            output_path="test_playback_now.png"
        )
        print(f"✓ PLAYBACK WORKS!")
        print(f"  Frame saved: {result}")
        return True
    except Exception as e:
        print(f"✗ Playback failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("NVR Connection Test")
    print("=" * 60)
    
    live_works = test_live_stream()
    playback_works = test_playback_now()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Live Stream: {'✓ WORKS' if live_works else '✗ FAILED'}")
    print(f"Playback: {'✓ WORKS' if playback_works else '✗ FAILED'}")
    
    if not live_works and not playback_works:
        print("\nBoth failed. Possible causes:")
        print("  - Wrong NVR brand (not Dahua)")
        print("  - Wrong IP/port")
        print("  - RTSP not enabled on NVR")
        print("  - Network/firewall issue")
