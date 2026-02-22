#!/usr/bin/env python3
"""Diagnose NVR RTSP connection and test different endpoints."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta

from cctv_search.nvr import DahuaNVRClient


def test_rtsp_url(url: str, description: str) -> bool:
    """Test if an RTSP URL is accessible."""
    print(f"\nTesting: {description}")
    print(f"URL: {url[:60]}...")
    
    cmd = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-timeout", "5000000",  # 5 second timeout
        "-i", url,
        "-frames:v", "1",
        "-f", "null",  # Don't save output, just test connection
        "-",
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            print(f"  ✓ SUCCESS")
            return True
        else:
            # Check for specific errors
            if "404" in result.stderr:
                print(f"  ✗ 404 Not Found (wrong endpoint)")
            elif "401" in result.stderr or "Unauthorized" in result.stderr:
                print(f"  ✗ 401 Unauthorized (check credentials)")
            elif "Connection refused" in result.stderr:
                print(f"  ✗ Connection refused (wrong port?)")
            else:
                print(f"  ✗ Error: {result.stderr[:100]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout (host unreachable)")
        return False
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        return False


def main():
    """Test various RTSP endpoints."""
    client = DahuaNVRClient()
    
    print("=" * 60)
    print("NVR RTSP Endpoint Diagnostics")
    print("=" * 60)
    print(f"Host: {client.host}")
    print(f"Port: {client.port}")
    print(f"Username: {client.username}")
    print(f"Password: {'*' * len(client.password)}")
    print()
    
    # Test timestamp
    timestamp = datetime(2026, 2, 19, 9, 0, 0)
    start_str = timestamp.strftime("%Y_%m_%d_%H_%M_%S")
    end_str = (timestamp + timedelta(seconds=1)).strftime("%Y_%m_%d_%H_%M_%S")
    
    # Different endpoint formats to try
    endpoints = [
        # Standard Dahua
        (f"rtsp://{client.username}:{client.password}@{client.host}:{client.port}/cam/playback?channel=1&starttime={start_str}&endtime={end_str}", 
         "Dahua Standard (/cam/playback)"),
        
        # Alternative Dahua paths
        (f"rtsp://{client.username}:{client.password}@{client.host}:{client.port}/playback?channel=1&starttime={start_str}&endtime={end_str}", 
         "Alternative (/playback)"),
        
        (f"rtsp://{client.username}:{client.password}@{client.host}:{client.port}/Streaming/Channels/101", 
         "Live Stream (Channel 1)"),
        
        (f"rtsp://{client.username}:{client.password}@{client.host}:{client.port}/user={client.username}&password={client.password}&channel=1&stream=0.sdp", 
         "Alternative format"),
    ]
    
    working_endpoints = []
    
    for url, desc in endpoints:
        if test_rtsp_url(url, desc):
            working_endpoints.append((url, desc))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if working_endpoints:
        print(f"\n✓ Found {len(working_endpoints)} working endpoint(s):")
        for url, desc in working_endpoints:
            print(f"  - {desc}")
    else:
        print("\n✗ No working endpoints found.")
        print("\nTroubleshooting suggestions:")
        print("  1. Check if NVR is Dahua brand (this code is for Dahua)")
        print("  2. Verify the channel number exists")
        print("  3. Check if recordings exist at the specified time")
        print("  4. Try accessing the NVR web interface to confirm IP/port")
        print("  5. Check if RTSP is enabled in NVR settings")
        print("  6. Verify username/password have playback permissions")


if __name__ == "__main__":
    main()
