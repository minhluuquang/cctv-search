"""NVR module for extracting frames from Network Video Recorders."""

from __future__ import annotations

from cctv_search.nvr.dahua import DahuaNVRClient, FrameRequest

__all__ = ["DahuaNVRClient", "FrameRequest"]
