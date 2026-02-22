"""Backward coarse-to-fine temporal search for fixed CCTV cameras.

This module implements the backward coarse-to-fine temporal search algorithm
for finding object instances in fixed CCTV camera footage.
"""

from __future__ import annotations

from cctv_search.search.algorithm import (
    BackwardTemporalSearch,
    BoundingBox,
    MockObjectDetector,
    MockVideoDecoder,
    ObjectDetection,
    ObjectTrack,
    Point,
    SearchResult,
    SearchStatus,
    SegmentationMask,
    backward_search,
)

__all__ = [
    "BackwardTemporalSearch",
    "BoundingBox",
    "MockObjectDetector",
    "MockVideoDecoder",
    "ObjectDetection",
    "ObjectTrack",
    "Point",
    "SearchResult",
    "SearchStatus",
    "SegmentationMask",
    "backward_search",
]
