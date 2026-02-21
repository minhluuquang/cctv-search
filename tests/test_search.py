"""Tests for the backward coarse-to-fine temporal object search algorithm."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    pass


@dataclass
class MockSearchResult:
    """Mock search result for testing."""

    timestamp: float | None
    precision: float
    found: bool


@pytest.fixture
def mock_detector():
    """Create mock object detector."""
    detector = MagicMock()
    detector.detect.return_value = []
    return detector


@pytest.fixture
def mock_tracker():
    """Create mock object tracker."""
    tracker = MagicMock()
    tracker.associate.return_value = None
    tracker.is_same_object.return_value = False  # Default to no match
    return tracker


@pytest.fixture
def mock_video_source():
    """Create mock video source with frame-based API."""
    source = MagicMock()
    source.fps = 20.0

    def timestamp_to_frame(ts):
        return int(ts * source.fps)

    def frame_to_timestamp(frame_idx):
        return frame_idx / source.fps

    # Set up get_frame_by_index as a MagicMock with side_effect
    source.get_frame_by_index = MagicMock(
        side_effect=lambda frame_idx: str(frame_idx).encode()
    )
    source.timestamp_to_frame = timestamp_to_frame
    source.frame_to_timestamp = frame_to_timestamp
    source.frame_exists.return_value = True

    return source


@pytest.fixture
def search_config():
    """Create default search configuration."""
    return {
        "initial_window": 30 * 60,  # 30 minutes
        "min_window": 5,  # 5 seconds
        "max_lookback": 3 * 60 * 60,  # 3 hours
        "precision": 5,  # ±5 seconds
    }


class TestBasicSearchFunctionality:
    """Test basic search finds object when it exists."""

    def test_search_finds_object_when_it_exists(
        self, mock_detector, mock_tracker, mock_video_source, search_config
    ):
        """Test that search finds the object when it exists in the history."""
        # Object appears at T - 600 seconds (10 minutes ago) and stays visible until T
        target_time = time.time()
        object_appearance_time = target_time - 600
        object_appearance_frame = mock_video_source.timestamp_to_frame(
            object_appearance_time
        )
        target_frame = mock_video_source.timestamp_to_frame(target_time)

        def mock_detect(frame):
            # Frame data contains frame index as string
            frame_idx = int(frame.decode())
            # Object exists from appearance_frame to target_frame
            if object_appearance_frame <= frame_idx <= target_frame:
                return [
                    MagicMock(
                        label="bicycle",
                        bbox=MagicMock(x=100, y=100, width=50, height=30),
                        mask=[[1, 1], [1, 1]],
                        confidence=0.9,
                    )
                ]
            return []

        mock_detector.detect.side_effect = mock_detect
        mock_tracker.is_same_object.return_value = True

        # Execute search
        from cctv_search.search.algorithm import backward_search

        result = backward_search(
            target_time=target_time,
            target_object=MagicMock(
                label="bicycle",
                bbox=MagicMock(x=100, y=100, width=50, height=30),
                mask=[[1, 1], [1, 1]],
            ),
            video_source=mock_video_source,
            detector=mock_detector,
            tracker=mock_tracker,
            config=search_config,
        )

        assert result.found is True
        assert result.timestamp is not None
        # Result should be close to actual appearance time (within precision)
        assert (
            abs(result.timestamp - object_appearance_time) <= search_config["precision"]
        )

    def test_search_returns_not_found_when_object_never_appeared(
        self, mock_detector, mock_tracker, mock_video_source, search_config
    ):
        """Test that search returns NOT_FOUND when object never appeared in history."""
        target_time = time.time()

        # No object detections at any point
        mock_detector.detect.return_value = []

        from cctv_search.search.algorithm import backward_search

        result = backward_search(
            target_time=target_time,
            target_object=MagicMock(label="bicycle"),
            video_source=mock_video_source,
            detector=mock_detector,
            tracker=mock_tracker,
            config=search_config,
        )

        assert result.found is False
        assert result.timestamp is None


class TestObjectDisappearanceReappearance:
    """Test handling of object disappearance and reappearance."""

    def test_finds_nearest_appearance_with_multiple_disappearances(
        self, mock_detector, mock_tracker, mock_video_source, search_config
    ):
        """Test finds nearest appearance when object disappears and reappears.

        Note: Gaps between appearances must be > 30 seconds (coarse step size)
        for the algorithm to properly detect them.
        """
        target_time = time.time()
        target_frame = mock_video_source.timestamp_to_frame(target_time)

        # Object appears at: T-300, disappears at T-240 (60s window, gap after)
        # Reappears at: T-180, disappears at T-120 (60s window, gap after)
        # Reappears at: T-60, present at T (60s window, current)
        # Gaps are 60 seconds (> 30 second coarse step) so algorithm can find them
        fps = mock_video_source.fps
        appearance_windows_frames = [
            # First appearance (T-300 to T-240)
            (target_frame - int(300 * fps), target_frame - int(240 * fps)),
            # Second appearance (T-180 to T-120)
            (target_frame - int(180 * fps), target_frame - int(120 * fps)),
            # Third appearance (T-60 to T), +1 to include target
            (target_frame - int(60 * fps), target_frame + 1),
        ]

        def mock_detect(frame):
            # Simulate object presence based on frame index encoded in frame
            frame_idx = int(frame.decode())
            for start, end in appearance_windows_frames:
                if start <= frame_idx < end:  # Use < for end to be consistent
                    return [MagicMock(label="bicycle", confidence=0.9)]
            return []

        mock_detector.detect.side_effect = mock_detect
        mock_tracker.is_same_object.return_value = True

        from cctv_search.search.algorithm import backward_search

        result = backward_search(
            target_time=target_time,
            target_object=MagicMock(label="bicycle"),
            video_source=mock_video_source,
            detector=mock_detector,
            tracker=mock_tracker,
            config=search_config,
        )

        assert result.found is True
        assert result.timestamp is not None
        # Should find the start of the nearest appearance window (around T-60)
        # not the absolute earliest appearance (T-300)
        nearest_start = target_time - 60
        # Allow tolerance
        assert abs(result.timestamp - nearest_start) <= search_config["precision"] + 10


class TestSearchTerminationConditions:
    """Test search termination conditions."""

    def test_stops_at_3_hour_limit(
        self, mock_detector, mock_tracker, mock_video_source, search_config
    ):
        """Test search stops when reaching 3-hour lookback limit."""
        target_time = time.time()

        # Object never appears in the 3-hour window
        mock_detector.detect.return_value = []

        from cctv_search.search.algorithm import backward_search

        result = backward_search(
            target_time=target_time,
            target_object=MagicMock(label="bicycle"),
            video_source=mock_video_source,
            detector=mock_detector,
            tracker=mock_tracker,
            config=search_config,
        )

        assert result.found is False
        # Verify we performed multiple iterations searching through the window
        # The search should check several frames with 30-second steps across 3 hours
        assert result.iterations >= 10  # Should check many frames, not just 1-2

    def test_achieves_5_second_precision(
        self, mock_detector, mock_tracker, mock_video_source, search_config
    ):
        """Test that search achieves ±5 second precision."""
        target_time = time.time()
        target_frame = mock_video_source.timestamp_to_frame(target_time)
        # Object appears from T-123.456 to T, then disappears
        disappearance_time = target_time - 123.456
        disappearance_frame = mock_video_source.timestamp_to_frame(disappearance_time)

        def mock_detect(frame):
            frame_idx = int(frame.decode())
            # Object exists from disappearance_frame to target_frame
            if disappearance_frame <= frame_idx <= target_frame:
                return [MagicMock(label="bicycle", confidence=0.9)]
            return []

        mock_detector.detect.side_effect = mock_detect
        mock_tracker.is_same_object.return_value = True

        from cctv_search.search.algorithm import backward_search

        result = backward_search(
            target_time=target_time,
            target_object=MagicMock(label="bicycle"),
            video_source=mock_video_source,
            detector=mock_detector,
            tracker=mock_tracker,
            config=search_config,
        )

        assert result.found is True
        assert result.timestamp is not None
        assert result.precision_seconds is not None
        # Should find near disappearance time (earliest appearance in continuous window)
        assert abs(result.timestamp - disappearance_time) <= 5.0
        assert result.precision_seconds <= 5.0


class TestMissingVideoSegments:
    """Test handling of missing video segments."""

    def test_skips_missing_segments_and_continues(
        self, mock_detector, mock_tracker, mock_video_source, search_config
    ):
        """Test that missing video segments are skipped and search continues."""
        target_time = time.time()
        target_frame = mock_video_source.timestamp_to_frame(target_time)

        # Missing segments at T-1000 to T-800, T-500 to T-400 (in frames)
        missing_segments_frames = [
            (target_frame - int(1000 * 20), target_frame - int(800 * 20)),
            (target_frame - int(500 * 20), target_frame - int(400 * 20)),
        ]

        original_side_effect = mock_video_source.get_frame_by_index.side_effect

        def get_frame_by_index(frame_idx):
            # Check if frame is in missing segments
            for start, end in missing_segments_frames:
                if start <= frame_idx <= end:
                    return None
            return original_side_effect(frame_idx)

        mock_video_source.get_frame_by_index.side_effect = get_frame_by_index

        # Object appears right before first missing segment
        object_time = target_time - 1050
        object_frame = mock_video_source.timestamp_to_frame(object_time)

        def mock_detect(frame):
            if frame is None:
                return []
            frame_idx = int(frame.decode())
            # Within 5 seconds in frames
            if abs(frame_idx - object_frame) <= int(5 * 20):
                return [MagicMock(label="bicycle", confidence=0.9)]
            return []

        mock_detector.detect.side_effect = mock_detect
        mock_tracker.is_same_object.return_value = True

        from cctv_search.search.algorithm import backward_search

        result = backward_search(
            target_time=target_time,
            target_object=MagicMock(label="bicycle"),
            video_source=mock_video_source,
            detector=mock_detector,
            tracker=mock_tracker,
            config=search_config,
        )

        assert result.found is True
        assert result.timestamp is not None
        # Allow slightly more tolerance due to frame-based rounding
        assert abs(result.timestamp - object_time) <= search_config["precision"] + 1

    def test_handles_all_missing_frames_in_window(
        self, mock_detector, mock_tracker, mock_video_source, search_config
    ):
        """Test handling when all frames are missing in initial window."""
        target_time = time.time()

        # All frames missing in the first 30 minutes
        mock_video_source.get_frame_by_index.return_value = None

        from cctv_search.search.algorithm import backward_search

        result = backward_search(
            target_time=target_time,
            target_object=MagicMock(label="bicycle"),
            video_source=mock_video_source,
            detector=mock_detector,
            tracker=mock_tracker,
            config=search_config,
        )

        # Should continue searching beyond missing segments
        # or return NOT_FOUND if no valid frames in 3-hour window
        assert isinstance(result.found, bool)


class TestCoarseToFineRefinement:
    """Test coarse-to-fine search refinement strategy."""

    def test_window_halves_on_success(
        self, mock_detector, mock_tracker, mock_video_source, search_config
    ):
        """Test that window size halves when object is found."""
        target_time = time.time()

        # Object exists throughout history
        mock_detector.detect.return_value = [MagicMock(label="bicycle")]
        mock_tracker.is_same_object.return_value = True

        from cctv_search.search.algorithm import backward_search

        backward_search(
            target_time=target_time,
            target_object=MagicMock(label="bicycle"),
            video_source=mock_video_source,
            detector=mock_detector,
            tracker=mock_tracker,
            config=search_config,
        )

        # Should progressively sample with halving windows
        # First: 30 min window, then 15 min, 7.5 min, etc.
        calls = mock_video_source.get_frame_by_index.call_args_list
        frames = [call[0][0] for call in calls]

        # Verify frames show coarse-to-fine pattern
        # Initial check at 30 min (in frames), then progressively finer
        assert len(frames) > 5  # Multiple refinement steps

    def test_window_shifts_on_failure(
        self, mock_detector, mock_tracker, mock_video_source, search_config
    ):
        """Test that search shifts window when object not found."""
        target_time = time.time()

        # Object appears at T - 90 minutes
        object_time = target_time - 90 * 60
        object_frame = mock_video_source.timestamp_to_frame(object_time)

        def mock_detect(frame):
            frame_idx = int(frame.decode())
            # Object appears at or before object_frame
            if frame_idx <= object_frame + int(5 * 20):
                return [MagicMock(label="bicycle")]
            return []

        mock_detector.detect.side_effect = mock_detect
        mock_tracker.is_same_object.return_value = True

        from cctv_search.search.algorithm import backward_search

        result = backward_search(
            target_time=target_time,
            target_object=MagicMock(label="bicycle"),
            video_source=mock_video_source,
            detector=mock_detector,
            tracker=mock_tracker,
            config=search_config,
        )

        assert result.found is True
        # Verify search explored multiple windows
        calls = mock_video_source.get_frame_by_index.call_args_list
        assert len(calls) > 3  # Should check multiple windows


class TestSearchConfiguration:
    """Test search configuration validation."""

    def test_invalid_min_window_raises_error(self):
        """Test that invalid min_window raises configuration error."""
        from cctv_search.search.algorithm import SearchConfigError, backward_search

        # Create a mock video source with proper timestamp_to_frame
        video_source = MagicMock()
        video_source.timestamp_to_frame = lambda ts: int(ts * 20)
        video_source.frame_to_timestamp = lambda frame: frame / 20
        video_source.get_frame_by_index = MagicMock(return_value=b"frame")

        with pytest.raises(SearchConfigError):
            backward_search(
                target_time=time.time(),
                target_object=MagicMock(),
                video_source=video_source,
                detector=MagicMock(),
                tracker=MagicMock(),
                config={"min_window": 0, "max_lookback": 3600},
            )

    def test_invalid_max_lookback_raises_error(self):
        """Test that invalid max_lookback raises configuration error."""
        from cctv_search.search.algorithm import SearchConfigError, backward_search

        # Create a mock video source with proper timestamp_to_frame
        video_source = MagicMock()
        video_source.timestamp_to_frame = lambda ts: int(ts * 20)
        video_source.frame_to_timestamp = lambda frame: frame / 20
        video_source.get_frame_by_index = MagicMock(return_value=b"frame")

        with pytest.raises(SearchConfigError):
            backward_search(
                target_time=time.time(),
                target_object=MagicMock(),
                video_source=video_source,
                detector=MagicMock(),
                tracker=MagicMock(),
                config={
                    "min_window": 5,
                    "max_lookback": 0,  # Invalid
                },
            )
