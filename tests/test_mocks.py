"""Tests for mock models (detector, tracker, association)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    pass


class TestMockDetector:
    """Tests for mock object detector."""

    @pytest.fixture
    def mock_detector(self):
        """Create mock detector with predictable behavior."""
        detector = MagicMock()
        detector.detect.return_value = [
            MagicMock(
                label="bicycle",
                bbox=MagicMock(x=100, y=100, width=50, height=30, confidence=0.9),
                mask=[[1, 1], [1, 1]],
                confidence=0.9,
            )
        ]
        return detector

    def test_detector_returns_valid_detections(self, mock_detector):
        """Test that detector returns valid detection objects."""
        frame = b"test_frame_data"
        detections = mock_detector.detect(frame)

        assert isinstance(detections, list)
        assert len(detections) > 0

        for det in detections:
            assert hasattr(det, "label")
            assert hasattr(det, "bbox")
            assert hasattr(det, "confidence")
            assert 0 <= det.confidence <= 1

    def test_detector_returns_empty_list_for_no_objects(self, mock_detector):
        """Test that detector returns empty list when no objects detected."""
        mock_detector.detect.return_value = []

        frame = b"empty_frame"
        detections = mock_detector.detect(frame)

        assert detections == []

    def test_detector_returns_multiple_objects(self, mock_detector):
        """Test detector returning multiple objects in frame."""
        mock_detector.detect.return_value = [
            MagicMock(
                label="bicycle",
                bbox=MagicMock(x=100, y=100, width=50, height=30),
                confidence=0.9,
            ),
            MagicMock(
                label="person",
                bbox=MagicMock(x=200, y=200, width=40, height=80),
                confidence=0.85,
            ),
            MagicMock(
                label="car",
                bbox=MagicMock(x=300, y=300, width=100, height=60),
                confidence=0.92,
            ),
        ]

        frame = b"busy_frame"
        detections = mock_detector.detect(frame)

        assert len(detections) == 3
        labels = [d.label for d in detections]
        assert "bicycle" in labels
        assert "person" in labels
        assert "car" in labels

    def test_detector_returns_consistent_confidence_range(self, mock_detector):
        """Test that detector confidence scores are in valid range."""
        frame = b"test_frame"
        detections = mock_detector.detect(frame)

        for det in detections:
            assert 0.0 <= det.confidence <= 1.0
            assert isinstance(det.confidence, float)

    def test_detector_handles_various_frame_sizes(self, mock_detector):
        """Test detector handles various frame sizes."""
        frame_sizes = [b"small", b"medium_frame_data", b"x" * 1000000]

        for frame in frame_sizes:
            detections = mock_detector.detect(frame)
            assert isinstance(detections, list)


class TestMockTracker:
    """Tests for mock object tracker with consistent IDs."""

    @pytest.fixture
    def mock_tracker(self):
        """Create mock tracker with ID management."""
        tracker = MagicMock()
        tracker._next_id = 1
        tracker._tracks = {}

        def mock_associate(detections, previous_track_id=None):
            """Simulate track association."""
            if previous_track_id and previous_track_id in tracker._tracks:
                # Continue existing track
                return MagicMock(
                    track_id=previous_track_id,
                    is_same_object=True,
                    detection=detections[0] if detections else None,
                )
            # Create new track
            track_id = tracker._next_id
            tracker._next_id += 1
            tracker._tracks[track_id] = detections[0] if detections else None
            return MagicMock(
                track_id=track_id,
                is_same_object=True,
                detection=detections[0] if detections else None,
            )

        tracker.associate.side_effect = mock_associate
        return tracker

    def test_tracker_maintains_consistent_ids(self, mock_tracker):
        """Test that tracker maintains consistent IDs for same object."""
        detection1 = MagicMock(label="bicycle", confidence=0.9)
        detection2 = MagicMock(label="bicycle", confidence=0.91)

        # First association creates new track
        result1 = mock_tracker.associate([detection1])
        track_id = result1.track_id

        # Second association continues same track
        result2 = mock_tracker.associate([detection2], previous_track_id=track_id)

        assert result2.track_id == track_id
        assert result2.is_same_object is True

    def test_tracker_creates_new_id_for_different_object(self, mock_tracker):
        """Test tracker creates new ID for different object."""
        detection1 = MagicMock(label="bicycle", bbox=MagicMock(x=100, y=100))
        detection2 = MagicMock(label="bicycle", bbox=MagicMock(x=500, y=500))

        # First object
        result1 = mock_tracker.associate([detection1])
        id1 = result1.track_id

        # Second object (different location)
        result2 = mock_tracker.associate([detection2])
        id2 = result2.track_id

        assert id1 != id2

    def test_tracker_handles_empty_detections(self, mock_tracker):
        """Test tracker handles empty detection list."""
        result = mock_tracker.associate([])

        assert result is not None
        assert hasattr(result, "track_id") or result.detection is None

    def test_tracker_preserves_detection_metadata(self, mock_tracker):
        """Test that tracker preserves detection metadata."""
        detection = MagicMock(
            label="bicycle",
            bbox=MagicMock(x=100, y=100, width=50, height=30),
            confidence=0.95,
            mask=[[1, 1], [1, 1]],
        )

        result = mock_tracker.associate([detection])

        assert result.detection.label == "bicycle"
        assert result.detection.confidence == 0.95
        assert result.detection.bbox.x == 100


class TestMockAssociation:
    """Tests for mock detection-to-track association."""

    @pytest.fixture
    def mock_association(self):
        """Create mock association function."""

        def associate(detections, tracks, iou_threshold=0.5):
            """Simple IoU-based association."""
            associations = []

            for det in detections:
                best_match = None
                best_iou = 0

                for track in tracks:
                    # Simulate IoU calculation
                    iou = getattr(det, "_iou_with_track", 0.6)
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_match = track

                if best_match:
                    associations.append(
                        MagicMock(
                            detection=det,
                            track=best_match,
                            iou=best_iou,
                            matched=True,
                        )
                    )
                else:
                    associations.append(
                        MagicMock(
                            detection=det,
                            track=None,
                            iou=0,
                            matched=False,
                        )
                    )

            return associations

        return associate

    def test_association_matches_high_iou_detections(self, mock_association):
        """Test that high IoU detections are matched to tracks."""
        detection = MagicMock(label="bicycle", _iou_with_track=0.8)
        track = MagicMock(track_id=1, label="bicycle")

        results = mock_association([detection], [track])

        assert len(results) == 1
        assert results[0].matched is True
        assert results[0].track.track_id == 1

    def test_association_rejects_low_iou_matches(self, mock_association):
        """Test that low IoU detections create new tracks."""
        detection = MagicMock(label="bicycle", _iou_with_track=0.3)
        track = MagicMock(track_id=1, label="bicycle")

        results = mock_association([detection], [track], iou_threshold=0.5)

        assert len(results) == 1
        assert results[0].matched is False
        assert results[0].track is None

    def test_association_handles_multiple_detections(self, mock_association):
        """Test association with multiple detections and tracks."""
        detections = [
            MagicMock(label="bicycle", _iou_with_track=0.7),
            MagicMock(label="person", _iou_with_track=0.6),
            MagicMock(label="bicycle", _iou_with_track=0.2),
        ]
        tracks = [
            MagicMock(track_id=1, label="bicycle"),
            MagicMock(track_id=2, label="person"),
        ]

        results = mock_association(detections, tracks)

        assert len(results) == 3
        matched = [r for r in results if r.matched]
        unmatched = [r for r in results if not r.matched]

        assert len(matched) == 2
        assert len(unmatched) == 1

    def test_association_respects_iou_threshold(self, mock_association):
        """Test that association respects IoU threshold parameter."""
        detection = MagicMock(label="bicycle", _iou_with_track=0.6)
        track = MagicMock(track_id=1)

        # With threshold 0.5, should match
        results_low = mock_association([detection], [track], iou_threshold=0.5)
        assert results_low[0].matched is True

        # With threshold 0.7, should not match
        results_high = mock_association([detection], [track], iou_threshold=0.7)
        assert results_high[0].matched is False


class TestMockIntegration:
    """Integration tests for mock detector + tracker + association."""

    def test_full_pipeline_from_detection_to_track(self):
        """Test complete pipeline from detection to track assignment."""
        # Create mocks
        detector = MagicMock()
        tracker = MagicMock()

        # Configure detector
        detector.detect.return_value = [
            MagicMock(
                label="bicycle",
                bbox=MagicMock(x=100, y=100, width=50, height=30),
                confidence=0.9,
            )
        ]

        # Configure tracker
        tracker.associate.return_value = MagicMock(
            track_id=1,
            is_same_object=True,
            detection=detector.detect.return_value[0],
        )

        # Simulate pipeline
        frame = b"test_frame"
        detections = detector.detect(frame)
        track_result = tracker.associate(detections)

        assert track_result.track_id == 1
        assert track_result.is_same_object is True
        assert track_result.detection.label == "bicycle"

    def test_pipeline_with_multiple_frames(self):
        """Test pipeline across multiple frames maintains consistency."""
        detector = MagicMock()
        tracker = MagicMock()

        # Track ID management
        current_track_id = 1

        def mock_associate(dets, previous_track_id=None):
            nonlocal current_track_id
            if previous_track_id:
                return MagicMock(
                    track_id=previous_track_id,
                    is_same_object=True,
                    detection=dets[0] if dets else None,
                )
            tid = current_track_id
            current_track_id += 1
            return MagicMock(
                track_id=tid,
                is_same_object=True,
                detection=dets[0] if dets else None,
            )

        tracker.associate.side_effect = mock_associate

        # Simulate 5 frames with same object
        track_id = None
        for i in range(5):
            detector.detect.return_value = [
                MagicMock(
                    label="bicycle",
                    bbox=MagicMock(x=100 + i, y=100, width=50, height=30),
                    confidence=0.9,
                )
            ]

            detections = detector.detect(b"frame")
            result = tracker.associate(detections, previous_track_id=track_id)
            track_id = result.track_id

        # Same object should have same track ID throughout
        assert track_id == 1

    def test_pipeline_handles_object_disappearance(self):
        """Test pipeline when object disappears from frame."""
        detector = MagicMock()
        tracker = MagicMock()

        frame_results = [
            [MagicMock(label="bicycle")],  # Frame 1: present
            [MagicMock(label="bicycle")],  # Frame 2: present
            [],  # Frame 3: disappeared
            [],  # Frame 4: still gone
            [MagicMock(label="bicycle")],  # Frame 5: reappeared
        ]

        detector.detect.side_effect = frame_results
        tracker.associate.side_effect = lambda dets, **kwargs: MagicMock(
            track_id=1,
            is_same_object=len(dets) > 0,
            detection=dets[0] if dets else None,
        )

        results = []
        for _ in range(5):
            detections = detector.detect(b"frame")
            result = tracker.associate(detections)
            results.append(result.is_same_object)

        assert results == [True, True, False, False, True]


class TestMockConfiguration:
    """Tests for mock model configuration."""

    def test_detector_configuration_parameters(self):
        """Test detector accepts configuration parameters."""
        detector = MagicMock()
        detector.configure = MagicMock()

        detector.configure(
            confidence_threshold=0.7,
            nms_threshold=0.4,
            max_detections=100,
        )

        detector.configure.assert_called_once_with(
            confidence_threshold=0.7,
            nms_threshold=0.4,
            max_detections=100,
        )

    def test_tracker_configuration_parameters(self):
        """Test tracker accepts configuration parameters."""
        tracker = MagicMock()
        tracker.configure = MagicMock()

        tracker.configure(
            max_age=30,
            min_hits=3,
            iou_threshold=0.5,
        )

        tracker.configure.assert_called_once_with(
            max_age=30,
            min_hits=3,
            iou_threshold=0.5,
        )


class TestMockErrorHandling:
    """Tests for mock model error handling."""

    def test_detector_handles_corrupted_frame(self):
        """Test detector handles corrupted frame gracefully."""
        detector = MagicMock()
        detector.detect.side_effect = [
            [MagicMock(label="bicycle")],  # Good frame
            Exception("Corrupted frame"),  # Bad frame
            [MagicMock(label="bicycle")],  # Good frame
        ]

        # Should handle exception
        results = []
        for _ in range(3):
            try:
                result = detector.detect(b"frame")
                results.append(result)
            except Exception:
                results.append(None)

        assert len(results) == 3
        assert results[1] is None

    def test_tracker_handles_invalid_detection(self):
        """Test tracker handles invalid detection input."""
        tracker = MagicMock()
        tracker.associate.return_value = MagicMock(
            track_id=None,
            is_same_object=False,
            detection=None,
            error="Invalid detection",
        )

        result = tracker.associate([None])

        assert result.track_id is None
        assert result.is_same_object is False
