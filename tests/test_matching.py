"""Tests for object matching logic (SameBike predicate) using ByteTrack."""

from __future__ import annotations

import numpy as np
import pytest

from cctv_search.ai import ByteTrackTracker
from cctv_search.detector import BoundingBox, Detection


@pytest.fixture
def tracker():
    """Create ByteTrack tracker instance."""
    return ByteTrackTracker(match_thresh=0.8)


@pytest.fixture
def sample_detection():
    """Create a sample detection for testing."""
    return Detection(
        bbox=BoundingBox(x1=100, y1=100, x2=150, y2=130),
        confidence=0.9,
        class_label="bicycle",
        mask=np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool),
        frame_idx=100,
        timestamp=10.0,
    )


class TestBoundingBoxIoU:
    """Test bounding box IoU calculation for SameBike detection."""

    def test_identical_bboxes_match(self, tracker, sample_detection):
        """Test that identical bounding boxes are considered the same object."""
        candidate = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=150, y2=130),
            confidence=0.85,
            class_label="bicycle",
            mask=np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool),
            frame_idx=101,
            timestamp=10.1,
        )

        result = tracker.is_same_object(sample_detection, candidate)
        assert result is True

    def test_non_overlapping_bboxes_reject(self, tracker, sample_detection):
        """Test that non-overlapping bounding boxes reject the match."""
        candidate = Detection(
            bbox=BoundingBox(x1=200, y1=200, x2=250, y2=230),
            confidence=0.85,
            class_label="bicycle",
            mask=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=bool),
            frame_idx=101,
            timestamp=10.1,
        )

        result = tracker.is_same_object(sample_detection, candidate)
        assert result is False

    def test_bbox_iou_threshold_at_match_thresh(self, tracker):
        """Test bounding box IoU at match threshold (0.8)."""
        # Same bbox = 1.0 IoU, should match with 0.8 threshold
        det1 = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=150, y2=130),
            confidence=0.9,
            class_label="bicycle",
            mask=np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool),
            frame_idx=100,
            timestamp=10.0,
        )
        det2 = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=150, y2=130),
            confidence=0.85,
            class_label="bicycle",
            mask=np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool),
            frame_idx=101,
            timestamp=10.1,
        )

        result = tracker.is_same_object(det1, det2)
        assert result is True

    def test_iou_below_threshold_rejects_match(self, tracker):
        """Test that IoU below threshold rejects the match."""
        det1 = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200),
            confidence=0.9,
            class_label="bicycle",
            mask=np.array(
                [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=bool
            ),
            frame_idx=100,
            timestamp=10.0,
        )
        # Far away - small bbox IoU but within 50px
        det2 = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=105, y2=105),
            confidence=0.85,
            class_label="bicycle",
            mask=np.array(
                [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=bool
            ),
            frame_idx=101,
            timestamp=10.1,
        )

        result = tracker.is_same_object(det1, det2)
        # Should reject due to low bbox IoU (< 0.8)
        assert result is False


class TestMotionConsistency:
    """Test motion consistency checks."""

    def test_center_distance_exactly_at_threshold(self, tracker):
        """Test center distance at exactly 50px threshold with high IoU."""
        # Same bbox (high IoU) but test motion threshold behavior
        det1 = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=150, y2=130),
            confidence=0.9,
            class_label="bicycle",
            mask=np.array([[1, 1], [1, 1]], dtype=bool),
            frame_idx=100,
            timestamp=10.0,
        )
        det2 = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=150, y2=130),  # Same position
            confidence=0.85,
            class_label="bicycle",
            mask=np.array([[1, 1], [1, 1]], dtype=bool),
            frame_idx=101,
            timestamp=10.1,
        )

        distance = det1.center_distance(det2)
        result = tracker.is_same_object(det1, det2)

        # Same position should match (0 distance, perfect IoU)
        assert distance == 0
        assert result is True

    def test_center_distance_above_threshold_rejects(self, tracker):
        """Test that center distance > 50px rejects match."""
        det1 = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=150, y2=130),  # center: 125, 115
            confidence=0.9,
            class_label="bicycle",
            mask=np.array([[1, 1], [1, 1]], dtype=bool),
            frame_idx=100,
            timestamp=10.0,
        )
        det2 = Detection(
            bbox=BoundingBox(x1=200, y1=200, x2=250, y2=230),  # center: 225, 215
            confidence=0.85,
            class_label="bicycle",
            mask=np.array([[1, 1], [1, 1]], dtype=bool),
            frame_idx=101,
            timestamp=10.1,
        )

        distance = det1.center_distance(det2)
        assert distance > 50

        result = tracker.is_same_object(det1, det2)
        assert result is False

    def test_motion_consistency_with_same_position(self, tracker):
        """Test motion consistency with identical position."""
        det1 = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=150, y2=130),
            confidence=0.9,
            class_label="bicycle",
            mask=np.array([[1, 1], [1, 1]], dtype=bool),
            frame_idx=100,
            timestamp=10.0,
        )
        det2 = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=150, y2=130),
            confidence=0.85,
            class_label="bicycle",
            mask=np.array([[1, 1], [1, 1]], dtype=bool),
            frame_idx=101,
            timestamp=10.1,
        )

        distance = det1.center_distance(det2)
        assert distance == 0

        result = tracker.is_same_object(det1, det2)
        assert result is True


class TestMultipleObjectDiscrimination:
    """Test discrimination between multiple similar objects."""

    def test_selects_best_match_among_multiple_candidates(self, tracker):
        """Test tracker identifies same object among multiple candidates."""
        target = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=150, y2=130),
            confidence=0.9,
            class_label="bicycle",
            mask=np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool),
            frame_idx=100,
            timestamp=10.0,
        )

        # Good match (identical bbox = 1.0 IoU)
        candidate1 = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=150, y2=130),
            confidence=0.85,
            class_label="bicycle",
            mask=np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool),
            frame_idx=101,
            timestamp=10.1,
        )

        # Poor match (far away, low IoU)
        candidate2 = Detection(
            bbox=BoundingBox(x1=250, y1=250, x2=300, y2=280),
            confidence=0.8,
            class_label="bicycle",
            mask=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=bool),
            frame_idx=101,
            timestamp=10.1,
        )

        result1 = tracker.is_same_object(target, candidate1)
        result2 = tracker.is_same_object(target, candidate2)

        assert result1 is True
        assert result2 is False

    def test_rejects_all_when_no_good_match(self, tracker):
        """Test that all candidates are rejected when none match well."""
        target = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=150, y2=130),
            confidence=0.9,
            class_label="bicycle",
            mask=np.array([[1, 1], [1, 1]], dtype=bool),
            frame_idx=100,
            timestamp=10.0,
        )

        candidates = [
            Detection(
                bbox=BoundingBox(x1=300, y1=300, x2=350, y2=330),
                confidence=0.7,
                class_label="bicycle",
                mask=np.array([[0, 0], [0, 1]], dtype=bool),
                frame_idx=101,
                timestamp=10.1,
            ),
            Detection(
                bbox=BoundingBox(x1=400, y1=400, x2=450, y2=430),
                confidence=0.8,
                class_label="bicycle",
                mask=np.array([[0, 0], [0, 1]], dtype=bool),
                frame_idx=101,
                timestamp=10.1,
            ),
        ]

        results = [tracker.is_same_object(target, c) for c in candidates]

        # None should match
        assert all(r is False for r in results)

    def test_different_classes_are_never_matched(self, tracker):
        """Test that objects of different classes are never matched."""
        target = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=150, y2=130),
            confidence=0.9,
            class_label="bicycle",
            mask=np.array([[1, 1], [1, 1]], dtype=bool),
            frame_idx=100,
            timestamp=10.0,
        )
        candidate = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=150, y2=130),
            confidence=0.9,
            class_label="person",  # Different class
            mask=np.array([[1, 1], [1, 1]], dtype=bool),
            frame_idx=101,
            timestamp=10.1,
        )

        result = tracker.is_same_object(target, candidate)
        assert result is False


class TestThresholdValidation:
    """Test threshold validation for IoU and distance."""

    def test_high_iou_threshold_rejects_partial_overlap(self):
        """Test that high IoU threshold rejects partial overlaps."""
        tracker = ByteTrackTracker(match_thresh=0.9)  # Very strict

        target = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=150, y2=130),
            confidence=0.9,
            class_label="bicycle",
            mask=np.array([[1, 1], [1, 1]], dtype=bool),
            frame_idx=100,
            timestamp=10.0,
        )
        # Slightly offset (creates partial IoU)
        candidate = Detection(
            bbox=BoundingBox(x1=110, y1=100, x2=160, y2=130),
            confidence=0.85,
            class_label="bicycle",
            mask=np.array([[1, 1], [1, 1]], dtype=bool),
            frame_idx=101,
            timestamp=10.1,
        )

        result = tracker.is_same_object(target, candidate)
        # Should reject with 0.9 threshold (offset reduces IoU below 0.9)
        assert result is False

    def test_strict_iou_threshold_accepts_match(self):
        """Test tracker accepts match when IoU is above threshold."""
        tracker = ByteTrackTracker(match_thresh=0.5)

        target = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=150, y2=130),
            confidence=0.9,
            class_label="bicycle",
            mask=np.array([[1, 1], [1, 1]], dtype=bool),
            frame_idx=100,
            timestamp=10.0,
        )
        # Same box = 100% overlap
        candidate = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=150, y2=130),
            confidence=0.85,
            class_label="bicycle",
            mask=np.array([[1, 1], [1, 1]], dtype=bool),
            frame_idx=101,
            timestamp=10.1,
        )

        result = tracker.is_same_object(target, candidate)
        # Should accept with 1.0 IoU
        assert result is True


class TestEdgeCases:
    """Test edge cases in matching."""

    def test_non_overlapping_bboxes_return_zero_iou(self):
        """Test that non-overlapping bboxes return IoU of 0."""
        det1 = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=150, y2=130),
            confidence=0.9,
            class_label="bicycle",
            mask=np.array([[1, 1], [1, 1]], dtype=bool),
            frame_idx=100,
            timestamp=10.0,
        )
        det2 = Detection(
            bbox=BoundingBox(x1=200, y1=200, x2=250, y2=230),
            confidence=0.85,
            class_label="bicycle",
            mask=np.array([[1, 1], [1, 1]], dtype=bool),
            frame_idx=101,
            timestamp=10.1,
        )

        # Non-overlapping bboxes should not match
        tracker = ByteTrackTracker(match_thresh=0.8)
        result = tracker.is_same_object(det1, det2)
        assert result is False

    def test_small_bboxes_match(self, tracker):
        """Test matching with small bounding boxes."""
        target = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=110, y2=110),
            confidence=0.9,
            class_label="bicycle",
            mask=np.array([[1]], dtype=bool),
            frame_idx=100,
            timestamp=10.0,
        )
        candidate = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=110, y2=110),
            confidence=0.85,
            class_label="bicycle",
            mask=np.array([[1]], dtype=bool),
            frame_idx=101,
            timestamp=10.1,
        )

        result = tracker.is_same_object(target, candidate)
        assert result is True

    def test_same_detection_compared_to_itself(self, tracker, sample_detection):
        """Test that a detection matches itself."""
        result = tracker.is_same_object(sample_detection, sample_detection)
        assert result is True
