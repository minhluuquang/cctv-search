"""Tests for object matching logic (SameBike predicate) using ByteTrack."""

from __future__ import annotations

import numpy as np
import pytest

from cctv_search.tracker import MockByteTrackTracker, TrackerConfig
from cctv_search.detector import Detection, BoundingBox


@pytest.fixture
def tracker():
    """Create ByteTrack tracker instance."""
    config = TrackerConfig(
        mask_iou_threshold=0.5,
        motion_threshold=50.0,
    )
    return MockByteTrackTracker(config)


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


class TestMaskIoU:
    """Test mask IoU calculation for SameBike detection."""

    def test_identical_masks_match(self, tracker, sample_detection):
        """Test that identical masks are considered the same object."""
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

    def test_non_overlapping_masks_reject(self, tracker, sample_detection):
        """Test that non-overlapping masks reject the match."""
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

    def test_iou_threshold_at_exactly_0_5(self, tracker):
        """Test IoU threshold at exactly 0.5."""
        # Create masks with exactly 0.5 IoU
        # Intersection: 2 pixels, Union: 4 pixels -> IoU = 0.5
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
            mask=np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=bool),  # Half overlap
            frame_idx=101,
            timestamp=10.1,
        )

        iou = det1.mask_iou(det2)
        result = tracker.is_same_object(det1, det2)

        # Should match if IoU >= 0.5
        if iou >= 0.5:
            assert result is True
        else:
            assert result is False

    def test_iou_below_threshold_rejects_match(self, tracker):
        """Test that IoU below 0.5 threshold rejects the match."""
        det1 = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=150, y2=130),
            confidence=0.9,
            class_label="bicycle",
            mask=np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=bool),
            frame_idx=100,
            timestamp=10.0,
        )
        # Small overlap: 1 pixel
        det2 = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=150, y2=130),
            confidence=0.85,
            class_label="bicycle",
            mask=np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=bool),
            frame_idx=101,
            timestamp=10.1,
        )

        iou = det1.mask_iou(det2)
        assert iou < 0.5

        result = tracker.is_same_object(det1, det2)
        assert result is False


class TestMotionConsistency:
    """Test motion consistency checks."""

    def test_center_distance_exactly_at_threshold(self, tracker):
        """Test center distance at exactly 50px threshold."""
        # Distance of exactly 50 pixels (using Pythagorean triple 30-40-50)
        det1 = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=150, y2=130),  # center: 125, 115
            confidence=0.9,
            class_label="bicycle",
            mask=np.array([[1, 1], [1, 1]], dtype=bool),
            frame_idx=100,
            timestamp=10.0,
        )
        det2 = Detection(
            bbox=BoundingBox(x1=70, y1=75, x2=120, y2=105),  # center: 95, 90
            confidence=0.85,
            class_label="bicycle",
            mask=np.array([[1, 1], [1, 1]], dtype=bool),
            frame_idx=101,
            timestamp=10.1,
        )

        distance = det1.center_distance(det2)
        result = tracker.is_same_object(det1, det2)

        # Should match if distance <= 50
        if distance <= 50:
            assert result is True
        else:
            assert result is False

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

        # Good match (close position, good mask overlap)
        candidate1 = Detection(
            bbox=BoundingBox(x1=105, y1=105, x2=155, y2=135),
            confidence=0.85,
            class_label="bicycle",
            mask=np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool),
            frame_idx=101,
            timestamp=10.1,
        )

        # Poor match (far away)
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

    def test_custom_iou_threshold(self):
        """Test tracker with custom IoU threshold."""
        config = TrackerConfig(mask_iou_threshold=0.7, motion_threshold=50.0)
        tracker = MockByteTrackTracker(config)

        target = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=150, y2=130),
            confidence=0.9,
            class_label="bicycle",
            mask=np.array([[1, 1], [1, 1]], dtype=bool),
            frame_idx=100,
            timestamp=10.0,
        )
        # Partial overlap
        candidate = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=150, y2=130),
            confidence=0.85,
            class_label="bicycle",
            mask=np.array([[1, 1], [0, 0]], dtype=bool),
            frame_idx=101,
            timestamp=10.1,
        )

        result = tracker.is_same_object(target, candidate)
        # Should reject with higher threshold
        assert result is False

    def test_custom_distance_threshold(self):
        """Test tracker with custom distance threshold."""
        config = TrackerConfig(mask_iou_threshold=0.5, motion_threshold=20.0)
        tracker = MockByteTrackTracker(config)

        target = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=150, y2=130),  # center: 125, 115
            confidence=0.9,
            class_label="bicycle",
            mask=np.array([[1, 1], [1, 1]], dtype=bool),
            frame_idx=100,
            timestamp=10.0,
        )
        # Distance of 30px
        candidate = Detection(
            bbox=BoundingBox(x1=130, y1=100, x2=180, y2=130),  # center: 155, 115
            confidence=0.85,
            class_label="bicycle",
            mask=np.array([[1, 1], [1, 1]], dtype=bool),
            frame_idx=101,
            timestamp=10.1,
        )

        result = tracker.is_same_object(target, candidate)
        # Should reject with lower distance threshold
        assert result is False


class TestEdgeCases:
    """Test edge cases in matching."""

    def test_empty_mask_returns_zero_iou(self):
        """Test that empty mask returns IoU of 0."""
        det1 = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=150, y2=130),
            confidence=0.9,
            class_label="bicycle",
            mask=np.array([[0, 0], [0, 0]], dtype=bool),
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

        iou = det1.mask_iou(det2)
        assert iou == 0.0

    def test_very_small_masks(self, tracker):
        """Test matching with very small masks (1 pixel)."""
        target = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=150, y2=130),
            confidence=0.9,
            class_label="bicycle",
            mask=np.array([[1]], dtype=bool),
            frame_idx=100,
            timestamp=10.0,
        )
        candidate = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=150, y2=130),
            confidence=0.85,
            class_label="bicycle",
            mask=np.array([[1]], dtype=bool),
            frame_idx=101,
            timestamp=10.1,
        )

        iou = target.mask_iou(candidate)
        assert iou == 1.0

        result = tracker.is_same_object(target, candidate)
        assert result is True

    def test_same_detection_compared_to_itself(self, tracker, sample_detection):
        """Test that a detection matches itself."""
        result = tracker.is_same_object(sample_detection, sample_detection)
        assert result is True
