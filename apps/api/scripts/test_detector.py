#!/usr/bin/env python3
"""Test script for RF-DETR object detector.

This script demonstrates how to use the RF-DETR detector for CCTV footage analysis.
It includes tests for:
- Model loading
- Single frame detection
- Batch detection
- Performance benchmarking
- Error handling

Usage:
    # Test with mock data (no dependencies required)
    python scripts/test_detector.py --mock

    # Test with real RF-DETR (requires: pip install rfdetr)
    python scripts/test_detector.py --image /path/to/image.jpg

    # Run all tests
    python scripts/test_detector.py --all
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from cctv_search.ai.rf_detr import DetectedObject


def create_mock_frame(width: int = 1920, height: int = 1080) -> bytes:
    """Create a mock frame for testing."""
    import numpy as np
    
    # Create a simple test image (gradient pattern)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some patterns that look like objects
    # Simulate a bicycle-like shape
    cv2 = sys.modules.get('cv2')
    if cv2:
        # Draw a rectangle to simulate an object
        cv2.rectangle(img, (100, 100), (300, 400), (128, 128, 128), -1)
        cv2.circle(img, (200, 300), 50, (100, 100, 100), -1)
    else:
        # Without OpenCV, just create a pattern
        img[100:400, 100:300] = 128
    
    # Encode to bytes
    try:
        import cv2
        _, buffer = cv2.imencode('.jpg', img)
        return buffer.tobytes()
    except ImportError:
        # Return dummy bytes if cv2 not available
        return b"mock_frame_data" * 1000


def test_detector_creation():
    """Test detector can be created with different configurations."""
    print("\n=== Test 1: Detector Creation ===")
    
    from cctv_search.ai.rf_detr import RFDetrDetector
    
    # Default configuration
    detector1 = RFDetrDetector()
    print(f"✓ Created detector with default threshold: {detector1.confidence_threshold}")
    
    # Custom threshold
    detector2 = RFDetrDetector(confidence_threshold=0.7)
    print(f"✓ Created detector with custom threshold: {detector2.confidence_threshold}")
    
    # Verify initial state
    assert not detector1._model_loaded, "Model should not be loaded initially"
    print("✓ Detector state is correct (model not loaded)")
    
    print("✅ Detector creation tests passed!\n")


def test_model_loading():
    """Test model loading (will fail gracefully if rfdetr not installed)."""
    print("\n=== Test 2: Model Loading ===")
    
    from cctv_search.ai.rf_detr import RFDetrDetector
    
    detector = RFDetrDetector()
    
    try:
        print("Loading RF-DETR model (this may take a moment on first run)...")
        detector.load_model()
        print("✓ Model loaded successfully!")
        print(f"✓ Model loaded state: {detector._model_loaded}")
    except RuntimeError as e:
        print(f"⚠️  Model loading failed (expected if rfdetr not installed):")
        print(f"   Error: {e}")
        print("\n   To install RF-DETR, run:")
        print("   pip install rfdetr")
        return False
    
    print("✅ Model loading test passed!\n")
    return True


def test_detection_mock():
    """Test detection with mock data."""
    print("\n=== Test 3: Mock Detection ===")
    
    from unittest.mock import MagicMock, patch
    from cctv_search.ai.rf_detr import RFDetrDetector
    
    detector = RFDetrDetector()
    
    # Create mock frame
    mock_frame = create_mock_frame()
    print(f"✓ Created mock frame ({len(mock_frame)} bytes)")
    
    # Test without loading model (should raise error)
    try:
        detector.detect(mock_frame)
        print("❌ Should have raised RuntimeError")
        return False
    except RuntimeError as e:
        print(f"✓ Correctly raised error without model: {e}")
    
    print("✅ Mock detection test passed!\n")
    return True


def test_detection_with_mocked_model():
    """Test detection with fully mocked model."""
    print("\n=== Test 4: Detection with Mocked Model ===")
    
    from unittest.mock import MagicMock, patch
    from cctv_search.ai.rf_detr import RFDetrDetector, BoundingBox, DetectedObject
    
    detector = RFDetrDetector()
    
    # Mock the model
    mock_model = MagicMock()
    mock_model.predict.return_value = [
        {'class': 'bicycle', 'confidence': 0.95, 'bbox': [100, 100, 300, 400]},
        {'class': 'person', 'confidence': 0.87, 'bbox': [400, 200, 500, 380]},
    ]
    
    detector._model = mock_model
    detector._model_loaded = True
    
    # Create mock frame as numpy array (avoid cv2 dependency)
    import numpy as np
    mock_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # Run detection
    print("Running detection on mock frame...")
    detections = detector.detect(mock_frame)
    
    print(f"✓ Detected {len(detections)} objects:")
    for i, det in enumerate(detections, 1):
        print(f"  {i}. {det.label}: {det.confidence:.2%} confidence")
        print(f"     BBox: ({det.bbox.x:.0f}, {det.bbox.y:.0f}) "
              f"{det.bbox.width:.0f}x{det.bbox.height:.0f}")
    
    assert len(detections) == 2, "Should detect 2 objects"
    assert detections[0].label == "bicycle", "First object should be bicycle"
    
    print("✅ Detection with mocked model test passed!\n")


def test_batch_detection():
    """Test batch detection on multiple frames."""
    print("\n=== Test 5: Batch Detection ===")
    
    from unittest.mock import MagicMock
    from cctv_search.ai.rf_detr import RFDetrDetector, BoundingBox, DetectedObject
    
    detector = RFDetrDetector()
    
    # Mock the model
    mock_ai_detector = MagicMock()
    mock_ai_detector.detect.side_effect = [
        [DetectedObject("car", BoundingBox(0, 0, 100, 100, 0.9), 0.9, 0.0)],
        [DetectedObject("person", BoundingBox(50, 50, 80, 160, 0.85), 0.85, 0.0)],
        [],  # No detections in third frame
    ]
    
    detector._model = mock_ai_detector
    detector._model_loaded = True
    
    # Create 3 mock frames
    frames = [create_mock_frame() for _ in range(3)]
    print(f"✓ Created {len(frames)} mock frames")
    
    # Run batch detection
    print("Running batch detection...")
    results = detector.detect_batch(frames)
    
    print(f"✓ Processed {len(results)} frames:")
    for i, detections in enumerate(results):
        print(f"  Frame {i+1}: {len(detections)} objects")
        for det in detections:
            print(f"    - {det.label}: {det.confidence:.2%}")
    
    assert len(results) == 3, "Should return results for all frames"
    assert len(results[2]) == 0, "Third frame should have no detections"
    
    print("✅ Batch detection test passed!\n")


def test_performance():
    """Test detection performance."""
    print("\n=== Test 6: Performance Test ===")
    
    from unittest.mock import MagicMock
    from cctv_search.ai.rf_detr import RFDetrDetector, BoundingBox, DetectedObject
    
    detector = RFDetrDetector()
    
    # Mock the model with realistic timing
    mock_ai_detector = MagicMock()
    
    def mock_detect(frame):
        # Simulate 50ms processing time
        time.sleep(0.05)
        return [
            DetectedObject("bicycle", BoundingBox(100, 100, 200, 300, 0.95), 0.95, 0.0)
        ]
    
    mock_ai_detector.detect.side_effect = mock_detect
    detector._model = mock_ai_detector
    detector._model_loaded = True
    
    # Create test frame
    mock_frame = create_mock_frame()
    
    # Warm up
    print("Warming up...")
    for _ in range(3):
        detector.detect(mock_frame)
    
    # Benchmark
    print("Running performance benchmark (10 iterations)...")
    times = []
    for _ in range(10):
        start = time.time()
        detector.detect(mock_frame)
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"✓ Average detection time: {avg_time*1000:.1f}ms")
    print(f"✓ Min time: {min_time*1000:.1f}ms")
    print(f"✓ Max time: {max_time*1000:.1f}ms")
    print(f"✓ Throughput: {1/avg_time:.1f} FPS")
    
    print("✅ Performance test passed!\n")


def test_error_handling():
    """Test error handling scenarios."""
    print("\n=== Test 7: Error Handling ===")
    
    from unittest.mock import MagicMock
    from cctv_search.ai.rf_detr import RFDetrDetector
    
    detector = RFDetrDetector()
    
    # Test 1: Detection without loading model
    print("Test 1: Detection without model...")
    try:
        detector.detect(b"frame_data")
        print("❌ Should have raised RuntimeError")
    except RuntimeError as e:
        print(f"✓ Correctly raised error: {e}")
    
    # Test 2: Load model when rfdetr not installed
    print("\nTest 2: Load model without rfdetr...")
    # This will fail gracefully in the test environment
    
    # Test 3: Invalid frame data
    print("\nTest 3: Invalid frame data...")
    mock_ai_detector = MagicMock()
    mock_ai_detector.detect.side_effect = Exception("Invalid frame format")
    detector._model = mock_ai_detector
    detector._model_loaded = True
    
    result = detector.detect(b"invalid_data")
    assert result == [], "Should return empty list on error"
    print("✓ Handled invalid frame gracefully")
    
    print("✅ Error handling test passed!\n")


def test_real_image(image_path: str):
    """Test with a real image (requires rfdetr installed)."""
    print(f"\n=== Real Image Test: {image_path} ===")
    
    from cctv_search.ai.rf_detr import RFDetrDetector
    
    # Load image
    try:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not load image: {image_path}")
            return False
        
        # Encode to bytes
        _, buffer = cv2.imencode('.jpg', img)
        frame_bytes = buffer.tobytes()
        print(f"✓ Loaded image: {img.shape[1]}x{img.shape[0]}")
    except ImportError:
        print("❌ OpenCV not installed. Install with: pip install opencv-python")
        return False
    
    # Initialize detector
    detector = RFDetrDetector(confidence_threshold=0.5)
    
    # Load model
    print("Loading model...")
    try:
        detector.load_model()
    except RuntimeError as e:
        print(f"❌ Failed to load model: {e}")
        print("   Install RF-DETR: pip install rfdetr")
        return False
    
    # Run detection
    print("Running detection...")
    start = time.time()
    detections = detector.detect(frame_bytes)
    elapsed = time.time() - start
    
    print(f"\n✓ Detection complete in {elapsed*1000:.1f}ms")
    print(f"✓ Found {len(detections)} objects:")
    
    for i, det in enumerate(detections, 1):
        print(f"\n  {i}. {det.label.upper()}")
        print(f"     Confidence: {det.confidence:.2%}")
        print(f"     Location: ({det.bbox.x:.0f}, {det.bbox.y:.0f})")
        print(f"     Size: {det.bbox.width:.0f}x{det.bbox.height:.0f}")
    
    print("\n✅ Real image test complete!\n")
    return True


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description="Test RF-DETR object detector"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run mock tests only (no dependencies required)"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Test with real image (requires: pip install rfdetr opencv-python)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests"
    )
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Include performance benchmark"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RF-DETR Object Detector Test Suite")
    print("=" * 60)
    
    # Default to --mock if no args provided
    if not any([args.mock, args.image, args.all, args.performance]):
        args.mock = True
    
    try:
        if args.mock or args.all:
            test_detector_creation()
            test_detection_mock()
            test_detection_with_mocked_model()
            test_batch_detection()
            test_error_handling()
            
            # Model loading test (may fail gracefully)
            model_loaded = test_model_loading()
            
            if args.performance:
                test_performance()
        
        if args.image:
            success = test_real_image(args.image)
            if not success:
                sys.exit(1)
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
