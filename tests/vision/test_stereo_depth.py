"""Tests for isolated camera calibration and stereo depth processing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from ratbot.vision.stereo_depth import StereoDepthService
from rt_200 import CameraTracker


class FakeMatcher:
    def __init__(self, disparity_pixels: float):
        self.disparity_pixels = disparity_pixels

    def compute(self, gray_left, gray_right):
        return np.full(gray_left.shape, self.disparity_pixels * 16.0, dtype=np.float32)


class StereoDepthServiceTests(unittest.TestCase):
    def test_missing_calibration_file_leaves_service_disabled(self):
        service = StereoDepthService(calibration_file="/definitely/missing/calibration.npz")

        self.assertFalse(service.calibration_enabled)
        self.assertFalse(service.stereo_calibration_enabled)

    def test_single_camera_calibration_loads_matrix_and_distortion(self):
        matrix = np.array([[100.0, 0.0, 32.0], [0.0, 101.0, 24.0], [0.0, 0.0, 1.0]])
        distortion = np.zeros(5)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "single.npz"
            np.savez(path, camera_matrix=matrix, dist_coeffs=distortion, rms_error=0.2)

            service = StereoDepthService(calibration_file=str(path))

        self.assertTrue(service.calibration_enabled)
        self.assertFalse(service.stereo_calibration_enabled)
        np.testing.assert_array_equal(service.camera_matrix, matrix)
        np.testing.assert_array_equal(service.dist_coeffs, distortion)

    def test_depth_is_unavailable_when_stereo_is_disabled(self):
        service = StereoDepthService()
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        self.assertIsNone(service.calculate_depth(frame, frame, 32, 32))
        self.assertEqual(service.last_depth_debug, "stereo disabled")

    def test_depth_uses_median_disparity_focal_length_and_baseline(self):
        service = StereoDepthService(min_texture_std=0.0, max_valid_mm=0.0)
        service.stereo_calibration_enabled = True
        service.stereo_matcher = FakeMatcher(32.0)
        service.stereo_focal_length = 100.0
        service.baseline = 50.0
        service.K1 = np.eye(3)
        service.D1 = np.zeros(5)
        textured = np.indices((64, 64)).sum(axis=0).astype(np.uint8)
        frame = np.repeat(textured[:, :, None], 3, axis=2)

        depth = service.calculate_depth(frame, frame, 32, 32)

        self.assertAlmostEqual(depth, 156.25)
        self.assertEqual(service.last_depth_debug, "disp 32.00px")

    def test_low_texture_region_is_rejected(self):
        service = StereoDepthService(min_texture_std=1.0)
        service.stereo_calibration_enabled = True
        service.stereo_matcher = FakeMatcher(16.0)
        service.stereo_focal_length = 100.0
        service.baseline = 50.0
        service.K1 = np.eye(3)
        service.D1 = np.zeros(5)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        self.assertIsNone(service.calculate_depth(frame, frame, 32, 32))
        self.assertEqual(service.last_depth_debug, "low texture 0.0")

    def test_negative_disparity_camera_order_is_supported(self):
        service = StereoDepthService(min_texture_std=0.0, max_valid_mm=0.0)
        service.stereo_calibration_enabled = True
        service.stereo_matcher = FakeMatcher(-20.0)
        service.stereo_disparity_sign = -1
        service.stereo_min_disparity = -192
        service.stereo_focal_length = 100.0
        service.baseline = 50.0
        service.K1 = np.eye(3)
        service.D1 = np.zeros(5)
        textured = np.indices((64, 64)).sum(axis=0).astype(np.uint8)
        frame = np.repeat(textured[:, :, None], 3, axis=2)

        depth = service.calculate_depth(frame, frame, 32, 32)

        self.assertAlmostEqual(depth, 250.0)
        self.assertEqual(service.last_depth_debug, "disp 20.00px")

    def test_depth_above_configured_limit_is_rejected(self):
        service = StereoDepthService(min_texture_std=0.0, max_valid_mm=200.0)
        service.stereo_calibration_enabled = True
        service.stereo_matcher = FakeMatcher(10.0)
        service.stereo_focal_length = 100.0
        service.baseline = 50.0
        service.K1 = np.eye(3)
        service.D1 = np.zeros(5)
        textured = np.indices((64, 64)).sum(axis=0).astype(np.uint8)
        frame = np.repeat(textured[:, :, None], 3, axis=2)

        self.assertIsNone(service.calculate_depth(frame, frame, 32, 32))
        self.assertEqual(service.last_depth_debug, "depth too far 0.50m")

    def test_rectification_failure_returns_original_frames(self):
        service = StereoDepthService()
        service.stereo_map_left = (object(), object())
        service.stereo_map_right = (object(), object())
        left = np.zeros((2, 2, 3), dtype=np.uint8)
        right = np.ones((2, 2, 3), dtype=np.uint8)

        with patch.object(cv2, "remap", side_effect=RuntimeError("bad map")):
            actual_left, actual_right = service.rectify_stereo_frames(left, right)

        self.assertIs(actual_left, left)
        self.assertIs(actual_right, right)


class CameraTrackerStereoWiringTests(unittest.TestCase):
    def test_runtime_calibration_status_delegates_to_stereo_service(self):
        tracker = CameraTracker(
            enable_servos=False,
            no_connect=True,
            enable_camera=False,
        )

        self.assertIs(
            tracker.calibration_enabled,
            tracker.stereo_depth.calibration_enabled,
        )
        self.assertIs(
            tracker.stereo_calibration_enabled,
            tracker.stereo_depth.stereo_calibration_enabled,
        )


if __name__ == "__main__":
    unittest.main()
