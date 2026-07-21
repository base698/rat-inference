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
        self.calls = 0

    def compute(self, gray_left, gray_right):
        self.calls += 1
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

    def test_stereo_rectification_uses_configured_image_size(self):
        service = StereoDepthService(image_size=(960, 720))
        service.K1 = np.eye(3)
        service.K2 = np.eye(3)
        service.D1 = np.zeros(5)
        service.D2 = np.zeros(5)
        service.R = np.eye(3)
        service.T = np.array([[10.0], [0.0], [0.0]])
        rectified_sizes = []
        map_sizes = []

        def fake_stereo_rectify(*args, **kwargs):
            rectified_sizes.append(args[4])
            p1 = np.array([[100.0, 0.0, 480.0, 0.0],
                           [0.0, 100.0, 360.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0]])
            p2 = p1.copy()
            p2[0, 3] = -1000.0
            return np.eye(3), np.eye(3), p1, p2, np.eye(4), None, None

        def fake_init_map(_k, _d, _r, _p, size, _type):
            map_sizes.append(size)
            return object(), object()

        with patch("cv2.stereoRectify", side_effect=fake_stereo_rectify), \
             patch("cv2.undistortPoints", return_value=np.array([[[480.0, 360.0]]], dtype=np.float32)), \
             patch("cv2.initUndistortRectifyMap", side_effect=fake_init_map):
            service.init_stereo_rectification()

        self.assertEqual(rectified_sizes, [(960, 720)])
        self.assertEqual(map_sizes, [(960, 720), (960, 720)])

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

    def test_depth_below_configured_limit_is_rejected(self):
        service = StereoDepthService(
            min_texture_std=0.0,
            min_valid_mm=300.0,
            max_valid_mm=0.0,
        )
        service.stereo_calibration_enabled = True
        service.stereo_matcher = FakeMatcher(25.0)
        service.stereo_focal_length = 100.0
        service.baseline = 50.0
        service.K1 = np.eye(3)
        service.D1 = np.zeros(5)
        textured = np.indices((64, 64)).sum(axis=0).astype(np.uint8)
        frame = np.repeat(textured[:, :, None], 3, axis=2)

        self.assertIsNone(service.calculate_depth(frame, frame, 32, 32))
        self.assertEqual(service.last_depth_debug, "depth too near 0.20m")

    def test_batch_depth_computes_one_disparity_map_and_returns_quality(self):
        service = StereoDepthService(min_texture_std=0.0, max_valid_mm=0.0)
        service.stereo_calibration_enabled = True
        matcher = FakeMatcher(25.0)
        service.stereo_matcher = matcher
        service.stereo_focal_length = 100.0
        service.baseline = 50.0
        service.K1 = np.array(
            [[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]]
        )
        service.D1 = np.zeros(5)
        textured = np.indices((64, 64)).sum(axis=0).astype(np.uint8)
        frame = np.repeat(textured[:, :, None], 3, axis=2)

        measurements = service.calculate_depths(
            frame,
            frame,
            [(20, 20), (40, 40)],
        )

        self.assertEqual(matcher.calls, 1)
        self.assertEqual(len(measurements), 2)
        self.assertTrue(all(item is not None for item in measurements))
        self.assertTrue(all(abs(item.depth_mm - 200.0) < 1e-6 for item in measurements))
        self.assertTrue(all(item.valid_ratio > 0.9 for item in measurements))
        self.assertTrue(all(item.confidence > 0.9 for item in measurements))
        self.assertTrue(all(item.covariance_camera.shape == (3, 3) for item in measurements))
        self.assertTrue(all(np.all(np.diag(item.covariance_camera) > 0) for item in measurements))

    def test_batch_depth_keeps_good_points_when_another_point_is_out_of_frame(self):
        service = StereoDepthService(min_texture_std=0.0, max_valid_mm=0.0)
        service.stereo_calibration_enabled = True
        service.stereo_matcher = FakeMatcher(20.0)
        service.stereo_focal_length = 100.0
        service.baseline = 50.0
        service.K1 = np.eye(3)
        service.D1 = np.zeros(5)
        textured = np.indices((64, 64)).sum(axis=0).astype(np.uint8)
        frame = np.repeat(textured[:, :, None], 3, axis=2)

        measurements = service.calculate_depths(frame, frame, [(32, 32), (999, 999)])

        self.assertIsNotNone(measurements[0])
        self.assertIsNone(measurements[1])

    def test_depth_point_uses_raw_left_camera_ray_not_rectified_projection(self):
        service = StereoDepthService(min_texture_std=0.0, max_valid_mm=0.0)
        service.stereo_calibration_enabled = True
        service.stereo_matcher = FakeMatcher(25.0)
        service.stereo_focal_length = 100.0
        service.baseline = 50.0
        service.K1 = np.array(
            [[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]]
        )
        service.D1 = np.zeros(5)
        service.P1 = np.array(
            [
                [100.0, 0.0, -250.0, 0.0],
                [0.0, 100.0, -100.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        service.rectify_stereo_point = lambda _x, _y: (40, 40)
        textured = np.indices((64, 64)).sum(axis=0).astype(np.uint8)
        frame = np.repeat(textured[:, :, None], 3, axis=2)

        measurements = service.calculate_depths(frame, frame, [(32, 32)])

        self.assertIsNotNone(measurements[0])
        np.testing.assert_allclose(
            measurements[0].point_camera_mm,
            [0.0, 0.0, 200.0],
        )

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
