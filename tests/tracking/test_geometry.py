"""Tests for camera projection and fixed turret-base frame geometry."""

from __future__ import annotations

import math
import unittest

import numpy as np

from ratbot.tracking.geometry import ServoKinematicsConfig, TurretFrameTransformer


class TurretFrameTransformerTests(unittest.TestCase):
    def make_transformer(self, **overrides):
        values = {
            "yaw_center_raw": 2200.0,
            "pitch_center_raw": 250.0,
            "yaw_raw_per_degree": 10.0,
            "pitch_raw_per_degree": 10.0,
            "yaw_sign": 1.0,
            "pitch_sign": -1.0,
            "camera_translation_mm": (0.0, 0.0, 0.0),
            "camera_mount_rpy_degrees": (0.0, 0.0, 0.0),
            "yaw_min_raw": 1600.0,
            "yaw_max_raw": 3100.0,
            "pitch_min_raw": 1.0,
            "pitch_max_raw": 500.0,
        }
        values.update(overrides)
        return TurretFrameTransformer(ServoKinematicsConfig(**values))

    def test_camera_projection_uses_opencv_axes(self):
        transformer = self.make_transformer()
        intrinsics = np.array(
            [[100.0, 0.0, 320.0], [0.0, 200.0, 240.0], [0.0, 0.0, 1.0]]
        )

        center = transformer.pixel_depth_to_camera(320, 240, 1000, intrinsics)
        offset = transformer.pixel_depth_to_camera(420, 440, 1000, intrinsics)

        np.testing.assert_allclose(center, [0.0, 0.0, 1000.0])
        np.testing.assert_allclose(offset, [1000.0, 1000.0, 1000.0])

    def test_neutral_camera_axes_map_to_base_forward_left_up(self):
        transformer = self.make_transformer()

        forward = transformer.camera_to_base([0, 0, 1000], 2200, 250)
        camera_right = transformer.camera_to_base([1000, 0, 0], 2200, 250)
        camera_down = transformer.camera_to_base([0, 1000, 0], 2200, 250)

        np.testing.assert_allclose(forward, [1000, 0, 0], atol=1e-9)
        np.testing.assert_allclose(camera_right, [0, -1000, 0], atol=1e-9)
        np.testing.assert_allclose(camera_down, [0, 0, -1000], atol=1e-9)

    def test_yaw_and_pitch_rotate_stationary_camera_point_in_base_frame(self):
        transformer = self.make_transformer()

        yaw_left = transformer.camera_to_base([0, 0, 1000], 3100, 250)
        pitch_up = transformer.camera_to_base([0, 0, 1000], 2200, 150)

        np.testing.assert_allclose(yaw_left, [0, 1000, 0], atol=1e-6)
        angle = math.radians(10.0)
        np.testing.assert_allclose(
            pitch_up,
            [1000 * math.cos(angle), 0, 1000 * math.sin(angle)],
            atol=1e-6,
        )

    def test_stationary_base_point_is_invariant_across_camera_pose_changes(self):
        transformer = self.make_transformer()
        fixed_base = np.array([1600.0, 400.0, -120.0])

        neutral_camera = transformer.base_to_camera(fixed_base, 2200, 250)
        turned_camera = transformer.base_to_camera(fixed_base, 2650, 150)

        np.testing.assert_allclose(
            transformer.camera_to_base(neutral_camera, 2200, 250),
            fixed_base,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            transformer.camera_to_base(turned_camera, 2650, 150),
            fixed_base,
            atol=1e-8,
        )
        self.assertFalse(np.allclose(neutral_camera, turned_camera))

    def test_camera_mount_translation_rotates_with_turret(self):
        transformer = self.make_transformer(camera_translation_mm=(100.0, 0.0, 50.0))

        neutral = transformer.camera_to_base([0, 0, 0], 2200, 250)
        yaw_left = transformer.camera_to_base([0, 0, 0], 3100, 250)

        np.testing.assert_allclose(neutral, [100, 0, 50], atol=1e-6)
        np.testing.assert_allclose(yaw_left, [0, 100, 50], atol=1e-6)

    def test_covariance_is_rotated_without_translation(self):
        transformer = self.make_transformer()
        covariance_camera = np.diag([1.0, 4.0, 9.0])

        covariance_base = transformer.camera_covariance_to_base(
            covariance_camera, 2200, 250
        )

        np.testing.assert_allclose(covariance_base, np.diag([9.0, 1.0, 4.0]))

    def test_base_position_converts_to_bounded_servo_setpoint(self):
        transformer = self.make_transformer()

        raw = transformer.base_position_to_servo_raw([1000, 1000, 1000])
        behind = transformer.base_position_to_servo_raw([-1000, 0, 0])

        self.assertAlmostEqual(raw["yaw_degrees"], 45.0)
        self.assertAlmostEqual(raw["pitch_degrees"], math.degrees(math.atan2(1000, math.sqrt(2_000_000))))
        self.assertEqual(raw["yaw"], 2650)
        self.assertEqual(raw["pitch"], 1)
        self.assertEqual(behind["yaw"], 3100)

    def test_extrinsic_aware_inverse_aim_recovers_servo_pose(self):
        transformer = self.make_transformer(
            camera_translation_mm=(85.0, -12.0, 42.0),
            camera_mount_rpy_degrees=(2.0, -3.0, 4.0),
        )
        expected_yaw = 2575
        expected_pitch = 185
        target = transformer.camera_to_base(
            [0.0, 0.0, 1800.0],
            expected_yaw,
            expected_pitch,
        )

        aim = transformer.base_position_to_servo_raw(target)

        self.assertAlmostEqual(aim["yaw"], expected_yaw, delta=1)
        self.assertAlmostEqual(aim["pitch"], expected_pitch, delta=1)
        camera_at_aim = transformer.base_to_camera(
            target,
            aim["yaw"],
            aim["pitch"],
        )
        self.assertAlmostEqual(camera_at_aim[0], 0.0, delta=2.0)
        self.assertAlmostEqual(camera_at_aim[1], 0.0, delta=2.0)
        self.assertGreater(camera_at_aim[2], 0.0)

    def test_invalid_servo_scales_are_rejected(self):
        with self.assertRaises(ValueError):
            self.make_transformer(yaw_raw_per_degree=0)
        with self.assertRaises(ValueError):
            self.make_transformer(pitch_sign=0)


if __name__ == "__main__":
    unittest.main()
