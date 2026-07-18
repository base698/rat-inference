"""Tests for image-space detection to servo-space observation conversion."""

from __future__ import annotations

import unittest

import numpy as np

from ratbot.robot.observation import ObservationConfig, TrackingObservationConverter


class FakeAiming:
    def __init__(self, depth_adjust=None):
        self.depth_adjust = depth_adjust
        self.calls = []

    def depth_adjust_px(self, depth_mm, focal_y):
        self.calls.append((depth_mm, focal_y))
        return self.depth_adjust


class FakeStereoDepth:
    def __init__(self):
        self.K1 = None
        self.camera_matrix = None


class TrackingObservationConverterTests(unittest.TestCase):
    def make_converter(self, pitch_scale=1.0, aiming=None, stereo=None):
        return TrackingObservationConverter(
            config=ObservationConfig(
                image_width=640,
                image_height=480,
                horizontal_fov_degrees=60.0,
                vertical_fov_degrees=45.0,
                yaw_min=1600,
                yaw_max=3100,
                pitch_min=1,
                pitch_max=500,
                yaw_range_degrees=180.0,
                pitch_range_degrees=55.0,
                pitch_tracking_scale=pitch_scale,
            ),
            aiming=aiming or FakeAiming(),
            stereo_depth=stereo or FakeStereoDepth(),
            crosshair_x=lambda yaw: 298,
            crosshair_y=lambda pitch: 199,
        )

    def test_crosshair_pixel_maps_to_current_servo_position(self):
        converter = self.make_converter()

        observation = converter.to_servo_target(
            target_x=298,
            target_y=199,
            current_yaw=2000,
            current_pitch=250,
        )

        self.assertEqual(observation["yaw"], 2000)
        self.assertEqual(observation["pitch"], 250)
        self.assertEqual(observation["yaw_offset_raw"], 0)
        self.assertEqual(observation["pitch_offset_raw"], 0)

    def test_pixel_offsets_convert_through_fov_and_axis_ranges(self):
        converter = self.make_converter()

        observation = converter.to_servo_target(
            target_x=398,
            target_y=299,
            current_yaw=2000,
            current_pitch=250,
        )

        self.assertEqual(observation["yaw_offset_raw"], 78)
        self.assertEqual(observation["pitch_offset_raw"], 85)
        self.assertEqual(observation["yaw"], 2078)
        self.assertEqual(observation["pitch"], 335)

    def test_pitch_tracking_scale_is_applied_before_clamping(self):
        converter = self.make_converter(pitch_scale=0.5)

        observation = converter.to_servo_target(
            target_x=298,
            target_y=299,
            current_yaw=2000,
            current_pitch=250,
        )

        self.assertEqual(observation["pitch_offset_raw"], 42)
        self.assertEqual(observation["pitch"], 292)

    def test_servo_targets_are_clamped_to_configured_bounds(self):
        converter = self.make_converter()

        observation = converter.to_servo_target(
            target_x=10000,
            target_y=-10000,
            current_yaw=3000,
            current_pitch=10,
        )

        self.assertEqual(observation["yaw"], 3100)
        self.assertEqual(observation["pitch"], 1)

    def test_depth_compensation_shifts_vertical_reference(self):
        aiming = FakeAiming(depth_adjust=12.4)
        stereo = FakeStereoDepth()
        stereo.K1 = np.array(
            [[100.0, 0.0, 0.0], [0.0, 200.0, 0.0], [0.0, 0.0, 1.0]]
        )
        converter = self.make_converter(aiming=aiming, stereo=stereo)

        observation = converter.to_servo_target(
            target_x=298,
            target_y=211,
            current_yaw=2000,
            current_pitch=250,
            depth_mm=1000.0,
        )

        self.assertEqual(aiming.calls, [(1000.0, 200.0)])
        self.assertEqual(observation["pixel_error_y"], 0)
        self.assertEqual(observation["pitch"], 250)

    def test_servo_raw_to_angle_is_inverse_for_each_axis(self):
        converter = self.make_converter()

        self.assertAlmostEqual(
            converter.servo_raw_to_angle(
                converter.angle_to_servo_raw(12.0, axis="yaw"),
                axis="yaw",
            ),
            12.0,
            delta=0.2,
        )
        self.assertAlmostEqual(
            converter.servo_raw_to_angle(
                converter.angle_to_servo_raw(12.0, axis="pitch"),
                axis="pitch",
            ),
            12.0,
            delta=0.2,
        )


if __name__ == "__main__":
    unittest.main()
