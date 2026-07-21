"""Tests for adapting selected 3D tracks to the bounded servo controller."""

from __future__ import annotations

import unittest

import numpy as np

from ratbot.tracking.aiming import WorldTrackBeliefAdapter
from ratbot.tracking.geometry import ServoKinematicsConfig, TurretFrameTransformer
from ratbot.tracking.models import Detection3D
from ratbot.tracking.multi_target import MultiTargetTracker, TrackManagerConfig


class FakeRobot:
    def __init__(
        self,
        yaw=2200,
        pitch=250,
        raw_per_degree=10,
        crosshair_x=320,
        crosshair_y=240,
    ):
        self.current_yaw = yaw
        self.current_pitch = pitch
        self.raw_per_degree = raw_per_degree
        self.crosshair_x = crosshair_x
        self.crosshair_y = crosshair_y
        self.stereo_depth = type(
            "FakeStereoDepth",
            (),
            {
                "K1": np.array(
                    [[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]]
                ),
                "camera_matrix": None,
            },
        )()

    def angle_to_servo_raw(self, angle_delta, axis="yaw"):
        return int(round(float(angle_delta) * self.raw_per_degree))

    def pixel_to_target_position(self, target_x, target_y, depth_mm=None):
        yaw_degrees = (float(target_x) - self.crosshair_x) * (60.0 / 640.0)
        pitch_degrees = (float(target_y) - self.crosshair_y) * (45.0 / 480.0)
        return {
            "yaw": self.current_yaw + self.angle_to_servo_raw(yaw_degrees, axis="yaw"),
            "pitch": self.current_pitch
            + self.angle_to_servo_raw(pitch_degrees, axis="pitch"),
            "angle_error_yaw": yaw_degrees,
            "angle_error_pitch": pitch_degrees,
        }


class WorldTrackBeliefAdapterTests(unittest.TestCase):
    def setUp(self):
        self.manager = MultiTargetTracker(
            TrackManagerConfig(confirm_hits=1, auto_select=True)
        )
        self.transformer = TurretFrameTransformer(
            ServoKinematicsConfig(
                yaw_center_raw=2200,
                pitch_center_raw=250,
                yaw_raw_per_degree=10,
                pitch_raw_per_degree=10,
                yaw_sign=1,
                pitch_sign=-1,
                yaw_min_raw=1600,
                yaw_max_raw=3100,
                pitch_min_raw=1,
                pitch_max_raw=500,
            )
        )

    def add_track(self, position, timestamp=10.0, confidence=0.9, center=None):
        self.manager.update(
            [Detection3D(
                position_base_mm=np.asarray(position, dtype=float),
                covariance_base=np.eye(3) * 25,
                confidence=confidence,
                classification="rat",
                measurement_time=timestamp,
                center=center,
            )],
            timestamp=timestamp,
        )

    def test_selected_world_position_becomes_servo_belief(self):
        self.add_track([1000, 1000, 0])
        adapter = WorldTrackBeliefAdapter(
            self.manager,
            self.transformer,
            aim_latency_seconds=0.0,
            min_confidence=0.2,
            max_age_seconds=1.0,
            clock=lambda: 10.1,
        )

        belief = adapter.get_active()

        self.assertEqual(belief["yaw"], 2650)
        self.assertEqual(belief["pitch"], 250)
        self.assertEqual(belief["track_id"], self.manager.selected_track_id)
        self.assertAlmostEqual(belief["age"], 0.1)

    def test_stale_or_unselected_track_produces_no_control_belief(self):
        self.add_track([1000, 0, 0])
        stale = WorldTrackBeliefAdapter(
            self.manager,
            self.transformer,
            max_age_seconds=0.5,
            clock=lambda: 11.0,
        )
        self.assertIsNone(stale.get_active())

        self.manager.clear_selection()
        fresh = WorldTrackBeliefAdapter(
            self.manager,
            self.transformer,
            clock=lambda: 10.1,
        )
        self.assertIsNone(fresh.get_active())

    def test_latency_prediction_leads_moving_target_without_mutating_track(self):
        self.add_track([1000, 0, 0])
        track = self.manager._tracks[self.manager.selected_track_id]
        track.filter.state[4] = 1000.0
        adapter = WorldTrackBeliefAdapter(
            self.manager,
            self.transformer,
            aim_latency_seconds=0.5,
            max_age_seconds=2.0,
            clock=lambda: 10.0,
        )

        belief = adapter.get_active()

        self.assertGreater(belief["yaw"], 2200)
        np.testing.assert_allclose(track.filter.position, [1000, 0, 0])

    def test_turret_relative_aim_uses_current_camera_basis(self):
        robot = FakeRobot(yaw=2325, pitch=210)
        position = self.transformer.camera_to_base(
            [300, 0, 1000],
            robot.current_yaw,
            robot.current_pitch,
        )
        self.add_track(position)
        adapter = WorldTrackBeliefAdapter(
            self.manager,
            self.transformer,
            aim_latency_seconds=0.0,
            max_age_seconds=1.0,
            robot=robot,
            clock=lambda: 10.1,
        )

        belief = adapter.get_active()

        self.assertGreater(belief["yaw"], robot.current_yaw)
        self.assertAlmostEqual(belief["pitch"], robot.current_pitch, delta=1)

    def test_turret_relative_aim_tracks_camera_vertical_error(self):
        robot = FakeRobot(yaw=2200, pitch=250)
        position = self.transformer.camera_to_base(
            [0, 250, 1000],
            robot.current_yaw,
            robot.current_pitch,
        )
        self.add_track(position)
        adapter = WorldTrackBeliefAdapter(
            self.manager,
            self.transformer,
            aim_latency_seconds=0.0,
            max_age_seconds=1.0,
            robot=robot,
            clock=lambda: 10.1,
        )

        belief = adapter.get_active()

        self.assertAlmostEqual(belief["yaw"], robot.current_yaw, delta=1)
        self.assertGreater(belief["pitch"], robot.current_pitch)

    def test_turret_relative_aim_uses_direct_world_ray_not_crosshair_projection(self):
        robot = FakeRobot(yaw=2200, pitch=250, crosshair_y=340)
        position = self.transformer.camera_to_base(
            [0, 100, 1000],
            robot.current_yaw,
            robot.current_pitch,
        )
        self.add_track(position)
        adapter = WorldTrackBeliefAdapter(
            self.manager,
            self.transformer,
            aim_latency_seconds=0.0,
            max_age_seconds=1.0,
            robot=robot,
            clock=lambda: 10.1,
        )

        belief = adapter.get_active()

        self.assertGreater(belief["pitch"], robot.current_pitch)
        self.assertEqual(belief["aim_source"], "world_ray")

    def test_robot_world_aim_ignores_fresh_center_by_default(self):
        robot = FakeRobot(yaw=2200, pitch=250, crosshair_y=340)
        noisy_position = self.transformer.camera_to_base(
            [0, -300, 1000],
            robot.current_yaw,
            robot.current_pitch,
        )
        self.add_track(noisy_position, center=(320, 340))
        adapter = WorldTrackBeliefAdapter(
            self.manager,
            self.transformer,
            aim_latency_seconds=0.0,
            max_age_seconds=1.0,
            robot=robot,
            clock=lambda: 10.1,
        )

        belief = adapter.get_active()

        self.assertLess(belief["pitch"], robot.current_pitch)

    def test_fresh_selected_center_preference_is_opt_in(self):
        robot = FakeRobot(yaw=2200, pitch=250, crosshair_y=340)
        noisy_position = self.transformer.camera_to_base(
            [0, -300, 1000],
            robot.current_yaw,
            robot.current_pitch,
        )
        self.add_track(noisy_position, center=(320, 340))
        adapter = WorldTrackBeliefAdapter(
            self.manager,
            self.transformer,
            aim_latency_seconds=0.0,
            max_age_seconds=1.0,
            prefer_fresh_center=True,
            robot=robot,
            clock=lambda: 10.1,
        )

        belief = adapter.get_active()

        self.assertAlmostEqual(belief["pitch"], robot.current_pitch, delta=1)

    def test_stale_selected_center_falls_back_to_world_aim(self):
        robot = FakeRobot(yaw=2200, pitch=250, crosshair_y=340)
        noisy_position = self.transformer.camera_to_base(
            [0, -300, 1000],
            robot.current_yaw,
            robot.current_pitch,
        )
        self.add_track(noisy_position, center=(320, 340))
        adapter = WorldTrackBeliefAdapter(
            self.manager,
            self.transformer,
            aim_latency_seconds=0.0,
            max_age_seconds=1.0,
            robot=robot,
            clock=lambda: 10.4,
        )

        belief = adapter.get_active()

        self.assertLess(belief["pitch"], robot.current_pitch)

    def test_missed_selected_center_falls_back_to_world_aim(self):
        robot = FakeRobot(yaw=2200, pitch=250, crosshair_y=340)
        noisy_position = self.transformer.camera_to_base(
            [0, -300, 1000],
            robot.current_yaw,
            robot.current_pitch,
        )
        self.add_track(noisy_position, center=(320, 340))
        self.manager.update([], timestamp=10.05)
        adapter = WorldTrackBeliefAdapter(
            self.manager,
            self.transformer,
            aim_latency_seconds=0.0,
            max_age_seconds=1.0,
            robot=robot,
            clock=lambda: 10.1,
        )

        belief = adapter.get_active()

        self.assertLess(belief["pitch"], robot.current_pitch)

    def test_robot_world_aim_does_not_lead_with_track_velocity(self):
        robot = FakeRobot(yaw=2200, pitch=250)
        position = self.transformer.camera_to_base(
            [0, 0, 1000],
            robot.current_yaw,
            robot.current_pitch,
        )
        self.add_track(position)
        track = self.manager._tracks[self.manager.selected_track_id]
        track.filter.state[4] = 1000.0
        adapter = WorldTrackBeliefAdapter(
            self.manager,
            self.transformer,
            aim_latency_seconds=0.5,
            max_age_seconds=1.0,
            robot=robot,
            clock=lambda: 10.1,
        )

        belief = adapter.get_active()

        self.assertAlmostEqual(belief["yaw"], robot.current_yaw, delta=1)
        self.assertEqual(belief["yaw_velocity"], 0.0)
        self.assertEqual(belief["pitch_velocity"], 0.0)
        self.assertEqual(belief["prediction_dt"], 0.0)


if __name__ == "__main__":
    unittest.main()
