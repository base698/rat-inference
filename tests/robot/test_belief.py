"""Characterization tests for angular target belief and servo control."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from ratbot.robot.belief import (
    AngularBeliefController,
    AngularTargetBelief,
    ServoBounds,
)


class AngularTargetBeliefTests(unittest.TestCase):
    def test_first_observation_reseeds_empty_belief(self):
        belief = AngularTargetBelief()

        with patch("ratbot.robot.belief.time.time", return_value=10.0):
            snapshot = belief.update(2100, 225, 0.8)

        self.assertTrue(snapshot["reseeded"])
        self.assertEqual(snapshot["reseed_reason"], "empty")
        self.assertFalse(snapshot["ignored"])
        self.assertEqual(snapshot["yaw"], 2100.0)
        self.assertEqual(snapshot["pitch"], 225.0)
        self.assertEqual(snapshot["yaw_velocity"], 0.0)
        self.assertEqual(snapshot["pitch_velocity"], 0.0)

    def test_nearby_observation_smooths_position_and_caps_axis_velocities(self):
        belief = AngularTargetBelief(
            update_alpha=0.5,
            pitch_update_alpha=0.5,
            velocity_alpha=1.0,
            pitch_velocity_alpha=1.0,
            max_velocity_raw_per_s=100,
            max_pitch_velocity_raw_per_s=20,
            reseed_distance_raw=0,
        )

        with patch("ratbot.robot.belief.time.time", side_effect=[10.0, 11.0]):
            belief.update(100, 100, 0.9)
            snapshot = belief.update(300, 200, 0.9)

        self.assertEqual(snapshot["yaw"], 200.0)
        self.assertEqual(snapshot["pitch"], 150.0)
        self.assertEqual(snapshot["yaw_velocity"], 100.0)
        self.assertEqual(snapshot["pitch_velocity"], 20.0)
        self.assertFalse(snapshot["reseeded"])

    def test_matching_second_large_jump_confirms_reseed(self):
        belief = AngularTargetBelief(
            miss_decay=0.5,
            reseed_distance_raw=100,
            reseed_confirmations=2,
            reseed_match_distance_raw=20,
            reseed_max_interval=1.0,
            max_prediction_age=0,
        )

        with patch("ratbot.robot.belief.time.time", side_effect=[1.0, 1.1, 1.2]):
            belief.update(2000, 200, 0.9)
            pending = belief.update(2200, 300, 0.9)
            confirmed = belief.update(2205, 305, 0.9)

        self.assertTrue(pending["ignored"])
        self.assertIn("pending", pending["ignored_reason"])
        self.assertFalse(pending["reseeded"])
        self.assertTrue(confirmed["reseeded"])
        self.assertIn("confirmed", confirmed["reseed_reason"])
        self.assertEqual(confirmed["yaw"], 2205.0)
        self.assertEqual(confirmed["pitch"], 305.0)
        self.assertEqual(confirmed["yaw_velocity"], 0.0)
        self.assertEqual(confirmed["pitch_velocity"], 0.0)

    def test_decay_eventually_deactivates_belief(self):
        belief = AngularTargetBelief(
            miss_decay=0.5,
            min_confidence=0.5,
            max_age=100,
        )

        with patch("ratbot.robot.belief.time.time", return_value=1.0):
            belief.update(2000, 200, 1.0)
        belief.decay()
        with patch("ratbot.robot.belief.time.time", return_value=1.1):
            self.assertIsNotNone(belief.get_active())
        belief.decay()
        with patch("ratbot.robot.belief.time.time", return_value=1.2):
            self.assertIsNone(belief.get_active())

    def test_active_snapshot_bounds_prediction_and_rejects_stale_state(self):
        belief = AngularTargetBelief(
            min_confidence=0.1,
            max_age=1.0,
            max_prediction_age=0.25,
        )
        with patch("ratbot.robot.belief.time.time", return_value=1.0):
            belief.update(100, 50, 1.0)
        belief.yaw_velocity = 40.0
        belief.pitch_velocity = -20.0

        with patch("ratbot.robot.belief.time.time", return_value=1.5):
            active = belief.get_active()
        self.assertIsNotNone(active)
        self.assertEqual(active["prediction_dt"], 0.25)
        self.assertEqual(active["yaw"], 110.0)
        self.assertEqual(active["pitch"], 45.0)

        with patch("ratbot.robot.belief.time.time", return_value=2.1):
            self.assertIsNone(belief.get_active())

    def test_clear_resets_all_tracking_state(self):
        belief = AngularTargetBelief()
        with patch("ratbot.robot.belief.time.time", side_effect=[1.0, 2.0]):
            belief.update(100, 50, 0.9)
            snapshot = belief.clear()

        self.assertIsNone(snapshot["yaw"])
        self.assertIsNone(snapshot["pitch"])
        self.assertEqual(snapshot["confidence"], 0.0)
        self.assertEqual(snapshot["yaw_velocity"], 0.0)
        self.assertEqual(snapshot["pitch_velocity"], 0.0)
        self.assertIsNone(belief.pending_reseed_yaw)
        self.assertIsNone(belief.pending_reseed_pitch)
        self.assertEqual(belief.pending_reseed_count, 0)


class StaticBelief:
    def __init__(self, yaw, pitch, confidence=1.0):
        self.snapshot = {
            "yaw": yaw,
            "pitch": pitch,
            "confidence": confidence,
            "age": 0.0,
            "yaw_velocity": 0.0,
            "pitch_velocity": 0.0,
            "prediction_dt": 0.0,
        }

    def get_active(self):
        return self.snapshot


class FakeRobot:
    camera_active = False

    pid_yaw_kp = 1.0
    pid_yaw_ki = 0.0
    pid_yaw_kd = 0.0
    pid_pitch_kp = 1.0
    pid_pitch_ki = 0.0
    pid_pitch_kd = 0.0
    pid_max_integral = 10.0

    def __init__(self, yaw=50, pitch=50):
        self.current_yaw = yaw
        self.current_pitch = pitch
        self.yaw_writes = []
        self.pitch_writes = []

    def angle_to_servo_raw(self, angle, axis):
        return angle

    def servo_raw_to_angle(self, raw, axis):
        return raw

    def set_yaw(self, yaw):
        self.current_yaw = yaw
        self.yaw_writes.append(yaw)

    def set_pitch(self, pitch):
        self.current_pitch = pitch
        self.pitch_writes.append(pitch)


class AngularBeliefControllerTests(unittest.TestCase):
    def make_controller(self, robot, belief, **kwargs):
        bounds = kwargs.pop("bounds", ServoBounds(0, 4095, 0, 4095))
        with patch("ratbot.robot.belief.time.time", return_value=1.0):
            return AngularBeliefController(
                robot=robot,
                belief=belief,
                bounds=bounds,
                **kwargs,
            )

    def test_deadband_suppresses_servo_writes(self):
        robot = FakeRobot()
        controller = self.make_controller(
            robot,
            StaticBelief(52, 48),
            deadband_raw=4,
        )

        with patch("ratbot.robot.belief.time.time", return_value=2.0):
            controller.track_once()

        self.assertEqual(robot.yaw_writes, [])
        self.assertEqual(robot.pitch_writes, [])

    def test_minimum_step_is_applied_outside_deadband(self):
        robot = FakeRobot(yaw=2000, pitch=250)
        controller = self.make_controller(
            robot,
            StaticBelief(2001, 250),
            deadband_raw=0,
            min_step_raw=3,
            max_yaw_step=0,
            max_pitch_step=0,
        )

        with patch("ratbot.robot.belief.time.time", return_value=2.0):
            controller.track_once()

        self.assertEqual(robot.current_yaw, 2003)
        self.assertEqual(robot.current_pitch, 250)

    def test_per_tick_step_limit_is_applied(self):
        robot = FakeRobot()
        controller = self.make_controller(
            robot,
            StaticBelief(100, 0),
            deadband_raw=0,
            min_step_raw=0,
            max_yaw_step=5,
            max_pitch_step=7,
        )

        with patch("ratbot.robot.belief.time.time", return_value=2.0):
            controller.track_once()

        self.assertEqual(robot.current_yaw, 55)
        self.assertEqual(robot.current_pitch, 43)

    def test_runtime_servo_bounds_are_applied(self):
        robot = FakeRobot(yaw=95, pitch=5)
        controller = self.make_controller(
            robot,
            StaticBelief(200, -100),
            bounds=ServoBounds(0, 100, 0, 100),
            deadband_raw=0,
            min_step_raw=0,
            max_yaw_step=0,
            max_pitch_step=0,
        )

        with patch("ratbot.robot.belief.time.time", return_value=2.0):
            controller.track_once()

        self.assertEqual(robot.current_yaw, 100)
        self.assertEqual(robot.current_pitch, 0)

    def test_reset_clears_pid_history(self):
        robot = FakeRobot()
        controller = self.make_controller(robot, StaticBelief(60, 40))
        controller.yaw_integral = 5.0
        controller.pitch_integral = -4.0
        controller.yaw_prev_error = 3.0
        controller.pitch_prev_error = -2.0

        with patch("ratbot.robot.belief.time.time", return_value=3.0):
            controller.reset()

        self.assertEqual(controller.yaw_integral, 0.0)
        self.assertEqual(controller.pitch_integral, 0.0)
        self.assertEqual(controller.yaw_prev_error, 0.0)
        self.assertEqual(controller.pitch_prev_error, 0.0)
        self.assertEqual(controller.last_time, 3.0)


class CameraTrackerBeliefWiringTests(unittest.TestCase):
    def test_off_center_belief_moves_tracker_without_missing_pid_config(self):
        from rt_200 import CameraTracker

        tracker = CameraTracker(
            enable_servos=False,
            no_connect=True,
            enable_camera=False,
        )
        initial_yaw = tracker.current_yaw
        initial_pitch = tracker.current_pitch

        with patch("ratbot.robot.belief.time.time", return_value=100.0):
            tracker.target_belief.update(
                initial_yaw + 100,
                initial_pitch + 50,
                confidence=1.0,
            )
            tracker.tracking_controller.track_once()

        self.assertGreater(tracker.current_yaw, initial_yaw)
        self.assertGreater(tracker.current_pitch, initial_pitch)


if __name__ == "__main__":
    unittest.main()
