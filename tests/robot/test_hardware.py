"""Tests for isolated tracking and trigger servo adapters."""

from __future__ import annotations

import tempfile
import threading
import unittest
import os
from pathlib import Path
from unittest.mock import Mock, patch

import rt_200
from ratbot.robot.belief import ServoBounds
from ratbot.robot.hardware import (
    TrackingServoController,
    TriggerServoConfig,
    TriggerServoController,
)
from rt_200 import CameraTracker


class FakeMotorBus:
    def __init__(self, yaw=2100, pitch=225):
        self.positions = {"yaw": yaw, "pitch": pitch}
        self.writes = []
        self.disconnected = False

    def read(self, register, motor_name, normalize=False):
        self.assert_register(register, "Present_Position", normalize)
        return self.positions[motor_name]

    def write(self, register, motor_name, value, normalize=False):
        self.assert_register(register, "Goal_Position", normalize)
        self.writes.append((motor_name, value))

    def disconnect(self):
        self.disconnected = True

    @staticmethod
    def assert_register(actual, expected, normalize):
        if actual != expected or normalize is not False:
            raise AssertionError((actual, expected, normalize))


class TrackingServoControllerTests(unittest.TestCase):
    def make_controller(self, **kwargs):
        defaults = dict(
            port="/dev/ttyACM0",
            enabled=True,
            bounds=ServoBounds(1600, 3100, 1, 500),
            yaw_center=2200,
            pitch_center=250,
            yaw_motor_id=1,
            pitch_motor_id=5,
        )
        defaults.update(kwargs)
        return TrackingServoController(**defaults)

    def test_simulation_updates_clamped_positions_without_bus(self):
        controller = self.make_controller()

        controller.set_yaw(9999)
        controller.set_pitch(-50)

        self.assertEqual(controller.current_yaw, 3100)
        self.assertEqual(controller.current_pitch, 1)

    def test_connected_controller_writes_unsigned_raw_positions(self):
        bus = FakeMotorBus()
        controller = self.make_controller()
        controller.motor_bus = bus
        controller.connected = True

        controller.raw_write("yaw", -1)

        self.assertEqual(bus.writes, [("yaw", 65535)])

    def test_read_positions_uses_bus_under_lock(self):
        bus = FakeMotorBus(yaw=2050, pitch=240)
        controller = self.make_controller()
        controller.motor_bus = bus
        controller.connected = True
        controller.lock = threading.Lock()

        self.assertEqual(controller.read_positions(), (2050, 240))

    def test_strict_measured_readback_fails_closed_on_bus_error(self):
        controller = self.make_controller()
        controller.motor_bus = Mock()
        controller.motor_bus.read.side_effect = RuntimeError("bus read failed")
        controller.connected = True

        with self.assertRaises(RuntimeError):
            controller.read_measured_positions()

    def test_disconnect_closes_bus_and_marks_controller_disconnected(self):
        bus = FakeMotorBus()
        controller = self.make_controller()
        controller.motor_bus = bus
        controller.connected = True

        controller.close()

        self.assertTrue(bus.disconnected)
        self.assertFalse(controller.connected)


class CameraTrackerHardwareWiringTests(unittest.TestCase):
    def test_tracker_preserves_robot_interface_in_simulation_mode(self):
        tracker = CameraTracker(
            enable_servos=False,
            no_connect=True,
            enable_camera=False,
        )

        tracker.set_yaw(9999)
        tracker.set_pitch(-10)

        self.assertEqual(tracker.current_yaw, 3100)
        self.assertEqual(tracker.current_pitch, 1)
        self.assertFalse(tracker.connected)
        self.assertFalse(tracker.trigger_servo_enabled)
        tracker.disconnect()

    def test_detection_snapshots_prune_old_files_and_throttle_writes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prior_dir = rt_200.DETECTIONS_DIR
            rt_200.DETECTIONS_DIR = temp_dir
            try:
                old_snapshot = Path(temp_dir) / "detection_old.jpg"
                old_snapshot.write_bytes(b"old")
                os.utime(old_snapshot, (0, 0))

                tracker = CameraTracker(
                    enable_servos=False,
                    no_connect=True,
                    enable_camera=False,
                )
                self.assertFalse(old_snapshot.exists())

                with patch("rt_200.cv2.imwrite", return_value=True):
                    first = tracker._save_detection_snapshot(None, now=100.0)
                    second = tracker._save_detection_snapshot(None, now=101.0)

                self.assertIsNotNone(first)
                self.assertIsNone(second)
                tracker.disconnect()
            finally:
                rt_200.DETECTIONS_DIR = prior_dir


class TriggerServoControllerTests(unittest.TestCase):
    def make_sysfs_tree(self, root: Path):
        chip = root / "pwmchip0"
        pwm = chip / "pwm0"
        pwm.mkdir(parents=True)
        (chip / "export").write_text("")
        (chip / "unexport").write_text("")
        (pwm / "period").write_text("")
        (pwm / "enable").write_text("")
        (pwm / "duty_cycle").write_text("")
        return chip, pwm

    def test_initialize_configures_period_enable_and_neutral_angle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _, pwm = self.make_sysfs_tree(root)
            controller = TriggerServoController(
                enabled=True,
                config=TriggerServoConfig(0, 0, 99, 38),
                sysfs_root=root,
                sleep=lambda _: None,
            )

            controller.initialize()

            expected_neutral = int(500000 + (99 / 180.0) * 2000000)
            self.assertEqual((pwm / "period").read_text(), "20000000")
            self.assertEqual((pwm / "enable").read_text(), "1")
            self.assertEqual((pwm / "duty_cycle").read_text(), str(expected_neutral))
            self.assertTrue(controller.enabled)

    def test_set_angle_clamps_to_supported_pulse_range(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _, pwm = self.make_sysfs_tree(root)
            controller = TriggerServoController(
                enabled=True,
                config=TriggerServoConfig(0, 0, 99, 38),
                sysfs_root=root,
                sleep=lambda _: None,
            )
            controller.pwm_path = pwm

            controller.set_angle(999)

            self.assertEqual((pwm / "duty_cycle").read_text(), "2500000")

    def test_fire_moves_to_action_then_returns_to_neutral(self):
        controller = TriggerServoController(
            enabled=True,
            config=TriggerServoConfig(0, 0, 99, 38),
            sleep=lambda _: None,
        )
        controller.set_angle = Mock()

        controller.fire()

        self.assertEqual(controller.set_angle.call_args_list[0].args, (38,))
        self.assertEqual(controller.set_angle.call_args_list[1].args, (99,))

    def test_close_disables_and_unexports_pwm_channel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chip, pwm = self.make_sysfs_tree(root)
            controller = TriggerServoController(
                enabled=True,
                config=TriggerServoConfig(0, 0, 99, 38),
                sysfs_root=root,
                sleep=lambda _: None,
            )
            controller.pwm_path = pwm

            controller.close()

            self.assertEqual((pwm / "enable").read_text(), "0")
            self.assertEqual((chip / "unexport").read_text(), "0")


if __name__ == "__main__":
    unittest.main()
