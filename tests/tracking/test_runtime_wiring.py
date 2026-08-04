"""Hardware-free composition tests for opt-in world-frame tracking."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from ratbot.vision.stereo_depth import StereoPointMeasurement
from rt_200 import CameraTracker


class CameraTrackerWorldWiringTests(unittest.TestCase):
    def test_measurement_context_uses_synchronous_measured_pose_not_commanded_goal(self):
        tracker = CameraTracker(
            enable_servos=False,
            no_connect=True,
            enable_camera=False,
            world_tracking=True,
        )
        tracker.current_yaw = 2900
        tracker.current_pitch = 450
        tracker.tracking_servos.read_measured_positions = MagicMock(
            return_value=(2111, 222)
        )

        with patch("rt_200.time.monotonic", return_value=12.5):
            context = tracker.capture_world_measurement_context()

        self.assertEqual(context, (12.5, 2111, 222))
        self.assertNotEqual(context[1:], (tracker.current_yaw, tracker.current_pitch))
        tracker.tracking_servos.read_measured_positions.assert_called_once_with()

    def test_world_measurements_create_tracks_and_drive_bounded_controller(self):
        tracker = CameraTracker(
            enable_servos=False,
            no_connect=True,
            enable_camera=False,
            world_tracking=True,
            world_actuation_enabled=True,
            world_calibration_validated=True,
            world_confirm_hits=1,
            world_yaw_raw_per_degree=10.0,
            world_pitch_raw_per_degree=10.0,
            world_yaw_sign=1.0,
            world_pitch_sign=-1.0,
        )
        measurement = StereoPointMeasurement(
            depth_mm=1000.0,
            disparity_px=20.0,
            valid_ratio=1.0,
            disparity_iqr_px=0.0,
            texture_std=10.0,
            confidence=1.0,
            point_camera_mm=np.array([-500.0, 0.0, 1000.0]),
            covariance_camera=np.eye(3) * 25.0,
        )
        detection = {
            "confidence": 0.9,
            "class_name": "rat",
            "bbox": (10, 10, 20, 20),
            "center": (15, 15),
        }

        tracks = tracker.update_world_tracks(
            [detection],
            [measurement],
            timestamp=10.0,
            yaw_raw=tracker.current_yaw,
            pitch_raw=tracker.current_pitch,
        )
        self.assertEqual(tracker.world_tracker.selected_track_id, tracks[0].id)
        tracks = tracker.world_tracker.get_tracks()
        initial_yaw = tracker.current_yaw
        tracker.world_belief.clock = lambda: 10.0
        controller = tracker.tracking_controller
        if hasattr(controller, "clock"):
            # velocity-form controller: fixed dt tick; bound is velocity * dt
            tick = {"now": 10.0}
            controller.clock = lambda: tick["now"]
            controller.last_time = 10.0
            tick["now"] = 10.05
            controller.track_once()
            step_bound = controller.max_yaw_velocity * 0.05 + 1
        else:
            controller.track_once()
            step_bound = tracker.max_yaw_step

        self.assertEqual(len(tracks), 1)
        self.assertTrue(tracks[0].selected)
        self.assertLess(tracker.current_yaw, initial_yaw)
        self.assertLessEqual(initial_yaw - tracker.current_yaw, step_bound)

    def test_world_tracking_is_shadow_only_until_calibration_and_actuation_are_enabled(self):
        tracker = CameraTracker(
            enable_servos=False,
            no_connect=True,
            enable_camera=False,
            world_tracking=True,
            world_confirm_hits=1,
        )
        measurement = StereoPointMeasurement(
            depth_mm=1000.0,
            disparity_px=20.0,
            valid_ratio=1.0,
            disparity_iqr_px=0.0,
            texture_std=10.0,
            confidence=1.0,
            point_camera_mm=np.array([-500.0, 0.0, 1000.0]),
            covariance_camera=np.eye(3) * 25.0,
        )
        initial = (tracker.current_yaw, tracker.current_pitch)
        tracker.update_world_tracks(
            [{"confidence": 0.9, "class_name": "rat", "bbox": (0, 0, 1, 1), "center": (0, 0)}],
            [measurement],
            timestamp=10.0,
            yaw_raw=tracker.current_yaw,
            pitch_raw=tracker.current_pitch,
        )
        tracker.tracking_controller.track_once()

        self.assertEqual((tracker.current_yaw, tracker.current_pitch), initial)
        self.assertFalse(tracker.world_actuation_enabled)

    def test_world_actuation_requires_explicit_calibration_validation(self):
        with self.assertRaises(ValueError):
            CameraTracker(
                enable_servos=False,
                no_connect=True,
                enable_camera=False,
                world_tracking=True,
                world_actuation_enabled=True,
                world_calibration_validated=False,
            )

    def test_world_updates_can_be_logged_as_replayable_jsonl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "world.jsonl"
            tracker = CameraTracker(
                enable_servos=False,
                no_connect=True,
                enable_camera=False,
                world_tracking=True,
                world_confirm_hits=1,
                world_log_path=str(log_path),
            )
            stereo = StereoPointMeasurement(
                depth_mm=1000.0,
                disparity_px=20.0,
                valid_ratio=1.0,
                disparity_iqr_px=0.0,
                texture_std=10.0,
                confidence=1.0,
                point_camera_mm=np.array([0.0, 0.0, 1000.0]),
                covariance_camera=np.eye(3) * 25.0,
            )
            tracker.update_world_tracks(
                [{"confidence": 0.9, "class_name": "rat", "bbox": (0, 0, 1, 1), "center": (0, 0)}],
                [stereo],
                timestamp=10.0,
                yaw_raw=tracker.current_yaw,
                pitch_raw=tracker.current_pitch,
            )

            rows = log_path.read_text(encoding="utf-8").splitlines()
            record = json.loads(rows[0])

        self.assertEqual(len(rows), 1)
        self.assertEqual(record["schema"], "ratbot.world_tracks.v1")
        self.assertEqual(record["selected_track_id"], 1)
        self.assertEqual(record["valid_3d_measurement_count"], 1)
        self.assertEqual(record["measurements"][0]["camera_point_mm"], [0.0, 0.0, 1000.0])
        self.assertEqual(record["measurements"][0]["base_point_mm"], [1000.0, 0.0, 0.0])
        self.assertIsNotNone(record["predicted_aim"])
        self.assertEqual(record["tracks"][0]["id"], 1)

    def test_ui_recording_captures_world_updates_into_session_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = CameraTracker(
                enable_servos=False,
                no_connect=True,
                enable_camera=False,
                world_tracking=True,
                world_confirm_hits=1,
                world_recordings_dir=temp_dir,
            )
            started = tracker.start_track_recording()
            stereo = StereoPointMeasurement(
                depth_mm=1000.0,
                disparity_px=20.0,
                valid_ratio=1.0,
                disparity_iqr_px=0.0,
                texture_std=10.0,
                confidence=1.0,
                point_camera_mm=np.array([0.0, 0.0, 1000.0]),
                covariance_camera=np.eye(3) * 25.0,
            )
            tracker.update_world_tracks(
                [{"confidence": 0.9, "class_name": "rat", "bbox": (1, 2, 3, 4), "center": (2, 3)}],
                [stereo],
                timestamp=10.0,
                yaw_raw=tracker.current_yaw,
                pitch_raw=tracker.current_pitch,
            )
            stopped = tracker.stop_track_recording()
            replay = tracker.load_track_recording(started["id"])

        self.assertEqual(stopped["frame_count"], 1)
        self.assertEqual(replay["frames"][0]["measurements"][0]["bbox"], [1, 2, 3, 4])
        self.assertEqual(replay["metadata"]["parameters"]["confirm_hits"], 1)

    def test_recording_disk_failure_stops_session_without_aborting_world_update(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = CameraTracker(
                enable_servos=False,
                no_connect=True,
                enable_camera=False,
                world_tracking=True,
                world_confirm_hits=1,
                world_recordings_dir=temp_dir,
            )
            tracker.start_track_recording()
            tracker.track_recordings.append = MagicMock(side_effect=OSError("disk full"))
            stereo = StereoPointMeasurement(
                depth_mm=1000.0,
                disparity_px=20.0,
                valid_ratio=1.0,
                disparity_iqr_px=0.0,
                texture_std=10.0,
                confidence=1.0,
                point_camera_mm=np.array([0.0, 0.0, 1000.0]),
                covariance_camera=np.eye(3) * 25.0,
            )

            tracks = tracker.update_world_tracks(
                [{"confidence": 0.9, "class_name": "rat", "bbox": (1, 2, 3, 4), "center": (2, 3)}],
                [stereo],
                timestamp=10.0,
                yaw_raw=tracker.current_yaw,
                pitch_raw=tracker.current_pitch,
            )

            self.assertEqual(len(tracks), 1)
            self.assertFalse(tracker.get_track_recording_status()["recording"])
            metadata = tracker.list_track_recordings()[0]
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["stop_reason"], "storage_error")

    def test_world_mode_without_depth_does_not_create_or_move_track(self):
        tracker = CameraTracker(
            enable_servos=False,
            no_connect=True,
            enable_camera=False,
            world_tracking=True,
            world_confirm_hits=1,
        )
        initial = (tracker.current_yaw, tracker.current_pitch)

        tracks = tracker.update_world_tracks(
            [{"confidence": 0.9, "class_name": "rat", "bbox": (0, 0, 1, 1), "center": (0, 0)}],
            [None],
            timestamp=10.0,
            yaw_raw=tracker.current_yaw,
            pitch_raw=tracker.current_pitch,
        )
        tracker.tracking_controller.track_once()

        self.assertEqual(tracks, [])
        self.assertEqual((tracker.current_yaw, tracker.current_pitch), initial)


if __name__ == "__main__":
    unittest.main()
