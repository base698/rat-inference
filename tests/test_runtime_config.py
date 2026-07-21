"""Tests for typed RT-200 CLI/runtime composition."""

from __future__ import annotations

import unittest

from ratbot.runtime_config import build_argument_parser, parse_runtime_config


RAW_CONFIG = {
    "camera": {
        "fov_horizontal": 60.0,
        "fov_vertical": 45.0,
        "width": 960,
        "height": 720,
    },
    "detection": {
        "model_path": "configured.engine",
        "confidence_threshold": 0.7,
        "imgsz": 640,
        "target_classes": ["item", "class0"],
    },
    "tracking": {
        "auto_tracking": {
            "smoothing_alpha": 0.45,
            "max_yaw_step": 45,
            "max_pitch_step": 35,
            "control_fps": 20,
            "belief_update_alpha": 0.75,
            "belief_pitch_update_alpha": 0.35,
            "belief_miss_decay": 0.9,
            "belief_min_confidence": 0.15,
            "belief_max_age": 1.0,
            "belief_deadband_raw": 4,
            "belief_min_step_raw": 3,
            "pitch_tracking_scale": 2.2,
            "belief_reseed_distance_raw": 160,
            "belief_reseed_confirmations": 2,
            "belief_reseed_match_distance_raw": 120,
            "belief_reseed_max_interval": 0.8,
            "belief_velocity_alpha": 0.45,
            "belief_pitch_velocity_alpha": 0.25,
            "belief_velocity_decay": 0.96,
            "belief_max_velocity_raw_per_s": 600,
            "belief_max_pitch_velocity_raw_per_s": 180,
            "belief_max_prediction_age": 0.45,
        }
    },
}

EXPECTED_HELP = {
    "help": "show this help message and exit",
    "port": "Serial port for servo connection",
    "disable_servos": "Disable tracking servo control (simulation mode)",
    "no_connect": "Skip servo connection attempt (web interface only)",
    "video_only": "Convenience mode: CSI camera web stream, no servos, no detection",
    "enable_camera": "Enable camera",
    "camera_id": "Camera device ID (default: 0)",
    "use_csi": "Use CSI camera with GStreamer pipeline (Jetson)",
    "invert_camera": "Invert camera 180 degrees for upside-down mounting",
    "model": "Path to YOLO model",
    "disable_detection": "Disable YOLO detection while keeping the camera stream enabled",
    "confidence": "Detection confidence threshold",
    "target_class": "YOLO class name to track. Repeat or comma-separate values. Default: all model classes",
    "imgsz": "Inference image size in pixels (default: 640)",
    "inference_fps": "Target inference loop FPS (default: 20)",
    "tracking_smoothing": "Detection center smoothing alpha for auto tracking: 0 disables, 1 follows raw detections (default: 0.45)",
    "max_yaw_step": "Maximum yaw raw-unit move per inference update for smoother tracking (default: 45)",
    "max_pitch_step": "Maximum pitch raw-unit move per control update for smoother tracking (default: 45)",
    "max_yaw_speed_raw_per_s": "Optional yaw raw-unit speed cap for smoother control interpolation",
    "max_pitch_speed_raw_per_s": "Optional pitch raw-unit speed cap for smoother control interpolation",
    "tracking_control_fps": "Servo control loop FPS for angular target belief tracking (default: 20)",
    "belief_update_alpha": "Angular belief update alpha from new detections: 0 ignores, 1 jumps to observation (default: 0.45)",
    "belief_pitch_update_alpha": "Optional pitch-only angular belief update alpha (default: same as --belief-update-alpha)",
    "belief_miss_decay": "Belief confidence decay per inference tick without detection (default: 0.94)",
    "belief_min_confidence": "Minimum angular belief confidence required to keep moving (default: 0.15)",
    "belief_max_age": "Maximum age in seconds for angular belief movement (default: 1.5)",
    "belief_deadband_raw": "Raw servo-unit error treated as centered by angular belief control (default: 4)",
    "belief_min_step_raw": "Minimum raw servo-unit correction while outside angular belief deadband (default: 3)",
    "pitch_tracking_scale": "Multiplier for vertical image error to pitch raw observation; higher uses more down/up pitch travel (default: 1.0)",
    "belief_reseed_distance_raw": "Raw-unit observation jump that starts a fresh angular belief instead of smoothing (default: 160)",
    "belief_velocity_alpha": "Angular belief velocity smoothing alpha from detections (default: 0.45)",
    "belief_pitch_velocity_alpha": "Optional pitch-only angular belief velocity alpha (default: same as --belief-velocity-alpha)",
    "belief_velocity_decay": "Angular belief velocity decay per inference tick without detection (default: 0.96)",
    "belief_max_velocity_raw_per_s": "Maximum angular belief target speed in raw servo units/sec (default: 600)",
    "belief_max_pitch_velocity_raw_per_s": "Optional pitch-only angular belief target speed cap (default: same as --belief-max-velocity-raw-per-s)",
    "belief_max_prediction_age": "Maximum seconds to extrapolate target belief after the latest detection (default: 0.45)",
    "belief_reseed_confirmations": "Matching far-jump detections required before reseeding angular belief (default: 2)",
    "belief_reseed_match_distance_raw": "Raw-unit distance for matching pending reseed detections (default: 120)",
    "belief_reseed_max_interval": "Maximum seconds between matching pending reseed detections (default: 0.8)",
    "belief_reseed_min_confidence": "Minimum confidence allowed to start/confirm a far-jump angular belief reseed (default: 0.55)",
    "calibration": "Path to camera calibration file (.npz, default: camera_calibration.npz)",
    "stereo": "Enable stereo mode for depth estimation (requires stereo calibration)",
    "baseline_override": "Override stereo baseline in mm (use if calibration baseline is incorrect)",
    "world_tracking": "Enable opt-in multi-target tracking in the fixed turret-base 3D frame",
    "enable_trigger": "Enable GPIO trigger servo",
    "api_host": "Host for FastAPI server",
    "api_port": "Port for FastAPI server",
}


class RuntimeConfigTests(unittest.TestCase):
    def parse(self, argv=(), *, exists=lambda path: False):
        return parse_runtime_config(
            argv=list(argv),
            raw_config=RAW_CONFIG,
            inference_fps_default=20,
            path_exists=exists,
        )

    def test_configured_defaults_are_typed_for_tracker_composition(self):
        runtime = self.parse()
        tracker = runtime.tracker

        self.assertEqual(tracker.port, "/dev/ttyACM0")
        self.assertTrue(tracker.enable_servos)
        self.assertFalse(tracker.enable_camera)
        self.assertEqual(tracker.model_path, None)
        self.assertEqual(tracker.requested_model, "configured.engine")
        self.assertEqual(tracker.confidence_threshold, 0.7)
        self.assertEqual(tracker.camera_width, 960)
        self.assertEqual(tracker.camera_height, 720)
        self.assertEqual(tracker.camera_fov_horizontal, 60.0)
        self.assertEqual(tracker.camera_fov_vertical, 45.0)
        self.assertEqual(tracker.target_classes, ("item", "class0"))
        self.assertEqual(tracker.max_pitch_step, 35)
        self.assertEqual(tracker.belief_pitch_update_alpha, 0.35)
        self.assertEqual(tracker.belief_max_pitch_velocity_raw_per_s, 180)
        self.assertEqual(runtime.api_host, "0.0.0.0")
        self.assertEqual(runtime.api_port, 8000)

    def test_all_cli_help_descriptions_preserve_the_previous_contract(self):
        parser = build_argument_parser(RAW_CONFIG, inference_fps_default=20)
        actual = {
            action.dest: action.help
            for action in parser._actions
            if action.option_strings
        }

        self.assertEqual(actual, EXPECTED_HELP)

    def test_video_only_applies_existing_camera_and_simulation_shortcut(self):
        runtime = self.parse(["--video-only"])
        tracker = runtime.tracker

        self.assertTrue(tracker.enable_camera)
        self.assertTrue(tracker.use_csi)
        self.assertFalse(tracker.enable_servos)
        self.assertTrue(tracker.no_connect)
        self.assertIsNone(tracker.model_path)

    def test_explicit_detection_and_target_overrides_are_preserved(self):
        runtime = self.parse([
            "--enable-camera",
            "--model", "override.pt",
            "--target-class", "rat,mouse",
            "--target-class", "item",
            "--confidence", "0.8",
            "--api-port", "9000",
        ])
        tracker = runtime.tracker

        self.assertEqual(tracker.model_path, "override.pt")
        self.assertEqual(tracker.target_classes, ("rat,mouse", "item"))
        self.assertEqual(tracker.confidence_threshold, 0.8)
        self.assertEqual(runtime.api_port, 9000)

    def test_stereo_uses_existing_default_calibration_when_available(self):
        expected = "tools/vision/calibration/output_recal/stereo_calibration.npz"
        runtime = self.parse(
            ["--stereo"],
            exists=lambda path: path == expected,
        )

        self.assertTrue(runtime.tracker.stereo_mode)
        self.assertEqual(runtime.tracker.calibration_file, expected)

    def test_world_tracking_is_opt_in_and_passes_calibration_defaults(self):
        disabled = self.parse().tracker
        enabled = self.parse(["--world-tracking"]).tracker

        self.assertFalse(disabled.world_tracking)
        self.assertTrue(enabled.world_tracking)
        self.assertTrue(enabled.stereo_mode)
        self.assertEqual(enabled.world_gate_distance_mm, 750.0)
        self.assertEqual(enabled.world_camera_translation_mm, (0.0, 0.0, 0.0))
        self.assertEqual(enabled.world_camera_mount_rpy_degrees, (0.0, 0.0, 0.0))
        self.assertEqual(enabled.world_pitch_sign, -1.0)
        self.assertFalse(enabled.world_api_recording_enabled)
        self.assertFalse(enabled.world_api_selection_enabled)
        self.assertFalse(enabled.world_actuation_enabled)
        self.assertFalse(enabled.world_calibration_validated)

    def test_world_safety_booleans_reject_string_values_fail_closed(self):
        for key in (
            "enabled",
            "actuation_enabled",
            "calibration_validated",
            "allow_remote_recording",
            "allow_remote_selection",
        ):
            for invalid in ("false", "no", "0", 0, None, []):
                with self.subTest(key=key, invalid=invalid):
                    raw = {
                        "tracking": {
                            "world_frame": {key: invalid},
                        }
                    }
                    with self.assertRaisesRegex(ValueError, key):
                        parse_runtime_config(
                            argv=[],
                            raw_config=raw,
                            inference_fps_default=20,
                            path_exists=lambda path: False,
                        )

    def test_world_safety_booleans_accept_actual_yaml_booleans(self):
        raw = {
            "tracking": {
                "world_frame": {
                    "enabled": True,
                    "actuation_enabled": True,
                    "calibration_validated": True,
                    "allow_remote_selection": True,
                }
            }
        }

        tracker = parse_runtime_config(
            argv=[],
            raw_config=raw,
            inference_fps_default=20,
            path_exists=lambda path: False,
        ).tracker

        self.assertTrue(tracker.world_tracking)
        self.assertTrue(tracker.world_actuation_enabled)
        self.assertTrue(tracker.world_calibration_validated)
        self.assertTrue(tracker.world_api_selection_enabled)

    def test_tracker_kwargs_exclude_display_only_runtime_fields(self):
        runtime = self.parse(["--enable-camera", "--disable-detection"])
        kwargs = runtime.tracker.to_tracker_kwargs()

        self.assertNotIn("requested_model", kwargs)
        self.assertNotIn("disable_detection", kwargs)
        self.assertTrue(kwargs["enable_camera"])
        self.assertIsNone(kwargs["model_path"])


if __name__ == "__main__":
    unittest.main()
