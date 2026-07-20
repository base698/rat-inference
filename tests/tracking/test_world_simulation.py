"""Deterministic software acceptance tests for turret motion and stable world tracks."""

from __future__ import annotations

import unittest

import numpy as np

from ratbot.tracking.geometry import ServoKinematicsConfig, TurretFrameTransformer
from ratbot.tracking.models import Detection3D
from ratbot.tracking.multi_target import MultiTargetTracker, TrackManagerConfig


class WorldTrackingSimulationTests(unittest.TestCase):
    def test_two_stationary_targets_keep_ids_and_positions_during_pan_tilt_sweep(self):
        transformer = TurretFrameTransformer(
            ServoKinematicsConfig(
                yaw_center_raw=2200,
                pitch_center_raw=250,
                yaw_raw_per_degree=10,
                pitch_raw_per_degree=10,
                yaw_sign=1,
                pitch_sign=-1,
                camera_translation_mm=(85.0, 0.0, 40.0),
            )
        )
        manager = MultiTargetTracker(
            TrackManagerConfig(
                gate_distance_mm=400,
                confirm_hits=2,
                max_misses=2,
                process_acceleration_std_mm_s2=10,
            )
        )
        fixed_targets = (
            np.array([2200.0, -450.0, -80.0]),
            np.array([1900.0, 550.0, 120.0]),
        )
        rng = np.random.default_rng(42)
        initial_ids = None
        tracks = []

        for index in range(30):
            timestamp = index * 0.05
            yaw_raw = 1800 + index * (800 / 29)
            pitch_raw = 170 + index * (160 / 29)
            detections = []
            for target_index, fixed_target in enumerate(fixed_targets):
                camera_point = transformer.base_to_camera(
                    fixed_target,
                    yaw_raw,
                    pitch_raw,
                )
                noisy_camera_point = camera_point + rng.normal(0.0, 3.0, 3)
                detections.append(
                    Detection3D(
                        position_base_mm=transformer.camera_to_base(
                            noisy_camera_point,
                            yaw_raw,
                            pitch_raw,
                        ),
                        covariance_base=transformer.camera_covariance_to_base(
                            np.eye(3) * 9.0,
                            yaw_raw,
                            pitch_raw,
                        ),
                        confidence=0.95,
                        classification="rat",
                        measurement_time=timestamp,
                        bbox=(target_index * 20, 0, target_index * 20 + 10, 10),
                    )
                )
            tracks = manager.update(detections, timestamp)
            if index == 1:
                initial_ids = [track.id for track in tracks]

        self.assertEqual([track.id for track in tracks], initial_ids)
        self.assertTrue(all(track.status == "confirmed" for track in tracks))
        for track, expected in zip(tracks, fixed_targets):
            np.testing.assert_allclose(track.position, expected, atol=12.0)
            self.assertLess(float(np.linalg.norm(track.velocity)), 50.0)


if __name__ == "__main__":
    unittest.main()
