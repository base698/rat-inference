"""Tests for adapting selected 3D tracks to the bounded servo controller."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from ratbot.tracking.aiming import WorldTrackBeliefAdapter
from ratbot.tracking.geometry import ServoKinematicsConfig, TurretFrameTransformer
from ratbot.tracking.models import Detection3D
from ratbot.tracking.multi_target import MultiTargetTracker, TrackManagerConfig


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

    def add_track(self, position, timestamp=10.0, confidence=0.9):
        self.manager.update(
            [Detection3D(
                position_base_mm=np.asarray(position, dtype=float),
                covariance_base=np.eye(3) * 25,
                confidence=confidence,
                classification="rat",
                measurement_time=timestamp,
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


if __name__ == "__main__":
    unittest.main()
