"""Tests for stable-ID multi-target tracking and explicit selection."""

from __future__ import annotations

import unittest

import numpy as np

from ratbot.tracking.models import Detection3D
from ratbot.tracking.multi_target import MultiTargetTracker, TrackManagerConfig


def detection(x, y=0, z=0, *, confidence=0.9, classification="rat", timestamp=0.0):
    return Detection3D(
        position_base_mm=np.array([x, y, z], dtype=float),
        covariance_base=np.eye(3) * 25.0,
        confidence=confidence,
        classification=classification,
        measurement_time=timestamp,
        bbox=(0, 0, 10, 10),
        center=(5, 5),
    )


class MultiTargetTrackerTests(unittest.TestCase):
    def make_tracker(self, **overrides):
        values = dict(
            gate_distance_mm=600.0,
            confirm_hits=2,
            max_misses=3,
            delete_after_seconds=2.0,
            process_acceleration_std_mm_s2=100.0,
            auto_select=True,
        )
        values.update(overrides)
        return MultiTargetTracker(TrackManagerConfig(**values))

    def test_non_finite_tracker_configuration_is_rejected(self):
        for key in (
            "gate_distance_mm",
            "delete_after_seconds",
            "process_acceleration_std_mm_s2",
            "confidence_decay",
        ):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    self.make_tracker(**{key: float("nan")})

    def test_out_of_order_manager_timestamp_is_rejected(self):
        tracker = self.make_tracker(confirm_hits=1)
        tracker.update([detection(0, timestamp=1.0)], timestamp=1.0)

        with self.assertRaises(ValueError):
            tracker.update([detection(1, timestamp=0.5)], timestamp=0.5)

    def test_track_confirms_after_configured_hits_and_auto_selects(self):
        tracker = self.make_tracker()

        first = tracker.update([detection(1000, timestamp=1.0)], timestamp=1.0)
        second = tracker.update([detection(1010, timestamp=2.0)], timestamp=2.0)

        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].status, "tentative")
        self.assertEqual(second[0].status, "confirmed")
        self.assertEqual(tracker.selected_track_id, second[0].id)

    def test_single_only_auto_select_waits_for_one_confirmed_track(self):
        tracker = self.make_tracker(
            confirm_hits=1,
            auto_select=True,
            auto_select_single_only=True,
        )

        tracker.update(
            [
                detection(100, confidence=0.9),
                detection(2000, confidence=0.8),
            ],
            timestamp=1.0,
        )

        self.assertIsNone(tracker.selected_track_id)

        tracks = tracker.update([detection(100, confidence=0.9)], timestamp=2.0)

        self.assertEqual(len(tracks), 2)
        self.assertEqual(tracker.selected_track_id, tracks[0].id)

    def test_assignments_are_one_to_one_and_new_detection_creates_new_track(self):
        tracker = self.make_tracker(confirm_hits=1, auto_select=False)
        original = tracker.update([detection(0), detection(2000)], timestamp=1.0)
        ids = {track.id for track in original}

        updated = tracker.update(
            [detection(100, timestamp=2.0), detection(2100, timestamp=2.0), detection(4000, timestamp=2.0)],
            timestamp=2.0,
        )

        self.assertEqual(len(updated), 3)
        self.assertTrue(ids.issubset({track.id for track in updated}))
        self.assertEqual(sum(track.hits for track in updated), 5)

    def test_assignment_maximizes_cardinality_before_minimizing_distance(self):
        tracker = self.make_tracker(
            confirm_hits=1,
            gate_distance_mm=5.0,
            auto_select=False,
            process_acceleration_std_mm_s2=0.0,
        )
        original = tracker.update([detection(0), detection(5)], timestamp=0.0)
        original_ids = {track.id for track in original}

        updated = tracker.update(
            [detection(4, timestamp=0.1), detection(8, timestamp=0.1)],
            timestamp=0.1,
        )

        self.assertEqual({track.id for track in updated}, original_ids)
        self.assertEqual(len(updated), 2)
        self.assertTrue(all(track.hits == 2 for track in updated))

    def test_class_mismatch_and_out_of_gate_measurements_do_not_steal_track(self):
        tracker = self.make_tracker(confirm_hits=1, auto_select=False)
        original = tracker.update([detection(0, classification="rat")], timestamp=1.0)[0]

        updated = tracker.update(
            [detection(100, classification="can", timestamp=2.0), detection(5000, classification="rat", timestamp=2.0)],
            timestamp=2.0,
        )

        old = next(track for track in updated if track.id == original.id)
        self.assertEqual(old.misses, 1)
        self.assertEqual(len(updated), 3)

    def test_motion_prediction_preserves_ids_for_two_targets(self):
        tracker = self.make_tracker(confirm_hits=1, gate_distance_mm=900, auto_select=False)
        first = tracker.update([detection(-1000), detection(1000)], timestamp=0.0)
        left_id = min(first, key=lambda track: track.position[0]).id
        right_id = max(first, key=lambda track: track.position[0]).id

        tracker.update([detection(-600, timestamp=1), detection(600, timestamp=1)], timestamp=1.0)
        tracker.update([detection(-150, timestamp=2), detection(150, timestamp=2)], timestamp=2.0)
        final = tracker.update(
            [detection(350, timestamp=3), detection(-350, timestamp=3)],
            timestamp=3.0,
        )

        self.assertGreater(next(t for t in final if t.id == left_id).position[0], 0)
        self.assertLess(next(t for t in final if t.id == right_id).position[0], 0)

    def test_selected_track_survives_short_misses_without_switching(self):
        tracker = self.make_tracker(confirm_hits=1, max_misses=2, auto_select=True)
        tracks = tracker.update(
            [detection(1000, confidence=0.95), detection(2000, confidence=0.8)],
            timestamp=1.0,
        )
        selected = tracker.selected_track_id
        other = next(track.id for track in tracks if track.id != selected)

        tracker.update([detection(2010, confidence=0.99, timestamp=2.0)], timestamp=2.0)

        self.assertEqual(tracker.selected_track_id, selected)
        self.assertNotEqual(tracker.selected_track_id, other)
        self.assertIsNotNone(tracker.get_selected_track(timestamp=2.1))

    def test_reacquired_confirmed_track_returns_to_confirmed_immediately(self):
        tracker = self.make_tracker(confirm_hits=2, max_misses=3)
        tracker.update([detection(0)], timestamp=0.0)
        confirmed = tracker.update([detection(10, timestamp=0.1)], timestamp=0.1)[0]
        self.assertEqual(confirmed.status, "confirmed")

        lost = tracker.update([], timestamp=0.2)[0]
        self.assertEqual(lost.status, "lost")
        reacquired = tracker.update(
            [detection(20, timestamp=0.3)],
            timestamp=0.3,
        )[0]

        self.assertEqual(reacquired.id, confirmed.id)
        self.assertEqual(reacquired.status, "confirmed")
        self.assertEqual(reacquired.misses, 0)

    def test_track_deletion_is_time_gated_not_inference_tick_gated(self):
        tracker = self.make_tracker(
            confirm_hits=1,
            max_misses=3,
            delete_after_seconds=1.5,
            auto_select=True,
        )
        original = tracker.update([detection(0, timestamp=0.0)], timestamp=0.0)[0]

        for index in range(1, 10):
            tracks = tracker.update([], timestamp=index * 0.05)

        self.assertEqual([track.id for track in tracks], [original.id])
        self.assertEqual(tracks[0].status, "lost")
        self.assertGreater(tracks[0].misses, tracker.config.max_misses)

        expired = tracker.update([], timestamp=1.6)
        self.assertEqual(expired, [])
        self.assertIsNone(tracker.selected_track_id)

    def test_dormant_track_reidentifies_with_old_id_inside_window(self):
        tracker = self.make_tracker(
            confirm_hits=1,
            delete_after_seconds=0.25,
            reidentify_after_seconds=8.0,
            auto_select=True,
        )
        original = tracker.update([detection(1000, timestamp=0.0)], timestamp=0.0)[0]
        self.assertEqual(tracker.selected_track_id, original.id)

        expired = tracker.update([], timestamp=0.3)
        self.assertEqual(expired, [])
        self.assertIsNone(tracker.selected_track_id)

        reacquired = tracker.update([detection(1025, timestamp=1.0)], timestamp=1.0)

        self.assertEqual([track.id for track in reacquired], [original.id])
        self.assertEqual(reacquired[0].status, "confirmed")
        self.assertEqual(reacquired[0].misses, 0)
        self.assertEqual(tracker.selected_track_id, original.id)
        self.assertTrue(tracker.last_assignments[0]["reidentified"])

    def test_dormant_track_does_not_reidentify_after_window_expires(self):
        tracker = self.make_tracker(
            confirm_hits=1,
            delete_after_seconds=0.25,
            reidentify_after_seconds=0.5,
            auto_select=False,
        )
        original = tracker.update([detection(1000, timestamp=0.0)], timestamp=0.0)[0]
        tracker.update([], timestamp=0.3)

        reacquired = tracker.update([detection(1025, timestamp=1.0)], timestamp=1.0)

        self.assertEqual(len(reacquired), 1)
        self.assertNotEqual(reacquired[0].id, original.id)
        self.assertNotIn("reidentified", tracker.last_assignments[0])

    def test_explicit_clear_selection_suppresses_future_auto_selection(self):
        tracker = self.make_tracker(confirm_hits=1, auto_select=True)
        first = tracker.update([detection(100)], timestamp=0.0)[0]
        self.assertEqual(tracker.selected_track_id, first.id)

        tracker.clear_selection()
        tracker.update([detection(105, timestamp=0.1)], timestamp=0.1)

        self.assertIsNone(tracker.selected_track_id)
        self.assertTrue(tracker.select_target(first.id))
        self.assertEqual(tracker.selected_track_id, first.id)

    def test_selected_track_expiry_does_not_auto_switch(self):
        tracker = self.make_tracker(confirm_hits=1, max_misses=0, auto_select=True)
        tracks = tracker.update(
            [detection(100, confidence=0.99), detection(2000, confidence=0.8)],
            timestamp=0.0,
        )
        selected = tracker.selected_track_id
        survivor = next(track for track in tracks if track.id != selected)

        tracker.update(
            [detection(2005, confidence=0.95, timestamp=0.1)],
            timestamp=0.1,
        )

        self.assertIsNone(tracker.selected_track_id)
        self.assertIsNotNone(next(track for track in tracker.get_tracks() if track.id == survivor.id))

    def test_manual_selection_clear_and_deletion_are_explicit(self):
        tracker = self.make_tracker(confirm_hits=1, max_misses=0, auto_select=False)
        tracks = tracker.update([detection(100), detection(2000)], timestamp=1.0)
        chosen = tracks[1].id

        self.assertTrue(tracker.select_target(chosen))
        self.assertEqual(tracker.selected_track_id, chosen)
        tracker.clear_selection()
        self.assertIsNone(tracker.selected_track_id)
        self.assertFalse(tracker.select_target(9999))

        tracker.select_target(chosen)
        tracker.update([], timestamp=2.0)
        self.assertIsNone(tracker.selected_track_id)

    def test_predicted_selected_snapshot_does_not_mutate_track(self):
        tracker = self.make_tracker(confirm_hits=1)
        track = tracker.update([detection(0, timestamp=1.0)], timestamp=1.0)[0]
        managed = tracker._tracks[track.id]
        managed.filter.state[3] = 100.0

        predicted = tracker.get_selected_track(timestamp=2.0, prediction_horizon=0.5)
        current = tracker.get_tracks()[0]

        self.assertAlmostEqual(predicted.position[0], 150.0)
        self.assertAlmostEqual(current.position[0], 0.0)


if __name__ == "__main__":
    unittest.main()
