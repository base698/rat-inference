"""Tests for durable track-recording sessions and parameterized replay."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ratbot.tracking.recording import TrackRecordingStore


class TrackRecordingStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = TrackRecordingStore(Path(self.temp.name))

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def frame(timestamp: float, x_mm: float, bbox=(10, 20, 30, 40)):
        return {
            "schema": "ratbot.world_tracks.v1",
            "recorded_at": "2026-07-21T12:00:00-04:00",
            "monotonic_time": timestamp,
            "measurements": [{
                "center": [20, 30],
                "bbox": list(bbox),
                "class": "rat",
                "detection_confidence": 0.9,
                "depth_confidence": 0.8,
                "base_point_mm": [x_mm, 0.0, 0.0],
                "base_covariance": np.eye(3).tolist(),
            }],
            "tracks": [],
            "assignments": [],
        }

    def test_start_append_stop_creates_session_directory_and_catalog_entry(self):
        started = self.store.start({"confirm_hits": 3, "gate_distance_mm": 750.0})
        self.store.append(self.frame(10.0, 1000.0))
        stopped = self.store.stop()

        session_dir = Path(self.temp.name) / started["id"]
        metadata = json.loads((session_dir / "metadata.json").read_text())
        rows = (session_dir / "observations.jsonl").read_text().splitlines()

        self.assertEqual(stopped["frame_count"], 1)
        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(metadata["parameters"]["confirm_hits"], 3)
        self.assertEqual(len(rows), 1)
        self.assertEqual(self.store.list_recordings()[0]["id"], started["id"])

    def test_store_rejects_nested_unknown_and_symlink_recording_ids(self):
        with self.assertRaises(KeyError):
            self.store.load("../metadata")
        with self.assertRaises(KeyError):
            self.store.load("missing")
        outside = Path(self.temp.name).parent / "outside-track-recording"
        outside.mkdir(exist_ok=True)
        link = Path(self.temp.name) / "linked-session"
        link.symlink_to(outside, target_is_directory=True)
        try:
            with self.assertRaises(KeyError):
                self.store.load("linked-session")
        finally:
            link.unlink()
            outside.rmdir()

    def test_reprocess_uses_confirmation_and_gate_parameters(self):
        started = self.store.start({"confirm_hits": 3, "gate_distance_mm": 750.0})
        self.store.append(self.frame(10.0, 1000.0))
        self.store.append(self.frame(10.1, 1010.0))
        self.store.stop()

        tentative = self.store.reprocess(started["id"], {
            "confirm_hits": 3,
            "gate_distance_mm": 100.0,
            "max_misses": 5,
            "delete_after_seconds": 1.5,
            "process_acceleration_std_mm_s2": 300.0,
            "confidence_decay": 0.85,
        })
        confirmed = self.store.reprocess(started["id"], {
            "confirm_hits": 2,
            "gate_distance_mm": 100.0,
            "max_misses": 5,
            "delete_after_seconds": 1.5,
            "process_acceleration_std_mm_s2": 300.0,
            "confidence_decay": 0.85,
        })

        self.assertEqual(tentative["frames"][-1]["tracks"][0]["status"], "tentative")
        self.assertEqual(confirmed["frames"][-1]["tracks"][0]["status"], "confirmed")
        self.assertEqual(confirmed["parameters"]["confirm_hits"], 2)

        gate_recording = self.store.start({})
        self.store.append(self.frame(20.0, 1000.0))
        self.store.append(self.frame(20.1, 1300.0))
        self.store.stop()
        narrow_gate = self.store.reprocess(
            gate_recording["id"],
            {"confirm_hits": 1, "gate_distance_mm": 100.0},
        )
        wide_gate = self.store.reprocess(
            gate_recording["id"],
            {"confirm_hits": 1, "gate_distance_mm": 500.0},
        )
        self.assertEqual(narrow_gate["frames"][-1]["assignments"][0]["track_id"], 2)
        self.assertEqual(wide_gate["frames"][-1]["assignments"][0]["track_id"], 1)
        for parameters in (
            {"max_misses": 101},
            {"delete_after_seconds": 0.0},
            {"gate_distance_mm": 10 ** 400},
        ):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                self.store.reprocess(gate_recording["id"], parameters)

    def test_frame_limit_finalizes_recording_instead_of_writing_forever(self):
        store = TrackRecordingStore(Path(self.temp.name) / "limited", max_frames=1)
        started = store.start({})

        store.append(self.frame(10.0, 1000.0))

        self.assertFalse(store.status()["recording"])
        loaded = store.load(started["id"])
        self.assertEqual(loaded["metadata"]["status"], "complete")
        self.assertEqual(loaded["metadata"]["stop_reason"], "frame_limit")
        self.assertEqual(len(loaded["frames"]), 1)

    def test_byte_limit_finalizes_before_writing_oversized_frame(self):
        store = TrackRecordingStore(Path(self.temp.name) / "byte-limited", max_bytes=10)
        started = store.start({})

        store.append(self.frame(10.0, 1000.0))

        loaded = store.load(started["id"])
        self.assertEqual(loaded["metadata"]["stop_reason"], "byte_limit")
        self.assertEqual(loaded["frames"], [])

    def test_active_recording_cannot_be_replayed_while_it_is_being_appended(self):
        started = self.store.start({})

        with self.assertRaisesRegex(RuntimeError, "stop the active recording"):
            self.store.load(started["id"])

    def test_delete_removes_stopped_session_but_rejects_active_recording(self):
        started = self.store.start({})
        self.store.append(self.frame(10.0, 1000.0))
        self.store.stop()

        deleted = self.store.delete(started["id"])

        self.assertTrue(deleted["success"])
        self.assertEqual(deleted["id"], started["id"])
        self.assertEqual(self.store.list_recordings(), [])
        with self.assertRaises(KeyError):
            self.store.load(started["id"])

        active = self.store.start({})
        with self.assertRaisesRegex(RuntimeError, "stop the active recording"):
            self.store.delete(active["id"])

    def test_restart_recovers_interrupted_recording_metadata_from_jsonl(self):
        started = self.store.start({})
        self.store.append(self.frame(10.0, 1000.0))

        restarted = TrackRecordingStore(self.temp.name)
        metadata = restarted.list_recordings()[0]

        self.assertEqual(metadata["id"], started["id"])
        self.assertEqual(metadata["status"], "interrupted")
        self.assertEqual(metadata["frame_count"], 1)

    def test_aggregate_session_limit_prevents_unbounded_directory_growth(self):
        store = TrackRecordingStore(
            Path(self.temp.name) / "session-limited",
            max_sessions=1,
            min_free_bytes=0,
        )
        store.start({})
        store.stop()

        with self.assertRaisesRegex(RuntimeError, "session limit"):
            store.start({})

    def test_aggregate_byte_quota_reserves_capacity_before_starting(self):
        store = TrackRecordingStore(
            Path(self.temp.name) / "quota-limited",
            max_bytes=1024 * 1024,
            max_total_bytes=2 * 1024 * 1024,
            min_free_bytes=0,
        )
        store.start({})
        store.stop()

        with self.assertRaisesRegex(RuntimeError, "storage quota"):
            store.start({})

    def test_unavailable_storage_does_not_break_store_construction(self):
        with patch.object(Path, "mkdir", side_effect=PermissionError("denied")):
            store = TrackRecordingStore(Path(self.temp.name) / "unavailable")

        self.assertFalse(store.status()["available"])
        with self.assertRaisesRegex(OSError, "storage unavailable"):
            store.start({})

    def test_metadata_failure_on_stop_clears_active_session(self):
        self.store.start({})
        with patch.object(self.store, "_write_metadata", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.store.stop()

        self.assertFalse(self.store.status()["recording"])

    def test_reprocess_rejects_output_size_amplification(self):
        store = TrackRecordingStore(
            Path(self.temp.name) / "output-limited",
            max_replay_output_bytes=10,
        )
        started = store.start({})
        store.append(self.frame(40.0, 1000.0))
        store.stop()

        with self.assertRaisesRegex(ValueError, "output-size limit"):
            store.reprocess(started["id"], {"confirm_hits": 1})

    def test_reprocess_rejects_cpu_work_amplification(self):
        store = TrackRecordingStore(
            Path(self.temp.name) / "work-limited",
            max_replay_work_units=1,
        )
        started = store.start({})
        store.append(self.frame(45.0, 1000.0))
        store.stop()

        with self.assertRaisesRegex(ValueError, "CPU work limit"):
            store.reprocess(started["id"], {"confirm_hits": 1})

    def test_reprocess_rejects_runtime_amplification(self):
        store = TrackRecordingStore(
            Path(self.temp.name) / "runtime-limited",
            max_replay_seconds=1e-9,
        )
        started = store.start({})
        store.append(self.frame(47.0, 1000.0))
        store.stop()

        with self.assertRaisesRegex(ValueError, "runtime limit"):
            store.reprocess(started["id"], {"confirm_hits": 1})

    def test_reprocess_rejects_malformed_measurement_collections(self):
        started = self.store.start({})
        malformed = self.frame(50.0, 1000.0)
        malformed["measurements"] = "not-a-list"
        self.store.append(malformed)
        self.store.stop()

        with self.assertRaisesRegex(ValueError, "measurements must be a list"):
            self.store.reprocess(started["id"], {})

    def test_reprocess_rejects_malformed_measurement_fields(self):
        started = self.store.start({})
        malformed = self.frame(51.0, 1000.0)
        malformed["measurements"][0]["detection_confidence"] = []
        self.store.append(malformed)
        self.store.stop()

        with self.assertRaisesRegex(ValueError, "detection_confidence must be numeric"):
            self.store.reprocess(started["id"], {})

        oversized_started = self.store.start({})
        oversized = self.frame(52.0, 1000.0)
        oversized["measurements"][0]["base_point_mm"][0] = 10 ** 400
        self.store.append(oversized)
        self.store.stop()
        with self.assertRaisesRegex(ValueError, "out-of-range number"):
            self.store.reprocess(oversized_started["id"], {})

    def test_oversized_metadata_is_not_parsed(self):
        started = self.store.start({})
        self.store.stop()
        metadata_path = Path(self.temp.name) / started["id"] / "metadata.json"
        metadata_path.write_text('"' + ('x' * (1024 * 1024)) + '"', encoding="utf-8")

        self.assertEqual(self.store.list_recordings(), [])
        with self.assertRaisesRegex(ValueError, "metadata exceeds"):
            self.store.load(started["id"])

    def test_reprocess_rejects_track_count_amplification(self):
        store = TrackRecordingStore(
            Path(self.temp.name) / "track-limited",
            max_replay_tracks=2,
        )
        started = store.start({})
        for index, x_mm in enumerate((1000.0, 2000.0, 3000.0)):
            store.append(self.frame(30.0 + index * 0.1, x_mm))
        store.stop()

        with self.assertRaisesRegex(ValueError, "track-count limit"):
            store.reprocess(
                started["id"],
                {"confirm_hits": 1, "gate_distance_mm": 1.0, "max_misses": 30},
            )


if __name__ == "__main__":
    unittest.main()
