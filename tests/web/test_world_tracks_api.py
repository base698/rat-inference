"""Tests for world-track observability and selection API routes."""

from __future__ import annotations

import asyncio
import tempfile
import unittest

from ratbot.web.control_api import ControlApiConfig, TrackerControlApi


class FakeTracker:
    connected = True
    current_yaw = 2200
    current_pitch = 250
    camera_active = False
    detection_count = 0
    trigger_servo_enabled = False
    world_tracking = True
    world_api_selection_enabled = True

    def __init__(self):
        self.selected = None

    def get_world_tracks(self):
        return [{"id": 7, "selected": self.selected == 7}]

    def select_world_target(self, target_id):
        if target_id != 7:
            return False
        self.selected = target_id
        return True

    def clear_world_selection(self):
        self.selected = None
        return True


class WorldTrackApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.api = TrackerControlApi(
            ControlApiConfig(
                yaw_min=1600,
                yaw_max=3100,
                yaw_center=2200,
                pitch_min=1,
                pitch_max=500,
                pitch_center=250,
                static_dir=self.temp.name,
                detections_dir=self.temp.name,
            ),
            lambda value: value,
            lambda value: value,
        )
        self.tracker = FakeTracker()
        self.api.set_tracker(self.tracker)

    def tearDown(self):
        self.temp.cleanup()

    def endpoint(self, path, method):
        for route in self.api.app.routes:
            if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
                return route.endpoint
        self.fail(f"missing {method} {path}")

    def test_get_tracks_exposes_mode_selection_and_tracks(self):
        response = asyncio.run(self.endpoint("/tracks", "GET")())

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'"world_tracking":true', response.body)
        self.assertIn(b'"id":7', response.body)

    def test_selection_mutations_are_disabled_by_default_gate(self):
        select = self.endpoint("/tracks/select", "POST")
        clear = self.endpoint("/tracks/clear-selection", "POST")
        self.tracker.world_api_selection_enabled = False

        selected = asyncio.run(select({"track_id": 7}))
        cleared = asyncio.run(clear())

        self.assertEqual(selected.status_code, 403)
        self.assertEqual(cleared.status_code, 403)
        self.assertIsNone(self.tracker.selected)

    def test_select_and_clear_selection_are_explicit(self):
        select = self.endpoint("/tracks/select", "POST")
        clear = self.endpoint("/tracks/clear-selection", "POST")

        selected = asyncio.run(select({"track_id": 7}))
        self.assertEqual(selected.status_code, 200)
        self.assertEqual(self.tracker.selected, 7)

        missing = asyncio.run(select({"track_id": 99}))
        self.assertEqual(missing.status_code, 404)
        invalid = asyncio.run(select({"track_id": "not-an-id"}))
        self.assertEqual(invalid.status_code, 400)
        boolean = asyncio.run(select({"track_id": True}))
        self.assertEqual(boolean.status_code, 400)
        fractional = asyncio.run(select({"track_id": 7.9}))
        self.assertEqual(fractional.status_code, 400)

        cleared = asyncio.run(clear())
        self.assertEqual(cleared.status_code, 200)
        self.assertIsNone(self.tracker.selected)


if __name__ == "__main__":
    unittest.main()
