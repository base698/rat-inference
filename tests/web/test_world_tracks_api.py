"""Tests for world-track observability, replay, recording, and selection routes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
import time
import unittest
from starlette.requests import Request

from ratbot.web.control_api import (
    ControlApiConfig,
    RequestBodyLimitMiddleware,
    TrackerControlApi,
)


class FakeTracker:
    connected = True
    current_yaw = 2200
    current_pitch = 250
    camera_active = False
    detection_count = 0
    trigger_servo_enabled = False
    world_tracking = True
    world_api_selection_enabled = True
    world_api_recording_enabled = True

    def __init__(self):
        self.selected = None
        self.recording = False
        self.recordings = [{"id": "20260721T120000-test", "frame_count": 4}]

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

    def get_track_recording_status(self):
        return {"recording": self.recording, "active": None}

    def start_track_recording(self):
        self.recording = True
        return {"id": "20260721T120001-live", "status": "recording"}

    def stop_track_recording(self):
        self.recording = False
        return {"id": "20260721T120001-live", "status": "complete", "frame_count": 2}

    def list_track_recordings(self):
        return self.recordings

    def load_track_recording(self, recording_id):
        if recording_id != "20260721T120000-test":
            raise KeyError(recording_id)
        return {"metadata": self.recordings[0], "frames": [{"monotonic_time": 1.0}]}

    def reprocess_track_recording(self, recording_id, parameters):
        if recording_id != "20260721T120000-test":
            raise KeyError(recording_id)
        return {"parameters": parameters, "frames": []}


class WorldTrackApiTests(unittest.TestCase):
    def test_request_body_limit_rejects_chunked_body_before_decode(self):
        sent = []
        chunks = iter((
            {"type": "http.request", "body": b"x" * 40_000, "more_body": True},
            {"type": "http.request", "body": b"x" * 40_000, "more_body": False},
        ))

        async def receive():
            return next(chunks)

        async def send(message):
            sent.append(message)

        async def inner_app(scope, receive, send):
            while True:
                message = await receive()
                if not message.get("more_body", False):
                    break

        scope = {
            "type": "http", "method": "POST", "path": "/api/track-recordings/x/reprocess",
            "headers": [], "query_string": b"", "http_version": "1.1",
            "scheme": "http", "server": ("test", 80), "client": ("test", 1),
        }
        asyncio.run(RequestBodyLimitMiddleware(inner_app)(scope, receive, send))

        response_start = next(message for message in sent if message["type"] == "http.response.start")
        self.assertEqual(response_start["status"], 413)

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
        response = asyncio.run(self.endpoint("/api/tracks/live", "GET")())

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'"world_tracking":true', response.body)
        self.assertIn(b'"id":7', response.body)

    def test_tracks_page_is_served_separately_from_live_json(self):
        (Path(self.temp.name) / "tracks.html").write_text("<h1>Track Replay</h1>")

        request = Request({"type": "http", "headers": [(b"accept", b"text/html")]})
        response = asyncio.run(self.endpoint("/tracks", "GET")(request))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(Path(response.path).name, "tracks.html")

    def test_tracks_preserves_json_compatibility_for_api_clients(self):
        request = Request({"type": "http", "headers": [(b"accept", b"application/json")]})

        response = asyncio.run(self.endpoint("/tracks", "GET")(request))

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'"world_tracking":true', response.body)

    def test_recording_start_stop_status_and_catalog_routes(self):
        status = asyncio.run(self.endpoint("/api/track-recordings/status", "GET")())
        started = asyncio.run(self.endpoint("/api/track-recordings/start", "POST")())
        stopped = asyncio.run(self.endpoint("/api/track-recordings/stop", "POST")())
        catalog = asyncio.run(self.endpoint("/api/track-recordings", "GET")())

        self.assertFalse(json.loads(status.body)["recording"])
        self.assertEqual(json.loads(started.body)["status"], "recording")
        self.assertEqual(json.loads(stopped.body)["frame_count"], 2)
        self.assertEqual(json.loads(catalog.body)["recordings"][0]["frame_count"], 4)

    def test_recording_mutations_are_disabled_by_default_gate(self):
        self.tracker.world_api_recording_enabled = False

        started = asyncio.run(self.endpoint("/api/track-recordings/start", "POST")())
        stopped = asyncio.run(self.endpoint("/api/track-recordings/stop", "POST")())
        catalog = asyncio.run(self.endpoint("/api/track-recordings", "GET")())
        load = asyncio.run(
            self.endpoint("/api/track-recordings/{recording_id}", "GET")(
                recording_id="20260721T120000-test"
            )
        )
        reprocess = asyncio.run(
            self.endpoint("/api/track-recordings/{recording_id}/reprocess", "POST")(
                recording_id="20260721T120000-test", request={}
            )
        )

        self.assertEqual(started.status_code, 403)
        self.assertEqual(stopped.status_code, 403)
        self.assertEqual(catalog.status_code, 403)
        self.assertEqual(load.status_code, 403)
        self.assertEqual(reprocess.status_code, 403)

    def test_recording_storage_error_returns_insufficient_storage(self):
        def fail_start():
            raise OSError("disk full")

        self.tracker.start_track_recording = fail_start
        response = asyncio.run(self.endpoint("/api/track-recordings/start", "POST")())

        self.assertEqual(response.status_code, 507)

    def test_recording_cannot_start_when_world_tracking_is_disabled(self):
        self.tracker.world_tracking = False

        started = asyncio.run(self.endpoint("/api/track-recordings/start", "POST")())

        self.assertEqual(started.status_code, 409)

    def test_recording_replay_and_parameterized_reprocess_routes(self):
        replay = self.endpoint("/api/track-recordings/{recording_id}", "GET")
        reprocess = self.endpoint("/api/track-recordings/{recording_id}/reprocess", "POST")

        loaded = asyncio.run(replay("20260721T120000-test"))
        tuned = asyncio.run(reprocess("20260721T120000-test", {"confirm_hits": 2}))
        missing = asyncio.run(replay("missing"))

        self.assertEqual(json.loads(loaded.body)["frames"][0]["monotonic_time"], 1.0)
        self.assertEqual(json.loads(tuned.body)["parameters"]["confirm_hits"], 2)
        self.assertEqual(missing.status_code, 404)

    def test_replay_api_rejects_concurrent_work_instead_of_queueing(self):
        original = self.tracker.load_track_recording

        def slow_load(recording_id):
            time.sleep(0.08)
            return original(recording_id)

        self.tracker.load_track_recording = slow_load
        load = self.endpoint("/api/track-recordings/{recording_id}", "GET")

        async def exercise():
            first_task = asyncio.create_task(load(recording_id="20260721T120000-test"))
            await asyncio.sleep(0.01)
            busy = await load(recording_id="beta")
            first = await first_task
            return first, busy

        first, busy = asyncio.run(exercise())
        self.assertEqual(first.status_code, 200)
        self.assertEqual(busy.status_code, 429)

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
