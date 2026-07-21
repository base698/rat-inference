"""Prometheus text metrics for the Ratbot control API."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import threading
import time
from typing import Mapping, Optional


PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def _label(value: object) -> str:
    text = str(value)
    return (
        text
        .replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace('"', '\\"')
    )


def _labels(**values: object) -> str:
    if not values:
        return ""
    return "{" + ",".join(f'{key}="{_label(value)}"' for key, value in values.items()) + "}"


def _float(value: object, *, default: Optional[float] = None) -> Optional[float]:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(number):
        return default
    return number


class RatbotMetrics:
    """Small in-process Prometheus collector with no external dependency."""

    def __init__(self):
        self._lock = threading.RLock()
        self._request_counts: Counter[tuple[str, str, str]] = Counter()
        self._request_duration_sums: defaultdict[tuple[str, str, str], float] = defaultdict(float)
        self._requests_in_progress = 0
        self._started_at = time.time()

    def begin_http_request(self) -> None:
        with self._lock:
            self._requests_in_progress += 1

    def finish_http_request(
        self,
        *,
        method: str,
        path: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        labels = (method.upper(), path, str(int(status_code)))
        duration = max(0.0, float(duration_seconds))
        with self._lock:
            self._requests_in_progress = max(0, self._requests_in_progress - 1)
            self._request_counts[labels] += 1
            self._request_duration_sums[labels] += duration

    def render(self, tracker) -> str:
        lines: list[str] = []
        self._append_process_metrics(lines)
        self._append_http_metrics(lines)
        self._append_tracker_metrics(lines, tracker)
        return "\n".join(lines) + "\n"

    @staticmethod
    def _help(lines: list[str], name: str, metric_type: str, help_text: str) -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {metric_type}")

    @staticmethod
    def _sample(lines: list[str], name: str, value: object, **labels: object) -> None:
        number = _float(value)
        if number is None:
            return
        lines.append(f"{name}{_labels(**labels)} {number:.12g}")

    def _append_process_metrics(self, lines: list[str]) -> None:
        self._help(lines, "ratbot_up", "gauge", "Ratbot API process is serving metrics.")
        self._sample(lines, "ratbot_up", 1)
        self._help(lines, "ratbot_start_time_seconds", "gauge", "Ratbot API process start time.")
        self._sample(lines, "ratbot_start_time_seconds", self._started_at)

    def _append_http_metrics(self, lines: list[str]) -> None:
        with self._lock:
            counts = dict(self._request_counts)
            sums = dict(self._request_duration_sums)
            in_progress = self._requests_in_progress

        self._help(
            lines,
            "ratbot_http_requests_total",
            "counter",
            "Total HTTP requests by method, route template, and status.",
        )
        for (method, path, status), count in sorted(counts.items()):
            self._sample(
                lines,
                "ratbot_http_requests_total",
                count,
                method=method,
                path=path,
                status=status,
            )

        self._help(
            lines,
            "ratbot_http_request_duration_seconds",
            "summary",
            "HTTP request duration by method, route template, and status.",
        )
        for (method, path, status), count in sorted(counts.items()):
            labels = {"method": method, "path": path, "status": status}
            self._sample(
                lines,
                "ratbot_http_request_duration_seconds_count",
                count,
                **labels,
            )
            self._sample(
                lines,
                "ratbot_http_request_duration_seconds_sum",
                sums.get((method, path, status), 0.0),
                **labels,
            )

        self._help(
            lines,
            "ratbot_http_requests_in_progress",
            "gauge",
            "HTTP requests currently being handled.",
        )
        self._sample(lines, "ratbot_http_requests_in_progress", in_progress)

    def _append_tracker_metrics(self, lines: list[str], tracker) -> None:
        self._help(lines, "ratbot_connected", "gauge", "Tracker servo connection state.")
        self._sample(lines, "ratbot_connected", bool(getattr(tracker, "connected", False)))
        self._help(lines, "ratbot_camera_active", "gauge", "Camera capture state.")
        self._sample(lines, "ratbot_camera_active", bool(getattr(tracker, "camera_active", False)))
        self._help(lines, "ratbot_yaw_position_raw", "gauge", "Current yaw servo raw position.")
        self._sample(lines, "ratbot_yaw_position_raw", getattr(tracker, "current_yaw", None))
        self._help(lines, "ratbot_pitch_position_raw", "gauge", "Current pitch servo raw position.")
        self._sample(lines, "ratbot_pitch_position_raw", getattr(tracker, "current_pitch", None))

        detection_data = self._detection_data(tracker)
        self._help(lines, "ratbot_detection_active", "gauge", "Whether the latest inference pass has a detection.")
        self._sample(lines, "ratbot_detection_active", detection_data.get("detection", False))
        self._help(lines, "ratbot_detection_confidence", "gauge", "Latest selected detection or track confidence.")
        self._sample(lines, "ratbot_detection_confidence", detection_data.get("confidence", 0.0))
        self._help(lines, "ratbot_detection_count_total", "counter", "Detections observed since process start.")
        self._sample(lines, "ratbot_detection_count_total", getattr(tracker, "detection_count", 0))

        depth_mm = _float(
            detection_data.get(
                "depth_mm",
                getattr(tracker, "latest_depth", None),
            )
        )
        self._help(lines, "ratbot_latest_depth_meters", "gauge", "Latest selected target depth in meters.")
        if depth_mm is not None:
            self._sample(lines, "ratbot_latest_depth_meters", depth_mm / 1000.0)

        self._help(lines, "ratbot_inference_fps", "gauge", "Most recent measured inference loop FPS.")
        self._sample(lines, "ratbot_inference_fps", getattr(tracker, "last_inference_fps", None))
        inference_device = getattr(tracker, "inference_device", None) or "auto"
        runtime_device = getattr(tracker, "inference_runtime_device", None) or "unknown"
        self._help(lines, "ratbot_inference_device_info", "gauge", "Configured and reported YOLO inference device.")
        self._sample(
            lines,
            "ratbot_inference_device_info",
            1,
            configured=inference_device,
            runtime=runtime_device,
        )
        controller = getattr(tracker, "tracking_controller", None)
        self._help(lines, "ratbot_tracking_control_fps", "gauge", "Most recent measured tracking control loop FPS.")
        self._sample(lines, "ratbot_tracking_control_fps", getattr(controller, "last_actual_fps", None))

        world_tracking = bool(getattr(tracker, "world_tracking", False))
        self._help(lines, "ratbot_world_tracking_enabled", "gauge", "World-frame tracking enabled state.")
        self._sample(lines, "ratbot_world_tracking_enabled", world_tracking)
        self._help(lines, "ratbot_world_actuation_enabled", "gauge", "World-frame actuation enabled state.")
        self._sample(
            lines,
            "ratbot_world_actuation_enabled",
            bool(getattr(tracker, "world_actuation_enabled", False)),
        )
        self._help(lines, "ratbot_world_calibration_validated", "gauge", "World-frame calibration validation gate.")
        self._sample(
            lines,
            "ratbot_world_calibration_validated",
            bool(getattr(tracker, "world_calibration_validated", False)),
        )

        tracks = self._tracks(tracker, detection_data)
        selected = detection_data.get(
            "selected_track_id",
            getattr(getattr(tracker, "world_tracker", None), "selected_track_id", None),
        )
        self._help(lines, "ratbot_world_selected_track_id", "gauge", "Selected world-track ID, or 0 when none is selected.")
        self._sample(lines, "ratbot_world_selected_track_id", selected or 0)
        self._help(lines, "ratbot_world_tracks", "gauge", "Current active world tracks.")
        self._sample(lines, "ratbot_world_tracks", len(tracks))

        by_status: Counter[str] = Counter(str(track.get("status", "unknown")) for track in tracks)
        self._help(lines, "ratbot_world_tracks_by_status", "gauge", "Current active world tracks by lifecycle status.")
        for status, count in sorted(by_status.items()):
            self._sample(lines, "ratbot_world_tracks_by_status", count, status=status)

        recording = self._recording_status(tracker)
        self._help(lines, "ratbot_track_recording_active", "gauge", "Track recording active state.")
        self._sample(lines, "ratbot_track_recording_active", bool(recording.get("recording", False)))

    @staticmethod
    def _detection_data(tracker) -> Mapping[str, object]:
        if tracker is None or not hasattr(tracker, "get_detection_data"):
            return {}
        try:
            data = tracker.get_detection_data()
        except Exception:
            return {}
        return data if isinstance(data, Mapping) else {}

    @staticmethod
    def _tracks(tracker, detection_data: Mapping[str, object]) -> list[Mapping[str, object]]:
        tracks = detection_data.get("tracks")
        if isinstance(tracks, list):
            return [track for track in tracks if isinstance(track, Mapping)]
        if tracker is None or not hasattr(tracker, "get_world_tracks"):
            return []
        try:
            fetched = tracker.get_world_tracks()
        except Exception:
            return []
        return [track for track in fetched if isinstance(track, Mapping)]

    @staticmethod
    def _recording_status(tracker) -> Mapping[str, object]:
        if tracker is None or not hasattr(tracker, "get_track_recording_status"):
            return {}
        try:
            status = tracker.get_track_recording_status()
        except Exception:
            return {}
        return status if isinstance(status, Mapping) else {}
