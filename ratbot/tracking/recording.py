"""Durable, bounded recording sessions and deterministic world-track replay."""

from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import re
import secrets
import shutil
import threading
import time
from typing import Any, Mapping, cast

import numpy as np

from .models import Detection3D
from .multi_target import MultiTargetTracker, TrackManagerConfig


_RECORDING_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_METADATA_RESERVE_BYTES = 1024 * 1024
_PARAMETER_DEFAULTS = {
    "gate_distance_mm": 750.0,
    "confirm_hits": 3,
    "max_misses": 5,
    "delete_after_seconds": 1.5,
    "reidentify_after_seconds": 8.0,
    "process_acceleration_std_mm_s2": 300.0,
    "confidence_decay": 0.85,
}


class TrackRecordingStore:
    """Own bounded track recordings and replay raw 3D observations safely."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_frames: int = 18_000,
        max_bytes: int = 64 * 1024 * 1024,
        max_sessions: int = 100,
        max_total_bytes: int = 1024 * 1024 * 1024,
        min_free_bytes: int = 512 * 1024 * 1024,
        max_replay_tracks: int = 32,
        max_replay_measurements_per_frame: int = 8,
        max_replay_work_units: int = 1_000_000,
        max_replay_seconds: float = 30.0,
        max_replay_output_bytes: int = 64 * 1024 * 1024,
    ):
        positive_limits = (
            max_frames, max_bytes, max_sessions, max_total_bytes,
            max_replay_tracks, max_replay_measurements_per_frame,
            max_replay_work_units,
            max_replay_output_bytes,
        )
        if min(positive_limits) < 1 or min_free_bytes < 0 or max_replay_seconds <= 0:
            raise ValueError("recording limits must be positive")
        self.root = Path(root)
        self.max_frames = int(max_frames)
        self.max_bytes = int(max_bytes)
        self.max_sessions = int(max_sessions)
        self.max_total_bytes = int(max_total_bytes)
        self.min_free_bytes = int(min_free_bytes)
        self.max_replay_tracks = int(max_replay_tracks)
        self.max_replay_measurements_per_frame = int(max_replay_measurements_per_frame)
        self.max_replay_work_units = int(max_replay_work_units)
        self.max_replay_seconds = float(max_replay_seconds)
        self.max_replay_output_bytes = int(max_replay_output_bytes)
        self._lock = threading.RLock()
        self._reprocess_lock = threading.Lock()
        self._active: dict[str, object] | None = None
        self._active_bytes = 0
        self._availability_error: str | None = None
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self._recover_interrupted()
        except OSError as exc:
            self._availability_error = str(exc)

    def _ensure_available(self) -> None:
        if self._availability_error is not None:
            raise OSError(f"track recording storage unavailable: {self._availability_error}")

    def _storage_usage(self) -> tuple[int, int]:
        """Return session count and bytes without following symlinks."""
        sessions = 0
        total_bytes = 0
        for path in self.root.iterdir():
            if path.is_symlink() or not path.is_dir():
                continue
            sessions += 1
            for name in ("metadata.json", "observations.jsonl"):
                item = path / name
                if item.is_file() and not item.is_symlink():
                    total_bytes += item.stat().st_size
        return sessions, total_bytes

    def _check_new_session_capacity(self) -> None:
        self._ensure_available()
        sessions, total_bytes = self._storage_usage()
        if sessions >= self.max_sessions:
            raise RuntimeError("track recording session limit reached")
        if total_bytes + self.max_bytes + _METADATA_RESERVE_BYTES > self.max_total_bytes:
            raise RuntimeError("track recording storage quota reached")
        required_free = self.min_free_bytes + self.max_bytes + _METADATA_RESERVE_BYTES
        if shutil.disk_usage(self.root).free < required_free:
            raise RuntimeError("insufficient free space for a track recording")

    @staticmethod
    def _now() -> datetime:
        return datetime.now().astimezone()

    def _session_dir(self, recording_id: str) -> Path:
        if not isinstance(recording_id, str) or not _RECORDING_ID.fullmatch(recording_id):
            raise KeyError(recording_id)
        path = self.root / recording_id
        if path.is_symlink() or not path.is_dir():
            raise KeyError(recording_id)
        try:
            path.resolve().relative_to(self.root.resolve())
        except ValueError as exc:
            raise KeyError(recording_id) from exc
        return path

    @staticmethod
    def _write_metadata(path: Path, metadata: Mapping[str, object]) -> None:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _read_metadata(path: Path) -> dict[str, object]:
        if path.is_symlink() or path.stat().st_size > _METADATA_RESERVE_BYTES:
            raise ValueError("recording metadata exceeds size or safety limits")
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("recording metadata must be an object")
        return metadata

    def _recover_interrupted(self) -> None:
        """Finalize sessions left open by a prior process crash."""
        for session_dir in self.root.iterdir():
            metadata_path = session_dir / "metadata.json"
            observations_path = session_dir / "observations.jsonl"
            if (
                session_dir.is_symlink()
                or not session_dir.is_dir()
                or not metadata_path.is_file()
            ):
                continue
            try:
                metadata = self._read_metadata(metadata_path)
                if metadata.get("status") != "recording":
                    continue
                count = 0
                first: float | None = None
                last: float | None = None
                if observations_path.is_file():
                    with observations_path.open("r", encoding="utf-8") as stream:
                        for line in stream:
                            if not line.strip():
                                continue
                            count += 1
                            try:
                                frame = json.loads(line)
                                timestamp = frame.get("monotonic_time")
                                if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool):
                                    first = float(timestamp) if first is None else first
                                    last = float(timestamp)
                            except json.JSONDecodeError:
                                pass
                metadata.update({
                    "status": "interrupted",
                    "stop_reason": "process_restart",
                    "stopped_at": self._now().isoformat(),
                    "frame_count": count,
                    "duration_seconds": max(0.0, (last or 0.0) - (first or 0.0)),
                })
                self._write_metadata(metadata_path, metadata)
            except (OSError, ValueError, json.JSONDecodeError):
                continue

    def start(self, parameters: Mapping[str, object]) -> dict[str, object]:
        with self._lock:
            if self._active is not None:
                raise RuntimeError("a track recording is already active")
            self._check_new_session_capacity()
            now = self._now()
            recording_id = f"{now.strftime('%Y%m%dT%H%M%S')}-{secrets.token_hex(3)}"
            session_dir = self.root / recording_id
            session_dir.mkdir(parents=False, exist_ok=False)
            metadata: dict[str, object] = {
                "id": recording_id,
                "schema": "ratbot.track_recording.v1",
                "status": "recording",
                "started_at": now.isoformat(),
                "stopped_at": None,
                "frame_count": 0,
                "duration_seconds": 0.0,
                "max_frames": self.max_frames,
                "max_bytes": self.max_bytes,
                "max_sessions": self.max_sessions,
                "max_total_bytes": self.max_total_bytes,
                "parameters": dict(parameters),
            }
            self._write_metadata(session_dir / "metadata.json", metadata)
            (session_dir / "observations.jsonl").touch()
            self._active = metadata
            self._active_bytes = 0
            return dict(metadata)

    def _stop_locked(self, reason: str, *, status: str = "complete") -> dict[str, object]:
        if self._active is None:
            raise RuntimeError("no track recording is active")
        metadata = dict(self._active)
        metadata.update({
            "status": status,
            "stop_reason": reason,
            "stopped_at": self._now().isoformat(),
        })
        session_dir = self.root / str(metadata["id"])
        try:
            self._write_metadata(session_dir / "metadata.json", metadata)
        finally:
            self._active = None
            self._active_bytes = 0
        return metadata

    def append(self, frame: Mapping[str, object]) -> None:
        with self._lock:
            if self._active is None:
                return
            encoded = json.dumps(dict(frame), separators=(",", ":")) + "\n"
            encoded_size = len(encoded.encode("utf-8"))
            if self._active_bytes + encoded_size > self.max_bytes:
                self._stop_locked("byte_limit")
                return
            session_dir = self.root / str(self._active["id"])
            with (session_dir / "observations.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(encoded)
            self._active_bytes += encoded_size
            raw_count = self._active.get("frame_count", 0)
            count = (raw_count if isinstance(raw_count, int) else 0) + 1
            self._active["frame_count"] = count
            timestamp = frame.get("monotonic_time")
            if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool):
                self._active.setdefault("first_monotonic_time", float(timestamp))
                self._active["last_monotonic_time"] = float(timestamp)
                first = float(cast(float, self._active["first_monotonic_time"]))
                self._active["duration_seconds"] = max(0.0, float(timestamp) - first)
            if count >= self.max_frames:
                self._stop_locked("frame_limit")

    def fail(self, message: str) -> None:
        """Mark an active session failed without propagating storage errors to inference."""
        with self._lock:
            if self._active is None:
                return
            self._active["error"] = str(message)[:500]
            try:
                self._stop_locked("storage_error", status="failed")
            except OSError:
                self._active = None
                self._active_bytes = 0

    def stop(self) -> dict[str, object]:
        with self._lock:
            return self._stop_locked("user")

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "available": self._availability_error is None,
                "error": self._availability_error,
                "recording": self._active is not None,
                "active": dict(self._active) if self._active is not None else None,
                "limits": {
                    "max_frames": self.max_frames,
                    "max_bytes": self.max_bytes,
                    "max_sessions": self.max_sessions,
                    "max_total_bytes": self.max_total_bytes,
                },
            }

    def list_recordings(self) -> list[dict[str, object]]:
        if self._availability_error is not None:
            return []
        recordings: list[dict[str, object]] = []
        for path in self.root.iterdir():
            metadata_path = path / "metadata.json"
            if path.is_symlink() or not path.is_dir() or not metadata_path.is_file():
                continue
            try:
                metadata = self._read_metadata(metadata_path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if metadata.get("id") == path.name:
                recordings.append(metadata)
        return sorted(recordings, key=lambda item: str(item.get("started_at", "")), reverse=True)

    def delete(self, recording_id: str) -> dict[str, object]:
        self._ensure_available()
        with self._lock:
            if self._active is not None and self._active.get("id") == recording_id:
                raise RuntimeError("stop the active recording before deleting it")
            session_dir = self._session_dir(recording_id)
            metadata: dict[str, object] = {"id": recording_id}
            try:
                metadata = self._read_metadata(session_dir / "metadata.json")
            except (OSError, ValueError, json.JSONDecodeError):
                pass
            shutil.rmtree(session_dir)
            return {
                "success": True,
                "id": recording_id,
                "frame_count": metadata.get("frame_count", 0),
            }

    def load(self, recording_id: str) -> dict[str, object]:
        self._ensure_available()
        with self._lock:
            if self._active is not None and self._active.get("id") == recording_id:
                raise RuntimeError("stop the active recording before replaying it")
        session_dir = self._session_dir(recording_id)
        observations_path = session_dir / "observations.jsonl"
        try:
            if observations_path.stat().st_size > self.max_bytes:
                raise ValueError(f"recording {recording_id!r} exceeds replay byte limit")
            metadata = self._read_metadata(session_dir / "metadata.json")
            frames: list[dict[str, object]] = []
            with observations_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    if len(frames) >= self.max_frames:
                        raise ValueError(f"recording {recording_id!r} exceeds replay frame limit")
                    frame = json.loads(line)
                    if not isinstance(frame, dict):
                        raise ValueError(f"recording {recording_id!r} contains a non-object frame")
                    frames.append(frame)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"recording {recording_id!r} is unreadable") from exc
        return {"metadata": metadata, "frames": frames}

    @staticmethod
    def _validated_parameters(values: Mapping[str, object]) -> dict[str, int | float]:
        unknown = set(values) - set(_PARAMETER_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown replay parameters: {', '.join(sorted(unknown))}")
        merged = {**_PARAMETER_DEFAULTS, **dict(values)}
        for key in ("confirm_hits", "max_misses"):
            value = merged[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")
        for key in set(_PARAMETER_DEFAULTS) - {"confirm_hits", "max_misses"}:
            value = merged[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be numeric")
            try:
                numeric_value = float(value)
            except (OverflowError, ValueError) as exc:
                raise ValueError(f"{key} is outside the supported numeric range") from exc
            if not math.isfinite(numeric_value):
                raise ValueError(f"{key} must be finite")
        validated = {
            "gate_distance_mm": float(cast(Any, merged["gate_distance_mm"])),
            "confirm_hits": int(cast(Any, merged["confirm_hits"])),
            "max_misses": int(cast(Any, merged["max_misses"])),
            "delete_after_seconds": float(cast(Any, merged["delete_after_seconds"])),
            "reidentify_after_seconds": float(
                cast(Any, merged["reidentify_after_seconds"])
            ),
            "process_acceleration_std_mm_s2": float(
                cast(Any, merged["process_acceleration_std_mm_s2"])
            ),
            "confidence_decay": float(cast(Any, merged["confidence_decay"])),
        }
        bounds = {
            "gate_distance_mm": (1.0, 10_000.0),
            "confirm_hits": (1, 100),
            "max_misses": (0, 30),
            "delete_after_seconds": (0.001, 10.0),
            "reidentify_after_seconds": (0.0, 30.0),
            "process_acceleration_std_mm_s2": (0.001, 10_000.0),
            "confidence_decay": (0.001, 1.0),
        }
        for key, (minimum, maximum) in bounds.items():
            value = validated[key]
            if value < minimum or value > maximum:
                raise ValueError(f"{key} must be between {minimum} and {maximum}")
        return validated

    @staticmethod
    def _finite_vector(value: object, length: int, name: str) -> tuple[float, ...]:
        if not isinstance(value, (list, tuple)) or len(value) != length:
            raise ValueError(f"{name} must contain {length} numbers")
        result: list[float] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(f"{name} must contain only numbers")
            try:
                number = float(item)
            except (OverflowError, ValueError) as exc:
                raise ValueError(f"{name} contains an out-of-range number") from exc
            if not math.isfinite(number):
                raise ValueError(f"{name} must contain only finite numbers")
            result.append(number)
        return tuple(result)

    @classmethod
    def _finite_matrix3(cls, value: object, name: str) -> np.ndarray:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"{name} must be a 3x3 numeric matrix")
        rows = [cls._finite_vector(row, 3, name) for row in value]
        return np.asarray(rows, dtype=float)

    @classmethod
    def _integer_vector(cls, value: object, length: int, name: str) -> tuple[int, ...]:
        numbers = cls._finite_vector(value, length, name)
        if any(not number.is_integer() for number in numbers):
            raise ValueError(f"{name} must contain only integers")
        return tuple(int(number) for number in numbers)

    @staticmethod
    def _finite_scalar(value: object, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        try:
            number = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"{name} is out of numeric range") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
        return number

    def reprocess(self, recording_id: str, parameters: Mapping[str, object]) -> dict[str, object]:
        if not self._reprocess_lock.acquire(blocking=False):
            raise RuntimeError("another recording is already being reprocessed")
        try:
            replay = self.load(recording_id)
            metadata = replay["metadata"]
            recorded = metadata.get("parameters", {}) if isinstance(metadata, dict) else {}
            base = {key: value for key, value in recorded.items() if key in _PARAMETER_DEFAULTS} if isinstance(recorded, dict) else {}
            tuned = self._validated_parameters({**base, **dict(parameters)})
            tracker = MultiTargetTracker(TrackManagerConfig(
                gate_distance_mm=float(tuned["gate_distance_mm"]),
                confirm_hits=int(tuned["confirm_hits"]),
                max_misses=int(tuned["max_misses"]),
                delete_after_seconds=float(tuned["delete_after_seconds"]),
                reidentify_after_seconds=float(tuned["reidentify_after_seconds"]),
                process_acceleration_std_mm_s2=float(
                    tuned["process_acceleration_std_mm_s2"]
                ),
                confidence_decay=float(tuned["confidence_decay"]),
                auto_select=True,
            ))
            frames: list[dict[str, object]] = []
            output_bytes = 0
            replay_work_units = 0
            prior_track_count = 0
            replay_deadline = time.monotonic() + self.max_replay_seconds
            for original in cast(list[dict[str, Any]], replay["frames"]):
                if time.monotonic() > replay_deadline:
                    raise ValueError("reprocessed recording exceeds runtime limit")
                timestamp_value = original.get("monotonic_time")
                if (
                    isinstance(timestamp_value, bool)
                    or not isinstance(timestamp_value, (int, float))
                ):
                    raise ValueError("recording frame has an invalid monotonic_time")
                timestamp = float(timestamp_value)
                detections: list[Detection3D] = []
                measurements = original.get("measurements", [])
                if not isinstance(measurements, list):
                    raise ValueError("recording frame measurements must be a list")
                if len(measurements) > self.max_replay_measurements_per_frame:
                    raise ValueError("recording exceeds per-frame replay measurement limit")
                for measurement in measurements:
                    if not isinstance(measurement, dict):
                        raise ValueError("recording measurement must be an object")
                    point = measurement.get("base_point_mm")
                    covariance = measurement.get("base_covariance")
                    if point is None or covariance is None:
                        continue
                    detection_confidence = self._finite_scalar(
                        measurement.get("detection_confidence", 0.0),
                        "detection_confidence",
                    )
                    depth_confidence = self._finite_scalar(
                        measurement.get("depth_confidence", 0.0),
                        "depth_confidence",
                    )
                    confidence = detection_confidence * depth_confidence
                    position = self._finite_vector(point, 3, "base_point_mm")
                    covariance_matrix = self._finite_matrix3(
                        covariance, "base_covariance"
                    )
                    bbox_value = measurement.get("bbox")
                    center_value = measurement.get("center")
                    bbox_numbers = (
                        self._integer_vector(bbox_value, 4, "bbox")
                        if bbox_value is not None else None
                    )
                    bbox = (
                        (
                            bbox_numbers[0], bbox_numbers[1],
                            bbox_numbers[2], bbox_numbers[3],
                        )
                        if bbox_numbers is not None else None
                    )
                    center_numbers = (
                        self._integer_vector(center_value, 2, "center")
                        if center_value is not None else None
                    )
                    center = (
                        (center_numbers[0], center_numbers[1])
                        if center_numbers is not None else None
                    )
                    classification = measurement.get("class")
                    if classification is not None and not isinstance(classification, str):
                        raise ValueError("class must be a string or null")
                    detections.append(Detection3D(
                        position_base_mm=np.asarray(position, dtype=float),
                        covariance_base=covariance_matrix,
                        confidence=max(0.0, min(1.0, confidence)),
                        classification=classification,
                        measurement_time=timestamp,
                        bbox=bbox,
                        center=center,
                    ))
                detection_count = len(detections)
                replay_work_units += (
                    max(1, prior_track_count)
                    * max(1, detection_count)
                    * (1 << detection_count)
                )
                if replay_work_units > self.max_replay_work_units:
                    raise ValueError("reprocessed recording exceeds CPU work limit")
                tracks = tracker.update(detections, timestamp)
                prior_track_count = tracker.managed_track_count()
                if prior_track_count > self.max_replay_tracks:
                    raise ValueError("reprocessed recording exceeds track-count limit")
                frame = dict(original)
                frame["tracks"] = [track.to_dict() for track in tracks]
                frame["assignments"] = list(tracker.last_assignments)
                frame["selected_track_id"] = tracker.selected_track_id
                output_bytes += len(json.dumps(frame, separators=(",", ":")).encode("utf-8"))
                if output_bytes > self.max_replay_output_bytes:
                    raise ValueError("reprocessed recording exceeds output-size limit")
                frames.append(frame)
            return {"metadata": metadata, "parameters": tuned, "frames": frames}
        finally:
            self._reprocess_lock.release()
