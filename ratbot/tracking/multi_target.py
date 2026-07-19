"""Stable-ID multi-target management in the fixed turret-base frame."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import threading
from typing import Optional

import numpy as np

from .kalman import ConstantVelocityKalman3D
from .models import Detection3D, TargetTrackSnapshot


@dataclass(frozen=True)
class TrackManagerConfig:
    gate_distance_mm: float = 750.0
    confirm_hits: int = 3
    max_misses: int = 5
    delete_after_seconds: float = 1.5
    process_acceleration_std_mm_s2: float = 300.0
    auto_select: bool = True
    confidence_decay: float = 0.85

    def __post_init__(self):
        finite_values = (
            self.gate_distance_mm,
            self.delete_after_seconds,
            self.process_acceleration_std_mm_s2,
            self.confidence_decay,
        )
        if not all(math.isfinite(float(value)) for value in finite_values):
            raise ValueError("tracker distances, timing, noise, and decay must be finite")
        if self.gate_distance_mm <= 0:
            raise ValueError("gate_distance_mm must be positive")
        if self.confirm_hits < 1 or self.max_misses < 0:
            raise ValueError("track hit/miss limits are invalid")
        if self.delete_after_seconds < 0 or self.process_acceleration_std_mm_s2 < 0:
            raise ValueError("time and process-noise values cannot be negative")
        if not 0 <= self.confidence_decay <= 1:
            raise ValueError("confidence_decay must be between zero and one")


@dataclass
class _ManagedTrack:
    id: int
    filter: ConstantVelocityKalman3D
    confidence: float
    classification: Optional[str]
    first_seen_time: float
    last_seen_time: float
    hits: int = 1
    consecutive_hits: int = 1
    misses: int = 0
    status: str = "tentative"
    bbox: Optional[tuple[int, int, int, int]] = None
    center: Optional[tuple[int, int]] = None


class MultiTargetTracker:
    """Predict, associate, update, and select independent 3D target tracks."""

    def __init__(self, config: TrackManagerConfig):
        self.config = config
        self._tracks: dict[int, _ManagedTrack] = {}
        self._next_id = 1
        self.selected_track_id: Optional[int] = None
        self._auto_select_suppressed = False
        self._lock = threading.RLock()
        self.last_assignments: list[dict[str, object]] = []
        self.last_update_time: Optional[float] = None

    @staticmethod
    def _classes_compatible(track: _ManagedTrack, detection: Detection3D) -> bool:
        return (
            track.classification is None
            or detection.classification is None
            or track.classification == detection.classification
        )

    def _associate(self, detections: list[Detection3D]):
        """Return maximum-cardinality, minimum-distance gated assignment.

        Dynamic programming masks the smaller bipartite side, which is practical
        for the small number of simultaneously tracked physical targets while
        avoiding a mandatory SciPy dependency.
        """
        track_ids = sorted(self._tracks)
        costs: dict[tuple[int, int], float] = {}
        for track_id in track_ids:
            track = self._tracks[track_id]
            for detection_index, detection in enumerate(detections):
                if not self._classes_compatible(track, detection):
                    continue
                distance = float(
                    np.linalg.norm(
                        track.filter.position - detection.position_base_mm
                    )
                )
                if distance <= self.config.gate_distance_mm:
                    costs[(track_id, detection_index)] = distance

        def better(candidate, incumbent):
            if candidate[0] != incumbent[0]:
                return candidate if candidate[0] > incumbent[0] else incumbent
            if not math.isclose(candidate[1], incumbent[1], abs_tol=1e-12):
                return candidate if candidate[1] < incumbent[1] else incumbent
            return candidate if candidate[2] < incumbent[2] else incumbent

        if len(track_ids) <= len(detections):
            track_index = {track_id: index for index, track_id in enumerate(track_ids)}

            @lru_cache(maxsize=None)
            def solve_by_detection(detection_index: int, used_track_mask: int):
                if detection_index >= len(detections):
                    return 0, 0.0, ()
                best = solve_by_detection(detection_index + 1, used_track_mask)
                for track_id in track_ids:
                    bit = 1 << track_index[track_id]
                    cost = costs.get((track_id, detection_index))
                    if used_track_mask & bit or cost is None:
                        continue
                    count, total, matches = solve_by_detection(
                        detection_index + 1,
                        used_track_mask | bit,
                    )
                    candidate = (
                        count + 1,
                        total + cost,
                        ((track_id, detection_index, cost),) + matches,
                    )
                    best = better(candidate, best)
                return best

            _, _, match_tuple = solve_by_detection(0, 0)
        else:
            detection_count = len(detections)

            @lru_cache(maxsize=None)
            def solve_by_track(track_index: int, used_detection_mask: int):
                if track_index >= len(track_ids):
                    return 0, 0.0, ()
                track_id = track_ids[track_index]
                best = solve_by_track(track_index + 1, used_detection_mask)
                for detection_index in range(detection_count):
                    bit = 1 << detection_index
                    cost = costs.get((track_id, detection_index))
                    if used_detection_mask & bit or cost is None:
                        continue
                    count, total, matches = solve_by_track(
                        track_index + 1,
                        used_detection_mask | bit,
                    )
                    candidate = (
                        count + 1,
                        total + cost,
                        ((track_id, detection_index, cost),) + matches,
                    )
                    best = better(candidate, best)
                return best

            _, _, match_tuple = solve_by_track(0, 0)

        matches = sorted(match_tuple)
        matched_tracks = {track_id for track_id, _, _ in matches}
        matched_detections = {index for _, index, _ in matches}
        return matches, matched_tracks, matched_detections

    def _new_track(self, detection: Detection3D) -> _ManagedTrack:
        track = _ManagedTrack(
            id=self._next_id,
            filter=ConstantVelocityKalman3D(
                detection.position_base_mm,
                detection.covariance_base,
                detection.measurement_time,
                process_acceleration_std_mm_s2=(
                    self.config.process_acceleration_std_mm_s2
                ),
            ),
            confidence=float(detection.confidence),
            classification=detection.classification,
            first_seen_time=detection.measurement_time,
            last_seen_time=detection.measurement_time,
            bbox=detection.bbox,
            center=detection.center,
        )
        if self.config.confirm_hits <= 1:
            track.status = "confirmed"
        self._tracks[track.id] = track
        self._next_id += 1
        return track

    def update(self, detections: list[Detection3D], timestamp: float) -> list[TargetTrackSnapshot]:
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        with self._lock:
            if self.last_update_time is not None and timestamp < self.last_update_time:
                raise ValueError("tracker updates must not move backward in time")
            self.last_update_time = timestamp
            for track in self._tracks.values():
                track.filter.predict_to(timestamp)

            matches, matched_track_ids, matched_detection_ids = self._associate(detections)
            assignments: list[dict[str, object]] = []
            for track_id, detection_index, distance in matches:
                track = self._tracks[track_id]
                detection = detections[detection_index]
                track.filter.update(
                    detection.position_base_mm,
                    detection.covariance_base,
                )
                track.confidence = min(
                    1.0,
                    max(
                        detection.confidence,
                        track.confidence * 0.75 + detection.confidence * 0.35,
                    ),
                )
                was_confirmed = track.status in {"confirmed", "lost"}
                track.last_seen_time = detection.measurement_time
                track.hits += 1
                track.consecutive_hits += 1
                track.misses = 0
                track.bbox = detection.bbox
                track.center = detection.center
                if was_confirmed or track.consecutive_hits >= self.config.confirm_hits:
                    track.status = "confirmed"
                assignments.append(
                    {
                        "track_id": track_id,
                        "detection_index": detection_index,
                        "distance_mm": distance,
                    }
                )

            for track_id, track in list(self._tracks.items()):
                if track_id in matched_track_ids:
                    continue
                track.misses += 1
                track.consecutive_hits = 0
                track.confidence *= self.config.confidence_decay
                track.status = "lost" if track.status == "confirmed" else track.status
                missing_seconds = max(0.0, timestamp - track.last_seen_time)
                if (
                    track.misses > self.config.max_misses
                    or (
                        self.config.delete_after_seconds > 0
                        and missing_seconds > self.config.delete_after_seconds
                    )
                ):
                    del self._tracks[track_id]
                    if self.selected_track_id == track_id:
                        self.selected_track_id = None
                        # Expiry is a stop condition, never permission to redirect
                        # actuation to another visible target.
                        self._auto_select_suppressed = True

            for detection_index, detection in enumerate(detections):
                if detection_index not in matched_detection_ids:
                    track = self._new_track(detection)
                    assignments.append(
                        {
                            "track_id": track.id,
                            "detection_index": detection_index,
                            "distance_mm": None,
                        }
                    )

            self.last_assignments = assignments
            if self.selected_track_id not in self._tracks:
                self.selected_track_id = None
            if (
                self.config.auto_select
                and not self._auto_select_suppressed
                and self.selected_track_id is None
            ):
                confirmed = [
                    track for track in self._tracks.values()
                    if track.status == "confirmed" and track.misses == 0
                ]
                if confirmed:
                    self.selected_track_id = max(
                        confirmed,
                        key=lambda track: (track.confidence, -track.id),
                    ).id
            return self._snapshots()

    def _snapshot(self, track: _ManagedTrack, *, state=None, covariance=None) -> TargetTrackSnapshot:
        state = track.filter.state if state is None else state
        covariance = track.filter.covariance if covariance is None else covariance
        return TargetTrackSnapshot(
            id=track.id,
            position=np.asarray(state[:3], dtype=float).copy(),
            velocity=np.asarray(state[3:], dtype=float).copy(),
            covariance=np.asarray(covariance, dtype=float).copy(),
            confidence=float(track.confidence),
            status=track.status,
            classification=track.classification,
            first_seen_time=track.first_seen_time,
            last_seen_time=track.last_seen_time,
            hits=track.hits,
            consecutive_hits=track.consecutive_hits,
            misses=track.misses,
            selected=track.id == self.selected_track_id,
            bbox=track.bbox,
            center=track.center,
        )

    def _snapshots(self) -> list[TargetTrackSnapshot]:
        return [self._snapshot(track) for track in sorted(self._tracks.values(), key=lambda item: item.id)]

    def get_tracks(self) -> list[TargetTrackSnapshot]:
        with self._lock:
            return self._snapshots()

    def select_target(self, target_id: int) -> bool:
        with self._lock:
            if target_id not in self._tracks:
                return False
            self.selected_track_id = int(target_id)
            self._auto_select_suppressed = False
            return True

    def clear_selection(self) -> None:
        with self._lock:
            self.selected_track_id = None
            self._auto_select_suppressed = True

    def clear(self) -> None:
        with self._lock:
            self._tracks.clear()
            self.selected_track_id = None
            self._auto_select_suppressed = False
            self.last_assignments = []

    def get_selected_track(
        self,
        *,
        timestamp: Optional[float] = None,
        prediction_horizon: float = 0.0,
    ) -> Optional[TargetTrackSnapshot]:
        with self._lock:
            if self.selected_track_id is None:
                return None
            track = self._tracks.get(self.selected_track_id)
            if track is None:
                return None
            if timestamp is None:
                target_time = track.filter.timestamp + max(0.0, prediction_horizon)
            else:
                target_time = max(track.filter.timestamp, float(timestamp)) + max(
                    0.0, float(prediction_horizon)
                )
            state = track.filter.predicted_state(target_time)
            covariance = track.filter.predicted_covariance(target_time)
            return self._snapshot(track, state=state, covariance=covariance)
