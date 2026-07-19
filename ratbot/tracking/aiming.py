"""Convert a selected world-frame target into the legacy servo-belief surface."""

from __future__ import annotations

import time


class WorldTrackBeliefAdapter:
    """Read selected 3D tracks through ``AngularBeliefController``'s API."""

    def __init__(
        self,
        manager,
        transformer,
        *,
        aim_latency_seconds=0.12,
        min_confidence=0.2,
        max_age_seconds=0.8,
        clock=time.monotonic,
    ):
        self.manager = manager
        self.transformer = transformer
        self.aim_latency_seconds = max(0.0, float(aim_latency_seconds))
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.max_age_seconds = max(0.0, float(max_age_seconds))
        self.clock = clock

    def get_active(self):
        now = float(self.clock())
        track = self.manager.get_selected_track(
            timestamp=now,
            prediction_horizon=self.aim_latency_seconds,
        )
        if track is None:
            return None
        age = max(0.0, now - track.last_seen_time)
        if (
            track.status not in {"confirmed", "lost"}
            or track.confidence < self.min_confidence
            or (self.max_age_seconds > 0 and age > self.max_age_seconds)
        ):
            return None

        try:
            aim = self.transformer.base_position_to_servo_raw(track.position)
            base_position = track.position - track.velocity * self.aim_latency_seconds
            base_aim = self.transformer.base_position_to_servo_raw(base_position)
            velocity_dt = 0.01
            future_position = track.position + track.velocity * velocity_dt
            future_aim = self.transformer.base_position_to_servo_raw(future_position)
        except ValueError:
            return None
        return {
            "yaw": aim["yaw"],
            "pitch": aim["pitch"],
            "base_yaw": base_aim["yaw"],
            "base_pitch": base_aim["pitch"],
            "yaw_velocity": (future_aim["yaw"] - aim["yaw"]) / velocity_dt,
            "pitch_velocity": (future_aim["pitch"] - aim["pitch"]) / velocity_dt,
            "prediction_dt": self.aim_latency_seconds,
            "confidence": track.confidence,
            "age": age,
            "track_id": track.id,
            "position_mm": track.position.copy(),
            "velocity_mm_s": track.velocity.copy(),
        }
