"""Angular target belief estimation and servo control."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ServoBounds:
    """Raw servo position limits used by the belief controller."""

    yaw_min: float
    yaw_max: float
    pitch_min: float
    pitch_max: float


class AngularTargetBelief:
    """Thread-safe angular target belief updated by vision observations."""

    def __init__(self, update_alpha=0.45, miss_decay=0.94,
                 min_confidence=0.15, max_age=1.5, reseed_distance_raw=160,
                 velocity_alpha=0.45, velocity_decay=0.96,
                 max_velocity_raw_per_s=600, max_prediction_age=0.45,
                 reseed_confirmations=2, reseed_match_distance_raw=120,
                 reseed_max_interval=0.8, reseed_min_confidence=0.55,
                 pitch_update_alpha=None, pitch_velocity_alpha=None,
                 max_pitch_velocity_raw_per_s=None):
        self.update_alpha = max(0.0, min(1.0, float(update_alpha)))
        self.pitch_update_alpha = (
            self.update_alpha if pitch_update_alpha is None
            else max(0.0, min(1.0, float(pitch_update_alpha)))
        )
        self.miss_decay = max(0.0, min(1.0, float(miss_decay)))
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.max_age = max(0.0, float(max_age))
        self.reseed_distance_raw = max(0.0, float(reseed_distance_raw))
        self.velocity_alpha = max(0.0, min(1.0, float(velocity_alpha)))
        self.pitch_velocity_alpha = (
            self.velocity_alpha if pitch_velocity_alpha is None
            else max(0.0, min(1.0, float(pitch_velocity_alpha)))
        )
        self.velocity_decay = max(0.0, min(1.0, float(velocity_decay)))
        self.max_velocity_raw_per_s = max(0.0, float(max_velocity_raw_per_s))
        self.max_pitch_velocity_raw_per_s = (
            self.max_velocity_raw_per_s if max_pitch_velocity_raw_per_s is None
            else max(0.0, float(max_pitch_velocity_raw_per_s))
        )
        self.max_prediction_age = max(0.0, float(max_prediction_age))
        self.reseed_confirmations = max(1, int(reseed_confirmations))
        self.reseed_match_distance_raw = max(0.0, float(reseed_match_distance_raw))
        self.reseed_max_interval = max(0.0, float(reseed_max_interval))
        reseed_min_confidence = (
            self.min_confidence
            if reseed_min_confidence is None
            else float(reseed_min_confidence)
        )
        self.reseed_min_confidence = max(
            self.min_confidence,
            min(1.0, reseed_min_confidence),
        )
        self.lock = threading.Lock()
        self.yaw = None
        self.pitch = None
        self.yaw_velocity = 0.0
        self.pitch_velocity = 0.0
        self.confidence = 0.0
        self.last_update = 0.0
        self.pending_reseed_yaw = None
        self.pending_reseed_pitch = None
        self.pending_reseed_time = 0.0
        self.pending_reseed_count = 0

    def _clamp_velocity(self, velocity, limit=None):
        if limit is None:
            limit = self.max_velocity_raw_per_s
        if limit <= 0:
            return 0.0
        return max(
            -limit,
            min(limit, float(velocity)),
        )

    def update(self, yaw, pitch, confidence):
        now = time.time()
        yaw = float(yaw)
        pitch = float(pitch)
        confidence = max(0.0, min(1.0, float(confidence)))

        with self.lock:
            age = now - self.last_update if self.last_update else float("inf")
            predicted = False
            if self.yaw is not None and self.pitch is not None and age != float("inf"):
                prediction_dt = min(age, self.max_prediction_age)
                if prediction_dt > 0:
                    self.yaw += self.yaw_velocity * prediction_dt
                    self.pitch += self.pitch_velocity * prediction_dt
                    predicted = True

            stale = self.max_age > 0 and age > self.max_age
            weak = self.confidence < self.min_confidence
            jump = 0.0
            if self.yaw is not None and self.pitch is not None:
                jump = max(abs(yaw - self.yaw), abs(pitch - self.pitch))
            large_jump = self.reseed_distance_raw > 0 and jump > self.reseed_distance_raw

            reseed_reason = None
            ignored_reason = None
            if self.yaw is None or self.pitch is None or self.confidence <= 0:
                reseed_reason = "empty"
            elif large_jump:
                if confidence < self.reseed_min_confidence:
                    ignored_reason = (
                        f"low-conf jump {jump:.0f} "
                        f"(conf {confidence:.2f} < {self.reseed_min_confidence:.2f})"
                    )
                else:
                    pending_age = now - self.pending_reseed_time if self.pending_reseed_time else float("inf")
                    pending_match = False
                    if self.pending_reseed_yaw is not None and self.pending_reseed_pitch is not None:
                        pending_distance = max(
                            abs(yaw - self.pending_reseed_yaw),
                            abs(pitch - self.pending_reseed_pitch),
                        )
                        pending_match = (
                            pending_age <= self.reseed_max_interval
                            and pending_distance <= self.reseed_match_distance_raw
                        )

                    if pending_match:
                        self.pending_reseed_count += 1
                    else:
                        self.pending_reseed_count = 1

                    self.pending_reseed_yaw = yaw
                    self.pending_reseed_pitch = pitch
                    self.pending_reseed_time = now

                    if self.pending_reseed_count >= self.reseed_confirmations:
                        reason_prefix = "stale " if stale else "weak " if weak else ""
                        reseed_reason = f"{reason_prefix}jump {jump:.0f} confirmed"
                    else:
                        reason_prefix = "stale " if stale else "weak " if weak else ""
                        ignored_reason = f"pending {reason_prefix}jump {jump:.0f}"
            elif stale:
                reseed_reason = "stale"
            elif weak:
                reseed_reason = "weak"

            if ignored_reason:
                self.confidence *= self.miss_decay
                self.yaw_velocity *= self.velocity_decay
                self.pitch_velocity *= self.velocity_decay
                if self.confidence < 0.01:
                    self.confidence = 0.0
                    self.yaw_velocity = 0.0
                    self.pitch_velocity = 0.0
                self.last_update = now
                snapshot = self.snapshot_locked(now)
                snapshot["reseeded"] = False
                snapshot["reseed_reason"] = None
                snapshot["ignored"] = True
                snapshot["ignored_reason"] = ignored_reason
                snapshot["predicted_before_update"] = predicted
                return snapshot

            if reseed_reason:
                self.yaw = yaw
                self.pitch = pitch
                self.yaw_velocity = 0.0
                self.pitch_velocity = 0.0
            else:
                previous_yaw = self.yaw
                previous_pitch = self.pitch
                dt = max(0.001, age)
                self.yaw += self.update_alpha * (yaw - self.yaw)
                self.pitch += self.pitch_update_alpha * (pitch - self.pitch)
                observed_yaw_velocity = (self.yaw - previous_yaw) / dt
                observed_pitch_velocity = (self.pitch - previous_pitch) / dt
                self.yaw_velocity += self.velocity_alpha * (
                    observed_yaw_velocity - self.yaw_velocity
                )
                self.pitch_velocity += self.pitch_velocity_alpha * (
                    observed_pitch_velocity - self.pitch_velocity
                )
                self.yaw_velocity = self._clamp_velocity(self.yaw_velocity)
                self.pitch_velocity = self._clamp_velocity(
                    self.pitch_velocity,
                    self.max_pitch_velocity_raw_per_s,
                )
                self.pending_reseed_yaw = None
                self.pending_reseed_pitch = None
                self.pending_reseed_time = 0.0
                self.pending_reseed_count = 0

            self.confidence = min(1.0, max(confidence, self.confidence * 0.85 + confidence * 0.35))
            self.last_update = now
            if reseed_reason:
                self.pending_reseed_yaw = None
                self.pending_reseed_pitch = None
                self.pending_reseed_time = 0.0
                self.pending_reseed_count = 0
            snapshot = self.snapshot_locked(now)
            snapshot["reseeded"] = reseed_reason is not None
            snapshot["reseed_reason"] = reseed_reason
            snapshot["ignored"] = False
            snapshot["ignored_reason"] = None
            snapshot["predicted_before_update"] = predicted
            return snapshot

    def clear(self):
        with self.lock:
            self.yaw = None
            self.pitch = None
            self.yaw_velocity = 0.0
            self.pitch_velocity = 0.0
            self.confidence = 0.0
            self.last_update = 0.0
            self.pending_reseed_yaw = None
            self.pending_reseed_pitch = None
            self.pending_reseed_time = 0.0
            self.pending_reseed_count = 0
            return self.snapshot_locked(time.time())

    def decay(self):
        with self.lock:
            if self.confidence <= 0:
                return

            self.confidence *= self.miss_decay
            self.yaw_velocity *= self.velocity_decay
            self.pitch_velocity *= self.velocity_decay
            if self.confidence < 0.01:
                self.confidence = 0.0
                self.yaw_velocity = 0.0
                self.pitch_velocity = 0.0

    def snapshot_locked(self, now=None, predict=False):
        if now is None:
            now = time.time()
        age = now - self.last_update if self.last_update else float("inf")
        yaw = self.yaw
        pitch = self.pitch
        prediction_dt = 0.0
        if predict and yaw is not None and pitch is not None and age != float("inf"):
            prediction_dt = min(age, self.max_prediction_age)
            yaw += self.yaw_velocity * prediction_dt
            pitch += self.pitch_velocity * prediction_dt
        return {
            "yaw": yaw,
            "pitch": pitch,
            "base_yaw": self.yaw,
            "base_pitch": self.pitch,
            "yaw_velocity": self.yaw_velocity,
            "pitch_velocity": self.pitch_velocity,
            "prediction_dt": prediction_dt,
            "confidence": self.confidence,
            "age": age,
        }

    def get_active(self):
        with self.lock:
            if self.yaw is None or self.pitch is None:
                return None

            snapshot = self.snapshot_locked(predict=True)
            if (
                snapshot["confidence"] < self.min_confidence
                or (self.max_age > 0 and snapshot["age"] > self.max_age)
            ):
                return None

            return snapshot


class AngularBeliefController:
    """Control loop that reads target belief and moves a robot interface toward it."""

    def __init__(self, robot, belief, bounds, control_fps=20,
                 max_yaw_step=45, max_pitch_step=45,
                 max_yaw_speed_raw_per_s=None,
                 max_pitch_speed_raw_per_s=None,
                 deadband_raw=4, min_step_raw=3):
        self.robot = robot
        self.belief = belief
        self.bounds = bounds
        self.control_fps = max(1.0, float(control_fps))
        self.max_yaw_step = max(0, int(max_yaw_step))
        self.max_pitch_step = max(0, int(max_pitch_step))
        self.max_yaw_speed_raw_per_s = self._optional_positive_float(
            max_yaw_speed_raw_per_s
        )
        self.max_pitch_speed_raw_per_s = self._optional_positive_float(
            max_pitch_speed_raw_per_s
        )
        self.deadband_raw = max(0, int(deadband_raw))
        self.min_step_raw = max(0, int(min_step_raw))
        self.yaw_integral = 0.0
        self.pitch_integral = 0.0
        self.yaw_prev_error = 0.0
        self.pitch_prev_error = 0.0
        self.last_time = time.time()
        self.loop_count = 0
        self.window_start = time.time()

    @staticmethod
    def _optional_positive_float(value):
        if value is None:
            return None
        value = float(value)
        if value <= 0:
            return None
        return value

    def reset(self):
        self.yaw_integral = 0.0
        self.pitch_integral = 0.0
        self.yaw_prev_error = 0.0
        self.pitch_prev_error = 0.0
        self.last_time = time.time()

    def _minimum_raw_step(self, raw_delta, raw_error):
        if (
            self.min_step_raw <= 0
            or abs(raw_error) <= self.deadband_raw
            or abs(raw_delta) >= self.min_step_raw
        ):
            return raw_delta

        return self.min_step_raw if raw_error > 0 else -self.min_step_raw

    def _limit_step(self, current_position, desired_position, max_step):
        if max_step <= 0:
            return desired_position

        delta = desired_position - current_position
        if abs(delta) <= max_step:
            return desired_position

        return current_position + (max_step if delta > 0 else -max_step)

    def _limit_speed(self, current_position, desired_position, max_speed, dt):
        if max_speed is None:
            return desired_position

        max_delta = max_speed * dt
        if max_delta <= 0:
            return current_position

        delta = desired_position - current_position
        if abs(delta) <= max_delta:
            return desired_position

        return current_position + (max_delta if delta > 0 else -max_delta)

    def track_once(self):
        belief = self.belief.get_active()
        if belief is None:
            return

        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        if dt < 0.001:
            dt = 0.001

        yaw_error_raw = belief["yaw"] - self.robot.current_yaw
        pitch_error_raw = belief["pitch"] - self.robot.current_pitch

        if abs(yaw_error_raw) <= self.deadband_raw and abs(pitch_error_raw) <= self.deadband_raw:
            return

        angle_error_yaw = self.robot.servo_raw_to_angle(yaw_error_raw, axis='yaw')
        angle_error_pitch = self.robot.servo_raw_to_angle(pitch_error_raw, axis='pitch')

        yaw_p = self.robot.pid_yaw_kp * angle_error_yaw
        self.yaw_integral += angle_error_yaw * dt
        self.yaw_integral = max(-self.robot.pid_max_integral, min(self.robot.pid_max_integral, self.yaw_integral))
        yaw_i = self.robot.pid_yaw_ki * self.yaw_integral
        yaw_d = self.robot.pid_yaw_kd * (angle_error_yaw - self.yaw_prev_error) / dt
        self.yaw_prev_error = angle_error_yaw

        pitch_p = self.robot.pid_pitch_kp * angle_error_pitch
        self.pitch_integral += angle_error_pitch * dt
        self.pitch_integral = max(-self.robot.pid_max_integral, min(self.robot.pid_max_integral, self.pitch_integral))
        pitch_i = self.robot.pid_pitch_ki * self.pitch_integral
        pitch_d = self.robot.pid_pitch_kd * (angle_error_pitch - self.pitch_prev_error) / dt
        self.pitch_prev_error = angle_error_pitch

        yaw_correction_raw = self.robot.angle_to_servo_raw(yaw_p + yaw_i + yaw_d, axis='yaw')
        pitch_correction_raw = self.robot.angle_to_servo_raw(pitch_p + pitch_i + pitch_d, axis='pitch')
        yaw_correction_raw = self._minimum_raw_step(yaw_correction_raw, yaw_error_raw)
        pitch_correction_raw = self._minimum_raw_step(pitch_correction_raw, pitch_error_raw)

        desired_yaw = self.robot.current_yaw + yaw_correction_raw
        desired_pitch = self.robot.current_pitch + pitch_correction_raw
        desired_yaw = max(self.bounds.yaw_min, min(self.bounds.yaw_max, desired_yaw))
        desired_pitch = max(self.bounds.pitch_min, min(self.bounds.pitch_max, desired_pitch))
        desired_yaw = self._limit_step(self.robot.current_yaw, desired_yaw, self.max_yaw_step)
        desired_pitch = self._limit_step(self.robot.current_pitch, desired_pitch, self.max_pitch_step)
        desired_yaw = self._limit_speed(
            self.robot.current_yaw,
            desired_yaw,
            self.max_yaw_speed_raw_per_s,
            dt,
        )
        desired_pitch = self._limit_speed(
            self.robot.current_pitch,
            desired_pitch,
            self.max_pitch_speed_raw_per_s,
            dt,
        )
        desired_yaw = int(round(desired_yaw))
        desired_pitch = int(round(desired_pitch))

        if desired_yaw == self.robot.current_yaw and desired_pitch == self.robot.current_pitch:
            return

        print(
            "   Belief control: "
            f"belief=({belief['yaw']:.1f}, {belief['pitch']:.1f}, conf={belief['confidence']:.2f}, age={belief['age']:.2f}s), "
            f"vel=({belief['yaw_velocity']:.0f}, {belief['pitch_velocity']:.0f}) raw/s, pred={belief['prediction_dt']:.2f}s, "
            f"error_raw=({yaw_error_raw:.1f}, {pitch_error_raw:.1f}), "
            f"move=({self.robot.current_yaw}->{desired_yaw}, {self.robot.current_pitch}->{desired_pitch})"
        )

        self.robot.set_yaw(desired_yaw)
        self.robot.set_pitch(desired_pitch)

    def run(self):
        while self.robot.camera_active:
            loop_start = time.time()
            self.track_once()
            self.loop_count += 1
            window_elapsed = time.time() - self.window_start
            if window_elapsed >= 5.0:
                actual_fps = self.loop_count / window_elapsed
                print(f"Tracking control actual FPS: {actual_fps:.1f} (target {self.control_fps:g})", flush=True)
                self.loop_count = 0
                self.window_start = time.time()
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, (1.0 / self.control_fps) - elapsed))

    def start(self):
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        print(f"Tracking control thread started ({self.control_fps:g} FPS target)")
