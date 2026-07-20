"""Hardware adapters for Feetech tracking servos and sysfs PWM trigger."""

from __future__ import annotations

import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .belief import ServoBounds


class TrackingServoController:
    """Own Feetech bus state and expose raw yaw/pitch position control."""

    def __init__(
        self,
        port,
        enabled,
        bounds: ServoBounds,
        yaw_center,
        pitch_center,
        yaw_motor_id,
        pitch_motor_id,
        bus_factory=None,
        motor_factory=None,
        norm_mode=None,
    ):
        self.port = port
        self.enabled = bool(enabled)
        self.bounds = bounds
        self.yaw_motor_id = yaw_motor_id
        self.pitch_motor_id = pitch_motor_id
        self.bus_factory = bus_factory
        self.motor_factory = motor_factory
        self.norm_mode = norm_mode
        self.connected = False
        self.motor_bus = None
        self.current_yaw = int(yaw_center)
        self.current_pitch = int(pitch_center)
        self.lock = threading.Lock()

    def read_measured_positions(self):
        """Read Present_Position or fail closed when connected hardware fails.

        In simulation/disconnected mode, commanded state is the simulated physical
        state. Connected callers performing geometry must never receive a command
        fallback disguised as measured pose.
        """
        if not self.connected:
            return self.current_yaw, self.current_pitch
        if not self.motor_bus:
            raise RuntimeError("servo bus is unavailable for measured readback")
        with self.lock:
            yaw_pos = self.motor_bus.read(
                "Present_Position", "yaw", normalize=False
            )
            pitch_pos = self.motor_bus.read(
                "Present_Position", "pitch", normalize=False
            )
        return int(yaw_pos), int(pitch_pos)

    def read_positions(self):
        """Read positions for display, retaining legacy fallback behavior."""
        try:
            return self.read_measured_positions()
        except Exception as exc:
            print(f"Error reading motor positions: {exc}")
            return self.current_yaw, self.current_pitch

    def connect(self):
        """Connect to the configured Feetech yaw and pitch servos."""
        previous_alarm_handler = signal.getsignal(signal.SIGALRM)

        def timeout_handler(signum, frame):
            raise TimeoutError("Connection timed out")

        try:
            if self.bus_factory is None or self.motor_factory is None:
                raise RuntimeError("Feetech libraries are unavailable")

            motors = {
                "yaw": self.motor_factory(
                    self.yaw_motor_id,
                    "sts3215",
                    self.norm_mode,
                ),
                "pitch": self.motor_factory(
                    self.pitch_motor_id,
                    "sts3215",
                    self.norm_mode,
                ),
            }
            self.motor_bus = self.bus_factory(self.port, motors)

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(5)
            try:
                self.motor_bus.connect(handshake=False)
            except TimeoutError:
                print(f"Connection timed out after 5 seconds on {self.port}")
                raise
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous_alarm_handler)

            self.connected = True
            self.current_yaw, self.current_pitch = self.read_positions()
            print(f"✓ Connected to tracking servos on {self.port}")
            print(
                f"  Yaw motor (ID {self.yaw_motor_id}): "
                f"{self.bounds.yaw_min}-{self.bounds.yaw_max} raw"
            )
            print(
                f"  Pitch motor (ID {self.pitch_motor_id}): "
                f"{self.bounds.pitch_min}-{self.bounds.pitch_max} raw"
            )
            print(
                f"  Current positions: Yaw={self.current_yaw}, "
                f"Pitch={self.current_pitch}"
            )
        except Exception as exc:
            print(f"Failed to connect to tracking servos: {exc}")
            self.connected = False
            self.motor_bus = None

    def raw_write(self, motor_name, value):
        """Write an unsigned 16-bit raw goal position to a motor."""
        if not self.motor_bus or not self.connected:
            print(f"[SERVO SIMULATION] Would move {motor_name} to {value}")
            return

        try:
            value = int(value) & 0xFFFF
            with self.lock:
                self.motor_bus.write(
                    "Goal_Position",
                    motor_name,
                    value,
                    normalize=False,
                )
        except Exception as exc:
            print(f"Error writing to {motor_name}: {exc}")

    def set_yaw(self, position):
        """Clamp and set the raw yaw position."""
        position = max(
            self.bounds.yaw_min,
            min(self.bounds.yaw_max, int(position)),
        )
        self.current_yaw = position
        if self.enabled:
            self.raw_write("yaw", position)

    def set_pitch(self, position):
        """Clamp and set the raw pitch position."""
        position = max(
            self.bounds.pitch_min,
            min(self.bounds.pitch_max, int(position)),
        )
        self.current_pitch = position
        if self.enabled:
            self.raw_write("pitch", position)

    def close(self):
        """Disconnect the motor bus and mark the adapter disconnected."""
        if self.motor_bus:
            try:
                self.motor_bus.disconnect()
                print("Disconnected from servos")
            except Exception:
                pass
        self.connected = False


@dataclass(frozen=True)
class TriggerServoConfig:
    pwm_chip: int
    pwm_channel: int
    neutral_angle: float
    action_angle: float


class TriggerServoController:
    """Control an action servo through Linux sysfs PWM."""

    def __init__(
        self,
        enabled,
        config: TriggerServoConfig,
        sysfs_root=Path("/sys/class/pwm"),
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.enabled = bool(enabled)
        self.config = config
        self.sysfs_root = Path(sysfs_root)
        self.sleep = sleep
        self.pwm_path: Optional[Path] = None

    def initialize(self):
        """Export and initialize the PWM channel at its neutral position."""
        try:
            chip_path = self.sysfs_root / f"pwmchip{self.config.pwm_chip}"
            pwm_path = chip_path / f"pwm{self.config.pwm_channel}"
            if not pwm_path.exists():
                (chip_path / "export").write_text(str(self.config.pwm_channel))
                self.sleep(0.2)

            self.pwm_path = pwm_path
            (pwm_path / "period").write_text("20000000")
            (pwm_path / "enable").write_text("1")
            self.set_angle(self.config.neutral_angle)

            print(
                f"✓ Trigger servo initialized on PWM chip "
                f"{self.config.pwm_chip} (Pin 15, GPIO12)"
            )
            print(
                f"  Neutral: {self.config.neutral_angle}°, "
                f"Trigger: {self.config.action_angle}°"
            )
        except Exception as exc:
            print(f"Failed to initialize trigger servo: {exc}")
            self.enabled = False
            self.pwm_path = None

    def set_angle(self, angle):
        """Set a clamped 0-180 degree trigger-servo angle."""
        if not self.enabled or self.pwm_path is None:
            return

        try:
            angle = max(0, min(180, angle))
            min_pulse = 500000
            max_pulse = 2500000
            pulse_width = int(
                min_pulse + (angle / 180.0) * (max_pulse - min_pulse)
            )
            (self.pwm_path / "duty_cycle").write_text(str(pulse_width))
            (self.pwm_path / "enable").write_text("1")
        except Exception as exc:
            print(f"Error setting trigger angle: {exc}")

    def fire(self):
        """Move to the action angle, hold, and return to neutral."""
        if not self.enabled:
            print("[TRIGGER SIMULATION] Would trigger servo")
            return

        print("Triggering action servo...")
        self.set_angle(self.config.action_angle)
        self.sleep(1)
        self.set_angle(self.config.neutral_angle)
        print("Trigger complete")

    def close(self):
        """Disable and unexport the PWM channel."""
        if not self.enabled or self.pwm_path is None:
            return

        try:
            (self.pwm_path / "enable").write_text("0")
            chip_path = self.sysfs_root / f"pwmchip{self.config.pwm_chip}"
            (chip_path / "unexport").write_text(str(self.config.pwm_channel))
        except Exception:
            pass
