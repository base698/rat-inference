"""Tracking-controller registry: drop-in controllers selected by configuration.

Every controller drives the same seam: it is constructed by a registered
builder, and must expose the loop protocol the tracker runtime relies on:

    track_once()        one control tick (read belief -> command servos)
    reset()             clear internal state (integrators, anchors)
    run() / start()     20Hz loop / daemon-thread starter
    control_fps         target loop rate
    last_actual_fps     measured loop rate (exported to /metrics)

Selection comes from ``tracking.controller`` in config.yaml ("angular",
"velocity", "rl", ...). New controllers register themselves here and become
available with zero changes to the tracker wiring:

    @register_controller("my-controller")
    def _build(robot, belief, bounds, control_fps, options):
        return MyController(...)

``options`` is the merged ``tracking.velocity_control``/controller options
mapping from config.yaml, so tuning stays a config edit + restart.
"""

from __future__ import annotations

from .belief import AngularBeliefController, VelocityFormController

_REGISTRY = {}


def register_controller(name):
    """Class-or-function decorator that adds a controller builder by name."""

    def decorator(builder):
        _REGISTRY[str(name)] = builder
        return builder

    return decorator


def available_controllers():
    return sorted(_REGISTRY)


def make_tracking_controller(name, *, robot, belief, bounds, control_fps, options=None):
    """Build the configured tracking controller.

    Raises ValueError (with the available names) for unknown controllers so a
    config typo fails loudly at startup instead of silently not tracking.
    """
    try:
        builder = _REGISTRY[str(name)]
    except KeyError:
        raise ValueError(
            f"unknown tracking controller {name!r}; "
            f"available: {', '.join(available_controllers())}"
        ) from None
    return builder(
        robot=robot,
        belief=belief,
        bounds=bounds,
        control_fps=control_fps,
        options=dict(options or {}),
    )


@register_controller("angular")
def _build_angular(robot, belief, bounds, control_fps, options):
    """Positional P(ID) controller stepping toward the belief each tick."""
    return AngularBeliefController(
        robot=robot,
        belief=belief,
        bounds=bounds,
        control_fps=control_fps,
        max_yaw_step=options.get("max_yaw_step", robot.max_yaw_step),
        max_pitch_step=options.get("max_pitch_step", robot.max_pitch_step),
        max_yaw_speed_raw_per_s=options.get(
            "max_yaw_speed_raw_per_s", robot.max_yaw_speed_raw_per_s
        ),
        max_pitch_speed_raw_per_s=options.get(
            "max_pitch_speed_raw_per_s", robot.max_pitch_speed_raw_per_s
        ),
        deadband_raw=options.get("deadband_raw", robot.belief_deadband_raw),
        min_step_raw=options.get("min_step_raw", robot.belief_min_step_raw),
    )


@register_controller("velocity")
def _build_velocity(robot, belief, bounds, control_fps, options):
    """Velocity-output controller integrated into position goals."""
    return VelocityFormController(
        robot=robot,
        belief=belief,
        bounds=bounds,
        control_fps=control_fps,
        kp_yaw=float(options.get("kp_yaw", 6.0)),
        kp_pitch=float(options.get("kp_pitch", 5.5)),
        max_yaw_velocity=float(
            options.get("max_yaw_velocity_raw_per_s", robot.max_yaw_step * control_fps)
        ),
        max_pitch_velocity=float(
            options.get("max_pitch_velocity_raw_per_s", robot.max_pitch_step * control_fps)
        ),
        max_accel=float(options.get("max_accel_raw_per_s2", 3500.0)),
        deadband_raw=options.get("deadband_raw", robot.belief_deadband_raw),
        damping_yaw=float(options.get("damping_yaw", 0.0)),
        damping_pitch=float(options.get("damping_pitch", 0.0)),
        reconcile_rate=float(options.get("reconcile_rate", 2.0)),
    )


class RLControllerStub(VelocityFormController):
    """Reinforcement-learning controller seam (NOT TRAINED YET).

    Inherits the velocity-form plumbing — integration into safe position
    goals, acceleration limiting, bounds clamping — and replaces only the
    velocity *decision* with a policy call. A trained policy plugs in by
    overriding :meth:`policy`; until then the stub commands zero velocity
    (the turret holds position), so selecting ``controller: rl`` in config
    is safe on hardware.

    Observation vector (matches the RL design notes in docs/math.md):
        [yaw_error, pitch_error,               # belief - commanded, raw units
         measured_yaw_vel, measured_pitch_vel, # Present_Velocity readback
         belief_yaw_vel, belief_pitch_vel,     # estimated target velocity
         confidence, belief_age,
         prev_action_yaw, prev_action_pitch]   # last commanded velocities

    Action: [yaw_velocity, pitch_velocity] normalized to [-1, 1], scaled by
    the controller's max velocities and then fed through the same
    accelerate/integrate/clamp pipeline as the classical controller.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_action = (0.0, 0.0)

    def build_observation(self, belief, measured_yaw_vel, measured_pitch_vel):
        yaw_error = belief["yaw"] - (self.cmd_yaw if self.cmd_yaw is not None else 0.0)
        pitch_error = belief["pitch"] - (self.cmd_pitch if self.cmd_pitch is not None else 0.0)
        return [
            yaw_error,
            pitch_error,
            measured_yaw_vel,
            measured_pitch_vel,
            float(belief.get("yaw_velocity", 0.0)),
            float(belief.get("pitch_velocity", 0.0)),
            float(belief.get("confidence", 0.0)),
            float(belief.get("age", 0.0)),
            self.prev_action[0],
            self.prev_action[1],
        ]

    def policy(self, observation):
        """Return (yaw_action, pitch_action) in [-1, 1]. Stub: hold still."""
        return 0.0, 0.0

    def _desired_velocities(self, belief, yaw_error, pitch_error,
                            measured_yaw_vel, measured_pitch_vel):
        obs = self.build_observation(belief, measured_yaw_vel, measured_pitch_vel)
        self.prev_action = tuple(
            max(-1.0, min(1.0, float(a))) for a in self.policy(obs)
        )
        return (
            self.prev_action[0] * self.max_yaw_velocity,
            self.prev_action[1] * self.max_pitch_velocity,
        )


@register_controller("rl")
def _build_rl(robot, belief, bounds, control_fps, options):
    return RLControllerStub(
        robot=robot,
        belief=belief,
        bounds=bounds,
        control_fps=control_fps,
        max_yaw_velocity=float(
            options.get("max_yaw_velocity_raw_per_s", robot.max_yaw_step * control_fps)
        ),
        max_pitch_velocity=float(
            options.get("max_pitch_velocity_raw_per_s", robot.max_pitch_step * control_fps)
        ),
        max_accel=float(options.get("max_accel_raw_per_s2", 3500.0)),
        deadband_raw=options.get("deadband_raw", robot.belief_deadband_raw),
    )
