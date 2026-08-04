"""Minimal MuJoCo aiming environment for the turret RL controller.

Deliberately dependency-light: plain ``mujoco`` bindings, no gymnasium — a
reset/step class with the conventional (obs, reward, done, info) contract so
any RL library (or hand-rolled loop) can drive it.

The action/observation contract mirrors ``RLControllerStub`` in
``ratbot/robot/controllers.py`` (see docs/math.md §7):

  action:      [yaw, pitch] in [-1, 1] -> desired angular velocity, scaled by
               MAX_VEL and integrated into position targets (same
               velocity-form pipeline as the real controller)
  observation: [yaw_err, pitch_err, yaw_vel, pitch_vel,
                target_yaw_vel, target_pitch_vel, confidence, age,
                prev_action_yaw, prev_action_pitch]

Rewards follow the named-terms pattern: aim error + control cost + smoothness,
each scaled, summed, multiplied by dt (shape sums stay comparable across
control rates).

Run the smoke test:  uv run --no-sync python sim/harness.py --test
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import mujoco

MODEL_PATH = Path(__file__).resolve().parent / "turret.xml"

SIM_DT = 0.005          # physics step (matches turret.xml)
CTRL_DT = 0.05          # control step -> 20 Hz, same as the real rig
MAX_VEL = 2.5           # rad/s, ballpark of 1240 raw/s through 8.33 raw/deg
REWARD_SCALES = {
    "aim": 1.0,          # negative angular error magnitude
    "control": 0.05,     # penalize large actions
    "smooth": 0.05,      # penalize action changes (jerk)
}


class TurretAimEnv:
    """Aim the turret at a (possibly moving) target point."""

    def __init__(self, episode_seconds=8.0, target_speed=0.3, seed=None):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.n_frames = int(CTRL_DT / SIM_DT)
        self.max_steps = int(episode_seconds / CTRL_DT)
        self.target_speed = float(target_speed)
        self.rng = np.random.default_rng(seed)

        self.yaw_id = self.model.joint("yaw").id
        self.pitch_id = self.model.joint("pitch").id
        self.target_mocap = self.model.body("target").mocapid[0]
        self.muzzle_site = self.model.site("muzzle").id

        self.prev_action = np.zeros(2)
        self.cmd = np.zeros(2)          # integrated position targets (rad)
        self.steps = 0
        self._target_angvel = np.zeros(2)

    # ------------------------------------------------------------------ helpers
    def _target_angles(self):
        """Yaw/pitch angles that would point the barrel at the target."""
        tx, ty, tz = self.data.mocap_pos[self.target_mocap]
        # pivot of the pitch stage sits ~0.26 m above the floor at the origin
        dx, dy, dz = tx, ty, tz - 0.26
        yaw = math.atan2(dy, dx)
        pitch = math.atan2(dz, math.hypot(dx, dy))
        return np.array([yaw, pitch])

    def _observation(self):
        target = self._target_angles()
        q = np.array([self.data.qpos[self.yaw_id], self.data.qpos[self.pitch_id]])
        qvel = np.array([self.data.qvel[self.yaw_id], self.data.qvel[self.pitch_id]])
        err = target - q
        return np.concatenate([
            err, qvel, self._target_angvel, [1.0, 0.0], self.prev_action,
        ]).astype(np.float32)

    # ------------------------------------------------------------------ API
    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        # random target on an arc in front of the turret
        r = self.rng.uniform(0.6, 1.4)
        yaw = self.rng.uniform(-1.0, 1.0)
        z = self.rng.uniform(0.15, 0.55)
        self.data.mocap_pos[self.target_mocap] = [r * math.cos(yaw), r * math.sin(yaw), z]
        self._target_angvel = self.rng.uniform(-1, 1, 2) * self.target_speed
        self.prev_action[:] = 0.0
        self.cmd = self._joint_angles()
        self.steps = 0
        mujoco.mj_forward(self.model, self.data)
        return self._observation()

    def _joint_angles(self):
        return np.array([self.data.qpos[self.yaw_id], self.data.qpos[self.pitch_id]])

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)

        # velocity-form: action is desired angular velocity, integrated into
        # clamped position targets (mirrors VelocityFormController)
        self.cmd = self.cmd + action * MAX_VEL * CTRL_DT
        lo, hi = self.model.jnt_range[self.yaw_id], self.model.jnt_range[self.pitch_id]
        self.cmd[0] = np.clip(self.cmd[0], lo[0], lo[1])
        self.cmd[1] = np.clip(self.cmd[1], hi[0], hi[1])
        self.data.ctrl[:] = self.cmd

        # drift the target (simple constant angular velocity on its arc)
        t = self.data.mocap_pos[self.target_mocap].copy()
        r = math.hypot(t[0], t[1])
        yaw = math.atan2(t[1], t[0]) + self._target_angvel[0] * CTRL_DT
        z = float(np.clip(t[2] + self._target_angvel[1] * CTRL_DT * 0.3, 0.1, 0.6))
        self.data.mocap_pos[self.target_mocap] = [r * math.cos(yaw), r * math.sin(yaw), z]

        for _ in range(self.n_frames):
            mujoco.mj_step(self.model, self.data)

        obs = self._observation()
        err = obs[:2]
        rewards = {
            "aim": -float(np.linalg.norm(err)),
            "control": -float(np.square(action).sum()),
            "smooth": -float(np.square(action - self.prev_action).sum()),
        }
        reward = sum(REWARD_SCALES[k] * v for k, v in rewards.items()) * CTRL_DT
        self.prev_action = action
        self.steps += 1
        done = self.steps >= self.max_steps
        info = {f"reward/{k}": v for k, v in rewards.items()}
        info["aim_error_rad"] = float(np.linalg.norm(err))
        return obs, reward, done, info
