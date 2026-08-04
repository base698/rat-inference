"""Smoke tests for the MuJoCo turret env (skipped when mujoco is absent)."""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from sim.turret_env import TurretAimEnv  # noqa: E402


def test_reset_and_observation_shape():
    env = TurretAimEnv(seed=1)
    obs = env.reset()
    assert obs.shape == (10,)
    assert np.isfinite(obs).all()


def test_proportional_policy_converges_on_static_target():
    env = TurretAimEnv(episode_seconds=6.0, target_speed=0.0, seed=3)
    obs = env.reset()
    initial = float(np.linalg.norm(obs[:2]))
    done = False
    while not done:
        action = np.clip(obs[:2] * 3.0, -1, 1)
        obs, _, done, info = env.step(action)
    assert info["aim_error_rad"] < 0.05
    assert info["aim_error_rad"] < initial


def test_actions_are_clamped_to_joint_limits():
    env = TurretAimEnv(episode_seconds=2.0, seed=5)
    env.reset()
    for _ in range(env.max_steps):
        _, _, done, _ = env.step([1.0, 1.0])
        if done:
            break
    yaw = env.data.qpos[env.yaw_id]
    assert yaw <= env.model.jnt_range[env.yaw_id][1] + 1e-3
