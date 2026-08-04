#!/usr/bin/env python3
"""Turret sim harness: headless smoke test or interactive viewer.

Headless (works everywhere, CI-safe):
    uv run --no-sync python sim/harness.py --test

Viewer (macOS needs mjpython — see sim/README.md):
    uv run --no-sync mjpython sim/harness.py

The built-in controller is a proportional aim policy driving the same
velocity-form action space the RL policy will use — it doubles as a sanity
baseline: any trained policy should at least beat it.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.turret_env import CTRL_DT, TurretAimEnv  # noqa: E402


def proportional_policy(obs):
    """Baseline: desired velocity proportional to angular error."""
    err = obs[:2]
    return np.clip(err * 3.0, -1.0, 1.0)


def run_test():
    env = TurretAimEnv(episode_seconds=6.0, target_speed=0.25, seed=7)
    obs = env.reset()
    initial_error = float(np.linalg.norm(obs[:2]))
    total_reward = 0.0
    done = False
    while not done:
        obs, reward, done, info = env.step(proportional_policy(obs))
        total_reward += reward
    final_error = info["aim_error_rad"]
    print(f"initial aim error: {initial_error:.3f} rad")
    print(f"final aim error:   {final_error:.3f} rad (moving target)")
    print(f"episode reward:    {total_reward:.3f}")
    ok = final_error < 0.15 and final_error < initial_error
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def run_viewer():
    import mujoco
    import mujoco.viewer

    env = TurretAimEnv(episode_seconds=1e9, target_speed=0.3, seed=None)
    obs = env.reset()
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            obs, _, _, _ = env.step(proportional_policy(obs))
            viewer.sync()
            # throttle to real time or the whole run flashes by
            sleep = CTRL_DT - (time.time() - step_start)
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    sys.exit(run_test() if "--test" in sys.argv else run_viewer())
