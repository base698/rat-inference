# Turret MuJoCo sim (RL harness)

Minimal, out-of-the-way simulation of the yaw/pitch turret for training the
RL controller (`ratbot/robot/controllers.py:RLControllerStub`). Nothing here
is imported by the runtime.

- `turret.xml` - MJCF: yaw+pitch hinges with position actuators, the real
  `turret-with-jetson.stl` as the rotating head (mm -> m via scale 0.001),
  and a mocap "can" target.
- `turret_env.py` - `TurretAimEnv`: reset/step env whose action space
  ([-1,1]^2 desired velocities, integrated into clamped position targets)
  and 10-float observation mirror `RLControllerStub`, so a policy trained
  here drops onto the hardware seam unchanged. Named reward terms
  (aim/control/smooth) scaled and summed x dt.
- `harness.py` - run it:

```bash
uv run --no-sync python sim/harness.py --test   # headless smoke test (CI-safe)
uv run --no-sync mjpython sim/harness.py        # interactive viewer (macOS: mjpython required)
```

Install deps: `uv pip install "mujoco>=3.9"` (or `uv sync --extra sim`).
macOS viewer note: `launch_passive` must run under `mjpython`; if it fails to
find libpython, symlink the dylib into the venv's lib dir (see
`workspaces/mujoco-tests/setup-macos-viewer.sh` for the idempotent fix).

Baseline: the built-in proportional policy converges to <0.05 rad on a
moving target - a trained policy should beat it on the smoothness terms.
