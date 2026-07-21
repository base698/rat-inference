# Ratbot Tracking TODO

## 2026-07-21 world tracking and laser alignment handoff

Current deployed state:

- Branch `main` is deployed on the Jetson at commit `c7dd544`.
- Live command used for the final can/world-tracking test:

```bash
./run.sh rt200 --model yolo11n.pt --confidence 0.30 --device 0
```

- The runtime is using CUDA (`inference_runtime_device: cuda:0`) and 960x720 CSI stereo frames.
- Current crosshair config is `tracking.crosshair.x_base: 439` and `y_base: 469`.
- At the last checked pose (`yaw=2199`, `pitch=247`) the API reported computed crosshair `441,441`.
- Docker cleanup was run on the Jetson with `docker system prune -af`, reclaiming about `47.95GB`. Active containers survived: `rat-inference` and `ratbot-node-exporter`.

Laser/crosshair status:

- At level, the laser and crosshair are effectively dead on after the final `2px` down and `2px` right adjustment.
- When pitching up, the apparent alignment error grows to roughly `35px` high. The error is smaller when aimed down.
- Do not keep tuning the global crosshair offset for this. The level pose is now right, so the remaining problem is likely pitch-dependent hardware geometry: laser/camera non-parallelism, mount flex, or pitch-axis parallax.
- Next software step is to gather `pitch raw`, `laser pixel error`, and `distance` at several pitch positions, then tune `tracking.pitch_compensation.points` or add a laser pitch-angle bias for world aiming.

Can detection status:

- The can was visible in the live frame on the chair/table area, right of center.
- Live logs showed `valid_3d=0/0`, meaning no accepted detector boxes at the current threshold.
- A one-off raw YOLO pass on the exact frame with `conf=0.05` found:

```text
chair 0.445
couch 0.431
cup 0.060 at the can location
```

- This explains the dropout: COCO YOLO does not have a can class, and this view only matched the can as `cup` at about `6%`, far below the live `0.30` confidence threshold.
- For tomorrow's can-only debugging, either restart with a very low confidence such as `0.05` and only `cup`/`bottle` targets, use a more COCO-like bottle/cup object, or add can examples to the custom model.

## 2026-07-18 live can tracking notes

Test command:

```bash
./run.sh rt200 --model yolo11n.pt --target-class bottle --target-class cup --confidence 0.35 --device 0
```

Current tuning state:

- Servo control loop is healthy at about `59 FPS` against the `60 FPS` target.
- Inference reaches about `19.9 FPS` after warmup, but dips during raw YOLO detection bursts. The TensorRT engine should remain the preferred rat-tracking path.
- The new far-jump confidence gate works for some false jumps. Examples from the log: detections `#73`, `#74`, `#78`, and `#79` were ignored as low-confidence jumps below `belief_reseed_min_confidence: 0.55`.

Observed failure:

- The robot still had one hard upward pitch command while the target/can was low in the frame.
- The clearest sequence was detections `#145` through `#148`.
- Before the snap, belief was low/down around pitch raw `427-441`.
- Detection `#145` reported `obs=(2005, 226)` and was held as `pending jump 201`.
- Detection `#146` reported `obs=(2011, 33)` and was held as `pending jump 402`.
- Detection `#147` reported `obs=(2026, 53)` with confidence `0.64` and became `reseed=jump 389 confirmed`.
- After that reseed, the control loop repeatedly commanded pitch raw downward numerically, e.g. `440->428`, `428->416`, `416->404`, continuing toward the new belief around `53`. Since lower raw pitch means physical up, this matches the visible hard-up snap that pushed the can out of view.

Next tuning TODO:

- Make jump reseeding stricter when there is a recent active belief, especially on pitch.
- Candidate: keep normal nearby belief updates unchanged, but require `3` matching far-jump detections or `confidence >= 0.70` before accepting a far-jump reseed from an active belief.
- Candidate: split stale/weak reacquisition from active-track reseeding. Stale/weak can reacquire more freely, but active-track far jumps should need stronger evidence.
- Candidate: add a pitch-specific jump gate so a large upward pitch reseed cannot override a recent downward/low belief unless detections are high confidence and consistent.
- Candidate: if the active belief decays near `belief_min_confidence`, clear/coast instead of accepting low-confidence stale/weak reseeds that can command a large pitch move.
