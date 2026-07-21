# Ratbot Tracking TODO

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
