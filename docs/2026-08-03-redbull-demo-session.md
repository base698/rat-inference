# 2026-08-03 — Red Bull demo build session

Everything that changed getting the turret demo-ready for the You Dirty Rat
talk. Reminder doc — the full detail is in the git log for today.

## What exists now

- **Model**: `runs/redbull/weights/best.pt` (+ `best.engine`, TensorRT fp16 @960,
  built on the Orin). yolo11n trained on `datasets/redbull` (150 imgs:
  83 rig captures / 26 scraped / 41 nano-banana synthetic). mAP50 0.995.
  **The class is `item`** (single-cls training renames it) — filter with
  `--target-class item`.
- **Dataset pipeline** (`datasets/redbull/README.md` for the workflow):
  collect_server.py (jetson:8010, clean frames via new `/raw-frame`),
  sync_captures.sh, autolabel_redbull.py (YOLO-World), review_server.py
  (localhost:8020, y/n), augment_backgrounds.py (Vertex nano banana; venue +
  `--ref-image` modes), audit_sheets.py.
- **Launcher**: `tools/jetson-ratbot` (installed at `~/bin/ratbot` on the
  Jetson). Mounts config.yaml and static/ into the container — tuning and UI
  edits need no rebuild. Autostart via docker `unless-stopped`; collect page
  via user crontab `@reboot`.

## Tracking fixes (in order discovered)

1. Reseed gates 160/120 -> 420/300: lateral motion stuttered through
   far-jump confirmations.
2. World-frame mode self-oscillates on stationary targets (stale-pose
   reprojection drags the track with the pan). Demo runs
   `--disable-world-tracking` (angular mode) + `world.actuation_enabled: false`.
3. Angular observations were anchored to the live servo pose; now anchored to
   a 30Hz pose history at `measurement_time - RATBOT_CAMERA_LATENCY_S`
   (0.05s, env-tunable in the launcher).
4. PID kd amplified commanded-vs-physical error sign flips into max-step
   square waves — kd is 0; P-only (kp 0.40 yaw / 0.38 pitch) is stable.
5. Final motion tuning: max step 62/38 per 20Hz tick, obs alpha 0.55,
   velocity decay 0.90 cap 400, deadband 8, inference 20fps (TensorRT).

## Demo day

- `ratbot start` after power-on (it was stopped on 08-03 — unless-stopped
  will NOT revive a stopped container).
- Offline venue plan: GL.iNet router, both laptop+jetson on it,
  `http://ubuntu.local:8000` (mDNS; no internet required). Pre-join the SSID
  and do one reboot test at home first (sudo at console).
- If the demo can sits >2m out: capture ~20 frames at that distance and
  retrain — the current model breathes below ~0.8 conf out there.
