# Jetson Track Recording and Replay Deployment/Test Plan

## Purpose

Deploy and hardware-test the new world-frame track recording and replay workbench on the Jetson when it is powered on. The first Jetson run is a **shadow-mode diagnostics test**: compute and record tracks, but do not allow world-frame tracking to command the servos.

## Current development status

- Branch: `feature/08-world-frame-tracking`
- Base commit before this uncommitted feature work: `6b6ac59 feat: add stable multi-target turret tracking`
- The feature is currently **uncommitted and unpushed**, so it must be committed and pushed before the Jetson can pull it.
- Local verification completed:
  - 133 tests passed
  - 37 subtests passed
  - MyPy passed across 9 source files
  - Python compilation and JavaScript syntax passed
  - Browser start/stop, catalog, 2D/3D replay, filtering, speeds, and reprocessing passed
  - Multiple fail-closed independent reviews ended with no P0/P1 findings

## What changed

### Runtime recording

The world-frame tracker can now create one durable directory per recording:

```text
run_logs/tracks/<recording-id>/
  metadata.json
  observations.jsonl
```

Each frame records accepted world-frame measurements, bounding boxes, confidence/covariance data, assignments, selected track, and resulting track snapshots. Recording write failures stop the recording without stopping live inference.

### Main control UI

The existing main page now has:

- Start recording
- Stop and save
- Recording status/frame count
- Link to the replay workbench

### Replay workbench

`/tracks` provides:

- Recording catalog and session loading
- Play, pause, timeline scrubbing, and 0.1×–2× speeds
- Stable-track selection/filtering
- 2D measurement heatmap and current bounding boxes
- 3D fixed-frame tracks, measurements, trails, labels, and axes
- Reprocessing through the canonical tracker with adjustable association and lifecycle parameters

Replay and reprocessing do not call robot actuator interfaces.

### API compatibility and safety

- `/api/tracks/live` is the explicit live-track JSON endpoint.
- Historical JSON-oriented `GET /tracks` clients remain supported through content negotiation.
- Recording/catalog/load/reprocess routes are fail-closed unless `allow_remote_recording` is enabled.
- World-frame actuation remains separately gated by `actuation_enabled` and `calibration_validated`.
- Request bodies, recordings, aggregate disk use, replay measurements/tracks/work/runtime/output, and concurrent replay operations are bounded.

## Safety configuration for the first Jetson test

Use shadow mode. Do **not** enable world-frame actuation during recording/replay validation.

```yaml
tracking:
  world_frame:
    enabled: true
    actuation_enabled: false
    calibration_validated: false

    recordings_dir: run_logs/tracks

    # Enable only while the Jetson API is restricted to the trusted LAN.
    allow_remote_recording: true

    # Leave disabled during the recording/replay test.
    allow_remote_selection: false
```

Important: `enabled: true` is needed to produce world-frame observations. The two actuation gates must remain false until camera-to-pivot translation, mount rotation, servo signs, and stationary-target behavior are physically validated.

## Jetson deployment checklist

### 1. Before powering on the Jetson

- [ ] Commit the completed feature on `feature/08-world-frame-tracking`.
- [ ] Push the branch to the remote.
- [ ] Record the commit SHA to deploy.
- [ ] Confirm no credentials, local recordings, or temporary preview files are included.

### 2. Update the Jetson checkout

From the RAT repository on the Jetson:

```bash
git status --short --branch
git fetch origin
git checkout feature/08-world-frame-tracking
git pull --ff-only origin feature/08-world-frame-tracking
uv sync --extra jetson
```

Before overwriting Jetson-specific settings, preserve its current configuration:

```bash
cp config.yaml "config.yaml.pre-track-replay-$(date +%Y%m%d-%H%M%S)"
```

Reapply any Jetson-only model, serial-port, camera, calibration, and servo settings after updating.

### 3. Preflight checks

- [ ] TensorRT engine exists: `runs/yolo11n-2025-10-23/weights/best.engine`.
- [ ] Stereo calibration exists: `tools/vision/calibration/output_recal/stereo_calibration.npz`.
- [ ] CSI cameras enumerate and produce frames.
- [ ] Servo controller is available at the expected port, normally `/dev/ttyACM0`.
- [ ] `run_logs/tracks` is writable.
- [ ] Sufficient free disk is available.
- [ ] Jetson API is accessible only from the trusted LAN before enabling remote recording.
- [ ] `actuation_enabled: false` and `calibration_validated: false` are confirmed in the running config.

## Staged Jetson test procedure

### Stage A — Software smoke test without servo connection

Start the application without connecting to the servos:

```bash
uv run --extra jetson python rt_200.py \
  --no-connect \
  --enable-camera \
  --use-csi \
  --stereo \
  --calibration tools/vision/calibration/output_recal/stereo_calibration.npz \
  --baseline-override 51.1 \
  --model runs/yolo11n-2025-10-23/weights/best.engine \
  --confidence 0.70 \
  --inference-fps 20
```

Verify:

- [ ] Main page loads at `http://<jetson-ip>:8000`.
- [ ] Live video and detections appear.
- [ ] `/api/tracks/live` returns world-tracking JSON.
- [ ] Recording controls show ready when world tracking and remote recording are enabled.
- [ ] No servo connection or actuator command is attempted.

### Stage B — Record a short session in shadow mode

Use the main page to:

1. Start recording.
2. Record 60–120 seconds containing:
   - a stationary target;
   - one moving target;
   - two targets crossing if practical;
   - brief missed detections/occlusion;
   - targets near image edges;
   - foreground/background depth boundaries.
3. Stop and save.

Verify on disk:

```bash
find run_logs/tracks -maxdepth 2 -type f -print
```

Expected for the new recording:

```text
run_logs/tracks/<recording-id>/metadata.json
run_logs/tracks/<recording-id>/observations.jsonl
```

Check that the metadata reports a completed session and a nonzero frame count. Confirm the JSONL grows while recording and remains readable after stopping.

### Stage C — Replay workbench validation

Open:

```text
http://<jetson-ip>:8000/tracks
```

Verify:

- [ ] The new session appears in the catalog.
- [ ] Play, pause, timeline, and every speed option work.
- [ ] Repeated Play clicks do not create overlapping playback loops.
- [ ] 2D mode shows measurements/heatmap and current bounding boxes.
- [ ] 3D mode shows fixed-frame tracks, trails, measurements, labels, and axes.
- [ ] Selecting one track hides other tracks and their assigned measurements/counts.
- [ ] Reprocessing changes track behavior when confirmation, gate, miss/delete, or process-noise settings change.
- [ ] Reprocessing reports that no robot commands were sent.
- [ ] Browser console remains free of errors and the page does not overflow at the Jetson display/browser size.

### Stage D — Runtime/performance observations

Capture:

- Actual inference FPS after warmup.
- CPU/GPU utilization and memory use during live recording.
- Browser responsiveness while recording and replaying.
- Reprocessing duration for the short session.
- Recording size and frame count.
- Whether track IDs remain stable during motion, crossings, occlusion, and image-edge exits.
- Whether world coordinates remain stable for a stationary target while the camera/turret pose changes.

If 20 FPS is not sustained with stereo depth and recording enabled, lower `--inference-fps` before changing image size or safety settings.

### Stage E — Connected-hardware shadow test

Only after Stage A–D pass, run the normal connected tracker command while keeping world-frame actuation disabled:

```bash
uv run --extra jetson python rt_200.py \
  --enable-camera \
  --use-csi \
  --stereo \
  --calibration tools/vision/calibration/output_recal/stereo_calibration.npz \
  --baseline-override 51.1 \
  --port /dev/ttyACM0 \
  --model runs/yolo11n-2025-10-23/weights/best.engine \
  --confidence 0.70 \
  --inference-fps 20 \
  --tracking-control-fps 20 \
  --belief-update-alpha 0.75 \
  --max-yaw-step 45 \
  --max-pitch-step 45 \
  --pitch-tracking-scale 2.2 \
  --belief-reseed-distance-raw 160
```

This may continue exercising the existing angular tracker, so use the project’s normal physical safety precautions. The new world-frame pipeline must still remain shadow-only until its separate calibration/actuation gates are intentionally approved.

## Failure and rollback criteria

Stop the test if:

- World replay or reprocessing causes any servo/trigger command.
- Recording failure interrupts camera inference or the control API.
- The recording directory grows past documented bounds.
- The API becomes unresponsive during replay/reprocessing.
- Stationary targets move substantially in world coordinates as turret pose changes.
- Track selection silently jumps to another target after the selected track disappears.
- Servo direction, camera translation, or camera mount rotation is still uncertain.

Rollback application code:

```bash
git checkout <previous-known-good-branch-or-commit>
uv sync --extra jetson
```

Restore the saved Jetson configuration if needed:

```bash
cp config.yaml.pre-track-replay-<timestamp> config.yaml
```

Record logs and preserve any failed recording session before rollback so it can be replayed off-device.

## Resource limits to expect

Default protections include:

- 18,000 frames or 64 MiB per recording.
- 100 recording sessions maximum.
- 1 GiB aggregate recording quota.
- 512 MiB free-space reserve plus capacity for a full new recording.
- 64 KiB HTTP request-body cap.
- 8 replay measurements per frame.
- 32 replay tracks.
- Bounded association-work budget.
- 30-second reprocessing runtime limit.
- 64 MiB replay output/response limit.
- One concurrent load/reprocessing job.

When a limit is reached, the operation should fail with a bounded error rather than affecting live inference or actuators.

## Results to bring back after the Jetson test

- Deployed commit SHA.
- Exact command and config used.
- Jetson model/JetPack/CUDA/TensorRT versions.
- Recording ID(s), frame counts, durations, and sizes.
- Actual inference FPS and reprocessing time.
- Screenshots of the main recording controls plus 2D and 3D replay views.
- Browser console errors, if any.
- Notes on ID stability, depth confidence, world-coordinate stability, and crossing behavior.
- Any recording or metadata error message.
- Confirmation that no world-frame actuator command occurred during replay/shadow testing.
