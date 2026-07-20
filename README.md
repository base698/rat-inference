# Rat Inference System

**T1000 for rats** - An AI-powered rat detection and tracking system using YOLO, with automated dataset generation and real-time servo tracking capabilities.

## Features

🎯 **YOLO Inference** - Fast rat detection using Ultralytics YOLO (v8/v11)
📊 **Dataset Generation** - AI-powered dataset creation using Vertex AI Gemini
🎥 **Real-time Tracking** - Camera-based tracking with servo control (Jetson Nano)
🏋️ **Model Training** - Train custom YOLO models with configurable image sizes
🔧 **Flexible Deployment** - Modular dependency groups for different use cases

## Quick Start

### Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- Google Cloud credentials (for dataset generation)

### Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <your-repo-url>
cd rat-inference

# Install dependencies based on your use case:

# For dataset generation
uv sync --extra dataset

# For Jetson Nano deployment
uv sync --extra jetson

# For model training only
uv sync
```

## Usage

### 1. Dataset Generation

Generate training images from reference photos using Vertex AI Gemini:

```bash
# Single prompt with validation
uv run python tools/vision/dataset/generate-dataset.py \
  --reference ref.jpg \
  --prompt "Make a rat appear facing right in the middle of screen" \
  --dataset rat \
  --validate

# Multiple prompts from file
uv run python tools/vision/dataset/generate-dataset.py \
  --reference ref.jpg \
  --prompts-file tools/vision/sample-prompts.txt \
  --dataset rat \
  --count 10 \
  --validate

# Multiple custom prompts
uv run python tools/vision/dataset/generate-dataset.py \
  --reference ref.jpg \
  --prompt "Add a rat in the center" \
  --prompt "Add a rat on the left side" \
  --dataset rat \
  --count 5
```

**Output:** Generated images are saved to `./datasets/{name}/unsorted/`

**Validation:** When `--validate` is enabled, Gemini 2.5 Pro evaluates image realism before saving.

### 2. Model Training

Train a custom YOLO model with configurable image size:

```bash
# Train with default settings (640px)
uv run python tools/vision/training/train.py --model-size n --epochs 100

# Train with larger image size (1024px)
uv run python tools/vision/training/train.py --model-size n --epochs 100 --imgsz 1024

# Train with medium model
uv run python tools/vision/training/train.py --model-size m --epochs 150 --batch 16 --imgsz 640
```

**Training Tips:**
- Start with `imgsz=640` for faster training
- Use larger sizes (800-1024) for better accuracy
- Smaller models (n, s) train faster but may be less accurate
- See `TRAINING_GUIDE.md` for detailed instructions

### 3. Inference

Run inference on images or videos:

```bash
# Single image with default size (640)
uv run python tools/vision/inference/inference.py --input image.jpg --model runs/best.pt --show

# Video with larger inference size (1024)
uv run python tools/vision/inference/inference.py --input video.mp4 --model runs/best.pt --imgsz 1024 --save

# Custom confidence threshold
uv run python tools/vision/inference/inference.py --input image.jpg --model runs/best.pt --conf 0.5 --imgsz 640
```

**Parameters:**
- `--imgsz`: Inference image size in pixels (default: 640)
- `--conf`: Confidence threshold (default: 0.25)
- `--show`: Display results
- `--save`: Save annotated output

### 4. Real-time Tracking (Jetson Nano)

Run the real-time camera tracker with servo control:

```bash
# With camera and detection using config.yaml defaults
uv run --extra jetson python rt_200.py \
  --enable-camera \
  --use-csi \
  --stereo \
  --calibration tools/vision/calibration/output_recal/stereo_calibration.npz \
  --baseline-override 51.1 \
  --port /dev/ttyACM0

# With larger inference size (1024px) - slower but more accurate
uv run --extra jetson python rt_200.py \
  --enable-camera \
  --use-csi \
  --stereo \
  --calibration tools/vision/calibration/output_recal/stereo_calibration.npz \
  --baseline-override 51.1 \
  --port /dev/ttyACM0 \
  --imgsz 1024 \
  --inference-fps 4

# Web interface only (no servos)
uv run python rt_200.py --no-connect --enable-camera --use-csi
```

The custom rat models currently expose a single YOLO class named `item`, not `rat`. The TensorRT engine may report that same single class as `class0` if metadata is missing. By default, `config.yaml` sets `target_classes: [item, class0]`; omit `--target-class` to use the configured target, or pass `--target-class all` to accept every class from the loaded model.

Current preferred rat auto-tracking command:

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

#### Stereo Depth + Laser Tracking

Current working command for CSI stereo cameras, Feetech tracking servos, stereo depth, and laser/camera vertical compensation without detection:

```bash
uv run --extra jetson python rt_200.py \
  --enable-camera \
  --use-csi \
  --disable-detection \
  --stereo \
  --calibration tools/vision/calibration/output_recal/stereo_calibration.npz \
  --baseline-override 51.1 \
  --port /dev/ttyACM0
```

Useful variants:

```bash
# Camera-only stereo test, no servos
uv run --extra jetson python rt_200.py \
  --enable-camera \
  --use-csi \
  --disable-detection \
  --disable-servos \
  --no-connect \
  --stereo \
  --calibration tools/vision/calibration/output_recal/stereo_calibration.npz \
  --baseline-override 51.1

# Simple video-only CSI stream, no servos, no detection
uv run python rt_200.py --video-only
```

The measured physical distance between stereo lens centers is `57.5mm`. The current live depth scale uses an empirical effective baseline override of `51.1mm`, which matched a tape-measured `1.6m` target after the calibrated baseline overestimated distance.

`config.yaml` contains two crosshair alignment systems:

- `pitch_compensation.points`: fallback pitch-to-Y correction when stereo depth is unavailable.
- `depth_crosshair_compensation`: depth-based Y correction for the vertical camera-to-laser offset.

Current laser geometry:

```yaml
depth_crosshair_compensation:
  enabled: true
  laser_vertical_offset_mm: 55.0
  reference_distance_mm: 1000.0
  max_adjust_px: 80.0
```

Positive `laser_vertical_offset_mm` means the laser exits below the camera center. The correction moves the crosshair down for closer targets and up for farther targets, relative to the `reference_distance_mm`.

**Access web interface:** `http://localhost:8000`

**Performance Notes:**
- `--inference-fps` is a scheduling target, not guaranteed throughput.
- The tracker logs `Inference actual FPS` every few seconds. In live 640px stereo tracking tests, the current Python/Ultralytics path measured about `1.3-1.5` actual FPS despite a 14 FPS target.
- The Jetson TensorRT engine at `runs/yolo11n-2025-10-23/weights/best.engine` is the preferred live path. The engine was exported from the 10-23 model for FP16 inference; Ultralytics does not accept `quantize=16` for this export, so use `half=True` when regenerating it.
- The first TensorRT live tracker pass held about `13.9` actual FPS against a 14 FPS cap after warmup. The current tuning pass raises the cap to `20` FPS; if the tracker cannot hold that rate with depth and servo control enabled, reduce depth frequency or the inference target before increasing image size.
- `imgsz=1024` is not recommended for live tracking until the 640px path is faster.

**Tracking Smoothness:**
- Vision detections update an angular target belief in turret coordinates. The servo controller runs separately and moves toward that belief, so motion can continue smoothly between missed detections.
- `--belief-update-alpha` controls how quickly new detections move the angular belief. Higher follows observations faster; lower is smoother but laggier. Current test value: `0.75`. Non-jump stale/weak beliefs reset directly to the first new observation, while far jumps use the confirmation gate below.
- `--belief-reseed-distance-raw` marks a far-away observation as a possible target jump. The first jump is ignored while the existing belief continues; a second matching jump within `--belief-reseed-max-interval` confirms reacquisition. Current test values: `160` raw units, `2` confirmations, `120` raw-unit match distance, `0.8s` interval.
- `--belief-velocity-alpha`, `--belief-velocity-decay`, `--belief-max-velocity-raw-per-s`, and `--belief-max-prediction-age` add a bounded target-velocity estimate to the belief. This lets the controller continue briefly toward a target that was moving when detections drop at the frame edge. Current test values: `0.45`, `0.96`, `600 raw/s`, `0.45s`.
- Pitch is damped separately because vertical detections are noisier: `--belief-pitch-update-alpha` is currently `0.35`, `--belief-pitch-velocity-alpha` is `0.25`, `--belief-max-pitch-velocity-raw-per-s` is `180`, and `--max-pitch-step` is `35`.
- `--tracking-control-fps` controls how often the servo controller reads belief and commands the robot.
- `--max-yaw-step` and `--max-pitch-step` cap per-control-tick servo moves in raw units. Lower values reduce jerk but slow convergence.
- `--pitch-tracking-scale` multiplies vertical image error before converting it into pitch raw units. The RT-200 pitch servo uses `1` as full up and `500` as full down; the current test value `2.2` gives below-crosshair detections more downward authority.
- Pitch derivative gain is currently `0.00` so the pitch loop follows the belief without derivative kick from servo readback lag.
- `tracking.video_fps` is currently `30` so inference has fresh camera frames; `tracking.motor_readback_fps` is currently `10` so the video overlay does not poll servo position on every frame.
- `--belief-miss-decay`, `--belief-min-confidence`, and `--belief-max-age` decide how long the robot keeps moving toward a target when detections disappear.
- The web UI has a `Clear Belief` button, backed by the robot control API, that clears autonomous target belief and PID state without recentering the servos or clearing detection history.
- The web UI syncs the yaw/pitch sliders from live motor readback during status polling, unless you are actively dragging a slider or a manual position write is in flight.
- `tracking.depth_min_texture_std` rejects stereo depth samples on blank/low-texture regions instead of showing unstable but plausible-looking distances. Current test value: `2.5`; lower is more permissive.
- `tracking.depth_min_valid_mm` rejects near-depth stereo failures before they can produce false `0.3m` readings on the floor. Current test value: `500.0`.
- `tracking.depth_max_valid_mm` rejects far-depth spikes before they can move the depth-compensated crosshair. Current test value: `6000.0`.
- `tracking.depth_adjust_smoothing_alpha` and `tracking.depth_adjust_missing_decay` smooth the depth-based visual crosshair Y offset so the overlay does not jump on noisy stereo samples.

**Pitch Homing Offset:**
- Feetech protocol 0 exposes `Homing_Offset`; the library notes `Present_Position = Actual_Position - Homing_Offset`.
- Use `./run.sh pitch-offset` to read the current pitch offset and limits.
- Use `./run.sh pitch-offset --delta -100` for a dry run that shows how a 100 raw-unit shift would change the same physical position.
- Add `--apply` only when you intentionally want to write persistent motor calibration.

**Model Notes:**
- `runs/yolo11n-2025-10-23/weights/best.engine` is the current preferred live tracking model on the Jetson.
- `runs/yolo11n-2025-10-23/weights/best.pt` is the source model for retraining, comparison, or regenerating the engine.
- `runs/yolo11n-2025-10-24/weights/best.pt` is kept for comparison but performed worse in live testing.
- Both custom rat models expose their target class as `item`; the TensorRT engine can appear as `class0` when metadata is absent.

Regenerate the TensorRT engine on the Jetson with the CUDA-enabled Docker base:

```bash
docker run --rm --runtime=nvidia --ipc=host --network=host \
  -v /home/base698/rat-inference:/app -w /app \
  ultralytics/ultralytics:latest-jetson-jetpack6 \
  yolo export \
    model=runs/yolo11n-2025-10-23/weights/best.pt \
    format=engine \
    imgsz=640 \
    half=True \
    device=0
```

#### Stereo Recalibration

Capture stereo checkerboard pairs with the web helper. The helper only saves stereo pairs when the checkerboard is detected in both cameras.

```bash
uv run --extra jetson python tools/vision/calibration/capture_calibration.py \
  --web \
  --use-csi \
  --stereo \
  --pattern 6x4 \
  --output tools/vision/calibration/images_recal \
  --port 8010
```

Open `http://<jetson-ip>:8010`, capture 40-60 good stereo pairs, then calibrate. Use the measured square size for `--square-size`; the last screen-based calibration used `21mm`.

```bash
uv run --extra jetson python tools/vision/calibration/calibrate_camera.py \
  --stereo \
  --left "tools/vision/calibration/images_recal/left/*.jpg" \
  --right "tools/vision/calibration/images_recal/right/*.jpg" \
  --pattern 6x4 \
  --square-size 21 \
  --output tools/vision/calibration/output_recal
```

The current recalibration produced:

```text
Left RMS: 0.065px
Right RMS: 0.046px
Stereo RMS: 1.486px
Solved baseline: 42.61mm
Physical lens spacing: 57.5mm
Current effective baseline override: 51.1mm
```

Stereo RMS is usable for testing but still high; a rigid printed target and more varied poses should improve it.

## Stable Multi-Target World-Frame Tracking (Opt-In)

`--world-tracking` enables the new 3D tracker in **shadow mode**. It projects every
YOLO detection with valid stereo depth into a fixed frame attached to the turret
base, predicts independent constant-velocity Kalman tracks, and preserves stable
IDs through short occlusions. Shadow mode computes, overlays, exposes, and logs
tracks but does not connect them to servo actuation.

```bash
python3 rt_200.py --enable-camera --stereo --world-tracking
```

This mode is deliberately **disabled by default**, and world-frame actuation has
an additional fail-closed gate. It is enabled only when both of these values are
set after completing the hardware acceptance procedure:

```yaml
tracking:
  world_frame:
    enabled: true
    actuation_enabled: true
    calibration_validated: true
```

All `tracking.world_frame` Boolean switches—including these two and
`enabled`/`allow_remote_selection`—must be real YAML booleans (`true`/`false`,
without quotes). String, number, null, and collection values are rejected instead
of being coerced. The existing angular tracker remains the default when world
tracking is disabled.
When world actuation is enabled, the tracker starts with no selected target and
requires an explicit stable-ID selection before the controller can move.

### Coordinate Convention

- OpenCV camera frame: `+X` right, `+Y` down, `+Z` forward.
- Turret-base frame: `+x` forward at neutral, `+y` left, `+z` up.
- Positive yaw turns left; positive pitch points up.
- The frame is fixed only while the physical turret base is stationary. It is
  not a global/map frame and must be reset if the base is moved.

### Required Hardware Calibration

Before enabling servo motion with world tracking:

1. Confirm `tracking.world_frame.yaw_center_raw` and `pitch_center_raw` put the
   optical axis at the chosen neutral pose.
2. Confirm a raw yaw increase turns the optical axis left. If not, set
   `yaw_sign: -1.0`.
3. Confirm a raw pitch increase points down. If not, set `pitch_sign: 1.0`.
4. Measure raw-units-per-degree over several commanded angles and replace
   `yaw_raw_per_degree` and `pitch_raw_per_degree`.
5. Measure the vector from the yaw/pitch pivot to the left camera optical center
   in neutral turret coordinates and set `camera_translation_mm`.
6. Measure residual camera roll/pitch/yaw mounting error and set
   `camera_mount_rpy_degrees`.
7. Put one stationary target at a surveyed position. Pan and tilt without moving
   the base. Its reported `(x, y, z)` should remain stable. Do not enable
   autonomous motion until this passes.

The stereo calibration file is the source of truth for focal length and
baseline. Do not copy a baseline from notes: project records have referenced
both 51.1 mm and 57.5 mm rigs. Use `--baseline-override` only after measuring the
physical camera pair used in that run.

### Track API

- `GET /tracks` — list track ID, position, velocity, covariance, lifecycle,
  confidence, selection, and last bounding box.
- `POST /tracks/select` with `{"track_id": 3}` — explicitly select an ID.
- `POST /tracks/clear-selection` — stop autonomous aiming without deleting
  current tracks.
- `POST /clear-belief` — clear both legacy angular belief and all world tracks.

The two track-selection mutation routes are denied with `403` unless
`tracking.world_frame.allow_remote_selection: true`. They are unauthenticated,
so enable that option only when the API is bound to or firewalled within a trusted
interface/network. Track IDs must be JSON integers; booleans, strings, and
fractional numbers are rejected.

The `/status` payload also includes `world_tracking`, `selected_track_id`,
`tracks`, and per-frame association diagnostics. Set
`tracking.world_frame.log_path` to append replayable `ratbot.world_tracks.v1`
JSONL records containing pose, detections, assignments, selection, track state,
and covariance. Logging is off by default.

### Current Scope and Deferred Work

Implemented: batched disparity (one map per frame), per-point quality and
covariance, camera-to-base transforms, fixed-frame Kalman tracks,
maximum-cardinality/minimum-distance gated one-to-one association, lifecycle
management, explicit target selection, extrinsic-aware latency-compensated aim,
fail-closed shadow/actuation gates, bounded servo control, API status, and
structured snapshots.

Deferred deliberately:

- Optimized Hungarian/Jonker-Volgenant association and appearance embeddings for
  dense scenes. The current exact dynamic-programming assignment prioritizes
  cardinality then distance and is intended for a small number of physical targets.
- Spherical occupancy grids and information-gain scanning. A dense fixed
  occupancy volume is not useful until the tracking frame and base-motion model
  are proven on hardware.
- True capture/exposure timestamps and interpolated servo pose. The current
  implementation snapshots `time.monotonic()` immediately after acquisition and
  synchronously reads servo `Present_Position`; connected readback failures skip
  the world update rather than substituting a commanded goal. Hardware exposure
  timestamps plus timestamped pose-history interpolation remain the next upgrade.
- Global/world mapping while the robot base moves. That requires base odometry
  or SLAM and a new transform layer.

## Project Structure

```
rat-inference/
├── rt_200.py                         # Real-time tracking CLI/server for the Jetson
├── config.yaml                       # Active robot, camera, tracking, and aiming config
├── ratbot/
│   ├── robot/
│   │   ├── aiming.py                 # Crosshair, pitch, and depth aiming compensation
│   │   └── interfaces.py             # TrackerRobot protocol used by the web controller
│   ├── vision/
│   │   ├── csi_camera.py             # CSI camera capture helper
│   │   └── yolo_inference.py         # Shared YOLO inference helpers
│   └── web/
│       └── control_api.py            # FastAPI control and streaming routes
├── static/
│   ├── index.html                    # Browser control UI
│   └── worker.js                     # Browser-side worker for frame handling
├── tools/
│   ├── README.md                     # Tool command index
│   ├── hardware/
│   │   ├── find_motors.py            # Feetech motor discovery
│   │   ├── gpio_test.py              # GPIO/PWM trigger diagnostics
│   │   ├── pitch_test.py             # Pitch servo test helper
│   │   ├── servo_test_sysfs.py       # Legacy sysfs PWM servo test
│   │   └── trigger_position_test.py  # Trigger servo position helper
│   └── vision/
│       ├── calibration/
│       │   ├── capture_calibration.py # Capture mono/stereo checkerboard images
│       │   ├── calibrate_camera.py    # Solve mono/stereo camera calibration
│       │   ├── images_recal/          # Current tracked stereo calibration captures
│       │   └── output_recal/          # Current tracked calibration .npz outputs
│       ├── dataset/
│       │   ├── dataset_cleaner.py     # Review/remove bad dataset images
│       │   ├── generate-dataset.py    # Generate candidate training images
│       │   ├── labeler.py             # GUI YOLO labeling helper
│       │   └── extract-frames.sh      # Extract frames from videos
│       ├── inference/
│       │   └── inference.py           # Standalone image/video YOLO inference CLI
│       ├── training/
│       │   └── train.py               # YOLO training CLI
│       ├── legacy/
│       │   └── main.py                # Older Roboflow/image experiment
│       ├── assets/                    # Sample/reference images
│       └── sample-prompts.txt         # Example prompts for generation
├── docs/
│   └── repository-cleanup-phase-*.md  # Cleanup and modularization notes
├── archive/
│   └── raspberry-pi/
│       └── rt_100.py                 # Legacy Raspberry Pi tracking/trap script
├── datasets/                         # Training datasets
│   └── rat/
│       ├── images/{train,val}/       # Training and validation images
│       ├── labels/{train,val}/       # YOLO labels
│       └── unsorted/                 # Generated or pending images
├── runs/                             # Training outputs
│   ├── yolo11n-2025-10-23/             # Current preferred live tracking model
│   │   └── weights/
│   │       ├── best.pt
│   │       └── best.engine
│   └── yolo11n-2025-10-24/             # Comparison model
│       └── weights/
│           └── best.pt
├── Dockerfile
├── run.sh
└── pyproject.toml                    # Dependencies with optional groups

Key Files:
- DEPENDENCIES.md - Dependency group details and conflict notes
- TRAINING_GUIDE.md - Model training instructions
- STEREO_DEPTH_FIX_SUMMARY.md - Stereo/depth/laser alignment repair log
```

Root-level utility wrappers have been removed. Use the `tools/hardware/...` and `tools/vision/...` paths for utility commands.

## Dependency Management

This project uses **optional dependency groups** to avoid conflicts:

### Available Groups

| Group | Packages | Use For |
|-------|----------|---------|
| **core** (default) | ultralytics, opencv, numpy | Training, inference |
| **dataset** | google-cloud-aiplatform | Dataset generation |
| **jetson** | lerobot, servos, fastapi | Jetson deployment |
| **optimize** | ncnn, onnx | Model optimization |
| **all** | All non-conflicting deps | Combined development |

### Installation Examples

```bash
# Dataset generation workflow
uv sync --extra dataset

# Jetson deployment
uv sync --extra jetson

# Combined: dataset + optimization
uv sync --extra dataset --extra optimize

# Everything
uv sync --extra all
```

**Important:** `tools/vision/legacy/main.py` (Roboflow inference) conflicts with `lerobot` and must be run in a separate venv.

See `DEPENDENCIES.md` for complete details.

## Image Size Configuration

All inference and training scripts now support the `--imgsz` parameter:

```bash
# Training
python tools/vision/training/train.py --imgsz 640   # Default
python tools/vision/training/train.py --imgsz 1024  # Higher accuracy (requires retraining)

# Inference
python tools/vision/inference/inference.py --imgsz 640   # Default
python tools/vision/inference/inference.py --imgsz 1024  # Larger inference size

# Real-time tracking
python rt_200.py --imgsz 640 --inference-fps 20   # Current real-time tracking target
python rt_200.py --imgsz 1024  # Higher accuracy, ~2-4 FPS on Jetson
```

**Notes:**
- Models trained at 640px can run inference at different sizes (e.g., 1024px)
- For best accuracy at 1024px, retrain the model with `--imgsz 1024`
- Larger sizes increase GPU memory usage and reduce FPS

## Jetson Nano Setup

### Prerequisites

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install system packages
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-libcamera libcamera-apps
sudo usermod -aG video $USER   # Log out/in after this
```

### Setup Virtual Environment

```bash
cd ~/rat-inference
uv venv -p /usr/bin/python3.11 --system-site-packages
source .venv/bin/activate
```

### Install Dependencies

```bash
# For Jetson deployment (includes servo control)
uv sync --extra jetson

# Verify installation
python -c "import libcamera; print('libcamera OK')"
python -c "from ultralytics import YOLO; print('YOLO OK')"
python -c "from lerobot.motors.feetech import FeetechMotorsBus; print('Servos OK')"
```

### Running

```bash
# Start the tracking system
uv run python rt_200.py \
  --port /dev/ttyACM0 \
  --enable-camera \
  --use-csi \
  --stereo \
  --calibration tools/vision/calibration/output_recal/stereo_calibration.npz \
  --baseline-override 51.1 \
  --model runs/yolo11n-2025-10-23/weights/best.engine \
  --confidence 0.70 \
  --imgsz 640 \
  --inference-fps 20

# Access web interface at http://<jetson-ip>:8000
```

## Model Performance

### Jetson Nano 8GB

| Image Size | FPS | Accuracy | Use Case |
|------------|-----|----------|----------|
| 640px | ~1.3-1.5 observed, 14 target cap | Good | Current live tracking |
| 800px | Lower | Better | Offline testing |
| 1024px | Much lower | Best | Offline/high-accuracy testing |

**Recommendation:** Use 640px at `--inference-fps 20` with the TensorRT engine, and optimize the depth/control loop before raising image size.

## Configuration Files

- **pyproject.toml** - Dependencies and project metadata
- **datasets/rat/rat_dataset.yaml** - Dataset configuration for training
- **runs/*/args.yaml** - Training run configuration

## Troubleshooting

### Dependency Conflicts

```bash
# Clear lock file and reinstall
rm uv.lock
uv sync --extra dataset
```

### Check Installed Packages

```bash
uv pip list
```

### Vertex AI Authentication

```bash
# Option 1: gcloud CLI (development)
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

# Option 2: Service account (production)
export GOOGLE_APPLICATION_CREDENTIALS='/path/to/key.json'
export GCP_PROJECT_ID='your-project-id'
export GCP_LOCATION='us-central1'
```
