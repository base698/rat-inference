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
  --baseline-override 57.5 \
  --port /dev/ttyACM0

# With larger inference size (1024px) - slower but more accurate
uv run --extra jetson python rt_200.py \
  --enable-camera \
  --use-csi \
  --stereo \
  --calibration tools/vision/calibration/output_recal/stereo_calibration.npz \
  --baseline-override 57.5 \
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
  --baseline-override 57.5 \
  --port /dev/ttyACM0 \
  --model runs/yolo11n-2025-10-23/weights/best.engine \
  --confidence 0.60 \
  --inference-fps 14 \
  --tracking-smoothing 0.45 \
  --max-yaw-step 45 \
  --max-pitch-step 28
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
  --baseline-override 57.5 \
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
  --baseline-override 57.5

# Simple video-only CSI stream, no servos, no detection
uv run python rt_200.py --video-only
```

The measured physical distance between stereo lens centers is `57.5mm`; pass it with `--baseline-override 57.5` so depth scale uses the real baseline instead of the solved calibration baseline.

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
- Keep `--inference-fps 14` as an upper cap for now. If the tracker cannot hold that rate with depth and servo control enabled, reduce depth frequency or the inference target before increasing image size.
- `imgsz=1024` is not recommended for live tracking until the 640px path is faster.

**Tracking Smoothness:**
- `--tracking-smoothing` low-pass filters the detected center before servo control. Lower values are smoother but laggier; `0.45` is the current default.
- `--max-yaw-step` and `--max-pitch-step` cap per-inference servo moves in raw units. Lower values reduce jerk but slow convergence.
- The current PID gains are intentionally more aggressive than the original values; smoothing and step caps keep that from snapping too hard between detections.

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
Measured baseline override: 57.5mm
```

Stereo RMS is usable for testing but still high; a rigid printed target and more varied poses should improve it.

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
python rt_200.py --imgsz 640 --inference-fps 14   # Current real-time tracking target
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
  --baseline-override 57.5 \
  --model runs/yolo11n-2025-10-23/weights/best.engine \
  --confidence 0.60 \
  --imgsz 640 \
  --inference-fps 14

# Access web interface at http://<jetson-ip>:8000
```

## Model Performance

### Jetson Nano 8GB

| Image Size | FPS | Accuracy | Use Case |
|------------|-----|----------|----------|
| 640px | ~1.3-1.5 observed, 14 target cap | Good | Current live tracking |
| 800px | Lower | Better | Offline testing |
| 1024px | Much lower | Best | Offline/high-accuracy testing |

**Recommendation:** Use 640px at `--inference-fps 14` for now, and optimize the inference pipeline before raising image size.

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
