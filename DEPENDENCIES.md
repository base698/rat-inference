# Dependency Management Guide

This project uses **uv** with optional dependency groups to avoid conflicts between different use cases.

## Dependency Groups

### Core (always installed)
- ultralytics (YOLO)
- opencv-python
- numpy
- pillow
- supervision

### Optional Groups

#### `dataset` - For dataset generation
```bash
uv sync --extra dataset
```
Installs: `google-cloud-aiplatform`

**Use for:**
- Running `tools/dataset/generate-dataset.py`
- Generating training images with Vertex AI Gemini

#### `jetson` - For Jetson Nano deployment
```bash
uv sync --extra jetson
```
Installs: `lerobot`, `feetech-servo-sdk`, `fastapi`, `uvicorn`

**Use for:**
- Running `rt_200.py` on Jetson Nano
- Servo control and real-time tracking

#### `optimize` - For model optimization
```bash
uv sync --extra optimize
```
Installs: `ncnn`, `onnx`

**Use for:**
- Model conversion and optimization
- Exporting models to different formats

#### `all` - Everything except conflicting packages
```bash
uv sync --extra all
```
Installs all non-conflicting optional dependencies.

## Common Usage Patterns

### Dataset Generation (Development Machine)
```bash
# Install dependencies
uv sync --extra dataset

# Generate training images
uv run python tools/dataset/generate-dataset.py --reference ref.jpg --prompt "Add a rat in center" --dataset rat --validate
```

### Model Training (Development Machine)
```bash
# Install core dependencies only
uv sync

# Train model
uv run python tools/training/train.py --model-size n --epochs 100 --imgsz 640

# Run inference
uv run python tools/inference/inference.py --input test.jpg --model runs/best.pt --imgsz 640
```

### Jetson Nano Deployment
```bash
# Install jetson dependencies
uv sync --extra jetson

# Run real-time tracker
uv run python rt_200.py --enable-camera --use-csi --enable-trigger --imgsz 640
```

### Combined Development (Dataset + Training)
```bash
# Install core + dataset + optimize
uv sync --extra dataset --extra optimize

# Generate data
uv run python tools/dataset/generate-dataset.py --reference ref.jpg --prompts-file prompts.txt --dataset rat --count 10

# Train model
uv run python tools/training/train.py --imgsz 640
```

## Important Notes

### About main.py (Legacy)
`main.py` uses Roboflow's `inference` package which **conflicts** with `lerobot` (used in rt_200.py).

If you need to run `main.py`, create a separate virtual environment:
```bash
python -m venv venv-legacy
source venv-legacy/bin/activate
pip install inference supervision pillow
python main.py
```

### Dependency Conflicts
The following packages conflict and cannot be installed together:
- ❌ `inference` (Roboflow) + `lerobot` (Servo control)
  - Reason: Different `av` (PyAV) version requirements

This is why we use optional dependency groups!

## Troubleshooting

### Clear lock file and reinstall
```bash
rm uv.lock
uv sync --extra dataset
```

### Check installed packages
```bash
uv pip list
```

### Run without installing
```bash
uv run --extra dataset python tools/dataset/generate-dataset.py --help
```
