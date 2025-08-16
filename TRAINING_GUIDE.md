# YOLOv8 Rat Detection Training Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
# or with uv:
uv pip install -e .
```

### 2. Train the Model

**Basic training with default settings:**
```bash
python train.py
```

**Recommended settings for better performance:**
```bash
python train.py --model-size m --epochs 150 --batch 32 --imgsz 640
```

**Train on CPU (slower but works without GPU):**
```bash
python train.py --device cpu --batch 8
```

**Resume training from checkpoint:**
```bash
python train.py --resume runs/train/rat_detector/weights/last.pt
```

### 3. Run Inference

**With trained model on image:**
```bash
python inference.py --input test_image.jpg --model runs/train/rat_detector/weights/best.pt --show --save
```

**With trained model on video:**
```bash
python inference.py --input test_video.mp4 --model runs/train/rat_detector/weights/best.pt --save --output detected_video.mp4
```

**Lower confidence threshold for more detections:**
```bash
python inference.py --input image.jpg --model runs/train/rat_detector/weights/best.pt --conf 0.1 --show
```

## Model Sizes

Choose based on your needs:
- `n` (nano): Fastest, least accurate, ~3MB
- `s` (small): Fast, good accuracy, ~11MB  
- `m` (medium): Balanced, ~25MB [RECOMMENDED]
- `l` (large): Slower, better accuracy, ~43MB
- `x` (extra-large): Slowest, best accuracy, ~68MB

## Training Tips

1. **Start with medium model** (`--model-size m`) for good balance
2. **Use at least 100 epochs** for decent results
3. **Enable caching** (`--cache True`) if you have enough RAM
4. **Monitor training**: Check `runs/train/rat_detector/` for plots and metrics
5. **Early stopping**: Training will stop early if no improvement for 50 epochs

## Common Issues

**Out of memory:**
- Reduce batch size: `--batch 8` or `--batch 4`
- Use smaller image size: `--imgsz 416`
- Use smaller model: `--model-size n`

**Slow training:**
- Enable AMP: `--amp True` (default)
- Use GPU if available
- Cache images: `--cache True`

**Poor detection results:**
- Train longer: `--epochs 200`
- Try larger model: `--model-size l`
- Adjust augmentations
- Check your dataset labels

## Dataset Structure

Your dataset is already properly structured:
```
datasets/rat/
├── rat_dataset.yaml     # Configuration file
├── images/
│   ├── train/           # Training images
│   └── val/             # Validation images
└── labels/
    ├── train/           # Training labels (YOLO format)
    └── val/             # Validation labels
```

## Monitoring Training

Training creates these outputs in `runs/train/rat_detector/`:
- `weights/best.pt` - Best model weights
- `weights/last.pt` - Latest checkpoint
- `results.csv` - Training metrics
- `confusion_matrix.png` - Performance visualization
- `train_batch*.jpg` - Training batch samples
- `val_batch*.jpg` - Validation batch samples