from ultralytics import YOLO
import argparse
import os
from pathlib import Path

def train_yolo(args):
    """
    Train a YOLOv8 model on the rat detection dataset
    """
    
    # Select model size
    model_name = f"yolov8{args.model_size}.pt"
    print(f"Using model: {model_name}")
    
    # Load a model
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming training from checkpoint: {args.resume}")
        model = YOLO(args.resume)
    else:
        print(f"Starting training from pretrained {model_name}")
        model = YOLO(model_name)  # Load pretrained model
    
    # Training arguments
    train_args = {
        "data": args.data,           # Dataset config path
        "epochs": args.epochs,        # Number of training epochs
        "imgsz": args.imgsz,         # Image size
        "batch": args.batch,         # Batch size
        "device": args.device,       # Device to use (0 for GPU, 'cpu' for CPU)
        "workers": args.workers,     # Number of data loader workers
        "optimizer": args.optimizer,  # Optimizer
        "lr0": args.lr0,             # Initial learning rate
        "lrf": args.lrf,             # Final learning rate factor
        "momentum": args.momentum,    # Momentum
        "weight_decay": args.weight_decay,  # Weight decay
        "warmup_epochs": args.warmup_epochs,  # Warmup epochs
        "warmup_momentum": args.warmup_momentum,  # Warmup momentum
        "warmup_bias_lr": args.warmup_bias_lr,   # Warmup bias learning rate
        "box": args.box,             # Box loss weight
        "cls": args.cls,             # Classification loss weight
        "dfl": args.dfl,             # DFL loss weight
        "patience": args.patience,    # Early stopping patience
        "save": True,                # Save checkpoints
        "save_period": args.save_period,  # Save checkpoint every n epochs
        "cache": args.cache,         # Cache images for faster training
        "augment": args.augment,     # Apply augmentations
        "degrees": args.degrees,     # Rotation augmentation
        "translate": args.translate, # Translation augmentation
        "scale": args.scale,         # Scale augmentation
        "shear": args.shear,         # Shear augmentation
        "perspective": args.perspective,  # Perspective augmentation
        "flipud": args.flipud,       # Flip up-down augmentation
        "fliplr": args.fliplr,       # Flip left-right augmentation
        "mosaic": args.mosaic,       # Mosaic augmentation
        "mixup": args.mixup,         # Mixup augmentation
        "copy_paste": args.copy_paste,  # Copy-paste augmentation
        "conf": args.conf,           # Confidence threshold for validation
        "iou": args.iou,             # IoU threshold for NMS
        "project": args.project,     # Project name
        "name": args.name,           # Experiment name
        "exist_ok": args.exist_ok,   # Overwrite existing experiment
        "pretrained": args.pretrained,  # Use pretrained weights
        "verbose": args.verbose,     # Verbose output
        "seed": args.seed,           # Random seed
        "deterministic": args.deterministic,  # Deterministic training
        "single_cls": args.single_cls,  # Single class training
        "rect": args.rect,           # Rectangular training
        "cos_lr": args.cos_lr,       # Cosine learning rate scheduler
        "close_mosaic": args.close_mosaic,  # Disable mosaic for final epochs
        "amp": args.amp,             # Automatic Mixed Precision training
        "plots": args.plots,         # Generate plots
    }
    
    # Remove None values to use defaults
    train_args = {k: v for k, v in train_args.items() if v is not None}
    
    # Train the model
    results = model.train(**train_args)
    
    # Evaluate model performance on validation set
    if args.validate:
        print("\nEvaluating model on validation set...")
        metrics = model.val()
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
    
    # Export model if requested
    if args.export_format:
        print(f"\nExporting model to {args.export_format} format...")
        model.export(format=args.export_format)
    
    print(f"\nTraining complete! Results saved to {results.save_dir}")
    return model, results

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for rat detection")
    
    # Model configuration
    parser.add_argument("--model-size", type=str, default="n", 
                       choices=["n", "s", "m", "l", "x"],
                       help="Model size: n(nano), s(small), m(medium), l(large), x(extra-large)")
    parser.add_argument("--data", type=str, default="datasets/rat/rat_dataset.yaml",
                       help="Path to dataset config file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume training")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="Image size (pixels)")
    parser.add_argument("--device", type=str, default="0",
                       help="Device to use (0 for GPU, 'cpu' for CPU)")
    parser.add_argument("--workers", type=int, default=8,
                       help="Number of data loader workers")
    
    # Optimizer settings
    parser.add_argument("--optimizer", type=str, default="SGD",
                       choices=["SGD", "Adam", "AdamW", "RMSProp"],
                       help="Optimizer")
    parser.add_argument("--lr0", type=float, default=0.01,
                       help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01,
                       help="Final learning rate factor")
    parser.add_argument("--momentum", type=float, default=0.937,
                       help="Momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0005,
                       help="Weight decay")
    
    # Warmup settings
    parser.add_argument("--warmup-epochs", type=float, default=3.0,
                       help="Warmup epochs")
    parser.add_argument("--warmup-momentum", type=float, default=0.8,
                       help="Warmup momentum")
    parser.add_argument("--warmup-bias-lr", type=float, default=0.1,
                       help="Warmup bias learning rate")
    
    # Loss weights
    parser.add_argument("--box", type=float, default=7.5,
                       help="Box loss weight")
    parser.add_argument("--cls", type=float, default=0.5,
                       help="Classification loss weight")
    parser.add_argument("--dfl", type=float, default=1.5,
                       help="DFL loss weight")
    
    # Augmentation settings
    parser.add_argument("--augment", type=bool, default=True,
                       help="Apply augmentations")
    parser.add_argument("--degrees", type=float, default=0.0,
                       help="Rotation augmentation (degrees)")
    parser.add_argument("--translate", type=float, default=0.1,
                       help="Translation augmentation")
    parser.add_argument("--scale", type=float, default=0.5,
                       help="Scale augmentation")
    parser.add_argument("--shear", type=float, default=0.0,
                       help="Shear augmentation (degrees)")
    parser.add_argument("--perspective", type=float, default=0.0,
                       help="Perspective augmentation")
    parser.add_argument("--flipud", type=float, default=0.0,
                       help="Flip up-down augmentation probability")
    parser.add_argument("--fliplr", type=float, default=0.5,
                       help="Flip left-right augmentation probability")
    parser.add_argument("--mosaic", type=float, default=1.0,
                       help="Mosaic augmentation probability")
    parser.add_argument("--mixup", type=float, default=0.0,
                       help="Mixup augmentation probability")
    parser.add_argument("--copy-paste", type=float, default=0.0,
                       help="Copy-paste augmentation probability")
    
    # Validation settings
    parser.add_argument("--conf", type=float, default=0.001,
                       help="Confidence threshold for validation")
    parser.add_argument("--iou", type=float, default=0.7,
                       help="IoU threshold for NMS")
    
    # Training options
    parser.add_argument("--patience", type=int, default=50,
                       help="Early stopping patience")
    parser.add_argument("--save-period", type=int, default=-1,
                       help="Save checkpoint every n epochs")
    parser.add_argument("--cache", type=bool, default=False,
                       help="Cache images for faster training")
    parser.add_argument("--pretrained", type=bool, default=True,
                       help="Use pretrained weights")
    parser.add_argument("--single-cls", type=bool, default=True,
                       help="Single class training")
    parser.add_argument("--rect", type=bool, default=False,
                       help="Rectangular training")
    parser.add_argument("--cos-lr", type=bool, default=False,
                       help="Use cosine learning rate scheduler")
    parser.add_argument("--close-mosaic", type=int, default=10,
                       help="Disable mosaic for final n epochs")
    parser.add_argument("--amp", type=bool, default=True,
                       help="Automatic Mixed Precision training")
    parser.add_argument("--plots", type=bool, default=True,
                       help="Generate plots")
    
    # Other settings
    parser.add_argument("--project", type=str, default="runs/train",
                       help="Project name")
    parser.add_argument("--name", type=str, default="rat_detector",
                       help="Experiment name")
    parser.add_argument("--exist-ok", type=bool, default=False,
                       help="Overwrite existing experiment")
    parser.add_argument("--verbose", type=bool, default=True,
                       help="Verbose output")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    parser.add_argument("--deterministic", type=bool, default=True,
                       help="Deterministic training")
    
    # Post-training options
    parser.add_argument("--validate", action="store_true",
                       help="Run validation after training")
    parser.add_argument("--export-format", type=str, default=None,
                       choices=["torchscript", "onnx", "openvino", "engine", 
                               "coreml", "saved_model", "pb", "tflite", 
                               "edgetpu", "tfjs", "paddle", "ncnn"],
                       help="Export format after training")
    
    args = parser.parse_args()
    
    # Train the model
    model, results = train_yolo(args)
    
    print("\nTraining completed successfully!")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    print(f"Last model saved at: {results.save_dir}/weights/last.pt")
    
    # Provide inference example
    print("\nTo run inference with the trained model:")
    print(f"python inference.py --model {results.save_dir}/weights/best.pt --image <image_path>")

if __name__ == "__main__":
    main()