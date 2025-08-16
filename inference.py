from ultralytics import YOLO
import cv2
import argparse
import os
from pathlib import Path
import numpy as np
import supervision as sv

def run_inference(args):
    """
    Run inference on images or videos using YOLOv8
    """
    
    # Load model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    # Check if input is image or video
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found!")
        return
    
    # Determine if input is image or video
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    
    if input_path.suffix.lower() in image_extensions:
        process_image(model, args)
    elif input_path.suffix.lower() in video_extensions:
        process_video(model, args)
    else:
        print(f"Unsupported file format: {input_path.suffix}")
        return

def process_image(model, args):
    """
    Process a single image
    """
    # Run inference
    results = model(
        args.input,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        agnostic_nms=args.agnostic_nms,
        max_det=args.max_det,
        classes=args.classes,
    )
    
    # Process results
    for r in results:
        # Get image
        img = r.orig_img
        
        # Get detections
        boxes = r.boxes
        
        if boxes is not None and len(boxes) > 0:
            print(f"\nDetected {len(boxes)} object(s):")
            
            # Print detection details
            for i, box in enumerate(boxes):
                cls = int(box.cls)
                conf = float(box.conf)
                xyxy = box.xyxy[0].cpu().numpy()
                
                # Get class name
                class_name = model.names[cls] if cls < len(model.names) else f"Class_{cls}"
                
                print(f"  {i+1}. {class_name} (confidence: {conf:.3f})")
                print(f"     Bounding box: [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")
            
            # Visualize if requested
            if args.show or args.save:
                # Convert to supervision format for better visualization
                detections = sv.Detections.from_ultralytics(r)
                
                # Annotate with boxes
                box_annotator = sv.BoxAnnotator(
                    thickness=args.line_thickness
                )
                
                # Annotate with labels
                label_annotator = sv.LabelAnnotator(
                    text_scale=args.text_scale,
                    text_thickness=1
                )
                
                # Create labels
                labels = [
                    f"{model.names[int(cls)]} {conf:.2f}"
                    for cls, conf in zip(detections.class_id, detections.confidence)
                ]
                
                # Annotate image with boxes
                annotated_img = box_annotator.annotate(
                    scene=img.copy(),
                    detections=detections
                )
                
                # Add labels
                annotated_img = label_annotator.annotate(
                    scene=annotated_img,
                    detections=detections,
                    labels=labels
                )
                
                # Save image
                if args.save:
                    output_path = Path(args.output) if args.output else Path("output.jpg")
                    cv2.imwrite(str(output_path), annotated_img)
                    print(f"\nAnnotated image saved to: {output_path}")
                
                # Show image
                if args.show:
                    cv2.imshow("YOLOv8 Detection", annotated_img)
                    print("\nPress any key to close the image window...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        else:
            print("\nNo objects detected in the image.")
            
            if args.show:
                cv2.imshow("YOLOv8 Detection", img)
                print("\nPress any key to close the image window...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

def process_video(model, args):
    """
    Process a video file
    """
    # Open video
    cap = cv2.VideoCapture(args.input)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    
    # Setup video writer if saving
    writer = None
    if args.save:
        output_path = Path(args.output) if args.output else Path("output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"\nSaving output to: {output_path}")
    
    # Setup annotators
    box_annotator = sv.BoxAnnotator(
        thickness=args.line_thickness,
        text_thickness=args.text_thickness,
        text_scale=args.text_scale
    )
    
    # Process frames
    frame_count = 0
    detection_count = 0
    
    print("\nProcessing video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames if requested
        if args.skip_frames > 0 and frame_count % (args.skip_frames + 1) != 0:
            continue
        
        # Run inference
        results = model(
            frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            agnostic_nms=args.agnostic_nms,
            max_det=args.max_det,
            classes=args.classes,
            verbose=False
        )
        
        # Process results
        for r in results:
            detections = sv.Detections.from_ultralytics(r)
            
            if len(detections) > 0:
                detection_count += len(detections)
                
                # Create labels
                labels = [
                    f"{model.names[int(cls)]} {conf:.2f}"
                    for cls, conf in zip(detections.class_id, detections.confidence)
                ]
                
                # Annotate frame
                annotated_frame = box_annotator.annotate(
                    scene=frame.copy(),
                    detections=detections,
                    labels=labels
                )
            else:
                annotated_frame = frame
            
            # Save frame
            if writer:
                writer.write(annotated_frame)
            
            # Show frame
            if args.show:
                cv2.imshow("YOLOv8 Video Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nVideo playback interrupted by user.")
                    break
        
        # Print progress
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"\nVideo processing complete!")
    print(f"  Frames processed: {frame_count}")
    print(f"  Total detections: {detection_count}")
    print(f"  Average detections per frame: {detection_count/frame_count:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference on images or videos")
    
    # Input/Output
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Path to input image or video")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Path to save output (annotated image/video)")
    parser.add_argument("--model", "-m", type=str, default="yolov8n.pt",
                       help="Path to YOLO model (e.g., yolov8n.pt, runs/train/rat_detector/weights/best.pt)")
    
    # Inference settings
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45,
                       help="IoU threshold for NMS")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="Inference image size")
    parser.add_argument("--device", type=str, default="0",
                       help="Device to use (0 for GPU, 'cpu' for CPU)")
    parser.add_argument("--max-det", type=int, default=300,
                       help="Maximum number of detections per image")
    parser.add_argument("--classes", type=int, nargs="+", default=None,
                       help="Filter by class indices (e.g., --classes 0 1 2)")
    parser.add_argument("--agnostic-nms", action="store_true",
                       help="Class-agnostic NMS")
    
    # Visualization settings
    parser.add_argument("--show", action="store_true",
                       help="Show results")
    parser.add_argument("--save", action="store_true",
                       help="Save results")
    parser.add_argument("--line-thickness", type=int, default=2,
                       help="Bounding box line thickness")
    parser.add_argument("--text-thickness", type=int, default=1,
                       help="Text thickness")
    parser.add_argument("--text-scale", type=float, default=0.5,
                       help="Text scale")
    
    # Video settings
    parser.add_argument("--skip-frames", type=int, default=0,
                       help="Skip n frames between inferences (for faster processing)")
    
    args = parser.parse_args()
    
    # Run inference
    run_inference(args)

if __name__ == "__main__":
    main()