from inference import get_model
import supervision as sv
from inference.core.utils.image_utils import load_image_bgr
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable CUDA

def main():
    # Check if image exists
    image_path = "rat-new3.jpeg"
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return
    
    try:
        # Load image
        image = load_image_bgr(image_path)
        print(f"Image loaded successfully: {image.shape}")
        
        # Initialize model - you might want to try different models
        # Options: "yolov8n-640", "yolov8s-640", "yolov8m-640", "yolov8l-640", "yolov8x-640"
        #model_name = "yolov8n-640"
        model_name = 'yolov8x-640'

        model = get_model(model_id=model_name)
        print("Model loaded successfully")
        
        # Run inference with lower confidence threshold
        # Lower confidence might help detect more objects
        results = model.infer(image, confidence=0.1)[0]
        
        # Convert to supervision format
        detections = sv.Detections.from_inference(results)
        
        print(f"Number of detections: {len(detections)}")
        print(f"Detections: {detections}")
        
        if len(detections) == 0:
            print("\nNo objects detected. Possible reasons:")
            print("1. The image doesn't contain objects from COCO dataset classes")
            print("2. Objects are too small or unclear")
            print("3. Try lowering the confidence threshold")
            print("\nCOCO classes include: person, bicycle, car, motorcycle, airplane,")
            print("bus, train, truck, boat, traffic light, fire hydrant, stop sign,")
            print("parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant,")
            print("bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase,")
            print("frisbee, skis, snowboard, sports ball, kite, baseball bat,")
            print("baseball glove, skateboard, surfboard, tennis racket, bottle,")
            print("wine glass, cup, fork, knife, spoon, bowl, banana, apple,")
            print("sandwich, orange, broccoli, carrot, hot dog, pizza, donut,")
            print("cake, chair, couch, potted plant, bed, dining table, toilet,")
            print("tv, laptop, mouse, remote, keyboard, cell phone, microwave,")
            print("oven, toaster, sink, refrigerator, book, clock, vase, scissors,")
            print("teddy bear, hair drier, toothbrush")
            
            # Still show the original image
            sv.plot_image(image)
        else:
            # Print detection details
            for i in range(len(detections)):
                if 'class_name' in detections.data:
                    class_name = detections.data['class_name'][i]
                else:
                    class_name = f"Class {detections.class_id[i]}"
                confidence = detections.confidence[i]
                bbox = detections.xyxy[i]
                print(f"Detection {i+1}: {class_name} (confidence: {confidence:.2f})")
                print(f"  Bounding box: {bbox}")
            
            # Annotate image
            annotator = sv.BoxAnnotator(thickness=4)
            annotated_image = annotator.annotate(scene=image.copy(), detections=detections)
            
            label_annotator = sv.LabelAnnotator(text_scale=1, text_thickness=2)
            
            # Create labels with class names and confidence
            labels = []
            for i in range(len(detections)):
                if 'class_name' in detections.data:
                    class_name = detections.data['class_name'][i]
                else:
                    class_name = f"Class {detections.class_id[i]}"
                confidence = detections.confidence[i]
                labels.append(f"{class_name} {confidence:.2f}")
            
            annotated_image = label_annotator.annotate(
                scene=annotated_image, 
                detections=detections,
                labels=labels
            )
            
            # Display the annotated image
            sv.plot_image(annotated_image)
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
