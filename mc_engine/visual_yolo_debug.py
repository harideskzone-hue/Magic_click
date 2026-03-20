import os
import cv2  # type: ignore
import glob
from ultralytics import YOLO  # type: ignore

# Ensure working directory is correctly scoped
import sys
sys.path.insert(0, r'c:\project\temp_magikCLick')
from pose_scorer import config as cfg  # type: ignore

def run_visual_debug():
    cfg.validate()
    person_det = YOLO(cfg.MODELS['person_detector'])
    
    input_dir = r"c:\project\temp_magikCLick\passed_images\passed_images"
    output_dir = r"c:\project\temp_magikCLick\debug_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Grab first 5 images for isolated testing
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))[:5]  # type: ignore
    
    print(f"Running detection on {len(image_paths)} images, saving to {output_dir}")
    
    for img_path in image_paths:
        fname = os.path.basename(img_path)
        img = cv2.imread(img_path)
        
        # Note: config threshold is 0.4. Ultralytics does NMS internally
        results = person_det(img, verbose=False, conf=cfg.DETECTION['person_conf'])
        boxes = results[0].boxes
        
        # Draw bounding boxes and confidences
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            conf = float(box.conf[0])
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            # Draw label
            label = f"Person: {conf:.2f}"
            cv2.putText(img, label, (x1, max(y1 - 10, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        
        out_path = os.path.join(output_dir, f"debug_{fname}")
        cv2.imwrite(out_path, img)
        print(f"[{len(boxes)} persons] -> {out_path}")

if __name__ == "__main__":
    run_visual_debug()
