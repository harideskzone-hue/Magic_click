import sys, os
sys.path.insert(0, r'c:\project\temp_magikCLick')

from pose_scorer import config as cfg
from ultralytics import YOLO
import cv2

# Just run person detection on 5 images and report counts
cfg.validate()
person_det = YOLO(cfg.MODELS['person_detector'])

img_dir = r'c:\project\temp_magikCLick\passed_images\passed_images'
images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')][:5]

for fname in images:
    path = os.path.join(img_dir, fname)
    bgr = cv2.imread(path)
    results = person_det(bgr, verbose=False, conf=cfg.DETECTION['person_conf'])
    boxes = results[0].boxes
    confs_str = [f"{float(b.conf[0]):.2f}" for b in boxes]
    print(f"{fname[:25]}  persons={len(boxes)}  confs={confs_str}")
