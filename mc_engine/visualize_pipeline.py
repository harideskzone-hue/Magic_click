import os
import cv2  # type: ignore
import numpy as np  # type: ignore
import sys
from ultralytics import YOLO  # type: ignore

# Add current dir to path for imports
sys.path.insert(0, r'c:\project\temp_magikCLick')

from pose_scorer import config as cfg  # type: ignore
from pose_scorer.preprocessor import prepare_image  # type: ignore
from pose_scorer.detection.yolo_detector import detect_person, detect_face  # type: ignore
from pose_scorer.detection.crop import make_person_crop, make_face_crop  # type: ignore
from pose_scorer.scorer import init_detectors, score_image  # type: ignore
from pose_scorer.aggregator import aggregate  # type: ignore

import mediapipe as mp  # type: ignore
from mediapipe.framework.formats import landmark_pb2  # type: ignore

def draw_pose(img, landmarks):
    if not landmarks: return img
    # Draw connections (subset for clarity)
    # MediaPipe pose indices: 11,12(shoulders), 23,24(hips), 13,14(elbows), 15,16(wrists)
    h, w = img.shape[:2]
    pts = {}
    for i in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
        lm = landmarks[i]
        pts[i] = (int(lm.x * w), int(lm.y * h))
        cv2.circle(img, pts[i], 4, (0, 255, 0), -1)
    
    conns = [(11,12), (11,13), (13,15), (12,14), (14,16), (11,23), (12,24), (23,24), (23,25), (25,27), (24,26), (26,28)]
    for start, end in conns:
        if start in pts and end in pts:
            cv2.line(img, pts[start], pts[end], (0, 255, 0), 2)
    return img

def draw_face(img, landmarks):
    if not landmarks: return img
    h, w = img.shape[:2]
    # Draw contours
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 1, (255, 255, 255), -1)
    
    # Highlight iris (468-471, 473-477)
    for i in list(range(468, 472)) + list(range(473, 478)):
        if i < len(landmarks):
            lm = landmarks[i]
            cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 2, (0, 255, 255), -1)
    return img

def create_viz(img_path, pd, fd, fl, pl, output_dir):
    fname = os.path.basename(img_path)
    config_dict = {
        'PREFLIGHT': cfg.PREFLIGHT, 'DETECTION': cfg.DETECTION, 'FRAME': cfg.FRAME,
        'CONFIDENCE': cfg.CONFIDENCE, 'CAMERA': cfg.CAMERA, 'FACE': cfg.FACE,
        'FACE_WEIGHTS': cfg.FACE_WEIGHTS, 'BODY_WEIGHTS': cfg.BODY_WEIGHTS,
        'GROUP_WEIGHTS': cfg.GROUP_WEIGHTS, 'SCORE_BANDS': cfg.SCORE_BANDS
    }

    # 1. Load & Preflight
    bgr, pf = prepare_image(img_path, config_dict)
    if bgr is None:
        print(f"Skipping {fname}: {pf['skip_reason']}")
        return
        
    assert bgr is not None  # type narrowing for Pyre
    full_h, full_w = bgr.shape[:2]

    # 2. Person detection
    # classes=[0] to ensure only persons
    person_bbox, pdet = detect_person(bgr, pd, config_dict)
    
    # 3. Face detection on person crop
    face_bbox = None
    person_crop = None
    if person_bbox:
        person_crop, p_off = make_person_crop(bgr, person_bbox, config_dict)
        face_bbox, fdet = detect_face(bgr, person_bbox, fd, config_dict)
    
    # 4. Score image to get data
    res = score_image(img_path, pd, fd, fl, pl)
    
    # ── PANEL ASSEMBLY ────────────────────────────────────────────────────────
    # Panel 1: Person Det
    p1 = bgr.copy()
    if person_bbox:
        cv2.rectangle(p1, (person_bbox[0], person_bbox[1]), (person_bbox[2], person_bbox[3]), (0, 255, 0), 2)
        cv2.putText(p1, f"Person {res['detection'].get('person_conf', 0):.2f}", (person_bbox[0], person_bbox[1]-10), 0, 0.7, (0, 255, 0), 2)
    
    # Panel 2: Face Det on Person Crop
    if person_crop is not None:
        p2 = person_crop.copy()
        if face_bbox:
            # bbox is global, convert to person-crop local
            lx1, ly1 = face_bbox[0] - p_off[0], face_bbox[1] - p_off[1]
            lx2, ly2 = face_bbox[2] - p_off[0], face_bbox[3] - p_off[1]
            cv2.rectangle(p2, (lx1, ly1), (lx2, ly2), (255, 0, 0), 2)
            cv2.putText(p2, f"Face {res['detection'].get('face_conf', 0):.2f}", (lx1, ly1-5), 0, 0.5, (255, 0, 0), 1)
    else:
        p2 = np.zeros((400, 300, 3), dtype=np.uint8)
        cv2.putText(p2, "No Person", (50, 200), 0, 1, (0,0,255), 2)

    # Panel 3: Face Landmarks & Iris on Face Crop
    face_crop_bgr, crop_meta = (None, None)
    if face_bbox:
        face_crop_bgr, crop_meta = make_face_crop(bgr, face_bbox, config_dict)
    
    if face_crop_bgr is not None:
        p3 = face_crop_bgr.copy()
        # Rerun Face Landmarker on crop just for visualization (since score_image doesn't return raw landmarks)
        rgb_crop = cv2.cvtColor(p3, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
        fl_res = fl.detect(mp_img)
        if fl_res.face_landmarks:
            draw_face(p3, fl_res.face_landmarks[0])
    else:
        p3 = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.putText(p3, "No Face", (100, 250), 0, 1, (0,0,255), 2)

    # Panel 4: Pose Landmarks on Full Image
    p4 = bgr.copy()
    rgb_full = cv2.cvtColor(p4, cv2.COLOR_BGR2RGB)
    mp_full = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_full)
    pl_res = pl.detect(mp_full)
    if pl_res.pose_landmarks:
        draw_pose(p4, pl_res.pose_landmarks[0])

    # ── RESIZE AND CONCAT ─────────────────────────────────────────────────────
    fh = 600
    def rsz(im, h=fh):
        asp = im.shape[1] / im.shape[0]
        return cv2.resize(im, (int(h * asp), h))

    # Text overlay
    overlay = np.zeros((220, 1400, 3), dtype=np.uint8)
    color = (0, 255, 0) if res['status'] == 'SCORED' else (0, 0, 255)
    cv2.putText(overlay, f"FILE: {fname}", (20, 40), 0, 0.8, (255,255,255), 2)
    cv2.putText(overlay, f"STATUS: {res['status']}", (20, 80), 0, 0.8, color, 2)
    cv2.putText(overlay, f"SCORE: {res['final_score']} ({res['score_band']})", (20, 120), 0, 1.2, (255,255,0), 3)
    if res.get('reject_reason'):
        cv2.putText(overlay, f"REASON: {res['reject_reason']}", (20, 160), 0, 0.7, (0,0,255), 2)

    # Details
    dx = 600
    if res.get('face_group'):
        cv2.putText(overlay, f"Face: {res['face_group'].get('group_score', '—')}", (dx, 40), 0, 0.7, (200,200,255), 2)
        for i, (k, v) in enumerate(res['face_group']['modules'].items()):
            txt = f"{k}: {v['score']:.1f}" if not v['skipped'] else f"{k}: SKIP"
            cv2.putText(overlay, txt, (dx, 75 + i*30), 0, 0.5, (180,180,180), 1)
    
    idx = 1000
    if res.get('body_group'):
        cv2.putText(overlay, f"Body: {res['body_group'].get('group_score', '—')}", (idx, 40), 0, 0.7, (200,200,255), 2)
        for i, (k, v) in enumerate(res['body_group']['modules'].items()):
            txt = f"{k}: {v['score']:.1f}" if not v['skipped'] else f"{k}: SKIP"
            cv2.putText(overlay, txt, (idx, 75 + i*30), 0, 0.5, (180,180,180), 1)

    # Concat
    top = np.hstack([rsz(p1), rsz(p2), rsz(p3), rsz(p4)])
    canvas = np.vstack([cv2.resize(overlay, (top.shape[1], 200)), top])
    
    out_path = os.path.join(output_dir, f"viz_{fname}")
    cv2.imwrite(out_path, canvas)
    print(f"[{res['status']}] -> {out_path}")

def main():
    cfg.validate()
    print("Loading models...")
    pd = YOLO(cfg.MODELS['person_detector'])
    fd = YOLO(cfg.MODELS['face_detector'])
    fl, pl = init_detectors()

    img_dir = r"c:\project\temp_magikCLick\passed_images\passed_images"
    out_dir = r"c:\project\temp_magikCLick\debug_output"
    os.makedirs(out_dir, exist_ok=True)

    paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Visualizing {len(paths)} images...")
    
    for p in paths[:10]: # type: ignore
        create_viz(p, pd, fd, fl, pl, out_dir)

if __name__ == "__main__":
    main()
