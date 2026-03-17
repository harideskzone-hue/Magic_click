import sys, traceback, json
sys.path.insert(0, r'c:\project\temp_magikCLick')

from pose_scorer import config as cfg
from pose_scorer.scorer import score_image, init_detectors
from ultralytics import YOLO

cfg.validate()
print('Config valid')
person_det = YOLO(cfg.MODELS['person_detector'])
face_det   = YOLO(cfg.MODELS['face_detector'])
print('YOLO loaded')
face_lm, pose_lm = init_detectors()
print('MediaPipe loaded')

img = r'c:\project\temp_magikCLick\passed_images\passed_images\04c25b56-0e33-49f6-9eda-6ab651a9f516.jpg'
try:
    result = score_image(img, person_det, face_det, face_lm, pose_lm)
    print('STATUS:', result['status'])
    print('SCORE:', result['final_score'])
    print('BAND:', result.get('score_band'))
    print('REJECT:', result.get('reject_reason'))
    if result.get('detection'):
        print('DETECTION:', result['detection'])
    if result.get('frame_check'):
        print('FRAME CHECK:', result['frame_check'])
    if result.get('face_group') and result['face_group'].get('modules'):
        for k, v in result['face_group']['modules'].items():
            print(f'  face/{k}: score={v.get("score")} skipped={v.get("skipped")} reason={v.get("skip_reason")}')
    if result.get('body_group') and result['body_group'].get('modules'):
        for k, v in result['body_group']['modules'].items():
            print(f'  body/{k}: score={v.get("score")} skipped={v.get("skipped")} reason={v.get("skip_reason")}')
except Exception as e:
    traceback.print_exc()
