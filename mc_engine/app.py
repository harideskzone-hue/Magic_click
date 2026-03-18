"""
Flask API backend for Best Photo Picker v3 UI.
Run: python app.py
"""
import os, json, tempfile, base64, threading
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory  # type: ignore
from flask_cors import CORS  # type: ignore

app = Flask(__name__, static_folder='ui', static_url_path='')
CORS(app)

# ── Global detector singletons (loaded once on first use) ───────────────────
_detectors = None
_lock = threading.Lock()

def get_detectors():
    global _detectors
    if _detectors is None:
        with _lock:
            if _detectors is None:
                from ultralytics import YOLO  # type: ignore
                from pose_scorer import config as cfg  # type: ignore
                from pose_scorer.scorer import init_detectors  # type: ignore
                person_det = YOLO(cfg.MODELS['person_detector'])
                face_det   = YOLO(cfg.MODELS['face_detector'])
                face_lm, pose_lm = init_detectors()
                _detectors = {
                    'person': person_det,
                    'face':   face_det,
                    'face_lm': face_lm,
                    'pose_lm': pose_lm,
                    'cfg': cfg
                }
    return _detectors

# ── Routes ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('ui', 'index.html')

@app.route('/api/status')
def api_status():
    try:
        from pose_scorer import config as cfg  # type: ignore
        cfg.validate()
        models_ready = all(
            Path(p).exists() for p in cfg.MODELS.values()
        )
        return jsonify({'status': 'ok', 'models_ready': models_ready})
    except Exception as e:
        return jsonify({'status': 'error', 'detail': str(e)}), 500

@app.route('/api/score', methods=['POST'])
def api_score():
    files = request.files.getlist('images')
    if not files:
        return jsonify({'error': 'No images uploaded'}), 400

    try:
        det = get_detectors()
    except Exception as e:
        return jsonify({'error': f'Model load failed: {e}. Run models/download_models.sh first.'}), 503

    from pose_scorer.scorer import score_image  # type: ignore

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            ext  = Path(f.filename).suffix.lower()
            path = os.path.join(tmp, f.filename)
            f.save(path)

            try:
                result = score_image(  # type: ignore
                    image_path      = path,
                    person_detector = det['person'],      # type: ignore
                    face_detector   = det['face'],        # type: ignore
                    face_landmarker = det['face_lm'],     # type: ignore
                    pose_landmarker = det['pose_lm'],     # type: ignore
                )
            except Exception as e:
                result = {
                    'image': f.filename, 'status': 'PIPELINE_ERROR',
                    'reject_reason': str(e)[:200],  # type: ignore
                    'final_score': None, 'score_band': '', 'rank': None,
                    'preflight': {}, 'detection': {}, 'frame_check': {},
                    'face_group': {}, 'body_group': {}, 'debug_image': '',
                }

            # For rejected images generate a lightweight debug image showing what was detected
            if isinstance(result, dict) and result.get('status') != 'SCORED' and 'debug_image' not in result:
                try:
                    import cv2, numpy as np, base64  # type: ignore
                    img = cv2.imread(path)
                    if img is not None:
                        img_h, img_w = img.shape[:2]
                        scale = min(1.0, 900 / img_w)
                        if scale < 1.0:
                            img = cv2.resize(img, (int(img_w * scale), int(img_h * scale)))
                        dh, dw = img.shape[:2]
                        region = img[0:50, 0:min(340, img.shape[1])].copy()
                        dark = np.full_like(region, (8, 10, 16))
                        img[0:region.shape[0], 0:region.shape[1]] = cv2.addWeighted(dark, 0.80, region, 0.20, 0)
                        cv2.putText(img, result.get('status',''), (10, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 255), 1, cv2.LINE_AA)
                        reason = result.get('reject_reason') or ''
                        cv2.putText(img, reason[:55], (10, 40),  # type: ignore
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1, cv2.LINE_AA)
                        # draw person bbox if available
                        det = result.get('detection') if isinstance(result, dict) else None
                        if isinstance(det, dict) and 'person_bbox' in det:  # type: ignore
                            pb = det['person_bbox']  # type: ignore
                            if isinstance(pb, list) and len(pb) == 4:
                                p = [int(v * scale) for v in pb]
                                cv2.rectangle(img, (p[0], p[1]), (p[2], p[3]), (40, 220, 80), 2)
                        # draw face bbox if available
                        if isinstance(det, dict) and 'face_bbox' in det:  # type: ignore
                            fb = det['face_bbox']  # type: ignore
                            if isinstance(fb, list) and len(fb) == 4:
                                f2 = [int(v * scale) for v in fb]
                                cv2.rectangle(img, (f2[0], f2[1]), (f2[2], f2[3]), (255, 120, 30), 2)
                        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        
                        if isinstance(result, dict):
                            result['debug_image'] = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()  # type: ignore
                except Exception:
                    pass

            results.append(result)

    # Rank scored images
    scored = [r for r in results if r['status'] == 'SCORED' and r['final_score'] is not None]
    scored.sort(key=lambda x: x['final_score'], reverse=True)
    for i, r in enumerate(scored):
        r['rank'] = i + 1

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5050)
