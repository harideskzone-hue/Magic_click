import os
import sys
import cv2  # type: ignore
import time
import subprocess
import threading
import queue
from ultralytics import YOLO  # type: ignore
import numpy as np  # type: ignore
from collections import deque

# Add current dir to path for imports
sys.path.insert(0, os.getcwd())
from pose_scorer import config as cfg  # type: ignore
import queue_manager

# Load .env
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

def _parse_camera_source(val: str):
    """Parse a camera source string: numeric string -> int (webcam), URL -> str."""
    if val.strip().lstrip('-').isdigit():
        return int(val.strip())
    return val.strip()


# ── Preflight Validation ──────────────────────────────────────────────────────
def _preflight():
    """
    Validate all critical dependencies before starting the pipeline.
    Returns True if all required checks pass.
    """
    _ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
    _MODELS_DIR = os.path.join(_ENGINE_DIR, "models")

    checks = []

    # 1. Model files
    required_models = ["yolo26n.pt", "face_landmarker.task", "pose_landmarker_full.task"]
    optional_models = ["yolo26n-face.pt"]

    for m in required_models:
        path = os.path.join(_MODELS_DIR, m)
        exists = os.path.isfile(path)
        size_mb = os.path.getsize(path) / 1e6 if exists else 0
        checks.append(("✓" if exists else "✗", m, f"{size_mb:.1f} MB" if exists else "MISSING (REQUIRED)"))
        if not exists:
            # Attempt auto-download
            try:
                sys.path.insert(0, _ENGINE_DIR)
                from model_manager import download_missing  # type: ignore
                download_missing()
            except Exception:
                pass

    for m in optional_models:
        path = os.path.join(_MODELS_DIR, m)
        exists = os.path.isfile(path)
        size_mb = os.path.getsize(path) / 1e6 if exists else 0
        checks.append(("✓" if exists else "⚠", m, f"{size_mb:.1f} MB" if exists else "missing (optional)"))

    # 2. Camera test
    cam_ok = False
    cam_detail = "UNAVAILABLE"
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cam_ok = True
            cam_detail = f"{w}×{h}"
            cap.release()
        else:
            cap.release()
    except Exception:
        pass
    checks.append(("✓" if cam_ok else "✗", "Camera 0", cam_detail))

    # 3. API health
    api_ok = False
    api_detail = "NOT RESPONDING"
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:5001/api/health", timeout=3) as resp:
            if resp.status == 200:
                import json as _json
                data = _json.loads(resp.read())
                api_ok = True
                api_detail = f"200 OK, {data.get('person_count', 0)} persons"
    except Exception:
        pass
    checks.append(("✓" if api_ok else "✗", "API health", api_detail))

    # 4. Session token
    token_ok = False
    if getattr(sys, 'frozen', False):
        _base = os.path.dirname(sys.executable)
    else:
        _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    _usr_dir = os.environ.get("MAGIC_CLICK_DATA", os.path.join(_base, "data"))
    _secret = os.path.join(_usr_dir, ".session_secret")
    if os.path.isfile(_secret):
        token_ok = True
    checks.append(("✓" if token_ok else "⚠", "Session token", "loaded" if token_ok else "missing (uploads may fail)"))

    # 5. Disk space
    import shutil
    total, used, free = shutil.disk_usage(os.path.expanduser("~"))
    free_gb = free / (1024**3)
    disk_ok = free_gb >= 0.5
    checks.append(("✓" if disk_ok else "⚠", "Disk space", f"{free_gb:.1f} GB free"))

    # Print table
    print("\n┌─ PREFLIGHT ──────────────────────────────────────────┐")
    for sym, name, detail in checks:
        print(f"│ {sym} {name:<28} ({detail:<20}) │")
    print("└──────────────────────────────────────────────────────┘\n")

    # Check for fatal failures
    fatal = any(sym == "✗" for sym, _, _ in checks)
    if fatal:
        print("[PREFLIGHT] ✗ Some required checks failed. See above.")
        return False
    return True


# ── Threaded camera reader ────────────────────────────────────────────────────
class CameraStream:
    def __init__(self, src):
        self.src = src
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            print(f"Failed to open {src}. Falling back to webcam 0.")
            self.stream = cv2.VideoCapture(0)
            if not self.stream.isOpened():
                print("Error: Could not open fallback webcam either.")
                self.stopped = True
                return

        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # Keep OpenCV's internal buffer as small as possible so we always
        # get the newest frame rather than reading stale buffered frames.
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret, self.frame = self.stream.read()
        self.stopped = False
        self.frame_id = 0
        self._lock = threading.Lock()

    def start(self):
        t = threading.Thread(target=self._update, daemon=True)
        t.start()
        return self

    def _update(self):
        """Drain OpenCV's internal network buffer with grab() in a tight loop.
        We only decode (retrieve) when the main thread actually wants a frame.
        This keeps the buffer empty so read() always returns the freshest frame."""
        while not self.stopped:
            # grab() signals the camera / advances the buffer pointer (cheap)
            grabbed = self.stream.grab()
            if not grabbed:
                time.sleep(0.005)
                continue
            # Decode the grabbed frame and expose it to the main thread
            ret, frame = self.stream.retrieve()
            if ret:
                with self._lock:
                    self.ret = ret
                    self.frame = frame
                    self.frame_id += 1

    def read(self):
        with self._lock:
            return self.ret, self.frame, self.frame_id

    def stop(self):
        self.stopped = True
        self.stream.release()


# ── Non-blocking YOLO inference thread ───────────────────────────────────────
class YoloDetectorThread:
    """
    Runs YOLO person-detection in a dedicated thread.
    We pass in a camera_id so the caller knows which camera this result is for.
    """
    def __init__(self, model, conf, min_height_ratio, face_model=None):
        self.model = model
        self.face_model = face_model   # optional fallback
        self.conf = conf
        self.min_height_ratio = min_height_ratio

        # Queue of depth-1: newest frame only
        self._q = queue.Queue(maxsize=1)
        self.result = False          # latest detection result
        self._stopped = False
        self._busy = False           # True while inference is running

    def start(self):
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
        return self

    def submit(self, small_frame):
        """Submit a (possibly downscaled) frame for detection. Non-blocking."""
        try:
            # Discard stale pending frame and replace with the newest one
            self._q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._q.put_nowait(small_frame)
        except queue.Full:
            pass  # Already has a frame queued — skip

    @property
    def is_busy(self):
        return self._busy

    def _run(self):
        while not self._stopped:
            try:
                frame = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            self._busy = True
            try:
                h = frame.shape[0]
                # Primary: YOLO person-body detection (class 0 = person)
                results = self.model(frame, classes=[0], conf=self.conf, verbose=False)
                boxes = results[0].boxes
                detected = False
                if len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        if (y2 - y1) / h >= self.min_height_ratio:
                            detected = True
                            break

                # Fallback: face detection — catches close-up/upper-body where
                # the full body silhouette is not visible inside the frame
                if not detected and self.face_model is not None:
                    try:
                        face_results = self.face_model(frame, conf=self.conf, verbose=False)
                        if face_results and len(face_results[0].boxes) > 0:
                            detected = True
                    except Exception:
                        pass

                self.result = detected
            except Exception:
                pass
            finally:
                self._busy = False

    def stop(self):
        self._stopped = True


class CameraProcessor:
    def __init__(self, cam_id, src, person_det_model, config_dict, face_det_model=None):
        self.cam_id = cam_id
        self.src = src
        self.stream = CameraStream(src)
        if self.stream.stopped:
            return
        self.stream.start()

        det_conf = config_dict['DETECTION']['person_conf']
        min_h_ratio = config_dict['DETECTION'].get('min_person_height_ratio', 0.15)
        self.yolo_thread = YoloDetectorThread(
            person_det_model, det_conf, min_h_ratio,
            face_model=face_det_model   # face is the fallback when body is absent
        )
        self.yolo_thread.start()

        self.last_scored_gray = None
        self.last_person_detected = False
        
        import typing
        self.is_recording = False
        self.video_writer: typing.Any = None
        self.video_fname: typing.Optional[str] = None
        self.last_person_time = 0.0
        self.frame_count_this_session = 0
        
        self.frame_times = deque(maxlen=30)
        self.display_fps = 15.0
        self.last_processed_frame_id = -1
        self.current_motion = 0.0
        
        # Output frame for rendering
        self.viz_frame = None

    def process_frame(self, motion_threshold=3.0, grace_period=2.0) -> None:
        if self.stream.stopped:
            return

        ret, frame, frame_id = self.stream.read()
        if not ret or frame is None:
            return

        if frame_id == self.last_processed_frame_id:
            return
        self.last_processed_frame_id = frame_id

        # Copy and flip
        frame = cv2.flip(frame, 1).copy()
        h_f, w_f = frame.shape[:2]

        # Cheap motion diff
        small = cv2.resize(frame, (320, 180))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if self.last_scored_gray is None:
            motion = float('inf')
        else:
            motion = float(np.mean(np.abs(gray - self.last_scored_gray)))
        self.current_motion = motion

        # Submit to YOLO if motion or person was not seen
        if motion > motion_threshold or not self.last_person_detected:
            self.yolo_thread.submit(small)
            self.last_scored_gray = gray

        self.last_person_detected = self.yolo_thread.result
        person_present = self.last_person_detected

        # FPS tracking
        now = time.time()
        self.frame_times.append(now)
        if len(self.frame_times) >= 10 and len(self.frame_times) % 10 == 0:
            recent = list(self.frame_times)[-10:]  # type: ignore
            self.display_fps = max(1.0, min(60.0, 9.0 / (recent[-1] - recent[0])))

        if person_present:
            self.last_person_time = now

        # Start recording
        if person_present and not self.is_recording:
            self.is_recording = True
            ts = int(now * 1000)
            self.video_fname = os.path.join(queue_manager.VIDEOS_DIR, f"cam{self.cam_id}_session_{ts}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            rec_fps = self.display_fps if 10 <= self.display_fps <= 45 else 25.0
            self.video_writer = cv2.VideoWriter(self.video_fname, fourcc, rec_fps, (w_f, h_f))
            self.frame_count_this_session = 0
            print(f"\n[CAM {self.cam_id}] RECORDING START: {self.video_fname} at {rec_fps:.1f} FPS")

        # Write / stop recording
        if self.is_recording:
            elapsed_no_person = now - self.last_person_time
            if elapsed_no_person > grace_period:
                print(f"\n[CAM {self.cam_id}] RECORDING STOP: No person for {grace_period}s.")
                self.is_recording = False
                if self.video_writer:
                    self.video_writer.release()
                    self.video_writer = None
                    
                # MULTI-CAMERA SJF QUEUE MODIFICATION
                print(f"[CAM {self.cam_id}] Submitting job to SJF queue with {self.frame_count_this_session} frames...")
                queue_manager.add_job(self.video_fname, self.frame_count_this_session)
                
            else:
                if self.video_writer:
                    self.video_writer.write(frame)
                    self.frame_count_this_session += 1

        # HUD Overlay
        viz = frame.copy()
        pw, ph = min(500, w_f), min(120, h_f)
        region = viz[0:ph, 0:pw].copy()
        dark   = np.full_like(region, (8, 10, 16))
        viz[0:ph, 0:pw] = cv2.addWeighted(dark, 0.78, region, 0.22, 0)

        cv2.putText(viz, f"CAM {self.cam_id} - AUTO RECORDING", (15, 30), 0, 0.8, (0, 255, 255), 2)
        status_text = "PERSON DETECTED" if person_present else "NO PERSON"
        color = (0, 255, 0) if person_present else (0, 0, 255)
        cv2.putText(viz, status_text, (15, 65), 0, 0.6, color, 2)
        cv2.putText(viz, f"Motion: {motion:.1f}", (15, 95), 0, 0.6, (200, 200, 200), 1)

        stable_label = f"MOTION {motion:.1f}" if motion > motion_threshold else "STABLE"
        label_color  = (0, 165, 255) if motion > motion_threshold else (0, 255, 0)
        cv2.putText(viz, stable_label, (w_f - 200, h_f - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2, cv2.LINE_AA)

        rec_status = "RECORDING" if self.is_recording else "STANDBY"
        rec_color  = (0, 0, 255) if self.is_recording else (150, 150, 150)
        cv2.putText(viz, rec_status, (w_f - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, rec_color, 2, cv2.LINE_AA)

        yolo_indicator = " [YOLO]" if self.yolo_thread.is_busy else ""
        cv2.putText(viz, f"{self.display_fps:.1f} FPS{yolo_indicator}", (w_f - 180, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1, cv2.LINE_AA)

        self.viz_frame = viz

    def stop(self):
        self.yolo_thread.stop()
        self.stream.stop()
        if self.video_writer:
            self.video_writer.release()

# ── Remote Camera IPC State ────────────────────────────────────────────────────
_CAMERA_STATE_FILE = os.environ.get(
    "CAMERAS_STATE_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "camera_state.json")
)

def _is_system_active() -> bool:
    if not os.path.exists(_CAMERA_STATE_FILE):
        return False
    try:
        import json
        with open(_CAMERA_STATE_FILE, "r") as f:
            data = json.load(f)
            return data.get("active", False)
    except Exception:
        return False

# ── Camera config from cameras.json ───────────────────────────────────────────
_CAMERAS_JSON = os.environ.get(
    "CAMERAS_JSON_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cameras.json")
)
_CAM_RELOAD_INTERVAL = 10.0   # seconds between hot-reload checks
_CAM_RETRY_DELAYS    = [2, 5, 10, 30]  # back-off seconds on stream failure


def _load_cameras_json() -> list:
    """Read cameras.json safely. Returns [] on any error."""
    try:
        with open(_CAMERAS_JSON, 'r') as f:
            data = json.load(f)
        return [c for c in data if isinstance(c, dict) and c.get('enabled', True)]
    except Exception:
        return []


def _env_camera_sources() -> list:
    """Fallback: read CAMERA_0/1/2 from env vars (backward compat)."""
    srcs = []
    for key in ("CAMERA_0", "CAMERA_1", "CAMERA_2"):
        val = os.environ.get(key, "DISABLED")
        if val.strip().upper() != "DISABLED":
            srcs.append({"id": key.lower(), "source": val.strip(), "label": key, "enabled": True})
    return srcs


def _get_active_cameras() -> list:
    """
    Return the list of enabled camera config dicts to use right now.
    Prefers cameras.json; falls back to CAMERA_x env vars.
    """
    configs = _load_cameras_json()
    if not configs:
        configs = _env_camera_sources()
    return configs


def _start_processor(cam_cfg: dict, person_det, config_dict, face_det=None) -> "CameraProcessor | None":
    """
    Attempt to start a CameraProcessor for a config entry.
    Returns None (and logs) if the stream cannot be opened.
    """
    src = _parse_camera_source(str(cam_cfg['source']))
    label = cam_cfg.get('label', cam_cfg['id'])
    print(f"  [CAM] Connecting → {label} ({src}) ...")
    proc = CameraProcessor(cam_cfg['id'], src, person_det, config_dict, face_det_model=face_det)
    if proc.stream.stopped:
        print(f"  [CAM] ✗ Failed to open {label} ({src})")
        return None
    print(f"  [CAM] ✓ Connected: {label}")
    return proc


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("--- Initializing Multi-Camera Auto-Recording Pipeline ---")
    cfg.validate()

    config_dict = {
        'PREFLIGHT': cfg.PREFLIGHT, 'DETECTION': cfg.DETECTION, 'FRAME': cfg.FRAME,
        'CONFIDENCE': cfg.CONFIDENCE, 'CAMERA': cfg.CAMERA, 'FACE': cfg.FACE,
        'BODY': cfg.BODY, 'FACE_WEIGHTS': cfg.FACE_WEIGHTS, 'BODY_WEIGHTS': cfg.BODY_WEIGHTS,
        'GROUP_WEIGHTS': cfg.GROUP_WEIGHTS, 'SCORE_BANDS': cfg.SCORE_BANDS,
        'DEBUG': cfg.DEBUG
    }

    print("Loading YOLO models (this may take a few seconds)...")
    person_det = YOLO(cfg.MODELS['person_detector'])
    # Load the face model as a fallback detection source
    try:
        face_det = YOLO(cfg.MODELS['face_detector'])
        print("  ✓ Face model loaded (used as fallback for close-up shots)")
    except Exception as e:
        face_det = None
        print(f"  ⚠ Face model unavailable (fallback disabled): {e}")

    # ── Initial camera load ─────────────────────────────────────────────────
    active_cfg   = _get_active_cameras()
    processors: dict[str, "CameraProcessor"] = {}   # cam_id → processor
    retry_times: dict[str, float]            = {}   # cam_id → next retry timestamp
    retry_counts: dict[str, int]             = {}   # cam_id → number of retries

    def _sync_processors(configs: list):
        """
        Reconcile running processors with the desired config list.
        Starts new cameras, stops removed ones. Modifies dict in-place.
        """
        desired_ids = {c['id'] for c in configs}

        # Stop processors for removed cameras
        removed_ids = set(processors.keys()) - desired_ids
        for cid in removed_ids:
            print(f"  [CAM] Stopping removed camera: {cid}")
            try:
                processors[cid].stop()
            except Exception as e:
                print(f"  [CAM] Warning during stop of {cid}: {e}")
            processors.pop(cid, None)
            retry_times.pop(cid, None)
            retry_counts.pop(cid, None)

        # Start processors for new cameras (not already running)
        for cam_cfg in configs:
            cid = cam_cfg['id']
            if cid in processors:
                continue  # already running
            now = time.time()
            if cid in retry_times and now < retry_times[cid]:
                continue  # still in back-off
            proc = _start_processor(cam_cfg, person_det, config_dict, face_det=face_det)
            if proc:
                processors[cid] = proc
                retry_times.pop(cid, None)
                retry_counts.pop(cid, None)
            else:
                # Back-off: pick next delay based on how many retries done
                count = retry_counts.get(cid, 0)
                delay = _CAM_RETRY_DELAYS[min(count, len(_CAM_RETRY_DELAYS) - 1)]
                retry_counts[cid] = count + 1
                retry_times[cid] = time.time() + delay
                print(f"  [CAM] Will retry {cid} in {delay}s")

    os.makedirs(queue_manager.VIDEOS_DIR, exist_ok=True)
    print("\n--- Multi-Camera System Ready (IDLE) ---")
    print("Awaiting Start signal from Dashboard http://localhost:5001/")

    last_reload = 0.0
    system_active = False

    try:
        while True:
            target_active = _is_system_active()
            
            if not target_active:
                if system_active:
                    print("\n  [SYSTEM] Remote stop signal received. Shutting down hardware...")
                    for proc in processors.values():
                        proc.stop()
                    processors.clear()
                    retry_times.clear()
                    retry_counts.clear()
                    try:
                        cv2.destroyAllWindows()
                    except Exception: pass
                    system_active = False
                    print("  [SYSTEM] Hardware released. Entering IDLE state.")
                    time.sleep(1.0) # Pause an extra beat to let hardware catch up
                
                time.sleep(0.5)
                continue
                
            else:
                if not system_active:
                    print("\n  [SYSTEM] Remote start signal received. Allocating hardware...")
                    try:
                        cv2.namedWindow('Multi-Camera Feed', cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('Multi-Camera Feed', 1280, 720)
                    except Exception as e:
                        print(f"  [CAM] Warning: UI not available in current environment ({e})")
                    system_active = True
                    last_reload = 0 # Force instant sync of cameras.json
                
            # ── Hot-reload cameras.json every _CAM_RELOAD_INTERVAL seconds ──
            now = time.time()
            if now - last_reload >= _CAM_RELOAD_INTERVAL:  # type: ignore
                new_cfg = _get_active_cameras()
                _sync_processors(new_cfg)
                last_reload = now

            # ── Check for crashed streams and attempt recovery ───────────────
            for cid, proc in list(processors.items()):
                if proc.stream.stopped:
                    print(f"  [CAM] Stream {cid} stopped unexpectedly. Scheduling retry...")
                    proc.stop()
                    processors.pop(cid, None)
                    retry_times[cid] = time.time() + _CAM_RETRY_DELAYS[0]

            # ── Process frames ───────────────────────────────────────────────
            viz_frames = []
            for proc in processors.values():
                proc.process_frame()
                if proc.viz_frame is not None:
                    small_viz = cv2.resize(proc.viz_frame, (640, 360))
                    viz_frames.append(small_viz)

            # ── Grid rendering ───────────────────────────────────────────────
            if viz_frames:
                if len(viz_frames) == 1:
                    grid = viz_frames[0]
                elif len(viz_frames) == 2:
                    grid = np.hstack((viz_frames[0], viz_frames[1]))
                elif len(viz_frames) == 3:
                    top = np.hstack((viz_frames[0], viz_frames[1]))
                    bot = np.hstack((viz_frames[2], np.zeros_like(viz_frames[0])))
                    grid = np.vstack((top, bot))
                else:
                    top = np.hstack((viz_frames[0], viz_frames[1]))
                    bot = np.hstack((viz_frames[2], viz_frames[3]))
                    grid = np.vstack((top, bot))
                cv2.imshow('Multi-Camera Feed', grid)
            else:
                # Show blank "waiting" frame
                blank = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Waiting for cameras...", (140, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
                cv2.putText(blank, "Add cameras at http://localhost:5001/", (70, 230),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
                cv2.imshow('Multi-Camera Feed', blank)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        for proc in processors.values():
            proc.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import json  # needed for _load_cameras_json

    # Run preflight checks
    if not _preflight():
        print("[FATAL] Preflight checks failed. Fix issues above and retry.")
        sys.exit(1)

    main()

