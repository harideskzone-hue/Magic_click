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
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

def _parse_camera_source(val: str):
    """Parse a camera source string: numeric string -> int (webcam), URL -> str."""
    if val.strip().lstrip('-').isdigit():
        return int(val.strip())
    return val.strip()


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
    def __init__(self, model, conf, min_height_ratio):
        self.model = model
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
                results = self.model(frame, classes=[0], conf=self.conf, verbose=False)
                boxes = results[0].boxes
                detected = False
                if len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        if (y2 - y1) / h >= self.min_height_ratio:
                            detected = True
                            break
                self.result = detected
            except Exception:
                pass
            finally:
                self._busy = False

    def stop(self):
        self._stopped = True


class CameraProcessor:
    def __init__(self, cam_id, src, person_det_model, config_dict):
        self.cam_id = cam_id
        self.src = src
        self.stream = CameraStream(src)
        if self.stream.stopped:
            return
        self.stream.start()

        det_conf = config_dict['DETECTION']['person_conf']
        min_h_ratio = config_dict['DETECTION'].get('min_person_height_ratio', 0.35)
        self.yolo_thread = YoloDetectorThread(person_det_model, det_conf, min_h_ratio)
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
            self.video_fname = f"captured_videos/cam{self.cam_id}_session_{ts}.mp4"
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

    print("Loading YOLO model (this may take a few seconds)...")
    person_det = YOLO(cfg.MODELS['person_detector'])

    # Load camera sources from .env
    # Edit CAMERA_0 and CAMERA_1 in .env to change IPs without touching this file.
    camera_sources = []
    for key in ("CAMERA_0", "CAMERA_1", "CAMERA_2"):
        val = os.environ.get(key, "DISABLED")
        if val.strip().upper() != "DISABLED":
            camera_sources.append(_parse_camera_source(val))
    
    processors = []
    for i, src in enumerate(camera_sources):
        print(f"Connecting to Camera {i}: {src} ...")
        p = CameraProcessor(i, src, person_det, config_dict)
        if not p.stream.stopped:
            processors.append(p)

    if not processors:
        print("No cameras available. Exiting.")
        return

    os.makedirs("captured_videos", exist_ok=True)

    print("\n--- Multi-Camera System Active ---")
    print("Press 'q' at any time to quit.")

    # We will display cameras in a grid.
    cv2.namedWindow('Multi-Camera Feed', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Multi-Camera Feed', 1280, 720)

    try:
        while True:
            viz_frames = []
            for p in processors:
                p.process_frame()
                if p.viz_frame is not None:
                    # Resize to 640x360 for grid rendering
                    small_viz = cv2.resize(p.viz_frame, (640, 360))
                    viz_frames.append(small_viz)
            
            # Simple grid rendering (handles up to 4 cameras nicely)
            if viz_frames:
                if len(viz_frames) == 1:
                    grid = viz_frames[0]
                elif len(viz_frames) == 2:
                    grid = np.hstack((viz_frames[0], viz_frames[1]))
                elif len(viz_frames) == 3:
                    top = np.hstack((viz_frames[0], viz_frames[1]))
                    bot = np.hstack((viz_frames[2], np.zeros_like(viz_frames[0])))
                    grid = np.vstack((top, bot))
                elif len(viz_frames) >= 4:
                    top = np.hstack((viz_frames[0], viz_frames[1]))
                    bot = np.hstack((viz_frames[2], viz_frames[3]))
                    grid = np.vstack((top, bot))
                
                cv2.imshow('Multi-Camera Feed', grid)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        for p in processors:
            p.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
