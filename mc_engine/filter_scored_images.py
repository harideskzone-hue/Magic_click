import cv2  # type: ignore
import os
import shutil
import json
import base64
import requests  # type: ignore

# Load .env so MC_DATABASE_URL can be configured without touching source code
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

_DB_URL = os.environ.get("MC_DATABASE_URL", "http://localhost:5001")
_MIN_SCORE_DEFAULT = int(os.environ.get("MC_MIN_SCORE", "60"))


def _queue_failed_upload(output_dir: str, fname: str, score: float):
    """Persist a failed upload to a local JSON queue for later retry."""
    import time as _t
    queue_path = os.path.join(output_dir, "failed_uploads.json")
    queue = []
    if os.path.exists(queue_path):
        try:
            with open(queue_path, "r") as f:
                queue = json.load(f)
        except Exception:
            pass
    queue.append({"image": fname, "score": score, "timestamp": _t.time()})
    try:
        with open(queue_path, "w") as f:
            json.dump(queue, f, indent=2)
        print(f"  [QUEUE] Saved {fname} to {queue_path} for later retry")
    except Exception as e:
        print(f"  [QUEUE ERROR] Could not save {fname}: {e}")


def filter_scored_images(input_dir, source_dir, output_dir, min_score=None, report_path=None):
    """
    Reads scoring results from report_path JSON, checks for SCORED status and min_score,
    then copies the corresponding original image from source_dir to output_dir.
    """
    if min_score is None:
        min_score = _MIN_SCORE_DEFAULT
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    passed_count: int  = 0
    missing_count: int = 0

    if not report_path or not os.path.exists(report_path):
        print(f"Report not found: {report_path}")
        return

    with open(report_path, 'r') as fp:
        results = json.load(fp)

    print(f"Filtering {len(results)} images from report...")

    for res in results:
        if res.get('status') != 'SCORED':
            continue
        
        score = res.get('final_score')
        if score is None or score < min_score:
            continue

        original_fname = res.get('image') or res.get('image_name')
        if not original_fname:
            continue

        src_path = os.path.join(source_dir, original_fname)

        if not os.path.exists(src_path):
            print(f"  [MISSING] {original_fname} not found in source_dir")
            missing_count += 1  # type: ignore
            continue

        # ── Copy raw image as-is, no overlay ──────────────────────────────
        _save_with_score(src_path, os.path.join(output_dir, original_fname), score)
        passed_count += 1  # type: ignore
        try:
            with open(src_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

            payload = {
                "img": f"data:image/jpeg;base64,{encoded_string}",
                "score": float(score),
            }
            upload_url = f"{_DB_URL}/api/add"

            # Read internal token (same approach as db_uploader.py)
            headers = {}
            import sys
            _usr_dir = os.environ.get("MAGIC_CLICK_DATA")
            if not _usr_dir:
                _usr_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
            
            _secret_path = os.path.join(_usr_dir, ".session_secret")
            
            try:
                with open(_secret_path, "r") as _sf:
                    headers["x-internal-token"] = _sf.read().strip()
            except Exception:
                pass  # Launcher may not have token yet; API will reject with 401

            r = None
            _MAX_UPLOAD_RETRIES = 3
            for attempt in range(1, _MAX_UPLOAD_RETRIES + 1):
                try:
                    r = requests.post(upload_url, json=payload, headers=headers, timeout=10)
                    r.raise_for_status()
                    print(f"  [DB UPLOAD OK] {original_fname} → {upload_url} (score:{score:.1f})")
                    break
                except Exception as upload_err:
                    if attempt < _MAX_UPLOAD_RETRIES:
                        wait = 2 ** attempt
                        print(f"  [DB UPLOAD RETRY {attempt}/{_MAX_UPLOAD_RETRIES}] {original_fname}: {upload_err} (retrying in {wait}s)")
                        import time as _time
                        _time.sleep(wait)
                    else:
                        detail = ""
                        if r is not None:
                            try:
                                resp_text = str(getattr(r, 'text', ''))
                                detail = f" body={resp_text[:200]}"  # type: ignore
                            except Exception:
                                pass
                        print(f"  [DB UPLOAD FAIL] {original_fname}: {upload_err}{detail}")
                        # Queue for later retry
                        _queue_failed_upload(output_dir, original_fname, float(score))
        except Exception as e:
            print(f"  [DB ERROR] {original_fname}: {e}")

    print(f"\nDone! {passed_count} raw images (score >= {min_score}) saved to '{output_dir}'.")
    if missing_count:
        print(f"  {missing_count} skipped — original not found in source_dir.")

def _save_with_score(src_path, out_path, score):
    img = cv2.imread(src_path)
    if img is None:
        shutil.copy2(src_path, out_path)
        return

    h, w = img.shape[:2]

    # Semi-transparent dark pill
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (220, 60), (8, 10, 16), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    # Score text
    cv2.putText(img, f"Score: {score:.1f}", (20, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (40, 220, 80), 2, cv2.LINE_AA)

    cv2.imwrite(out_path, img)


if __name__ == "__main__":
    in_dir      = r"C:\projects\temp_magikCLick\batch_viz_3"
    source_dir  = r"C:\projects\temp_magikCLick\captured_videos\all_frames"
    out_dir     = r"C:\projects\temp_magikCLick\batch_viz_3\scored_raw"
    report_path = r"C:\projects\temp_magikCLick\batch_viz_3\results.json"  # ← was None before

    if os.path.exists(in_dir):
        filter_scored_images(in_dir, source_dir, out_dir, min_score=75, report_path=report_path)
    else:
        print(f"Input directory not found: {in_dir}")