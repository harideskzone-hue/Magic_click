"""
db_uploader.py — Best-Frame → mc_database uploader
====================================================
Reads results.json from a scored session, picks the single highest-scoring
SCORED frame, encodes it as base64, and POSTs it to mc_database /api/add.

Called automatically by post_process_video.py — zero manual steps required.

Configuration (optional, via mc_engine/.env):
  MC_DATABASE_URL=http://localhost:5000   (default)
"""

import os
import json
import base64
import time
import logging

# Load .env if available
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

try:
    import requests  # type: ignore
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MC_DATABASE_URL = os.environ.get("MC_DATABASE_URL", "http://localhost:5001")
_MAX_RETRIES    = 3
_RETRY_BASE_S   = 1.0   # back-off: 1s → 2s → 4s


# ── Public API ────────────────────────────────────────────────────────────────

def upload_best_frame(session_scored_dir: str, session_name: str) -> dict | None:
    """
    Find the best-scored image in session_scored_dir/results.json, then upload
    it to mc_database via POST /api/add.

    Returns the API response dict on success, or None on failure.
    Called by post_process_video.py automatically — no manual trigger needed.
    """
    if not _HAS_REQUESTS:
        log.warning("[DB UPLOAD] 'requests' not installed. Skipping upload.")
        return None

    report_path = os.path.join(session_scored_dir, "results.json")
    if not os.path.exists(report_path):
        log.warning(f"[DB UPLOAD] results.json not found in {session_scored_dir}. Skipping.")
        return None

    # ── 1. Load results ───────────────────────────────────────────────────────
    try:
        with open(report_path, "r") as fp:
            results = json.load(fp)
    except Exception as exc:
        log.error(f"[DB UPLOAD] Failed to read results.json: {exc}")
        return None

    # ── 2. Pick best SCORED frame ─────────────────────────────────────────────
    best = _pick_best_frame(results)
    if best is None:
        log.info(f"[DB UPLOAD] No SCORED frames in session '{session_name}'. Nothing to upload.")
        return None

    fname  = best.get("image") or best.get("image_name", "")
    score  = float(best.get("final_score", 0.0))
    img_path = os.path.join(session_scored_dir, fname)

    # Fallback: look in the raw directory (sibling of scored dir)
    if not os.path.exists(img_path):
        raw_dir = session_scored_dir.replace("_scored", "_raw")
        img_path = os.path.join(raw_dir, fname)

    if not os.path.exists(img_path):
        log.warning(f"[DB UPLOAD] Best frame '{fname}' not found on disk. Skipping.")
        return None

    # ── 3. Encode ─────────────────────────────────────────────────────────────
    try:
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        img_data_url = f"data:image/jpeg;base64,{b64}"
    except Exception as exc:
        log.error(f"[DB UPLOAD] Failed to encode image: {exc}")
        return None

    # ── 4. Upload with retries ────────────────────────────────────────────────
    payload = {"img": img_data_url, "score": score}
    url = f"{MC_DATABASE_URL}/api/add"

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if data.get("success"):
                pid = data.get("person_id", "unknown")
                log.info(
                    f"[DB UPLOAD] OK  session='{session_name}'  "
                    f"frame='{fname}'  score={score:.1f}  person_id={pid}"
                )
                print(
                    f"[DB UPLOAD] ✓ Uploaded best frame '{fname}' (score={score:.1f}) "
                    f"→ person_id={pid}"
                )
                return data
            else:
                log.warning(f"[DB UPLOAD] API returned success=false: {data.get('error')}")
                return None

        except requests.exceptions.ConnectionError:
            wait = _RETRY_BASE_S * (2 ** (attempt - 1))
            if attempt < _MAX_RETRIES:
                log.warning(f"[DB UPLOAD] mc_database not reachable (attempt {attempt}). Retry in {wait:.0f}s...")
                time.sleep(wait)
            else:
                log.error(
                    f"[DB UPLOAD] FAILED after {_MAX_RETRIES} retries — "
                    f"mc_database unreachable at {MC_DATABASE_URL}"
                )
                print(f"[DB UPLOAD] ✗ Upload failed after {_MAX_RETRIES} retries (connection error)")
                return None

        except Exception as exc:
            log.error(f"[DB UPLOAD] Unexpected error on attempt {attempt}: {exc}")
            if attempt >= _MAX_RETRIES:
                print(f"[DB UPLOAD] ✗ Upload failed: {exc}")
                return None
            time.sleep(_RETRY_BASE_S)

    return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pick_best_frame(results: list) -> dict | None:
    """Return the result dict with the highest final_score among SCORED entries."""
    scored = [
        r for r in results
        if isinstance(r, dict)
        and r.get("status") == "SCORED"
        and r.get("final_score") is not None
    ]
    if not scored:
        return None
    return max(scored, key=lambda r: float(r["final_score"]))


# ── Quick sanity test (run directly) ─────────────────────────────────────────
if __name__ == "__main__":
    import sys, tempfile, struct

    # Create a minimal valid 1×1 JPEG for testing
    # JPEG bytes for a 1x1 white pixel
    TINY_JPG = bytes([
        0xFF,0xD8,0xFF,0xE0,0x00,0x10,0x4A,0x46,0x49,0x46,0x00,0x01,0x01,0x00,
        0x00,0x01,0x00,0x01,0x00,0x00,0xFF,0xDB,0x00,0x43,0x00,0x08,0x06,0x06,
        0x07,0x06,0x05,0x08,0x07,0x07,0x07,0x09,0x09,0x08,0x0A,0x0C,0x14,0x0D,
        0x0C,0x0B,0x0B,0x0C,0x19,0x12,0x13,0x0F,0x14,0x1D,0x1A,0x1F,0x1E,0x1D,
        0x1A,0x1C,0x1C,0x20,0x24,0x2E,0x27,0x20,0x22,0x2C,0x23,0x1C,0x1C,0x28,
        0x37,0x29,0x2C,0x30,0x31,0x34,0x34,0x34,0x1F,0x27,0x39,0x3D,0x38,0x32,
        0x3C,0x2E,0x33,0x34,0x32,0xFF,0xC0,0x00,0x0B,0x08,0x00,0x01,0x00,0x01,
        0x01,0x01,0x11,0x00,0xFF,0xC4,0x00,0x1F,0x00,0x00,0x01,0x05,0x01,0x01,
        0x01,0x01,0x01,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x02,
        0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0xFF,0xC4,0x00,0xB5,0x10,
        0x00,0x02,0x01,0x03,0x03,0x02,0x04,0x03,0x05,0x05,0x04,0x04,0x00,0x00,
        0x01,0x7D,0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,
        0x13,0x51,0x61,0x07,0x22,0x71,0x14,0x32,0x81,0x91,0xA1,0x08,0x23,0x42,
        0xB1,0xC1,0x15,0x52,0xD1,0xF0,0x24,0x33,0x62,0x72,0x82,0x09,0x0A,0x16,
        0x17,0x18,0x19,0x1A,0x25,0x26,0x27,0x28,0x29,0x2A,0x34,0x35,0x36,0x37,
        0x38,0x39,0x3A,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4A,0x53,0x54,0x55,
        0x56,0x57,0x58,0x59,0x5A,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6A,0x73,
        0x74,0x75,0x76,0x77,0x78,0x79,0x7A,0x83,0x84,0x85,0x86,0x87,0x88,0x89,
        0x8A,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9A,0xA2,0xA3,0xA4,0xA5,
        0xA6,0xA7,0xA8,0xA9,0xAA,0xB2,0xB3,0xB4,0xB5,0xB6,0xB7,0xB8,0xB9,0xBA,
        0xC2,0xC3,0xC4,0xC5,0xC6,0xC7,0xC8,0xC9,0xCA,0xD2,0xD3,0xD4,0xD5,0xD6,
        0xD7,0xD8,0xD9,0xDA,0xE1,0xE2,0xE3,0xE4,0xE5,0xE6,0xE7,0xE8,0xE9,0xEA,
        0xF1,0xF2,0xF3,0xF4,0xF5,0xF6,0xF7,0xF8,0xF9,0xFA,0xFF,0xDA,0x00,0x08,
        0x01,0x01,0x00,0x00,0x3F,0x00,0xFB,0xD2,0x8A,0x28,0x03,0xFF,0xD9
    ])

    with tempfile.TemporaryDirectory() as tmp:
        # Write test image
        img_path = os.path.join(tmp, "test_frame.jpg")
        with open(img_path, "wb") as f:
            f.write(TINY_JPG)

        # Write fake results.json
        results_path = os.path.join(tmp, "results.json")
        with open(results_path, "w") as f:
            json.dump([
                {"image": "test_frame.jpg", "status": "SCORED", "final_score": 78.5},
                {"image": "bad_frame.jpg",  "status": "REJECTED", "final_score": None},
            ], f)

        print(f"[TEST] Uploading to {MC_DATABASE_URL}/api/add ...")
        result = upload_best_frame(tmp, "test_session")
        if result:
            print(f"[TEST] PASSED — person_id={result.get('person_id')}")
        else:
            print("[TEST] FAILED or mc_database not running (expected if server is down)")
