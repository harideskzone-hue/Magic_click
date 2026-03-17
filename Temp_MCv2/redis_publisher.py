"""
redis_publisher.py — System A (AI Machine)
==========================================
Automatically called by post_process_video.py after each scoring session.
Reads the results.json from the scored session directory and publishes
each scored image to the Redis channel 'mc:scores' on the remote server.

CONFIGURATION:
  Set REDIS_HOST in .env (or as an environment variable) to point to
  the IP address of the remote machine (System B).

  Example:  REDIS_HOST=192.168.1.50
"""

import os
import json
import time
import socket
import logging

# Load .env if present (dotenv is optional — falls back to os.environ)
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

# ── Configuration ─────────────────────────────────────────────────────────────
REDIS_HOST    = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT    = int(os.environ.get("REDIS_PORT", 6379))
REDIS_CHANNEL = os.environ.get("REDIS_CHANNEL", "mc:scores")

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[Publisher] %(message)s")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_redis_client():
    """Return a Redis client. Returns None (with a warning) if unavailable."""
    try:
        import redis
        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
            socket_connect_timeout=3,   # fail fast — don't block the AI pipeline
            socket_timeout=5,
        )
        client.ping()   # confirm the connection is live
        return client
    except ImportError:
        log.warning("'redis' package not installed. Run: pip install redis")
        return None
    except Exception as exc:
        log.warning(
            f"Redis not reachable at {REDIS_HOST}:{REDIS_PORT} — {exc}. "
            "Skipping publish. Pipeline continues normally."
        )
        return None


# ── Main API ──────────────────────────────────────────────────────────────────
def publish_session_results(session_scored_dir: str, session_name: str) -> None:
    """
    Read results.json from session_scored_dir and publish all scored
    images to Redis. Called automatically by post_process_video.py.

    Parameters
    ----------
    session_scored_dir : str
        Path to the directory containing results.json produced by score_folder.py.
    session_name : str
        Unique session identifier (e.g. 'session_1741234567890').
    """
    report_path = os.path.join(session_scored_dir, "results.json")
    if not os.path.exists(report_path):
        log.warning(f"results.json not found in {session_scored_dir}. Nothing to publish.")
        return

    try:
        with open(report_path, "r") as fp:
            results = json.load(fp)
    except Exception as exc:
        log.error(f"Failed to read results.json: {exc}")
        return

    client = _get_redis_client()
    if client is None:
        return  # Graceful degradation — pipeline does NOT crash

    published   = 0
    system_name = socket.gethostname()

    try:
        for entry in results:
            # Build a clean, minimal payload
            payload = {
                "event":        "score_result",
                "session":      session_name,
                "machine":      system_name,
                "timestamp":    entry.get("timestamp", time.time()),
                "image":        entry.get("image") or entry.get("image_name", "unknown"),
                "status":       entry.get("status", "UNKNOWN"),
                "final_score":  entry.get("final_score"),
                "band":         entry.get("band", ""),
                "face_score":   entry.get("face_score"),
                "body_score":   entry.get("body_score"),
                "frame_score":  entry.get("frame_score"),
                "reject_reason": entry.get("reject_reason", ""),
            }
            client.publish(REDIS_CHANNEL, json.dumps(payload))
            published += 1

        # Publish a session-done sentinel so the dashboard knows the session ended
        done_payload = {
            "event":   "session_done",
            "session": session_name,
            "machine": system_name,
            "total":   published,
            "timestamp": time.time(),
        }
        client.publish(REDIS_CHANNEL, json.dumps(done_payload))

        log.info(
            f"Published {published} results for session '{session_name}' "
            f"to Redis {REDIS_HOST}:{REDIS_PORT} / channel '{REDIS_CHANNEL}'"
        )

    except Exception as exc:
        log.error(f"Error while publishing to Redis: {exc}")
    finally:
        try:
            client.close()
        except Exception:
            pass


# ── Quick test (run directly) ─────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        publish_session_results(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python redis_publisher.py <scored_dir> <session_name>")
        print(f"Current config: REDIS_HOST={REDIS_HOST}  PORT={REDIS_PORT}  CHANNEL={REDIS_CHANNEL}")
