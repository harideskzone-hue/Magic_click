"""
Job queue manager — SQLite backend (SJF scheduling).

All files (jobs.db, captured_videos/) are stored under the user-writable
app-data directory, never inside the root-owned /Applications/MagicClick/.
"""
import sqlite3
import os
import sys
import time

# ── User data directory (same logic as bootstrap.py) ─────────────────────────
def _user_data_dir() -> str:
    if sys.platform == "darwin":
        return os.path.join(os.path.expanduser("~"), "Library", "Application Support", "MagicClick")
    elif sys.platform == "win32":
        return os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "MagicClick")
    else:
        return os.path.join(os.path.expanduser("~"), ".magic_click")

USER_DATA   = _user_data_dir()
VIDEOS_DIR  = os.path.join(USER_DATA, "captured_videos")
DB_PATH     = os.path.join(VIDEOS_DIR, "jobs.db")

os.makedirs(VIDEOS_DIR, exist_ok=True)


def _get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with _get_conn() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_path TEXT UNIQUE NOT NULL,
                frame_count INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'PENDING',
                created_at REAL NOT NULL,
                started_at REAL,
                completed_at REAL
            )
        ''')
        conn.commit()

def add_job(video_path: str, frame_count: int) -> "int | None":
    """Adds a new job to the queue."""
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            '''
            INSERT INTO jobs (video_path, frame_count, status, created_at)
            VALUES (?, ?, 'PENDING', ?)
            ''',
            (video_path, frame_count, time.time())
        )
        conn.commit()
        return cursor.lastrowid

def get_shortest_job():
    """Retrieves the pending job with the lowest frame count (SJF). Marks it RUNNING."""
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM jobs
            WHERE status = 'PENDING'
            ORDER BY frame_count ASC, created_at ASC
            LIMIT 1
        ''')
        row = cursor.fetchone()
        if row is None:
            return None
        job_id = row['id']
        cursor.execute('''
            UPDATE jobs
            SET status = 'RUNNING', started_at = ?
            WHERE id = ? AND status = 'PENDING'
        ''', (time.time(), job_id))
        if cursor.rowcount > 0:
            conn.commit()
            return dict(row)
        return None

def mark_job_completed(job_id: int):
    with _get_conn() as conn:
        conn.execute('''
            UPDATE jobs SET status = 'COMPLETED', completed_at = ? WHERE id = ?
        ''', (time.time(), job_id))
        conn.commit()

def mark_job_failed(job_id: int):
    with _get_conn() as conn:
        conn.execute('''
            UPDATE jobs SET status = 'FAILED', completed_at = ? WHERE id = ?
        ''', (time.time(), job_id))
        conn.commit()

def get_pending_count() -> int:
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM jobs WHERE status = 'PENDING'")
        return cursor.fetchone()[0]

# Initialize tables when imported
init_db()
