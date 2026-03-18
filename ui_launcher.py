"""
ui_launcher.py — Magic Click Unified System Launcher
======================================================
Single-command entry point for the full pipeline:

    python ui_launcher.py

Starts (in order):
  1. mc_database  — FastAPI face-embedding server (port 5000)
  2. job_worker   — SJF video-scoring queue processor
  3. live_scorer  — Multi-camera person capture

Handles:
  - Active health-poll to confirm API is ready before proceeding
  - Stdout streaming with per-process prefixes
  - Process crash detection with automatic worker restart
  - Graceful shutdown on Ctrl-C or SIGTERM (kills all children)
"""

import subprocess
import sys
import time
import os
import signal
import threading
import webbrowser

# ── Config ────────────────────────────────────────────────────────────────────
API_URL          = "http://localhost:5001"
PIPELINE_URL     = f"{API_URL}/"          # existing mc_database dashboard
HEALTH_URL       = f"{API_URL}/api/health"
API_READY_TIMEOUT  = 60   # seconds to wait for mc_database to start
WORKER_RESTART_DELAY = 5  # seconds between worker crash restarts
BASE = os.path.dirname(os.path.abspath(__file__))

# Shared shutdown flag
_shutdown = threading.Event()

# Dynamically calculate the venv python path so it runs the AI models even if launched from a system context
IS_WINDOWS = sys.platform == "win32"
VENV_PYTHON = os.path.join(BASE, ".venv", "Scripts", "python.exe") if IS_WINDOWS else os.path.join(BASE, ".venv", "bin", "python3")
# Fallback to sys.executable just in case it's run without the bootstrapper
PYTHON_EXE = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable


# ─── Process factories ────────────────────────────────────────────────────────

def _start_api() -> subprocess.Popen:
    print("[Launcher] ▶  Starting mc_database (FastAPI, port 5001)…")
    return subprocess.Popen(
        [PYTHON_EXE, "-u", "run.py"],
        cwd=os.path.join(BASE, "mc_database"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def _start_worker() -> subprocess.Popen:
    print("[Launcher] ▶  Starting job_worker (SJF queue)…")
    return subprocess.Popen(
        [PYTHON_EXE, "-u", "job_worker.py"],
        cwd=os.path.join(BASE, "Temp_MCv2"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def _start_camera() -> subprocess.Popen:
    print("[Launcher] ▶  Starting live_scorer (camera capture)…")
    return subprocess.Popen(
        [PYTHON_EXE, "-u", "live_scorer.py"],
        cwd=os.path.join(BASE, "Temp_MCv2"),
        # camera uses OpenCV windows — don't capture stdout so it can interact
        stdout=None,
        stderr=None,
    )


# ─── Utilities ────────────────────────────────────────────────────────────────

def _tail(proc: subprocess.Popen, prefix: str):
    """Stream a process's stdout to the console with a prefix label."""
    try:
        if proc.stdout:
            for line in iter(proc.stdout.readline, ""):  # type: ignore
                if _shutdown.is_set():
                    break
                print(f"[{prefix}] {line}", end="", flush=True)
    except Exception:
        pass


def _wait_for_api(timeout: int = API_READY_TIMEOUT) -> bool:
    """
    Poll GET /api/health until it responds with 200.
    Returns True when ready, False on timeout.
    """
    try:
        import urllib.request, urllib.error  # stdlib only — no requests needed here
    except ImportError:
        return False

    deadline = time.time() + timeout
    attempt  = 0
    while time.time() < deadline and not _shutdown.is_set():
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        attempt += 1
        if attempt % 5 == 0:
            remaining = int(deadline - time.time())
            print(f"[Launcher] ⏳ Waiting for API… ({remaining}s remaining)")
        time.sleep(1)
    return False


def _monitor_worker(get_worker_fn):
    """
    Background thread: restarts the job_worker if it crashes unexpectedly.
    Stops when _shutdown is set.
    """
    while not _shutdown.is_set():
        proc = get_worker_fn()
        code = proc.wait()
        if _shutdown.is_set():
            break
        if code != 0:
            print(f"[Launcher] ⚠  job_worker exited (code {code}). Restarting in {WORKER_RESTART_DELAY}s…")
            time.sleep(WORKER_RESTART_DELAY)
            new_proc = _start_worker()
            # Swap the reference and tail it
            t = threading.Thread(target=_tail, args=(new_proc, "WORKER"), daemon=True)
            t.start()
            # Update the mutable reference via closure
            _worker_holder[0] = new_proc  # type: ignore
        else:
            break  # clean exit, don't restart


# ─── Graceful shutdown ────────────────────────────────────────────────────────

def _terminate_all(procs: list):
    """Send SIGTERM to all living child processes, then wait."""
    print("\n[Launcher] 🛑 Shutting down all services…")
    _shutdown.set()
    for p in procs:
        if p and p.poll() is None:
            try:
                p.terminate()
            except Exception:
                pass
    # Give them 5s to exit gracefully, then SIGKILL
    deadline = time.time() + 5
    for p in procs:
        if p:
            try:
                remaining = max(0.1, deadline - time.time())
                p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    p.kill()
                except Exception:
                    pass
    print("[Launcher] ✓  All services stopped.")


# ─── Main ─────────────────────────────────────────────────────────────────────

_worker_holder = [None]  # mutable reference for monitor thread


def main():
    print()
    print("╔══════════════════════════════════════════╗")
    print("║    Magic Click — Unified Pipeline v2     ║")
    print("╚══════════════════════════════════════════╝")
    print()

    procs = []
    cam_proc    = None
    worker_proc = None

    def handle_signal(sig, frame):
        _terminate_all(procs)
        sys.exit(0)

    signal.signal(signal.SIGINT,  handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # ── 1. Start mc_database ────────────────────────────────────────────────
    api_proc = _start_api()
    procs.append(api_proc)
    t_api = threading.Thread(target=_tail, args=(api_proc, "DB "), daemon=True)
    t_api.start()

    # ── 2. Wait for API readiness ───────────────────────────────────────────
    print(f"[Launcher] ⏳ Polling {HEALTH_URL} (timeout={API_READY_TIMEOUT}s)…")
    ready = _wait_for_api(API_READY_TIMEOUT)

    if not ready:
        print("[Launcher] ✗  mc_database did not start in time. Check mc_database/requirements.txt.")
        _terminate_all(procs)
        sys.exit(1)

    print(f"[Launcher] ✓  API ready at {API_URL}")

    # ── 3. Start job_worker ─────────────────────────────────────────────────
    worker_proc = _start_worker()
    _worker_holder[0] = worker_proc  # type: ignore
    procs.append(worker_proc)
    t_worker = threading.Thread(target=_tail, args=(worker_proc, "WORK"), daemon=True)
    t_worker.start()

    # Monitor + auto-restart worker if it crashes
    t_mon = threading.Thread(
        target=_monitor_worker,
        args=(lambda: _worker_holder[0],),
        daemon=True
    )
    t_mon.start()

    # ── 4. Open browser ─────────────────────────────────────────────────────
    print(f"[Launcher] 🌐 Opening pipeline dashboard → {PIPELINE_URL}")
    time.sleep(0.5)  # tiny delay so the OS focus lands correctly
    webbrowser.open(PIPELINE_URL)

    # ── 5. Start live_scorer ────────────────────────────────────────────────
    cam_proc = _start_camera()
    procs.append(cam_proc)

    print()
    print("┌─────────────────────────────────────────────┐")
    print(f"│  Dashboard  → {PIPELINE_URL:<30}│")
    print(f"│  API Docs   → {API_URL}/docs              │")
    print("│  Press Ctrl-C or close the camera window     │")
    print("│  to shut down everything cleanly.            │")
    print("└─────────────────────────────────────────────┘")
    print()

    # ── 6. Block until camera exits (user presses 'q') ──────────────────────
    try:
        cam_proc.wait()
    except KeyboardInterrupt:
        pass  # signal handler above will clean up

    _terminate_all(procs)


if __name__ == "__main__":
    main()
