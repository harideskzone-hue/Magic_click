"""
Magic Click Unified Bootstrapper  v3.0
=======================================
Cross-platform entry point for the Magic Click pipeline.

Features
--------
* Python 3.10+ guard shown as a native GUI dialog
* Lock-file prevents two simultaneous launches
* Internet connectivity check before first-run install
* Virtual-environment creation + dependency install (with retry)
* Premium dark-mode Tkinter progress window
* Structured log file written to magic_click_setup.log
* Hands off to ui_launcher.py via os.execv (macOS/Linux) or subprocess (Windows)
"""

import sys
import os
import subprocess
import threading
import time
import socket
import logging
import tempfile
import tkinter as tk
from tkinter import ttk, messagebox

# ── 0. Logging ─────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
# Log to /tmp so it is always writable regardless of install location permissions
LOG_PATH     = os.path.join(tempfile.gettempdir(), "magic_click_setup.log")
logging.basicConfig(
    filename=LOG_PATH,
    filemode="a",
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("bootstrap")
log.info("─" * 60)
log.info("Bootstrap started  (PID %s, Python %s)", os.getpid(), sys.version.split()[0])

# ── 1. Python version guard ────────────────────────────────────────────────────
if sys.version_info < (3, 10):
    _r = tk.Tk(); _r.withdraw()
    messagebox.showerror(
        "Python Version Error",
        f"Magic Click requires Python 3.10, 3.11, or 3.12.\n"
        f"You have Python {sys.version_info.major}.{sys.version_info.minor}.\n\n"
        "Please install Python 3.11 and try again.",
    )
    log.error("Unsupported Python version: %s", sys.version)
    sys.exit(1)

if sys.version_info >= (3, 13):
    _r = tk.Tk(); _r.withdraw()
    messagebox.showerror(
        "Python Version Error",
        f"Magic Click uses AI libraries which do not yet have stable builds for Python 3.13+.\n"
        f"You have Python {sys.version_info.major}.{sys.version_info.minor}.\n\n"
        "Please install Python 3.11 from python.org and try again.",
    )
    log.error("Unsupported Python version (too new): %s", sys.version)
    sys.exit(1)

# ── 2. Constants ───────────────────────────────────────────────────────────────
os.chdir(SCRIPT_DIR)

IS_WINDOWS   = sys.platform == "win32"
IS_MAC       = sys.platform == "darwin"
VENV_DIR     = os.path.join(SCRIPT_DIR, ".venv")
VENV_PYTHON  = (
    os.path.join(VENV_DIR, "Scripts", "python.exe") if IS_WINDOWS
    else os.path.join(VENV_DIR, "bin", "python3")
)
LOCK_FILE    = os.path.join(tempfile.gettempdir(), "magic_click.lock")
MAX_RETRIES  = 3       # pip install retry attempts
INTERNET_HOST = "8.8.8.8"
INTERNET_PORT = 53
INTERNET_TIMEOUT = 5   # seconds

# ── 3. Duplicate-launch lock ───────────────────────────────────────────────────
def _acquire_lock() -> bool:
    """Return True if this process owns the lock, False if another is running."""
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE) as f:
                pid = int(f.read().strip())
            # On Unix psutil is optional — use /proc or kill(pid,0) as a fallback
            try:
                os.kill(pid, 0)          # raises OSError if process doesn't exist
                return False             # another instance is alive
            except OSError:
                pass                     # stale lock – reclaim it
        except (ValueError, OSError):
            pass
    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))
    log.info("Lock acquired: %s", LOCK_FILE)
    return True

def _release_lock():
    try:
        os.remove(LOCK_FILE)
        log.info("Lock released.")
    except OSError:
        pass

# ── 4. Internet check ──────────────────────────────────────────────────────────
def _check_internet() -> bool:
    try:
        socket.setdefaulttimeout(INTERNET_TIMEOUT)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(
            (INTERNET_HOST, INTERNET_PORT)
        )
        log.info("Internet connectivity: OK")
        return True
    except Exception as exc:
        log.warning("Internet check failed: %s", exc)
        return False

# ── 5. Premium GUI — Dark Cobalt theme matching the Magic Click dashboard ──────
BG      = "#0D1117"      # deep near-black background
SURFACE = "#161B22"      # card surface
BORDER  = "#21262D"      # subtle border
FG      = "#E6EDF3"      # primary text
FG_DIM  = "#8B949E"      # muted text
ACCENT  = "#2F81F7"      # cobalt blue accent
SUCCESS = "#3FB950"      # lush green
ERROR   = "#F85149"      # vibrant red
ACC_DARK= "#1C6FCA"      # darker cobalt for pressed state


class BootstrapGUI:
    """Dark cobalt animated launcher window — matches Magic Click dashboard theme."""

    STAGES = [
        ("🔍", "System Check"),
        ("⚙️",  "Environment"),
        ("📦", "Packages"),
        ("🧠", "AI Models"),
        ("🚀", "Launching"),
    ]
    # Estimated seconds per stage (for time-estimate label)
    _STAGE_TIMES = [2, 5, 180, 60, 5]

    # Spinner dot frames
    _DOTS = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Magic Click — Setup")
        self.root.geometry("560x310")
        self.root.resizable(False, False)
        self.root.configure(bg=BG)
        self.root.eval("tk::PlaceWindow . center")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ttk style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame",       background=BG)
        style.configure("Card.TFrame",  background=SURFACE)
        style.configure("TLabel",       background=BG, foreground=FG)
        style.configure(
            "TProgressbar",
            troughcolor=BORDER,
            background=ACCENT,
            bordercolor=BG,
            lightcolor=ACCENT,
            darkcolor=ACCENT,
            thickness=10,
        )

        outer = tk.Frame(self.root, bg=BG, padx=28, pady=20)
        outer.pack(fill=tk.BOTH, expand=True)

        # ── Logo + title row ────────────────────────────────────────────────────
        top = tk.Frame(outer, bg=BG)
        top.pack(fill=tk.X, pady=(0, 16))

        # Animated pulsing bolt
        self._bolt_lbl = tk.Label(top, text="⚡", font=("Helvetica", 32), bg=BG, fg=ACCENT)
        self._bolt_lbl.pack(side=tk.LEFT, padx=(0, 14))

        title_col = tk.Frame(top, bg=BG)
        title_col.pack(side=tk.LEFT)
        tk.Label(title_col, text="Magic Click",
                 font=("Helvetica", 22, "bold"), bg=BG, fg=FG).pack(anchor=tk.W)
        tk.Label(title_col, text="AI vision pipeline · setting up your environment",
                 font=("Helvetica", 9), bg=BG, fg=FG_DIM).pack(anchor=tk.W)

        # Spinner dot in top-right
        self._spinner_var = tk.StringVar(value=self._DOTS[0])
        tk.Label(top, textvariable=self._spinner_var,
                 font=("Courier", 18), bg=BG, fg=ACCENT).pack(side=tk.RIGHT)

        # ── 5-stage progress strip ───────────────────────────────────────────────
        strip = tk.Frame(outer, bg=BG)
        strip.pack(fill=tk.X, pady=(0, 14))
        self._stage_badges: list[tk.Label] = []
        self._stage_txts:   list[tk.Label] = []
        for i, (icon, label) in enumerate(self.STAGES):
            col = tk.Frame(strip, bg=BG)
            col.pack(side=tk.LEFT, padx=(0, 16))
            badge = tk.Label(col, text=icon, font=("Helvetica", 18),
                             bg=BORDER, fg=FG_DIM, padx=4, pady=2, relief="flat")
            badge.pack(anchor=tk.W)
            txt = tk.Label(col, text=label, font=("Helvetica", 8),
                           bg=BG, fg=FG_DIM)
            txt.pack(anchor=tk.W)
            self._stage_badges.append(badge)
            self._stage_txts.append(txt)

        # ── Status label with animated spinner prefix ────────────────────────────
        self._status_var = tk.StringVar(value="Initialising…")
        self._status_lbl = tk.Label(outer, textvariable=self._status_var,
                                    font=("Helvetica", 11, "bold"), bg=BG, fg=FG,
                                    anchor=tk.W)
        self._status_lbl.pack(fill=tk.X, pady=(0, 6))

        # ── Progress bar + pct + time estimate ──────────────────────────────────
        bar_row = tk.Frame(outer, bg=BG)
        bar_row.pack(fill=tk.X, pady=(0, 6))
        self._progress = ttk.Progressbar(
            bar_row,
            mode="determinate", maximum=100.0,
        )
        self._progress.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._pct_var = tk.StringVar(value="0%")
        tk.Label(bar_row, textvariable=self._pct_var,
                 font=("Menlo", 10, "bold"), bg=BG, fg=ACCENT, width=5
                 ).pack(side=tk.RIGHT, padx=(8, 0))
        self._progress["value"] = 0.0

        # Time estimate
        self._eta_var = tk.StringVar(value="")
        tk.Label(outer, textvariable=self._eta_var,
                 font=("Helvetica", 9), bg=BG, fg=FG_DIM, anchor=tk.W
                 ).pack(fill=tk.X, pady=(0, 4))

        # ── Detail log line ──────────────────────────────────────────────────────
        self._detail_var = tk.StringVar(value="")
        tk.Label(outer, textvariable=self._detail_var,
                 font=("Menlo", 8), bg=BG, fg=FG_DIM,
                 anchor=tk.W, wraplength=500).pack(fill=tk.X)

        self._closing = False
        self._dot_idx = 0
        self._current_stage = 0
        self._stage_start = time.time()
        self._animate()

    # ── Animation loop ──────────────────────────────────────────────────────────
    def _animate(self):
        if self._closing:
            return
        self._dot_idx = (self._dot_idx + 1) % len(self._DOTS)
        self._spinner_var.set(self._DOTS[self._dot_idx])

        # Pulse active stage badge between SURFACE and a cobalt tint
        pulse_on = (self._dot_idx % 4 < 2)
        if self._current_stage < len(self._stage_badges):
            b = self._stage_badges[self._current_stage]
            b.config(bg=ACC_DARK if pulse_on else ACCENT)

        # Time estimate
        elapsed = time.time() - self._stage_start
        stage_est = self._STAGE_TIMES[min(self._current_stage, len(self._STAGE_TIMES)-1)]
        remaining = max(0, stage_est - int(elapsed))
        if remaining > 0:
            self._eta_var.set(f"⏱  ~{remaining}s remaining in this stage")
        else:
            self._eta_var.set("⏳  Almost done — please wait…")

        self.root.after(100, self._animate)

    # ── public API ──────────────────────────────────────────────────────────────
    def set_stage(self, index: int):
        self.root.after(0, self._apply_stage, index)

    def _apply_stage(self, index: int):
        self._current_stage = index
        self._stage_start = time.time()
        for i, (badge, txt) in enumerate(zip(self._stage_badges, self._stage_txts)):
            if i < index:      # done — green
                badge.config(bg=SUCCESS,  fg="#0D1117")
                txt.config(fg=SUCCESS)
            elif i == index:   # active — cobalt
                badge.config(bg=ACCENT,   fg="#FFFFFF")
                txt.config(fg=FG)
            else:              # pending — dim
                badge.config(bg=BORDER,   fg=FG_DIM)
                txt.config(fg=FG_DIM)

    def update_status(self, main: str, detail: str = ""):
        self.root.after(0, self._set_labels, main, detail)

    def _set_labels(self, main: str, detail: str):
        self._status_var.set(main)
        self._detail_var.set(detail)

    def set_progress(self, value: float):
        self.root.after(0, self._set_progress, value)

    def _set_progress(self, value: float):
        v = max(0.0, min(100.0, float(value)))
        self._progress["value"] = v
        self._pct_var.set(f"{int(v)}%")
        if v >= 100:
            s = ttk.Style()
            s.configure("TProgressbar", background=SUCCESS,
                         lightcolor=SUCCESS, darkcolor=SUCCESS)

    def error_and_exit(self, title: str, message: str):
        self.root.after(0, self._show_error, title, message)

    def _show_error(self, title: str, message: str):
        messagebox.showerror(title, message)
        self._closing = True
        self.root.destroy()
        _release_lock()
        sys.exit(1)

    def close(self):
        self.root.after(500, self._do_close)  # type: ignore[arg-type]

    def _do_close(self):
        self._closing = True
        self.root.destroy()

    def _on_close(self):
        if not self._closing:
            if messagebox.askyesno(
                "Cancel Setup?",
                "Magic Click setup is in progress.\n\nAre you sure you want to quit?",
            ):
                _release_lock()
                self.root.destroy()
                sys.exit(0)


# ── 6. First-run dialog (shown before GUI, before venv exists) ─────────────────
def _show_first_run_dialog() -> bool:
    """
    Returns True  → user wants to proceed.
    Returns False → user chose Cancel.
    Native macOS AppleScript; Tkinter fallback elsewhere.
    """
    title   = "Magic Click — First Run"
    heading = "Welcome to Magic Click!"
    body    = (
        "This is your first time launching Magic Click.\n\n"
        "The setup wizard will:\n"
        "  • Create an isolated Python environment\n"
        "  • Download AI models & packages (~500 MB)\n"
        "  • Launch the camera dashboard in your browser\n\n"
        "An internet connection is required for the initial setup.\n"
        "Subsequent launches will be instant."
    )

    if IS_MAC:
        script = (
            f'button returned of (display dialog "{body}" '
            f'buttons {{"Cancel", "▶  Launch"}} '
            f'default button "▶  Launch" '
            f'with title "{heading}")'
        )
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True,
        )
        log.info("AppleScript dialog result: %r", result.stdout.strip())
        return result.stdout.strip() == "▶  Launch"

    # Universal fallback — Tkinter dialog
    root = tk.Tk(); root.withdraw()
    ok = messagebox.askokcancel(heading, body)
    root.destroy()
    return ok


# ── 7. Setup worker ───────────────────────────────────────────────────────────
def run_setup(gui: BootstrapGUI):
    """Runs in a background thread: creates venv, installs deps, launches."""

    mc_reqs       = os.path.join(SCRIPT_DIR, "mc_database", "requirements.txt")
    pipeline_reqs = os.path.join(SCRIPT_DIR, "mc_engine",    "requirements.txt")

    # ── helper: run a command with retry ──────────────────────────────────────
    def run_cmd(cmd, stage_desc: str, start_pct: float, end_pct: float) -> int:
        for attempt in range(1, MAX_RETRIES + 1):
            log.info("CMD attempt %d/%d: %s", attempt, MAX_RETRIES, cmd)
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True, bufsize=1, universal_newlines=True,
            )
            current_pct = float(start_pct)
            gui.set_progress(current_pct)

            if process.stdout:
                for line in process.stdout:  # type: ignore[attr-defined]
                    remaining    = float(end_pct) - current_pct
                    current_pct += remaining * 0.02
                    gui.set_progress(current_pct)
                    clean = line.strip()
                    if clean:
                        if len(clean) > 70:
                            clean = "…" + clean[-68:]  # type: ignore[index]
                        log.debug("[pip] %s", clean)
                        gui.update_status(stage_desc, clean)

            code = process.wait()
            gui.set_progress(end_pct)

            if code == 0:
                log.info("CMD succeeded (attempt %d)", attempt)
                return 0

            log.warning("CMD failed (code %d, attempt %d)", code, attempt)
            if attempt < MAX_RETRIES:
                gui.update_status(
                    f"⚠  Retry {attempt}/{MAX_RETRIES - 1}…",
                    "Retrying in 3 seconds…",
                )
                time.sleep(3)

        return code   # non-zero after all retries

    # ── Stage 0: environment ───────────────────────────────────────────────────
    try:
        gui.set_stage(0)

        if not os.path.exists(VENV_DIR) or not os.path.exists(VENV_PYTHON):
            gui.update_status("Creating virtual environment…", "One moment…")
            gui.set_progress(3)
            log.info("Creating venv at %s", VENV_DIR)
            import venv  # type: ignore[import]
            venv.create(VENV_DIR, with_pip=True)
            log.info("Venv created OK")
        gui.set_progress(10)

        # ── Stage 1: packages ──────────────────────────────────────────────────
        gui.set_stage(1)

        # Upgrade pip
        code = run_cmd(
            [VENV_PYTHON, "-m", "pip", "install", "--upgrade", "pip"],
            "Preparing installer…", 10, 18,
        )
        if code != 0:
            gui.error_and_exit(
                "Pip Upgrade Failed",
                "Could not upgrade pip.\n\n"
                "Please check your internet connection and try again.\n"
                f"Details are in: {LOG_PATH}",
            )
            return

        # mc_database requirements
        if os.path.exists(mc_reqs):
            code = run_cmd(
                [VENV_PYTHON, "-m", "pip", "install", "--timeout", "180", "-r", mc_reqs],
                "Installing core database module…", 18, 45,
            )
            if code != 0:
                gui.error_and_exit(
                    "Installation Failed",
                    "Failed to install database requirements after "
                    f"{MAX_RETRIES} attempts.\n\n"
                    "Please check your internet connection and try again.\n"
                    f"Full log: {LOG_PATH}",
                )
                return

        # AI pipeline requirements
        if os.path.exists(pipeline_reqs):
            code = run_cmd(
                [VENV_PYTHON, "-m", "pip", "install", "--timeout", "180", "-r", pipeline_reqs],
                "Installing AI models (this may take a few minutes)…", 45, 95,
            )
            if code != 0:
                gui.error_and_exit(
                    "Installation Failed",
                    "Failed to install AI pipeline requirements after "
                    f"{MAX_RETRIES} attempts.\n\n"
                    "Please check your internet connection and try again.\n"
                    f"Full log: {LOG_PATH}",
                )
                return

        # ── Download required AI model files if missing ────────────────────
        models_dir = os.path.join(SCRIPT_DIR, "mc_engine", "models")
        os.makedirs(models_dir, exist_ok=True)
        MODEL_URLS = {
            "yolo26n.pt":                "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
            "yolo26n-face.pt":           "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolo26n-face.pt",
            "face_landmarker.task":      "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        }
        for fname, url in MODEL_URLS.items():
            dest = os.path.join(models_dir, fname)
            if not os.path.exists(dest):
                gui.update_status(f"Downloading {fname}…", "One-time model download")
                gui.set_progress(90)
                log.info("Downloading model: %s → %s", url, dest)
                run_cmd(
                    [VENV_PYTHON, "-c",
                     f"import urllib.request; urllib.request.urlretrieve('{url}', '{dest}')"],
                    f"Downloading {fname}…", 90, 95,
                )
            else:
                log.info("Model already present: %s", fname)

        # ── Stage 2: launch ────────────────────────────────────────────────────
        gui.set_stage(2)
        gui.update_status("All done! Launching the pipeline…", "Opening dashboard…")
        gui.set_progress(100)
        log.info("Setup complete — handing off to ui_launcher.py")

        time.sleep(1.2)   # let green bar render
        gui.close()
        time.sleep(0.5)

        launcher = os.path.join(SCRIPT_DIR, "ui_launcher.py")
        if IS_WINDOWS:
            subprocess.run([VENV_PYTHON, launcher])
            sys.exit(0)
        else:
            os.execv(VENV_PYTHON, [VENV_PYTHON, launcher])

    except Exception as exc:
        log.exception("Unhandled error in run_setup")
        gui.error_and_exit(
            "Unexpected Error",
            f"An unexpected error occurred:\n\n{exc}\n\nFull log: {LOG_PATH}",
        )
    finally:
        _release_lock()


# ── 8. Main ────────────────────────────────────────────────────────────────────
def main():
    log.info("main() entered")

    # If we are already inside the venv, skip setup and launch directly
    if os.path.abspath(sys.executable) == os.path.abspath(VENV_PYTHON):
        log.info("Already in venv — launching directly")
        launcher = os.path.join(SCRIPT_DIR, "ui_launcher.py")
        if IS_WINDOWS:
            sys.exit(subprocess.run([sys.executable, launcher]).returncode)
        else:
            os.execv(sys.executable, [sys.executable, launcher])

    # Prevent duplicate launches
    if not _acquire_lock():
        root = tk.Tk(); root.withdraw()
        messagebox.showwarning(
            "Already Running",
            "Magic Click is already starting up in another window.\n\n"
            "Please wait for it to finish.",
        )
        root.destroy()
        log.warning("Duplicate launch blocked by lock file.")
        sys.exit(0)

    is_first_run = not os.path.exists(VENV_PYTHON)

    # Show the first-run welcome dialog
    if is_first_run:
        if not _check_internet():
            root = tk.Tk(); root.withdraw()
            messagebox.showerror(
                "No Internet Connection",
                "Magic Click needs to download AI packages on the first run.\n\n"
                "Please connect to the internet and try again.",
            )
            root.destroy()
            log.error("First-run blocked: no internet connection.")
            _release_lock()
            sys.exit(1)

        if not _show_first_run_dialog():
            log.info("User cancelled at first-run dialog.")
            _release_lock()
            sys.exit(0)

    # Show premium GUI and run setup in background thread
    gui = BootstrapGUI()
    threading.Thread(target=run_setup, args=(gui,), daemon=True).start()
    gui.root.mainloop()


if __name__ == "__main__":
    main()
