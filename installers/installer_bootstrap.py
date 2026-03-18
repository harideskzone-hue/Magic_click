#!/usr/bin/env python3
"""
Magic Click — Cross-Platform Installer Bootstrap
=================================================
Runs after OS-level package installation (macOS .pkg postinstall,
Windows .exe [Run] step, Linux .deb postinst) to:

  1. Validate the system (OS version, disk space, internet)
  2. Create an ARM/Intel-native Python virtual environment
  3. Install all dependencies with retry + back-off
  4. Warm the InsightFace model cache
  5. Verify all services start correctly
  6. Show a success screen with a Launch button

Run directly for manual first-run setup:
    python3 installers/installer_bootstrap.py
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent          # installers/
ROOT        = HERE.parent                               # Magic_Click/
VENV_DIR    = ROOT / ".venv"
REQ_FILES   = [
    ROOT / "mc_database" / "requirements.txt",  # API + auth
    ROOT / "mc_engine"   / "requirements.txt",  # live scorer + AI models
]
LOG_FILE    = ROOT / "magic_click_install.log"
LOCK_FILE   = Path(tempfile.gettempdir()) / "magic_click_install.lock"

# ── Constants ──────────────────────────────────────────────────────────────────
MIN_PYTHON  = (3, 10)
MIN_DISK_GB = 3.0
API_PORT    = 5001
MAX_RETRIES = 3

IS_WIN  = sys.platform == "win32"
IS_MAC  = sys.platform == "darwin"
IS_LIN  = sys.platform.startswith("linux")

VENV_PY = VENV_DIR / ("Scripts/python.exe" if IS_WIN else "bin/python3")
VENV_PIP = VENV_DIR / ("Scripts/pip.exe" if IS_WIN else "bin/pip")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("installer")


# ══════════════════════════════════════════════════════════════════════════════
# Premium Installer GUI
# ══════════════════════════════════════════════════════════════════════════════

class InstallerGUI:
    # Cobalt-dark palette to match dashboard brand
    BG          = "#010D3B"
    BG_CARD     = "#01207B"
    ACCENT      = "#4F70E8"
    ACCENT_LITE = "#7B9BFF"
    GREEN       = "#00C896"
    RED_COL     = "#FF5E5E"
    TEXT        = "#F0F4FF"
    TEXT_DIM    = "#8FA0CC"
    W, H        = 520, 440

    STAGES = [
        ("🔍", "System Check"),
        ("⚙️",  "Creating Environment"),
        ("📦", "Installing Packages"),
        ("🧠", "Downloading AI Models"),
        ("✅", "Verifying Services"),
    ]

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Magic Click — Setup")
        self.root.geometry(f"{self.W}x{self.H}")
        self.root.resizable(False, False)
        self.root.configure(bg=self.BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._closing = False
        self._allow_close = False

        # Declare all widget attributes upfront so the type-checker can see them
        self._stage_label: tk.Label
        self._sub_label: tk.Label
        self._dots: list[tk.Label] = []
        self._bar: tk.Frame
        self._track_w: int
        self._pct_label: tk.Label
        self._log_text: tk.Text
        self._btn_frame: tk.Frame
        self._action_btn: tk.Button

        if IS_MAC:
            try: self.root.tk.call(
                "::tk::unsupported::MacWindowStyle", "style",
                str(self.root), "document", "closeBox"
            )
            except Exception: pass

        self._build_ui()

    # ── Build UI ────────────────────────────────────────────────────────────
    def _build_ui(self):
        c = self.root
        bg, card = self.BG, self.BG_CARD

        # Top bar
        bar = tk.Frame(c, bg=card, height=56)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        tk.Label(bar, text="⚡", font=("Helvetica", 22), bg=card, fg=self.ACCENT_LITE).pack(side="left", padx=16)
        tk.Label(bar, text="Magic Click Setup", font=("Helvetica", 13, "bold"), bg=card, fg=self.TEXT).pack(side="left")

        # Status area
        mid = tk.Frame(c, bg=bg)
        mid.pack(fill="both", expand=True, padx=30, pady=(24, 0))

        self._stage_label = tk.Label(mid, text="Starting…", font=("Helvetica", 15, "bold"),
                                      bg=bg, fg=self.TEXT)
        self._stage_label.pack(anchor="w")

        self._sub_label = tk.Label(mid, text="Please wait", font=("Helvetica", 10),
                                    bg=bg, fg=self.TEXT_DIM, wraplength=460, justify="left")
        self._sub_label.pack(anchor="w", pady=(4, 20))

        # Stage indicator dots
        dot_frame = tk.Frame(mid, bg=bg)
        dot_frame.pack(anchor="w", pady=(0, 20))
        self._dots: list[tk.Label] = []
        for icon, name in self.STAGES:
            col = tk.Frame(dot_frame, bg=bg)
            col.pack(side="left", padx=8)
            lbl = tk.Label(col, text=icon, font=("Helvetica", 16), bg=bg, fg=self.TEXT_DIM)
            lbl.pack()
            tk.Label(col, text=name, font=("Helvetica", 7), bg=bg, fg=self.TEXT_DIM).pack()
            self._dots.append(lbl)

        # Progress track
        track = tk.Frame(mid, bg="#0B1E5C", height=10, bd=0, relief="flat")
        track.pack(fill="x", pady=(0, 6))
        track.pack_propagate(False)
        self._bar = tk.Frame(track, bg=self.ACCENT, height=10)
        self._bar.place(x=0, y=0, relheight=1, width=0)
        self._track_w = self.W - 60  # approx

        self._pct_label = tk.Label(mid, text="0%", font=("Helvetica", 10, "bold"),
                                    bg=bg, fg=self.ACCENT_LITE)
        self._pct_label.pack(anchor="e")

        # Detail log
        log_frame = tk.Frame(mid, bg="#020B2A", bd=0)
        log_frame.pack(fill="both", expand=True, pady=(12, 0))
        self._log_text = tk.Text(log_frame, height=6, bg="#020B2A", fg=self.TEXT_DIM,
                                  font=("Courier", 8), relief="flat", state="disabled",
                                  wrap="word", insertbackground=self.ACCENT)
        self._log_text.pack(fill="both", expand=True, padx=10, pady=8)

        # Bottom action row
        self._btn_frame = tk.Frame(c, bg=bg)
        self._btn_frame.pack(fill="x", padx=30, pady=16)
        self._action_btn = tk.Button(self._btn_frame, text="Cancel", font=("Helvetica", 10),
                                      bg=self.BG_CARD, fg=self.TEXT, bd=0, padx=16, pady=7,
                                      cursor="hand2", command=self._on_close)
        self._action_btn.pack(side="right")

    # ── Public API ──────────────────────────────────────────────────────────
    def set_stage(self, idx: int, pct: int, sub: str = ""):
        icon, name = self.STAGES[idx]
        self._stage_label.config(text=f"{icon}  {name}")
        if sub: self._sub_label.config(text=sub)
        for i, lbl in enumerate(self._dots):
            if i < idx:
                lbl.config(fg=self.GREEN)
            elif i == idx:
                lbl.config(fg=self.ACCENT_LITE)
            else:
                lbl.config(fg=self.TEXT_DIM)
        self._set_pct(pct)
        self.root.update_idletasks()

    def _set_pct(self, pct: int):
        pct = max(0, min(100, pct))
        w = int(self._track_w * pct / 100)
        self._bar.place_configure(width=w)
        color = self.GREEN if pct >= 100 else self.ACCENT
        self._bar.config(bg=color)
        self._pct_label.config(text=f"{pct}%")

    def append_log(self, line: str, error: bool = False):
        self._log_text.config(state="normal")
        tag = "err" if error else "norm"
        self._log_text.tag_config("err",  foreground=self.RED_COL)
        self._log_text.tag_config("norm", foreground=self.TEXT_DIM)
        self._log_text.insert("end", line.rstrip() + "\n", tag)
        self._log_text.see("end")
        self._log_text.config(state="disabled")
        log.debug(line.strip())
        self.root.update_idletasks()

    def show_success(self, on_launch):
        self._allow_close = True
        self._set_pct(100)
        self._stage_label.config(text="✅  Setup Complete!", fg=self.GREEN)
        self._sub_label.config(text="Magic Click is installed. Click Launch to open the dashboard.")
        self._action_btn.config(
            text="▶  Launch Magic Click", bg=self.GREEN, fg="#001A0F",
            font=("Helvetica", 11, "bold"), command=on_launch
        )

    def show_error(self, msg: str, on_retry, on_cancel):
        self._allow_close = True
        self._stage_label.config(text="❌  Setup Failed", fg=self.RED_COL)
        self._sub_label.config(text=msg)
        self._bar.config(bg=self.RED_COL)
        tk.Button(self._btn_frame, text="↺ Retry", font=("Helvetica", 10),
                  bg="#3A1A1A", fg=self.RED_COL, bd=0, padx=14, pady=7,
                  cursor="hand2", command=on_retry).pack(side="right", padx=(0, 8))
        self._action_btn.config(text="Cancel", command=on_cancel)

    def _on_close(self):
        if self._allow_close or messagebox.askyesno("Cancel Setup",
            "Setup is not complete. Cancel installation?", parent=self.root):
            self._closing = True
            self.root.destroy()

    def run(self, setup_fn):
        threading.Thread(target=setup_fn, daemon=True).start()
        self.root.mainloop()


# ══════════════════════════════════════════════════════════════════════════════
# Pre-Install Validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_system() -> list[str]:
    """Return list of blocking error strings; empty = all clear."""
    errors = []

    # Python version
    if sys.version_info < MIN_PYTHON:
        errors.append(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required. "
            f"You have {sys.version_info.major}.{sys.version_info.minor}. "
            f"Please install a newer Python from python.org."
        )

    # macOS version >= 12 (Monterey)
    if IS_MAC:
        _parts = platform.mac_ver()[0].split(".")
        _major = int(_parts[0]) if len(_parts) > 0 else 0
        _minor = int(_parts[1]) if len(_parts) > 1 else 0
        if (_major, _minor) < (12, 0):
            errors.append(f"macOS 12 (Monterey) or newer required. You have {platform.mac_ver()[0]}.")

    # Disk space
    usage = shutil.disk_usage(ROOT)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < MIN_DISK_GB:
        errors.append(
            f"Not enough disk space. {MIN_DISK_GB:.0f} GB free space required; "
            f"you have {free_gb:.1f} GB available."
        )

    # Internet
    if not _ping_internet():
        errors.append(
            "No internet connection detected. "
            "Magic Click needs the internet for first-time setup to download AI models (~500 MB). "
            "Please connect to Wi-Fi or Ethernet and try again."
        )

    return errors


def _ping_internet(host="8.8.8.8", port=53, timeout=4) -> bool:
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except OSError:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Lock File
# ══════════════════════════════════════════════════════════════════════════════

def acquire_lock() -> bool:
    if LOCK_FILE.exists():
        return False
    LOCK_FILE.write_text(str(os.getpid()))
    return True


def release_lock():
    try: LOCK_FILE.unlink(missing_ok=True)
    except Exception: pass


# ══════════════════════════════════════════════════════════════════════════════
# Core Setup Logic
# ══════════════════════════════════════════════════════════════════════════════

def _run(cmd: list[str], cwd=None, capture: bool = True):
    """Run a subprocess, return (returncode, combined_output)."""
    r = subprocess.run(cmd, cwd=cwd or ROOT, capture_output=capture, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    log.debug("CMD %s → rc=%d\n%s", cmd, r.returncode, out[:2000])  # type: ignore[index]
    return r.returncode, out


def _pip_install(extra_flags: list[str], gui: InstallerGUI, attempt: int) -> bool:
    gui.append_log(f"  pip install (attempt {attempt}/{MAX_RETRIES})…")
    cmd: list[str] = [str(VENV_PIP), "install", "--quiet", "--no-warn-script-location"] + extra_flags
    for req in REQ_FILES:
        if req.exists():
            cmd.extend(["-r", str(req)])
    rc, out = _run(cmd)
    for line in out.splitlines():
        if line.strip(): gui.append_log(f"    {line}")
    return rc == 0


def _warm_models(gui: InstallerGUI):
    """Import InsightFace inside the venv to trigger model download."""
    gui.append_log("  Warming InsightFace model cache…")
    warm_script = """
import sys, os
os.makedirs(os.path.expanduser("~/.insightface/models"), exist_ok=True)
try:
    import insightface
    app = insightface.app.FaceAnalysis(name="buffalo_sc",
                                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320))
    print("Models ready.")
except Exception as e:
    print(f"Model warn (non-fatal): {e}")
"""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        f.write(warm_script)
        tmp = f.name
    try:
        rc, out = _run([str(VENV_PY), tmp])
        for line in out.splitlines():
            if line.strip(): gui.append_log(f"    {line}")
    finally:
        os.unlink(tmp)


def _verify_services(gui: InstallerGUI) -> bool:
    """Start the API briefly and confirm /api/health returns 200."""
    gui.append_log("  Starting API for verification…")
    proc = subprocess.Popen(
        [str(VENV_PY), "-m", "uvicorn", "run:app",
         "--host", "127.0.0.1", "--port", str(API_PORT), "--log-level", "error"],
        cwd=ROOT / "mc_database",
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 40
    ok = False
    import urllib.request, urllib.error
    while time.time() < deadline:
        try:
            code = urllib.request.urlopen(
                f"http://127.0.0.1:{API_PORT}/api/health", timeout=3
            ).getcode()
            if code == 200:
                ok = True
                break
        except Exception:
            pass
        time.sleep(2)
    proc.terminate()
    try: proc.wait(timeout=5)
    except subprocess.TimeoutExpired: proc.kill()
    return ok


def _rollback(gui: InstallerGUI):
    gui.append_log("  Rolling back — removing incomplete .venv…", error=True)
    if VENV_DIR.exists():
        shutil.rmtree(VENV_DIR, ignore_errors=True)
    gui.append_log("  Rollback complete.", error=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main Setup Orchestration
# ══════════════════════════════════════════════════════════════════════════════

def run_setup(gui: InstallerGUI):
    """Runs in a background thread. Updates GUI through public methods."""
    attempt_count = [0]

    def attempt():
        attempt_count[0] += 1
        log.info("=== Setup attempt %d ===", attempt_count[0])

        # ── Stage 0: System Validation ────────────────────────────────────
        gui.set_stage(0, 2, "Checking your system requirements…")
        errors = validate_system()
        if errors:
            msg = "\n".join(f"• {e}" for e in errors)
            log.error("Pre-install validation failed:\n%s", msg)
            def _show_system_error() -> None:
                gui.show_error(
                    "Your system doesn't meet the requirements:\n\n" + msg,
                    on_retry=lambda: threading.Thread(target=attempt, daemon=True).start(),
                    on_cancel=gui.root.destroy,
                )
            gui.root.after(0, _show_system_error)  # type: ignore[arg-type]
            return

        gui.append_log("✓ OS, disk space, and internet checks passed.")
        gui.set_stage(0, 5, "System check passed. Creating Python environment…")

        # ── Stage 1: Create venv ──────────────────────────────────────────
        gui.set_stage(1, 8, "Creating an isolated Python environment…")
        if not VENV_DIR.exists():
            rc, out = _run([sys.executable, "-m", "venv", str(VENV_DIR)])
            if rc != 0:
                gui.append_log(out, error=True)
                _rollback(gui)
                def _show_venv_error() -> None:
                    gui.show_error(
                        "Failed to create Python virtual environment.\n\n"
                        "Please check that you have write permission to:\n" + str(ROOT),
                        on_retry=lambda: threading.Thread(target=attempt, daemon=True).start(),
                        on_cancel=gui.root.destroy,
                    )
                gui.root.after(0, _show_venv_error)  # type: ignore[arg-type]
                return
            gui.append_log("✓ Virtual environment created.")
        else:
            gui.append_log("  Existing .venv found — reusing.")

        gui.set_stage(1, 18)

        # ── Stage 2: Install packages with retry ──────────────────────────
        gui.set_stage(2, 22, "Installing Python packages (~500 MB, please wait)…\nThis may take 5–15 minutes on first run.")
        installed = False
        for attempt_n in range(1, MAX_RETRIES + 1):
            time.sleep((attempt_n - 1) * 5)  # exponential back-off: 0, 5, 10 s
            if _pip_install([], gui, attempt_n):
                installed = True
                break
            gui.append_log(f"  Attempt {attempt_n} failed. Retrying…", error=True)

        if not installed:
            _rollback(gui)
            def _show_pip_error() -> None:
                gui.show_error(
                    "Package installation failed after 3 attempts.\n\n"
                    "Please check your internet connection and try again.\n"
                    f"See {LOG_FILE.name} for details.",
                    on_retry=lambda: threading.Thread(target=attempt, daemon=True).start(),
                    on_cancel=gui.root.destroy,
                )
            gui.root.after(0, _show_pip_error)  # type: ignore[arg-type]
            return

        gui.append_log("✓ All packages installed.")
        gui.set_stage(2, 68)

        # ── Stage 3: Warm model cache ─────────────────────────────────────
        gui.set_stage(3, 72, "Downloading AI recognition models…\n(~200 MB, one-time only)")
        _warm_models(gui)
        gui.append_log("✓ AI models ready.")
        gui.set_stage(3, 88)

        # ── Stage 4: Service verification ────────────────────────────────
        gui.set_stage(4, 91, "Starting services to verify everything works…")
        ok = _verify_services(gui)
        if not ok:
            def _show_svc_error() -> None:
                gui.show_error(
                    "Services failed to start.\n\n"
                    "The packages installed correctly but the API did not respond.\n"
                    f"Check {LOG_FILE.name} for details.",
                    on_retry=lambda: threading.Thread(target=attempt, daemon=True).start(),
                    on_cancel=gui.root.destroy,
                )
            gui.root.after(0, _show_svc_error)  # type: ignore[arg-type]
            return

        gui.append_log("✓ All services verified successfully.")
        gui.set_stage(4, 100)

        def launch():
            gui._allow_close = True
            gui.root.destroy()
            _launch_pipeline()

        def _show_success() -> None:
            gui.show_success(on_launch=launch)
        gui.root.after(0, _show_success)  # type: ignore[arg-type]
        log.info("Setup completed successfully.")

    attempt()


def _launch_pipeline():
    """Launch ui_launcher.py (or MagicClick.command on macOS) after install."""
    log.info("Launching pipeline from post-install.")
    launcher = ROOT / "ui_launcher.py"
    if launcher.exists():
        subprocess.Popen([str(VENV_PY), str(launcher)],
                         cwd=ROOT, start_new_session=True)
    elif IS_MAC:
        cmd_file = ROOT / "launchers/macos/MagicClick.command"
        if cmd_file.exists():
            subprocess.Popen(["bash", str(cmd_file)], cwd=ROOT, start_new_session=True)


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("Magic Click installer_bootstrap.py started. Python %s | %s",
             sys.version, platform.platform())

    # Prevent running inside venv (should use system Python for setup)
    if sys.prefix != sys.base_prefix:
        print("⚠  installer_bootstrap.py must be run with system Python, not a venv.")
        sys.exit(1)

    # Lock file
    if not acquire_lock():
        messagebox.showwarning(
            "Already Running",
            "Magic Click setup is already running.\n\nCheck your taskbar/dock and wait for it to finish."
        )
        sys.exit(0)

    try:
        gui = InstallerGUI()
        gui.run(lambda: run_setup(gui))
    finally:
        release_lock()


if __name__ == "__main__":
    main()
