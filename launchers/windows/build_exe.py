# build_exe.py
# Uses PyInstaller to bundle bootstrap.py into a single MagicClick.exe
# Run this on a Windows machine: python launchers/windows/build_exe.py

import os
import subprocess
import sys

DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(DIR, "..", ".."))
BOOTSTRAP_PY = os.path.join(ROOT_DIR, "bootstrap.py")

print("Checking for PyInstaller...")
try:
    import PyInstaller  # type: ignore
except ImportError:
    print("PyInstaller not found. Installing...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)

print("\nBuilding MagicClick.exe...")
# --onefile: creates a single .exe
# --windowed: hides the background terminal window (since bootstrap has a GUI)
# --name: the output filename
# --distpath: where to put the .exe (back in the windows launcher folder)
# --workpath: temp build files
cmd = [
    sys.executable, "-m", "PyInstaller",
    "--onefile",
    "--windowed",
    "--name", "MagicClick",
    "--distpath", DIR,
    "--workpath", os.path.join(DIR, "build"),
    "--specpath", DIR,
    BOOTSTRAP_PY
]

result = subprocess.run(cmd)

if result.returncode == 0:
    print(f"\n✓ Success! Your executable is ready at:")
    print(f"  {os.path.join(DIR, 'MagicClick.exe')}")
    print("\nYou can distribute this .exe to Windows users.")
    print("They just double-click it, and it will auto-install everything and launch.")
else:
    print("\n✗ Build failed.")
    sys.exit(1)
