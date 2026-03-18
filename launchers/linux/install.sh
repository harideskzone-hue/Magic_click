#!/bin/bash
# install.sh — Linux setup script
# Generates the magic-click.desktop and installs it to the user applications menu

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
DESKTOP_FILE="$HOME/.local/share/applications/magic-click.desktop"

# Find Python
check_py_version() {
    local py="$1"
    local ver=$("$py" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
    local major=$(echo "$ver" | cut -d. -f1)
    local minor=$(echo "$ver" | cut -d. -f2)
    if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; then return 0; fi
    return 1
}

find_python() {
    for cmd in python3 python; do
        local py=$(command -v $cmd 2>/dev/null)
        if [ -n "$py" ] && check_py_version "$py"; then echo "$py"; return 0; fi
    done
    
    # Common Linux alternative paths or deadsnakes PPA paths
    for v in 3.14 3.13 3.12 3.11 3.10; do
        if [ -x "/usr/bin/python$v" ] && check_py_version "/usr/bin/python$v"; then echo "/usr/bin/python$v"; return 0; fi
        if [ -x "/usr/local/bin/python$v" ] && check_py_version "/usr/local/bin/python$v"; then echo "/usr/local/bin/python$v"; return 0; fi
    done
    return 1
}

PYTHON=$(find_python)
if [ -z "$PYTHON" ]; then
    echo "Error: Python 3.10 or newer was not found."
    echo "Please install it using your package manager (e.g., 'sudo apt install python3' or 'sudo dnf install python3') and try again."
    exit 1
fi

PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✓ Python $PY_VER OK ($PYTHON)"

mkdir -p "$HOME/.local/share/applications"

cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Version=1.0
Name=Magic Click
GenericName=AI Vision Launcher
Comment=One-click AI-powered person detection and photo capture system
Exec=bash -c 'cd "$PROJECT_ROOT" && $PYTHON bootstrap.py'
Terminal=false
Type=Application
Categories=Science;Video;AudioVideo;
StartupNotify=true
EOF

chmod +x "$DESKTOP_FILE"

echo "✓ Installed magic-click.desktop"
echo "  You can now launch Magic Click from your application menu."
