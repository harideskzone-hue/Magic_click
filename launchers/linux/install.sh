#!/bin/bash
# install.sh — Linux setup script
# Generates the magic-click.desktop and installs it to the user applications menu

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
DESKTOP_FILE="$HOME/.local/share/applications/magic-click.desktop"

# Find Python
PYTHON=$(command -v python3 2>/dev/null)
if [ -z "$PYTHON" ]; then
    echo "Error: python3 not found. Please install Python 3.10+ and retry."
    exit 1
fi

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
