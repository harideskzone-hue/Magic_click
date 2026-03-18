#!/bin/bash
# build_app.sh
# Packages the MagicClick.command into a double-clickable macOS .app bundle.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
APP_NAME="MagicClick.app"
APP_DIR="$DIR/$APP_NAME"

# Clean previous build
rm -rf "$APP_DIR"

# Create directory structure
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"

# Create the executable wrapper
cat > "$APP_DIR/Contents/MacOS/run" << 'RUNEOF'
#!/bin/bash
# Resolve the absolute project root  (app is at launchers/macos/MagicClick.app)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../../../.." && pwd )"
cd "$PROJECT_ROOT" || exit 1

# Determine python3 executable
PYTHON=$(command -v python3 2>/dev/null)
if [ -z "$PYTHON" ]; then
    osascript -e 'display alert "Python Not Found" message "Python 3 could not be found on this Mac.\n\nPlease install Python 3.10+ from python.org and try again." as critical'
    exit 1
fi

# Show native launch dialog
VENV_EXISTS=0
[ -f ".venv/bin/python3" ] && VENV_EXISTS=1

if [ "$VENV_EXISTS" -eq 1 ]; then
    # Quick-launch dialog for returning users
    MSG="Magic Click is ready to start.\n\nYour AI pipeline will open in your browser."
    BUTTON_LABEL="▶  Launch"
else
    # First-run dialog (bootstrap.py will also show its own detailed dialog)
    MSG="Starting Magic Click for the first time.\n\nA one-time setup (~500 MB) is needed.\nPlease ensure you are connected to the internet."
    BUTTON_LABEL="▶  Set Up & Launch"
fi

CHOICE=$(osascript -e "button returned of (display dialog \"$MSG\" buttons {\"Cancel\", \"$BUTTON_LABEL\"} default button \"$BUTTON_LABEL\" with title \"Magic Click 2.0\")" 2>/dev/null)

if [ "$CHOICE" != "$BUTTON_LABEL" ]; then
    exit 0
fi

# Launch bootstrap.py silently in background (bootstrap shows its own GUI)
nohup "$PYTHON" -u bootstrap.py > /dev/null 2>&1 &
RUNEOF

# Make it executable
chmod +x "$APP_DIR/Contents/MacOS/run"

# Create Info.plist
cat > "$APP_DIR/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>run</string>
    <key>CFBundleIdentifier</key>
    <string>com.magicclick.studio</string>
    <key>CFBundleName</key>
    <string>MagicClick</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>2.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
</dict>
</plist>
EOF

echo "✓ Created $APP_NAME"
echo "You can now double-click '$APP_DIR' in Finder to start the pipeline."
