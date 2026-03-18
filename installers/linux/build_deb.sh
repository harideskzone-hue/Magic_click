#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Magic Click — Linux .deb Package Builder
# ══════════════════════════════════════════════════════════════════════════════
# Usage (from project root):
#   bash installers/linux/build_deb.sh
#
# Output: installers/linux/dist/magic-click_2.0.0_amd64.deb
#
# Requirements: dpkg-deb (pre-installed on all Debian/Ubuntu systems)
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
DIST_DIR="$SCRIPT_DIR/dist"
PKG_NAME="magic-click"
VERSION="2.0.0"
ARCH="amd64"
INSTALL_PREFIX="/opt/magic-click"
DEB_FILE="$DIST_DIR/${PKG_NAME}_${VERSION}_${ARCH}.deb"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   Magic Click — Linux .deb Builder       ║"
echo "╚══════════════════════════════════════════╝"
echo ""

command -v dpkg-deb >/dev/null 2>&1 || { echo "✗ dpkg-deb not found. Install with: sudo apt install dpkg-dev"; exit 1; }

# ── 1. Clean + create deb structure ──────────────────────────────────────────
echo "▶  Preparing build directory…"
rm -rf "$BUILD_DIR" "$DIST_DIR"
mkdir -p \
    "$BUILD_DIR/DEBIAN" \
    "$BUILD_DIR$INSTALL_PREFIX" \
    "$BUILD_DIR/usr/share/applications" \
    "$BUILD_DIR/usr/share/icons/hicolor/128x128/apps" \
    "$DIST_DIR"

# ── 2. Copy project files ─────────────────────────────────────────────────────
echo "▶  Copying Magic Click files…"
rsync -a \
    --exclude ".venv/" \
    --exclude "__pycache__/" \
    --exclude "*.pyc" \
    --exclude ".git/" \
    --exclude "captured_videos/" \
    --exclude "captured_shots/" \
    --exclude "data/db/" \
    "$PROJECT_ROOT/" "$BUILD_DIR$INSTALL_PREFIX/"

# ── 3. Compute installed size (in KB) ─────────────────────────────────────────
INSTALLED_SIZE=$(du -sk "$BUILD_DIR$INSTALL_PREFIX" | cut -f1)

# ── 4. DEBIAN/control ─────────────────────────────────────────────────────────
echo "▶  Writing DEBIAN/control…"
cat > "$BUILD_DIR/DEBIAN/control" << EOF
Package: $PKG_NAME
Version: $VERSION
Architecture: $ARCH
Maintainer: Magic Click Team <support@magicclick.local>
Installed-Size: $INSTALLED_SIZE
Depends: python3 (>= 3.10), python3-venv, python3-pip, libgl1
Recommends: python3-tk
Section: science
Priority: optional
Homepage: http://localhost:5001/
Description: Magic Click AI Vision System
 One-click AI-powered person detection and photo capture system.
 On first run, automatically installs all Python dependencies and
 downloads AI models. Subsequent launches are instant.
EOF

# ── 5. DEBIAN/postinst ────────────────────────────────────────────────────────
echo "▶  Writing DEBIAN/postinst…"
cat > "$BUILD_DIR/DEBIAN/postinst" << 'POSTINST'
#!/bin/bash
set -e
INSTALL_PATH="/opt/magic-click"
LOG="/tmp/magic_click_postinst.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== Magic Click Post-Install (Linux) ==="

# Make scripts executable
chmod +x "$INSTALL_PATH/launchers/linux/install.sh" 2>/dev/null || true
chmod +x "$INSTALL_PATH/installers/macos/scripts/"* 2>/dev/null || true
find "$INSTALL_PATH/installers" -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true

# Install .desktop entry
bash "$INSTALL_PATH/launchers/linux/install.sh" >> "$LOG" 2>&1 || log "⚠ Desktop install warned (non-fatal)"

# Find python3
PYTHON=$(command -v python3 2>/dev/null)
[ -n "$PYTHON" ] || { log "python3 not found — skipping bootstrap"; exit 0; }

# Launch installer_bootstrap.py in background for the current user
# Use DISPLAY so the Tk GUI can appear
if [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; then
    log "▶  Launching installer_bootstrap.py…"
    DISPLAY="${DISPLAY:-:0}" "$PYTHON" "$INSTALL_PATH/installers/installer_bootstrap.py" \
        >> "$LOG" 2>&1 &
else
    log "⚠ No display detected — run manually: python3 $INSTALL_PATH/installers/installer_bootstrap.py"
fi

log "=== Post-install complete ==="
exit 0
POSTINST
chmod 755 "$BUILD_DIR/DEBIAN/postinst"

# ── 6. DEBIAN/prerm (cleanup on uninstall) ────────────────────────────────────
cat > "$BUILD_DIR/DEBIAN/prerm" << 'PRERM'
#!/bin/bash
INSTALL_PATH="/opt/magic-click"
rm -rf "$INSTALL_PATH/.venv" 2>/dev/null || true
rm -f "$HOME/.local/share/applications/magic-click.desktop" 2>/dev/null || true
exit 0
PRERM
chmod 755 "$BUILD_DIR/DEBIAN/prerm"

# ── 7. .desktop file ─────────────────────────────────────────────────────────
cat > "$BUILD_DIR/usr/share/applications/magic-click.desktop" << EOF
[Desktop Entry]
Version=1.0
Name=Magic Click
GenericName=AI Vision Launcher
Comment=One-click AI person detection and photo capture system
Exec=bash -c 'cd $INSTALL_PREFIX && python3 installers/installer_bootstrap.py'
Terminal=false
Type=Application
Categories=Science;Video;
StartupNotify=true
EOF

# ── 8. Build the .deb ────────────────────────────────────────────────────────
echo "▶  Building .deb package…"
dpkg-deb --build --root-owner-group "$BUILD_DIR" "$DEB_FILE"

# ── 9. Report ────────────────────────────────────────────────────────────────
echo ""
echo "┌──────────────────────────────────────────────────────────┐"
echo "│  ✅  Build Successful                                      │"
echo "├──────────────────────────────────────────────────────────┤"
printf "│  Output:  %-46s │\n" "$DEB_FILE"
SIZE=$(du -sh "$DEB_FILE" | cut -f1)
printf "│  Size:    %-46s │\n" "$SIZE"
echo "├──────────────────────────────────────────────────────────┤"
echo "│  Install: sudo dpkg -i $PKG_NAME\_$VERSION\_$ARCH.deb     │"
echo "│  Remove:  sudo dpkg -r $PKG_NAME                          │"
echo "└──────────────────────────────────────────────────────────┘"
echo ""
