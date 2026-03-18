#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Magic Click — macOS .pkg Installer Builder
# ══════════════════════════════════════════════════════════════════════════════
# Usage (from project root):
#   bash installers/macos/build_pkg.sh
#
# Output: installers/macos/dist/MagicClick_Installer.pkg
#
# Requirements: macOS with Xcode Command Line Tools (pkgbuild + productbuild)
# Optional:     Apple Developer ID for code signing (removes Gatekeeper warning)
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
DIST_DIR="$SCRIPT_DIR/dist"
PKG_ROOT="$BUILD_DIR/pkg_root"            # files to be installed
SCRIPTS_DIR="$SCRIPT_DIR/scripts"         # pre/postinstall scripts
RESOURCES_DIR="$SCRIPT_DIR/resources"     # welcome.html, background.png
COMPONENT_PKG="$BUILD_DIR/MagicClick_component.pkg"
FINAL_PKG="$DIST_DIR/MagicClick_Installer.pkg"

INSTALL_PATH="/Applications/MagicClick"   # where files land on the user's Mac
IDENTIFIER="com.magicclick.installer"
VERSION="2.0.0"

# ── Optional: Developer ID for signing (set env var to enable) ────────────────
SIGN_IDENTITY="${SIGN_IDENTITY:-}"        # e.g. "Developer ID Installer: Your Name (TEAMID)"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   Magic Click — macOS .pkg Builder       ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── 1. Check prerequisites ─────────────────────────────────────────────────────
echo "▶  Checking prerequisites…"
command -v pkgbuild      >/dev/null 2>&1 || { echo "✗ pkgbuild not found. Install Xcode Command Line Tools: xcode-select --install"; exit 1; }
command -v productbuild  >/dev/null 2>&1 || { echo "✗ productbuild not found. Install Xcode Command Line Tools."; exit 1; }
echo "✓ pkgbuild and productbuild found."

# ── 2. Clean + create build dirs ─────────────────────────────────────────────
echo "▶  Preparing build directories…"
rm -rf "$BUILD_DIR" "$DIST_DIR"
mkdir -p "$PKG_ROOT$INSTALL_PATH" "$BUILD_DIR" "$DIST_DIR"

# ── 3. Copy project files into pkg_root ──────────────────────────────────────
echo "▶  Copying Magic Click files…"
rsync -a \
    --exclude ".venv/" \
    --exclude "__pycache__/" \
    --exclude "*.pyc" \
    --exclude ".git/" \
    --exclude "captured_videos/" \
    --exclude "captured_shots/" \
    --exclude "data/db/" \
    "$PROJECT_ROOT/" "$PKG_ROOT$INSTALL_PATH/"
echo "✓ Files copied."

# ── 4. Build component package (pkgbuild) ─────────────────────────────────────
echo "▶  Building component package…"
pkgbuild_args=(
    --root     "$PKG_ROOT"
    --scripts  "$SCRIPTS_DIR"
    --identifier "$IDENTIFIER"
    --version  "$VERSION"
    --install-location "/"
    "$COMPONENT_PKG"
)
if [ -n "$SIGN_IDENTITY" ]; then
    pkgbuild_args+=(--sign "$SIGN_IDENTITY")
    echo "   Signing with: $SIGN_IDENTITY"
fi
pkgbuild "${pkgbuild_args[@]}"
echo "✓ Component package built."

# ── 5. Generate distribution XML ──────────────────────────────────────────────
echo "▶  Generating distribution XML…"
DIST_XML="$BUILD_DIR/distribution.xml"
cat > "$DIST_XML" << DISTXML
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="1">
    <title>Magic Click 2.0</title>
    <organization>com.magicclick</organization>
    <allowed-os-versions>
        <os-version min="12.0"/>
    </allowed-os-versions>
    <volume-check>
        <allowed-volumes>
            <volume attributes="writable"/>
        </allowed-volumes>
    </volume-check>
    <welcome    file="welcome.html"    mime-type="text/html"/>
    <conclusion file="conclusion.html" mime-type="text/html"/>
    <background file="background.png"  mime-type="image/png"
                alignment="bottomleft" scaling="tofit"/>
    <choices-outline>
        <line choice="default"/>
    </choices-outline>
    <choice id="default" title="Magic Click" description="Install Magic Click AI Vision System">
        <pkg-ref id="$IDENTIFIER"/>
    </choice>
    <pkg-ref id="$IDENTIFIER" version="$VERSION" onConclusion="none">MagicClick_component.pkg</pkg-ref>
</installer-gui-script>
DISTXML
echo "✓ Distribution XML generated."

# ── 6. Build final .pkg (productbuild) ─────────────────────────────────────────
echo "▶  Building final installer package…"
productbuild_args=(
    --distribution "$DIST_XML"
    --resources    "$RESOURCES_DIR"
    --package-path "$BUILD_DIR"
    "$FINAL_PKG"
)
if [ -n "$SIGN_IDENTITY" ]; then
    productbuild_args+=(--sign "$SIGN_IDENTITY")
fi
productbuild "${productbuild_args[@]}"
echo "✓ Installer package built."

# ── 7. Report ─────────────────────────────────────────────────────────────────
echo ""
echo "┌──────────────────────────────────────────────────────────┐"
echo "│  ✅  Build Successful                                      │"
echo "├──────────────────────────────────────────────────────────┤"
printf "│  Output:  %-46s │\n" "$FINAL_PKG"
SIZE=$(du -sh "$FINAL_PKG" | cut -f1)
printf "│  Size:    %-46s │\n" "$SIZE"
if [ -z "$SIGN_IDENTITY" ]; then
echo "│                                                            │"
echo "│  ⚠  Not signed. Users will see a Gatekeeper warning.     │"
echo "│     Right-click → Open to bypass, or set SIGN_IDENTITY   │"
echo "│     env var with your Apple Developer ID to sign.         │"
fi
echo "└──────────────────────────────────────────────────────────┘"
echo ""
