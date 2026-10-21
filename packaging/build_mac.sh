#!/bin/bash
# Build dist/Lightyear.app and dist/Lightyear-<version>-macos-<arch>.dmg.
#
#   SIGN_ID="Developer ID Application: …" (or its SHA-1)   sign with hardened runtime
#   NOTARY_PROFILE=lightyear-notary                        also notarise + staple (needs SIGN_ID); create once with
#       xcrun notarytool store-credentials lightyear-notary --apple-id <id> --team-id <team> --password <app-specific>
#
# Needs a build env with conda-forge scientific packages (pip's scipy can fail to load on recent macOS):
#   mamba create -p build/env -c conda-forge python=3.10 numpy=1.26 scipy=1.11 scikit-learn pandas=2.1 \
#         pyarrow matplotlib h5py anndata=0.11 pyyaml threadpoolctl
#   build/env/bin/pip install -e ".[app]" pyinstaller
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-build/env/bin/python}
VERSION=$($PY -c "import lightyear; print(lightyear.__version__)")
SIGN_ID=${SIGN_ID:-}
NOTARY_PROFILE=${NOTARY_PROFILE:-}
[ -f packaging/Lightyear.icns ] || $PY packaging/make_icon.py packaging
(cd packaging && ../$PY -m PyInstaller lightyear.spec --noconfirm --distpath ../dist --workpath ../build/pyi)
APP=dist/Lightyear.app
rm -rf dist/Lightyear   # the unbundled copy; the .app has everything
"$APP/Contents/MacOS/Lightyear" --help > /dev/null   # smoke test: the bundle starts

notarise() {  # file to submit
  echo "notarising $1 …"
  xcrun notarytool submit "$1" --keychain-profile "$NOTARY_PROFILE" --wait --timeout 60m | tee build/notary.log
  grep -q "status: Accepted" build/notary.log || {
    id=$(awk '/^  id:/{print $2; exit}' build/notary.log)
    xcrun notarytool log "$id" --keychain-profile "$NOTARY_PROFILE"; exit 1; }
}

if [ -n "$SIGN_ID" ]; then
  # PyInstaller re-signs everything ad hoc; put back Anthropic's Developer ID signature on the Claude binary
  # (it carries the JIT entitlements Bun needs) and sign everything else ourselves, inside out.
  CLAUDE_SRC=$($PY -c "import claude_agent_sdk, pathlib; print(pathlib.Path(claude_agent_sdk.__file__).parent / '_bundled' / 'claude')")
  CLAUDE_DST=$(find "$APP" -path "*_bundled/claude" -type f -not -type l | head -1)
  cp "$CLAUDE_SRC" "$CLAUDE_DST"
  echo "signing Mach-O files …"
  find "$APP/Contents" -type f -not -path "$CLAUDE_DST" -print0 | while IFS= read -r -d '' f; do
    if file -b "$f" | grep -q "Mach-O"; then echo "$f"; fi
  done | grep -v "/Contents/MacOS/Lightyear$" > build/macho.txt
  tr '\n' '\0' < build/macho.txt | xargs -0 -n 50 \
    codesign --force --timestamp --options runtime --sign "$SIGN_ID"
  codesign --force --timestamp --options runtime --entitlements packaging/entitlements.plist \
    --sign "$SIGN_ID" "$APP/Contents/MacOS/Lightyear"
  codesign --force --timestamp --options runtime --entitlements packaging/entitlements.plist --sign "$SIGN_ID" "$APP"
  codesign --verify --deep --strict "$APP"
  echo "signed $(wc -l < build/macho.txt | tr -d ' ') libraries + app"
  if [ -n "$NOTARY_PROFILE" ]; then
    ditto -c -k --keepParent "$APP" build/Lightyear-notarise.zip
    notarise build/Lightyear-notarise.zip
    xcrun stapler staple "$APP"
    rm -f build/Lightyear-notarise.zip
  fi
fi

DMG=dist/Lightyear-$VERSION-macos-$(uname -m).dmg
rm -f "$DMG"
STAGE=build/dmg; rm -rf "$STAGE"; mkdir -p "$STAGE"; cp -R "$APP" "$STAGE/"; ln -s /Applications "$STAGE/Applications"
hdiutil create -volname "Lightyear" -srcfolder "$STAGE" -ov -format UDZO "$DMG" > /dev/null
rm -rf "$STAGE"
if [ -n "$SIGN_ID" ]; then
  codesign --force --timestamp --sign "$SIGN_ID" "$DMG"
  if [ -n "$NOTARY_PROFILE" ]; then notarise "$DMG"; xcrun stapler staple "$DMG"; fi
fi
du -sh "$APP" "$DMG"
[ -n "$NOTARY_PROFILE" ] && spctl -a -vv -t install "$DMG" || true
