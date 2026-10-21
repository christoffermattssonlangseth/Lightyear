#!/bin/bash
# Build dist/Lightyear.app and dist/Lightyear-<version>-macos-arm64.dmg.
#
# Needs a build env with conda-forge scientific wheels (pip's scipy can fail to load on recent macOS):
#   mamba create -p build/env -c conda-forge python=3.10 numpy=1.26 scipy=1.11 scikit-learn pandas=2.1 \
#         pyarrow matplotlib h5py anndata=0.11 pyyaml threadpoolctl
#   build/env/bin/pip install -e ".[app]" pyinstaller
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-build/env/bin/python}
VERSION=$($PY -c "import lightyear; print(lightyear.__version__)")
[ -f packaging/Lightyear.icns ] || $PY packaging/make_icon.py packaging
(cd packaging && ../$PY -m PyInstaller lightyear.spec --noconfirm --distpath ../dist --workpath ../build/pyi)
APP=dist/Lightyear.app
rm -rf dist/Lightyear   # the unbundled copy; the .app has everything
"$APP/Contents/MacOS/Lightyear" --help > /dev/null   # smoke test: the bundle starts
DMG=dist/Lightyear-$VERSION-macos-$(uname -m).dmg
rm -f "$DMG"
STAGE=build/dmg; rm -rf "$STAGE"; mkdir -p "$STAGE"; cp -R "$APP" "$STAGE/"; ln -s /Applications "$STAGE/Applications"
hdiutil create -volname "Lightyear" -srcfolder "$STAGE" -ov -format UDZO "$DMG" > /dev/null
rm -rf "$STAGE"
du -sh "$APP" "$DMG"
