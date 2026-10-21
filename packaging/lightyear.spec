# PyInstaller spec for Lightyear.app (macOS). Build with packaging/build_mac.sh.
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

from lightyear import __version__ as VERSION

datas, binaries, hidden = [], [], []
for pkg in ("claude_agent_sdk", "webview", "uvicorn", "keyring"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hidden += h
hidden += collect_submodules("mcp", filter=lambda n: not n.startswith("mcp.cli"))  # mcp.cli exits without typer
datas += collect_data_files("mcp")
datas += collect_data_files("lightyear")
hidden += collect_submodules("lightyear")
# what run_python scripts commonly reach for, beyond what Lightyear itself imports
hidden += ["scipy.stats", "scipy.spatial", "scipy.sparse", "scipy.cluster.hierarchy", "sklearn.linear_model",
           "sklearn.decomposition", "sklearn.neighbors", "sklearn.cluster", "sklearn.metrics"]

a = Analysis(["lightyear_app.py"], pathex=[], binaries=binaries, datas=datas, hiddenimports=hidden,
             excludes=["tkinter", "IPython", "jupyter", "notebook", "pytest", "torch", "napari", "PyQt5", "PyQt6",
                       "PySide2", "PySide6"],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Lightyear", console=False, argv_emulation=False,
          codesign_identity=None, entitlements_file=None)
coll = COLLECT(exe, a.binaries, a.datas, name="Lightyear")
app = BUNDLE(coll, name="Lightyear.app", icon="Lightyear.icns", bundle_identifier="io.github.christoffermattssonlangseth.lightyear",
             version=VERSION,
             info_plist={"CFBundleShortVersionString": VERSION, "NSHighResolutionCapable": True,
                         "LSMinimumSystemVersion": "12.0", "NSRequiresAquaSystemAppearance": False})
