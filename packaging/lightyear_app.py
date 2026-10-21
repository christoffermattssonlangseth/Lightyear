"""Entry point of the bundled app. Double-clicked: open the chat app in a window. With arguments: the normal
`lightyear` CLI, which is also how the app re-enters itself for the MCP server (`serve`) and run_python
scripts (`run-script`)."""
import multiprocessing
import os
import sys

from lightyear.runtime import app_dir

# PyInstaller's matplotlib hook points MPLCONFIGDIR at a fresh temp dir, which rebuilds the font cache (~20 s)
# on every start of every subprocess; keep it in the app dir instead.
os.environ["MPLCONFIGDIR"] = str(app_dir() / "matplotlib")

from lightyear.cli import main  # noqa: E402

if __name__ == "__main__":
    multiprocessing.freeze_support()
    args = [a for a in sys.argv[1:] if not a.startswith("-psn_")]  # Finder adds a process serial number
    main(args or ["gui", "--window"])
