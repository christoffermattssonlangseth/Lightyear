"""Where Lightyear is running: from source / pip, or as the bundled desktop app (PyInstaller).

The bundled app has no `python` to call, so subprocesses (MCP server, run_python scripts) re-enter the
app binary through hidden CLI subcommands instead.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

FROZEN = bool(getattr(sys, "frozen", False))


def server_cmd() -> list[str]:
    """Command that starts the MCP server for the store in $LIGHTYEAR_STORE."""
    return [sys.executable, "serve"] if FROZEN else [sys.executable, "-m", "lightyear.server"]


def python_cmd() -> list[str]:
    """Command prefix that runs a Python script file."""
    return [sys.executable, "run-script"] if FROZEN else [sys.executable]


def app_dir() -> Path:
    """Per-user settings directory ($LIGHTYEAR_HOME overrides)."""
    if os.environ.get("LIGHTYEAR_HOME"):
        p = Path(os.environ["LIGHTYEAR_HOME"])
    elif sys.platform == "darwin":
        p = Path.home() / "Library" / "Application Support" / "Lightyear"
    else:
        p = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "lightyear"
    p.mkdir(parents=True, exist_ok=True)
    return p


def skill_text() -> str:
    """The query-dataset skill (how to work), without its front matter."""
    import re
    text = (Path(__file__).parent / "skill.md").read_text()
    return re.sub(r"^---.*?---\s*", "", text, flags=re.S)
