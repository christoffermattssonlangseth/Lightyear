"""Which model providers are usable on this machine, and how.

Claude: an Anthropic API key, or (running from source only) the local Claude Code sign-in. Anthropic does
not allow third-party apps built on the Agent SDK to offer claude.ai login, so the bundled app is API-key
only for Claude unless LIGHTYEAR_ALLOW_CLAUDE_LOGIN=1.
Codex: the official Codex CLI must be installed; it is used with its own sign-in (`codex login`, the user's
ChatGPT plan or key) or with an OpenAI API key that the app passes as CODEX_API_KEY.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from ..runtime import FROZEN
from .settings import Settings, get_key, mask

ALLOW_CLAUDE_LOGIN = (not FROZEN) or os.environ.get("LIGHTYEAR_ALLOW_CLAUDE_LOGIN") == "1"
# A Finder-launched app gets a bare PATH; look where package managers put CLIs (codex is a node script).
EXTRA_PATH = ["/opt/homebrew/bin", "/usr/local/bin", str(Path.home() / ".local/bin"),
              str(Path.home() / ".npm-global/bin"), str(Path.home() / ".bun/bin"), str(Path.home() / ".volta/bin")]


def search_path() -> str:
    parts = os.environ.get("PATH", "").split(os.pathsep)
    return os.pathsep.join(parts + [p for p in EXTRA_PATH if p not in parts])


def claude_binary() -> str | None:
    try:
        import claude_agent_sdk
        b = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
        if b.exists():
            return str(b)
    except ImportError:
        pass
    return shutil.which("claude", path=search_path())


def codex_binary() -> str | None:
    return shutil.which("codex", path=search_path())


def _run(cmd: list[str], env: dict | None = None, timeout: int = 20) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL,
                           env={**os.environ, "PATH": search_path(), **(env or {})})
        return r.returncode, (r.stdout + r.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, str(e)


def claude_login_status() -> dict:
    b = claude_binary()
    if not b:
        return {"available": False, "detail": "Claude Code not found"}
    env = {k: "" for k in ("ANTHROPIC_API_KEY",)}
    code, out = _run([b, "auth", "status"], env=env)
    try:
        d = json.loads(out[out.index("{"):])
    except ValueError:
        return {"available": False, "detail": out[-200:] or f"exit {code}"}
    ok = bool(d.get("loggedIn"))
    who = d.get("email") or d.get("orgName") or ""
    return {"available": ok, "detail": f"signed in{' as ' + who if who else ''} ({d.get('authMethod', '?')})"
            if ok else "not signed in - run `claude auth login` in a terminal"}


def codex_login_status() -> dict:
    b = codex_binary()
    if not b:
        return {"installed": False, "available": False,
                "detail": "Codex CLI not found - install it (npm i -g @openai/codex or brew install codex)"}
    code, out = _run([b, "login", "status"])
    line = out.splitlines()[-1] if out else ""
    ok = code == 0 and "logged in" in line.lower()
    return {"installed": True, "available": ok, "path": b,
            "detail": line if ok else "not signed in - run `codex login` in a terminal"}


def validate_key(provider: str, key: str) -> tuple[bool, str]:
    """One cheap authenticated request to the provider's own API (the key goes nowhere else)."""
    import httpx
    if provider == "claude":
        url, headers = "https://api.anthropic.com/v1/models", {"x-api-key": key, "anthropic-version": "2023-06-01"}
    else:
        url, headers = "https://api.openai.com/v1/models", {"Authorization": f"Bearer {key}"}
    try:
        r = httpx.get(url, headers=headers, timeout=15)
    except httpx.HTTPError as e:
        return False, f"could not reach the API: {e}"
    if r.status_code == 200:
        return True, "key accepted"
    try:
        msg = r.json().get("error", {}).get("message", "")
    except ValueError:
        msg = r.text[:200]
    return False, f"HTTP {r.status_code}: {msg}"


def status(s: Settings | None = None) -> dict:
    """Everything the setup screen shows. Keys are only ever returned masked."""
    s = s or Settings.load()
    claude_key, codex_key = get_key("claude"), get_key("codex")
    claude_login = claude_login_status() if ALLOW_CLAUDE_LOGIN else {"available": False, "detail": "not offered"}
    codex = codex_login_status()
    out = {
        "provider": s.provider, "auth": s.auth, "model": s.model, "stores_dir": s.stores_dir,
        "claude": {"login_allowed": ALLOW_CLAUDE_LOGIN, "login": claude_login,
                   "key": mask(claude_key), "sdk": _has_sdk()},
        "codex": {"installed": codex["installed"], "login": codex, "key": mask(codex_key)},
    }
    out["ready"] = ready(s, out)
    return out


def _has_sdk() -> bool:
    try:
        import claude_agent_sdk  # noqa: F401
        return True
    except ImportError:
        return False


def ready(s: Settings, st: dict) -> bool:
    if s.provider == "claude":
        if not st["claude"]["sdk"]:
            return False
        return bool(st["claude"]["key"]) if s.auth.get("claude") == "api_key" else st["claude"]["login"]["available"]
    if s.provider == "codex":
        if not st["codex"]["installed"]:
            return False
        return bool(st["codex"]["key"]) if s.auth.get("codex") == "api_key" else st["codex"]["login"]["available"]
    return False
