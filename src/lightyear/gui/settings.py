"""App settings (settings.json in the app dir) and API keys (system keychain).

Keys never go into settings.json: they live in the OS keychain via `keyring` (macOS Keychain, Windows
Credential Manager, Secret Service). Without a usable keyring they fall back to a 0600 file.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..runtime import app_dir

SERVICE = "lightyear"
KEY_NAMES = {"claude": "anthropic_api_key", "codex": "openai_api_key"}


@dataclass
class Settings:
    provider: str | None = None                 # "claude" | "codex"; None = not set up yet
    auth: dict = field(default_factory=lambda: {"claude": "api_key", "codex": "login"})  # "login" | "api_key"
    model: dict = field(default_factory=dict)   # optional per-provider model override
    stores_dir: str | None = None

    @classmethod
    def load(cls) -> "Settings":
        p = app_dir() / "settings.json"
        if not p.exists():
            return cls()
        d = json.loads(p.read_text())
        s = cls()
        for k, v in d.items():
            if hasattr(s, k):
                setattr(s, k, {**getattr(s, k), **v} if isinstance(v, dict) else v)
        return s

    def save(self):
        (app_dir() / "settings.json").write_text(json.dumps(asdict(self), indent=1))


def _keyring():
    try:
        import keyring
        from keyring.backends import fail
        kr = keyring.get_keyring()
        return None if isinstance(kr, fail.Keyring) else keyring
    except Exception:
        return None


def _fallback() -> Path:
    return app_dir() / "keys.json"


def get_key(provider: str) -> str | None:
    name = KEY_NAMES[provider]
    kr = _keyring()
    if kr is not None:
        try:
            v = kr.get_password(SERVICE, name)
            if v:
                return v
        except Exception:
            pass
    p = _fallback()
    return json.loads(p.read_text()).get(name) if p.exists() else None


def set_key(provider: str, key: str | None) -> str:
    """Store (or with None, delete) a provider key; returns where it is kept."""
    name = KEY_NAMES[provider]
    kr = _keyring()
    if kr is not None:
        try:
            if key:
                kr.set_password(SERVICE, name, key)
            else:
                try:
                    kr.delete_password(SERVICE, name)
                except Exception:
                    pass
            _drop_fallback(name)
            return "keychain"
        except Exception:
            pass
    p = _fallback()
    d = json.loads(p.read_text()) if p.exists() else {}
    if key:
        d[name] = key
    else:
        d.pop(name, None)
    p.write_text(json.dumps(d))
    os.chmod(p, 0o600)
    return str(p)


def _drop_fallback(name: str):
    p = _fallback()
    if p.exists():
        d = json.loads(p.read_text())
        if d.pop(name, None) is not None:
            p.write_text(json.dumps(d))


def mask(key: str | None) -> str | None:
    return None if not key else f"{key[:7]}…{key[-4:]}"
