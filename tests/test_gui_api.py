"""First-run and setup flows of the chat app API, as a new user meets them (no settings, no folders)."""
import shutil

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from lightyear.gui import app as gui
from lightyear.gui import auth, settings


@pytest.fixture
def fresh(tmp_path, monkeypatch, toy):
    """A brand-new user: empty home, no settings, default datasets folder missing, nobody signed in."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LIGHTYEAR_HOME", str(tmp_path / "appdir"))
    monkeypatch.setattr(settings, "_keyring", lambda: None)
    monkeypatch.setattr(gui, "DEFAULT_STORES", home / "Lightyear" / "stores")
    monkeypatch.setattr(gui, "STORES_OVERRIDE", None)
    monkeypatch.setattr(auth, "claude_login_status", lambda: {"available": False, "detail": "not signed in"})
    monkeypatch.setattr(auth, "codex_login_status",
                        lambda: {"installed": False, "available": False, "detail": "Codex CLI not found"})
    store = toy[1].path
    return TestClient(gui.app), home, store


def test_first_run_status_has_no_errors(fresh):
    c, home, _ = fresh
    r = c.get("/api/setup")
    assert r.status_code == 200
    st = r.json()
    assert st["provider"] is None and not st["ready"]
    assert st["stores_dir"] == str(home / "Lightyear" / "stores")
    assert c.get("/api/datasets").json() == []
    assert c.get("/").status_code == 200
    assert c.get("/favicon.ico").status_code == 200
    for asset in ("app.js", "style.css", "favicon.png", "vendor/marked.min.js", "vendor/purify.min.js"):
        assert c.get(f"/static/{asset}").status_code == 200, asset


def test_start_with_default_folder_creates_it(fresh):
    """The setup page always sends the prefilled default folder; that must never be an error."""
    c, home, _ = fresh
    st = c.get("/api/setup").json()
    r = c.post("/api/setup", json={"provider": "claude", "auth": {"claude": "api_key"},
                                   "model": {"claude": ""}, "stores_dir": st["stores_dir"]})
    assert r.status_code == 200, r.text
    assert (home / "Lightyear" / "stores").is_dir()
    assert not r.json()["ready"]          # no key yet: the page asks for one, not for a folder
    assert c.get("/api/datasets").json() == []


def test_missing_folder_is_explained(fresh):
    c, home, _ = fresh
    r = c.post("/api/setup", json={"provider": "claude", "stores_dir": str(home / "nope")})
    assert r.status_code == 400
    assert "Folder not found" in r.json()["detail"] and "Advanced" in r.json()["detail"]


def test_pointing_at_stores_lists_them(fresh, tmp_path):
    c, _, store = fresh
    root = tmp_path / "mystores"
    root.mkdir()
    shutil.copytree(store, root / "toy")
    r = c.post("/api/setup", json={"provider": "codex", "stores_dir": str(root)})
    assert r.status_code == 200 and r.json()["stores_dir"] == str(root)
    ds = c.get("/api/datasets").json()
    assert [d["id"] for d in ds] == ["toy"]
    assert c.get("/api/toy/card").status_code == 200
    assert c.get("/api/toy/claims").json() == []


def test_chat_before_setup_asks_for_setup(fresh, tmp_path):
    c, _, store = fresh
    root = tmp_path / "s"
    root.mkdir()
    shutil.copytree(store, root / "toy")
    c.post("/api/setup", json={"stores_dir": str(root)})
    with c.websocket_connect("/ws/toy") as ws:
        assert ws.receive_json()["type"] == "setup_needed"


def test_claude_login_refused_when_not_offered(fresh, monkeypatch):
    c, _, _ = fresh
    monkeypatch.setattr(auth, "ALLOW_CLAUDE_LOGIN", False)
    r = c.post("/api/setup", json={"provider": "claude", "auth": {"claude": "login"}})
    assert r.status_code == 400 and "API key" in r.json()["detail"]


def test_codex_start_ignores_unusable_claude_mode(fresh, monkeypatch):
    """Choosing Codex must work whatever the (hidden) Claude radio holds."""
    c, _, _ = fresh
    monkeypatch.setattr(auth, "ALLOW_CLAUDE_LOGIN", False)
    r = c.post("/api/setup", json={"provider": "codex", "auth": {"claude": "login", "codex": "login"}})
    assert r.status_code == 200, r.text
    assert r.json()["provider"] == "codex" and r.json()["auth"]["claude"] == "api_key"


def test_hidden_attribute_wins_over_css():
    assert "[hidden] { display: none !important; }" in (gui.STATIC / "style.css").read_text()
