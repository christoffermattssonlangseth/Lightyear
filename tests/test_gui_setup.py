"""Setup screen plumbing: settings/keys storage, Codex command line and event translation."""
import json

import pytest

pytest.importorskip("fastapi")

from lightyear.gui import runners, settings
from lightyear.gui.settings import Settings


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LIGHTYEAR_HOME", str(tmp_path))
    monkeypatch.setattr(settings, "_keyring", lambda: None)  # never touch the real keychain in tests
    return tmp_path


def test_settings_roundtrip(home):
    s = Settings.load()
    assert s.provider is None and s.auth == {"claude": "api_key", "codex": "login"}
    s.provider, s.auth["codex"], s.stores_dir = "codex", "api_key", "/x"
    s.save()
    t = Settings.load()
    assert (t.provider, t.auth["codex"], t.auth["claude"], t.stores_dir) == ("codex", "api_key", "api_key", "/x")


def test_keys_fallback_file_is_private(home):
    assert settings.get_key("claude") is None
    where = settings.set_key("claude", "sk-ant-abcdefghijk1234")
    assert where.endswith("keys.json") and (home / "keys.json").stat().st_mode & 0o077 == 0
    assert settings.get_key("claude") == "sk-ant-abcdefghijk1234"
    assert settings.mask("sk-ant-abcdefghijk1234") == "sk-ant-…1234"
    assert "sk-ant" not in (home / "settings.json").read_text() if (home / "settings.json").exists() else True
    settings.set_key("claude", None)
    assert settings.get_key("claude") is None


def test_codex_args_lock_down_tools(toy):
    store = toy[1].path
    s = Settings(provider="codex")
    a = runners.codex_args(store, s, "hello", None)
    assert a[:2] == ["exec", "--json"] and a[-2:] == ["--", "hello"]
    disabled = {a[i + 1] for i, x in enumerate(a) if x == "--disable"}
    assert {"shell_tool", "unified_exec", "apps", "plugins", "view_image"} <= disabled
    cfg = dict(a[i + 1].split("=", 1) for i, x in enumerate(a) if x == "-c")
    assert cfg["sandbox_mode"] == '"read-only"' and cfg["approval_policy"] == '"never"'
    assert json.loads(cfg["developer_instructions"]).startswith("You are Lightyear")
    assert str(store) in cfg["mcp_servers.lightyear.env"]
    resumed = runners.codex_args(store, s, "again", "abc")
    assert resumed[:3] == ["exec", "resume", "abc"]


def test_codex_event_translation():
    st = {"thread_id": None, "seen": set(), "tools": 0, "t0": 0.0, "finished": False}
    ev = []
    for e in [
        {"type": "thread.started", "thread_id": "T1"},
        {"type": "item.completed", "item": {"id": "e", "type": "error", "message": "Under-development features enabled: x"}},
        {"type": "item.started", "item": {"id": "i1", "type": "mcp_tool_call", "server": "lightyear",
                                          "tool": "dataset_card", "arguments": {}, "status": "in_progress"}},
        {"type": "item.completed", "item": {"id": "i1", "type": "mcp_tool_call", "tool": "dataset_card",
                                            "result": {"content": [{"type": "text", "text": '{"result": "# card"}'},
                                                                   {"type": "image", "data": "AAA"}]},
                                            "error": None, "status": "completed"}},
        {"type": "item.completed", "item": {"id": "i2", "type": "mcp_tool_call", "tool": "bad", "arguments": {"x": 1},
                                            "result": None, "error": {"message": "boom"}, "status": "failed"}},
        {"type": "item.completed", "item": {"id": "m", "type": "agent_message", "text": "There are *n* cells."}},
        {"type": "error", "message": "Reconnecting... 1/5"},
        {"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 20}},
    ]:
        ev += runners.codex_events(e, st)
    assert st["thread_id"] == "T1" and st["finished"]
    kinds = [e["type"] for e in ev]
    assert kinds == ["tool_use", "tool_result", "tool_use", "tool_result", "text", "done"]
    assert ev[1]["text"] == "# card" and ev[1]["images"] == ["AAA"] and not ev[1]["is_error"]
    assert ev[3]["is_error"] and "boom" in ev[3]["text"]
    assert ev[-1]["turns"] == 2 and ev[-1]["tokens"] == 120


def test_ready_rules(home):
    from lightyear.gui import auth
    st = {"claude": {"sdk": True, "key": None, "login": {"available": True}},
          "codex": {"installed": True, "key": "sk-…abcd", "login": {"available": False}}}
    assert not auth.ready(Settings(), st)
    assert not auth.ready(Settings(provider="claude"), st)                       # api_key mode, no key
    assert auth.ready(Settings(provider="claude", auth={"claude": "login", "codex": "login"}), st)
    assert not auth.ready(Settings(provider="codex"), st)                        # login mode, not signed in
    assert auth.ready(Settings(provider="codex", auth={"claude": "api_key", "codex": "api_key"}), st)
