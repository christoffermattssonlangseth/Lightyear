"""Agent back ends for the chat app. Each turns one question into a stream of UI events:

    {"type": "text", "text"}                       a finished block of assistant text
    {"type": "delta", "text"}                      streamed text (Claude only)
    {"type": "tool_use", "id", "name", "input"}
    {"type": "tool_result", "id", "name", "text", "images", "is_error"}
    {"type": "done", "turns", "duration_s", "cost_usd" | "tokens"}
    {"type": "error", "text"}

Both back ends get the same Lightyear MCP server as their only tools (no shell, no file access), so the model
sees only aggregates and figures computed on this machine.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Awaitable, Callable

from ..runtime import server_cmd, skill_text
from .auth import codex_binary, search_path
from .settings import Settings, get_key

Emit = Callable[[dict], Awaitable[None]]


def system_prompt(store: Path) -> str:
    meta = json.loads((store / "meta.json").read_text())
    return "\n\n".join([
        f"You are Lightyear, an agent that knows the spatial omics dataset `{meta.get('name', store.name)}` and "
        "answers questions about it in a local app. Your only tools are the Lightyear tools; they compute on the "
        "full data locally and return aggregates and figures. Figures returned by tools are shown to the user "
        "automatically - refer to them, don't describe how to open them. Answer in Markdown; keep answers "
        "focused; italicise mouse gene symbols.",
        f"Dataset description:\n{meta.get('dataset_description', '')}",
        "How to work:\n" + skill_text()])


def server_env(store: Path) -> dict:
    return {"LIGHTYEAR_STORE": str(store), "OPENBLAS_NUM_THREADS": "4"}


def unwrap(text: str) -> str:
    """str-returning MCP tools arrive as structured {"result": "..."}."""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and set(obj) == {"result"} and isinstance(obj["result"], str):
            return obj["result"]
    except ValueError:
        pass
    return text


def _content(items) -> tuple[str, list[str]]:
    texts, images = [], []
    for c in items if isinstance(items, list) else [items]:
        if isinstance(c, str):
            texts.append(c)
        elif isinstance(c, dict) and c.get("type") == "text":
            texts.append(c.get("text", ""))
        elif isinstance(c, dict) and c.get("type") == "image":
            images.append((c.get("source") or {}).get("data") or c.get("data"))
    return unwrap("\n".join(texts)), [i for i in images if i]


class Runner:
    name = "?"

    async def start(self): ...
    async def ask(self, text: str, emit: Emit): ...
    async def interrupt(self): ...
    async def close(self): ...


# -- Claude (Agent SDK) -------------------------------------------------------------------
class ClaudeRunner(Runner):
    name = "claude"

    def __init__(self, store: Path, settings: Settings):
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
        cmd = server_cmd()
        env = {}
        if settings.auth.get("claude") == "api_key":
            env["ANTHROPIC_API_KEY"] = get_key("claude") or ""
        opts = ClaudeAgentOptions(
            mcp_servers={"lightyear": {"type": "stdio", "command": cmd[0], "args": cmd[1:], "env": server_env(store)}},
            tools=[], allowed_tools=["mcp__lightyear"], setting_sources=[], strict_mcp_config=True,
            system_prompt=system_prompt(store), cwd=str(store), include_partial_messages=True, max_turns=60,
            env=env, model=settings.model.get("claude") or None)
        self.client = ClaudeSDKClient(options=opts)
        self.tool_names: dict[str, str] = {}

    async def start(self):
        await self.client.connect()

    async def ask(self, text: str, emit: Emit):
        from claude_agent_sdk import (AssistantMessage, ResultMessage, StreamEvent, TextBlock, ToolResultBlock,
                                      ToolUseBlock, UserMessage)
        await self.client.query(text)
        async for m in self.client.receive_response():
            if isinstance(m, StreamEvent):
                ev = m.event
                if ev.get("type") == "content_block_delta" and ev.get("delta", {}).get("type") == "text_delta":
                    await emit({"type": "delta", "text": ev["delta"]["text"]})
            elif isinstance(m, AssistantMessage):
                for b in m.content:
                    if isinstance(b, TextBlock):
                        await emit({"type": "text", "text": b.text})
                    elif isinstance(b, ToolUseBlock):
                        name = b.name.split("__")[-1]
                        self.tool_names[b.id] = name
                        await emit({"type": "tool_use", "id": b.id, "name": name, "input": b.input})
            elif isinstance(m, UserMessage) and isinstance(m.content, list):
                for b in m.content:
                    if isinstance(b, ToolResultBlock):
                        txt, imgs = _content(b.content)
                        await emit({"type": "tool_result", "id": b.tool_use_id,
                                    "name": self.tool_names.get(b.tool_use_id, ""),
                                    "text": txt, "images": imgs, "is_error": bool(b.is_error)})
            elif isinstance(m, ResultMessage):
                await emit({"type": "done", "turns": m.num_turns, "duration_s": m.duration_ms / 1000,
                            "cost_usd": m.total_cost_usd, "is_error": m.is_error})

    async def interrupt(self):
        await self.client.interrupt()

    async def close(self):
        await self.client.disconnect()


# -- Codex (CLI, `codex exec --json`) -------------------------------------------------------
# Everything except the MCP tools is switched off; the read-only sandbox is a second fence.
CODEX_DISABLE = ["shell_tool", "unified_exec", "browser_use", "browser_use_external", "computer_use",
                 "in_app_browser", "image_generation", "apps", "plugins", "multi_agent", "goals", "tool_suggest",
                 "skill_search", "view_image", "sleep_tool", "memories"]
# Codex 0.15x only lists tools from servers speaking the current MCP revision behind this flag.
CODEX_ENABLE = ["mcp_2026_07_28"]


def toml(v) -> str:
    return json.dumps(v, ensure_ascii=False)  # a JSON string / list is a valid TOML value


def codex_args(store: Path, settings: Settings, prompt: str, thread_id: str | None) -> list[str]:
    cmd = server_cmd()
    a = ["exec"] + (["resume", thread_id] if thread_id else [])
    a += ["--json", "--skip-git-repo-check", "--ignore-user-config", "--ignore-rules"]
    for f in CODEX_DISABLE:
        a += ["--disable", f]
    for f in CODEX_ENABLE:
        a += ["--enable", f]
    env = "{" + ", ".join(f"{k} = {toml(v)}" for k, v in server_env(store).items()) + "}"
    cfg = {"sandbox_mode": toml("read-only"), "approval_policy": toml("never"), "web_search": toml("disabled"),
           "suppress_unstable_features_warning": "true", "project_doc_max_bytes": "0",
           "developer_instructions": toml(system_prompt(store)),
           "mcp_servers.lightyear.command": toml(cmd[0]), "mcp_servers.lightyear.args": toml(cmd[1:]),
           "mcp_servers.lightyear.env": env, "mcp_servers.lightyear.startup_timeout_sec": "120",
           "mcp_servers.lightyear.tool_timeout_sec": "1800",
           "mcp_servers.lightyear.default_tools_approval_mode": toml("approve")}
    for k, v in cfg.items():
        a += ["-c", f"{k}={v}"]
    if settings.model.get("codex"):
        a += ["-m", settings.model["codex"]]
    return a + ["--", prompt]


def codex_events(e: dict, state: dict) -> list[dict]:
    """Translate one `codex exec --json` event into UI events."""
    t, it = e.get("type"), e.get("item") or {}
    out = []
    if t == "thread.started":
        state["thread_id"] = e.get("thread_id")
    elif t in ("item.started", "item.completed") and it.get("type") == "mcp_tool_call":
        if t == "item.started" or it["id"] not in state["seen"]:
            state["seen"].add(it["id"])
            state["tools"] += 1
            out.append({"type": "tool_use", "id": it["id"], "name": it.get("tool", ""),
                        "input": it.get("arguments") or {}})
        if t == "item.completed":
            res, err = it.get("result") or {}, it.get("error")
            txt, imgs = _content(res.get("content") or [])
            if err:
                txt = (txt + "\n" if txt else "") + str(err.get("message", err))
            out.append({"type": "tool_result", "id": it["id"], "name": it.get("tool", ""), "text": txt,
                        "images": imgs, "is_error": bool(err) or it.get("status") == "failed"})
    elif t == "item.completed" and it.get("type") == "agent_message":
        out.append({"type": "text", "text": it.get("text", "")})
    elif t == "item.completed" and it.get("type") == "error":
        msg = it.get("message", "")
        if "Under-development features" not in msg:
            out.append({"type": "text", "text": f"_Codex: {msg}_"})
    elif t == "turn.completed":
        u = e.get("usage") or {}
        out.append({"type": "done", "turns": state["tools"], "duration_s": time.time() - state["t0"],
                    "tokens": (u.get("input_tokens") or 0) + (u.get("output_tokens") or 0)})
        state["finished"] = True
    elif t == "error":  # transient ("Reconnecting... 2/5"); kept in case the run dies without turn.failed
        state["last_error"] = e.get("message", "")
    elif t == "turn.failed":
        err = e.get("error") or {}
        out.append({"type": "error", "text": err.get("message", str(err)) if isinstance(err, dict) else str(err)})
        state["finished"] = True
    return out


class CodexRunner(Runner):
    name = "codex"

    def __init__(self, store: Path, settings: Settings):
        self.store, self.settings = store, settings
        self.binary = codex_binary()
        if not self.binary:
            raise RuntimeError("Codex CLI not found")
        self.thread_id: str | None = None
        self.proc: asyncio.subprocess.Process | None = None

    def _env(self) -> dict:
        env = {**os.environ, "PATH": search_path()}
        env.pop("CODEX_API_KEY", None)
        if self.settings.auth.get("codex") == "api_key":
            env["CODEX_API_KEY"] = get_key("codex") or ""
        return env

    async def ask(self, text: str, emit: Emit):
        state = {"thread_id": self.thread_id, "seen": set(), "tools": 0, "t0": time.time(), "finished": False}
        self.proc = await asyncio.create_subprocess_exec(
            self.binary, *codex_args(self.store, self.settings, text, self.thread_id), cwd=str(self.store),
            env=self._env(), stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, limit=256 * 1024 * 1024)  # tool results carry base64 figures
        err_tail: list[str] = []

        async def drain():
            async for line in self.proc.stderr:
                err_tail.append(line.decode(errors="replace").rstrip())
                del err_tail[:-30]
        drainer = asyncio.create_task(drain())
        final = []  # held back until codex has exited, so the UI can't send the next question too early
        async for line in self.proc.stdout:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            for ev in codex_events(e, state):
                if ev["type"] in ("done", "error"):
                    final.append(ev)
                else:
                    await emit(ev)
        code = await self.proc.wait()
        await drainer
        self.thread_id = state["thread_id"] or self.thread_id
        for ev in final:
            await emit(ev)
        if not state["finished"]:
            msg = "stopped" if code < 0 else (state.get("last_error") or
                                               "\n".join(l for l in err_tail if l and "Reading additional" not in l))
            await emit({"type": "error", "text": f"Codex exited ({code}): {msg[-1500:] or 'no output'}"})
        self.proc = None

    async def interrupt(self):
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()

    async def close(self):
        await self.interrupt()


def make_runner(store: Path, settings: Settings) -> Runner:
    if settings.provider == "codex":
        return CodexRunner(store, settings)
    if settings.provider == "claude":
        return ClaudeRunner(store, settings)
    raise RuntimeError("no model provider set up yet")
