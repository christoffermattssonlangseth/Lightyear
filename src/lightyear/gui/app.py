"""Lightyear local chat app.

    lightyear gui [--stores stores] [--port 8765]

A FastAPI server on localhost. Each browser chat is one Claude Agent SDK session whose only tools are the
Lightyear MCP tools for the chosen dataset (no shell, no file access), so the agent sees exactly what it
sees in Claude Code: aggregates and figures computed locally. Uses the local Claude Code login.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from .. import tools
from ..claims import fingerprint, load_ledger
from ..store import Store

STATIC = Path(__file__).parent / "static"
SKILL = Path(__file__).resolve().parents[3] / ".claude" / "skills" / "query-dataset" / "SKILL.md"
LEDGER_TOOLS = {"test_claim", "recheck_claims"}

app = FastAPI(title="Lightyear")
STORES: dict[str, Path] = {}


def discover(root: Path) -> dict[str, Path]:
    return {p.parent.name: p.parent.resolve() for p in sorted(Path(root).glob("*/meta.json"))}


def _store(ds: str) -> Store:
    if ds not in STORES:
        raise HTTPException(404, f"unknown dataset {ds!r}")
    return Store(STORES[ds])


# -- read-only API -------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/static/{name}")
def static(name: str):
    p = (STATIC / name).resolve()
    if p.parent != STATIC.resolve() or not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/api/datasets")
def datasets():
    out = []
    for ds, path in STORES.items():
        meta = json.loads((path / "meta.json").read_text())
        desc = (meta.get("dataset_description") or "").strip()
        sch = meta["schema"]
        out.append({"id": ds, "name": meta.get("name", ds), "description": desc,
                    "n_cells": sch["n_cells"], "n_genes": sch["n_genes"], "roles": sch["roles"]})
    return out


@app.get("/api/{ds}/claims")
def claims(ds: str):
    s = _store(ds)
    rows = []
    for c in load_ledger(s):
        h = c["history"][-1]
        try:
            stale = fingerprint(s, c["spec"]) != h["fingerprint"]
        except Exception:
            stale = True
        rows.append({"id": c["id"], "statement": c["statement"], "verdict": h["verdict"],
                     "effect": h["summary"]["effect"], "n_a": h["summary"]["n_a"], "n_b": h["summary"]["n_b"],
                     "checked": h["run_at"], "stale": stale, "n_runs": len(c["history"])})
    return rows


@app.get("/api/{ds}/claims/{cid}")
def claim(ds: str, cid: str):
    try:
        text, fig = tools.show_claim(_store(ds), cid)
    except KeyError as e:
        raise HTTPException(404, str(e))
    return {"markdown": text, "figure": base64.b64encode(Path(fig).read_bytes()).decode()}


@app.get("/api/{ds}/findings")
def findings(ds: str):
    return {"markdown": tools.read_findings(_store(ds))}


@app.get("/api/{ds}/card")
def card(ds: str):
    return {"markdown": tools.card(_store(ds))}


# -- agent session ---------------------------------------------------------------------
def system_prompt(ds: str) -> str:
    meta = json.loads((STORES[ds] / "meta.json").read_text())
    skill = SKILL.read_text() if SKILL.exists() else ""
    skill = re.sub(r"^---.*?---\s*", "", skill, flags=re.S)  # drop front matter
    return "\n\n".join([
        f"You are Lightyear, an agent that knows the spatial omics dataset `{meta.get('name', ds)}` and answers "
        "questions about it in a local web app. Your only tools are the Lightyear tools (mcp__lightyear__*); "
        "they compute on the full data locally and return aggregates and figures. Figures returned by tools are "
        "shown to the user automatically - refer to them, don't describe how to open them. Answer in Markdown; "
        "keep answers focused; italicise mouse gene symbols.",
        f"Dataset description:\n{meta.get('dataset_description', '')}",
        "How to work:\n" + skill])


def agent_options(ds: str):
    from claude_agent_sdk import ClaudeAgentOptions
    env = {"LIGHTYEAR_STORE": str(STORES[ds]), "OPENBLAS_NUM_THREADS": "4"}
    return ClaudeAgentOptions(
        mcp_servers={"lightyear": {"type": "stdio", "command": sys.executable, "args": ["-m", "lightyear.server"],
                                   "env": env}},
        tools=[], allowed_tools=["mcp__lightyear"], setting_sources=[], strict_mcp_config=True,
        system_prompt=system_prompt(ds), cwd=str(STORES[ds]), include_partial_messages=True, max_turns=60)


def _tool_result_payload(block) -> dict:
    content = block.content if isinstance(block.content, list) else [block.content]
    texts, images = [], []
    for c in content:
        if isinstance(c, dict) and c.get("type") == "image":
            src = c.get("source") or {}
            images.append(src.get("data") or c.get("data"))
        elif isinstance(c, dict) and c.get("type") == "text":
            texts.append(c.get("text", ""))
        elif isinstance(c, str):
            texts.append(c)
    text = "\n".join(texts)
    try:  # str-returning tools arrive as structured {"result": "..."}
        obj = json.loads(text)
        if isinstance(obj, dict) and set(obj) == {"result"} and isinstance(obj["result"], str):
            text = obj["result"]
    except ValueError:
        pass
    return {"text": text, "images": [i for i in images if i], "is_error": bool(block.is_error)}


@app.websocket("/ws/{ds}")
async def chat(ws: WebSocket, ds: str):
    from claude_agent_sdk import (AssistantMessage, ClaudeSDKClient, ResultMessage, StreamEvent, TextBlock,
                                  ToolResultBlock, ToolUseBlock, UserMessage)
    await ws.accept()
    if ds not in STORES:
        await ws.send_json({"type": "error", "text": f"unknown dataset {ds!r}"})
        await ws.close()
        return
    client = ClaudeSDKClient(options=agent_options(ds))
    await client.connect()
    tool_names: dict[str, str] = {}
    busy = asyncio.Lock()

    async def run(text: str):
        async with busy:
            await client.query(text)
            async for m in client.receive_response():
                if isinstance(m, StreamEvent):
                    ev = m.event
                    if ev.get("type") == "content_block_delta" and ev.get("delta", {}).get("type") == "text_delta":
                        await ws.send_json({"type": "delta", "text": ev["delta"]["text"]})
                elif isinstance(m, AssistantMessage):
                    for b in m.content:
                        if isinstance(b, TextBlock):
                            await ws.send_json({"type": "text", "text": b.text})
                        elif isinstance(b, ToolUseBlock):
                            name = b.name.split("__")[-1]
                            tool_names[b.id] = name
                            await ws.send_json({"type": "tool_use", "id": b.id, "name": name, "input": b.input})
                elif isinstance(m, UserMessage) and isinstance(m.content, list):
                    for b in m.content:
                        if isinstance(b, ToolResultBlock):
                            name = tool_names.get(b.tool_use_id, "")
                            await ws.send_json({"type": "tool_result", "id": b.tool_use_id, "name": name,
                                                **_tool_result_payload(b)})
                            if name in LEDGER_TOOLS or name == "add_finding":
                                await ws.send_json({"type": "ledger_changed"})
                elif isinstance(m, ResultMessage):
                    await ws.send_json({"type": "done", "turns": m.num_turns, "duration_s": m.duration_ms / 1000,
                                        "cost_usd": m.total_cost_usd, "is_error": m.is_error})

    task = None
    try:
        await ws.send_json({"type": "ready"})
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "ask" and msg.get("text", "").strip():
                if task and not task.done():
                    await ws.send_json({"type": "error", "text": "still working on the previous question"})
                    continue
                task = asyncio.create_task(run(msg["text"]))
                task.add_done_callback(lambda t: t.exception() and asyncio.ensure_future(
                    ws.send_json({"type": "error", "text": f"{type(t.exception()).__name__}: {t.exception()}"})))
            elif msg.get("type") == "interrupt":
                await client.interrupt()
    except WebSocketDisconnect:
        pass
    finally:
        if task and not task.done():
            task.cancel()
        await client.disconnect()


def _free_port(host: str, port: int) -> int:
    import socket
    for p in range(port, port + 30):
        with socket.socket() as sk:
            if sk.connect_ex((host, p)) != 0:
                return p
    raise SystemExit(f"no free port in {port}-{port + 29}")


def main(stores: str = "stores", host: str = "127.0.0.1", port: int = 8790, open_browser: bool = True):
    import uvicorn
    port = _free_port(host, port)
    STORES.update(discover(Path(stores)))
    if not STORES:
        raise SystemExit(f"no stores found under {stores}/ (expected */meta.json)")
    print(f"Lightyear: {', '.join(STORES)} -> http://{host}:{port}")
    if open_browser:
        import threading
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
