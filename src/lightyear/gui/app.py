"""Lightyear local chat app.

    lightyear gui [--stores DIR] [--port 8790]

A FastAPI server on localhost. Each browser chat is one agent session (Claude via the Agent SDK, or Codex via
its CLI) whose only tools are the Lightyear MCP tools for the chosen dataset (no shell, no file access), so the
model sees exactly what it sees in Claude Code: aggregates and figures computed locally. The first run shows a
setup screen to pick the provider and sign in (see auth.py for what is offered and why).
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .. import tools
from ..claims import fingerprint, load_ledger
from ..runtime import FROZEN
from ..store import Store
from . import auth
from .runners import make_runner
from .settings import Settings, set_key

STATIC = Path(__file__).parent / "static"
LEDGER_TOOLS = {"test_claim", "recheck_claims", "add_finding"}
DEFAULT_STORES = Path.home() / "Lightyear" / "stores" if FROZEN else Path("stores")

app = FastAPI(title="Lightyear")
STORES_OVERRIDE: Path | None = None   # --stores on the command line wins over the settings
WINDOW = None                         # the pywebview window, when running as the desktop app


def stores_root() -> Path:
    if STORES_OVERRIDE:
        return STORES_OVERRIDE
    s = Settings.load()
    return Path(s.stores_dir) if s.stores_dir else DEFAULT_STORES.resolve()


def discover(root: Path) -> dict[str, Path]:
    return {p.parent.name: p.parent.resolve() for p in sorted(Path(root).glob("*/meta.json"))}


def _stores() -> dict[str, Path]:
    return discover(stores_root())


def _store(ds: str) -> Store:
    stores = _stores()
    if ds not in stores:
        raise HTTPException(404, f"unknown dataset {ds!r}")
    return Store(stores[ds])


# -- read-only API -------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/static/{name:path}")
def static(name: str):
    p = (STATIC / name).resolve()
    if not p.is_relative_to(STATIC.resolve()) or not p.is_file():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/api/datasets")
def datasets():
    out = []
    for ds, path in _stores().items():
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


# -- setup: model provider, sign-in, keys ---------------------------------------------------
@app.get("/api/setup")
def setup_status():
    st = auth.status()
    st["stores_dir"] = str(stores_root())
    st["frozen"] = FROZEN
    st["can_pick_folder"] = WINDOW is not None and not STORES_OVERRIDE
    return st


@app.post("/api/pick-folder")
def pick_folder():
    """Native folder chooser (only in the app window; a browser page can't see real paths)."""
    if WINDOW is None:
        raise HTTPException(400, "folder picker only available in the app window")
    import webview
    picked = WINDOW.create_file_dialog(webview.FileDialog.FOLDER, directory=str(stores_root().parent))
    return {"path": picked[0] if picked else None}


@app.post("/api/setup")
def setup_save(body: dict):
    s = Settings.load()
    if body.get("provider") in ("claude", "codex"):
        s.provider = body["provider"]
    for prov, mode in (body.get("auth") or {}).items():
        if prov in ("claude", "codex") and mode in ("login", "api_key"):
            if prov == "claude" and mode == "login" and not auth.ALLOW_CLAUDE_LOGIN:
                raise HTTPException(400, "Claude sign-in is not offered in the app; use an API key")
            s.auth[prov] = mode
    for prov, model in (body.get("model") or {}).items():
        if prov in ("claude", "codex"):
            s.model[prov] = (model or "").strip() or None
    if body.get("stores_dir") and not STORES_OVERRIDE:
        d = Path(os.path.expanduser(body["stores_dir"])).resolve()
        if d == DEFAULT_STORES.resolve():
            d.mkdir(parents=True, exist_ok=True)  # the default is ours to create
        if not d.is_dir():
            raise HTTPException(400, f"Folder not found: {d}. Under Advanced, set the datasets folder to the folder "
                                     "that holds your dataset stores.")
        s.stores_dir = str(d)
    s.save()
    return setup_status()


@app.post("/api/setup/key")
def setup_key(body: dict):
    prov, key = body.get("provider"), (body.get("key") or "").strip()
    if prov not in ("claude", "codex") or not key:
        raise HTTPException(400, "provider and key required")
    ok, detail = auth.validate_key(prov, key)
    if not ok:
        return {"ok": False, "detail": detail}
    where = set_key(prov, key)
    return {"ok": True, "detail": f"{detail}; saved to {where}"}


@app.delete("/api/setup/key/{prov}")
def setup_key_delete(prov: str):
    if prov not in ("claude", "codex"):
        raise HTTPException(404)
    set_key(prov, None)
    return setup_status()


# -- agent session ---------------------------------------------------------------------
@app.websocket("/ws/{ds}")
async def chat(ws: WebSocket, ds: str):
    await ws.accept()
    stores = _stores()
    settings = Settings.load()
    if ds not in stores:
        await ws.send_json({"type": "error", "text": f"unknown dataset {ds!r}"})
        await ws.close()
        return
    if not auth.ready(settings, auth.status(settings)):
        await ws.send_json({"type": "setup_needed"})
        await ws.close()
        return
    try:
        runner = make_runner(stores[ds], settings)
        await runner.start()
    except Exception as e:
        await ws.send_json({"type": "error", "text": f"could not start {settings.provider}: {e}"})
        await ws.close()
        return

    async def emit(ev: dict):
        await ws.send_json(ev)
        if ev["type"] == "tool_result" and ev.get("name") in LEDGER_TOOLS:
            await ws.send_json({"type": "ledger_changed"})

    async def run(text: str):
        try:
            await runner.ask(text, emit)
        except Exception as e:
            await ws.send_json({"type": "error", "text": f"{type(e).__name__}: {e}"})

    task = None
    try:
        await ws.send_json({"type": "ready", "provider": settings.provider})
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "ask" and msg.get("text", "").strip():
                if task and not task.done():
                    await ws.send_json({"type": "error", "text": "still working on the previous question"})
                    continue
                task = asyncio.create_task(run(msg["text"]))
            elif msg.get("type") == "interrupt":
                await runner.interrupt()
    except WebSocketDisconnect:
        pass
    finally:
        if task and not task.done():
            task.cancel()
        await runner.close()


def _free_port(host: str, port: int) -> int:
    import socket
    for p in range(port, port + 30):
        with socket.socket() as sk:
            if sk.connect_ex((host, p)) != 0:
                return p
    raise SystemExit(f"no free port in {port}-{port + 29}")


def main(stores: str | None = None, host: str = "127.0.0.1", port: int = 8790, open_browser: bool = True,
         window: bool = False):
    """Serve the app. window=True opens it in a native window (pywebview) and quits when that closes."""
    import threading
    import uvicorn
    global STORES_OVERRIDE
    port = _free_port(host, port)
    if FROZEN and not stores:
        DEFAULT_STORES.mkdir(parents=True, exist_ok=True)
    if stores:
        STORES_OVERRIDE = Path(stores).resolve()
    url = f"http://{host}:{port}"
    print(f"Lightyear: datasets in {stores_root()} -> {url}")
    if window:
        try:
            import webview
        except ImportError:
            window = False
    if not window:
        if open_browser:
            import webbrowser
            threading.Timer(1.5, lambda: webbrowser.open(url)).start()
        uvicorn.run(app, host=host, port=port, log_level="warning")
        return
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    import time
    while not server.started:
        time.sleep(0.05)
    global WINDOW
    WINDOW = webview.create_window("Lightyear", url, width=1360, height=900, min_size=(760, 560))
    webview.start()
    server.should_exit = True
