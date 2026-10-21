"""Release gate: run the *built* app as a brand-new user and walk through the setup screen in a browser.

    python packaging/check_first_run.py dist/Lightyear.app [--ask]

The app runs with an empty HOME (no settings, no ~/Lightyear, no CLI sign-ins), exactly as after a download.
Checks: the setup screen opens without errors; Start with the prefilled defaults gives the right next step
(not an error); the bundled `ingest` builds a store; once a dataset and a provider exist the chat connects
and the side panels load. With Codex signed in on this machine (~/.codex), the last part uses it; --ask also
sends one real question. Any failure exits non-zero, so build_mac.sh stops before notarising.
Needs numpy, anndata and playwright (with Chrome) in the Python running this script.
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright


def fail(msg: str):
    print(f"FIRST-RUN CHECK FAILED: {msg}")
    sys.exit(1)


def toy_h5ad(path: Path):
    rng = np.random.default_rng(0)
    n, types = 3000, ["Astro", "Micro", "Oligo", "Neuron"]
    obs = pd.DataFrame({"cell_type": rng.choice(types, n), "sample": rng.choice([f"s{i}" for i in range(6)], n)})
    obs["condition"] = np.where(obs["sample"].isin(["s0", "s1", "s2"]), "ctrl", "treated")
    X = rng.poisson(1.0, (n, 40)).astype("float32")
    a = ad.AnnData(np.log1p(X), obs=obs, var=pd.DataFrame(index=[f"Gene{i}" for i in range(40)]))
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obsm["spatial"] = rng.uniform(0, 2000, (n, 2))
    a.write_h5ad(path)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("app")
    ap.add_argument("--ask", action="store_true", help="also ask one real question (uses the provider)")
    args = ap.parse_args()
    binary = Path(args.app) / "Contents" / "MacOS" / "Lightyear"
    if not binary.exists():
        fail(f"no app binary at {binary}")

    tmp = Path(tempfile.mkdtemp(prefix="lightyear-firstrun-"))
    home = tmp / "home"
    home.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("LIGHTYEAR_")}
    env["HOME"] = str(home)
    real_codex = Path.home() / ".codex"
    if (real_codex / "auth.json").exists():
        env["CODEX_HOME"] = str(real_codex)   # let Codex see this machine's sign-in; nothing else of ours
    port = free_port()
    log = open(tmp / "app.log", "w")
    proc = subprocess.Popen([str(binary), "gui", "--no-browser", "--port", str(port)], env=env,
                            stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):  # first start rebuilds the font cache in the fresh home
            try:
                urllib.request.urlopen(url + "/api/setup", timeout=2)
                break
            except OSError:
                if proc.poll() is not None:
                    fail(f"app exited ({proc.returncode}); log:\n" + (tmp / "app.log").read_text()[-3000:])
                time.sleep(1)
        else:
            fail("app did not start within 120 s")
        print(f"app up at {url} (home {home})")

        with sync_playwright() as p:
            b = p.chromium.launch(channel="chrome")
            pg = b.new_page(viewport={"width": 1300, "height": 860})
            errors = []
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.on("response", lambda r: r.status >= 400 and errors.append(f"HTTP {r.status} {r.url}"))
            pg.on("console", lambda m: m.type == "error" and errors.append(f"console: {m.text} {m.location.get('url', '')}"))

            # 1. a fresh install opens on the setup screen, with nothing shown as an error
            pg.goto(url)
            pg.wait_for_selector("#setupdlg[open]", timeout=30000)
            pg.wait_for_timeout(500)
            bad = pg.text_content("#setup-msg").strip()
            if bad:
                fail(f"setup screen shows a message on open: {bad!r}")
            pg.screenshot(path=str(tmp / "1_setup.png"))

            # 2. Start with every default as prefilled: must ask for the next step, not complain about folders
            pg.click("label.provider:has(input[value=claude])")
            pg.click("#setup-start")
            pg.wait_for_function("document.querySelector('#setup-msg').textContent.length > 0", timeout=20000)
            msg = pg.text_content("#setup-msg")
            if "folder" in msg.lower() or "error" in msg.lower():
                fail(f"Start with defaults shows {msg!r}")
            if not (home / "Lightyear" / "stores").is_dir():
                fail("default datasets folder ~/Lightyear/stores was not created")
            print(f"defaults ok: {msg!r}")

            # 3. the bundled CLI ingests a dataset into the default folder
            toy_h5ad(tmp / "toy.h5ad")
            r = subprocess.run([str(binary), "ingest", str(tmp / "toy.h5ad"), str(home / "Lightyear" / "stores" / "toy")],
                               env=env, capture_output=True, text=True, timeout=900)
            if r.returncode or not (home / "Lightyear" / "stores" / "toy" / "meta.json").exists():
                fail(f"bundled ingest failed:\n{(r.stdout + r.stderr)[-3000:]}")
            print("bundled ingest ok")

            # 4. with a signed-in provider, setup completes and the chat connects
            pg.click("#setup-recheck")
            pg.wait_for_timeout(1500)
            codex_ok = pg.evaluate("setup && setup.codex.login.available")
            if not codex_ok:
                print("Codex not signed in on this machine: skipping the connect check")
            else:
                pg.click("label.provider:has(input[value=codex])")
                pg.click("label.opt:has(input[name=codex-auth][value=login])")
                pg.click("#setup-start")
                pg.wait_for_function("!document.querySelector('#setupdlg').open", timeout=20000)
                pg.wait_for_function("document.querySelector('#status').textContent.startsWith('ready')", timeout=60000)
                for tab in ("claims", "findings", "card"):
                    pg.click(f".tabs button[data-tab={tab}]")
                    pg.wait_for_timeout(400)
                    if not pg.text_content(f"#tab-{tab}").strip():
                        fail(f"{tab} panel is empty")
                print("setup complete, chat connected, panels load")
                if args.ask:
                    pg.fill("#input", "How many cells and cell types are in this dataset? One line.")
                    pg.press("#input", "Enter")
                    pg.wait_for_selector(".msg.done", timeout=300000)
                    if pg.query_selector(".msg .err"):
                        fail("question produced an error: " + pg.text_content(".msg .err"))
                    print("question answered")
                pg.screenshot(path=str(tmp / "2_ready.png"))
            if errors:
                fail("browser errors: " + "; ".join(errors[:5]))
            b.close()
        print(f"FIRST-RUN CHECK PASSED (screenshots in {tmp})")
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
