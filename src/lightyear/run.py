"""Run agent-written analysis code against a store, reproducibly.

Each call becomes a numbered script in <store>/analyses/ with a fixed prelude, runs in a
subprocess (time-limited), and returns stdout (truncated) plus any figures it produced.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

from .store import Store

PRELUDE = '''\
# --- lightyear prelude (auto-generated) ---
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightyear import load, stats, spatial, analysis, tools, plots
s = load({store!r})                    # Store: s.obs, s.xy, s.gene(name), s.mask(filters), s.knn, s.roles
OUT = __import__("pathlib").Path({out!r})  # write tables/figures here
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
# --- analysis: {title} ---
'''

EPILOGUE = '''
# --- lightyear epilogue: save any open figures ---
for _i in plt.get_fignums():
    plt.figure(_i).savefig(OUT / f"fig{_i}.png", dpi=110, bbox_inches="tight")
'''


def run_python(s: Store, code: str, title: str = "analysis", timeout: int = 900, max_chars: int = 12000):
    root = (s.path / "analyses").resolve()
    root.mkdir(exist_ok=True)
    n = len(list(root.glob("[0-9][0-9][0-9]_*.py"))) + 1
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")[:50] or "analysis"
    out = root / f"{n:03d}_{slug}"
    out.mkdir(exist_ok=True)
    script = root / f"{n:03d}_{slug}.py"
    script.write_text(PRELUDE.format(store=str(s.path.resolve()), out=str(out.resolve()), title=title)
                      + code.strip() + "\n" + EPILOGUE)
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=timeout,
                           cwd=out, env={"OPENBLAS_NUM_THREADS": "4", **os.environ})
        text, ok = r.stdout + (("\n[stderr]\n" + r.stderr[-4000:]) if r.returncode else ""), r.returncode == 0
    except subprocess.TimeoutExpired as e:
        text, ok = (e.stdout or "") + f"\n[timed out after {timeout}s]", False
    if len(text) > max_chars:
        text = text[:max_chars // 2] + f"\n... [{len(text) - max_chars} chars truncated] ...\n" + text[-max_chars // 2:]
    figs = sorted(p for p in out.glob("*.png") if p.stat().st_mtime >= t0 - 1)
    (out / "stdout.txt").write_text(text)
    header = f"[{'ok' if ok else 'FAILED'} in {time.time() - t0:.0f}s] script: {script}\n"
    return header + text, figs
