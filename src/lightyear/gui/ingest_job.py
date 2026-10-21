"""Add a dataset from the app: inspect an .h5ad (obs only), then run `lightyear ingest` in a subprocess.

The store is built in a hidden `.ingest-<id>` folder next to the others and renamed when complete, so a
half-built store never shows up as a dataset. One ingest runs at a time.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

from ..runtime import cli_cmd
from ..schema import inspect_h5ad

ROLE_KEYS = ("sample", "cell_type", "group", "section")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_-").lower()[:40] or "dataset"


def _looks_like_counts(path: Path, layer: str | None) -> bool:
    """True if the first stored values are all whole numbers (raw counts, not log-normalised)."""
    with h5py.File(path, "r") as f:
        X = f["layers"][layer] if layer else f["X"]
        vals = X["data"][:5000] if isinstance(X, h5py.Group) else X[:min(50, X.shape[0])]
    vals = np.asarray(vals, dtype=float).ravel()
    vals = vals[vals != 0]
    return vals.size > 0 and bool(np.all(vals == np.round(vals))) and vals.max() > 1


def inspect(path: str) -> dict:
    p = Path(path).expanduser()
    if p.suffix != ".h5ad" or not p.is_file():
        raise ValueError(f"Not an .h5ad file: {p}")
    schema, _ = inspect_h5ad(p)
    columns = [{"name": c.name, "n_unique": c.n_unique, "role": c.role, "level": c.level,
                "examples": [v for v, _ in c.top_values[:4]]}
               for c in schema.columns.values()
               if c.top_values and c.role not in ("coordinate", "qc") and 1 < c.n_unique <= 500]
    layers = schema.structures.get("layers", [])
    return {"path": str(p.resolve()), "n_cells": schema.n_cells, "n_genes": schema.n_genes,
            "spatial": schema.spatial_key, "layers": layers, "x_counts": _looks_like_counts(p, None),
            "columns": columns, "roles": {k: schema.roles.get(k) for k in ROLE_KEYS},
            "id": slug(p.stem), "name": p.stem}


class Job:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = "idle"   # idle | running | done | failed | cancelled
        self.id = self.error = None
        self.log: list[str] = []
        self.proc: subprocess.Popen | None = None
        self.t0 = 0.0

    def status(self) -> dict:
        return {"state": self.state, "id": self.id, "error": self.error, "log": self.log[-40:],
                "elapsed_s": round(time.time() - self.t0) if self.t0 else 0}

    def start(self, root: Path, spec: dict) -> dict:
        path = Path(spec.get("path") or "").expanduser()
        sid = (spec.get("id") or "").strip()
        roles = {k: v for k, v in (spec.get("roles") or {}).items() if k in ROLE_KEYS and v}
        if not path.is_file():
            raise ValueError(f"File not found: {path}")
        if not ID_RE.match(sid):
            raise ValueError("Short name: letters, digits, - and _ only (no spaces)")
        if (root / sid).exists():
            raise ValueError(f"A dataset called {sid!r} already exists in {root}")
        for k in ("sample", "cell_type"):
            if not roles.get(k):
                raise ValueError("Choose the replicate column and the cell type column")
        cfg = {"name": (spec.get("name") or sid).strip(), "roles": roles}
        for k in ("description", "organism"):
            if (spec.get(k) or "").strip():
                cfg[k] = spec[k].strip()
        layer = spec.get("layer") or None
        if layer == "X":
            layer = None
        with self.lock:
            if self.state == "running":
                raise ValueError("Another dataset is still being added")
            root.mkdir(parents=True, exist_ok=True)
            tmp = root / f".ingest-{sid}"
            shutil.rmtree(tmp, ignore_errors=True)  # leftover of an interrupted run of ours
            tmp.mkdir()
            (tmp / "lightyear.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            cmd = cli_cmd() + ["ingest", str(path), str(tmp), "--config", str(tmp / "lightyear.yaml")]
            if layer:
                cmd += ["--layer", layer]
            self.state, self.id, self.error, self.log, self.t0 = "running", sid, None, [], time.time()
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                         stdin=subprocess.DEVNULL, cwd=str(root))
            threading.Thread(target=self._watch, args=(self.proc, tmp, root / sid), daemon=True).start()
        return self.status()

    def _watch(self, proc: subprocess.Popen, tmp: Path, final: Path):
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                self.log.append(line)
                del self.log[:-400]
        code = proc.wait()
        with self.lock:
            if self.state == "cancelled":
                shutil.rmtree(tmp, ignore_errors=True)
            elif code == 0 and (tmp / "meta.json").exists():
                tmp.rename(final)
                self.state = "done"
            else:
                shutil.rmtree(tmp, ignore_errors=True)
                tail = [l for l in self.log if l.strip()][-1:] or [f"exit code {code}"]
                self.state, self.error = "failed", tail[0]
            self.proc = None

    def cancel(self) -> dict:
        with self.lock:
            if self.state == "running" and self.proc:
                self.state = "cancelled"
                self.proc.terminate()
        return self.status()


JOB = Job()
