"""A query-friendly on-disk copy of a dataset.

Layout of a store directory:
    meta.json        schema, roles, provenance
    obs.parquet      all cell-level metadata (categoricals preserved)
    spatial.npy      (n_cells, 2) float32 coordinates
    expr.npy         (n_genes, n_cells) float16, gene-major -> one gene is one contiguous read
    card/            dataset card + precomputed tables (written by ingest)
    findings.md      notes the agent accumulates across sessions

Gene-major dense float16 keeps per-gene queries O(n_cells) with no sparse
decoding; it is used for targeted panels (Xenium/MERSCOPE/CosMx, a few hundred
to a few thousand genes). Whole-transcriptome data would need a sparse CSC
variant (not implemented yet).
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .schema import inspect_h5ad, read_spatial

MAX_DENSE_ELEMENTS = 6e9  # ~12 GB at float16


def _x_group(f: h5py.File, layer: str | None):
    return f["X"] if layer in (None, "X") else f["layers"][layer]


def _write_expr(path_h5ad, out: Path, n_cells: int, n_genes: int, layer: str | None, chunk: int):
    mm = np.lib.format.open_memmap(out / "expr.npy", mode="w+", dtype=np.float16, shape=(n_genes, n_cells))
    with h5py.File(path_h5ad, "r") as f:
        X = _x_group(f, layer)
        enc = X.attrs.get("encoding-type", "array") if isinstance(X, h5py.Group) else "array"
        enc = enc.decode() if isinstance(enc, bytes) else enc
        if enc == "csr_matrix":
            indptr = X["indptr"][:]
            for r0 in range(0, n_cells, chunk):
                r1 = min(r0 + chunk, n_cells)
                a, b = indptr[r0], indptr[r1]
                data = X["data"][a:b]
                idx = X["indices"][a:b]
                rows = np.repeat(np.arange(r1 - r0), np.diff(indptr[r0:r1 + 1]))
                block = np.zeros((n_genes, r1 - r0), dtype=np.float32)
                block[idx, rows] = data
                mm[:, r0:r1] = block.astype(np.float16)
        elif enc == "csc_matrix":
            indptr = X["indptr"][:]
            for g in range(n_genes):
                a, b = indptr[g], indptr[g + 1]
                col = np.zeros(n_cells, dtype=np.float16)
                col[X["indices"][a:b]] = X["data"][a:b]
                mm[g] = col
        else:  # dense
            for r0 in range(0, n_cells, chunk):
                r1 = min(r0 + chunk, n_cells)
                mm[:, r0:r1] = np.asarray(X[r0:r1], dtype=np.float32).T.astype(np.float16)
    mm.flush()
    del mm


def build_store(path_h5ad, out, overrides: dict | None = None, layer: str | None = None,
                chunk: int = 100_000, name: str | None = None) -> "Store":
    """Convert an .h5ad into a store directory. `overrides` comes from the dataset yaml."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    overrides = overrides or {}
    schema, obs = inspect_h5ad(path_h5ad, overrides)

    if schema.n_cells * schema.n_genes > MAX_DENSE_ELEMENTS:
        raise NotImplementedError(
            f"{schema.n_cells} cells x {schema.n_genes} genes is too large for the dense store; "
            "subset genes or add a sparse backend")

    obs = obs.copy()
    obs.index = obs.index.astype(str)
    for c in obs.columns:
        if obs[c].dtype == object:
            obs[c] = obs[c].astype(str).astype("category")
    obs.to_parquet(out / "obs.parquet")

    xy, _ = read_spatial(path_h5ad, obs)
    np.save(out / "spatial.npy", xy)

    _write_expr(path_h5ad, out, schema.n_cells, schema.n_genes, layer, chunk)

    meta = {
        "name": name or Path(path_h5ad).stem,
        "source": str(Path(path_h5ad).resolve()),
        "expr_layer": layer or "X",
        "expr_description": overrides.get("expr_description", ""),
        "dataset_description": overrides.get("description", ""),
        "schema": schema.to_dict(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return Store(out)


class Store:
    """Read side of a store. Everything is lazy; expr is a memmap."""

    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        s = self.meta["schema"]
        self.genes: list[str] = s["genes"]
        self.gene_index = {g: i for i, g in enumerate(self.genes)}
        self._gene_lower = {g.lower(): g for g in self.genes}
        self.roles: dict = s["roles"]
        self.columns: dict = s["columns"]
        self._obs = None
        self._xy = None
        self._expr = None

    # -- lazy data ---------------------------------------------------------
    @property
    def obs(self) -> pd.DataFrame:
        if self._obs is None:
            self._obs = pd.read_parquet(self.path / "obs.parquet")
        return self._obs

    @property
    def xy(self) -> np.ndarray:
        if self._xy is None:
            self._xy = np.load(self.path / "spatial.npy")
        return self._xy

    @property
    def expr(self) -> np.ndarray:
        if self._expr is None:
            self._expr = np.load(self.path / "expr.npy", mmap_mode="r")
        return self._expr

    @property
    def n_cells(self) -> int:
        return self.meta["schema"]["n_cells"]

    @property
    def card_dir(self) -> Path:
        d = self.path / "card"
        d.mkdir(exist_ok=True)
        return d

    # -- helpers -----------------------------------------------------------
    def resolve_gene(self, g: str) -> str:
        if g in self.gene_index:
            return g
        hit = self._gene_lower.get(g.lower())
        if hit is None:
            close = [x for x in self.genes if x.lower().startswith(g.lower()[:3])][:10]
            raise KeyError(f"gene {g!r} not in panel; similar: {close}")
        return hit

    def gene(self, g: str, mask: np.ndarray | None = None) -> np.ndarray:
        v = np.asarray(self.expr[self.gene_index[self.resolve_gene(g)]], dtype=np.float32)
        return v if mask is None else v[mask]

    def role(self, r: str) -> str | None:
        return self.roles.get(r)

    def mask(self, filters: dict | None) -> np.ndarray:
        """filters: {column: value | [values]} ; numeric columns accept {">": x, "<": y}."""
        m = np.ones(self.n_cells, dtype=bool)
        for col, val in (filters or {}).items():
            if col not in self.obs.columns:
                raise KeyError(f"unknown column {col!r}")
            s = self.obs[col]
            if isinstance(val, dict):
                v = s.to_numpy(float)
                for op, x in val.items():
                    m &= {">": v > x, ">=": v >= x, "<": v < x, "<=": v <= x}[op]
            else:
                vals = [str(x) for x in (val if isinstance(val, (list, tuple)) else [val])]
                if isinstance(s.dtype, pd.CategoricalDtype):
                    keep = np.isin(s.cat.categories.astype(str), vals)
                    codes = s.cat.codes.to_numpy()
                    m &= (codes >= 0) & keep[np.maximum(codes, 0)]
                else:
                    m &= s.astype(str).isin(vals).to_numpy()
        return m

    def findings_path(self) -> Path:
        return self.path / "findings.md"


def load(path) -> Store:
    return Store(path)
