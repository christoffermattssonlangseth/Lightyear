"""A query-friendly on-disk copy of a dataset.

Layout of a store directory:
    meta.json        schema, roles, provenance
    obs.parquet      all cell-level metadata (categoricals preserved)
    spatial.npy      (n_cells, 2) float32 coordinates
    expr.npy         (n_genes, n_cells) float16, gene-major -> one gene is one contiguous read
      or expr_csc/   indptr.npy (n_genes+1), indices.npy (int32 cell ids), data.npy (float16): sparse
                     gene-major, used when the dense matrix would be too large (e.g. Xenium 5K)
    card/            dataset card + precomputed tables (written by ingest)
    findings.md      notes the agent accumulates across sessions

Gene-major dense float16 keeps per-gene queries O(n_cells) with no sparse
decoding; it is used for targeted panels (a few hundred genes). Larger panels use
the sparse CSC variant: still one contiguous slice per gene.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .schema import inspect_h5ad, read_spatial

MAX_DENSE_ELEMENTS = 6e9  # ~12 GB at float16
DENSE_BYTES_LIMIT = 4e9  # above this the store is written sparse (CSC) unless forced


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


def _write_expr_csc(path_h5ad, out: Path, n_cells: int, n_genes: int, layer: str | None, chunk: int,
                    log=print):
    """CSR (cells x genes) in the h5ad -> gene-major CSC on disk, two passes over row chunks."""
    d = out / "expr_csc"
    d.mkdir(exist_ok=True)
    with h5py.File(path_h5ad, "r") as f:
        X = _x_group(f, layer)
        enc = X.attrs.get("encoding-type", "array") if isinstance(X, h5py.Group) else "array"
        enc = enc.decode() if isinstance(enc, bytes) else enc
        if enc != "csr_matrix":
            raise NotImplementedError(f"sparse store needs a CSR matrix, got {enc}")
        indptr = X["indptr"][:].astype(np.int64)
        nnz = int(indptr[-1])
        counts = np.zeros(n_genes, np.int64)
        for a in range(0, nnz, 200_000_000):
            counts += np.bincount(X["indices"][a:a + 200_000_000], minlength=n_genes)
        gptr = np.r_[0, np.cumsum(counts)]
        np.save(d / "indptr.npy", gptr)
        ind = np.lib.format.open_memmap(d / "indices.npy", "w+", np.int32, (nnz,))
        dat = np.lib.format.open_memmap(d / "data.npy", "w+", np.float16, (nnz,))
        cursor = gptr[:-1].copy()
        for r0 in range(0, n_cells, chunk):
            r1 = min(r0 + chunk, n_cells)
            a, b = indptr[r0], indptr[r1]
            g = X["indices"][a:b]
            v = X["data"][a:b]
            rows = np.repeat(np.arange(r0, r1, dtype=np.int32), np.diff(indptr[r0:r1 + 1]))
            order = np.argsort(g, kind="stable")  # stable: cells stay ascending within each gene
            g, v, rows = g[order], v[order], rows[order]
            c = np.bincount(g, minlength=n_genes)
            seg_start = np.r_[0, np.cumsum(c)[:-1]]
            pos = cursor[g] + (np.arange(len(g)) - seg_start[g])
            ind[pos] = rows
            dat[pos] = v.astype(np.float16)
            cursor += c
            log(f"  expression: {r1:,}/{n_cells:,} cells")
        ind.flush(); dat.flush()
        del ind, dat
    assert (cursor == gptr[1:]).all()


def build_store(path_h5ad, out, overrides: dict | None = None, layer: str | None = None,
                chunk: int = 100_000, name: str | None = None, sparse: bool | None = None, log=print) -> "Store":
    """Convert an .h5ad into a store directory. `overrides` comes from the dataset yaml."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    overrides = overrides or {}
    schema, obs = inspect_h5ad(path_h5ad, overrides)

    if sparse is None:
        sparse = schema.n_cells * schema.n_genes * 2 > DENSE_BYTES_LIMIT
    if not sparse and schema.n_cells * schema.n_genes > MAX_DENSE_ELEMENTS:
        raise NotImplementedError(f"{schema.n_cells} cells x {schema.n_genes} genes is too large to store dense")

    obs = obs.copy()
    obs.index = obs.index.astype(str)
    for c in obs.columns:
        if obs[c].dtype == object:
            obs[c] = obs[c].astype(str).astype("category")
    obs.to_parquet(out / "obs.parquet")

    xy, _ = read_spatial(path_h5ad, obs)
    np.save(out / "spatial.npy", xy)

    if sparse:
        _write_expr_csc(path_h5ad, out, schema.n_cells, schema.n_genes, layer, min(chunk, 50_000), log)
    else:
        _write_expr(path_h5ad, out, schema.n_cells, schema.n_genes, layer, chunk)

    meta = {
        "name": name or Path(path_h5ad).stem,
        "source": str(Path(path_h5ad).resolve()),
        "expr_layer": layer or "X",
        "expr_format": "csc" if sparse else "dense",
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
        for name in self.meta.get("derived", {}):  # older stores leaked derived columns into the schema
            s["columns"].pop(name, None)
        self.columns: dict = {**s["columns"], **self.meta.get("derived", {})}
        self._obs = None
        self._xy = None
        self._expr = None
        self._knn = None
        self._strcols: dict = {}

    # -- lazy data ---------------------------------------------------------
    @property
    def obs(self) -> pd.DataFrame:
        if self._obs is None:
            obs = pd.read_parquet(self.path / "obs.parquet")
            d = self.path / "derived.parquet"
            if d.exists():
                der = pd.read_parquet(d)
                der.index = obs.index
                obs = pd.concat([obs.drop(columns=[c for c in der.columns if c in obs.columns]), der], axis=1)
            self._obs = obs
        return self._obs

    @property
    def xy(self) -> np.ndarray:
        if self._xy is None:
            self._xy = np.load(self.path / "spatial.npy")
        return self._xy

    @property
    def sparse(self) -> bool:
        return self.meta.get("expr_format") == "csc"

    @property
    def expr(self):
        """Dense: (n_genes, n_cells) memmap. Sparse: (indptr, indices, data) memmaps of the CSC store."""
        if self._expr is None:
            if self.sparse:
                d = self.path / "expr_csc"
                self._expr = (np.load(d / "indptr.npy"), np.load(d / "indices.npy", mmap_mode="r"),
                              np.load(d / "data.npy", mmap_mode="r"))
            else:
                self._expr = np.load(self.path / "expr.npy", mmap_mode="r")
        return self._expr

    def gene_nz(self, g: str) -> tuple[np.ndarray, np.ndarray]:
        """(cell indices, values) of the non-zero entries of a panel gene."""
        g = self.resolve_gene(g)
        if g not in self.gene_index:
            v = self.gene(g)
            nz = np.flatnonzero(v)
            return nz, v[nz]
        j = self.gene_index[g]
        if self.sparse:
            ptr, ind, dat = self.expr
            return np.asarray(ind[ptr[j]:ptr[j + 1]]), np.asarray(dat[ptr[j]:ptr[j + 1]], np.float32)
        v = np.asarray(self.expr[j], np.float32)
        nz = np.flatnonzero(v)
        return nz, v[nz]

    @property
    def knn(self) -> tuple[np.ndarray, np.ndarray]:
        """(indices, distances) of each cell's spatial nearest neighbours within its section, nearest first
        (self excluded). Built at ingest; see spatial.ensure_knn."""
        if self._knn is None:
            self._knn = (np.load(self.path / "knn_idx.npy", mmap_mode="r"),
                         np.load(self.path / "knn_dist.npy", mmap_mode="r"))
        return self._knn

    @property
    def n_cells(self) -> int:
        return self.meta["schema"]["n_cells"]

    @property
    def card_dir(self) -> Path:
        d = self.path / "card"
        d.mkdir(exist_ok=True)
        return d

    # -- helpers -----------------------------------------------------------
    def save_meta(self):
        (self.path / "meta.json").write_text(json.dumps(self.meta, indent=1, default=str))

    def add_column(self, name: str, values, description: str, role: str = "derived", overwrite: bool = True):
        """Persist a new per-cell column (niche labels, gates, scores, distances) so all tools can use it."""
        if name in self.meta["schema"]["columns"]:
            raise ValueError(f"{name!r} is an original column; pick another name")
        if name in self.obs.columns and not overwrite:
            raise ValueError(f"{name!r} exists")
        values = pd.Series(np.asarray(values), index=self.obs.index, name=name)
        if values.dtype == object:
            values = values.astype("category")
        d = self.path / "derived.parquet"
        der = pd.read_parquet(d) if d.exists() else pd.DataFrame(index=np.arange(self.n_cells))
        der.index = np.arange(self.n_cells)
        der[name] = values.to_numpy() if not isinstance(values.dtype, pd.CategoricalDtype) else values.values
        der.to_parquet(d)
        n_unique = int(values.nunique())
        info = {"name": name, "dtype": str(values.dtype), "n_unique": n_unique, "role": role, "level": "cell",
                "description": description, "top_values": [], "numeric_summary": {}}
        if n_unique <= 50 or not pd.api.types.is_float_dtype(values):
            info["top_values"] = [[str(k), int(v)] for k, v in values.astype(str).value_counts().head(40).items()]
        self.meta.setdefault("derived", {})[name] = info
        self.columns[name] = info
        self.save_meta()
        self._obs = None
        self._strcols = {}
        return info

    def strcol(self, name: str) -> np.ndarray:
        """Column as a cached numpy array of str (fast repeated lookups of sample / group / cell type)."""
        if name not in self._strcols:
            c = self.obs[name]
            if isinstance(c.dtype, pd.CategoricalDtype):
                cats = np.asarray(c.cat.categories.astype(str), dtype=object)
                codes = c.cat.codes.to_numpy()
                v = np.where(codes >= 0, cats[np.maximum(codes, 0)], "nan")
            else:
                v = c.astype(str).to_numpy()
            self._strcols[name] = v.astype(str)
        return self._strcols[name]

    def is_numeric_column(self, name: str) -> bool:
        return name in self.obs.columns and pd.api.types.is_numeric_dtype(self.obs[name]) \
            and self.obs[name].dtype != bool

    def resolve_gene(self, g: str) -> str:
        """Gene symbol (case-insensitive) or a numeric obs column such as a gene-set score."""
        if g in self.gene_index:
            return g
        if self.is_numeric_column(g):
            return g
        hit = self._gene_lower.get(g.lower())
        if hit is None:
            close = [x for x in self.genes if x.lower().startswith(g.lower()[:3])][:10]
            raise KeyError(f"gene {g!r} not in panel; similar: {close}")
        return hit

    def gene(self, g: str, mask: np.ndarray | None = None) -> np.ndarray:
        g = self.resolve_gene(g)
        if g in self.gene_index and self.sparse:
            idx, val = self.gene_nz(g)
            v = np.zeros(self.n_cells, np.float32)
            v[idx] = val
        elif g in self.gene_index:
            v = np.asarray(self.expr[self.gene_index[g]], dtype=np.float32)
        else:
            v = self.obs[g].to_numpy(np.float32)
        return v if mask is None else v[mask]

    def role(self, r: str) -> str | None:
        if r == "section":  # spatial unit (one coordinate frame / tissue piece); defaults to the sample
            return self.roles.get("section") or self.roles.get("sample")
        return self.roles.get(r)

    def mask(self, filters: dict | None) -> np.ndarray:
        """filters: {column: value | [values]}; {column: {"not": [values]}} excludes;
        numeric columns and genes accept {">": x, "<=": y}, e.g. {"Spp1": {">": 1}}."""
        m = np.ones(self.n_cells, dtype=bool)
        for col, val in (filters or {}).items():
            if col not in self.obs.columns:
                try:
                    v = self.gene(col)
                except KeyError:
                    raise KeyError(f"unknown column or gene {col!r}") from None
                if not isinstance(val, dict):
                    raise ValueError(f"gene filter {col!r} needs a comparison, e.g. {{'>': 0}}")
                for op, x in val.items():
                    m &= {">": v > x, ">=": v >= x, "<": v < x, "<=": v <= x}[op]
                continue
            s = self.obs[col]
            if isinstance(val, dict) and "not" in val:
                m &= ~self.mask({col: val["not"]})
            elif isinstance(val, dict):
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
