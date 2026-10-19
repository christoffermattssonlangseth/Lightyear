"""Light-weight inspection of an .h5ad and inference of what each obs column *means*.

Only obs / var / obsm['spatial'] / uns descriptions are read here; X, layers and
obsp are never loaded (they can be many GB).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem

# name patterns, checked in order; first match wins
ROLE_PATTERNS = [
    ("sample", r"^(sample|sample_?id|section|slide|orig\.ident|library|specimen|gsm)$"),
    ("cell_type", r"(cell_?type|celltype|annotation|cell_?class|subclass|major_?type)"),
    ("cluster", r"(leiden|louvain|cluster|seurat_clusters)"),
    ("niche", r"(niche|domain|region|compartment|cellcharter|neighbou?rhood)"),
    ("group", r"(group|condition|treatment|disease|genotype|timepoint|stage|batch|experiment|sex|age|replicate)"),
    ("coordinate", r"^(x|y|z)(_centroid|_um|_location)?$|centroid|^(row|col)$"),
    ("qc", r"(count|n_genes|area|codeword|probe|qc_|mito|pct_|log1p)"),
]

SPATIAL_OBSM_KEYS = ("spatial", "X_spatial", "spatial_coords")


@dataclass
class ColumnInfo:
    name: str
    dtype: str
    n_unique: int
    role: str = "other"
    level: str = "cell"  # "cell" or "sample" (constant within every sample)
    description: str = ""
    top_values: list = field(default_factory=list)
    numeric_summary: dict = field(default_factory=dict)


@dataclass
class Schema:
    n_cells: int
    n_genes: int
    genes: list
    columns: dict  # name -> ColumnInfo
    roles: dict    # role -> primary column name
    spatial_key: str | None
    x_encoding: str
    graphs: dict
    structures: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["genes"] = list(self.genes)
        return d


def read_obs(path) -> pd.DataFrame:
    with h5py.File(path, "r") as f:
        return read_elem(f["obs"])


def read_spatial(path, obs: pd.DataFrame | None = None) -> tuple[np.ndarray, str]:
    with h5py.File(path, "r") as f:
        if "obsm" in f:
            for k in SPATIAL_OBSM_KEYS:
                if k in f["obsm"]:
                    xy = np.asarray(f["obsm"][k][:, :2], dtype=np.float32)
                    return xy, f"obsm/{k}"
    if obs is not None:
        for xc, yc in (("x_centroid", "y_centroid"), ("x", "y"), ("center_x", "center_y")):
            if xc in obs and yc in obs:
                return obs[[xc, yc]].to_numpy(np.float32), f"obs/{xc},{yc}"
    raise ValueError("no spatial coordinates found (obsm['spatial'] or x/y obs columns)")


def _uns_descriptions(f: h5py.File) -> tuple[dict, dict]:
    """Pick up free-text column/graph descriptions some pipelines store in uns."""
    cols, graphs = {}, {}
    if "uns" not in f:
        return cols, graphs
    for key, target in (("obs_columns", cols), ("graphs", graphs)):
        if key in f["uns"]:
            try:
                d = read_elem(f["uns"][key])
            except Exception:
                continue
            for k, v in d.items():
                # keys may be "a, b, c" sharing one description
                for name in re.split(r",\s*", str(k)):
                    target[name] = str(v)
    return cols, graphs


def guess_role(name: str) -> str:
    low = name.lower()
    for role, pat in ROLE_PATTERNS:
        if re.search(pat, low):
            return role
    return "other"


def describe_columns(obs: pd.DataFrame, sample_col: str | None, descriptions: dict) -> dict:
    out = {}
    for c in obs.columns:
        s = obs[c]
        info = ColumnInfo(name=c, dtype=str(s.dtype), n_unique=int(s.nunique(dropna=True)))
        info.role = guess_role(c)
        info.description = descriptions.get(c, "")
        is_cat = isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == bool or s.dtype == object
        if is_cat or info.n_unique <= 30:
            vc = s.astype("object").value_counts()
            info.top_values = [[str(k), int(v)] for k, v in vc.head(40).items()]
        elif np.issubdtype(s.dtype, np.number):
            q = np.nanquantile(s.to_numpy(float), [0, 0.05, 0.5, 0.95, 1])
            info.numeric_summary = dict(zip(["min", "p05", "median", "p95", "max"], map(float, q)))
        if sample_col and c != sample_col and info.n_unique <= 200:
            per_sample = obs.groupby(sample_col, observed=True)[c].nunique(dropna=False)
            if (per_sample <= 1).all():
                info.level = "sample"
        out[c] = info
    return out


def pick_primary(columns: dict, n_cells: int) -> dict:
    """Choose one column per role (the most plausible), leaving the rest as alternatives."""
    roles = {}
    by_role: dict[str, list[ColumnInfo]] = {}
    for info in columns.values():
        by_role.setdefault(info.role, []).append(info)

    samp = [c for c in by_role.get("sample", []) if 1 < c.n_unique < n_cells / 50]
    if samp:
        # prefer the most specific sample identifier
        roles["sample"] = max(samp, key=lambda c: (c.n_unique, "sample" in c.name.lower())).name

    ct = [c for c in by_role.get("cell_type", []) if 1 < c.n_unique <= 300]
    if ct:
        # shortest name tends to be the curated label ("cell_type" over "cell_type_cluster")
        roles["cell_type"] = min(ct, key=lambda c: (len(c.name), c.n_unique)).name
    elif by_role.get("cluster"):
        roles["cell_type"] = min(by_role["cluster"], key=lambda c: c.n_unique).name

    grp = [c for c in by_role.get("group", []) if c.level == "sample" and 1 < c.n_unique <= 50]
    if grp:
        pref = ["group", "condition", "treatment", "disease", "genotype"]
        grp.sort(key=lambda c: next((i for i, p in enumerate(pref) if p in c.name.lower()), 99))
        roles["group"] = grp[0].name
    return roles


def inspect_h5ad(path, overrides: dict | None = None) -> tuple[Schema, pd.DataFrame]:
    overrides = overrides or {}
    with h5py.File(path, "r") as f:
        genes = [g.decode() if isinstance(g, bytes) else str(g) for g in f["var"]["_index"][:]] \
            if "_index" in f["var"] else list(read_elem(f["var"]).index)
        X = f["X"]
        x_encoding = X.attrs.get("encoding-type", "array") if isinstance(X, h5py.Group) else "array"
        if isinstance(x_encoding, bytes):
            x_encoding = x_encoding.decode()
        col_desc, graph_desc = _uns_descriptions(f)
        graphs = {k: graph_desc.get(k, "") for k in (f["obsp"].keys() if "obsp" in f else [])}
        structures = {k: list(f[k].keys()) for k in ("obsm", "layers", "uns") if k in f}

    obs = read_obs(path)
    col_desc.update(overrides.get("descriptions", {}))

    sample_col = overrides.get("roles", {}).get("sample")
    if sample_col is None:
        cands = [c for c in obs.columns if guess_role(c) == "sample" and 1 < obs[c].nunique() < len(obs) / 50]
        sample_col = max(cands, key=lambda c: (obs[c].nunique(), "sample" in c.lower())) if cands else None

    columns = describe_columns(obs, sample_col, col_desc)
    roles = pick_primary(columns, len(obs))
    roles.update(overrides.get("roles", {}))
    for col, role in overrides.get("column_roles", {}).items():
        if col in columns:
            columns[col].role = role

    try:
        _, spatial_key = read_spatial(path, obs)
    except ValueError:
        spatial_key = None

    schema = Schema(
        n_cells=len(obs), n_genes=len(genes), genes=genes, columns=columns, roles=roles,
        spatial_key=spatial_key, x_encoding=x_encoding, graphs=graphs, structures=structures,
    )
    return schema, obs
