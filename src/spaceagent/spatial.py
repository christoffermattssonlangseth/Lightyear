"""Spatial-neighbour analyses on the store's per-section kNN graph.

The graph (knn_idx.npy / knn_dist.npy, nearest first, self excluded) is built once at
ingest. Everything here returns per-sample / per-group aggregates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import stats
from .store import Store

K_STORED = 30


# -- graph -------------------------------------------------------------------
def ensure_knn(s: Store, k: int = K_STORED, log=print):
    if (s.path / "knn_idx.npy").exists():
        idx = np.load(s.path / "knn_idx.npy", mmap_mode="r")
        if idx.shape[1] >= k:
            return
    sections = stats.codes_for(s.obs, [s.role("sample")])[0]
    n = s.n_cells
    idx = np.lib.format.open_memmap(s.path / "knn_idx.npy", "w+", np.int32, (n, k))
    dist = np.lib.format.open_memmap(s.path / "knn_dist.npy", "w+", np.float16, (n, k))
    xy = s.xy
    for sec in np.unique(sections):
        cells = np.flatnonzero(sections == sec)
        kk = min(k, len(cells) - 1)
        d, nb = cKDTree(xy[cells]).query(xy[cells], k=kk + 1, workers=-1)
        nb, d = cells[nb[:, 1:]], d[:, 1:]
        if kk < k:  # tiny section: pad with self at infinite distance
            nb = np.c_[nb, np.repeat(cells[:, None], k - kk, 1)]
            d = np.c_[d, np.full((len(cells), k - kk), np.inf)]
        idx[cells], dist[cells] = nb, d
    idx.flush(); dist.flush()
    s._knn = None
    log(f"  kNN graph k={k} built")


def neighbours(s: Store, k: int = 15, max_dist: float | None = None):
    """(idx, valid) for the first k neighbours; valid=False beyond max_dist (e.g. across tissue gaps)."""
    idx, dist = s.knn
    if k > idx.shape[1]:
        raise ValueError(f"store has k={idx.shape[1]} neighbours; rebuild with ensure_knn(k={k})")
    idx = np.asarray(idx[:, :k])
    valid = np.ones(idx.shape, bool) if max_dist is None else np.asarray(dist[:, :k], np.float32) <= max_dist
    return idx, valid


def pair_counts(s: Store, column: str, k: int = 10) -> tuple[dict, list[str]]:
    """Per sample, counts of (centre category, neighbour category) over the kNN graph."""
    codes, cats = stats.codes_for(s.obs, [column])
    C = len(cats)
    idx, _ = neighbours(s, k)
    smp = s.obs[s.role("sample")].astype(str).to_numpy()
    out = {}
    for sm in sorted(set(smp)):
        cells = np.flatnonzero(smp == sm)
        flat = np.repeat(codes[cells], k) * C + codes[idx[cells]].ravel()
        out[sm] = np.bincount(flat, minlength=C * C).reshape(C, C)
    return out, [str(c) for c in cats]


def composition_matrix(s: Store, column: str, k: int = 15, max_dist: float | None = None,
                       cells: np.ndarray | None = None):
    """Fraction of each category of `column` among each cell's k neighbours -> (n_cells or len(cells), C)."""
    codes, cats = stats.codes_for(s.obs, [column])
    idx, valid = neighbours(s, k, max_dist)
    if cells is not None:
        idx, valid = idx[cells], valid[cells]
    C = len(cats)
    rows = np.broadcast_to(np.arange(len(idx))[:, None], idx.shape)[valid]
    flat = rows.astype(np.int64) * C + codes[idx[valid]]
    comp = np.bincount(flat, minlength=len(idx) * C).reshape(len(idx), C).astype(np.float32)
    comp /= np.maximum(comp.sum(1, keepdims=True), 1)
    return comp, [str(c) for c in cats]


def smoothed_expression(s: Store, genes: list[str] | None = None, k: int = 15, max_dist: float | None = None,
                        n_pcs: int = 20, seed: int = 0):
    """UTAG-style features: each cell's expression averaged with its neighbours, reduced by PCA."""
    from sklearn.decomposition import PCA
    genes = [s.resolve_gene(g) for g in (genes or s.genes)]
    idx, valid = neighbours(s, k, max_dist)
    w = valid.astype(np.float32)
    denom = 1 + w.sum(1)
    X = np.empty((s.n_cells, len(genes)), np.float32)
    for j, g in enumerate(genes):
        v = s.gene(g)
        X[:, j] = (v + (v[idx] * w).sum(1)) / denom
    X -= X.mean(0)
    X /= X.std(0) + 1e-6
    rng = np.random.default_rng(seed)
    fit = rng.choice(s.n_cells, min(s.n_cells, 200_000), replace=False)
    pca = PCA(n_components=min(n_pcs, len(genes) - 1), random_state=seed).fit(X[fit])
    return pca.transform(X).astype(np.float32)


# -- niches -----------------------------------------------------------------
def name_clusters(comp_mean: pd.DataFrame, overall: pd.Series, prefix: str) -> dict:
    """'N3: Microglia | Oligodendrocyte' from the two most enriched (and abundant) categories."""
    names = {}
    for cl, row in comp_mean.iterrows():
        score = row * np.log2((row + 1e-3) / (overall + 1e-3)).clip(lower=0)
        top = score.sort_values(ascending=False).index[:2]
        names[cl] = f"{prefix}{cl}: " + " | ".join(top)
    return names


def compute_niches(s: Store, name: str, n_niches: int = 8, features: str = "composition", k: int = 15,
                   max_dist: float | None = None, column: str | None = None, seed: int = 0) -> dict:
    """Cluster cells by their neighbourhood and save the labels as a new column `name`.

    features="composition": cell-type fractions among the k neighbours (classic niche / CN analysis).
    features="expression":  neighbour-smoothed expression PCs (UTAG-like; no annotation needed).
    """
    from sklearn.cluster import MiniBatchKMeans
    column = column or s.role("cell_type")
    if features == "composition":
        X, _ = composition_matrix(s, column, k, max_dist)
    elif features == "expression":
        X = smoothed_expression(s, k=k, max_dist=max_dist, seed=seed)
    else:
        raise ValueError("features must be 'composition' or 'expression'")
    km = MiniBatchKMeans(n_clusters=n_niches, random_state=seed, batch_size=8192, n_init=5).fit(X)
    lab = km.labels_
    # niche label = what the cells in it are (own type), which reads better than the neighbour mix
    own = pd.crosstab(lab, s.obs[column].astype(str).to_numpy(), normalize="index")
    overall = s.obs[column].astype(str).value_counts(normalize=True)
    names = name_clusters(own, overall.reindex(own.columns).fillna(0), "N")
    labels = pd.Categorical([names[l] for l in lab], categories=[names[i] for i in sorted(names)])
    desc = (f"compute_niches: MiniBatchKMeans k={n_niches} on {features} of {k} spatial neighbours"
            f"{f' within {max_dist}' if max_dist else ''} (cell types from `{column}`)")
    s.add_column(name, labels, desc, role="niche")
    return {"column": name, "description": desc,
            "composition": own.rename(index=names), "n_cells": pd.Series(lab).map(names).value_counts()}


# -- neighbourhood statistics -------------------------------------------------
def neighbourhood_of(s: Store, cells: dict, k: int = 15, column: str | None = None, max_dist: float | None = None,
                     by: str | None = None) -> pd.DataFrame:
    """Mean neighbour composition around the selected cells, per sample, vs the section background."""
    column = column or s.role("cell_type")
    samp = s.role("sample")
    m = s.mask(cells)
    comp, cats = composition_matrix(s, column, k, max_dist, cells=np.flatnonzero(m))
    smp = s.obs[samp].astype(str).to_numpy()
    per = pd.DataFrame(comp, columns=cats).groupby(smp[m]).mean()
    bg = stats.composition(s.obs, column, [samp]).reindex(index=per.index, columns=cats).fillna(0)
    log2e = np.log2((per + 1e-3) / (bg + 1e-3))
    out = pd.concat({"frac": per, "log2_vs_section": log2e}, axis=1)
    if by:
        g = s.obs.groupby(samp, observed=True)[by].first().astype(str).reindex(per.index)
        out = out.groupby(g.to_numpy()).mean()
    return out


def nhood_enrichment(s: Store, samples: list[str] | None = None, k: int = 10, n_perm: int = 50,
                     column: str | None = None, seed: int = 0) -> pd.DataFrame:
    """squidpy-style neighbourhood enrichment z-scores: observed pair counts vs label permutations
    within each section. Returns long table (sample, centre, neighbour, observed, z)."""
    column = column or s.role("cell_type")
    samp = s.role("sample")
    codes, cats = stats.codes_for(s.obs, [column])
    C = len(cats)
    idx, _ = neighbours(s, k)
    smp = s.obs[samp].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    rows = []
    for sm in samples or sorted(set(smp)):
        cells = np.flatnonzero(smp == sm)
        local = np.full(s.n_cells, -1, np.int64)
        local[cells] = np.arange(len(cells))
        nb = local[idx[cells]]  # neighbours stay within the section
        lab = codes[cells]

        def count(l):
            return np.bincount(np.repeat(l, k) * C + l[nb].ravel(), minlength=C * C).reshape(C, C)

        obs = count(lab)
        perm = np.stack([count(rng.permutation(lab)) for _ in range(n_perm)])
        z = (obs - perm.mean(0)) / (perm.std(0) + 1e-9)
        for i in range(C):
            for j in range(C):
                rows.append((sm, str(cats[i]), str(cats[j]), int(obs[i, j]), float(z[i, j])))
    return pd.DataFrame(rows, columns=["sample", "centre", "neighbour", "observed", "z"])


def co_occurrence(s: Store, a: dict, b: dict, distances: list[float] | None = None,
                  samples: list[str] | None = None, max_cells: int = 20_000, seed: int = 0) -> pd.DataFrame:
    """P(b within distance d of an a cell) / P(b in the section) for a range of distances (ratio > 1 =
    co-occurrence). Per sample."""
    distances = distances or [10, 20, 30, 50, 75, 100, 150, 200, 300]
    samp = s.role("sample")
    smp = s.obs[samp].astype(str).to_numpy()
    ma, mb = s.mask(a), s.mask(b)
    rng = np.random.default_rng(seed)
    rows = []
    for sm in samples or sorted(set(smp[ma])):
        sec = smp == sm
        ia = np.flatnonzero(sec & ma)
        if len(ia) == 0:
            continue
        ia = rng.choice(ia, min(len(ia), max_cells), replace=False)
        tree_all = cKDTree(s.xy[sec])
        tree_b = cKDTree(s.xy[sec & mb]) if (sec & mb).any() else None
        p_b = (sec & mb).sum() / sec.sum()
        for d in distances:
            n_all = tree_all.query_ball_point(s.xy[ia], d, workers=-1, return_length=True) - 1
            n_b = tree_b.query_ball_point(s.xy[ia], d, workers=-1, return_length=True) if tree_b else 0
            n_b = n_b - mb[ia]  # don't count the centre cell itself
            ratio = (n_b.sum() / max(n_all.sum(), 1)) / max(p_b, 1e-9)
            rows.append((sm, d, float(ratio)))
    return pd.DataFrame(rows, columns=["sample", "distance", "ratio"])


def morans_i(s: Store, genes: list[str] | None = None, samples: list[str] | None = None, k: int = 10,
             max_cells: int | None = None) -> pd.DataFrame:
    """Moran's I per gene per section on the kNN graph (row-standardised). High = spatially patterned."""
    genes = [s.resolve_gene(g) for g in (genes or s.genes)]
    samp = s.role("sample")
    smp = s.obs[samp].astype(str).to_numpy()
    idx, _ = neighbours(s, k)
    out = {}
    for sm in samples or sorted(set(smp)):
        cells = np.flatnonzero(smp == sm)
        nb = idx[cells]
        res = {}
        for g in genes:
            v = s.gene(g)
            z = v - v[cells].mean()
            den = (z[cells] ** 2).sum()
            res[g] = float((z[cells] * z[nb].mean(1)).sum() / den) if den > 0 else np.nan
        out[sm] = res
    return pd.DataFrame(out)


def gradient(s: Store, feature: str, target: dict, cells: dict | None = None,
             bins: list[float] | None = None, by: str | None = None) -> pd.DataFrame:
    """Mean of a gene/score in `cells` as a function of distance to the nearest `target` cell."""
    bins = bins or [0, 25, 50, 100, 200, 400, 800, np.inf]
    samp = s.role("sample")
    sec = stats.codes_for(s.obs, [samp])[0]
    src, tgt = s.mask(cells), s.mask(target)
    d = stats.distance_to(s.xy, sec, src & ~tgt, tgt)
    v = s.gene(feature)
    keep = src & ~tgt & ~np.isnan(d)
    df = pd.DataFrame({"sample": s.obs[samp].astype(str).to_numpy()[keep],
                       "bin": pd.cut(d[keep], bins, right=False).astype(str), "v": v[keep]})
    t = df.groupby(["sample", "bin"])["v"].agg(["size", "mean"]).reset_index()
    order = [str(x) for x in pd.cut([], bins, right=False).categories]
    t["bin"] = pd.Categorical(t["bin"], categories=order, ordered=True)
    t = t[t["size"] >= 10]
    if by:
        t["group"] = s.obs.groupby(samp, observed=True)[by].first().astype(str).reindex(t["sample"]).to_numpy()
        return t.pivot_table(index="bin", columns="group", values="mean", aggfunc="mean", observed=False)
    return t.pivot_table(index="bin", columns="sample", values="mean", observed=False)
