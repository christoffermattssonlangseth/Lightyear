"""Aggregations shared by the ingest step and the query tools.

Everything reduces cells to groups (cell type, sample, niche, ...) before it is
returned, so tool outputs stay small and never expose cell-level values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.spatial import cKDTree

from .store import Store


def codes_for(obs: pd.DataFrame, cols: list[str], mask: np.ndarray | None = None):
    """Integer group code per cell for the combination of `cols` (-1 = masked out)."""
    per_col = [pd.factorize(obs[c].astype(str), sort=True) for c in cols]
    combined = np.zeros(len(obs), dtype=np.int64)
    for codes_c, uniq_c in per_col:
        combined = combined * len(uniq_c) + codes_c
    codes, uniq = pd.factorize(combined, sort=True)
    parts = []
    for codes_c, uniq_c in reversed(per_col):
        parts.append(np.asarray(uniq_c)[uniq % len(uniq_c)])
        uniq = uniq // len(uniq_c)
    parts = parts[::-1]
    if len(cols) == 1:
        index = pd.Index(parts[0], name=cols[0])
    else:
        index = pd.MultiIndex.from_arrays(parts, names=cols)
    codes = codes.astype(np.int64)
    if mask is not None:
        codes = np.where(mask, codes, -1)
    return codes, index


def group_stats(store: Store, cols: list[str], genes: list[str] | None = None,
                mask: np.ndarray | None = None, min_cells: int = 1):
    """Mean expression and fraction expressing (>0) per group, for each gene.

    Returns (mean, frac, n_cells): mean/frac are DataFrames groups x genes.
    """
    codes, index = codes_for(store.obs, cols, mask)
    k = len(index)
    keep = codes >= 0
    c = codes[keep]
    n = np.bincount(c, minlength=k).astype(float)
    genes = [store.resolve_gene(g) for g in (genes or store.genes)]
    mean = np.empty((k, len(genes)), dtype=np.float32)
    frac = np.empty((k, len(genes)), dtype=np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        for j, g in enumerate(genes):
            v = store.gene(g)[keep]
            mean[:, j] = np.bincount(c, weights=v, minlength=k) / n
            frac[:, j] = np.bincount(c, weights=(v > 0), minlength=k) / n
    ncell = pd.Series(n.astype(int), index=index, name="n_cells")
    ok = ncell.to_numpy() >= min_cells
    mean = pd.DataFrame(mean, index=index, columns=genes)[ok]
    frac = pd.DataFrame(frac, index=index, columns=genes)[ok]
    return mean, frac, ncell[ok]


def composition(obs: pd.DataFrame, of: str, by: list[str], mask: np.ndarray | None = None,
                normalize: bool = True) -> pd.DataFrame:
    """Counts (or fractions within each `by` group) of the categories of `of`."""
    df = obs if mask is None else obs[mask]
    tab = pd.crosstab([df[b].astype(str) for b in by], df[of].astype(str))
    if normalize:
        tab = tab.div(tab.sum(axis=1), axis=0)
    return tab


def markers(mean: pd.DataFrame, frac: pd.DataFrame, n_top: int = 10, min_frac: float = 0.1) -> dict:
    """One-vs-rest markers from group means (log-normalised data -> difference is a log fold)."""
    out = {}
    for grp in mean.index:
        rest = mean.drop(index=grp).mean(axis=0)
        rest_frac = frac.drop(index=grp).mean(axis=0)
        df = pd.DataFrame({
            "mean": mean.loc[grp], "frac": frac.loc[grp],
            "lfc_vs_rest": mean.loc[grp] - rest, "frac_rest": rest_frac,
        })
        df = df[df["frac"] >= min_frac].sort_values("lfc_vs_rest", ascending=False).head(n_top)
        out[str(grp)] = df.round(3)
    return out


def mask_panel_gaps(s: Store, pb: pd.DataFrame) -> pd.DataFrame:
    """Set pseudobulk values to NaN where a gene was not on that sample's panel."""
    p = s.card_dir / "tables" / "panel_gaps.parquet"
    if not p.exists():
        return pb
    pb = pb.copy()
    for g, r in pd.read_parquet(p).iterrows():
        if g in pb.columns:
            pb.loc[pb.index.intersection(r.missing_in.split(";")), g] = np.nan
    return pb


def compare_samples(pb: pd.DataFrame, sample_groups: pd.Series, a: str, b: str,
                    min_mean: float = 0.05) -> pd.DataFrame:
    """Sample-level comparison of pseudobulk means (rows = samples, cols = genes).

    Welch t-test across samples (cells are not independent replicates).
    """
    ga = pb.loc[sample_groups[sample_groups == a].index.intersection(pb.index)]
    gb = pb.loc[sample_groups[sample_groups == b].index.intersection(pb.index)]
    if len(ga) < 1 or len(gb) < 1:
        return pd.DataFrame()
    res = pd.DataFrame({
        f"mean_{a}": ga.mean(), f"mean_{b}": gb.mean(),
    })
    res["delta"] = res[f"mean_{a}"] - res[f"mean_{b}"]
    if len(ga) >= 2 and len(gb) >= 2:
        with np.errstate(all="ignore"):
            t, p = sps.ttest_ind(ga, gb, equal_var=False, axis=0, nan_policy="omit")
        res["t"] = t
        res["p"] = p
    res["n_a"], res["n_b"] = ga.notna().sum(), gb.notna().sum()
    res = res[(res[f"mean_{a}"] >= min_mean) | (res[f"mean_{b}"] >= min_mean)]
    return res.sort_values("delta", key=np.abs, ascending=False)


def neighbor_counts(xy: np.ndarray, labels: np.ndarray, n_labels: int, sections: np.ndarray,
                    k: int = 10, mask: np.ndarray | None = None) -> dict:
    """Per-section counts of (centre label, neighbour label) pairs over the kNN graph.

    Returns {section: (n_labels x n_labels) counts}. Neighbours never cross sections.
    """
    out = {}
    for sec in np.unique(sections):
        idx = np.flatnonzero(sections == sec)
        if mask is not None:
            idx = idx[mask[idx]]
        if len(idx) <= k:
            continue
        tree = cKDTree(xy[idx])
        _, nb = tree.query(xy[idx], k=k + 1, workers=-1)
        centre = np.repeat(labels[idx], k)
        neigh = labels[idx][nb[:, 1:]].ravel()
        out[sec] = np.bincount(centre * n_labels + neigh, minlength=n_labels ** 2).reshape(n_labels, n_labels)
    return out


def enrichment_log2oe(counts: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """log2(observed / expected) where expected assumes neighbours drawn from the section's labels."""
    tot = counts.sum()
    row = counts.sum(axis=1, keepdims=True)
    col = counts.sum(axis=0, keepdims=True)
    exp = row * col / max(tot, 1)
    return np.log2((counts + eps) / (exp + eps))


def distance_to(xy: np.ndarray, sections: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """For each source cell, distance to the nearest target cell in the same section (nan if none)."""
    d = np.full(len(xy), np.nan, dtype=np.float32)
    for sec in np.unique(sections[source]):
        s = np.flatnonzero(source & (sections == sec))
        t = np.flatnonzero(target & (sections == sec))
        if len(t) == 0:
            continue
        dist, _ = cKDTree(xy[t]).query(xy[s], k=1, workers=-1)
        d[s] = dist
    return d
