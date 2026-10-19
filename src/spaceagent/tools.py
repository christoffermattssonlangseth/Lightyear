"""Query functions behind the MCP tools. Each returns small markdown and, where useful, a figure path.

They are plain Python so they can also be used from a notebook:
    from spaceagent import load, tools
    s = load("stores/cuprizone"); print(tools.expression(s, ["Spp1"], by=["cell_type"]))
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from . import stats
from .ingest import _md_table
from .store import Store


def _sample_groups(s: Store, col: str) -> pd.Series:
    return s.obs.groupby(s.role("sample"), observed=True)[col].first().astype(str)


def card(s: Store, section: str | None = None) -> str:
    secs = json.loads((s.card_dir / "sections.json").read_text())
    if section is None:
        return (s.card_dir / "card.md").read_text()
    if section not in secs:
        return f"unknown section {section!r}; available: {list(secs)}"
    return secs[section]


def column_values(s: Store, column: str, by: str | None = None, max_values: int = 80) -> str:
    c = s.obs[column]
    info = s.columns.get(column, {})
    head = f"`{column}` role={info.get('role')} level={info.get('level')} dtype={c.dtype}"
    if info.get("description"):
        head += f"\n{info['description']}"
    if pd.api.types.is_numeric_dtype(c) and c.dtype != bool and c.nunique() > 30:
        grp = c.groupby(s.obs[by].astype(str), observed=True) if by else None
        desc = (grp.describe() if grp is not None else c.describe().to_frame().T)
        return head + "\n\n" + _md_table(desc)
    if by:
        tab = pd.crosstab(c.astype(str), s.obs[by].astype(str))
        return head + "\n\n" + _md_table(tab.head(max_values))
    vc = c.astype(str).value_counts().head(max_values).rename("n_cells").to_frame()
    return head + "\n\n" + _md_table(vc)


def list_genes(s: Store, pattern: str | None = None) -> str:
    import re
    genes = [g for g in s.genes if not pattern or re.search(pattern, g, re.I)]
    return f"{len(genes)} genes" + (f" matching {pattern!r}" if pattern else "") + ":\n" + ", ".join(genes)


def expression(s: Store, genes: list[str], by: list[str], filters: dict | None = None,
               per_sample: bool = False, min_cells: int = 10) -> str:
    """Mean log-normalised expression and % expressing per group."""
    genes = [s.resolve_gene(g) for g in genes]
    m = s.mask(filters)
    cols = ([s.role("sample")] if per_sample and s.role("sample") not in by else []) + list(by)
    mean, frac, n = stats.group_stats(s, cols, genes, mask=m, min_cells=min_cells)
    out = pd.DataFrame({"n_cells": n})
    for g in genes:
        out[f"{g} mean"] = mean[g].round(3)
        out[f"{g} %>0"] = (100 * frac[g]).round(1)
    note = f"{int(m.sum()):,} cells after filters {filters or {}}"
    gap = _panel_note(s, genes)
    return note + (f"\n{gap}" if gap else "") + "\n\n" + _md_table(out.reset_index().set_index(cols[0]), max_rows=200)


def _panel_note(s: Store, genes: list[str]) -> str:
    p = s.card_dir / "tables" / "panel_gaps.parquet"
    if not p.exists():
        return ""
    gaps = pd.read_parquet(p)
    hit = gaps[gaps.index.isin(genes)]
    return "\n".join(f"WARNING: {g} not measured in samples {r.missing_in}" for g, r in hit.iterrows())


def composition(s: Store, of: str, by: list[str], filters: dict | None = None,
                normalize: bool = True, summarize_by_group: str | None = None) -> str:
    """Fractions (or counts) of `of` categories within each `by` group.

    summarize_by_group: if given (e.g. 'group'), compute per sample then report mean ± sd per group.
    """
    m = s.mask(filters)
    if summarize_by_group:
        samp = s.role("sample")
        tab = stats.composition(s.obs, of, [samp], mask=m, normalize=normalize)
        sg = _sample_groups(s, summarize_by_group)
        tab["_g"] = sg.reindex(tab.index).to_numpy()
        mean = tab.groupby("_g").mean().T
        sd = tab.groupby("_g").std().T
        n = tab.groupby("_g").size()
        out = mean.round(4).astype(str) + " ± " + sd.round(4).astype(str)
        out.index.name = of
        return (f"mean ± sd over samples (n per group: {n.to_dict()}); filters {filters or {}}\n\n"
                + _md_table(out, max_rows=200))
    tab = stats.composition(s.obs, of, by, mask=m, normalize=normalize)
    return f"filters {filters or {}}\n\n" + _md_table(tab.round(4), max_rows=200)


def compare_groups(s: Store, a: str, b: str, column: str | None = None, cell_type: str | None = None,
                   genes: list[str] | None = None, filters: dict | None = None, n_top: int = 25) -> str:
    """Sample-level pseudobulk comparison between two values of a sample-level column."""
    column = column or s.role("group")
    samp = s.role("sample")
    f = dict(filters or {})
    if cell_type:
        f[s.role("cell_type")] = cell_type
    m = s.mask(f)
    mean, _, n = stats.group_stats(s, [samp], genes, mask=m, min_cells=20)
    res = stats.compare_samples(stats.mask_panel_gaps(s, mean), _sample_groups(s, column), str(a), str(b))
    if res.empty:
        return "no samples with ≥20 cells in one of the groups"
    gap = _panel_note(s, list(res.index[:n_top]))
    hdr = (f"{a} vs {b} on `{column}`; cells: {f or 'all'}; samples with ≥20 cells: "
           f"{res['n_a'].max()} vs {res['n_b'].max()}. delta = difference in mean log-normalised expression.")
    return hdr + (f"\n{gap}" if gap else "") + "\n\n" + _md_table(res.head(n_top).round(4))


def neighbourhood(s: Store, centre: str | None = None, group: str | None = None, n_top: int = 30) -> str:
    """Precomputed cell-type co-localisation (log2 observed/expected over kNN)."""
    df = pd.read_parquet(s.card_dir / "tables" / "nhood_by_group.parquet")
    if centre:
        df = df[df["centre"] == centre]
    if group:
        df = df[df["group"] == group]
    if centre and not group:
        piv = df.pivot_table(index="neighbour", columns="group", values="log2_oe")
        return f"neighbours of {centre}: log2 O/E by group\n\n" + _md_table(piv.round(3).sort_values(piv.columns[0], ascending=False))
    df = df.sort_values("log2_oe", ascending=False).head(n_top)
    return _md_table(df.set_index("group").round(3))


def distance(s: Store, source: dict, target: dict, by: str | None = None, max_um: float | None = None) -> str:
    """Distance from each source cell to its nearest target cell (same section), summarised per sample/group.

    source/target are filter dicts, e.g. {"cell_type": "Microglia"} and {"cell_type": "Oligodendrocyte"}.
    """
    samp = s.role("sample")
    sec = stats.codes_for(s.obs, [samp])[0]
    src, tgt = s.mask(source), s.mask(target)
    d = stats.distance_to(s.xy, sec, src, tgt)
    df = pd.DataFrame({"sample": s.obs[samp].astype(str).to_numpy()[src], "d": d[src]}).dropna()
    per = df.groupby("sample")["d"].agg(n="size", median="median", p25=lambda x: x.quantile(.25),
                                       p75=lambda x: x.quantile(.75))
    if max_um:
        per[f"frac_within_{max_um:g}um"] = df.groupby("sample")["d"].apply(lambda x: (x <= max_um).mean())
    out = f"distance (coordinate units) from {source} to nearest {target}\n\n" + _md_table(per.round(2))
    if by:
        per["_g"] = _sample_groups(s, by).reindex(per.index).to_numpy()
        out += "\n\nmean of per-sample values by group:\n\n" + _md_table(per.groupby("_g").mean(numeric_only=True).round(2))
    return out


def gene_spatial_association(s: Store, gene: str, near: dict, radius: float = 50.0,
                             cells: dict | None = None, by: str | None = None) -> str:
    """Expression of `gene` in `cells` that are within `radius` of any `near` cell vs further away."""
    samp = s.role("sample")
    sec = stats.codes_for(s.obs, [samp])[0]
    src, tgt = s.mask(cells), s.mask(near)
    d = stats.distance_to(s.xy, sec, src, tgt)
    v = s.gene(gene)
    df = pd.DataFrame({"sample": s.obs[samp].astype(str).to_numpy(), "near": d <= radius, "v": v})[src & ~np.isnan(d)]
    t = df.groupby(["sample", "near"])["v"].agg(["size", "mean", lambda x: (x > 0).mean()])
    t.columns = ["n", "mean", "frac>0"]
    t = t.unstack("near")
    out = f"{gene} in {cells or 'all cells'}: within {radius} of {near} (True) vs beyond (False)\n\n" + _md_table(t.round(3))
    if by:
        t2 = t.copy()
        t2["_g"] = _sample_groups(s, by).reindex(t2.index).to_numpy()
        out += "\n\nby group (mean over samples):\n\n" + _md_table(t2.groupby("_g").mean().round(3))
    return out


def add_finding(s: Store, title: str, text: str) -> str:
    p = s.findings_path()
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    with p.open("a") as f:
        f.write(f"\n## {title}\n_{stamp}_\n\n{text.strip()}\n")
    return f"saved to {p}"


def read_findings(s: Store) -> str:
    p = s.findings_path()
    return p.read_text() if p.exists() else "no findings recorded yet"
