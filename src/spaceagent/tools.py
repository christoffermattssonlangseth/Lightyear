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
    return hdr + (f"\n{gap}" if gap else "") + "\n\n" + _md_table(res.head(n_top).round({c: 4 for c in res.columns if c != "p"}))


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


# -- annotation / derived columns ------------------------------------------------
def describe_structures(s: Store) -> str:
    """What the dataset contains beyond obs: source obsm/obsp/layers/uns, store graph, derived columns."""
    sch = s.meta["schema"]
    lines = [f"source: {s.meta['source']}", f"expression: {s.meta['expr_layer']} {s.meta.get('expr_description', '')}",
             f"coordinates: {sch['spatial_key']}"]
    for k in ("obsm", "layers", "uns"):
        if sch.get("structures", {}).get(k):
            lines.append(f"{k}: " + ", ".join(sch["structures"][k]))
    for g, d in (sch.get("graphs") or {}).items():
        lines.append(f"source obsp graph `{g}`: {d}")
    if (s.path / "knn_idx.npy").exists():
        lines.append(f"store spatial kNN graph: k={s.knn[0].shape[1]} per section (used by all neighbour tools)")
    der = s.meta.get("derived", {})
    if der:
        lines.append("\nderived columns (created by tools, usable everywhere):")
        lines += [f"- `{n}` ({d['role']}): {d['description']}" for n, d in der.items()]
    return "\n".join(lines)


def markers(s: Store, column: str, filters: dict | None = None, n_top: int = 10) -> str:
    from .analysis import markers_for
    lines = [f"one-vs-rest markers of `{column}` (lfc = mean log-expr difference vs other categories)"
             + (f", cells {filters}" if filters else "")]
    for cat, (n, df) in markers_for(s, column, filters, n_top).items():
        lines.append(f"- **{cat}** (n={n:,}): " +
                     ", ".join(f"{g} ({r.lfc_vs_rest:.2f}, {100 * r.frac:.0f}%)" for g, r in df.iterrows()))
    return "\n".join(lines)


def annotate(s: Store, column: str, filters: dict | None = None, tissue: str | None = None) -> str:
    from .analysis import annotate_suggest
    df = annotate_suggest(s, column, filters, tissue=tissue)
    return ("Marker-database matches (PanglaoDB + CellMarker, hypergeometric on panel genes). Hints only: "
            "judge with tissue context and marker specificity.\n\n" + _md_table(df, max_rows=100))


def gate(s: Store, name: str, feature: str, threshold: float, within: dict | None = None, above: bool = True) -> str:
    from .analysis import gate as _gate
    r = _gate(s, name, feature, threshold, within, above)
    return (f"created `{name}`: {r['description']}; {r['n_true']:,} of {r['n_within']:,} cells True. "
            f"Use it in filters ({{'{name}': True}}), composition(of='{name}') or case_control.")


def score_genes(s: Store, name: str, genes: list[str]) -> str:
    from .analysis import score_genes as _score
    r = _score(s, name, genes)
    return f"created numeric column `{name}`: {r['description']}. Use it like a gene in expression/compare/plots."


def compute_niches(s: Store, name: str, n_niches: int = 8, features: str = "composition", k: int = 15,
                   max_dist: float | None = None) -> str:
    from .spatial import compute_niches as _cn
    r = _cn(s, name, n_niches, features, k, max_dist)
    comp = r["composition"]
    lines = [f"created `{name}`: {r['description']}", "", "own cell-type composition of each niche (top 4):"]
    for niche, row in comp.iterrows():
        top = row.sort_values(ascending=False).head(4)
        lines.append(f"- **{niche}** ({int(r['n_cells'].get(niche, 0)):,} cells): " +
                     ", ".join(f"{c} {100 * v:.0f}%" for c, v in top.items()))
    if s.role("group"):
        lines += ["", composition(s, name, [s.role("sample")], summarize_by_group=s.role("group"))]
    return "\n".join(lines)


# -- spatial ------------------------------------------------------------------
def neighbourhood_of(s: Store, cells: dict, k: int = 15, column: str | None = None, by: str | None = None) -> str:
    from .spatial import neighbourhood_of as _no
    df = _no(s, cells, k, column, by=by)
    frac = df["frac"].T.round(3)
    enr = df["log2_vs_section"].T.round(2)
    return (f"neighbours (k={k}) of cells {cells}: mean fraction of each type, and log2 enrichment vs the "
            "section's overall composition\n\nfraction:\n\n" + _md_table(frac, max_rows=100)
            + "\n\nlog2 enrichment:\n\n" + _md_table(enr, max_rows=100))


def nhood_enrichment(s: Store, samples: list[str] | None = None, centre: str | None = None, k: int = 10,
                     n_perm: int = 50, by: str | None = None) -> str:
    from .spatial import nhood_enrichment as _ne
    df = _ne(s, samples, k, n_perm)
    if centre:
        df = df[df["centre"] == centre]
    if by:
        df["grp"] = _sample_groups(s, by).reindex(df["sample"]).to_numpy()
        piv = df.pivot_table(index=["centre", "neighbour"], columns="grp", values="z").round(1)
    else:
        piv = df.pivot_table(index=["centre", "neighbour"], columns="sample", values="z").round(1)
    piv = piv.reindex(piv.abs().max(axis=1).sort_values(ascending=False).index)
    return (f"neighbourhood enrichment z-scores (k={k}, {n_perm} label permutations within each section; "
            "|z|>2 ≈ beyond chance)\n\n" + _md_table(piv.reset_index().set_index("centre"), max_rows=60))


def co_occurrence(s: Store, a: dict, b: dict, distances: list[float] | None = None, by: str | None = None) -> str:
    from .spatial import co_occurrence as _co
    df = _co(s, a, b, distances)
    if by:
        df["grp"] = _sample_groups(s, by).reindex(df["sample"]).to_numpy()
        piv = df.pivot_table(index="distance", columns="grp", values="ratio")
    else:
        piv = df.pivot_table(index="distance", columns="sample", values="ratio")
    return (f"co-occurrence of {b} around {a}: P(b within d)/P(b in section); >1 = enriched\n\n"
            + _md_table(piv.round(2)))


def spatial_autocorrelation(s: Store, genes: list[str] | None = None, samples: list[str] | None = None,
                            n_top: int = 30) -> str:
    from .spatial import morans_i
    if samples is None:
        smp = s.obs[s.role("sample")].astype(str)
        samples = list(smp.drop_duplicates().iloc[:3])
    df = morans_i(s, genes, samples)
    df["mean"] = df.mean(1)
    df = df.sort_values("mean", ascending=False)
    return (f"Moran's I on the 10-NN graph in samples {samples} (high = spatially patterned, ~0 = random)\n\n"
            + _md_table(df.head(n_top).round(3)))


def gradient(s: Store, feature: str, target: dict, cells: dict | None = None, bins: list[float] | None = None,
             by: str | None = None) -> str:
    from .spatial import gradient as _gr
    df = _gr(s, feature, target, cells, bins, by)
    return (f"mean {feature} in {cells or 'all cells'} by distance (coordinate units) to nearest {target}"
            + (f", averaged over samples per `{by}`" if by else "") + "\n\n" + _md_table(df.round(3)))


# -- case / control ---------------------------------------------------------------
def case_control(s: Store, a: str, b: str, column: str | None = None, n_top: int = 15) -> str:
    from .analysis import case_control as _cc
    df = _cc(s, a, b, column)
    out_dir = s.path / "analyses"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"case_control_{a}_vs_{b}.parquet"
    df.to_parquet(path)
    n_a, n_b = int(df["n_a"].iat[0]), int(df["n_b"].iat[0])
    shown = ["feature", "source", "mean_a", "mean_b", "effect", "effect_unit", "hedges_g", "p"]
    lines = [f"# {a} (n={n_a} samples) vs {b} (n={n_b})",
             "Every feature is a per-sample value; Hedges g = standardised difference across samples. "
             f"Full table: {path}", ""]
    meaningful = df[((df.effect_unit == "log2 ratio") & (df.effect.abs() >= 0.5)) |
                    ((df.effect_unit == "delta") & (df.effect.abs() >= 0.15))]
    pan = meaningful[meaningful.pan_celltype]
    if len(pan):
        g = pan["feature"].str.split(": ", n=1).str[1]
        summ = pan.assign(gene=g).groupby("gene").agg(n_celltypes=("source", "nunique"),
                                                       mean_effect=("effect", "mean"))
        lines += ["## ⚠ Pan-cell-type shifts (likely spillover / ambient, excluded from rankings below)", "",
                  "These genes move the same way in most cell types. Find the cell type where they truly change "
                  "(largest effect, highest expression) and check with gene_near before calling them regulated.", "",
                  _md_table(summ.sort_values("n_celltypes", ascending=False).round(3)), ""]
        meaningful = meaningful[~meaningful.pan_celltype]
    lines += ["## Strongest separating features overall", "", _md_table(meaningful[shown].head(n_top).set_index("feature"))]
    for kind in ("composition", "niche", "neighbourhood", "expression"):
        sub = meaningful[meaningful.kind == kind]
        if len(sub):
            lines += ["", f"## {kind}", "", _md_table(sub[shown].head(n_top).set_index("feature"))]
    return "\n".join(lines)


def separability(s: Store, a: str, b: str, column: str | None = None, cells: dict | None = None,
                 features: str = "expression") -> str:
    from .analysis import separability as _sep
    r = _sep(s, a, b, column, cells, features)
    lines = [f"leave-one-sample-out logistic regression, {a} vs {b}, cells {cells or 'all'}, features={features}",
             f"held-out AUC = {r['auc']:.3f} on {r['n_cells']:,} cells (0.5 = indistinguishable)", "",
             "mean predicted P(a) per held-out sample:", "", _md_table(r["per_sample"].set_index("sample").round(3)), "",
             f"features pushing towards {a}: " + ", ".join(f"{k} ({v:+.2f})" for k, v in r[f'higher_in_{a}'].items()),
             f"features pushing towards {b}: " + ", ".join(f"{k} ({v:+.2f})" for k, v in r[f'higher_in_{b}'].items())]
    return "\n".join(lines)


def ligand_receptor(s: Store, sender: str | None = None, receiver: str | None = None, a: str | None = None,
                    b: str | None = None, column: str | None = None, n_top: int = 30) -> str:
    from .analysis import ligand_receptor as _lr
    df = _lr(s, sender, receiver, a, b, column, n_top=n_top)
    if df.empty:
        return "no ligand-receptor pairs with both partners on the panel and expressed"
    hdr = ("OmniPath ligand-receptor pairs on the panel. score = mean ligand (sender) x mean receptor (receiver). "
           "colocalisation_log2_oe = how much the sender and receiver types neighbour each other (>0 = adjacent). "
           "Expression-based inference: a hypothesis, not evidence of signalling.")
    return hdr + "\n\n" + _md_table(df.round(4), max_rows=n_top)
