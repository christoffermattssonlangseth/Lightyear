"""Figures returned to the agent as PNG files under <store>/figures/."""
from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import stats  # noqa: E402
from .store import Store  # noqa: E402

PALETTE = plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors


def _out(s: Store, stem: str):
    d = s.path / "figures"
    d.mkdir(exist_ok=True)
    return d / (re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)[:120] + ".png")


def spatial(s: Store, sample: str, color: str, filters: dict | None = None,
            bbox: list[float] | None = None, highlight: list[str] | None = None, point_size: float | None = None):
    """Scatter one section coloured by a gene or an obs column.

    highlight: for a categorical colour, draw only these categories in colour (rest grey).
    bbox: [xmin, xmax, ymin, ymax] to zoom.
    """
    samp = s.role("sample")
    m = s.mask({samp: sample})
    xy = s.xy
    if bbox:
        x0, x1, y0, y1 = bbox
        m &= (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    sel = s.mask(filters) & m if filters else m
    fig, ax = plt.subplots(figsize=(8, 8), dpi=110)
    ps = point_size or max(0.2, min(6.0, 4e4 / max(m.sum(), 1)))
    ax.scatter(xy[m & ~sel, 0], xy[m & ~sel, 1], s=ps, c="#dddddd", linewidths=0, rasterized=True)
    if color in s.obs.columns and not pd.api.types.is_float_dtype(s.obs[color]):
        lab = s.obs[color].astype(str).to_numpy()
        cats = sorted(pd.unique(lab[sel]))
        shown = [c for c in cats if not highlight or c in highlight]
        if highlight:
            rest = sel & ~np.isin(lab, shown)
            ax.scatter(xy[rest, 0], xy[rest, 1], s=ps, c="#cccccc", linewidths=0, rasterized=True)
        for i, c in enumerate(shown):
            k = sel & (lab == c)
            ax.scatter(xy[k, 0], xy[k, 1], s=ps, color=PALETTE[i % len(PALETTE)], label=c, linewidths=0, rasterized=True)
        ax.legend(markerscale=max(1, 8 / ps), fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    else:
        v = s.obs[color].to_numpy(float) if color in s.obs.columns else s.gene(color)
        idx = np.flatnonzero(sel)
        idx = idx[np.argsort(v[idx])]  # high values on top
        vmax = np.nanquantile(v[idx], 0.99) if len(idx) else 1
        sc = ax.scatter(xy[idx, 0], xy[idx, 1], s=ps, c=v[idx], cmap="magma_r" if color not in s.obs else "viridis",
                        vmin=0 if color not in s.obs else None, vmax=vmax or None, linewidths=0, rasterized=True)
        fig.colorbar(sc, ax=ax, shrink=0.6, label=color)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(f"{sample}: {color}" + (f" | {filters}" if filters else ""), fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    p = _out(s, f"spatial_{sample}_{color}_{filters or ''}_{bbox or ''}")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def dotplot(s: Store, genes: list[str], by: str, filters: dict | None = None):
    genes = [s.resolve_gene(g) for g in genes]
    mean, frac, _ = stats.group_stats(s, [by], genes, mask=s.mask(filters), min_cells=10)
    fig, ax = plt.subplots(figsize=(0.5 * len(genes) + 3, 0.35 * len(mean) + 1.5), dpi=110)
    yy, xx = np.meshgrid(np.arange(len(mean)), np.arange(len(genes)), indexing="ij")
    z = mean.to_numpy() / np.maximum(mean.to_numpy().max(axis=0, keepdims=True), 1e-9)
    sc = ax.scatter(xx.ravel(), yy.ravel(), s=250 * frac.to_numpy().ravel(), c=z.ravel(), cmap="Reds",
                    vmin=0, vmax=1, edgecolors="#999999", linewidths=0.3)
    ax.set_xticks(range(len(genes)), genes, rotation=90, fontstyle="italic")
    ax.set_yticks(range(len(mean)), [str(i) for i in mean.index])
    ax.set_xlim(-0.6, len(genes) - 0.4); ax.set_ylim(len(mean) - 0.4, -0.6)
    fig.colorbar(sc, ax=ax, shrink=0.5, label="mean (scaled per gene)")
    ax.set_title(f"size = fraction >0; {filters or ''}", fontsize=8)
    p = _out(s, f"dot_{'-'.join(genes)}_{by}_{filters or ''}")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def sample_bars(s: Store, gene: str, group: str, cell_type: str | None = None):
    """Per-sample pseudobulk of a gene, grouped - the honest picture for few-replicate contrasts."""
    samp = s.role("sample")
    f = {s.role("cell_type"): cell_type} if cell_type else None
    mean, _, n = stats.group_stats(s, [samp], [gene], mask=s.mask(f), min_cells=20)
    sg = s.obs.groupby(samp, observed=True)[group].first().astype(str)
    df = pd.DataFrame({"v": mean.iloc[:, 0], "g": sg.reindex(mean.index)}).sort_values("g")
    fig, ax = plt.subplots(figsize=(max(4, 0.35 * len(df) + 2), 3.2), dpi=110)
    groups = list(dict.fromkeys(df["g"]))
    ax.bar(range(len(df)), df["v"], color=[PALETTE[groups.index(g) % 20] for g in df["g"]])
    ax.set_xticks(range(len(df)), df.index, rotation=90, fontsize=7)
    ax.set_ylabel(f"mean {gene}")
    ax.set_title(f"{gene} per sample{' in ' + cell_type if cell_type else ''}", fontsize=9)
    for i, g in enumerate(groups):
        ax.bar(0, 0, color=PALETTE[i % 20], label=g)
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
    p = _out(s, f"bars_{gene}_{group}_{cell_type or ''}")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p
