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
    """Scatter coloured by a gene or an obs column. `sample` is a sample (all its sections/pieces drawn
    side by side, shared colours) or a single section.

    highlight: for a categorical colour, draw only these categories in colour (rest grey).
    bbox: [xmin, xmax, ymin, ymax] to zoom (coordinates of a single section).
    """
    samp, secc = s.role("sample"), s.role("section")
    if secc != samp and str(sample) in set(s.strcol(secc)):
        secs = [str(sample)]
        base = s.mask({secc: sample})
    else:
        base = s.mask({samp: sample})
        if not base.any():
            raise KeyError(f"{sample!r} is not a value of {samp!r}" + (f" or {secc!r}" if secc != samp else ""))
        secs = sorted(set(s.strcol(secc)[base]))[:6]
    xy = s.xy
    if bbox:
        x0, x1, y0, y1 = bbox
        base &= (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    sel_all = s.mask(filters) & base if filters else base
    categorical = color in s.obs.columns and not pd.api.types.is_float_dtype(s.obs[color])
    if categorical:
        lab = s.obs[color].astype(str).to_numpy()
        cats = sorted(pd.unique(lab[sel_all]))
        shown = [c for c in cats if not highlight or c in highlight]
    else:
        v = s.obs[color].to_numpy(float) if color in s.obs.columns else s.gene(color)
        vmax = np.nanquantile(v[sel_all], 0.99) if sel_all.any() else 1
    n = len(secs)
    fig, axes = plt.subplots(1, n, figsize=(min(8 * n, 24), 8 if n == 1 else 24 / n), dpi=110, squeeze=False)
    sec_arr = s.strcol(secc)
    for ax, se in zip(axes[0], secs):
        m = base & (sec_arr == se)
        sel = sel_all & m
        ps = point_size or max(0.2, min(6.0, 4e4 / max(m.sum(), 1)))
        ax.scatter(xy[m & ~sel, 0], xy[m & ~sel, 1], s=ps, c="#dddddd", linewidths=0, rasterized=True)
        if categorical:
            if highlight:
                rest = sel & ~np.isin(lab, shown)
                ax.scatter(xy[rest, 0], xy[rest, 1], s=ps, c="#cccccc", linewidths=0, rasterized=True)
            for i, c in enumerate(shown):
                k = sel & (lab == c)
                ax.scatter(xy[k, 0], xy[k, 1], s=ps, color=PALETTE[i % len(PALETTE)], label=c, linewidths=0,
                           rasterized=True)
        else:
            idx = np.flatnonzero(sel)
            idx = idx[np.argsort(v[idx])]  # high values on top
            sc = ax.scatter(xy[idx, 0], xy[idx, 1], s=ps, c=v[idx],
                            cmap="magma_r" if color not in s.obs else "viridis",
                            vmin=0 if color not in s.obs else None, vmax=vmax or None, linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(se if n > 1 else f"{sample}: {color}" + (f" | {filters}" if filters else ""), fontsize=9)
    if categorical:
        handles, labels = axes[0][0].get_legend_handles_labels()
        for ax in axes[0][1:]:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h); labels.append(l)
        fig.legend(handles, labels, markerscale=4, fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5),
                   frameon=False)
    else:
        fig.colorbar(sc, ax=axes[0].tolist(), shrink=0.6, label=color)
    if n > 1:
        fig.suptitle(f"{sample}: {color}" + (f" | {filters}" if filters else ""), fontsize=10)
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


def claim(s: Store, c: dict):
    """Per-sample values behind a claim, by group, with the verdict."""
    h = c["history"][-1]
    sp = c["spec"]
    df = pd.DataFrame({"v": pd.Series(h["per_sample"], dtype=float), "g": pd.Series(h["groups"])}).dropna()
    order = [sp["a"]] + ([sp["b"]] if sp["b"] is not None else [])
    fig, ax = plt.subplots(figsize=(1.4 * len(order) + 2.2, 3.2), dpi=110)
    for i, grp in enumerate(order):
        d = df[df["g"] == str(grp)]
        x = i + np.linspace(-0.12, 0.12, max(len(d), 1))[:len(d)]
        ax.scatter(x, d["v"], color=PALETTE[2 * i], s=30, zorder=3)
        for xi, (smp, yi) in zip(x, d["v"].items()):
            ax.annotate(smp, (xi, yi), fontsize=5, xytext=(4, 0), textcoords="offset points", va="center")
        if len(d):
            ax.hlines(d["v"].mean(), i - 0.25, i + 0.25, color="k", lw=1)
    if sp["b"] is None or sp["kind"] == "perturbation":
        ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.set_xticks(range(len(order)), order, fontsize=8)
    ax.set_xlim(-0.6, len(order) - 0.4)
    unit = {"expression": "mean log-expression", "composition": "log2 share",
            "colocalisation": "log2 O/E", "context": "predicted log-expression (context only)",
            "perturbation": "predicted change (log)"}[sp["kind"]]
    ax.set_ylabel(unit, fontsize=8)
    ax.set_title(f"{c['id'] or 'unrecorded'}: {h['verdict'].upper()}\n{c['statement'][:70]}", fontsize=8)
    p = _out(s, f"claim_{c['id'] or 'tmp'}")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p
