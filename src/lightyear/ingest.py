"""Ingest: learn a dataset once, write a dataset card the agent reads first.

The card is markdown (for the LLM and for humans) plus parquet tables under
card/tables/ that the query tools reuse instead of recomputing.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import spatial, stats
from .store import Store, build_store

SECTIONS = ["overview", "columns", "samples", "cell_types", "markers", "panel",
            "niches", "neighbourhoods", "contrasts", "caveats"]


def load_config(path) -> dict:
    return yaml.safe_load(Path(path).read_text()) if path else {}


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3g}", max_rows: int = 60) -> str:
    df = df.head(max_rows)
    cols = [str(c) for c in df.columns]
    idx_name = df.index.name or ""
    lines = ["| " + " | ".join([str(idx_name)] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for i, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(c.replace("|", "\\|") for c in [str(i)] + cells) + " |")
    return "\n".join(lines)


class CardBuilder:
    def __init__(self, store: Store, config: dict, k_neighbors: int = 10, log=print):
        self.s = store
        self.cfg = config
        self.k = k_neighbors
        self.log = log
        self.tables = store.card_dir / "tables"
        self.tables.mkdir(parents=True, exist_ok=True)
        self.sections: dict[str, str] = {}
        self.caveats: list[str] = list(config.get("caveats", []))
        self.sample = store.role("sample")
        self.ct = store.role("cell_type")
        self.group = store.role("group")

    def save(self, name: str, df: pd.DataFrame):
        out = df.copy()
        out.columns = [str(c) for c in out.columns]
        out.reset_index().to_parquet(self.tables / f"{name}.parquet")

    # -- sections ----------------------------------------------------------
    def overview(self):
        m, o = self.s.meta, self.s.obs
        lines = [f"# Dataset card: {m['name']}", ""]
        if m.get("dataset_description"):
            lines += [m["dataset_description"].strip(), ""]
        lines += [
            f"- cells: {len(o):,}; genes: {len(self.s.genes):,}",
            f"- source: `{m['source']}` (expression from `{m['expr_layer']}`"
            + (f": {m['expr_description']}" if m.get("expr_description") else "") + ")",
            f"- coordinates: `{m['schema']['spatial_key']}`",
            f"- key columns: " + ", ".join(f"{r} = `{c}`" for r, c in self.s.roles.items()),
        ]
        if self.sample:
            lines.append(f"- samples/sections: {o[self.sample].nunique()}")
        if self.group:
            counts = o.groupby(self.group, observed=True)[self.sample].nunique() if self.sample else None
            if counts is not None:
                lines.append("- samples per group: " + ", ".join(f"{k} ({v})" for k, v in counts.items()))
        graphs = m["schema"].get("graphs") or {}
        if graphs:
            lines.append("- precomputed graphs in source obsp: " + ", ".join(graphs))
        self.sections["overview"] = "\n".join(lines)

    def columns(self):
        rows = []
        for name, c in self.s.columns.items():
            if c["top_values"]:
                vals = ", ".join(v for v, _ in c["top_values"][:8])
                if c["n_unique"] > 8:
                    vals += ", ..."
            elif c["numeric_summary"]:
                ns = c["numeric_summary"]
                vals = f"median {ns['median']:.3g} (p05 {ns['p05']:.3g}, p95 {ns['p95']:.3g})"
            else:
                vals = ""
            rows.append({"column": name, "role": c["role"], "level": c["level"],
                         "n_unique": c["n_unique"], "values": vals, "description": c["description"]})
        df = pd.DataFrame(rows).set_index("column")
        self.save("columns", df)
        self.sections["columns"] = "## Columns\n\n`level=sample` means constant within each sample.\n\n" + _md_table(df, max_rows=200)

    def samples(self):
        if not self.sample:
            return
        o = self.s.obs
        sample_cols = [n for n, c in self.s.columns.items() if c["level"] == "sample" and n != self.sample
                       and c["n_unique"] <= 50]
        df = o.groupby(self.sample, observed=True).agg(n_cells=(self.sample, "size"))
        for c in sample_cols:
            df[c] = o.groupby(self.sample, observed=True)[c].first().astype(str)
        for q in ("total_counts", "transcript_counts", "n_genes", "n_genes_by_counts"):
            if q in o.columns:
                df[f"median_{q}"] = o.groupby(self.sample, observed=True)[q].median()
                med = df[f"median_{q}"]
                low = med[med < 0.5 * med.median()]
                for smp, v in low.items():
                    self.caveats.append(f"Sample `{smp}` has low depth (median {q} {v:.0f} vs dataset median "
                                        f"{med.median():.0f}); differences there may be technical.")
                break
        self.save("samples", df)
        self.sections["samples"] = "## Samples\n\n" + _md_table(df)
        # nested covariates: e.g. every group lives in exactly one batch (skip per-sample identifiers)
        n_samp = len(df)
        coarse = [c for c in sample_cols if 1 < df[c].nunique() < n_samp]
        for a in coarse:
            outer = [b for b in coarse if b != a and df[b].nunique() < df[a].nunique()
                     and len(df[[a, b]].drop_duplicates()) == df[a].nunique()]
            if outer:
                self.caveats.append(f"`{a}` is nested in {', '.join(f'`{b}`' for b in outer)} (each {a} value occurs "
                                    f"in only one of each); differences between {a} values from different "
                                    f"{'/'.join(outer)} are confounded with them.")

    def cell_types(self):
        if not self.ct:
            return
        o = self.s.obs
        counts = o[self.ct].astype(str).value_counts().rename("n_cells").to_frame()
        counts["fraction"] = counts["n_cells"] / counts["n_cells"].sum()
        text = ["## Cell types", "", f"Column `{self.ct}`.", "", _md_table(counts)]
        self.save("celltype_counts", counts)
        if self.sample:
            per_sample = stats.composition(o, self.ct, [self.sample])
            self.save("celltype_by_sample", per_sample)
            if self.group:
                sg = o.groupby(self.sample, observed=True)[self.group].first().astype(str)
                ps = per_sample.copy()
                ps["_g"] = sg.reindex(ps.index).to_numpy()
                mean = ps.groupby("_g").mean().T
                sd = ps.groupby("_g").std().T
                mean.index.name = "cell type (mean fraction of sample)"
                self.save("celltype_by_group_mean", mean)
                self.save("celltype_by_group_sd", sd)
                text += ["", f"Mean fraction per sample, by `{self.group}`:", "", _md_table(mean)]
        self.sections["cell_types"] = "\n".join(text)

    def markers(self):
        if not self.ct:
            return
        t0 = time.time()
        mean, frac, n = stats.group_stats(self.s, [self.ct])
        self.save("celltype_mean", mean)
        self.save("celltype_frac", frac)
        mk = stats.markers(mean, frac, n_top=10)
        rows = []
        lines = ["## Marker genes", "", "Top genes per cell type vs the mean of other types "
                 "(lfc = difference in mean log-normalised expression; % = fraction of cells >0).", ""]
        for grp, df in mk.items():
            lines.append(f"- **{grp}** (n={n.get(grp, 0):,}): " +
                         ", ".join(f"{g} ({r.lfc_vs_rest:.2f}, {100 * r.frac:.0f}%)" for g, r in df.iterrows()))
            rows += [{"cell_type": grp, "gene": g, **r.to_dict()} for g, r in df.iterrows()]
        self.save("markers", pd.DataFrame(rows).set_index("cell_type"))
        self.sections["markers"] = "\n".join(lines)
        self.log(f"  markers {time.time() - t0:.0f}s")

    def panel(self):
        """Genes that are detected in some samples but essentially absent from others -> panel differences."""
        if not self.sample:
            return
        t0 = time.time()
        _, frac, _ = stats.group_stats(self.s, [self.sample])
        self.save("gene_frac_by_sample", frac)
        absent = frac < 1e-4
        present = frac > 5e-3
        gaps = [(g, list(frac.index[absent[g]])) for g in frac.columns if absent[g].any() and present[g].any()]
        lines = ["## Panel coverage", ""]
        if gaps:
            # group genes by the identical set of missing samples
            by_set: dict[tuple, list] = {}
            for g, miss in gaps:
                by_set.setdefault(tuple(miss), []).append(g)
            lines.append(f"{len(gaps)} genes look unmeasured (≈0 detection) in some samples while detected elsewhere "
                         "- likely different gene panels. Treat zeros there as missing, not as absent expression.")
            for miss, genes in by_set.items():
                lines.append(f"- missing in {', '.join(map(str, miss))}: {', '.join(sorted(genes))}")
            self.caveats.append(f"{len(gaps)} genes are not measured in all samples (see Panel coverage).")
            pd.DataFrame([{"gene": g, "missing_in": ";".join(map(str, m))} for g, m in gaps]) \
                .set_index("gene").to_parquet(self.tables / "panel_gaps.parquet")
        else:
            lines.append("All genes are detected in every sample.")
        self.sections["panel"] = "\n".join(lines)
        self.log(f"  panel {time.time() - t0:.0f}s")

    def niches(self):
        cols = [n for n, c in self.s.columns.items()
                if c["role"] == "niche" and 1 < c["n_unique"] <= 50 and n != self.ct]
        cols = self.cfg.get("niche_columns", cols)
        if not cols or not self.ct:
            return
        o = self.s.obs
        lines = ["## Niches / regions", ""]
        for nc in cols:
            comp = stats.composition(o, self.ct, [nc])
            self.save(f"niche_composition__{nc}", comp)
            desc = self.s.columns[nc]["description"]
            lines.append(f"### `{nc}`" + (f" - {desc}" if desc else ""))
            for niche, row in comp.iterrows():
                top = row.sort_values(ascending=False).head(4)
                n_cells = int((o[nc].astype(str) == niche).sum())
                lines.append(f"- **{niche}** ({n_cells:,} cells): " +
                             ", ".join(f"{k} {100 * v:.0f}%" for k, v in top.items()))
            if self.sample and self.group:
                per = stats.composition(o, nc, [self.sample])
                sg = o.groupby(self.sample, observed=True)[self.group].first().astype(str)
                per["_g"] = sg.reindex(per.index).to_numpy()
                mean = per.groupby("_g").mean().T
                mean.index.name = f"{nc} (mean fraction of sample)"
                self.save(f"niche_by_group__{nc}", mean)
                lines += ["", _md_table(mean)]
            lines.append("")
        self.sections["niches"] = "\n".join(lines)

    def neighbourhoods(self):
        if not (self.ct and self.sample):
            return
        t0 = time.time()
        o = self.s.obs
        counts, names = spatial.pair_counts(self.s, self.ct, k=self.k)
        rows = []
        for sec, cnt in counts.items():
            lo = stats.enrichment_log2oe(cnt)
            for i, a in enumerate(names):
                for j, b in enumerate(names):
                    rows.append((sec, a, b, int(cnt[i, j]), float(lo[i, j])))
        df = pd.DataFrame(rows, columns=["sample", "centre", "neighbour", "count", "log2_oe"])
        df.set_index("sample").to_parquet(self.tables / "nhood_by_sample.parquet")
        lines = ["## Neighbourhoods", "",
                 f"kNN (k={self.k}) within each section; log2(observed/expected) of neighbour cell types, "
                 "averaged over samples. Positive = co-localised beyond chance.", ""]
        if self.group:
            sg = o.groupby(self.sample, observed=True)[self.group].first().astype(str)
            df["group"] = sg.reindex(df["sample"]).to_numpy()
            agg = df.groupby(["group", "centre", "neighbour"])["log2_oe"].mean().reset_index()
        else:
            agg = df.groupby(["centre", "neighbour"])["log2_oe"].mean().reset_index()
            agg["group"] = "all"
        agg.to_parquet(self.tables / "nhood_by_group.parquet")
        off = agg[agg["centre"] < agg["neighbour"]]
        for g, sub in off.groupby("group"):
            top = sub.sort_values("log2_oe", ascending=False).head(8)
            lines.append(f"- **{g}** strongest heterotypic pairs: " +
                         ", ".join(f"{r.centre}–{r.neighbour} ({r.log2_oe:+.2f})" for r in top.itertuples()))
        self.sections["neighbourhoods"] = "\n".join(lines)
        self.log(f"  neighbourhoods {time.time() - t0:.0f}s")

    def contrasts(self):
        cons = self.cfg.get("contrasts", [])
        if not cons or not (self.ct and self.sample):
            return
        t0 = time.time()
        o = self.s.obs
        mean, _, n = stats.group_stats(self.s, [self.sample, self.ct], min_cells=20)
        self.save("pseudobulk_sample_celltype", mean)
        self.save("pseudobulk_ncells", n.to_frame())
        lines = ["## Condition contrasts", "",
                 "Per cell type, sample-level pseudobulk (mean log-normalised expression per sample, ≥20 cells). "
                 "delta = mean(a) − mean(b); p from Welch t-test across samples (few replicates - treat as "
                 "effect sizes, not significance).", ""]
        allres = []
        for con in cons:
            col, a, b = con.get("column", self.group), str(con["a"]), str(con["b"])
            sg = o.groupby(self.sample, observed=True)[col].first().astype(str)
            lines.append(f"### {a} vs {b} (`{col}`)")
            for ct in sorted(mean.index.get_level_values(1).unique()):
                pb = stats.mask_panel_gaps(self.s, mean.xs(ct, level=1))
                res = stats.compare_samples(pb, sg, a, b)
                if res.empty:
                    continue
                res = res[res["n_a"] + res["n_b"] >= 3]
                top = res.head(8)
                if top.empty:
                    continue
                res = res.assign(contrast=f"{a}_vs_{b}", cell_type=ct)
                allres.append(res.rename(columns={f"mean_{a}": "mean_a", f"mean_{b}": "mean_b"}))
                lines.append(f"- **{ct}**: " + ", ".join(
                    f"{g} {r.delta:+.2f}" + (f" (p={r.p:.2g})" if "p" in res and not np.isnan(r.p) else "")
                    for g, r in top.iterrows()))
            lines.append("")
        if allres:
            full = pd.concat(allres)
            full.index.name = "gene"
            full.reset_index().to_parquet(self.tables / "contrasts.parquet")
            lines += self._global_shifts(full)
        self.sections["contrasts"] = "\n".join(lines)
        self.log(f"  contrasts {time.time() - t0:.0f}s")

    def _global_shifts(self, full: pd.DataFrame, min_delta: float = 0.2, min_share: float = 0.75) -> list[str]:
        """Genes shifting the same way in most cell types: usually spillover/ambient RNA from a strongly
        changed cell population (or tissue-wide composition), not regulation in each cell type."""
        out = []
        for con, sub in full.reset_index().groupby("contrast"):
            n_ct = sub["cell_type"].nunique()
            if n_ct < 4:
                continue
            for sign, word in ((1, "up"), (-1, "down")):
                hit = sub[sign * sub["delta"] >= min_delta].groupby("gene")["cell_type"].nunique()
                hit = hit[hit >= min_share * n_ct].sort_values(ascending=False)
                if len(hit):
                    genes = ", ".join(f"{g} ({k}/{n_ct})" for g, k in hit.items())
                    msg = (f"{con}: {genes} {word} by ≥{min_delta} in most cell types - either spillover / "
                           "ambient signal from the cells that really change, or a genuine tissue-wide response "
                           "(e.g. interferon / MHC-I in inflammation). Don't call these regulated in a given cell "
                           "type without test_claim (its specificity test separates the two by expression level "
                           "relative to the top expresser) or gene_near.")
                    out.append(f"- ⚠ {msg}")
                    self.caveats.append(msg)
        return [""] + out if out else []

    def finish(self):
        lines = ["## Caveats", ""] + [f"- {c}" for c in dict.fromkeys(self.caveats)] if self.caveats \
            else ["## Caveats", "", "- none detected automatically"]
        self.sections["caveats"] = "\n".join(lines)
        ordered = [self.sections[k] for k in SECTIONS if k in self.sections]
        (self.s.card_dir / "card.md").write_text("\n\n".join(ordered) + "\n")
        (self.s.card_dir / "sections.json").write_text(json.dumps(self.sections, indent=1))

    def run(self):
        for step in SECTIONS[:-1]:
            self.log(f"card: {step}")
            getattr(self, step)()
        self.finish()


def ingest(h5ad, out, config_path=None, layer=None, rebuild_store=False, k_neighbors=10, log=print) -> Store:
    cfg = load_config(config_path)
    out = Path(out)
    layer = layer or cfg.get("layer")
    if rebuild_store or not (out / "meta.json").exists():
        log(f"building store {out} from {h5ad}")
        t0 = time.time()
        store = build_store(h5ad, out, overrides=cfg, layer=layer, name=cfg.get("name"), sparse=cfg.get("sparse"),
                            log=log)
        log(f"  store built in {time.time() - t0:.0f}s")
    else:
        store = Store(out)
        cols = store.meta["schema"]["columns"]  # apply config edits that don't need a rebuild
        for col, role in cfg.get("column_roles", {}).items():
            if col in cols:
                cols[col]["role"] = role
        for col, desc in cfg.get("descriptions", {}).items():
            if col in cols:
                cols[col]["description"] = desc
        store.meta["dataset_description"] = cfg.get("description", store.meta.get("dataset_description", ""))
        store.save_meta()
        store = Store(out)
    if cfg.get("niche_columns"):
        store.meta["niche_columns"] = cfg["niche_columns"]
        store.save_meta()
    if cfg.get("exclude_cell_types"):
        store.meta["exclude_cell_types"] = cfg["exclude_cell_types"]
        store.save_meta()
    if cfg.get("organism"):
        store.meta["organism"] = cfg["organism"]
        store.save_meta()
    if store.role("sample"):
        spatial.ensure_knn(store, log=log)
    CardBuilder(store, cfg, k_neighbors=k_neighbors, log=log).run()
    log(f"card written to {store.card_dir / 'card.md'}")
    return store
