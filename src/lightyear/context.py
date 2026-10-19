"""Context model: a learned map from a cell's spatial neighbourhood to its expression state.

For each cell type, all panel genes are regressed (ridge, per-gene penalty) on the cell-type composition of
the cell's neighbourhood at two scales (k=10, k=30; fractions plus square roots, so "a few neighbours" and
"mostly surrounded by" are distinct) and local density. Fitting is cross-fitted by sample: every cell's
prediction comes from a model that never saw its animal, so performance numbers and in-silico experiments
are out-of-sample.

What the model supports:
- how much of a gene's variation within a cell type the neighbourhood predicts (cell- and sample-level)
- which neighbour types drive a gene (in-silico removal)
- counterfactuals: remove / replace neighbour types and predict the change
- how much of an a-vs-b difference the neighbourhood alone predicts

It is associational: "cells with more X neighbours express more g", not proof that X causes g.
Counterfactual neighbourhoods are checked against the training data (support) so the model is not asked
about neighbourhoods it never saw.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from . import spatial, stats
from .store import Store

SCALES = (10, 30)
LAMBDAS = (1e-1, 1e0, 1e1, 1e2, 1e3)
MIN_CELLS = 500
SUPPORT_N = 20000
BLAS_THREADS = 4  # OpenBLAS's default thread count thrashes on small dense products (100x slower on an M1)


# -- features ------------------------------------------------------------------------
def _comps(s: Store, cells: np.ndarray) -> tuple[list[np.ndarray], list[str]]:
    ct = s.role("cell_type")
    out = []
    for k in SCALES:
        c, cats = spatial.composition_matrix(s, ct, k, cells=cells)
        out.append(c)
    return out, cats


def _density(s: Store, cells: np.ndarray) -> np.ndarray:
    _, dist = s.knn
    return np.log1p(np.asarray(dist[cells, :SCALES[0]], np.float32).mean(1))[:, None]


def _design(comps: list[np.ndarray], dens: np.ndarray) -> np.ndarray:
    parts = []
    for c in comps:
        parts += [c, np.sqrt(c)]
    return np.hstack(parts + [dens]).astype(np.float32)


def _group_onehot(groups: np.ndarray, levels: list[str], override: str | None = None) -> np.ndarray:
    """Condition indicators (first level = reference). Lets the model learn neighbourhood effects within
    condition instead of using the neighbourhood as a proxy for the condition."""
    g = np.full(len(groups), str(override)) if override is not None else np.asarray(groups, str)
    return np.stack([g == lv for lv in levels[1:]], 1).astype(np.float32) if len(levels) > 1 \
        else np.zeros((len(g), 0), np.float32)


def counterfactual(comps: list[np.ndarray], cats: list[str], remove: list[str] | None = None,
                   replace: dict | None = None) -> list[np.ndarray]:
    """Edit neighbourhood compositions: `remove` types (others renormalised) and/or `replace` {from: to}
    (the fraction of `from` is given to `to`). Rows left empty become NaN."""
    ix = {c: i for i, c in enumerate(cats)}
    for name in list(remove or []) + list((replace or {}).keys()) + list((replace or {}).values()):
        if name not in ix:
            raise KeyError(f"{name!r} is not a cell type; options: {cats}")
    out = []
    for c in comps:
        c = c.copy()
        for f, t in (replace or {}).items():
            c[:, ix[t]] += c[:, ix[f]]
            c[:, ix[f]] = 0
        if remove:
            c[:, [ix[r] for r in remove]] = 0
            tot = c.sum(1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                c = np.where(tot > 0, c / tot, np.nan)
        out.append(c.astype(np.float32))
    return out


# -- fitting --------------------------------------------------------------------------
def _model_dir(s: Store, name: str):
    return s.path / "models" / name


def _folds(samples: list[str], groups: pd.Series, max_folds: int = 8) -> dict:
    """Sample -> fold. Leave-one-sample-out when there are few samples, else group-balanced folds."""
    if len(samples) <= max_folds:
        return {smp: i for i, smp in enumerate(samples)}
    order = sorted(samples, key=lambda x: (groups[x], x))
    return {smp: i % 5 for i, smp in enumerate(order)}


def _panel_gap_genes(s: Store, samples: list[str]) -> set[str]:
    p = s.card_dir / "tables" / "panel_gaps.parquet"
    if not p.exists():
        return set()
    gaps = pd.read_parquet(p)
    return {g for g, r in gaps.iterrows() if set(r.missing_in.split(";")) & set(samples)}


@threadpool_limits.wrap(limits=BLAS_THREADS, user_api="blas")
def _variable_genes(s: Store, cells_mask: np.ndarray, n: int, min_frac: float = 0.01) -> list[str]:
    """The n most variable panel genes over the training cells (expressed in >= min_frac of them)."""
    m = cells_mask.sum()
    rows = []
    for g in s.genes:
        idx, val = s.gene_nz(g)
        val = val[cells_mask[idx]]
        mu = val.sum() / m
        rows.append((g, len(val) / m, (val ** 2).sum() / m - mu ** 2))
    df = pd.DataFrame(rows, columns=["gene", "frac", "var"])
    return df[df.frac >= min_frac].nlargest(n, "var")["gene"].tolist()


def fit(s: Store, name: str, groups: list[str] | None = None, cell_types: list[str] | None = None,
        group_column: str | None = None, max_cells: int = 100_000, seed: int = 0, log=print,
        genes: list[str] | None = None, max_genes: int = 1000) -> pd.DataFrame:
    """Fit cross-fitted context models for each cell type on the samples of `groups` (default: all).
    Panels larger than `max_genes` are reduced to the most variable genes plus any `genes` asked for.
    Saves to <store>/models/<name>/ and returns the per-(cell type, gene) performance table."""
    samp, ctc = s.role("sample"), s.role("cell_type")
    group_column = group_column or s.role("group")
    sg = s.obs.groupby(samp, observed=True)[group_column].first().astype(str)
    samples = sorted(sg.index if not groups else sg.index[sg.isin([str(g) for g in groups])])
    if len(samples) < 3:
        raise ValueError("need at least 3 samples to cross-fit")
    fold_of = _folds(samples, sg)
    gaps = _panel_gap_genes(s, samples)
    rng = np.random.default_rng(seed)
    smp_arr = s.strcol(samp)
    ct_arr = s.strcol(ctc)
    in_train = np.isin(smp_arr, samples)
    asked = [s.resolve_gene(g) for g in (genes or [])]
    pool = s.genes if len(s.genes) <= max_genes else _variable_genes(s, in_train, max_genes)
    genes = [g for g in dict.fromkeys(pool + asked) if g not in gaps]
    types = cell_types or [c for c, n in pd.Series(ct_arr[in_train]).value_counts().items() if n >= MIN_CELLS]
    # subsample per (cell type, sample), then read every gene row once for all cell types
    picks = {}
    for ct in types:
        cells = np.flatnonzero(in_train & (ct_arr == ct))
        cap = max(1, max_cells // len(samples))
        keep = []
        for smp in samples:
            c = cells[smp_arr[cells] == smp]
            keep.append(c if len(c) <= cap else np.sort(rng.choice(c, cap, replace=False)))
        picks[ct] = np.concatenate(keep)
    allc = np.concatenate(list(picks.values()))
    pos = {ct: slice(o, o + len(p)) for ct, p, o in
           zip(picks, picks.values(), np.cumsum([0] + [len(p) for p in picks.values()])[:-1])}
    Y_all = np.empty((len(allc), len(genes)), np.float16)
    for j, g in enumerate(genes):
        Y_all[:, j] = s.gene(g)[allc]
    comps_all, cats = _comps(s, allc)
    levels = sorted(set(sg[samples]))
    grp_arr = s.strcol(group_column)
    X_all = np.hstack([_design(comps_all, _density(s, allc)), _group_onehot(grp_arr[allc], levels)])
    d = _model_dir(s, name)
    d.mkdir(parents=True, exist_ok=True)
    perf, arrays = [], {}
    for ct in types:
        sl = pos[ct]
        X, Y = X_all[sl].astype(np.float64), Y_all[sl].astype(np.float64)
        cs = smp_arr[picks[ct]]
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Xs = np.c_[np.ones(len(X)), (X - mu) / sd]
        F = Xs.shape[1]
        fold = np.array([fold_of[x] for x in cs])
        nf = fold.max() + 1
        Wf = np.zeros((nf, len(LAMBDAS), F, Y.shape[1]))
        sse = np.zeros((len(LAMBDAS), Y.shape[1]))
        # condition-only baseline: held-out cells predicted by their condition's mean in the training folds
        cg = grp_arr[picks[ct]]
        sse_base = np.zeros(Y.shape[1])
        for f in range(nf):
            tr, te = fold != f, fold == f
            for lv in np.unique(cg[te]):
                ref = tr & (cg == lv)
                mu_lv = Y[ref].mean(0) if ref.any() else Y[tr].mean(0)
                sse_base += ((Y[te & (cg == lv)] - mu_lv) ** 2).sum(0)
        ss_codes, ss_names = pd.factorize(cs, sort=True)
        pmeans = np.zeros((len(LAMBDAS), len(ss_names), Y.shape[1]))
        P = np.empty(Y.shape, np.float32)
        for f in range(nf):
            tr = fold != f
            A, B = Xs[tr].T @ Xs[tr], Xs[tr].T @ Y[tr]
            for li, lam in enumerate(LAMBDAS):
                R = lam * tr.sum() / 1000 * np.eye(F)
                R[0, 0] = 0
                Wf[f, li] = np.linalg.solve(A + R, B)
        for li in range(len(LAMBDAS)):  # held-out predictions, one penalty at a time
            for f in range(nf):
                te = fold == f
                P[te] = Xs[te] @ Wf[f, li]
            sse[li] = ((P - Y) ** 2).sum(0)
            for j in range(len(ss_names)):
                pmeans[li, j] = P[ss_codes == j].mean(0)
        best = sse.argmin(0)
        sst = ((Y - Y.mean(0)) ** 2).sum(0)
        r2 = 1 - sse[best, np.arange(Y.shape[1])] / np.maximum(sst, 1e-9)
        # sample level: held-out sample means of observed vs predicted
        ob = pd.DataFrame(np.stack([Y[ss_codes == j].mean(0) for j in range(len(ss_names))]),
                          index=ss_names, columns=genes)
        pr = pd.DataFrame(pmeans[best, :, np.arange(Y.shape[1])].T, index=ss_names, columns=genes)
        between = ((ob - ob.mean()) ** 2).sum()
        r2_s = 1 - ((ob - pr) ** 2).sum() / between.where(between > 1e-9)
        r2_ctx = 1 - sse[best, np.arange(Y.shape[1])] / np.maximum(sse_base, 1e-9)
        perf.append(pd.DataFrame({"cell_type": ct, "gene": genes, "r2_context": r2_ctx, "r2_cell": r2,
                                  "r2_sample": r2_s.to_numpy(),
                                  "mean": Y.mean(0), "sd_between_samples": ob.std().to_numpy(),
                                  "lambda": np.array(LAMBDAS)[best], "n_train": len(Y)}))
        # per-fold and full-data weights at each gene's chosen lambda
        A, B = Xs.T @ Xs, Xs.T @ Y
        Wfull = np.zeros((F, Y.shape[1]))
        for li, lam in enumerate(LAMBDAS):
            R = lam * len(Y) / 1000 * np.eye(F)
            R[0, 0] = 0
            sel = best == li
            if sel.any():
                Wfull[:, sel] = np.linalg.solve(A + R, B[:, sel])
        Wsel = np.take_along_axis(Wf, best[None, None, None, :].repeat(nf, 0).repeat(F, 2), 1)[:, 0]
        key = _key(ct)
        arrays[f"{key}__W"] = Wsel.astype(np.float32)
        arrays[f"{key}__Wfull"] = Wfull.astype(np.float32)
        arrays[f"{key}__mu"], arrays[f"{key}__sd"] = mu.astype(np.float32), sd.astype(np.float32)
        # support reference: training neighbourhoods (composition only)
        ref = np.hstack([c[sl] for c in comps_all])
        ref = ref[rng.choice(len(ref), min(SUPPORT_N, len(ref)), replace=False)]
        dd, _ = cKDTree(ref).query(ref, k=2)
        arrays[f"{key}__ref"] = ref.astype(np.float32)
        arrays[f"{key}__ref_thr"] = np.float32(np.quantile(dd[:, 1], 0.99))
        log(f"  {ct}: n={len(Y)}, median r2_context={np.median(r2_ctx):.3f}")
    np.savez_compressed(d / "weights.npz", **arrays)
    tm, _, _ = stats.group_stats(s, [ctc], genes, mask=in_train, min_cells=MIN_CELLS)
    tm.index = tm.index.astype(str)
    tm.to_parquet(d / "type_means.parquet")
    perf = pd.concat(perf, ignore_index=True)
    perf.to_parquet(d / "performance.parquet")
    meta = {"name": name, "samples": samples, "fold_of": fold_of, "genes": genes, "cats": cats,
            "group_levels": levels,
            "types": types, "group_column": group_column, "groups": groups, "scales": list(SCALES),
            "dropped_panel_gap_genes": sorted(gaps), "max_cells": max_cells}
    (d / "model.json").write_text(json.dumps(meta, indent=1))
    return perf


def _key(ct: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in ct)


# -- using a fitted model ---------------------------------------------------------------------
class ContextModel:
    def __init__(self, s: Store, name: str):
        d = _model_dir(s, name)
        if not (d / "model.json").exists():
            have = [p.name for p in (s.path / "models").glob("*")] if (s.path / "models").exists() else []
            raise KeyError(f"no context model {name!r}; fitted: {have}")
        self.s, self.name = s, name
        self.meta = json.loads((d / "model.json").read_text())
        self.w = np.load(d / "weights.npz")
        self.perf = pd.read_parquet(d / "performance.parquet")
        self.gi = {g: i for i, g in enumerate(self.meta["genes"])}
        self.type_means = pd.read_parquet(d / "type_means.parquet")

    def cells_of(self, filters: dict | None, cell_type: str | None = None) -> np.ndarray:
        m = self.s.mask(filters)
        if cell_type:
            m &= self.s.mask({self.s.role("cell_type"): cell_type})
        return np.flatnonzero(m)

    def single_type(self, cells: np.ndarray) -> str:
        ct = self.s.strcol(self.s.role("cell_type"))[cells]
        u = np.unique(ct)
        if len(u) != 1:
            raise ValueError(f"cells must be one cell type (got {list(u)[:5]}); add a cell_type filter")
        if u[0] not in self.meta["types"]:
            raise KeyError(f"{u[0]!r} not modelled; modelled: {self.meta['types']}")
        return str(u[0])

    def gene_cols(self, genes: list[str] | None) -> list[str]:
        if not genes:
            return self.meta["genes"]
        out = [self.s.resolve_gene(g) for g in genes]
        missing = [g for g in out if g not in self.gi]
        if missing:
            raise KeyError(f"not modelled: {missing}")
        return out

    @threadpool_limits.wrap(limits=BLAS_THREADS, user_api="blas")
    def predict(self, cells: np.ndarray, genes: list[str], comps: list[np.ndarray] | None = None,
                cross_fit: bool = True, group: str | None = None) -> np.ndarray:
        """Predicted expression (cells x genes). Cells from training samples use the model that never saw
        their sample (cross_fit); other cells use the full-data model. group: predict as if every cell were
        in this condition (default: its own)."""
        ct = self.single_type(cells)
        k = _key(ct)
        if comps is None:
            comps, _ = _comps(self.s, cells)
        grp = self.s.strcol(self.meta["group_column"])[cells]
        X = np.hstack([_design(comps, _density(self.s, cells)),
                       _group_onehot(grp, self.meta["group_levels"], group)]).astype(np.float64)
        Xs = np.c_[np.ones(len(X)), (X - self.w[f"{k}__mu"]) / self.w[f"{k}__sd"]]
        gi = [self.gi[g] for g in genes]
        smp = self.s.strcol(self.s.role("sample"))[cells]
        out = Xs @ self.w[f"{k}__Wfull"][:, gi]
        if cross_fit:
            W = self.w[f"{k}__W"]
            for smp_name, f in self.meta["fold_of"].items():
                sel = smp == smp_name
                if sel.any():
                    out[sel] = Xs[sel] @ W[f][:, gi]
        return out.astype(np.float32)

    def support(self, cells: np.ndarray, comps: list[np.ndarray], max_check: int = 20000, seed: int = 0) -> float:
        """Share of (counterfactual) neighbourhoods within the training distribution."""
        k = _key(self.single_type(cells))
        q = np.hstack(comps)
        q = q[~np.isnan(q).any(1)]
        if len(q) > max_check:
            q = q[np.random.default_rng(seed).choice(len(q), max_check, replace=False)]
        if not len(q):
            return 0.0
        d, _ = cKDTree(self.w[f"{k}__ref"]).query(q, k=1)
        return float(np.mean(d <= self.w[f"{k}__ref_thr"]))


def _per_sample_mean(s: Store, cells: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    smp = s.strcol(s.role("sample"))[cells]
    return pd.DataFrame(values).groupby(smp).mean()


@threadpool_limits.wrap(limits=BLAS_THREADS, user_api="blas")
def perturb(s: Store, name: str, cells: dict, genes: list[str] | None = None, remove: list[str] | None = None,
            replace: dict | None = None) -> dict:
    """In-silico neighbourhood edit for `cells` (one cell type). Returns per-sample mean predicted expression
    with actual vs edited neighbourhoods, the delta, how many cells were affected and the share of edited
    neighbourhoods inside the training distribution."""
    m = ContextModel(s, name)
    idx = m.cells_of(cells)
    m.single_type(idx)
    genes = m.gene_cols(genes)
    comps, cats = _comps(s, idx)
    cf = counterfactual(comps, cats, remove, replace)
    ok = ~np.isnan(np.hstack(cf)).any(1)
    changed = ok & (np.abs(np.hstack(cf) - np.hstack(comps)).sum(1) > 1e-6)
    base = m.predict(idx[ok], genes, [c[ok] for c in comps])
    edit = m.predict(idx[ok], genes, [c[ok] for c in cf])
    smp = s.strcol(s.role("sample"))[idx[ok]]
    actual = pd.DataFrame(base, columns=genes).groupby(smp).mean()
    delta = pd.DataFrame(edit - base, columns=genes).groupby(smp).mean()
    delta_changed = pd.DataFrame((edit - base)[changed[ok]], columns=genes).groupby(smp[changed[ok]]).mean()
    frac_changed = pd.Series(changed[ok], index=smp).groupby(level=0).mean()
    return {"actual": actual, "delta": delta, "delta_in_changed_cells": delta_changed,
            "frac_cells_changed": frac_changed, "support": m.support(idx[ok][changed[ok]], [c[ok][changed[ok]] for c in cf]),
            "support_actual": m.support(idx[ok], [c[ok] for c in comps]),
            "n_dropped_empty": int((~ok).sum()), "genes": genes}


@threadpool_limits.wrap(limits=BLAS_THREADS, user_api="blas")
def explain(s: Store, name: str, a: str, b: str, cells: dict, genes: list[str] | None = None) -> pd.DataFrame:
    """Split the observed a-vs-b difference (mean of per-sample means) into
    - neighbourhood: predicted difference from the cells' actual neighbourhoods, condition held at b
    - condition: predicted difference from switching the condition, neighbourhoods held as they are in a
    - residual: what the model does not account for.
    All predictions are cross-fitted (each animal predicted by a model that never saw it)."""
    m = ContextModel(s, name)
    gcol = m.meta["group_column"]
    for g in (a, b):
        if str(g) not in m.meta["group_levels"]:
            raise KeyError(f"{g!r} not in the model's training groups {m.meta['group_levels']}")
    idx = m.cells_of({**cells, gcol: [a, b]})
    ct = m.single_type(idx)
    genes = m.gene_cols(genes)
    obs_ = _per_sample_mean(s, idx, np.stack([s.gene(g)[idx] for g in genes], 1))
    at_b = _per_sample_mean(s, idx, m.predict(idx, genes, group=b))
    at_a = _per_sample_mean(s, idx, m.predict(idx, genes, group=a))
    for d in (obs_, at_b, at_a):
        d.columns = genes
    sg = s.obs.groupby(s.role("sample"), observed=True)[gcol].first().astype(str).reindex(obs_.index)
    A, B = sg == str(a), sg == str(b)
    out = pd.DataFrame({"observed_delta": obs_[A].mean() - obs_[B].mean(),
                        "neighbourhood_part": at_b[A].mean() - at_b[B].mean(),
                        "condition_part": (at_a[A] - at_b[A]).mean()})
    out["residual"] = out["observed_delta"] - out["neighbourhood_part"] - out["condition_part"]
    with np.errstate(all="ignore"):
        out["share_neighbourhood"] = out["neighbourhood_part"] / out["observed_delta"]
    perf = m.perf[m.perf.cell_type == ct].set_index("gene")
    out["model_r2_context"] = perf["r2_context"].reindex(out.index)
    return out.sort_values("observed_delta", key=np.abs, ascending=False)


@threadpool_limits.wrap(limits=BLAS_THREADS, user_api="blas")
def drivers(s: Store, name: str, cell_type: str, genes: list[str], cells: dict | None = None,
            max_cells: int = 30000, seed: int = 0) -> pd.DataFrame:
    """For each neighbour type: mean predicted change in each gene when that type is removed from the
    neighbourhoods of `cell_type` cells (only cells that have it). Negative = the neighbour type is
    associated with higher expression."""
    m = ContextModel(s, name)
    idx = m.cells_of(cells, cell_type)
    if len(idx) > max_cells:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_cells, replace=False))
    genes = m.gene_cols(genes)
    comps, cats = _comps(s, idx)
    base = m.predict(idx, genes, comps)
    rows = []
    for c in cats:
        cf = counterfactual(comps, cats, remove=[c])
        has = (comps[0][:, cats.index(c)] > 0) | (comps[1][:, cats.index(c)] > 0)
        ok = has & ~np.isnan(np.hstack(cf)).any(1)
        if ok.sum() < 50:
            continue
        d = (m.predict(idx[ok], genes, [x[ok] for x in cf]) - base[ok]).mean(0)
        rows.append({"neighbour_type": c, "share_of_cells_with_it": float(has.mean()),
                     **{g: float(v) for g, v in zip(genes, d)}})
    return pd.DataFrame(rows).set_index("neighbour_type")


SPILL_LOG = 0.5  # driver type expresses the gene >= this many log units above the cell itself


def source_gap(m: ContextModel, cell_type: str, driver: str, gene: str) -> float:
    tm = m.type_means
    return float(tm.loc[driver, gene] - tm.loc[cell_type, gene]) if driver in tm.index else np.nan


def atlas(s: Store, name: str, cell_type: str, n_top: int = 15, min_mean: float = 0.05,
          genes: list[str] | None = None) -> pd.DataFrame:
    """The genes of `cell_type` whose expression the neighbourhood predicts best (held-out), each with the
    neighbour type most associated with higher / lower expression, and a label: 'spillover-like' when the
    neighbour that raises it expresses the gene far more than the cell itself (transcripts picked up from
    neighbours), otherwise 'state-like' (a context-dependent cell state)."""
    m = ContextModel(s, name)
    perf = m.perf[(m.perf.cell_type == cell_type) & (m.perf["mean"] >= min_mean)].set_index("gene")
    if perf.empty:
        raise KeyError(f"{cell_type!r} not modelled; modelled: {m.meta['types']}")
    top = m.gene_cols(genes) if genes else perf.sort_values("r2_context", ascending=False).index[:n_top].tolist()
    dr = drivers(s, name, cell_type, top)
    rows = []
    for g in top:
        col = dr[g]
        up, down = col.idxmin(), col.idxmax()  # removing `up` lowers g -> it is associated with higher g
        gap = source_gap(m, cell_type, up, g)
        label = "spillover-like" if up != cell_type and gap >= SPILL_LOG else "state-like"
        rows.append({"gene": g, "r2_context": perf.loc[g, "r2_context"], "r2_cell": perf.loc[g, "r2_cell"],
                     "mean": perf.loc[g, "mean"], "raised_by": up, "removal_effect": col[up],
                     "raiser_minus_self": gap, "lowered_by": down, "label": label})
    return pd.DataFrame(rows).set_index("gene")
