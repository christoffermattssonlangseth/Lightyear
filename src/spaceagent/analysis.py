"""Higher-level analyses: annotation support, gates/scores, case-control, separability, ligand-receptor."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import resources, spatial, stats
from .store import Store


def _sample_groups(s: Store, col: str) -> pd.Series:
    return s.obs.groupby(s.role("sample"), observed=True)[col].first().astype(str)


def organism(s: Store) -> str:
    return s.meta.get("organism") or "mouse"


# -- annotation ---------------------------------------------------------------
def markers_for(s: Store, column: str, filters: dict | None = None, n_top: int = 10,
                min_frac: float = 0.1, min_cells: int = 20) -> dict:
    mean, frac, n = stats.group_stats(s, [column], mask=s.mask(filters), min_cells=min_cells)
    mk = stats.markers(mean, frac, n_top=n_top, min_frac=min_frac)
    return {k: (int(n.get(k, 0)), v) for k, v in mk.items()}


def annotate_suggest(s: Store, column: str, filters: dict | None = None, n_markers: int = 15,
                     tissue: str | None = None, n_suggest: int = 3) -> pd.DataFrame:
    """For each category of `column`: its top panel markers and the marker-database cell types whose
    marker lists overlap them most (hypergeometric over panel genes). A hint for annotation, not a label."""
    db = resources.marker_db(organism(s))
    if tissue:
        db = db[db["tissue"].str.contains(tissue, case=False, na=False)]
    panel = set(s.genes)
    db = db[db["gene"].isin(panel)]
    sets = db.groupby("cell_type")["gene"].agg(set)
    sets = sets[sets.map(len) >= 2]
    universe = len(panel & set(db["gene"]))
    rows = []
    for cat, (n, df) in markers_for(s, column, filters, n_top=n_markers).items():
        top = set(df.index) & set(db["gene"])
        hits = []
        for ct, genes in sets.items():
            ov = top & genes
            if len(ov) < 2:
                continue
            p = sps.hypergeom.sf(len(ov) - 1, universe, len(genes), len(top))
            hits.append((p, ct, sorted(ov)))
        hits.sort()
        rows.append({column: cat, "n_cells": n, "top_markers": ", ".join(df.index[:10]),
                     "suggestions": "; ".join(f"{ct} (p={p:.1g}: {', '.join(ov)})" for p, ct, ov in hits[:n_suggest])
                     or "no database match"})
    return pd.DataFrame(rows).set_index(column)


def gate(s: Store, name: str, feature: str, threshold: float, within: dict | None = None,
         above: bool = True) -> dict:
    """Boolean column: cells (inside `within`) whose gene/score is above (or below) threshold."""
    v = s.gene(feature)
    m = (v > threshold) if above else (v <= threshold)
    m &= s.mask(within)
    desc = f"gate: {feature} {'>' if above else '<='} {threshold}" + (f" within {within}" if within else "")
    s.add_column(name, m, desc, role="gate")
    return {"column": name, "description": desc, "n_true": int(m.sum()), "n_within": int(s.mask(within).sum())}


def score_genes(s: Store, name: str, genes: list[str], n_ctrl: int = 50, seed: int = 0) -> dict:
    """Gene-set score = mean log-expression of the set minus mean of expression-matched random panel genes
    (scanpy score_genes logic, on the panel)."""
    genes = [s.resolve_gene(g) for g in genes]
    means = pd.Series({g: float(np.mean(s.gene(g)[::50])) for g in s.genes})
    bins = pd.qcut(means.rank(method="first"), 10, labels=False)
    rng = np.random.default_rng(seed)
    ctrl = set()
    for g in genes:
        pool = [x for x in means.index[bins == bins[g]] if x not in genes] \
            or [x for x in means.index if x not in genes]  # tiny panels
        ctrl |= set(rng.choice(pool, min(len(pool), max(1, n_ctrl // len(genes))), replace=False))
    sc = np.mean([s.gene(g) for g in genes], 0) - np.mean([s.gene(g) for g in ctrl], 0)
    desc = f"score_genes: {', '.join(genes)} minus {len(ctrl)} expression-matched control genes"
    s.add_column(name, sc.astype(np.float32), desc, role="score")
    return {"column": name, "description": desc, "genes": genes}


# -- case / control -------------------------------------------------------------
def _effect(va: np.ndarray, vb: np.ndarray):
    ma, mb = np.nanmean(va, 0), np.nanmean(vb, 0)
    na, nb = np.sum(~np.isnan(va), 0), np.sum(~np.isnan(vb), 0)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sa, sb = np.nanvar(va, 0, ddof=1), np.nanvar(vb, 0, ddof=1)
        sp = np.sqrt(((na - 1) * sa + (nb - 1) * sb) / np.maximum(na + nb - 2, 1))
        g = (ma - mb) / np.maximum(sp, 1e-3) * (1 - 3 / (4 * (na + nb) - 9))
        p = sps.ttest_ind(va, vb, equal_var=False, axis=0, nan_policy="omit").pvalue \
            if min(len(va), len(vb)) >= 2 else np.full(len(ma), np.nan)
    return ma, mb, g, np.asarray(p)


def _feature_table(name: str, per_sample: pd.DataFrame, A: list, B: list, kind: str, log_ratio: bool):
    va, vb = per_sample.reindex(A).to_numpy(float), per_sample.reindex(B).to_numpy(float)
    ma, mb, g, p = _effect(va, vb)
    df = pd.DataFrame({"feature": per_sample.columns.astype(str), "kind": kind, "source": name,
                       "mean_a": ma, "mean_b": mb, "hedges_g": g, "p": p})
    df["effect"] = np.log2((ma + 1e-3) / (mb + 1e-3)) if log_ratio else ma - mb
    df["effect_unit"] = "log2 ratio" if log_ratio else "delta"
    return df


def case_control(s: Store, a: str, b: str, column: str | None = None, cell_types: list[str] | None = None,
                 min_frac: float = 0.005) -> pd.DataFrame:
    """Everything that separates samples of `a` from samples of `b`, at the sample level:
    cell-type composition, niche/region abundance, neighbourhood (co-localisation) structure,
    and per-cell-type pseudobulk expression. Returns one long table ranked by |Hedges g|."""
    column = column or s.role("group")
    samp, ct = s.role("sample"), s.role("cell_type")
    sg = _sample_groups(s, column)
    A, B = list(sg.index[sg == str(a)]), list(sg.index[sg == str(b)])
    if not A or not B:
        raise ValueError(f"no samples for {a!r} or {b!r} in {column!r}; values: {sorted(sg.unique())}")
    keep = s.mask({column: [a, b]})
    tabs = []
    # composition
    comp = stats.composition(s.obs, ct, [samp], mask=keep)
    t = _feature_table(ct, comp, A, B, "composition", True)
    tabs.append(t[(t.mean_a >= min_frac) | (t.mean_b >= min_frac)])
    # niches / regions (any categorical niche-like column incl. derived ones), skipping columns that are
    # just relabelled copies of one already used (e.g. niche vs niche_id)
    seen = []
    for col, info in s.columns.items():
        if info.get("role") not in ("niche", "gate") or col == ct or col not in s.obs:
            continue
        if s.obs[col].nunique() > 50:
            continue
        codes = pd.factorize(s.obs[col])[0]
        if any(len(np.unique(codes * (o.max() + 1) + o)) == len(np.unique(codes)) == len(np.unique(o))
               for o in seen):
            continue
        seen.append(codes)
        comp = stats.composition(s.obs, col, [samp], mask=keep)
        t = _feature_table(col, comp, A, B, "niche", True)
        tabs.append(t[(t.mean_a >= min_frac) | (t.mean_b >= min_frac)])
    # neighbourhood structure from the card (log2 O/E per sample)
    p = s.card_dir / "tables" / "nhood_by_sample.parquet"
    if p.exists():
        nh = pd.read_parquet(p)
        nh = nh[nh["centre"] <= nh["neighbour"]]
        wide = nh.assign(f=nh["centre"] + " ~ " + nh["neighbour"]).pivot_table(
            index="sample", columns="f", values="log2_oe")
        tabs.append(_feature_table("neighbourhood log2 O/E", wide, A, B, "neighbourhood", False))
    # expression within each cell type (sample pseudobulk, panel gaps masked)
    mean, _, _ = stats.group_stats(s, [samp, ct], mask=keep, min_cells=20)
    for c in cell_types or sorted(mean.index.get_level_values(1).unique()):
        if c not in mean.index.get_level_values(1):
            continue
        pb = stats.mask_panel_gaps(s, mean.xs(c, level=1))
        t = _feature_table(c, pb, A, B, "expression", False)
        t["feature"] = c + ": " + t["feature"]
        tabs.append(t[(t.mean_a >= 0.05) | (t.mean_b >= 0.05)])
    out = pd.concat(tabs, ignore_index=True)
    out["n_a"], out["n_b"] = len(A), len(B)
    out["pan_celltype"] = _pan_celltype_flags(out)
    out = out.reindex(out["hedges_g"].abs().sort_values(ascending=False).index)
    return out.reset_index(drop=True)


def _pan_celltype_flags(df: pd.DataFrame, min_delta: float = 0.15, min_share: float = 0.6) -> np.ndarray:
    """Expression features whose gene shifts the same way in most cell types: typically spillover /
    ambient RNA from the population that really changes, not regulation in each cell type."""
    ex = df[df.kind == "expression"]
    gene = ex["feature"].str.split(": ", n=1).str[1]
    n_ct = ex["source"].nunique()
    flags = np.zeros(len(df), bool)
    if n_ct < 4:
        return flags
    for sign in (1, -1):
        hit = (sign * ex["effect"] >= min_delta).groupby(gene).sum()
        pan = set(hit.index[hit >= min_share * n_ct])
        flags[ex.index[gene.isin(pan) & (sign * ex["effect"] > 0)]] = True
    return flags


def separability(s: Store, a: str, b: str, column: str | None = None, cells: dict | None = None,
                 features: str = "expression", k: int = 15, max_cells_per_sample: int = 2000,
                 seed: int = 0) -> dict:
    """Can cells from `a` be told apart from cells from `b`? Leave-one-sample-out logistic regression
    (so the score reflects held-out animals, not memorised sections). Returns AUC, per-sample mean
    P(a) and the most informative features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    column = column or s.role("group")
    samp = s.role("sample")
    m = s.mask(cells) & s.mask({column: [a, b]})
    smp = s.obs[samp].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.flatnonzero(m & (smp == x)),
                                     min(max_cells_per_sample, int((m & (smp == x)).sum())), replace=False)
                          for x in np.unique(smp[m])])
    parts, names = [], []
    if features in ("expression", "both"):
        X = np.stack([s.gene(g)[idx] for g in s.genes], 1)
        gaps = s.card_dir / "tables" / "panel_gaps.parquet"
        if gaps.exists():  # genes missing from some panels would separate batches, not biology
            drop = set(pd.read_parquet(gaps).index)
            keep = [i for i, g in enumerate(s.genes) if g not in drop]
            X, gl = X[:, keep], [s.genes[i] for i in keep]
        else:
            gl = list(s.genes)
        parts.append(X); names += gl
    if features in ("neighbourhood", "both"):
        comp, cats = spatial.composition_matrix(s, s.role("cell_type"), k, cells=idx)
        parts.append(comp); names += [f"nbr:{c}" for c in cats]
    X = np.hstack(parts)
    y = (s.obs[column].astype(str).to_numpy()[idx] == str(a)).astype(int)
    groups = smp[idx]
    pred = np.zeros(len(idx))
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.1, max_iter=300).fit(sc.transform(X[tr]), y[tr])
        pred[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    auc = roc_auc_score(y, pred)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.1, max_iter=300).fit(sc.transform(X), y)
    coef = pd.Series(clf.coef_[0], index=names).sort_values()
    per = pd.DataFrame({"sample": groups, "label": np.where(y == 1, a, b), "p_a": pred}) \
        .groupby(["sample", "label"])["p_a"].mean().reset_index()
    return {"auc": float(auc), "n_cells": len(idx), "per_sample": per,
            f"higher_in_{a}": coef[::-1].head(12).round(3), f"higher_in_{b}": coef.head(12).round(3)}


# -- ligand-receptor ------------------------------------------------------------
def _lr_pairs(s: Store) -> pd.DataFrame:
    lr = resources.ligand_receptor(organism(s))
    panel = set(s.genes)
    ok = lr["ligand"].str.split("_").map(lambda x: set(x) <= panel) & \
        lr["receptor"].str.split("_").map(lambda x: set(x) <= panel)
    return lr[ok].reset_index(drop=True)


def ligand_receptor(s: Store, sender: str | None = None, receiver: str | None = None,
                    a: str | None = None, b: str | None = None, column: str | None = None,
                    min_frac: float = 0.1, n_top: int = 30) -> pd.DataFrame:
    """Ligand-receptor candidates between cell types, restricted to panel genes.

    score = mean(ligand in sender) x mean(receptor in receiver) (complex = min over subunits), kept when
    both are expressed in >= min_frac of cells. Each pair is annotated with the spatial co-localisation of
    sender and receiver (log2 O/E from the kNN graph) - communication needs proximity.
    With a/b (values of `column`), scores are computed per sample and compared (delta, Hedges g)."""
    ct, samp = s.role("cell_type"), s.role("sample")
    lr = _lr_pairs(s)
    if lr.empty:
        return pd.DataFrame()
    genes = sorted({g for x in pd.concat([lr.ligand, lr.receptor]) for g in x.split("_")})

    def subunit(df, cplx, how):
        cols = cplx.split("_")
        return df[cols].min(axis=1) if len(cols) > 1 else df[cols[0]]

    def scores(mask):
        mean, frac, _ = stats.group_stats(s, [ct], genes, mask=mask, min_cells=20)
        rows = []
        senders = [sender] if sender else list(mean.index)
        receivers = [receiver] if receiver else list(mean.index)
        for r in lr.itertuples():
            lm, lf = subunit(mean, r.ligand, "min"), subunit(frac, r.ligand, "min")
            rm, rf = subunit(mean, r.receptor, "min"), subunit(frac, r.receptor, "min")
            for sd in senders:
                if sd not in lm.index or lf[sd] < min_frac:
                    continue
                for rc in receivers:
                    if rc not in rm.index or rf[rc] < min_frac:
                        continue
                    rows.append((r.ligand, r.receptor, sd, rc, float(lm[sd] * rm[rc]), float(lf[sd]),
                                 float(rf[rc]), r.n_sources))
        return pd.DataFrame(rows, columns=["ligand", "receptor", "sender", "receiver", "score",
                                           "frac_ligand", "frac_receptor", "n_sources"])

    nh_p = s.card_dir / "tables" / "nhood_by_group.parquet"
    nh = pd.read_parquet(nh_p).groupby(["centre", "neighbour"])["log2_oe"].mean() if nh_p.exists() else None

    if a is None:
        out = scores(None).sort_values("score", ascending=False)
    else:
        column = column or s.role("group")
        sg = _sample_groups(s, column)
        per = []
        for sm in sg.index[sg.isin([str(a), str(b)])]:
            t = scores(s.mask({samp: sm}))
            t["sample"], t["grp"] = sm, sg[sm]
            per.append(t)
        per = pd.concat(per)
        key = ["ligand", "receptor", "sender", "receiver"]
        wide = per.pivot_table(index="sample", columns=key, values="score").fillna(0)
        A, B = list(sg.index[sg == str(a)]), list(sg.index[sg == str(b)])
        ma, mb, g, p = _effect(wide.reindex(A).to_numpy(), wide.reindex(B).to_numpy())
        out = pd.DataFrame(list(wide.columns), columns=key)
        out["score_a"], out["score_b"], out["delta"], out["hedges_g"], out["p"] = ma, mb, ma - mb, g, p
        out = out.reindex(out["delta"].abs().sort_values(ascending=False).index)
    if nh is not None and len(out):
        out["colocalisation_log2_oe"] = [nh.get((r.sender, r.receiver), np.nan) for r in out.itertuples()]
    return out.head(n_top).reset_index(drop=True)
