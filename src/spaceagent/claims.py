"""Claims ledger: findings as re-runnable, falsifiable statements.

A claim is a structured spec (what is measured, in which cells, between which groups, in which
direction). Testing it reduces the data to one value per sample, then tries to break the claim:

- direction   observed effect has the claimed sign and a meaningful size
- replicates  >= 2 samples per group with data
- separation  share of (a, b) sample pairs ordered as claimed (1.0 = every case beyond every control)
- leave_one_out  sign and >= half the size survive dropping any single sample
- permutation exact sample-label permutation p, with the smallest p the design allows
- spillover   (expression) effect survives in cells with no neighbour of the main source cell type
- niche       (expression) effect survives within niches, i.e. is not a niche-composition shift
- confident   (expression, co-localisation) effect survives in cells with strong identity markers
- specificity (caveat only) the same shift is not equally present in all other cells

Verdicts: refuted / not_supported / underpowered / fragile / robust. Each test run is stored in
<store>/claims.json with a fingerprint of the data it used, so claims are re-checked when the
annotation, derived columns or expression change.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import json

import numpy as np
import pandas as pd

from . import spatial, stats
from .store import Store

KINDS = ("expression", "composition", "colocalisation", "context", "perturbation")
MODEL_KINDS = ("context", "perturbation")
MIN_EFFECT = {"expression": 0.1, "composition": 0.5, "colocalisation": 0.5, "context": 0.1, "perturbation": 0.1}
MIN_CELLS = 20
CORE = ("direction", "replicates", "separation", "leave_one_out", "permutation", "spillover", "niche",
        "confident", "model_fit", "support", "not_spillover")  # specificity is core only when the population is not the main expresser


# -- spec -----------------------------------------------------------------------
def make_spec(s: Store, kind: str, feature: str, a: str, b: str | None = None, cells: dict | None = None,
              direction: str = "up", column: str | None = None, group_column: str | None = None,
              k: int = 10, min_effect: float | None = None, stratify: str | None = None,
              model: str | None = None, remove: list[str] | None = None, replace: dict | None = None) -> dict:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if direction not in ("up", "down"):
        raise ValueError("direction must be 'up' or 'down' (a relative to b, or relative to expected)")
    group_column = group_column or s.role("group")
    column = column or s.role("cell_type")
    if b is None and kind not in ("colocalisation", "perturbation"):
        raise ValueError("a one-group claim (b=None) is only meaningful for colocalisation (vs expected) "
                         "and perturbation (vs no change)")
    extra = {}
    if kind in MODEL_KINDS:
        from .context import ContextModel
        if not model:
            raise ValueError(f"kind {kind!r} needs model= (a fitted context model)")
        cm = ContextModel(s, model)
        cm.single_type(cm.cells_of(cells))
        feature = cm.gene_cols([feature])[0]
        if kind == "perturbation" and not (remove or replace):
            raise ValueError("perturbation needs remove=[...] and/or replace={from: to}")
        extra = {"model": model} if kind == "context" else {"model": model, "remove": remove or [],
                                                              "replace": replace or {}}
    elif kind == "expression":
        feature = s.resolve_gene(feature)
    elif feature not in set(s.obs[column].astype(str).unique()):
        raise ValueError(f"{feature!r} is not a value of {column!r}")
    groups = set(s.obs[group_column].astype(str).unique())
    for g in (a, b):
        if g is not None and str(g) not in groups:
            raise ValueError(f"{g!r} is not a value of {group_column!r}")
    s.mask(cells)  # validates filters
    return {"kind": kind, "feature": feature, "a": a, "b": b, "cells": cells or {}, "direction": direction,
            "column": column, "group_column": group_column, "k": int(k),
            "min_effect": float(min_effect if min_effect is not None else MIN_EFFECT[kind]),
            "stratify": stratify, **extra}


def describe(spec: dict) -> str:
    cells = ", ".join(f"{k}={v}" for k, v in spec["cells"].items()) or "all cells"
    arrow = "higher" if spec["direction"] == "up" else "lower"
    ref = spec["b"] if spec["b"] is not None else "expected"
    what = {"expression": f"{spec['feature']} expression in [{cells}]",
            "composition": f"share of {spec['feature']} among [{cells}]",
            "colocalisation": f"{spec['feature']} among the {spec['k']} nearest neighbours of [{cells}] (log2 O/E)",
            "context": f"{spec['feature']} in [{cells}] as predicted from neighbourhood alone, condition held at "
                       f"{spec['b']} (model {spec.get('model')})",
            "perturbation": f"change in predicted {spec['feature']} in [{cells}] when neighbours are edited "
                            f"(remove {spec.get('remove')}, replace {spec.get('replace')}; model {spec.get('model')})"}
    if spec["kind"] == "perturbation" and spec["b"] is None:
        return f"{what['perturbation']} is {'positive' if spec['direction'] == 'up' else 'negative'} in {spec['a']}"
    return f"{what[spec['kind']]} is {arrow} in {spec['a']} than in {ref}"


# -- per-sample measurement -----------------------------------------------------
def _sample_codes(s: Store):
    return stats.codes_for(s.obs, [s.role("sample")])


def per_sample(s: Store, spec: dict, extra: np.ndarray | None = None) -> pd.Series:
    """One value per sample (NaN where too few cells / gene not on that sample's panel).
    expression: mean log-expression; composition: log2 share; colocalisation: log2 O/E;
    context: mean predicted log-expression from neighbourhood alone; perturbation: mean predicted change
    in the cells whose neighbourhood the edit changes."""
    codes, samples = _sample_codes(s)
    groups = [spec["a"]] + ([spec["b"]] if spec["b"] is not None else [])
    m = s.mask(spec["cells"]) & s.mask({spec["group_column"]: groups})
    if extra is not None:
        m &= extra
    S = len(samples)
    n = np.bincount(codes[m], minlength=S).astype(float)
    with np.errstate(all="ignore"):
        if spec["kind"] in MODEL_KINDS:
            val, n = _model_per_sample(s, spec, m, codes, S)
        elif spec["kind"] == "expression":
            v = np.bincount(codes[m], weights=s.gene(spec["feature"])[m], minlength=S) / n
            val = pd.DataFrame({spec["feature"]: v}, index=samples.astype(str))
            val = stats.mask_panel_gaps(s, val)[spec["feature"]].to_numpy()
        else:
            is_f = (s.obs[spec["column"]].astype(str) == spec["feature"]).to_numpy()
            if spec["kind"] == "composition":
                hit = np.bincount(codes[m & is_f], minlength=S)
                val = np.log2((hit + 0.5) / (n + 0.5))
            else:
                idx, _ = spatial.neighbours(s, spec["k"])
                cells = np.flatnonzero(m)
                frac_nb = is_f[idx[cells]].mean(1)
                obs_ = np.bincount(codes[cells], weights=frac_nb, minlength=S) / n
                exp_ = np.bincount(codes, weights=is_f, minlength=S) / np.bincount(codes, minlength=S)
                val = np.log2((obs_ + 1e-3) / (exp_ + 1e-3))
    out = pd.Series(val, index=samples.astype(str), name="value")
    out[n < MIN_CELLS] = np.nan
    sg = s.obs.groupby(s.role("sample"), observed=True)[spec["group_column"]].first().astype(str)
    return out[sg.reindex(out.index).isin([str(g) for g in groups]).to_numpy()]


def _model_per_sample(s: Store, spec: dict, m: np.ndarray, codes: np.ndarray, S: int):
    from .context import ContextModel, _comps, counterfactual
    cm = ContextModel(s, spec["model"])
    cells = np.flatnonzero(m)
    if spec["kind"] == "context":  # neighbourhood part only: every cell predicted at the reference condition
        v = cm.predict(cells, [spec["feature"]], group=spec["b"])[:, 0]
    else:
        comps, cats = _comps(s, cells)
        cf = counterfactual(comps, cats, spec["remove"], spec["replace"])
        ok = ~np.isnan(np.hstack(cf)).any(1) & (np.abs(np.hstack(cf) - np.hstack(comps)).sum(1) > 1e-6)
        cells = cells[ok]
        v = (cm.predict(cells, [spec["feature"]], [c[ok] for c in cf]) -
             cm.predict(cells, [spec["feature"]], [c[ok] for c in comps]))[:, 0]
    n = np.bincount(codes[cells], minlength=S).astype(float)
    with np.errstate(all="ignore"):
        return np.bincount(codes[cells], weights=v, minlength=S) / n, n


def _model_tests(s: Store, spec: dict, eff: float, add):
    from .context import SPILL_LOG, ContextModel, _comps, counterfactual, drivers, source_gap
    cm = ContextModel(s, spec["model"])
    pop_idx = cm.cells_of(spec["cells"])
    ct = cm.single_type(pop_idx)
    g = spec["feature"]
    pf = cm.perf[(cm.perf.cell_type == ct) & (cm.perf.gene == g)].iloc[0]
    add("model_fit", bool(pf.r2_context >= 0.02 and pf.r2_sample >= 0.5),
        f"held-out fit for {g} in {ct}: r2 beyond condition {pf.r2_context:.3f} (need ≥ 0.02), sample-level "
        f"r2 {pf.r2_sample:.3f} (need ≥ 0.5)")
    if spec["kind"] == "perturbation":
        movers = list(spec["remove"]) + list(spec["replace"].keys())
        gm = s.mask({spec["group_column"]: spec["a"]})
        cells = np.intersect1d(pop_idx, np.flatnonzero(gm))
        comps, cats = _comps(s, cells)
        cf = counterfactual(comps, cats, spec["remove"], spec["replace"])
        ok = ~np.isnan(np.hstack(cf)).any(1) & (np.abs(np.hstack(cf) - np.hstack(comps)).sum(1) > 1e-6)
        sup = cm.support(cells[ok], [c[ok] for c in cf]) if ok.any() else 0.0
        add("support", sup >= 0.8, f"{sup:.0%} of edited neighbourhoods in {spec['a']} resemble training data")
    else:
        dr = drivers(s, spec["model"], ct, [g])[g]
        movers = [dr.idxmin()]
    gaps = {x: source_gap(cm, ct, x, g) for x in movers if x != ct}
    worst = max(gaps, key=gaps.get) if gaps else None
    add("not_spillover", not (worst and gaps[worst] >= SPILL_LOG),
        (f"{worst} expresses {g} {gaps[worst]:+.2f} log units vs {ct}" if worst else "driver is the cell type itself")
        + (" - effect indistinguishable from transcripts picked up from it" if worst and gaps[worst] >= SPILL_LOG
           else ""))


def _groups(s: Store, spec: dict, idx) -> pd.Series:
    sg = s.obs.groupby(s.role("sample"), observed=True)[spec["group_column"]].first().astype(str)
    return sg.reindex(idx)


def _effect(v: pd.Series, g: pd.Series, spec: dict) -> float:
    ok = v.notna()
    va = v[ok & (g == str(spec["a"]))]
    if spec["b"] is None:
        return float(va.mean()) if len(va) else np.nan
    vb = v[ok & (g == str(spec["b"]))]
    return float(va.mean() - vb.mean()) if len(va) and len(vb) else np.nan


def _signed(x: float, spec: dict) -> float:
    return x if spec["direction"] == "up" else -x


# -- tests ------------------------------------------------------------------------
def _survives(base: float, x: float, spec: dict, share: float = 0.5) -> bool:
    return bool(np.isfinite(x) and _signed(x, spec) >= share * _signed(base, spec))


def _permutation(v: pd.Series, g: pd.Series, spec: dict, obs: float, max_perm: int = 5000):
    v = v.dropna()
    g = g.reindex(v.index)
    x = v.to_numpy()
    if spec["b"] is None:  # sign flips around 0
        n = len(x)
        flips = np.array(list(itertools.product([1, -1], repeat=n))) if n <= 12 else \
            np.random.default_rng(0).choice([1, -1], (max_perm, n))
        null = (flips * x).mean(1)
    else:
        isa = (g == str(spec["a"])).to_numpy()
        na, n = int(isa.sum()), len(x)
        combos = list(itertools.combinations(range(n), na))
        if len(combos) > max_perm:
            rng = np.random.default_rng(0)
            combos = [tuple(rng.choice(n, na, replace=False)) for _ in range(max_perm)]
        null = []
        for c in combos:
            sel = np.zeros(n, bool)
            sel[list(c)] = True
            null.append(x[sel].mean() - x[~sel].mean())
        null = np.array(null)
    sn = null if spec["direction"] == "up" else -null
    p = float(np.mean(sn >= _signed(obs, spec) - 1e-12))
    return p, 1.0 / len(null)


def _shared_shift(s: Store, spec: dict, eff: float, pop: np.ndarray, g: pd.Series):
    """Share of other cell types (enough cells in every sample) whose same contrast is >= half the claimed
    effect in the claimed direction; plus the cell type with the highest mean expression."""
    groups = [spec["a"], spec["b"]]
    m = s.mask({spec["group_column"]: groups}) & ~pop
    mean, _, n = stats.group_stats(s, [s.role("sample"), s.role("cell_type")], [spec["feature"]], mask=m,
                                   min_cells=MIN_CELLS)
    wide = mean.iloc[:, 0].unstack(1)
    gap = stats.mask_panel_gaps(s, pd.DataFrame({spec["feature"]: 0.0}, index=wide.index))[spec["feature"]].isna()
    wide[gap.to_numpy()] = np.nan
    hits, n_types = 0, 0
    for ct in wide.columns:
        v = wide[ct]
        e = _effect(v, g.reindex(v.index), spec)
        if np.isfinite(e) and v.notna().sum() >= 4:
            n_types += 1
            hits += _signed(e, spec) >= 0.5 * _signed(eff, spec)
    allm, _, _ = stats.group_stats(s, [s.role("cell_type")], [spec["feature"]],
                                   mask=s.mask({spec["group_column"]: groups}), min_cells=MIN_CELLS)
    top = str(allm.iloc[:, 0].idxmax())
    return (hits / n_types if n_types else 0.0), n_types, top


def _niche_column(s: Store, spec: dict) -> str | None:
    if spec.get("stratify"):
        return spec["stratify"]
    for col, info in s.columns.items():
        if info.get("role") == "niche" and col in s.obs and col != spec["column"] \
                and 2 <= s.obs[col].nunique() <= 30 and not any(k == col for k in spec["cells"]):
            return col
    return None


def _population_type(s: Store, spec: dict) -> str | None:
    v = spec["cells"].get(spec["column"])
    if isinstance(v, list) and len(v) == 1:
        v = v[0]
    return v if isinstance(v, str) else None


def _identity_markers(s: Store, cell_type: str, exclude: str, n: int = 5) -> list[str]:
    p = s.card_dir / "tables" / "markers.parquet"
    if p.exists():
        mk = pd.read_parquet(p)
        mk = mk[(mk["cell_type"] == cell_type) & (mk["gene"] != exclude)]
        if len(mk):
            return mk.sort_values("lfc_vs_rest", ascending=False)["gene"].head(n).tolist()
    mean, frac, _ = stats.group_stats(s, [s.role("cell_type")], min_cells=MIN_CELLS)
    mk = stats.markers(mean, frac, n_top=n + 1).get(cell_type)
    return [] if mk is None else [g for g in mk.index if g != exclude][:n]


def run_tests(s: Store, spec: dict) -> dict:
    v = per_sample(s, spec)
    g = _groups(s, spec, v.index)
    eff = _effect(v, g, spec)
    me = spec["min_effect"]
    tests = []

    def add(name, passed, detail, core=None):
        tests.append({"test": name, "passed": passed, "detail": detail,
                      "core": name in CORE if core is None else core})

    # direction / size
    if not np.isfinite(eff):
        add("direction", None, "no measurable effect (too few cells or gene missing)")
    else:
        add("direction", _signed(eff, spec) >= me,
            f"effect {eff:+.3f} (claimed {spec['direction']}, minimum |effect| {me})")
    # replicates
    na = int((v.notna() & (g == str(spec["a"]))).sum())
    nb = int((v.notna() & (g == str(spec["b"]))).sum()) if spec["b"] is not None else None
    add("replicates", na >= 2 and (nb is None or nb >= 2),
        f"{na} samples in {spec['a']}" + (f", {nb} in {spec['b']}" if nb is not None else ""))
    # separation
    va = v[v.notna() & (g == str(spec["a"]))].to_numpy()
    if spec["b"] is None:
        sep = float(np.mean(_signed(va, spec) > 0)) if len(va) else np.nan
        sep_txt = f"{sep:.2f} of {spec['a']} samples on the claimed side of expected"
    else:
        vb = v[v.notna() & (g == str(spec["b"]))].to_numpy()
        d = _signed(va[:, None] - vb[None, :], spec)
        sep = float(np.mean(d > 0)) if d.size else np.nan
        sep_txt = f"{sep:.2f} of {spec['a']}×{spec['b']} sample pairs ordered as claimed"
    add("separation", bool(sep == 1.0) if np.isfinite(sep) else None, sep_txt)
    # leave one out
    if np.isfinite(eff) and v.notna().sum() > 2:
        loo = {smp: _effect(v.drop(smp), g.drop(smp), spec) for smp in v.dropna().index}
        worst = min(loo, key=lambda k: _signed(loo[k], spec) if np.isfinite(loo[k]) else -np.inf)
        add("leave_one_out", all(_survives(eff, x, spec) for x in loo.values()),
            f"weakest without {worst}: {loo[worst]:+.3f}")
    else:
        add("leave_one_out", None, "too few samples")
    # permutation
    if np.isfinite(eff) and v.notna().sum() >= 3:
        p, pmin = _permutation(v, g, spec, eff)
        add("permutation", p <= max(0.05, pmin + 1e-9),
            f"p={p:.3g} (smallest possible with this design: {pmin:.3g})")
    else:
        p, pmin = np.nan, np.nan
        add("permutation", None, "too few samples")

    if spec["kind"] == "expression" and np.isfinite(eff):
        ct_col = s.role("cell_type")
        pop = s.mask(spec["cells"])
        # spillover: drop cells touching the main non-population source of the gene
        expr = s.gene(spec["feature"])
        grp_a = s.mask({spec["group_column"]: spec["a"]})
        ctc, cts = stats.codes_for(s.obs, [ct_col])
        m_out = grp_a & ~pop & (ctc >= 0)
        if m_out.any():
            means = np.bincount(ctc[m_out], weights=expr[m_out], minlength=len(cts)) / \
                np.maximum(np.bincount(ctc[m_out], minlength=len(cts)), 1)
            counts = np.bincount(ctc[m_out], minlength=len(cts))
            means[counts < 100] = -np.inf
            src = int(np.argmax(means))
            idx, _ = spatial.neighbours(s, min(spec["k"], 10))
            is_src = (ctc == src) & ~pop
            isolated = ~is_src[idx].any(1)
            e2 = _effect(per_sample(s, spec, isolated), g, spec)
            add("spillover", _survives(eff, e2, spec) if np.isfinite(e2) else None,
                f"main other source {cts[src]} (mean {means[src]:.2f} in {spec['a']}); "
                f"effect in cells with no {cts[src]} among nearest {min(spec['k'], 10)}: {e2:+.3f}")
        # niche stratification
        nc = _niche_column(s, spec)
        if nc:
            parts, weights = [], []
            for lev in s.obs[nc].astype(str).unique():
                ex = s.mask({nc: lev})
                vv = per_sample(s, spec, ex)
                e = _effect(vv, g.reindex(vv.index), spec)
                if np.isfinite(e):
                    parts.append(e)
                    weights.append(int((pop & ex).sum()))
            e2 = float(np.average(parts, weights=weights)) if parts else np.nan
            add("niche", _survives(eff, e2, spec) if np.isfinite(e2) else None,
                f"cell-weighted effect within `{nc}` strata ({len(parts)} usable): {e2:+.3f}")
        # specificity: does the gene shift the same way in most other cell types? If so and the population
        # is not the gene's main expresser, the shift is the signature of ambient RNA / segmentation spillover.
        shared, n_types, top_type = _shared_shift(s, spec, eff, pop, g)
        if n_types:
            pop_ct = _population_type(s, spec)
            ambient = top_type != pop_ct
            add("specificity", shared < 0.6,
                f"{shared:.0%} of {n_types} other cell types shift the same way (≥ half the effect); "
                f"highest expresser: {top_type}" + (" - likely ambient/spillover" if shared >= 0.6 and ambient else ""),
                core=ambient)

    if spec["kind"] in MODEL_KINDS and np.isfinite(eff):
        _model_tests(s, spec, eff, add)

    if spec["kind"] in ("expression", "colocalisation") and np.isfinite(eff):
        ct = _population_type(s, spec)
        if ct:
            genes = _identity_markers(s, ct, spec["feature"])
            if genes:
                sc = np.mean([s.gene(x) for x in genes], 0)
                pop = s.mask(spec["cells"])
                codes, _ = _sample_codes(s)
                conf = np.zeros(s.n_cells, bool)
                for c in np.unique(codes[pop]):
                    cc = pop & (codes == c)
                    conf[cc] = sc[cc] >= np.median(sc[cc])
                e2 = _effect(per_sample(s, spec, conf), g, spec)
                add("confident", _survives(eff, e2, spec) if np.isfinite(e2) else None,
                    f"effect in the half of {ct} with the strongest identity markers "
                    f"({', '.join(genes)}): {e2:+.3f}")

    verdict = _verdict(tests, eff, spec)
    a_vals = v[g == str(spec["a"])]
    summary = {"effect": eff, "mean_a": float(a_vals.mean()),
               "mean_b": float(v[g == str(spec["b"])].mean()) if spec["b"] is not None else None,
               "n_a": na, "n_b": nb, "perm_p": p, "perm_p_min": pmin}
    return {"verdict": verdict, "summary": summary, "tests": tests,
            "per_sample": {k: (None if pd.isna(x) else round(float(x), 4)) for k, x in v.items()},
            "groups": g.to_dict()}


def _verdict(tests: list[dict], eff: float, spec: dict) -> str:
    t = {x["test"]: x["passed"] for x in tests}
    if t.get("direction") is None:
        return "not_supported"
    if t["direction"] is False:  # wrong sign beyond the minimum effect = refuted, otherwise too small
        return "refuted" if _signed(eff, spec) <= -spec["min_effect"] else "not_supported"
    if t.get("replicates") is False:
        return "underpowered"
    if any(x["passed"] is False for x in tests if x["core"]):
        return "fragile"
    return "robust"


# -- ledger -------------------------------------------------------------------------
def _ledger_path(s: Store):
    return s.path / "claims.json"


def load_ledger(s: Store) -> list[dict]:
    p = _ledger_path(s)
    return json.loads(p.read_text()) if p.exists() else []


def _save_ledger(s: Store, claims: list[dict]):
    _ledger_path(s).write_text(json.dumps(claims, indent=1, default=float))


def fingerprint(s: Store, spec: dict) -> str:
    """Hash of everything the claim reads: the columns it uses (values per cell) and the expression file."""
    cols = {s.role("sample"), spec["group_column"], spec["column"], s.role("cell_type"), *spec["cells"],
            spec["feature"]}  # feature counts when it is a derived score
    nc = _niche_column(s, spec)
    if nc:
        cols.add(nc)
    h = hashlib.sha1(json.dumps(spec, sort_keys=True, default=str).encode())
    if spec.get("model"):
        w = s.path / "models" / spec["model"] / "weights.npz"
        h.update(f"{w.stat().st_mtime if w.exists() else 'missing'}".encode())
    for c in sorted(x for x in cols if x and x in s.obs.columns):
        h.update(c.encode())
        h.update(pd.util.hash_pandas_object(s.obs[c].astype(str), index=False).to_numpy().tobytes())
    e = s.path / "expr.npy"
    h.update(f"{e.stat().st_size}:{int(e.stat().st_mtime)}".encode())
    return h.hexdigest()[:16]


def test_claim(s: Store, statement: str, spec: dict, record: bool = True) -> dict:
    res = run_tests(s, spec)
    entry = {"run_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "fingerprint": fingerprint(s, spec),
             **res}
    claim = {"id": None, "statement": statement, "spec": spec, "history": [entry]}
    if record:
        claims = load_ledger(s)
        same = [c for c in claims if c["spec"] == spec]
        if same:
            claim = same[0]
            claim["statement"] = statement
            claim["history"].append(entry)
        else:
            claim["id"] = f"C{len(claims) + 1:03d}"
            claims.append(claim)
        _save_ledger(s, claims)
    return claim


def recheck(s: Store, ids: list[str] | None = None, force: bool = False) -> list[dict]:
    """Re-run claims whose data fingerprint changed (or all, with force). Returns change records."""
    claims = load_ledger(s)
    changes = []
    for c in claims:
        if ids and c["id"] not in ids:
            continue
        last = c["history"][-1]
        try:
            fp = fingerprint(s, c["spec"])
        except Exception as e:  # a column the claim used is gone
            changes.append({"id": c["id"], "statement": c["statement"], "before": last["verdict"],
                            "after": "error", "note": str(e)})
            continue
        if fp == last["fingerprint"] and not force:
            continue
        try:
            res = run_tests(s, c["spec"])
        except Exception as e:
            changes.append({"id": c["id"], "statement": c["statement"], "before": last["verdict"],
                            "after": "error", "note": str(e)})
            continue
        c["history"].append({"run_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "fingerprint": fp, **res})
        changes.append({"id": c["id"], "statement": c["statement"], "before": last["verdict"],
                        "after": res["verdict"], "note": "data changed" if fp != last["fingerprint"] else "forced"})
    _save_ledger(s, claims)
    return changes
