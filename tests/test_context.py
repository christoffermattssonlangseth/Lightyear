import numpy as np

from spaceagent import context


def test_context_model_recovers_planted_effect(toy):
    _, s = toy
    perf = context.fit(s, "toy_ctx", max_cells=4000, log=lambda *_: None)
    p = perf.set_index(["cell_type", "gene"])
    assert p.loc[("A", "GeneCtx"), "r2_context"] > 0.3
    assert p.loc[("A", "Noise"), "r2_context"] < 0.05
    assert p.loc[("T", "GeneDE"), "r2_cell"] > 0.3 > p.loc[("T", "GeneDE"), "r2_context"]  # condition, not context
    # removing T neighbours lowers GeneCtx in A cells; removing B does little
    dr = context.drivers(s, "toy_ctx", "A", ["GeneCtx"])
    assert dr.loc["T", "GeneCtx"] < -0.3 and abs(dr.loc["B", "GeneCtx"]) < 0.1
    r = context.perturb(s, "toy_ctx", {"cell_type": "A"}, ["GeneCtx", "Noise"], remove=["T"])
    assert (r["delta_in_changed_cells"]["GeneCtx"] < -0.3).all()
    assert (r["delta"]["Noise"].abs() < 0.05).all()
    assert 0 < r["frac_cells_changed"].max() < 0.5 and r["support_actual"] > 0.9
    # replacing all B neighbours of A cells with T makes neighbourhoods never seen in training
    r = context.perturb(s, "toy_ctx", {"cell_type": "A"}, ["GeneCtx"], replace={"B": "T"})
    assert r["support"] < 0.5
    # GeneDE in T: the treat effect is intrinsic, not neighbourhood
    ex = context.explain(s, "toy_ctx", "treat", "ctrl", {"cell_type": "T"}, ["GeneDE"])
    assert ex.loc["GeneDE", "observed_delta"] > 1 and abs(ex.loc["GeneDE", "share_neighbourhood"]) < 0.2
    assert abs(ex.loc["GeneDE", "condition_part"] - ex.loc["GeneDE", "observed_delta"]) < 0.2


def test_model_claims(toy):
    from spaceagent import claims
    _, s = toy
    if not (s.path / "models" / "toy_ctx").exists():
        context.fit(s, "toy_ctx", max_cells=4000, log=lambda *_: None)

    def run(**kw):
        h = claims.test_claim(s, "t", claims.make_spec(s, **kw), record=False)["history"][-1]
        return h["verdict"], {x["test"]: x for x in h["tests"]}

    v, t = run(kind="perturbation", feature="GeneCtx", a="treat", cells={"cell_type": "A"}, remove=["T"],
               direction="down", model="toy_ctx")
    # planted effect; T cells do not express GeneCtx themselves, so it is not spillover from them
    assert v == "robust" and t["model_fit"]["passed"] and t["not_spillover"]["passed"] and t["support"]["passed"]
    v, t = run(kind="perturbation", feature="GeneCtx", a="treat", cells={"cell_type": "A"}, replace={"B": "T"},
               direction="up", model="toy_ctx")
    assert t["support"]["passed"] is False and v == "fragile"
    # context alone does not predict the treat effect on GeneDE in T cells
    v, _ = run(kind="context", feature="GeneDE", a="treat", b="ctrl", cells={"cell_type": "T"}, model="toy_ctx")
    assert v in ("not_supported", "refuted")
