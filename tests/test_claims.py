import numpy as np

from spaceagent import claims, spatial, tools


def _verdict(s, **kw):
    spec = claims.make_spec(s, **kw)
    return claims.test_claim(s, "t", spec, record=False)["history"][-1]


def test_true_claims_are_robust(toy):
    _, s = toy
    h = _verdict(s, kind="expression", feature="GeneDE", a="treat", b="ctrl", cells={"cell_type": "T"})
    assert h["verdict"] == "robust" and h["summary"]["effect"] > 1
    t = {x["test"]: x for x in h["tests"]}
    assert t["separation"]["passed"] and t["spillover"]["passed"] and t["confident"]["passed"]
    assert t["permutation"]["detail"].endswith("0.167)")  # 2 vs 2 samples: 1/6 is the floor
    h = _verdict(s, kind="colocalisation", feature="B", a="treat", cells={"cell_type": "A"})
    assert h["verdict"] == "robust" and h["summary"]["effect"] > 0.5
    h = _verdict(s, kind="composition", feature="T", a="treat", b="ctrl", min_effect=0.3)
    assert h["verdict"] == "robust"


def test_false_claims_are_caught(toy):
    _, s = toy
    assert _verdict(s, kind="expression", feature="GeneDE", a="treat", b="ctrl", cells={"cell_type": "T"},
                    direction="down")["verdict"] == "refuted"
    assert _verdict(s, kind="expression", feature="Noise", a="treat", b="ctrl",
                    cells={"cell_type": "A"})["verdict"] == "not_supported"
    assert _verdict(s, kind="colocalisation", feature="T", a="treat", cells={"cell_type": "A"},
                    )["verdict"] == "refuted"


def test_spillover_is_caught(toy):
    _, s = toy
    # contamination: A cells pick up GeneDE from neighbouring T cells (only at the A|T border)
    idx, _ = spatial.neighbours(s, 10)
    is_t = (s.obs.cell_type == "T").to_numpy()
    de = s.gene("GeneDE")
    spill = 5 * (de[idx] * is_t[idx]).mean(1) + np.where(is_t, de, 0)
    s.add_column("Spill", spill.astype(np.float32), "toy contamination", role="score")
    h = _verdict(s, kind="expression", feature="Spill", a="treat", b="ctrl", cells={"cell_type": "A"},
                 min_effect=0.02)
    t = {x["test"]: x for x in h["tests"]}
    assert h["summary"]["effect"] > 0.02 and t["direction"]["passed"]
    assert t["spillover"]["passed"] is False and "T" in t["spillover"]["detail"]
    assert h["verdict"] == "fragile"


def test_ledger_and_recheck(toy):
    _, s = toy
    s.add_column("Spill2", s.gene("GeneDE"), "copy", role="score")
    md, fig = tools.test_claim(s, "GeneDE-like score up in treated T cells", "expression", "Spill2",
                               "treat", "ctrl", cells={"cell_type": "T"})
    assert "ROBUST" in md and fig.exists()
    cid = md.split()[1]
    assert cid in tools.list_claims(s) and "recheck" not in tools.list_claims(s)
    assert "nothing re-run" in tools.recheck_claims(s)
    # the score is redefined (e.g. annotation fix) -> claim goes stale and flips on recheck
    s.add_column("Spill2", s.gene("Noise"), "now noise", role="score")
    assert "recheck" in tools.list_claims(s)
    out = tools.recheck_claims(s)
    assert "⚠" in out and "not_supported" in out
    md, _ = tools.show_claim(s, cid)
    assert "History: robust" in md


def test_ambient_shift_is_caught(toy):
    _, s = toy
    # a T-cell gene that leaks into every cell type in treated samples (ambient RNA)
    treat = (s.obs.group == "treat").to_numpy()
    is_t = (s.obs.cell_type == "T").to_numpy()
    rng = np.random.default_rng(1)
    amb = np.where(is_t, 3.0, 0.2) + 0.6 * treat + rng.normal(0, 0.05, s.n_cells)
    s.add_column("Amb", amb.astype(np.float32), "toy ambient", role="score")
    h = _verdict(s, kind="expression", feature="Amb", a="treat", b="ctrl", cells={"cell_type": "A"})
    t = {x["test"]: x for x in h["tests"]}
    assert t["specificity"]["passed"] is False and t["specificity"]["core"] and "ambient" in t["specificity"]["detail"]
    assert h["verdict"] == "fragile"
    # the same shift in the main expresser is not penalised for it
    h = _verdict(s, kind="expression", feature="Amb", a="treat", b="ctrl", cells={"cell_type": "T"})
    assert not {x["test"]: x for x in h["tests"]}["specificity"]["core"]
