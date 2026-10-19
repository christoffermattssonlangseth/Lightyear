import numpy as np
import pandas as pd
import pytest

from lightyear import analysis, resources, run, spatial, tools


@pytest.fixture(autouse=True)
def offline_resources(monkeypatch):
    db = pd.DataFrame({"source": "x", "cell_type": ["Alpha cells"] * 2 + ["Tee cells"] * 2 + ["Other"] * 2,
                       "tissue": "toy", "gene": ["MarkA", "Noise", "MarkT", "GeneDE", "MarkB", "GeneGap"]})
    lr = pd.DataFrame({"ligand": ["GeneDE", "MarkB"], "receptor": ["MarkA", "Absent"], "n_sources": [3, 1]})
    monkeypatch.setattr(resources, "marker_db", lambda organism="mouse": db)
    monkeypatch.setattr(resources, "ligand_receptor", lambda organism="mouse": lr)


def test_knn_graph(toy):
    _, s = toy
    idx, dist = s.knn
    smp = s.obs["sample_id"].astype(str).to_numpy()
    assert idx.shape == (s.n_cells, spatial.K_STORED)
    assert (smp[np.asarray(idx[:, :10])] == smp[:, None]).all()  # never crosses sections
    assert (np.diff(np.asarray(dist[:, :10], float), axis=1) >= -1e-2).all()  # nearest first


def test_derived_columns_and_filters(toy):
    _, s = toy
    out = tools.gate(s, "de_pos_T", "GeneDE", 1.0, within={"cell_type": "T"})
    assert "de_pos_T" in s.obs and s.obs["de_pos_T"].sum() > 0
    assert not s.obs.loc[s.obs.cell_type != "T", "de_pos_T"].any()
    assert "created" in out
    tools.score_genes(s, "t_score", ["MarkT", "GeneDE"])
    assert s.obs.groupby("cell_type", observed=True)["t_score"].mean().idxmax() == "T"
    assert "t_score mean" in tools.expression(s, ["t_score"], ["cell_type"])  # scores act like genes
    assert s.mask({"GeneDE": {">": 1}}).sum() == (s.gene("GeneDE") > 1).sum()
    assert s.mask({"cell_type": {"not": ["T"]}}).sum() == (s.obs.cell_type != "T").sum()
    assert "de_pos_T" in tools.describe_structures(s)


def test_niches_recover_layout(toy):
    _, s = toy
    r = spatial.compute_niches(s, "toy_niche", n_niches=2, k=10)
    comp = r["composition"]
    # one niche is the A+B half, the other the T half
    assert comp["T"].max() > 0.9 and comp["T"].min() < 0.1
    e = spatial.compute_niches(s, "toy_uta", n_niches=2, features="expression", k=10)
    lab = s.obs["toy_uta"].astype(str)
    agree = pd.crosstab(lab, s.obs["toy_niche"].astype(str)).max(axis=1).sum() / len(lab)
    assert agree > 0.9
    assert "toy_niche" in tools.composition(s, "toy_niche", ["sample_id"], summarize_by_group="group")


def test_neighbour_stats(toy):
    _, s = toy
    nb = spatial.neighbourhood_of(s, {"cell_type": "A"}, k=10, by="group")
    assert (nb["frac"]["T"] < 0.05).all() and (nb["log2_vs_section"]["B"] > 0).all()
    z = spatial.nhood_enrichment(s, ["s1"], k=10, n_perm=20)
    zz = z.set_index(["centre", "neighbour"])["z"]
    assert zz[("A", "B")] > 2 and zz[("A", "T")] < -2
    co = spatial.co_occurrence(s, {"cell_type": "A"}, {"cell_type": "T"}, [20, 600], samples=["s1"])
    assert co.set_index("distance")["ratio"][20] < co.set_index("distance")["ratio"][600]
    mi = spatial.morans_i(s, ["MarkT", "Noise"], ["s1"])
    assert mi.loc["MarkT", "s1"] > 0.3 > abs(mi.loc["Noise", "s1"])
    gr = spatial.gradient(s, "MarkB", {"cell_type": "T"}, {"cell_type": "B"})
    assert gr.notna().any().any()


def test_case_control_and_separability(toy):
    _, s = toy
    cc = analysis.case_control(s, "treat", "ctrl")
    top_expr = cc[cc.kind == "expression"].iloc[0]
    assert top_expr.feature == "T: GeneDE" and top_expr.effect > 1
    t_comp = cc[(cc.kind == "composition") & (cc.feature == "T")].iloc[0]
    assert t_comp.effect > 0.3
    assert "Strongest separating" in tools.case_control(s, "treat", "ctrl")
    sep_t = analysis.separability(s, "treat", "ctrl", cells={"cell_type": "T"}, max_cells_per_sample=300)
    sep_a = analysis.separability(s, "treat", "ctrl", cells={"cell_type": "A"}, max_cells_per_sample=300)
    assert sep_t["auc"] > 0.9 and sep_a["auc"] < 0.7
    assert "GeneDE" in sep_t["higher_in_treat"].index[:2]


def test_annotation_and_lr(toy):
    _, s = toy
    ann = analysis.annotate_suggest(s, "cell_type")
    assert "Tee cells" in ann.loc["T", "suggestions"]
    assert "MarkT" in tools.markers(s, "cell_type").split("**T**")[1].splitlines()[0]
    lr = analysis.ligand_receptor(s)
    assert list(lr.iloc[0][["ligand", "receptor", "sender", "receiver"]]) == ["GeneDE", "MarkA", "T", "A"]
    assert "Absent" not in set(lr.receptor)
    d = analysis.ligand_receptor(s, sender="T", receiver="A", a="treat", b="ctrl")
    assert d.iloc[0].delta > 0


def test_run_python(toy):
    _, s = toy
    text, figs = run.run_python(s, "print(s.obs.cell_type.value_counts().to_dict())\n"
                                   "plt.plot([1, 2]); print('done')", title="toy check")
    assert text.startswith("[ok") and "done" in text and len(figs) == 1
    text, _ = run.run_python(s, "raise ValueError('boom')", title="fail")
    assert "FAILED" in text and "boom" in text
    assert len(list((s.path / "analyses").glob("*.py"))) == 2
