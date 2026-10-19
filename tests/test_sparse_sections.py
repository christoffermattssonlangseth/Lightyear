import numpy as np

from lightyear import context, plots, spatial, stats


def test_sparse_store_matches_source(toy_pieces):
    a, s = toy_pieces
    assert s.sparse and (s.path / "expr_csc").exists() and not (s.path / "expr.npy").exists()
    X = a.X.toarray()
    for j in (0, 1, 7):
        assert np.allclose(s.gene(a.var_names[j]), X[:, j], atol=1e-2)
    mean, frac, _ = stats.group_stats(s, ["cell_type"], ["MarkA", "MarkT"])
    assert mean.loc["A", "MarkA"] > 1 > mean.loc["T", "MarkA"]
    assert np.isclose(mean.loc["A", "MarkA"], X[(a.obs.cell_type == "A").to_numpy(), 0].mean(), atol=1e-2)


def test_spatial_ops_stay_within_pieces(toy_pieces):
    _, s = toy_pieces
    idx, _ = s.knn
    piece = s.strcol("piece")
    assert (piece[np.asarray(idx[:, :10])] == piece[:, None]).all()
    nb = spatial.neighbourhood_of(s, {"cell_type": "A"}, k=10)
    assert nb["frac"]["A"].min() > 0.99  # pieces overlap in coordinates but are never neighbours
    co = spatial.co_occurrence(s, {"cell_type": "A"}, {"cell_type": "T"}, [20, 100])
    assert (co["ratio"] < 0.01).all()
    d = stats.distance_to(s.xy, stats.codes_for(s.obs, [s.role("section")])[0], s.mask({"cell_type": "A"}),
                          s.mask({"cell_type": "T"}))
    assert np.isnan(d[s.mask({"cell_type": "A"})]).all()
    p = plots.spatial(s, "a1", "cell_type")  # an animal -> one panel per piece
    assert p.exists()


def test_context_on_sparse_with_gene_cap(toy_pieces):
    _, s = toy_pieces
    perf = context.fit(s, "pc", max_cells=2000, max_genes=10, genes=["G40"], log=lambda *_: None)
    g = set(perf.gene)
    assert len(g) <= 11 and "G40" in g
