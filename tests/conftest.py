import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.spatial import cKDTree

from lightyear import ingest


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    """4 sections, 2 groups. Planted: GeneDE up in 'treat' T cells; A and B cells cluster together;
    GeneGap unmeasured in s4; GeneCtx in A cells rises with the share of T among their 10 nearest neighbours."""
    rng = np.random.default_rng(0)
    rows, obs, xy = [], [], []
    genes = ["MarkA", "MarkB", "MarkT", "GeneDE", "GeneGap", "Noise", "GeneCtx"]
    for si, (smp, grp) in enumerate([("s1", "ctrl"), ("s2", "ctrl"), ("s3", "treat"), ("s4", "treat")]):
        n = 1500
        # A and B in the left half, T in the right half
        ct = rng.choice(["A", "B", "T"], n, p=[0.25, 0.25, 0.5] if grp == "treat" else [1 / 3] * 3)
        x = np.where(ct == "T", rng.uniform(500, 1000, n), rng.uniform(0, 500, n))
        y = rng.uniform(0, 1000, n)
        X = rng.poisson(0.2, (n, len(genes))).astype(float)
        X[ct == "A", 0] += 3
        X[ct == "B", 1] += 3
        X[ct == "T", 2] += 3
        X[:, 4] = rng.poisson(1.0, n)
        if grp == "treat":
            X[ct == "T", 3] += 4
        if smp == "s4":
            X[:, 4] = 0
        _, nb = cKDTree(np.c_[x, y]).query(np.c_[x, y], k=11)
        frac_t = (ct[nb[:, 1:]] == "T").mean(1)
        X[:, 6] = rng.poisson(0.2, n) + np.where(ct == "A", rng.poisson(8 * frac_t), 0)
        rows.append(np.log1p(X))
        obs.append(pd.DataFrame({"sample_id": smp, "group": grp, "cell_type": ct,
                                 "total_counts": X.sum(1)}))
        xy.append(np.c_[x, y])
    a = ad.AnnData(sp.csr_matrix(np.vstack(rows).astype(np.float32)),
                   obs=pd.concat(obs, ignore_index=True), var=pd.DataFrame(index=genes))
    a.obs.index = a.obs.index.astype(str)
    for c in ("sample_id", "group", "cell_type"):
        a.obs[c] = a.obs[c].astype("category")
    a.obsm["spatial"] = np.vstack(xy)
    d = tmp_path_factory.mktemp("toy")
    a.write_h5ad(d / "toy.h5ad")
    cfg = d / "cfg.yaml"
    cfg.write_text("contrasts:\n  - {column: group, a: treat, b: ctrl}\n")
    store = ingest(d / "toy.h5ad", d / "store", cfg, log=lambda *_: None)
    return a, store


@pytest.fixture(scope="module")
def toy_pieces(tmp_path_factory):
    """Sparse store; 4 animals x 2 tissue pieces that share one coordinate frame (as when pieces of an animal
    come from different sections). Piece P2 is all T cells, P1 all A cells: any neighbour or distance
    computation that mixes pieces will find T next to A."""
    rng = np.random.default_rng(1)
    obs, xy, rows = [], [], []
    for ai, (animal, grp) in enumerate([("a1", "ctrl"), ("a2", "ctrl"), ("a3", "treat"), ("a4", "treat")]):
        for piece, ct in (("P1", "A"), ("P2", "T")):
            n = 800
            obs.append(pd.DataFrame({"animal": animal, "piece": f"{animal}_{piece}", "group": grp,
                                     "cell_type": ct}, index=[f"{animal}{piece}{i}" for i in range(n)]))
            xy.append(rng.uniform(0, 500, (n, 2)))
            X = rng.poisson(0.1, (n, 50)).astype(float)
            X[:, 0 if ct == "A" else 1] += 3
            rows.append(np.log1p(X))
    a = ad.AnnData(sp.csr_matrix(np.vstack(rows).astype(np.float32)), obs=pd.concat(obs),
                   var=pd.DataFrame(index=["MarkA", "MarkT"] + [f"G{i}" for i in range(48)]))
    for c in ("animal", "piece", "group", "cell_type"):
        a.obs[c] = a.obs[c].astype("category")
    a.obsm["spatial"] = np.vstack(xy)
    d = tmp_path_factory.mktemp("pieces")
    a.write_h5ad(d / "p.h5ad")
    cfg = d / "cfg.yaml"
    cfg.write_text("sparse: true\nroles: {sample: animal, section: piece, cell_type: cell_type, group: group}\n")
    return a, ingest(d / "p.h5ad", d / "store", cfg, log=lambda *_: None)
