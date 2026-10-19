import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from spaceagent import ingest


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    """4 sections, 2 groups. Planted: GeneDE up in 'treat' T cells; A and B cells cluster together;
    GeneGap unmeasured in s4."""
    rng = np.random.default_rng(0)
    rows, obs, xy = [], [], []
    genes = ["MarkA", "MarkB", "MarkT", "GeneDE", "GeneGap", "Noise"]
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
