import asyncio

import numpy as np
import pandas as pd

from spaceagent import tools, plots


def test_roles_inferred(toy):
    _, s = toy
    assert s.roles == {"sample": "sample_id", "cell_type": "cell_type", "group": "group"}
    assert s.columns["group"]["level"] == "sample"


def test_expr_roundtrip(toy):
    a, s = toy
    X = a.X.toarray()
    for j, g in enumerate(a.var_names):
        np.testing.assert_allclose(s.gene(g), X[:, j], atol=2e-3)
    assert s.resolve_gene("markt") == "MarkT"


def test_card_findings(toy):
    _, s = toy
    card = tools.card(s)
    for sec in ("Cell types", "Marker genes", "Panel coverage", "Neighbourhoods", "Condition contrasts", "Caveats"):
        assert sec in card
    assert "GeneGap" in tools.card(s, "panel") and "s4" in tools.card(s, "panel")
    assert "MarkT" in tools.card(s, "markers").split("**T**")[1].splitlines()[0]
    nh = pd.read_parquet(s.card_dir / "tables" / "nhood_by_group.parquet")
    ab = nh[(nh.centre == "A") & (nh.neighbour == "B")].log2_oe.mean()
    at = nh[(nh.centre == "A") & (nh.neighbour == "T")].log2_oe.mean()
    assert ab > 0 > at
    con = pd.read_parquet(s.card_dir / "tables" / "contrasts.parquet")
    top = con[con.cell_type == "T"].iloc[0]
    assert top.gene == "GeneDE" and top.delta > 1


def test_tools(toy):
    _, s = toy
    assert "GeneDE mean" in tools.expression(s, ["GeneDE"], ["cell_type"], {"group": "treat"})
    assert "WARNING: GeneGap" in tools.expression(s, ["GeneGap"], ["group"])
    assert "±" in tools.composition(s, "cell_type", ["sample_id"], summarize_by_group="group")
    out = tools.compare_groups(s, "treat", "ctrl", cell_type="T")
    rows = [l for l in out.splitlines() if l.startswith("| ") and "mean_treat" not in l]
    assert rows[0].startswith("| GeneDE")
    assert "GeneGap" not in out.split("WARNING")[0]  # s4 zeros are treated as missing, not as low
    assert "median" in tools.distance(s, {"cell_type": "A"}, {"cell_type": "T"}, by="group")
    assert s.mask({"total_counts": {">": 1e9}}).sum() == 0
    tools.add_finding(s, "t", "GeneDE up in treat T")
    assert "GeneDE up" in tools.read_findings(s)


def test_plots(toy):
    _, s = toy
    for p in (plots.spatial(s, "s3", "GeneDE"), plots.spatial(s, "s1", "cell_type", highlight=["A"]),
              plots.dotplot(s, ["MarkA", "MarkT"], "cell_type"), plots.sample_bars(s, "GeneDE", "group", "T")):
        assert p.exists() and p.stat().st_size > 1000


def test_mcp_server(toy, monkeypatch):
    _, s = toy
    from spaceagent import server
    monkeypatch.setattr(server, "STORE_PATH", str(s.path))
    monkeypatch.setattr(server, "_store", None)
    names = {t.name for t in asyncio.run(server.mcp.list_tools())}
    assert {"dataset_card", "expression", "compare_groups", "plot_spatial"} <= names
    res = asyncio.run(server.mcp.call_tool("expression", {"genes": ["GeneDE"], "by": ["group"]}))
    assert "GeneDE" in str(res)
