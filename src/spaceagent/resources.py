"""External knowledge bases, downloaded once and cached in ~/.cache/spaceagent.

- PanglaoDB + CellMarker 2.0 marker lists (annotation suggestions)
- OmniPath ligand-receptor interactions (ligrecextra), per organism
"""
from __future__ import annotations

import io
import os
import urllib.request
from pathlib import Path

import pandas as pd

CACHE = Path(os.environ.get("SPACEAGENT_CACHE", Path.home() / ".cache" / "spaceagent"))
_SA = "https://raw.githubusercontent.com/Brogrammer94/SpatialAgent/main/data/"
URLS = {
    "panglao": _SA + "PanglaoDB_markers_27_Mar_2020.tsv",
    "cellmarker": _SA + "Cell_marker_All.csv",
}
TAXON = {"mouse": 10090, "human": 9606, "rat": 10116}


def _fetch(name: str, url: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / name
    if not p.exists():
        with urllib.request.urlopen(url, timeout=120) as r:
            p.write_bytes(r.read())
    return p


def _species_case(sym: str, organism: str) -> str:
    return sym.capitalize() if organism != "human" else sym.upper()


def marker_db(organism: str = "mouse") -> pd.DataFrame:
    """Long table (source, cell_type, tissue, gene) from PanglaoDB + CellMarker, gene symbols in the
    organism's case convention."""
    p = CACHE / f"markers_{organism}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    pg = pd.read_csv(_fetch("panglao.tsv", URLS["panglao"]), sep="\t")
    sp = "Mm" if organism == "mouse" else "Hs"
    pg = pg[pg["species"].str.contains(sp, na=False)]
    pg = pd.DataFrame({"source": "PanglaoDB", "cell_type": pg["cell type"], "tissue": pg["organ"],
                       "gene": pg["official gene symbol"].astype(str)})
    cm = pd.read_csv(_fetch("cellmarker.csv", URLS["cellmarker"]), low_memory=False)
    cm = cm[cm["species"].str.lower() == ("mouse" if organism == "mouse" else "human")]
    cm = pd.DataFrame({"source": "CellMarker", "cell_type": cm["cell_name"], "tissue": cm["tissue_type"],
                       "gene": cm["Symbol"].astype(str)})
    db = pd.concat([pg, cm]).dropna()
    db["gene"] = db["gene"].map(lambda g: _species_case(g, organism))
    db = db.drop_duplicates()
    db.to_parquet(p)
    return db


def ligand_receptor(organism: str = "mouse") -> pd.DataFrame:
    """OmniPath ligand-receptor pairs: ligand, receptor (complexes as 'A_B'), n_sources."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"lr_{organism}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    url = ("https://omnipathdb.org/interactions?datasets=ligrecextra&genesymbols=yes&fields=sources"
           f"&format=tsv&organisms={TAXON[organism]}")
    with urllib.request.urlopen(url, timeout=120) as r:
        df = pd.read_csv(io.BytesIO(r.read()), sep="\t")
    df = pd.DataFrame({"ligand": df["source_genesymbol"], "receptor": df["target_genesymbol"],
                       "n_sources": df["sources"].str.count(";") + 1}).drop_duplicates(["ligand", "receptor"])
    df.to_parquet(p)
    return df
