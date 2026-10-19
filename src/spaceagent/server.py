"""MCP server exposing one dataset store to an agent.

    SPACEAGENT_STORE=stores/cuprizone python -m spaceagent.server

Every tool returns aggregates (per group / per sample) or a rendered figure,
never cell-level tables.
"""
from __future__ import annotations

import os
import sys
from typing import Any

from mcp.server.mcpserver import Image, MCPServer

from . import plots, tools
from .store import Store

STORE_PATH = os.environ.get("SPACEAGENT_STORE") or (sys.argv[1] if len(sys.argv) > 1 else None)
_store: Store | None = None


def S() -> Store:
    global _store
    if _store is None:
        if not STORE_PATH:
            raise RuntimeError("set SPACEAGENT_STORE to a store directory")
        _store = Store(STORE_PATH)
    return _store


FILTERS_DOC = ("filters: {column: value | [values]} on obs columns; numeric columns take "
               "{column: {'>': x, '<=': y}}. Example: {'cell_type': 'Microglia', 'group': ['A','B']}.")

mcp = MCPServer(
    name="spaceagent",
    instructions=(
        "Tools over one spatial omics dataset. Start every session with dataset_card() and read_findings(). "
        "Use column names exactly as listed in the card. Treat samples (not cells) as replicates for any "
        "group comparison, and check the card's caveats (panel gaps, confounding) before concluding. "
        + FILTERS_DOC
    ),
)


@mcp.tool()
def dataset_card(section: str | None = None) -> str:
    """The precomputed dataset card (overview, columns, samples, cell types, markers, panel coverage,
    niches, neighbourhoods, condition contrasts, caveats). Pass a section name to get only that part."""
    return tools.card(S(), section)


@mcp.tool()
def column_values(column: str, by: str | None = None) -> str:
    """Values/counts of one obs column, optionally cross-tabulated against another column."""
    return tools.column_values(S(), column, by)


@mcp.tool()
def list_genes(pattern: str | None = None) -> str:
    """Genes in the panel, optionally filtered by a case-insensitive regex."""
    return tools.list_genes(S(), pattern)


@mcp.tool()
def expression(genes: list[str], by: list[str], filters: dict[str, Any] | None = None,
               per_sample: bool = False) -> str:
    """Mean log-normalised expression and % cells expressing for genes, per group of `by` columns.
    per_sample=True adds the sample column so you can see replicate spread."""
    return tools.expression(S(), genes, by, filters, per_sample)


@mcp.tool()
def composition(of: str, by: list[str], filters: dict[str, Any] | None = None, normalize: bool = True,
                summarize_by_group: str | None = None) -> str:
    """Fractions of the categories of column `of` within groups of `by`. With summarize_by_group
    (e.g. the group/condition column), fractions are computed per sample and reported as mean ± sd per group,
    which is the right way to compare conditions."""
    return tools.composition(S(), of, by, filters, normalize, summarize_by_group)


@mcp.tool()
def compare_groups(a: str, b: str, column: str | None = None, cell_type: str | None = None,
                   genes: list[str] | None = None, filters: dict[str, Any] | None = None, n_top: int = 25) -> str:
    """Differential expression a vs b of a sample-level column (default: the group column), using
    per-sample pseudobulk within an optional cell type / filter, Welch t-test across samples."""
    return tools.compare_groups(S(), a, b, column, cell_type, genes, filters, n_top)


@mcp.tool()
def neighbourhood(centre: str | None = None, group: str | None = None, n_top: int = 30) -> str:
    """Cell-type co-localisation (log2 observed/expected among spatial kNN), precomputed per group.
    Give `centre` to see all neighbours of one cell type across groups."""
    return tools.neighbourhood(S(), centre, group, n_top)


@mcp.tool()
def distance(source: dict[str, Any], target: dict[str, Any], by: str | None = None,
             max_um: float | None = None) -> str:
    """Distance from each source cell to the nearest target cell in the same section, summarised per
    sample (median, IQR, optional fraction within max_um) and optionally by a group column.
    source/target are filter dicts."""
    return tools.distance(S(), source, target, by, max_um)


@mcp.tool()
def gene_near(gene: str, near: dict[str, Any], radius: float = 50.0, cells: dict[str, Any] | None = None,
              by: str | None = None) -> str:
    """Is `gene` higher in `cells` lying within `radius` of `near` cells than in those further away?
    Reports per sample (and per group with `by`)."""
    return tools.gene_spatial_association(S(), gene, near, radius, cells, by)


@mcp.tool()
def plot_spatial(sample: str, color: str, filters: dict[str, Any] | None = None,
                 bbox: list[float] | None = None, highlight: list[str] | None = None) -> Image:
    """Image of one section coloured by a gene or obs column. bbox=[xmin,xmax,ymin,ymax] zooms;
    highlight=[categories] greys out all other categories."""
    return Image(path=plots.spatial(S(), sample, color, filters, bbox, highlight))


@mcp.tool()
def plot_dotplot(genes: list[str], by: str, filters: dict[str, Any] | None = None) -> Image:
    """Dot plot of genes across categories of `by` (size = % expressing, colour = scaled mean)."""
    return Image(path=plots.dotplot(S(), genes, by, filters))


@mcp.tool()
def plot_sample_bars(gene: str, group: str, cell_type: str | None = None) -> Image:
    """Per-sample pseudobulk bars of one gene coloured by group - shows replicate spread."""
    return Image(path=plots.sample_bars(S(), gene, group, cell_type))


# -- interpretation / annotation ---------------------------------------------------
@mcp.tool()
def describe_structures() -> str:
    """Data structures beyond obs: source obsm/obsp/layers/uns keys and graph descriptions, the store's
    spatial kNN graph, and derived columns created in earlier sessions (niches, gates, scores)."""
    return tools.describe_structures(S())


@mcp.tool()
def markers(column: str, filters: dict[str, Any] | None = None, n_top: int = 10) -> str:
    """One-vs-rest marker genes for every category of any categorical column (cell types, clusters,
    subclusters, niches, gates) - use to interpret what an annotation or cluster is."""
    return tools.markers(S(), column, filters, n_top)


@mcp.tool()
def annotate(column: str, filters: dict[str, Any] | None = None, tissue: str | None = None) -> str:
    """Suggest identities for each category of `column` by matching its top panel markers to PanglaoDB +
    CellMarker (optionally restricted to a tissue, e.g. 'brain'). For checking or (re)annotating clusters."""
    return tools.annotate(S(), column, filters, tissue)


@mcp.tool()
def gate(name: str, feature: str, threshold: float, within: dict[str, Any] | None = None,
         above: bool = True) -> str:
    """Create a boolean column `name` = cells (within a filter) with gene/score above threshold,
    e.g. name='Spp1_pos_microglia', feature='Spp1', threshold=1, within={'cell_type': 'Microglia'}.
    The new column works in every other tool."""
    return tools.gate(S(), name, feature, threshold, within, above)


@mcp.tool()
def score_genes(name: str, genes: list[str]) -> str:
    """Create a numeric gene-set score column (e.g. a DAM or interferon signature) usable like a gene."""
    return tools.score_genes(S(), name, genes)


# -- spatial neighbours / niches ----------------------------------------------------
@mcp.tool()
def compute_niches(name: str, n_niches: int = 8, features: str = "composition", k: int = 15,
                   max_dist: float | None = None) -> str:
    """Cluster cells by their spatial neighbourhood into `n_niches` niches and save them as column `name`.
    features='composition' = cell-type mix of the k nearest neighbours (needs annotation);
    features='expression' = neighbour-smoothed expression (UTAG-like, annotation-free).
    Returns niche composition and per-group abundance."""
    return tools.compute_niches(S(), name, n_niches, features, k, max_dist)


@mcp.tool()
def neighbourhood_of(cells: dict[str, Any], k: int = 15, column: str | None = None, by: str | None = None) -> str:
    """What surrounds the selected cells? Mean composition of their k nearest neighbours (by `column`,
    default cell type), per sample or per group (`by`), and log2 enrichment vs the section background."""
    return tools.neighbourhood_of(S(), cells, k, column, by)


@mcp.tool()
def nhood_enrichment(samples: list[str] | None = None, centre: str | None = None, k: int = 10,
                     n_perm: int = 50, by: str | None = None) -> str:
    """Permutation-tested neighbourhood enrichment (squidpy-style z-scores) of cell-type pairs within
    sections. Restrict `samples` for speed; `by` averages z per group."""
    return tools.nhood_enrichment(S(), samples, centre, k, n_perm, by)


@mcp.tool()
def co_occurrence(a: dict[str, Any], b: dict[str, Any], distances: list[float] | None = None,
                  by: str | None = None) -> str:
    """Co-occurrence ratio of cells `b` around cells `a` as a function of distance (in coordinate units):
    tells at what scale two populations associate or avoid each other."""
    return tools.co_occurrence(S(), a, b, distances, by)


@mcp.tool()
def spatial_autocorrelation(genes: list[str] | None = None, samples: list[str] | None = None,
                            n_top: int = 30) -> str:
    """Moran's I per gene (spatially variable genes). Defaults to all genes in three samples."""
    return tools.spatial_autocorrelation(S(), genes, samples, n_top)


@mcp.tool()
def gradient(feature: str, target: dict[str, Any], cells: dict[str, Any] | None = None,
             bins: list[float] | None = None, by: str | None = None) -> str:
    """How a gene/score in `cells` changes with distance to the nearest `target` cell (e.g. microglial
    Spp1 vs distance to a lesion niche). Binned, per sample or per group."""
    return tools.gradient(S(), feature, target, cells, bins, by)


# -- case / control ------------------------------------------------------------------
@mcp.tool()
def case_control(a: str, b: str, column: str | None = None, n_top: int = 15) -> str:
    """What separates group a from group b? Sample-level comparison of cell-type composition, niche
    abundance (incl. derived niches/gates), neighbourhood structure and per-cell-type expression, ranked by
    effect size. Start here for any 'how do cases differ from controls' question."""
    return tools.case_control(S(), a, b, column, n_top)


@mcp.tool()
def separability(a: str, b: str, column: str | None = None, cells: dict[str, Any] | None = None,
                 features: str = "expression") -> str:
    """Can a and b be told apart at the cell level, on held-out samples? Leave-one-sample-out
    classifier (features: 'expression', 'neighbourhood' or 'both'), returns AUC and the driving features.
    Use within one cell type to ask whether that population changed state."""
    return tools.separability(S(), a, b, column, cells, features)


@mcp.tool()
def ligand_receptor(sender: str | None = None, receiver: str | None = None, a: str | None = None,
                    b: str | None = None, column: str | None = None, n_top: int = 30) -> str:
    """Candidate ligand-receptor interactions (OmniPath, panel genes only) between cell types, annotated
    with how co-localised sender and receiver are. Give a/b to rank pairs that change between groups."""
    return tools.ligand_receptor(S(), sender, receiver, a, b, column, n_top)


# -- free analysis -------------------------------------------------------------------
@mcp.tool(structured_output=False)
def run_python(code: str, title: str = "analysis") -> list:
    """Run Python against the dataset for anything the other tools can't express. Preloaded: np, pd, plt,
    s (the store: s.obs, s.xy, s.gene(name), s.mask(filters), s.knn), modules stats/spatial/analysis/
    tools/plots, OUT (output dir). Print aggregated results only (no cell-level dumps). Open matplotlib
    figures are saved and returned. The script is saved under analyses/ for reproducibility."""
    from .run import run_python as _run
    text, figs = _run(S(), code, title)
    return [text] + [Image(path=p) for p in figs[:4]]


@mcp.tool()
def add_finding(title: str, text: str) -> str:
    """Record a finding (with the numbers and tool calls that support it) in the dataset's findings log,
    so later sessions can build on it."""
    return tools.add_finding(S(), title, text)


@mcp.tool()
def read_findings() -> str:
    """Findings recorded in previous sessions."""
    return tools.read_findings(S())


def main():
    mcp.run()


if __name__ == "__main__":
    main()
