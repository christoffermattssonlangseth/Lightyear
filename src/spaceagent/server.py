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
