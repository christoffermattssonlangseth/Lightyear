# SpaceAgent

Ask questions of a spatial omics dataset in plain language. The agent reads a precomputed
**dataset card** and answers with numbers, per-sample replication and figures computed locally.

```
.h5ad ──ingest──▶ store/ (obs.parquet, spatial.npy, expr.npy gene-major float16, knn_idx/dist.npy)
                    └─ card/ card.md + tables/*.parquet   ◀── "what the agent knows"
                    └─ findings.md                        ◀── what it learned in past sessions
                           │
                   MCP server (spaceagent.server) ──▶ Claude Code / any MCP client
```

## Ingest a dataset

```bash
pip install -e '.[test]'
spaceagent inspect path/to/data.h5ad                 # check inferred roles first
spaceagent ingest path/to/data.h5ad stores/<name> --config configs/<name>.yaml
```

The config is optional: it pins column roles (sample / cell_type / group), adds column
descriptions, picks niche columns, lists condition contrasts and adds known caveats
(see `configs/cuprizone.yaml`). Without it, roles are inferred from column names and from
which columns are constant within each sample.

The card contains: overview, column dictionary, samples (with depth and covariate nesting),
cell-type abundance per group, marker genes, **panel coverage** (genes unmeasured in some
samples), niche composition, kNN co-localisation per group, per-cell-type condition contrasts
(sample-level pseudobulk) and auto-detected caveats.

## Query it

Point `.mcp.json` at the store and start Claude Code in this folder. The `query-dataset` skill
runs a Plan → Act → Conclude → Verify loop over 29 tools:

| Area | Tools |
|---|---|
| Orientation | `dataset_card`, `describe_structures`, `column_values`, `list_genes`, `read_findings`, `add_finding` |
| Annotation | `markers` (any categorical column), `annotate` (PanglaoDB + CellMarker matching), `gate`, `score_genes` |
| Expression | `expression`, `compare_groups`, `plot_dotplot`, `plot_sample_bars` |
| Spatial neighbours | `neighbourhood_of`, `neighbourhood`, `nhood_enrichment` (permutation z), `co_occurrence` (vs distance), `distance`, `gene_near`, `gradient`, `spatial_autocorrelation` (Moran's I), `plot_spatial` |
| Niches | `compute_niches` (cell-type composition, or UTAG-like neighbour-smoothed expression), `composition` |
| Case vs control | `case_control` (composition, niches, neighbourhoods and expression ranked by sample-level effect size, with pan-cell-type spillover flagged), `separability` (leave-one-sample-out classifier), `ligand_receptor` (OmniPath, co-localisation-annotated) |
| Free analysis | `run_python`: saved, numbered scripts in `<store>/analyses/` that return stdout and figures |

New labels created by tools (niches, gates, scores) are saved to `derived.parquet` and work in every
tool and in later sessions. Inspired in part by [SpatialAgent](https://github.com/Brogrammer94/SpatialAgent)
(Wang, Regev et al. 2025), adapted to aggregate-only tools over a fast local store.

The same functions work in a notebook:

```python
from spaceagent import load, tools
s = load("stores/cuprizone")
print(tools.compare_groups(s, "NoRecov_CupRap", "NoRecov_Cntl", cell_type="Microglia"))
```

## Limits (v0.1)

- Dense gene-major store, which suits targeted panels (Xenium/MERSCOPE/CosMx). Whole-transcriptome data needs a sparse backend.
- Expression is assumed to be log-normalised. Use `--layer` to choose another layer.
- One store per server process.
- External knowledge (marker DBs, OmniPath ligand-receptor) is downloaded on first use to `~/.cache/spaceagent`.
- `run_python` runs arbitrary code locally; nothing enforces aggregate-only output. Keep that in mind for human data.
