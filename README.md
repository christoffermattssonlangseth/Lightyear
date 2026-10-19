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

## What makes it different: a claims ledger

Spatial case–control studies usually have 2–3 animals per group, so over-claiming is easy. In SpaceAgent
a finding is a structured, re-runnable **claim**. `test_claim` reduces it to one value per sample and tries
to break it:

| test | asks |
|---|---|
| direction / replicates | right sign, meaningful size, ≥ 2 samples per group |
| separation | is every case sample beyond every control sample? |
| leave_one_out | does it survive dropping any single animal? |
| permutation | exact sample-label permutation p, reported with the smallest p the design allows |
| spillover | does it survive in cells with no neighbour of the main other source cell type? |
| specificity | does the gene move the same way in most cell types (ambient / segmentation signature)? |
| niche | does it hold within niches, or is it a niche-composition shift? |
| confident | does it hold in cells with strong identity markers (misannotation, doublets)? |

Verdicts (`robust`, `fragile`, `underpowered`, `not_supported`, `refuted`) are kept in `<store>/claims.json`
with a fingerprint of the data each claim used. When the annotation, a derived column or the expression
changes, `list_claims` marks claims stale and `recheck_claims` re-runs them and reports flipped verdicts.

On cupriZONE (acute CupRap vs control, 3 v 3) *Spp1*, *Cybb* up and *Tmem119* down in microglia are
`robust`. *Mbp* down in microglia and *Gfap* up in oligodendrocytes, both known spillover artefacts,
come out `fragile`: the ambient signature is found from the data, with no gene list.

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
runs a Plan → Act → Conclude → Verify loop over 33 tools:

| Area | Tools |
|---|---|
| Orientation | `dataset_card`, `describe_structures`, `column_values`, `list_genes`, `read_findings`, `add_finding` |
| Annotation | `markers` (any categorical column), `annotate` (PanglaoDB + CellMarker matching), `gate`, `score_genes` |
| Expression | `expression`, `compare_groups`, `plot_dotplot`, `plot_sample_bars` |
| Spatial neighbours | `neighbourhood_of`, `neighbourhood`, `nhood_enrichment` (permutation z), `co_occurrence` (vs distance), `distance`, `gene_near`, `gradient`, `spatial_autocorrelation` (Moran's I), `plot_spatial` |
| Niches | `compute_niches` (cell-type composition, or UTAG-like neighbour-smoothed expression), `composition` |
| Case vs control | `case_control` (composition, niches, neighbourhoods and expression ranked by sample-level effect size, with pan-cell-type spillover flagged), `separability` (leave-one-sample-out classifier), `ligand_receptor` (OmniPath, co-localisation-annotated) |
| Claims ledger | `test_claim`, `list_claims`, `show_claim`, `recheck_claims` |
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

## Limits (v0.3)

- Dense gene-major store, which suits targeted panels (Xenium/MERSCOPE/CosMx). Whole-transcriptome data needs a sparse backend.
- Expression is assumed to be log-normalised. Use `--layer` to choose another layer.
- One store per server process.
- External knowledge (marker DBs, OmniPath ligand-receptor) is downloaded on first use to `~/.cache/spaceagent`.
- `run_python` runs arbitrary code locally; nothing enforces aggregate-only output. Keep that in mind for human data.
