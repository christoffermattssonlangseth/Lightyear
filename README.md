# SpaceAgent

Ask questions of a spatial omics dataset in plain language. The agent reads a precomputed
**dataset card** and answers with numbers, per-sample replication and figures computed locally.

```
.h5ad ──ingest──▶ store/ (obs.parquet, spatial.npy, expr.npy gene-major float16)
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

Point `.mcp.json` at the store and start Claude Code in this folder. Tools:
`dataset_card`, `column_values`, `list_genes`, `expression`, `composition`, `compare_groups`,
`neighbourhood`, `distance`, `gene_near`, `plot_spatial`, `plot_dotplot`, `plot_sample_bars`,
`add_finding`, `read_findings`. The `query-dataset` skill sets out how they should be used.

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
