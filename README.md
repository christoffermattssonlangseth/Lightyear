# Lightyear

**L**earns, **I**nterrogates, **G**enerates **H**ypotheses, **T**ests, **Y**ields **E**vidence **A**nd **R**efutations.

Ask questions of a spatial omics dataset in plain language. The agent reads a precomputed
**dataset card** and answers with numbers, per-sample replication and figures computed locally.

```
.h5ad ──ingest──▶ store/ (obs.parquet, spatial.npy, expression gene-major: dense expr.npy or sparse expr_csc/, knn)
                    └─ card/ card.md + tables/*.parquet   ◀── "what the agent knows"
                    └─ findings.md                        ◀── what it learned in past sessions
                           │
                   MCP server (lightyear.server) ──▶ Claude Code / any MCP client
```

## What makes it different: a claims ledger

Spatial case–control studies usually have 2–3 animals per group, so over-claiming is easy. In Lightyear
a finding is a structured, re-runnable **claim**. `test_claim` reduces it to one value per sample and tries
to break it:

| test | asks |
|---|---|
| direction / replicates | right sign, meaningful size, ≥ 2 samples per group |
| separation | is every case sample beyond every control sample? |
| leave_one_out | does it survive dropping any single animal? |
| permutation | exact sample-label permutation p, reported with the smallest p the design allows |
| spillover | does it survive in cells with no neighbour of the main other source cell type? |
| clean_cells | does it survive in cells with no transcripts of the main source cell type's specific markers? |
| specificity | does the gene move the same way in most cell types? If so and this population is not its main expresser the verdict is `ambiguous`: diffuse ambient RNA (e.g. *Gfap*, *Mbp* in processes) and genuine tissue-wide responses (interferon / MHC-I) look identical in the data, so the agent must decide with prior knowledge and say so |
| niche | does it hold within niches, or is it a niche-composition shift? |
| confident | does it hold in cells with strong identity markers (misannotation, doublets)? |

Verdicts (`robust`, `ambiguous`, `fragile`, `underpowered`, `not_supported`, `refuted`) are kept in `<store>/claims.json`
with a fingerprint of the data each claim used. When the annotation, a derived column or the expression
changes, `list_claims` marks claims stale and `recheck_claims` re-runs them and reports flipped verdicts.

On cupriZONE (acute CupRap vs control, 3 v 3) *Spp1*, *Cybb* up and *Tmem119* down in microglia are
`robust`. *Mbp* down in microglia and *Gfap* up in oligodendrocytes, both known spillover artefacts,
come out `ambiguous`, as does interferon-driven *H2-D1* in oligodendrocytes in RRMAP2 EAE: the data flag
the tissue-wide pattern; whether it is artefact or biology is a knowledge call the agent has to make explicitly.

## A learned model of the tissue: what if the neighbours were different?

`fit_context_model` learns, per cell type, how a cell's expression depends on who surrounds it: ridge
regression of all panel genes on the cell-type composition of its 10 and 30 nearest neighbours (fractions and
square roots), local density and condition. It is **cross-fitted by animal**: every reported number and every
prediction comes from a model that never saw that animal. With condition as a covariate, the neighbourhood
effects are learned within condition, so the neighbourhood is not used as a proxy for "this is a lesioned mouse".

| tool | answers |
|---|---|
| `context_genes` | which genes of a cell type depend on the neighbourhood (`r2_context` = held-out variance explained beyond condition), which neighbour raises/lowers each, and whether that is **spillover-like** (the neighbour expresses the gene far more: picked-up transcripts) or **state-like** |
| `explain_difference` | splits a case–control change into a *neighbourhood part* (different surroundings), a *condition part* (same surroundings, condition switched) and a residual |
| `perturb_neighbourhood` | in-silico edit (remove neighbour types, replace one with another), predicted change per animal, with a **support check**: are the edited neighbourhoods like anything seen in training? |

Model results enter the claims ledger as `kind="context"` / `kind="perturbation"`, with extra tests for held-out
model fit, support and spillover. Phrase them as associations, not causes.

On cupriZONE (acute CupRap vs control, 6 animals, fit in ~40 s):
- In microglia, 86% of the *Fn1* increase and 60% of the *Spp1* increase go with the cells' surroundings
  (other microglia: lesion clusters).
- *Cd68* (30%) and *Tmem119* (42%) shift mostly with condition across all microglia.
- The top "context genes" of most cell types are neighbours' markers (*Ppp1r1b* from MSNs, *Slc17a7* from
  excitatory neurons, *Mbp* from oligodendrocytes), correctly labelled spillover-like.
- "Remove all microglial neighbours" is flagged as extrapolation (60% support).

## Ingest a dataset

```bash
pip install -e '.[test]'
lightyear inspect path/to/data.h5ad                 # check inferred roles first
lightyear ingest path/to/data.h5ad stores/<name> --config configs/<name>.yaml
```

The config is optional: it pins column roles (sample = replicate unit, section = spatial unit if
different, e.g. tissue pieces; cell_type; group), can build a group column from others
(`combine_columns`, e.g. model + stage), adds column
descriptions, picks niche columns, lists condition contrasts and adds known caveats
(see `configs/cuprizone.yaml`). Without it, roles are inferred from column names and from
which columns are constant within each sample.

The card contains: overview, column dictionary, samples (with depth and covariate nesting),
cell-type abundance per group, marker genes, **panel coverage** (genes unmeasured in some
samples), niche composition, kNN co-localisation per group, per-cell-type condition contrasts
(sample-level pseudobulk) and auto-detected caveats.

Panels too large to store densely (e.g. Xenium 5K) are written as a sparse gene-major (CSC) store;
everything else works the same.

## Datasets

| server | store | data |
|---|---|---|
| `lightyear-cuprizone` | `stores/cuprizone` | Xenium 307 genes, mouse forebrain, cuprizone-rapamycin / OSM (GSE266690), 2.36 M cells, 19 sections |
| `lightyear-rrmap2` | `stores/rrmap2` | Xenium 5K, mouse spinal cord, chronic + relapsing-remitting EAE, 1.38 M cells, 67 animals / 158 tissue pieces |

```bash
lightyear ingest <cupriZONE h5ad> stores/cuprizone --config configs/cuprizone.yaml
lightyear ingest <RRMAP2 annotated h5ad> stores/rrmap2 --config configs/rrmap2.yaml
```

## Query it

Each store gets its own MCP server in `.mcp.json`; start Claude Code in this folder. The `query-dataset` skill
runs a Plan → Act → Conclude → Verify loop over 37 tools:

| Area | Tools |
|---|---|
| Orientation | `dataset_card`, `describe_structures`, `column_values`, `list_genes`, `read_findings`, `add_finding` |
| Annotation | `markers` (any categorical column), `annotate` (PanglaoDB + CellMarker matching), `gate`, `score_genes` |
| Expression | `expression`, `compare_groups`, `plot_dotplot`, `plot_sample_bars` |
| Spatial neighbours | `neighbourhood_of`, `neighbourhood`, `nhood_enrichment` (permutation z), `co_occurrence` (vs distance), `distance`, `gene_near`, `gradient`, `spatial_autocorrelation` (Moran's I), `plot_spatial` |
| Niches | `compute_niches` (cell-type composition, or UTAG-like neighbour-smoothed expression), `composition` |
| Case vs control | `case_control` (composition, niches, neighbourhoods and expression ranked by sample-level effect size, with pan-cell-type spillover flagged), `separability` (leave-one-sample-out classifier), `ligand_receptor` (OmniPath, co-localisation-annotated) |
| Context model | `fit_context_model`, `context_genes`, `explain_difference`, `perturb_neighbourhood` |
| Claims ledger | `test_claim`, `list_claims`, `show_claim`, `recheck_claims` |
| Free analysis | `run_python`: saved, numbered scripts in `<store>/analyses/` that return stdout and figures |

New labels created by tools (niches, gates, scores) are saved to `derived.parquet` and work in every
tool and in later sessions. Inspired in part by [SpatialAgent](https://github.com/Brogrammer94/SpatialAgent)
(Wang, Regev et al. 2025), adapted to aggregate-only tools over a fast local store.

The same functions work in a notebook:

```python
from lightyear import load, tools
s = load("stores/cuprizone")
print(tools.compare_groups(s, "NoRecov_CupRap", "NoRecov_Cntl", cell_type="Microglia"))
```

## Limits (v0.4)

- Gene-major store (dense or sparse CSC): fast per-gene queries; whole-matrix operations go gene by gene.
- Expression is assumed to be log-normalised. Use `--layer` to choose another layer.
- One store per server process (one server per dataset); cross-dataset questions are answered by querying both.
- The context model is linear in composition features (with square-root terms). It sees cell-type composition,
  not neighbours' expression states, and it is associational.
- External knowledge (marker DBs, OmniPath ligand-receptor) is downloaded on first use to `~/.cache/lightyear`.
- `run_python` runs arbitrary code locally; nothing enforces aggregate-only output. Keep that in mind for human data.
