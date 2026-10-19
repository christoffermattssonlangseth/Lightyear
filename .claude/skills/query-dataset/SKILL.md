---
name: query-dataset
description: Answer a biological question about the loaded spatial omics dataset with the spaceagent MCP tools - interpret annotations and niches, analyse spatial neighbourhoods, compare cases vs controls, run custom analyses. Grounded in numbers, per-sample replication and figures. Use whenever the user asks what is in the data, where something is, what surrounds what, whether something changes between conditions, or what separates groups.
---

# Answering a question about the dataset: Plan → Act → Conclude → Verify

## 0. Orient (once per session)
`dataset_card()`, `read_findings()`, `describe_structures()`. Note the role columns (sample / cell_type /
group), the caveats (confounding, panel gaps, pan-cell-type shifts), the niche columns and any derived
columns from earlier sessions. Use names exactly as the card spells them.

## 1. Plan
Write 2-5 steps before calling tools: which cells (filters), which readout, which comparison (each
treatment against **its own** control), which tool per step, and what result would answer the question
either way. If the question uses a concept the data lacks ("lesion", "activated"), say which column, niche,
gate or score you'll use as a proxy and why.

## 2. Act - pick the tool by question type

| Question | Tools |
|---|---|
| What is this cluster / annotation? Is the annotation right? | `markers(column)`, `annotate(column, tissue=...)`, `plot_dotplot` |
| Define a population ("*Spp1*+ microglia", "DAM score") | `gate`, `score_genes` → then use the new column anywhere |
| What surrounds X? Who neighbours whom? | `neighbourhood_of(cells, by=group)`, `neighbourhood`, `nhood_enrichment`, `co_occurrence` (scale) |
| Tissue regions / niches | card niches, `compute_niches` (composition or annotation-free `expression`), `composition(of=niche, summarize_by_group=group)`, `markers(niche)` |
| Does X change near Y? | `gradient(feature, target, cells, by)`, `gene_near`, `distance` |
| Spatially patterned genes | `spatial_autocorrelation` |
| Case vs control - what separates them? | `case_control(a, b)` first; then `separability(a, b, cells={cell_type: ...}, features='both')` per population of interest; `compare_groups` for details; `plot_sample_bars` |
| Who might signal to whom? | `ligand_receptor(sender, receiver, a, b)` - only with co-localisation support, a hypothesis |
| Anything else | `run_python(code, title)` - store API: `s.obs`, `s.xy`, `s.gene(g)`, `s.mask(filters)`, `s.knn`, modules `stats`, `spatial`, `analysis`; print aggregates only |

Filters: `{column: value | [values]}`, `{column: {"not": [...]}}`, numeric/gene `{"Spp1": {">": 1}}`.

## 3. Look
Make at least one figure per spatial claim (`plot_spatial`, zoom with `bbox`, `highlight`) and per-sample
bars for condition claims, and actually inspect them. Use `run_python` for custom plots.

## 4. Conclude
Direct answer first, then the key numbers (per group, n samples), figures, caveats, and what would firm it
up. Italicise mouse gene symbols (*Spp1*).

## 5. Verify (before sending)
Check every claim in the answer against tool output from this session:
- Is each number copied from a tool result (not recalled or rounded into a different claim)?
- Replicates = samples. With n=2-3 per group, report effect sizes and per-sample values; p-values and
  Hedges g with tiny n are unstable, and nhood z-scores grow with cell number (compare between groups,
  not to 2).
- Gene flagged as pan-cell-type shift / spillover, or missing from a panel? Then don't call it regulated
  in that cell type; check `gene_near` / the source population.
- Could a composition or niche shift explain an apparent expression change? Compare within cell type and
  niche.
- Contrast within one batch/experiment only.
- Ligand-receptor and marker-database matches are hypotheses.
Drop or qualify any claim that fails.

## 6. Record
`add_finding(title, text)` for non-trivial results: the numbers, the exact tool calls / analysis script
paths, and the caveats, so the next session starts from them.
