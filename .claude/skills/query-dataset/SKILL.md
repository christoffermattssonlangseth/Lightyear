---
name: query-dataset
description: Answer a biological question about the loaded spatial omics dataset using the spaceagent MCP tools - grounded in numbers, per-sample replication and figures. Use whenever the user asks what is in the data, where something is, whether something changes between conditions, or which cells sit near which.
---

# Answering a question about the dataset

1. **Orient (once per session).** Call `dataset_card()` and `read_findings()`. Note the role columns
   (sample / cell_type / group), the caveats, and the panel gaps. Use column and category names exactly
   as the card spells them.

2. **Translate the question** into data terms before calling tools: which cells (filters), which
   readout (gene expression, composition, distance, co-localisation), which comparison (which groups,
   and which control belongs to which treatment). If the question uses a term the data doesn't have
   (e.g. "lesion"), say which column/category you are using as a proxy and why.

3. **Query, smallest first.** Prefer precomputed card tables, then aggregate tools
   (`expression`, `composition`, `compare_groups`, `distance`, `gene_near`, `neighbourhood`).
   Use `per_sample=True` / `summarize_by_group` so replicate spread is visible.
   For anything the tools can't express, write Python against the store
   (`from spaceagent import load, stats; s = load(<store>)`) - keep outputs aggregated.

4. **Look.** Make at least one figure for spatial claims (`plot_spatial`, zoom with `bbox`,
   `highlight` categories) and per-sample bars for condition claims (`plot_sample_bars`), and look at it.

5. **Check before concluding.**
   - Replicates are samples, not cells. With n=2-3 per group report effect sizes and per-sample
     values; don't lean on p-values.
   - Gene not measured in some samples? (WARNING lines / Panel coverage) - say so.
   - Is the contrast within one batch/experiment? Never compare treatment to another experiment's control.
   - Could a cell-type composition shift explain an apparent expression change? Check within cell type.
   - Segmentation spillover: a non-canonical gene in a cell type next to cells that express it highly
     is often spillover - check `gene_near`.

6. **Answer** with: the direct answer first, the key numbers (per group, n samples), the figure(s),
   caveats, and what would firm it up. Italicise mouse gene symbols (*Spp1*).

7. **Record** non-trivial results with `add_finding(title, text)` including the numbers and the exact
   tool calls, so the next session starts from them.
