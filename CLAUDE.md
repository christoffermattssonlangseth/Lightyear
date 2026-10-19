# Lightyear

An agent that "learns" a spatial omics dataset once (ingest → store + dataset card) and then answers
questions about it through MCP tools (`src/lightyear/server.py`, registered in `.mcp.json`).

- Answering data questions: follow `.claude/skills/query-dataset/SKILL.md`.
- Stores served (one MCP server each, see `.mcp.json`; git-ignored, rebuild with the README command):
  `lightyear-cuprizone` -> `stores/cuprizone` (Xenium 307 genes, dense) and
  `lightyear-rrmap2` -> `stores/rrmap2` (Xenium 5K EAE spinal cord, sparse CSC; replicate = animal
  `sample_name`, spatial unit = tissue piece `kmeans_split_id`). Ask which dataset a question is about if unclear.
- Python env: miniforge base (`/Users/chrislangseth/miniforge3/bin/python`, py3.10, mcp 2.x - `MCPServer`, not FastMCP).
- Tests: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q` (the base env's napari pytest plugin is broken).
- Tools must return aggregates (per group/sample) or figures, never cell-level tables - the same design
  must work later on human/GDPR datasets where only aggregates may leave the machine.
