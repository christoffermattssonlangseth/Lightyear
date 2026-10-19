"""lightyear ingest <h5ad> <store> [--config cfg.yaml]  |  lightyear inspect <h5ad>  |  lightyear serve <store>"""
from __future__ import annotations

import argparse
import json
import os

from .ingest import ingest, load_config
from .schema import inspect_h5ad


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lightyear")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("inspect", help="print inferred column roles (no store written)")
    a.add_argument("h5ad")
    a.add_argument("--config")

    a = sub.add_parser("ingest", help="build the store and dataset card")
    a.add_argument("h5ad")
    a.add_argument("store")
    a.add_argument("--config")
    a.add_argument("--layer", help="expression layer (default X; should be log-normalised)")
    a.add_argument("--rebuild-store", action="store_true", help="re-export expression even if the store exists")
    a.add_argument("-k", type=int, default=10, help="spatial neighbours for co-localisation")

    a = sub.add_parser("serve", help="run the MCP server over stdio")
    a.add_argument("store")

    args = ap.parse_args(argv)
    if args.cmd == "inspect":
        schema, _ = inspect_h5ad(args.h5ad, load_config(args.config))
        print(json.dumps({"n_cells": schema.n_cells, "n_genes": schema.n_genes, "roles": schema.roles,
                          "spatial": schema.spatial_key,
                          "columns": {k: f"{v.role}/{v.level} ({v.n_unique})" for k, v in schema.columns.items()}},
                         indent=1))
    elif args.cmd == "ingest":
        ingest(args.h5ad, args.store, args.config, args.layer, args.rebuild_store, args.k)
    elif args.cmd == "serve":
        os.environ["LIGHTYEAR_STORE"] = args.store
        from . import server
        server.STORE_PATH = args.store
        server.main()


if __name__ == "__main__":
    main()
