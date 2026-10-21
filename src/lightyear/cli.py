"""lightyear ingest <h5ad> <store> [--config cfg.yaml] | inspect <h5ad> | serve [store] | gui [--stores dir]"""
from __future__ import annotations

import argparse
import json
import os
import sys

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

    a = sub.add_parser("serve", help="run the MCP server over stdio (store from the argument or $LIGHTYEAR_STORE)")
    a.add_argument("store", nargs="?")

    a = sub.add_parser("gui", help="local chat app over all stores in a directory")
    a.add_argument("--stores", help="folder of stores (default: the app setting, else ./stores)")
    a.add_argument("--port", type=int, default=8790, help="first port to try (next free one is used)")
    a.add_argument("--no-browser", action="store_true")
    a.add_argument("--window", action="store_true", help="open in a native window (needs pywebview)")

    a = sub.add_parser("run-script", help=argparse.SUPPRESS)  # bundled app: stands in for `python script.py`
    a.add_argument("script")

    args = ap.parse_args(argv)
    if args.cmd == "inspect":
        schema, _ = inspect_h5ad(args.h5ad, load_config(args.config))
        print(json.dumps({"n_cells": schema.n_cells, "n_genes": schema.n_genes, "roles": schema.roles,
                          "spatial": schema.spatial_key,
                          "columns": {k: f"{v.role}/{v.level} ({v.n_unique})" for k, v in schema.columns.items()}},
                         indent=1))
    elif args.cmd == "ingest":
        sys.stdout.reconfigure(line_buffering=True)  # the app shows the progress lines as they come
        ingest(args.h5ad, args.store, args.config, args.layer, args.rebuild_store, args.k)
    elif args.cmd == "serve":
        store = args.store or os.environ.get("LIGHTYEAR_STORE")
        if not store:
            ap.error("serve needs a store (argument or $LIGHTYEAR_STORE)")
        os.environ["LIGHTYEAR_STORE"] = store
        from . import server
        server.STORE_PATH = store
        server.main()
    elif args.cmd == "gui":
        from .gui.app import main as gui_main
        gui_main(args.stores, port=args.port, open_browser=not args.no_browser, window=args.window)
    elif args.cmd == "run-script":
        import runpy
        sys.argv = [args.script]
        runpy.run_path(args.script, run_name="__main__")


if __name__ == "__main__":
    main()
