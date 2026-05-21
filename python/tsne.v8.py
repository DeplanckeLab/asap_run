from __future__ import annotations
import argparse
import json
import sys
import h5py
import numpy as np
from pathlib import Path
from typing import Any, Optional

## Constants
DEFAULT_NP_DTYPE       = np.float64
LOOM_CHUNK_GENES       = 64
LOOM_CHUNK_CELLS       = 64
LOOM_COMPRESSION       = "gzip"
LOOM_COMPRESSION_LEVEL = 2
LOOM_DTYPE             = "float64"  # on-disk precision: "float32" or "float64"
OUTPUT_JSON_NAME       = "output.json"

_CELL_ID_PATH = "/col_attrs/CellID"
_GENE_ID_PATH = "/row_attrs/_StableID"

## Error handling
class ErrorJSON(Exception):
    def __init__(self, message: str, output_path: str = None):
        super().__init__(message)
        payload = {"displayed_error": message}
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        else:
            print(json.dumps(payload, ensure_ascii=False), file=sys.stdout)
        sys.exit(1)

## JSON serializer
def _json_default(obj: Any):
    if isinstance(obj, np.bool_):    return bool(obj)
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    return str(obj)

## LOOM helpers
def _read_string_dataset(f: h5py.File, path: str) -> list[str]:
    raw = f[path][()]
    if isinstance(raw, (bytes, np.bytes_)):
        return [raw.decode("utf-8", errors="replace")]
    return [v.decode("utf-8", errors="replace") if isinstance(v, (bytes, np.bytes_)) else str(v) for v in raw.ravel()]

def _write_dataset_2d(f_out: h5py.File, path: str, data: np.ndarray) -> int:
    """Write a 2D matrix (rows x cols). Returns on-disk size in bytes."""
    nrows, ncols = data.shape
    chunk = (min(nrows, LOOM_CHUNK_CELLS), min(ncols, LOOM_CHUNK_GENES))
    if path in f_out:
        del f_out[path]
    grp = "/".join(path.split("/")[:-1])
    if grp:
        f_out.require_group(grp)
    ds = f_out.create_dataset(path, data=data.astype(LOOM_DTYPE, copy=False),
                               shape=data.shape, chunks=chunk,
                               compression=LOOM_COMPRESSION, compression_opts=LOOM_COMPRESSION_LEVEL)
    return int(ds.id.get_storage_size())

def _open_loom_with_retry(path: str, mode: str, max_wait: int = 600) -> tuple:
    """Try to open a LOOM/HDF5 file, retrying every second if locked.
    Returns (h5py.File, wait_time_seconds)."""
    import time as _time
    elapsed = 0
    while True:
        try:
            return h5py.File(path, mode), elapsed
        except OSError as e:
            if elapsed >= max_wait:
                ErrorJSON(f"Could not open {path!r} in mode {mode!r} after {max_wait}s: {e}")
            _time.sleep(1)
            elapsed += 1

## Main t-SNE function
def tsne(args):
    input_path = Path(args.f).resolve()
    if not input_path.is_file():
        ErrorJSON(f"Input LOOM file not found: {args.f}")

    out_dir = Path(args.o).resolve() if args.o else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    output_json_path = str(out_dir / OUTPUT_JSON_NAME) if out_dir else None

    if not args.output_meta.startswith("/"):
        ErrorJSON(f"--output_meta must start with / (e.g. /col_attrs/tsne), got: {args.output_meta!r}")

    try:
        import os as _os
        _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        _os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        import scanpy as sc
        from importlib.metadata import version as _pkg_version
        tool_version = _pkg_version("scanpy")
    except ImportError:
        ErrorJSON("scanpy is not installed. Install it with: pip install scanpy")

    warnings: list[str] = []

    result: dict = {
        "parameters": {},
        "tsne":       {"tool": "scanpy", "tool_version": tool_version,
                       "n_components": args.n_components, "n_dims": -1,
                       "perplexity": args.perplexity},
        "metadata":   [],
        "wait_time":  0,
    }

    f_in, wt = _open_loom_with_retry(str(input_path), "r")
    result["wait_time"] += wt
    with f_in:
        src_path = args.input_meta
        if src_path not in f_in:
            ErrorJSON(f"Source path {src_path!r} not found in LOOM file.")
        node = f_in[src_path]
        if not isinstance(node, h5py.Dataset) or len(node.shape) != 2:
            ErrorJSON(f"Expected a 2-D dataset at {src_path}, got shape {getattr(node, 'shape', '?')}")

        ca_len   = f_in[_CELL_ID_PATH].shape[0]
        cell_ids = _read_string_dataset(f_in, _CELL_ID_PATH)

        s0, s1 = int(node.shape[0]), int(node.shape[1])
        # Orientation: Python-written = (n_cells, n_pcs); R-written = (n_pcs, n_cells) → transpose.
        if s0 == ca_len:
            emb_cxp = node[:, :].astype(DEFAULT_NP_DTYPE)
        else:
            emb_cxp = node[:, :].astype(DEFAULT_NP_DTYPE).T

    n_cells, n_pcs = emb_cxp.shape

    # Resolve n_dims
    n_dims = min(args.n_dims, n_pcs) if args.n_dims is not None else n_pcs
    if args.n_dims is not None and args.n_dims > n_pcs:
        warnings.append(f"--n_dims ({args.n_dims}) exceeds available PCs ({n_pcs}); using all {n_pcs}.")

    # Perplexity must be less than n_cells / 3
    max_perplexity = (n_cells - 1) / 3
    perplexity = args.perplexity
    if perplexity >= max_perplexity:
        warnings.append(f"perplexity ({perplexity}) capped to {int(max_perplexity)} (must be < n_cells/3).")
        perplexity = float(int(max_perplexity))

    result["parameters"] = {
        "loom_path":        str(input_path),
        "input_loom_path":  args.input_meta,
        "output_loom_path": args.output_meta,
        "method":           args.method,
        "n_dims":           n_dims,
        "n_components":     args.n_components,
        "perplexity":       perplexity,
        "random_state":     args.random_state,
        "learning_rate":    args.learning_rate,
    }
    result["tsne"]["n_dims"]     = n_dims
    result["tsne"]["perplexity"] = perplexity

    # Build AnnData with PCA embeddings and run t-SNE
    import anndata
    adata               = anndata.AnnData(X=np.zeros((n_cells, 1), dtype=np.float32))
    adata.obs_names     = list(cell_ids)
    adata.obsm["X_pca"] = emb_cxp[:, :n_dims].astype(DEFAULT_NP_DTYPE)

    sc.tl.tsne(
        adata,
        n_pcs        = n_dims,
        perplexity   = perplexity,
        n_components = args.n_components,
        random_state = args.random_state,
        learning_rate= args.learning_rate,
        use_rep      = "X_pca",
    )

    tsne_embeddings = adata.obsm["X_tsne"].astype(DEFAULT_NP_DTYPE)   # (n_cells x n_components)

    # Append t-SNE embeddings into the LOOM
    f_rw, wt = _open_loom_with_retry(str(input_path), "r+")
    result["wait_time"] += wt
    with f_rw:
        out_path = args.output_meta
        if out_path in f_rw or out_path.lstrip("/") in f_rw:
            warnings.append(f"Path '{out_path}' already exists and will be overwritten.")
        size_tsne = _write_dataset_2d(f_rw, out_path, tsne_embeddings)

    result["metadata"] = [{"name": args.output_meta, "on": "CELL", "type": "NUMERIC",
                            "nber_rows": n_cells, "nber_cols": args.n_components,
                            "dataset_size": size_tsne, "imported": 0}]

    if warnings:
        result["warnings"] = warnings

    json_str = json.dumps(result, default=_json_default, ensure_ascii=False, separators=(",", ":"))
    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as fj:
            fj.write(json_str)
    else:
        print(json_str)

## CLI
HELP_TEXT = """
t-SNE Script — scanpy backend

Reads PCA cell embeddings from a LOOM file, runs sc.tl.tsne, appends t-SNE
embeddings into the input LOOM, and writes output.json.

Options:
  -f             Input LOOM file.                                            [required]
  --input_meta   Path to PCA cell embeddings in the LOOM.                   [required]
                   Example: /col_attrs/pca_Py_scaled_ln
  --output_meta  Path where t-SNE embeddings are stored.                    [required]
                   Example: /col_attrs/tsne_Py_scaled_ln
  --method       t-SNE method: tsne                                          [required]
  -o             Output folder for output.json. [optional, default: stdout]

  -- sc.tl.tsne parameters ─────────────────────────────────────────────────────
  --n_dims          Number of PCA dims to use (NULL = all).  [default: all]
  --n_components    Number of t-SNE dimensions.              [default: 2]
  --perplexity      t-SNE perplexity.                        [default: 30]
  --random_state    Random seed.                             [default: 0]
  --learning_rate   Learning rate ('auto' or float).         [default: auto]

  --help         Show this message and exit.
"""

def main():
    if "--help" in sys.argv:
        print(HELP_TEXT)
        sys.exit(0)

    parser = argparse.ArgumentParser(description="t-SNE Script (scanpy)", add_help=False)
    parser.add_argument("-f",            required=True)
    parser.add_argument("-o",            required=False, default=None)
    parser.add_argument("--input_meta",  required=True)
    parser.add_argument("--output_meta", required=True)
    parser.add_argument("--method",      required=True, choices=["tsne"])
    parser.add_argument("--n_dims",       type=int,   default=None)
    parser.add_argument("--n_components", type=int,   default=2)
    parser.add_argument("--perplexity",   type=float, default=30.0)
    parser.add_argument("--random_state", type=int,   default=0)
    parser.add_argument("--learning_rate",type=str,   default="auto")

    args = parser.parse_args()
    tsne(args)

if __name__ == "__main__":
    main()