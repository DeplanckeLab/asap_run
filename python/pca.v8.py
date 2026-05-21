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

## Main PCA function
def pca(args):
    input_path = Path(args.f).resolve()
    if not input_path.is_file():
        ErrorJSON(f"Input LOOM file not found: {args.f}")

    out_dir = Path(args.o).resolve() if args.o else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    output_json_path = str(out_dir / OUTPUT_JSON_NAME) if out_dir else None

    if not args.output_meta.startswith("/"):
        ErrorJSON(f"--output_meta must start with / (e.g. /col_attrs/pca), got: {args.output_meta!r}")

    try:
        import scanpy as sc
        from importlib.metadata import version as _pkg_version
        tool_version = _pkg_version("scanpy")
    except ImportError:
        ErrorJSON("scanpy is not installed. Install it with: pip install scanpy")

    warnings: list[str] = []

    result: dict = {
        "parameters": {},
        "pca":        {"tool": "scanpy", "tool_version": tool_version,
                       "n_pcs": args.n_pcs, "n_features": -1,
                       "variance": None, "variance_ratio": None},
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

        s0, s1 = int(node.shape[0]), int(node.shape[1])
        ra_len = f_in[_GENE_ID_PATH].shape[0]
        ca_len = f_in[_CELL_ID_PATH].shape[0]
        if s0 == ca_len and s1 == ra_len:
            n_genes, n_cells = s1, s0
            matrix_gxc = node[:, :].astype(DEFAULT_NP_DTYPE).T   # genes x cells
        else:
            n_genes, n_cells = s0, s1
            matrix_gxc = node[:, :].astype(DEFAULT_NP_DTYPE)

        cell_ids = _read_string_dataset(f_in, _CELL_ID_PATH)
        gene_ids = _read_string_dataset(f_in, _GENE_ID_PATH)

        # Read HVG flags if --features is provided
        if args.features:
            if args.features not in f_in:
                ErrorJSON(f"--features path {args.features!r} not found in LOOM file.")
            hvg_flags = f_in[args.features][()].astype(int).ravel()
            if len(hvg_flags) != n_genes:
                ErrorJSON(f"--features dataset length ({len(hvg_flags)}) does not match number of genes ({n_genes}).")
            feature_mask = hvg_flags == 1
            if not feature_mask.any():
                ErrorJSON("No genes selected by --features (all flags are 0).")
        else:
            feature_mask = np.ones(n_genes, dtype=bool)

    n_features   = int(feature_mask.sum())
    n_pcs_actual = min(args.n_pcs, n_features - 1, n_cells - 1)
    if n_pcs_actual < args.n_pcs:
        warnings.append(f"n_pcs capped from {args.n_pcs} to {n_pcs_actual} (min(n_features, n_cells) - 1).")

    result["parameters"] = {
        "loom_path":        str(input_path),
        "input_loom_path":  args.input_meta,
        "output_loom_path": args.output_meta,
        "method":           args.method,
        "n_pcs":            n_pcs_actual,
        "features":         args.features,
        "zero_center":      args.zero_center,
        "svd_solver":       args.svd_solver,
        "random_state":     args.random_state,
        "chunked":          args.chunked,
        "chunk_size":       args.chunk_size,
    }
    result["pca"]["n_pcs"]      = n_pcs_actual
    result["pca"]["n_features"] = n_features

    # Build AnnData (cells x genes), subsetting to selected features
    import anndata
    mat_cx = matrix_gxc.T.astype(DEFAULT_NP_DTYPE)   # cells x genes
    if not feature_mask.all():
        mat_cx = mat_cx[:, feature_mask]
    adata = anndata.AnnData(X=mat_cx)

    sc.pp.pca(
        adata,
        n_comps      = n_pcs_actual,
        zero_center  = args.zero_center,
        svd_solver   = args.svd_solver,
        random_state = args.random_state,
        chunked      = args.chunked,
        chunk_size   = args.chunk_size if args.chunked else None,
    )

    cell_embeddings = adata.obsm["X_pca"].astype(DEFAULT_NP_DTYPE)   # (n_cells x n_pcs)
    variance        = adata.uns["pca"]["variance"].astype(DEFAULT_NP_DTYPE)
    variance_ratio  = adata.uns["pca"]["variance_ratio"].astype(DEFAULT_NP_DTYPE)

    result["pca"]["variance"]       = variance.tolist()
    result["pca"]["variance_ratio"] = variance_ratio.tolist()

    # Append cell embeddings into the LOOM
    f_rw, wt = _open_loom_with_retry(str(input_path), "r+")
    result["wait_time"] += wt
    with f_rw:
        out_path = args.output_meta
        if out_path in f_rw or out_path.lstrip("/") in f_rw:
            warnings.append(f"Path '{out_path}' already exists and will be overwritten.")
        size_ce = _write_dataset_2d(f_rw, out_path, cell_embeddings)

    result["metadata"] = [{"name": args.output_meta, "on": "CELL", "type": "NUMERIC",
                            "nber_rows": n_cells, "nber_cols": n_pcs_actual,
                            "dataset_size": size_ce, "imported": 0}]

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
PCA Script — scanpy backend

Options:
  -f             Input LOOM file.                                            [required]
  --input_meta   Path to the scaled/normalized matrix inside the LOOM.      [required]
  --output_meta  Path where cell embeddings (cells x n_pcs) are stored.     [required]
                   Examples: /col_attrs/pca   /col_attrs/pca_1_scaled_ln
  --method       PCA method: pca                                             [required]
  -o             Output folder for output.json. [optional, default: stdout]

  -- sc.pp.pca parameters (defaults match scanpy) ---------------------------------
  --n_pcs           Number of principal components.          [default: 50]
  --features        Path to a LOOM /row_attrs/ dataset with 0/1 HVG flags
                      (e.g. /row_attrs/highly_variable). NULL = all genes.
                                                             [default: NULL]
  --no_zero_center  Disable zero-centering.                  [default: zero_center=True]
  --svd_solver      SVD solver: arpack | randomized | auto.  [default: arpack]
  --random_state    Random seed.                              [default: 0]
  --chunked         Process in chunks (memory efficient).     [default: False]
  --chunk_size      Chunk size (only used with --chunked).    [default: None]

  --help         Show this message and exit.
"""

def main():
    if "--help" in sys.argv:
        print(HELP_TEXT)
        sys.exit(0)

    parser = argparse.ArgumentParser(description="PCA Script (scanpy)", add_help=False)
    parser.add_argument("-f",            required=True)
    parser.add_argument("-o",            required=False, default=None)
    parser.add_argument("--input_meta",  required=True)
    parser.add_argument("--output_meta", required=True)
    parser.add_argument("--method",      required=True, choices=["pca"])
    parser.add_argument("--n_pcs",          type=int,   default=50)
    parser.add_argument("--features",       type=str,   default=None)
    parser.add_argument("--zero_center",    dest="zero_center", action="store_true",  default=True)
    parser.add_argument("--no_zero_center", dest="zero_center", action="store_false")
    parser.add_argument("--svd_solver",     type=str,   default="arpack")
    parser.add_argument("--random_state",   type=int,   default=0)
    parser.add_argument("--chunked",        action="store_true", default=False)
    parser.add_argument("--chunk_size",     type=int,   default=None)

    args = parser.parse_args()
    pca(args)

if __name__ == "__main__":
    main()