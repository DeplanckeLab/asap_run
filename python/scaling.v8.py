from __future__ import annotations
import argparse
import json
import sys
import h5py
import numpy as np
from pathlib import Path
from typing import Any, Optional

## Constants (matching parse_v8 / normalize conventions)
DEFAULT_NP_DTYPE       = np.float64
LOOM_CHUNK_GENES       = 64
LOOM_CHUNK_CELLS       = 64
LOOM_COMPRESSION       = "gzip"
LOOM_COMPRESSION_LEVEL = 2
LOOM_DTYPE             = "float32"
OUTPUT_JSON_NAME       = "output.json"

_CELL_ID_PATH = "/col_attrs/CellID"
_GENE_ID_PATH = "/row_attrs/_StableID"

## Error handling (same pattern as parse_v8.py)
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

## JSON serializer (handles numpy scalars, same as parse_v8.py)
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

def _write_dataset(f_out: h5py.File, path: str, data: np.ndarray, n_genes: int, n_cells: int) -> int:
    """Write a genes×cells matrix with standard chunking/compression. Returns on-disk size in bytes."""
    chunk = (min(n_genes, LOOM_CHUNK_GENES), min(n_cells, LOOM_CHUNK_CELLS))
    if path in f_out:
        del f_out[path]
    ds = f_out.create_dataset(path, data=data.astype(LOOM_DTYPE, copy=False), shape=(n_genes, n_cells), chunks=chunk, compression=LOOM_COMPRESSION, compression_opts=LOOM_COMPRESSION_LEVEL)
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


## Main scaling function
def scale(args):
    input_path = Path(args.f).resolve()
    if not input_path.is_file():
        ErrorJSON(f"Input LOOM file not found: {args.f}")

    out_dir = Path(args.o).resolve() if args.o else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    output_json_path = str(out_dir / OUTPUT_JSON_NAME)

    if not args.output_meta.startswith("/layers/"):
        ErrorJSON(f"--output_meta must be a path under /layers/ (e.g. /layers/scaled_1_toto), got: {args.output_meta!r}")

    try:
        import scanpy as sc
        from importlib.metadata import version as _pkg_version
        tool_version = _pkg_version("scanpy")
    except ImportError:
        ErrorJSON("scanpy is not installed. Install it with: pip install scanpy")

    warnings: list[str] = []

    result: dict = {
        "parameters": {},
        "scaling":    {
            "tool":         "scanpy",
            "tool_version": tool_version,
            "zero_center":  args.zero_center,
            "max_value":    args.max_value,
        },
        "metadata":   [],
        "wait_time":  0,
        "warnings":   [],
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
            # Matrix stored cells×genes — transpose to canonical genes×cells
            n_genes, n_cells = s1, s0
            matrix_gxc = node[:, :].astype(DEFAULT_NP_DTYPE).T
        else:
            # Standard genes×cells
            n_genes, n_cells = s0, s1
            matrix_gxc = node[:, :].astype(DEFAULT_NP_DTYPE)

        result["parameters"] = {
            "loom_path":        str(input_path),
            "input_loom_path":  args.input_meta,
            "output_loom_path": args.output_meta,
            "method":           args.method,
            "zero_center":      args.zero_center,
            "max_value":        args.max_value,
        }

        # Build AnnData (cells × genes) and run sc.pp.scale
        import anndata
        adata = anndata.AnnData(X=matrix_gxc.T)   # cells × genes
        sc.pp.scale(adata, zero_center=args.zero_center, max_value=args.max_value)
        scaled_gxc = adata.X.T   # back to genes × cells

    # Append scaled layer directly into the input LOOM (no copy needed)
    f_rw, wt = _open_loom_with_retry(str(input_path), "r+")
    result["wait_time"] += wt
    with f_rw:
        if args.output_meta in f_rw:
            warnings.append(f"Path '{args.output_meta}' already exists in the LOOM file and will be overwritten.")
            del f_rw[args.output_meta]
        f_rw.require_group("/layers")
        scaled_size = _write_dataset(f_rw, args.output_meta, scaled_gxc, n_genes, n_cells)

    # Metadata entry — mirrors the Metadata dataclass from parse_v8.py
    is_count_scaled = int(np.all(scaled_gxc == np.floor(scaled_gxc)))
    result["metadata"].append({
        "name":            args.output_meta,
        "on":              "EXPRESSION_MATRIX",
        "type":            "NUMERIC",
        "nber_cols":       n_cells,
        "nber_rows":       n_genes,
        "dataset_size":    scaled_size,
        "is_count_table":  is_count_scaled,
        "imported":        0,
    })

    if warnings:
        result["warnings"] = warnings
    else:
        del result["warnings"]

    json_str = json.dumps(result, default=_json_default, ensure_ascii=False, separators=(",", ":"))
    if args.o:
        with open(output_json_path, "w", encoding="utf-8") as fj:
            fj.write(json_str)
    else:
        print(json_str)  # stdout only when -o is not set

## CLI
HELP_TEXT = """
Scaling Script — scanpy backend

Reads a LOOM file, scales the selected layer (per-gene mean centering and/or
std normalization), appends the scaled layer directly into the input LOOM file,
and writes output.json.

Options:
  -f             Input LOOM file.                                            [required]
  --input_meta   Path to the matrix to scale inside the LOOM file.          [required]
                   Examples: /layers/normalized   /layers/normalized_ln
  --output_meta  Path where the scaled matrix is stored in the LOOM file.   [required]
                 Must be under /layers/.
                   Examples: /layers/scaled   /layers/scaled_1_toto
  --method       Scaling method: scale                                       [required]
  -o             Output folder.                       [optional, default: input dir]

  ── scale parameters (defaults match sc.pp.scale) ────────────────────────────────
  --zero_center  Center each gene to mean zero.            [default: True]
  --no_zero_center  Disable centering (sets zero_center=False).
  --max_value    Clip values to this threshold after scaling.
                   [default: None → no clipping]

  --help         Show this message and exit.

Output JSON scaling block:
  tool          str  — always 'scanpy'
  tool_version  str  — installed scanpy version
  zero_center   bool — whether mean-centering was applied
  max_value     num  — clip threshold (null if not set)

Output JSON metadata entry (mirrors parse_v8.py Metadata dataclass):
  name            str  — LOOM-internal path of the scaled layer (= --output_meta)
  on              str  — always 'EXPRESSION_MATRIX'
  type            str  — always 'NUMERIC'
  nber_cols       int  — number of cells
  nber_rows       int  — number of genes
  dataset_size    int  — on-disk compressed size in bytes
  is_count_table  int  — 1 if all values are integers, 0 otherwise
  imported        int  — always 0 (generated, not imported from source)

Mandatory parameters: -f, --input_meta, --output_meta, --method
"""

def main():
    if "--help" in sys.argv:
        print(HELP_TEXT)
        sys.exit(0)

    parser = argparse.ArgumentParser(description="Scaling Script (scanpy)", add_help=False)

    # Core I/O — all mandatory except -o
    parser.add_argument("-f",            required=True,  metavar="Input LOOM file")
    parser.add_argument("-o",            required=False, metavar="Output folder", default=None)
    parser.add_argument("--input_meta",  required=True,  metavar="LOOM-internal path to scale (e.g. /layers/normalized)")
    parser.add_argument("--output_meta", required=True,  metavar="LOOM-internal path for the scaled output (e.g. /layers/scaled_1_toto)")
    parser.add_argument("--method",      required=True,  choices=["scale"], metavar="Scaling method")

    # scale parameters (defaults match sc.pp.scale)
    parser.add_argument("--zero_center",    dest="zero_center", action="store_true",  default=True,  help="[default: True]")
    parser.add_argument("--no_zero_center", dest="zero_center", action="store_false",                help="Disable centering")
    parser.add_argument("--max_value",      type=float,         default=None,                        help="[default: None = no clipping]")

    args = parser.parse_args()
    scale(args)

if __name__ == "__main__":
    main()