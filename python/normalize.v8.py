from __future__ import annotations
# OpenBLAS in the numpy/scipy wheels supports at most 64 threads. On nodes with more cores, OpenMP code that calls BLAS from
# every thread (e.g. scikit-learn neighbour search in t-SNE) overflows that limit and segfaults ("OpenBLAS warning:
# precompiled NUM_THREADS exceeded ... Segmentation fault"). Cap both pools before numpy/scipy/sklearn are loaded.
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _n = _os.environ.get(_v, "").strip()
    _os.environ[_v] = str(max(1, min(int(_n) if _n.isdigit() else (_os.cpu_count() or 1), 64)))
import argparse
import json
import sys
import h5py
import numpy as np
from pathlib import Path
from typing import Any, Optional

## Constants (matching parse_v8 conventions)
DEFAULT_NP_DTYPE       = np.float64  # in-memory computation precision (always float64)
LOOM_CHUNK_GENES       = 64
LOOM_CHUNK_CELLS       = 64
LOOM_COMPRESSION       = "gzip"
LOOM_COMPRESSION_LEVEL = 2
LOOM_DTYPE             = "float64"  # on-disk precision: "float32" or "float64"
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


## Block helpers (bounded RAM: the matrix is never held in memory as a whole)
BLOCK_BYTES = 64 * 1024**2  # target size of one float64 block

def _block_len(other_dim: int, chunk: int) -> int:
    """Block length so that one float64 (block x other_dim) slab stays near BLOCK_BYTES; a multiple of the loom chunk."""
    n = BLOCK_BYTES // (8 * max(1, other_dim))
    return max(chunk, n // chunk * chunk)

def _read_cells(node: h5py.Dataset, c0: int, c1: int, cells_first: bool) -> np.ndarray:
    """genes x cells float64 block for cells [c0, c1)."""
    if cells_first:
        return node[c0:c1, :].astype(DEFAULT_NP_DTYPE, copy=False).T
    return node[:, c0:c1].astype(DEFAULT_NP_DTYPE, copy=False)

## Normalization implementations (mirror sc.pp.normalize_total + sc.pp.log1p, one block of cells at a time)
def _cell_totals(node: h5py.Dataset, blocks: list, n_genes: int, n_cells: int, cells_first: bool, exclude_highly_expressed: bool, max_fraction: float) -> np.ndarray:
    """Per-cell totals; with exclude_highly_expressed, totals over genes whose max per-cell fraction is <= max_fraction."""
    cell_sums = np.empty(n_cells, dtype=DEFAULT_NP_DTYPE)
    gene_max_frac = np.zeros(n_genes, dtype=DEFAULT_NP_DTYPE) if exclude_highly_expressed else None
    for c0, c1 in blocks:
        blk  = _read_cells(node, c0, c1, cells_first)
        sums = blk.sum(axis=0)
        cell_sums[c0:c1] = sums
        if exclude_highly_expressed:
            with np.errstate(divide="ignore", invalid="ignore"):
                np.divide(blk, sums[np.newaxis, :], out=blk, where=sums[np.newaxis, :] != 0)
            blk[:, sums == 0] = 0
            np.maximum(gene_max_frac, blk.max(axis=1), out=gene_max_frac)
    if exclude_highly_expressed:
        keep = gene_max_frac <= max_fraction
        for c0, c1 in blocks:
            cell_sums[c0:c1] = _read_cells(node, c0, c1, cells_first)[keep, :].sum(axis=0)
    return cell_sums

def _log_label(base: Optional[float]) -> tuple[str, Optional[float]]:
    """base: None → natural log (ln) | 2 → log2(x+1) | 10 → log10(x+1) | other → log_base(x+1)."""
    if base is None: return "ln (natural log, log1p)", None
    if base == 2:    return "log2 (log2(x+1))", 2.0
    if base == 10:   return "log10 (log10(x+1))", 10.0
    return f"log{base} (log{base}(x+1))", float(base)

## Main normalization function
def normalize(args):
    input_path = Path(args.f).resolve()
    if not input_path.is_file():
        ErrorJSON(f"Input LOOM file not found: {args.f}")

    out_dir = Path(args.o).resolve() if args.o else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    output_json_path = str(out_dir / OUTPUT_JSON_NAME)

    if not args.output_meta.startswith("/layers/"):
        ErrorJSON(f"--output_meta must be a path under /layers/ (e.g. /layers/normalized_1_toto), got: {args.output_meta!r}")

    try:
        import scanpy as sc
        from importlib.metadata import version as _pkg_version
        tool_version = _pkg_version("scanpy")
    except ImportError:
        ErrorJSON("scanpy is not installed. Install it with: pip install scanpy")

    warnings: list[str] = []

    result: dict = {
        "parameters":       {},
        "normalization":    {"tool": "scanpy", "tool_version": tool_version, "is_log_transformed": False, "log_type": None, "log_base": None},
        "metadata":         [],
        "wait_time":        0,
        "warnings":         [],
    }

    if args.method != "normalize_total":
        ErrorJSON(f"Unknown normalization method: {args.method!r}. Available: normalize_total")
    result["parameters"] = {
        "loom_path":                str(input_path),
        "input_loom_path":          args.input_meta,
        "output_loom_path":         args.output_meta,
        "method":                   args.method,
        "target_sum":               args.target_sum,
        "exclude_highly_expressed": args.exclude_highly_expressed,
        "max_fraction":             args.max_fraction if args.exclude_highly_expressed else None,
        "key_added":                args.key_added,
    }
    if args.log:
        log_label, log_base_used = _log_label(args.log_base)
        result["normalization"]["is_log_transformed"] = True
        result["normalization"]["log_type"]           = log_label
        result["normalization"]["log_base"]           = log_base_used

    # The normalized layer is built block by block in a scratch HDF5 file, then copied into the LOOM:
    # RAM stays near one block, and a failure never leaves a half-written layer in the LOOM.
    import os, tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=".normalize_", suffix=".h5", dir=str(out_dir))
    os.close(tmp_fd)
    try:
        f_in, wt = _open_loom_with_retry(str(input_path), "r")
        result["wait_time"] += wt
        with f_in, h5py.File(tmp_path, "w") as f_tmp:
            src_path = args.input_meta
            if src_path not in f_in:
                ErrorJSON(f"Source path {src_path!r} not found in LOOM file.")
            node = f_in[src_path]
            if not isinstance(node, h5py.Dataset) or len(node.shape) != 2:
                ErrorJSON(f"Expected a 2-D dataset at {src_path}, got shape {getattr(node, 'shape', '?')}")

            s0, s1 = int(node.shape[0]), int(node.shape[1])
            ra_len = f_in[_GENE_ID_PATH].shape[0]
            ca_len = f_in[_CELL_ID_PATH].shape[0]
            cells_first = s0 == ca_len and s1 == ra_len   # matrix stored cells×genes (else canonical genes×cells)
            n_genes, n_cells = (s1, s0) if cells_first else (s0, s1)
            step   = _block_len(n_genes, LOOM_CHUNK_CELLS)
            blocks = [(c0, min(c0 + step, n_cells)) for c0 in range(0, n_cells, step)]

            # Pass 1: per-cell totals → effective target sum (median of non-zero totals unless --target_sum)
            cell_sums = _cell_totals(node, blocks, n_genes, n_cells, cells_first, args.exclude_highly_expressed, args.max_fraction)
            non_zero  = cell_sums[cell_sums > 0]
            effective_target = float(args.target_sum) if args.target_sum is not None else (float(np.median(non_zero)) if len(non_zero) > 0 else 1.0)
            result["parameters"]["effective_target_sum"] = effective_target

            # Pass 2: normalize (+ log1p) each block of cells in place and write it out
            out = f_tmp.create_dataset("layer", shape=(n_genes, n_cells), dtype=LOOM_DTYPE,
                                       chunks=(min(n_genes, LOOM_CHUNK_GENES), min(n_cells, LOOM_CHUNK_CELLS)),
                                       compression=LOOM_COMPRESSION, compression_opts=LOOM_COMPRESSION_LEVEL)
            is_count_norm = 1
            for c0, c1 in blocks:
                blk  = _read_cells(node, c0, c1, cells_first)
                sums = cell_sums[np.newaxis, c0:c1]
                blk *= effective_target
                with np.errstate(divide="ignore", invalid="ignore"):
                    np.divide(blk, sums, out=blk, where=sums != 0)
                blk[:, cell_sums[c0:c1] == 0] = 0
                if args.log:
                    np.log1p(blk, out=blk)
                    if log_base_used is not None:
                        blk /= np.log(log_base_used)
                if is_count_norm and not np.all(blk == np.floor(blk)):
                    is_count_norm = 0
                out[:, c0:c1] = blk

        # Append normalized layer into the input LOOM
        f_rw, wt = _open_loom_with_retry(str(input_path), "r+")
        result["wait_time"] += wt
        with f_rw, h5py.File(tmp_path, "r") as f_tmp:
            if args.output_meta in f_rw:
                warnings.append(f"Path '{args.output_meta}' already exists in the LOOM file and will be overwritten.")
                del f_rw[args.output_meta]
            f_rw.require_group("/layers")
            f_rw.copy(f_tmp["layer"], args.output_meta)
            norm_ds = f_rw[args.output_meta]
            if args.log:
                # Record the log base on the layer so downstream steps (hvg.v8.py, flavor=seurat) can undo the log correctly.
                norm_ds.attrs["log1p_base"] = float(log_base_used) if log_base_used is not None else float(np.e)
            norm_size = int(norm_ds.id.get_storage_size())
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Metadata entry — mirrors the Metadata dataclass from parse_v8.py
    result["metadata"].append({
        "name":            args.output_meta,
        "on":              "EXPRESSION_MATRIX",
        "type":            "NUMERIC",
        "nber_cols":       n_cells,
        "nber_rows":       n_genes,
        "dataset_size":    norm_size,
        "is_count_table":  is_count_norm,
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
Normalization Script — scanpy backend

Reads a LOOM file (from parse.py), normalizes the selected path, appends the
normalized layer directly into the input LOOM file, and writes output.json.

Options:
  -f             Input LOOM file (from parse.py).                           [required]
  --input_meta   Path to the matrix to normalize inside the LOOM file.     [required]
                   Examples: /matrix   /layers/spliced   /layers/toto
  --output_meta  Path where the normalized matrix is stored in the output   [required]
                 LOOM file. Must be under /layers/.
                   Examples: /layers/normalized   /layers/normalized_1_toto
  --method       Normalization method: normalize_total                      [required]
  -o             Output folder.                      [optional, default: input dir]

  ── normalize_total parameters (defaults match sc.pp.normalize_total) ─────────────
  --target_sum                 Normalize each cell to this many total counts.
                                 [default: None → normalized to median total count]
  --exclude_highly_expressed   Exclude genes whose per-cell fraction exceeds
                                 --max_fraction from the normalization factor.
                                 [default: False]
  --max_fraction               Max per-cell fraction before a gene is excluded
                                 (only used when --exclude_highly_expressed is set).
                                 [default: 0.05]
  --key_added                  Record per-cell totals under this key in the JSON.
                                 [default: None]

  ── Log transformation, applied after normalization (defaults match sc.pp.log1p) ───
  --log         Apply log1p transformation after normalization.  [default: not applied]
  --log_base    Log base: None → ln (natural log) | 2 → log2 | 10 → log10
                  [default: None → ln]

  --help        Show this message and exit.

Output JSON normalization block:
  is_log_transformed  bool   — whether log1p was applied
  log_type            str    — e.g. "ln (natural log, log1p)" or "log2 (log2(x+1))"
  log_base            float  — 2.0, 10.0, or null for natural log

Output JSON metadata entry (mirrors parse_v8.py Metadata dataclass):
  name            str  — LOOM-internal path of the normalized layer (= --output_meta)
  on              str  — always "EXPRESSION_MATRIX"
  type            str  — always "NUMERIC"
  dataset_size    int  — on-disk compressed size in bytes
  is_count_table  int  — 1 if all values are integers, 0 otherwise
  imported        int  — always 0 (generated, not imported from source)

Mandatory parameters: -f, --input_meta, --output_meta, --method
"""

def main():
    if "--help" in sys.argv:
        print(HELP_TEXT)
        sys.exit(0)

    parser = argparse.ArgumentParser(description="Normalization Script (scanpy)", add_help=False)

    # Core I/O — -f, --input_meta, --output_meta, --method are all mandatory
    parser.add_argument("-f",            required=True,  metavar="Input LOOM file")
    parser.add_argument("-o",            required=False, metavar="Output folder", default=None)
    parser.add_argument("--input_meta",  required=True,  metavar="LOOM-internal path to normalize (e.g. /matrix or /layers/toto)")
    parser.add_argument("--output_meta", required=True,  metavar="LOOM-internal path for the normalized output (e.g. /layers/normalized_1_toto)")
    parser.add_argument("--method",      required=True,  choices=["normalize_total"], metavar="Normalization method")

    # normalize_total parameters (defaults match sc.pp.normalize_total)
    parser.add_argument("--target_sum",               type=float, default=None,  help="[default: None = median]")
    parser.add_argument("--exclude_highly_expressed", action="store_true",       help="[default: False]")
    parser.add_argument("--max_fraction",             type=float, default=0.05,  help="[default: 0.05]")
    parser.add_argument("--key_added",                type=str,   default=None,  help="[default: None]")

    # Log transformation parameters (defaults match sc.pp.log1p)
    parser.add_argument("--log",      action="store_true",            help="[default: not applied]")
    parser.add_argument("--log_base", type=float, default=None,       help="[default: None = ln]")

    args = parser.parse_args()
    normalize(args)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # uncaught library error: still report a displayed_error (ErrorJSON itself exits via SystemExit)
        import traceback
        traceback.print_exc(file=sys.stderr)
        ErrorJSON(f"Unexpected error: {type(e).__name__}: {e}")
