from __future__ import annotations
import argparse
import json
import sys
import h5py
import numpy as np
from pathlib import Path
from typing import Any, Optional

## Constants (matching parse_v8 conventions)
DEFAULT_NP_DTYPE       = np.float64
LOOM_CHUNK_GENES       = 1024
LOOM_CHUNK_CELLS       = 1024
LOOM_COMPRESSION       = "gzip"
LOOM_COMPRESSION_LEVEL = 4
LOOM_DTYPE             = "float32"
NORMALIZED_LAYER       = "/layers/normalized"
OUTPUT_LOOM_NAME       = "normalized.loom"
OUTPUT_JSON_NAME       = "output.json"

_CELL_ID_CANDIDATES = ["/col_attrs/CellID", "/col_attrs/cell_id", "/col_attrs/barcode", "/col_attrs/Barcode", "/col_attrs/barcodes", "/col_attrs/obs_names"]
_GENE_ID_CANDIDATES = ["/row_attrs/Gene", "/row_attrs/Accession", "/row_attrs/Original_Gene", "/row_attrs/gene_name", "/row_attrs/gene", "/row_attrs/var_names"]

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

def _infer_cell_ids(f: h5py.File, n_cells: int) -> list[str]:
    for p in _CELL_ID_CANDIDATES:
        if p in f and isinstance(f[p], h5py.Dataset):
            vals = _read_string_dataset(f, p)
            if len(vals) == n_cells:
                return vals
    return [f"Cell_{i+1}" for i in range(n_cells)]

def _infer_gene_ids(f: h5py.File, n_genes: int) -> list[str]:
    for p in _GENE_ID_CANDIDATES:
        if p in f and isinstance(f[p], h5py.Dataset):
            vals = _read_string_dataset(f, p)
            if len(vals) == n_genes:
                return vals
    return [f"Gene_{i+1}" for i in range(n_genes)]

def _write_dataset(f_out: h5py.File, path: str, data: np.ndarray, n_genes: int, n_cells: int) -> int:
    """Write a genes×cells matrix with standard chunking/compression. Returns on-disk size in bytes."""
    chunk = (min(n_genes, LOOM_CHUNK_GENES), min(n_cells, LOOM_CHUNK_CELLS))
    if path in f_out:
        del f_out[path]
    ds = f_out.create_dataset(path, data=data.astype(LOOM_DTYPE, copy=False), shape=(n_genes, n_cells), chunks=chunk, compression=LOOM_COMPRESSION, compression_opts=LOOM_COMPRESSION_LEVEL)
    return int(ds.id.get_storage_size())

def _copy_metadata_from_loom(f_in: h5py.File, f_out: h5py.File, skip_top_keys: set[str], warnings: list[str]) -> None:
    """Copy all top-level groups from f_in to f_out, skipping keys in skip_top_keys."""
    for key in f_in.keys():
        if key in skip_top_keys or key in f_out:
            continue
        try:
            f_in.copy(key, f_out)
        except Exception as e:
            warnings.append(f"Could not copy '{key}' from input LOOM: {e}")

## Normalization implementations
def _normalize_total_impl(mat_cxg: np.ndarray, target_sum: Optional[float], exclude_highly_expressed: bool, max_fraction: float) -> tuple[np.ndarray, float]:
    """
    Mirror of sc.pp.normalize_total logic.
    mat_cxg: (cells × genes). Returns (normalized_cxg, effective_target_sum).
    """
    mat_cxg   = mat_cxg.astype(DEFAULT_NP_DTYPE, copy=True)
    cell_sums = mat_cxg.sum(axis=1)
    if exclude_highly_expressed:
        with np.errstate(divide="ignore", invalid="ignore"):
            fracs = np.divide(mat_cxg, cell_sums[:, np.newaxis], out=np.zeros_like(mat_cxg), where=cell_sums[:, np.newaxis] != 0)
        cell_sums = mat_cxg[:, fracs.max(axis=0) <= max_fraction].sum(axis=1)
    non_zero  = cell_sums[cell_sums > 0]
    effective = float(target_sum) if target_sum is not None else (float(np.median(non_zero)) if len(non_zero) > 0 else 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        normed = np.divide(mat_cxg * effective, cell_sums[:, np.newaxis], out=np.zeros_like(mat_cxg), where=cell_sums[:, np.newaxis] != 0)
    return normed, effective

def _log1p_transform(mat: np.ndarray, base: Optional[float]) -> tuple[np.ndarray, str, Optional[float]]:
    """
    Apply log1p transformation.
    base: None → natural log (ln) | 2 → log2(x+1) | 10 → log10(x+1) | other → log_base(x+1).
    Returns (transformed_matrix, log_type_label, base_or_None).
    """
    result = np.log1p(mat)
    if base is None: return result, "ln (natural log, log1p)", None
    if base == 2:    return result / np.log(2),  "log2 (log2(x+1))", 2.0
    if base == 10:   return result / np.log(10), "log10 (log10(x+1))", 10.0
    return result / np.log(base), f"log{base} (log{base}(x+1))", float(base)

## Main normalization function
def normalize(args):
    input_path = Path(args.f).resolve()
    if not input_path.is_file():
        ErrorJSON(f"Input LOOM file not found: {args.f}")

    out_dir = Path(args.o).resolve() if args.o else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    output_loom_path = str(out_dir / OUTPUT_LOOM_NAME)
    output_json_path = str(out_dir / OUTPUT_JSON_NAME)

    meta_path = Path(args.input_meta).resolve()
    if not meta_path.is_file():
        ErrorJSON(f"Input metadata JSON not found: {args.input_meta}")
    with open(meta_path, "r", encoding="utf-8") as fj:
        input_meta_list = json.load(fj).get("metadata", [])

    try:
        import scanpy as sc
        tool_version = sc.__version__
    except ImportError:
        ErrorJSON("scanpy is not installed. Install it with: pip install scanpy")

    warnings: list[str] = []

    result: dict = {
        "input_path":    str(input_path),
        "output_path":   output_loom_path,
        "tool":          "scanpy",
        "tool_version":  tool_version,
        "method":        args.method,
        "nber_rows":     -1,
        "nber_cols":     -1,
        "layer_path":    NORMALIZED_LAYER,
        "parameters":    {},
        "normalization": {"is_log_transformed": False, "log_type": None, "log_base": None},
        "statistics":    {},
        "metadata":      input_meta_list,
        "warnings":      [],
    }

    with h5py.File(str(input_path), "r") as f_in:
        # "matrix" → /matrix; any other value → /layers/<layer>
        src_path = "/matrix" if args.layer == "matrix" else f"/layers/{args.layer}"
        if src_path not in f_in:
            ErrorJSON(f"Source path {src_path!r} not found in LOOM file.")
        node = f_in[src_path]
        if not isinstance(node, h5py.Dataset) or len(node.shape) != 2:
            ErrorJSON(f"Expected a 2-D dataset at {src_path}, got shape {getattr(node, 'shape', '?')}")

        n_genes, n_cells = int(node.shape[0]), int(node.shape[1])
        result["nber_rows"] = n_genes
        result["nber_cols"] = n_cells
        matrix_gxc = node[:, :].astype(DEFAULT_NP_DTYPE)   # (n_genes × n_cells)

        result["statistics"]["min_value_before"]    = float(matrix_gxc.min())
        result["statistics"]["max_value_before"]    = float(matrix_gxc.max())
        result["statistics"]["nber_zeros_before"]   = int(np.sum(matrix_gxc == 0))
        result["statistics"]["median_depth_before"] = float(np.median(matrix_gxc.sum(axis=0)))

        if args.method == "normalize_total":
            result["parameters"] = {
                "source_layer":             args.layer,
                "target_sum":               args.target_sum,
                "exclude_highly_expressed": args.exclude_highly_expressed,
                "max_fraction":             args.max_fraction if args.exclude_highly_expressed else None,
                "key_added":                args.key_added,
            }
            normed_cxg, effective_target = _normalize_total_impl(matrix_gxc.T, args.target_sum, args.exclude_highly_expressed, args.max_fraction)
            result["parameters"]["effective_target_sum"] = effective_target
            normalized_gxc = normed_cxg.T   # back to (n_genes × n_cells)
        else:
            ErrorJSON(f"Unknown normalization method: {args.method!r}. Available: normalize_total")

        if args.log:
            normalized_gxc, log_label, log_base_used = _log1p_transform(normalized_gxc, args.log_base)
            result["normalization"]["is_log_transformed"] = True
            result["normalization"]["log_type"]           = log_label
            result["normalization"]["log_base"]           = log_base_used

        result["statistics"]["min_value_after"]    = float(normalized_gxc.min())
        result["statistics"]["max_value_after"]    = float(normalized_gxc.max())
        result["statistics"]["nber_zeros_after"]   = int(np.sum(normalized_gxc == 0))
        result["statistics"]["median_depth_after"] = float(np.median(normalized_gxc.sum(axis=0)))

        with h5py.File(output_loom_path, "w") as f_out:
            for grp in ["attrs", "col_attrs", "row_attrs", "layers", "row_graphs", "col_graphs"]:
                f_out.require_group(grp)
            f_out["attrs"].create_dataset("LOOM_SPEC_VERSION", data="3.0.0", dtype=h5py.string_dtype(encoding="utf-8"))
            _write_dataset(f_out, "/matrix", matrix_gxc, n_genes, n_cells)
            norm_size = _write_dataset(f_out, NORMALIZED_LAYER, normalized_gxc, n_genes, n_cells)
            result["statistics"]["normalized_layer_disk_size_bytes"] = norm_size
            _copy_metadata_from_loom(f_in, f_out, skip_top_keys={"matrix", "attrs"}, warnings=warnings)

    if warnings:
        result["warnings"] = warnings
    else:
        del result["warnings"]

    json_str = json.dumps(result, default=_json_default, ensure_ascii=False, indent=2)
    if args.o:
        with open(output_json_path, "w", encoding="utf-8") as fj:
            fj.write(json_str)
    else:
        print(json_str)  # stdout only when -o is not set

## CLI
HELP_TEXT = """
Normalization Script — scanpy backend

Reads a LOOM file (from parse.py), normalizes the selected layer, writes a new LOOM
(/layers/normalized = normalized data, /matrix = raw counts preserved) and output.json.

Options:
  -f             Input LOOM file (from parse.py).                           [required]
  --input_meta   Input metadata JSON (output.json from parse.py).           [required]
  --method       Normalization method: normalize_total                      [required]
  --layer        LOOM layer to normalize. Use "matrix" for /matrix.         [required]
                   Examples: "matrix", "spliced", "unspliced"
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
"""

def main():
    if "--help" in sys.argv:
        print(HELP_TEXT)
        sys.exit(0)

    parser = argparse.ArgumentParser(description="Normalization Script (scanpy)", add_help=False)

    # Core I/O — -f, --input_meta, --method, --layer are all mandatory
    parser.add_argument("-f",           required=True,  metavar="Input LOOM file")
    parser.add_argument("-o",           required=False, metavar="Output folder", default=None)
    parser.add_argument("--input_meta", required=True,  metavar="Input JSON (from parse.py)")
    parser.add_argument("--method",     required=True,  choices=["normalize_total"], metavar="Normalization method")
    parser.add_argument("--layer",      required=True,  metavar='LOOM layer ("matrix" or layer name)')

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
    main()
