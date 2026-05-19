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
LOOM_COMPRESSION       = "gzip"
LOOM_COMPRESSION_LEVEL = 2
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

def _write_dataset_1d_int(f_out: h5py.File, path: str, data: np.ndarray) -> int:
    """Write a 1D integer array (0/1 HVG flags). Returns on-disk size in bytes."""
    n = len(data)
    if path in f_out:
        del f_out[path]
    grp = "/".join(path.split("/")[:-1])
    if grp:
        f_out.require_group(grp)
    ds = f_out.create_dataset(path, data=data.astype(np.int32, copy=False),
                               shape=(n,), chunks=(min(n, 1024),),
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

## Main HVG function
def hvg(args):
    input_path = Path(args.f).resolve()
    if not input_path.is_file():
        ErrorJSON(f"Input LOOM file not found: {args.f}")

    out_dir = Path(args.o).resolve() if args.o else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    output_json_path = str(out_dir / OUTPUT_JSON_NAME) if out_dir else None
    plot_dir         = out_dir if out_dir else input_path.parent
    plot_png_path    = str(plot_dir / "plot.hvg.png")
    plot_json_path   = str(plot_dir / "plot.hvg.json")

    if not args.output_meta.startswith("/row_attrs/"):
        ErrorJSON(f"--output_meta must be under /row_attrs/ (e.g. /row_attrs/highly_variable), got: {args.output_meta!r}")

    try:
        import scanpy as sc
        from importlib.metadata import version as _pkg_version
        tool_version = _pkg_version("scanpy")
    except ImportError:
        ErrorJSON("scanpy is not installed. Install it with: pip install scanpy")

    warnings: list[str] = []

    result: dict = {
        "parameters": {},
        "hvg":        {"tool": "scanpy", "tool_version": tool_version,
                       "method": args.method, "n_top_genes": args.n_top_genes, "n_hvg_found": -1},
        "metadata":   [],
        "plots":      [],
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
            matrix_gxc = node[:, :].astype(DEFAULT_NP_DTYPE).T
        else:
            n_genes, n_cells = s0, s1
            matrix_gxc = node[:, :].astype(DEFAULT_NP_DTYPE)

        gene_ids = _read_string_dataset(f_in, _GENE_ID_PATH)
        cell_ids = _read_string_dataset(f_in, _CELL_ID_PATH)

    result["parameters"] = {
        "loom_path":        str(input_path),
        "input_loom_path":  args.input_meta,
        "output_loom_path": args.output_meta,
        "method":           args.method,
        "n_top_genes":      args.n_top_genes,
        "min_mean":         args.min_mean         if args.method in ("seurat", "seurat_v3", "cell_ranger") else None,
        "max_mean":         args.max_mean         if args.method in ("seurat", "seurat_v3", "cell_ranger") else None,
        "min_disp":         args.min_disp         if args.method in ("seurat", "seurat_v3", "cell_ranger") else None,
        "span":             args.span             if args.method == "seurat_v3"                            else None,
        "n_bins":           args.n_bins,
        "batch_key":        args.batch_key,
    }

    # Build AnnData (cells x genes) and run sc.pp.highly_variable_genes
    import anndata
    adata = anndata.AnnData(X=matrix_gxc.T.astype(DEFAULT_NP_DTYPE))
    adata.obs_names = list(cell_ids)
    adata.var_names = list(gene_ids)

    try:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes = args.n_top_genes,
            flavor      = args.method,
            min_mean    = args.min_mean,
            max_mean    = args.max_mean,
            min_disp    = args.min_disp,
            span        = args.span,
            n_bins      = args.n_bins,
            batch_key   = args.batch_key,
            inplace     = True,
        )
    except (ValueError, ImportError) as e:
        ErrorJSON(f"sc.pp.highly_variable_genes failed with method '{args.method}': {e}. "
                  f"For 'seurat_v3' this usually means skmisc is missing or has a binary incompatibility "
                  f"with the installed numpy version (try: pip install --upgrade skmisc).")

    # scanpy adds adata.var["highly_variable"] as a boolean column
    hvg_flags   = adata.var["highly_variable"].values.astype(np.int32)
    n_hvg_found = int(hvg_flags.sum())
    result["hvg"]["n_hvg_found"] = n_hvg_found

    # Append HVG flags into the LOOM
    f_rw, wt = _open_loom_with_retry(str(input_path), "r+")
    result["wait_time"] += wt
    with f_rw:
        out_path = args.output_meta
        if out_path in f_rw:
            warnings.append(f"Path '{out_path}' already exists and will be overwritten.")
        size_hvg = _write_dataset_1d_int(f_rw, out_path, hvg_flags)

    ## ── Generate plots ───────────────────────────────────────────────────────
    try:
        import plotly.graph_objects as go
        import plotly.io as pio

        var_df = adata.var.copy()
        # Determine y-axis column based on method
        if args.method == "seurat_v3":
            y_col, y_label = "variances_norm",    "Normalized variance"
        else:
            y_col, y_label = "dispersions_norm",  "Normalized dispersion"
        x_label = "Mean expression"

        hvg_mask    = var_df["highly_variable"].values
        x_all       = var_df["means"].values
        y_all       = var_df[y_col].values
        names_all   = var_df.index.tolist()

        fig = go.Figure()
        # Non-HVG trace
        non_mask = ~hvg_mask
        fig.add_trace(go.Scattergl(
            x=x_all[non_mask], y=y_all[non_mask],
            mode="markers", name="Other genes",
            text=[names_all[i] for i in range(len(names_all)) if non_mask[i]],
            hovertemplate="%{text}<br>mean=%{x:.3f}<br>" + y_label + "=%{y:.3f}<extra></extra>",
            marker=dict(color="#CCCCCC", size=3, opacity=0.6)
        ))
        # HVG trace
        fig.add_trace(go.Scattergl(
            x=x_all[hvg_mask], y=y_all[hvg_mask],
            mode="markers", name="HVG",
            text=[names_all[i] for i in range(len(names_all)) if hvg_mask[i]],
            hovertemplate="%{text}<br>mean=%{x:.3f}<br>" + y_label + "=%{y:.3f}<extra></extra>",
            marker=dict(color="#E87C3E", size=5, opacity=0.9)
        ))

        fig.update_layout(
            title=f"Highly Variable Genes — {args.method} ({n_hvg_found}/{n_genes} selected)",
            xaxis_title=x_label, yaxis_title=y_label,
            template="simple_white", width=800, height=600,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        )

        # Save plotly JSON
        pio.write_json(fig, plot_json_path)
        result["plots"].append({"path": plot_json_path, "type": "plotly_json"})

        # Save PNG (requires kaleido)
        try:
            pio.write_image(fig, plot_png_path, format="png", width=800, height=600, scale=1.5)
            result["plots"].append({"path": plot_png_path, "type": "png"})
        except Exception as e_png:
            warnings.append(f"PNG export failed (kaleido may not be installed): {e_png}")

    except Exception as e_plot:
        warnings.append(f"Plot generation failed: {e_plot}")

    result["metadata"] = [{"name": args.output_meta, "on": "GENE", "type": "INTEGER",
                            "nber_rows": n_genes, "nber_cols": 1,
                            "dataset_size": size_hvg, "imported": 0}]

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
HVG Script — scanpy backend

Options:
  -f             Input LOOM file.                                            [required]
  --input_meta   Path to the count/normalized matrix inside the LOOM.       [required]
  --output_meta  Path in /row_attrs/ where the 0/1 HVG flag is stored.      [required]
                   Example: /row_attrs/highly_variable
  --method       HVG method: seurat | seurat_v3 | cell_ranger               [required]
                   seurat, cell_ranger : expects log-normalized data as --input_meta
                   seurat_v3          : expects raw counts as --input_meta
  -o             Output folder for output.json. [optional, default: stdout]

  -- sc.pp.highly_variable_genes parameters --------------------------------------
  --n_top_genes   Number of top variable genes.              [default: 2000]
  --min_mean      Min mean expression threshold.             [default: 0.0125]
  --max_mean      Max mean expression threshold.             [default: 3.0]
  --min_disp      Min dispersion threshold.                  [default: 0.5]
  --span          Span for loess (seurat_v3 only).           [default: 0.3]
  --n_bins        Number of bins for mean expression.        [default: 20]
  --batch_key     Column in obs for batch correction.        [default: None]

  --help         Show this message and exit.
"""

def main():
    if "--help" in sys.argv:
        print(HELP_TEXT)
        sys.exit(0)

    parser = argparse.ArgumentParser(description="HVG Script (scanpy)", add_help=False)
    parser.add_argument("-f",            required=True)
    parser.add_argument("-o",            required=False, default=None)
    parser.add_argument("--input_meta",  required=True)
    parser.add_argument("--output_meta", required=True)
    parser.add_argument("--method",      required=True, choices=["seurat", "seurat_v3", "cell_ranger"])
    parser.add_argument("--n_top_genes", type=int,   default=2000)
    parser.add_argument("--min_mean",    type=float, default=0.0125)
    parser.add_argument("--max_mean",    type=float, default=3.0)
    parser.add_argument("--min_disp",    type=float, default=0.5)
    parser.add_argument("--span",        type=float, default=0.3)
    parser.add_argument("--n_bins",      type=int,   default=20)
    parser.add_argument("--batch_key",   type=str,   default=None)

    args = parser.parse_args()
    hvg(args)

if __name__ == "__main__":
    main()