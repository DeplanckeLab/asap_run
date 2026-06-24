from __future__ import annotations
import argparse
import json
import sys
import h5py
import numpy as np
from pathlib import Path
from typing import Any, Optional

DEFAULT_NP_DTYPE       = np.float64
LOOM_COMPRESSION       = "gzip"
LOOM_COMPRESSION_LEVEL = 2
LOOM_DTYPE             = "float64"
OUTPUT_JSON_NAME       = "output.json"
_CELL_ID_PATH = "/col_attrs/CellID"
_GENE_ID_PATH = "/row_attrs/_StableID"

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

def _json_default(obj: Any):
    if isinstance(obj, np.bool_):    return bool(obj)
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    return str(obj)

def _write_1d_float(f_out: h5py.File, path: str, data: np.ndarray) -> int:
    n = len(data)
    if path in f_out: del f_out[path]
    grp = "/".join(path.split("/")[:-1])
    if grp: f_out.require_group(grp)
    ds = f_out.create_dataset(path, data=data.astype(LOOM_DTYPE, copy=False), shape=(n,),
                               chunks=(min(n, 1024),), compression=LOOM_COMPRESSION, compression_opts=LOOM_COMPRESSION_LEVEL)
    return int(ds.id.get_storage_size())

def _write_1d_int(f_out: h5py.File, path: str, data: np.ndarray) -> int:
    n = len(data)
    if path in f_out: del f_out[path]
    grp = "/".join(path.split("/")[:-1])
    if grp: f_out.require_group(grp)
    ds = f_out.create_dataset(path, data=data.astype(np.int32, copy=False), shape=(n,),
                               chunks=(min(n, 1024),), compression=LOOM_COMPRESSION, compression_opts=LOOM_COMPRESSION_LEVEL)
    return int(ds.id.get_storage_size())

def _open_loom_with_retry(path: str, mode: str, max_wait: int = 600) -> tuple:
    import time as _time
    elapsed = 0
    while True:
        try:
            return h5py.File(path, mode), elapsed
        except OSError as e:
            if elapsed >= max_wait:
                ErrorJSON(f"Could not open {path!r} in mode {mode!r} after {max_wait}s: {e}")
            _time.sleep(1); elapsed += 1

def doublet_score(args):
    input_path = Path(args.f).resolve()
    if not input_path.is_file():
        ErrorJSON(f"Input LOOM file not found: {args.f}")

    out_dir = Path(args.o).resolve() if args.o else None
    if out_dir: out_dir.mkdir(parents=True, exist_ok=True)
    output_json_path = str(out_dir / OUTPUT_JSON_NAME) if out_dir else None
    try:
        import os as _os
        _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        _os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        import scrublet as scr
        from importlib.metadata import version as _pkg_version
        tool_version = _pkg_version("scrublet")
    except ImportError:
        ErrorJSON("scrublet is not installed. Install it with: pip install scrublet")

    warnings: list[str] = []

    result: dict = {
        "parameters":    {},
        "doublet_score": {"tool": "scrublet", "tool_version": tool_version,
                          "doublet_rate_auto": False,
                          "threshold_auto": None, "threshold_auto_n_doublets": -1,
                          "threshold_used": None, "nExp": -1,
                          "n_doublets_called": -1, "n_singlets_called": -1},
        "metadata":      [],
        "wait_time":     0,
    }

    # Read count matrix
    f_in, wt = _open_loom_with_retry(str(input_path), "r")
    result["wait_time"] += wt
    with f_in:
        src_path = args.input_meta
        if src_path not in f_in:
            ErrorJSON(f"Source path {src_path!r} not found in LOOM file.")
        node = f_in[src_path]
        if not isinstance(node, h5py.Dataset) or len(node.shape) != 2:
            ErrorJSON(f"Expected a 2-D dataset at {src_path}.")

        ca_len = f_in[_CELL_ID_PATH].shape[0]
        s0, s1 = int(node.shape[0]), int(node.shape[1])

        # Scrublet needs cells × genes (dense or sparse)
        # Orientation detection: compare s0/s1 against n_cells
        from scipy.sparse import csc_matrix
        if s0 == ca_len:
            # Python-written: on-disk (n_cells, n_genes) — cells already on axis 0
            counts_matrix = csc_matrix(node[:, :])
            n_cells, n_genes = s0, s1
        else:
            # R-written: on-disk (n_genes, n_cells) due to hdf5r transpose — transpose back
            counts_matrix = csc_matrix(node[:, :].T)
            n_cells, n_genes = s1, s0

    # Expected doublet rate
    if args.doublet_rate is not None:
        expected_dr = args.doublet_rate
        doublet_rate_auto = False
    else:
        expected_dr = min((0.8 * n_cells) / 100000, 0.10)
        doublet_rate_auto = True

    nExp = round(expected_dr * n_cells)

    result["parameters"] = {
        "loom_path":               str(input_path),
        "input_loom_path":         args.input_meta,
        "output_score_loom_path":  args.output_score_meta,
        "output_call_loom_path":   args.output_call_meta,
        "method":                  args.method,
        "n_pcs":                   args.n_pcs,
        "expected_doublet_rate":   expected_dr,
        "random_state":            args.random_state,
    }
    result["doublet_score"]["doublet_rate_auto"] = doublet_rate_auto
    result["doublet_score"]["nExp"]              = nExp

    # Validate input looks like raw counts (non-negative, not all tiny floats)
    sample = counts_matrix[:100, :].toarray() if hasattr(counts_matrix, 'toarray') else counts_matrix[:100, :]
    if sample.min() < 0:
        ErrorJSON(f"--input_meta appears to contain negative values (min={sample.min():.3f}). "
                  f"Scrublet requires raw integer counts, not normalized or scaled data. "
                  f"Use the raw count matrix (e.g. /matrix).")
    if sample.max() < 1 and n_genes > 100:
        ErrorJSON(f"--input_meta values are all < 1 (max={sample.max():.4f}). "
                  f"Scrublet requires raw integer counts, not normalized data. "
                  f"Use the raw count matrix (e.g. /matrix).")

    # Run Scrublet
    scrub = scr.Scrublet(counts_matrix, expected_doublet_rate=expected_dr, random_state=args.random_state)
    doublet_scores, predicted_doublets_auto = scrub.scrub_doublets(n_prin_comps=args.n_pcs, verbose=False)
    doublet_scores = np.array(doublet_scores, dtype=DEFAULT_NP_DTYPE)

    threshold_auto  = float(scrub.threshold_)
    auto_n_doublets = int(np.sum(predicted_doublets_auto))
    result["doublet_score"]["threshold_auto"]          = threshold_auto
    result["doublet_score"]["threshold_auto_n_doublets"] = auto_n_doublets

    # Calling strategy: auto-threshold by default; --doublet_rate overrides with top-N
    if args.doublet_rate is not None:
        # Override: call exactly nExp top-scoring cells as doublets
        calls = np.zeros(n_cells, dtype=np.int32)
        top_idx = np.argsort(doublet_scores)[::-1][:nExp]
        calls[top_idx] = 1
        threshold_used = None
        pass  # override noted in doublet_score.threshold_used = None
    else:
        # Default: use scrublet's auto-threshold
        calls = predicted_doublets_auto.astype(np.int32)
        threshold_used = threshold_auto

    n_doublets_called = int(calls.sum())
    result["doublet_score"]["threshold_used"]    = threshold_used
    result["doublet_score"]["doublet_rate"]      = round(n_doublets_called / n_cells, 6)
    result["doublet_score"]["n_doublets_called"] = n_doublets_called
    result["doublet_score"]["n_singlets_called"] = int(n_cells - n_doublets_called)

    # Write to LOOM
    f_rw, wt = _open_loom_with_retry(str(input_path), "r+")
    result["wait_time"] += wt
    with f_rw:
        for p in [args.output_score_meta, args.output_call_meta]:
            if p in f_rw or p.lstrip("/") in f_rw:
                warnings.append(f"Path '{p}' already exists and will be overwritten.")
        size_score = _write_1d_float(f_rw, args.output_score_meta, doublet_scores)
        size_call  = _write_1d_int(f_rw,   args.output_call_meta,  calls)

    result["metadata"] = [
        {"name": args.output_score_meta, "on": "CELL", "type": "NUMERIC",  "nber_rows": n_cells, "nber_cols": 1, "dataset_size": size_score, "imported": 0},
        {"name": args.output_call_meta,  "on": "CELL", "type": "INTEGER",  "nber_rows": n_cells, "nber_cols": 1, "dataset_size": size_call,  "imported": 0},
    ]

    if warnings: result["warnings"] = warnings

    json_str = json.dumps(result, default=_json_default, ensure_ascii=False, separators=(",", ":"))
    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as fj: fj.write(json_str)
    else:
        print(json_str)

HELP_TEXT = """
Doublet Score Script — scrublet backend

Reads a raw count matrix, runs scrublet, writes score + binary call into LOOM.

Default calling uses scrublet's auto-threshold. Use --doublet_rate to override
with top-N calling (more robust when auto-threshold is unreliable).

Options:
  -f                     Input LOOM file.                                   [required]
  --input_meta           Path to the raw count matrix.                      [required]
  --output_score_meta    Path for doublet score in /col_attrs/.             [required]
  --output_call_meta     Path for binary call (0/1) in /col_attrs/.         [required]
  --method               scrublet                                            [required]
  -o                     Output folder for output.json and plots. [optional]

  --n_pcs                PCs for scrublet.                    [default: 30]
  --doublet_rate         Override auto-threshold: call top X fraction
                           of cells as doublets.              [default: None → auto]
  --random_state         Random seed.                         [default: 42]

  --help                 Show this message and exit.
"""

def main():
    if "--help" in sys.argv:
        print(HELP_TEXT); sys.exit(0)

    parser = argparse.ArgumentParser(description="Doublet Score Script (scrublet)", add_help=False)
    parser.add_argument("-f",                     required=True)
    parser.add_argument("-o",                     required=False, default=None)
    parser.add_argument("--input_meta",           required=True)
    parser.add_argument("--output_score_meta",    required=True)
    parser.add_argument("--output_call_meta",     required=True)
    parser.add_argument("--method",               required=True, choices=["scrublet"])
    parser.add_argument("--n_pcs",                type=int,   default=30)
    parser.add_argument("--doublet_rate",         type=float, default=None)
    parser.add_argument("--random_state",         type=int,   default=42)

    args = parser.parse_args()
    doublet_score(args)

if __name__ == "__main__":
    main()