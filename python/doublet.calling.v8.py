from __future__ import annotations
import argparse
import json
import sys
import h5py
import numpy as np
from pathlib import Path
from typing import Any, Optional

## Constants
LOOM_COMPRESSION       = "gzip"
LOOM_COMPRESSION_LEVEL = 2
OUTPUT_JSON_NAME       = "output.json"

_CELL_ID_PATH = "/col_attrs/CellID"

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
def _write_dataset_1d_int(f_out: h5py.File, path: str, data: np.ndarray) -> int:
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

## Main calling function
def doublet_call(args):
    input_path = Path(args.f).resolve()
    if not input_path.is_file():
        ErrorJSON(f"Input LOOM file not found: {args.f}")

    out_dir = Path(args.o).resolve() if args.o else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    output_json_path = str(out_dir / OUTPUT_JSON_NAME) if out_dir else None

    warnings: list[str] = []

    result: dict = {
        "parameters":   {},
        "doublet_call": {"method": args.method, "threshold_used": None,
                         "n_doublets_called": -1, "n_singlets_called": -1,
                         "doublet_rate": None},
        "metadata":     [],
        "wait_time":    0,
    }

    # Read score from LOOM
    f_in, wt = _open_loom_with_retry(str(input_path), "r")
    result["wait_time"] += wt
    with f_in:
        score_path = args.input_score_meta
        if score_path not in f_in:
            ErrorJSON(f"Score path {score_path!r} not found in LOOM file.")
        scores  = f_in[score_path][()].astype(np.float64).ravel()
        ca_len  = f_in[_CELL_ID_PATH].shape[0]

    n_cells = len(scores)
    if n_cells != ca_len:
        ErrorJSON(f"Score length ({n_cells}) does not match cell count ({ca_len}).")

    # Apply threshold strategy
    calls = np.zeros(n_cells, dtype=np.int32)

    if args.method == "threshold":
        calls[scores >= args.threshold] = 1
        threshold_used = float(args.threshold)
    elif args.method == "top_n":
        n_d     = min(args.n_doublets, n_cells)
        top_idx = np.argsort(scores)[::-1][:n_d]
        calls[top_idx] = 1
        # threshold_used = minimum score among called doublets
        threshold_used = float(scores[top_idx[-1]]) if n_d > 0 else None
    elif args.method == "top_pct":
        n_d     = round(args.doublet_rate * n_cells)
        top_idx = np.argsort(scores)[::-1][:n_d]
        calls[top_idx] = 1
        threshold_used = float(scores[top_idx[-1]]) if n_d > 0 else None
    elif args.method == "valley":  # histogram valley detection on the observed score distribution
        from scipy.signal import find_peaks
        hist_counts, bin_edges = np.histogram(scores, bins=100)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        valleys, _ = find_peaks(-hist_counts.astype(float))
        if len(valleys) > 0:
            mid = (scores.max() + scores.min()) / 2
            best_valley = valleys[np.argmin(np.abs(bin_centers[valleys] - mid))]
            threshold_used = float(bin_centers[best_valley])
        else:
            threshold_used = float(np.mean(scores) + np.std(scores))
            warnings.append("No valley found in score distribution; using mean+std fallback.")
        calls[scores >= threshold_used] = 1
    else:  # auto — uses skimage.filters.threshold_minimum (scrublet's actual algorithm)
        try:
            from skimage.filters import threshold_minimum
            threshold_used = float(threshold_minimum(scores))
        except Exception as e_thresh:
            threshold_used = float(np.mean(scores) + np.std(scores))
            warnings.append(f"skimage threshold_minimum failed ({e_thresh}); using mean+std fallback.")
        calls[scores >= threshold_used] = 1

    n_doublets = int(calls.sum())
    n_singlets = n_cells - n_doublets
    doublet_rate_actual = round(n_doublets / n_cells, 6)

    result["parameters"] = {
        "loom_path":              str(input_path),
        "input_score_loom_path":  args.input_score_meta,
        "output_call_loom_path":  args.output_call_meta,
        "method":                 args.method,
        "threshold":              args.threshold,
        "n_doublets":             args.n_doublets,
        "doublet_rate":           args.doublet_rate,   # null unless user explicitly passed --doublet_rate
    }
    result["doublet_call"]["threshold_used"]    = threshold_used
    result["doublet_call"]["n_doublets_called"] = n_doublets
    result["doublet_call"]["n_singlets_called"] = n_singlets
    result["doublet_call"]["doublet_rate"]      = doublet_rate_actual

    # Write binary call to LOOM
    f_rw, wt = _open_loom_with_retry(str(input_path), "r+")
    result["wait_time"] += wt
    with f_rw:
        out_path = args.output_call_meta
        if out_path in f_rw or out_path.lstrip("/") in f_rw:
            warnings.append(f"Path '{out_path}' already exists and will be overwritten.")
        size_call = _write_dataset_1d_int(f_rw, out_path, calls)

    result["metadata"] = [{"name": args.output_call_meta, "on": "CELL", "type": "INTEGER",
                            "nber_rows": n_cells, "nber_cols": 1,
                            "dataset_size": size_call, "imported": 0}]

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
Doublet Call Script (Python)

Re-calls doublets from a pre-computed doublet score saved in a LOOM file.
Tool-agnostic: works with scores from DoubletFinder, scrublet, or any other tool.

Four strategies (--method):
  threshold   Score > --threshold → Doublet.    Requires --threshold.
  top_n       Top --n_doublets cells → Doublet. Requires --n_doublets.
  top_pct     Top --doublet_rate fraction.      Requires --doublet_rate.
  auto        Re-detect threshold automatically from the score distribution
              using bimodal valley detection (same logic as scrublet).

Options:
  -f                     Input LOOM file.                                   [required]
  --input_score_meta     Path to the doublet score in /col_attrs/.          [required]
                           Example: /col_attrs/doublet_score_scr
  --output_call_meta     Path in /col_attrs/ for the new binary call.       [required]
                           Example: /col_attrs/doublet_call_scr_t02
  --method               threshold | top_n | top_pct                        [required]
  -o                     Output folder for output.json. [optional: stdout]

  -- Parameters (method-specific) ─────────────────────────────────────────────
  --threshold            Fixed threshold (method=threshold).
  --n_doublets           Number of doublets to call (method=top_n).
  --doublet_rate         Fraction of cells to call as doublets (method=top_pct).

  --help                 Show this message and exit.

Output JSON doublet_call block:
  threshold_used   float  — actual score cutoff applied
                             threshold: the value passed
                             top_n / top_pct: min score among called doublets
                             auto: detected valley in score histogram
  doublet_rate     float  — actual fraction of cells called as doublets
"""

def main():
    if "--help" in sys.argv:
        print(HELP_TEXT)
        sys.exit(0)

    parser = argparse.ArgumentParser(description="Doublet Call Script", add_help=False)
    parser.add_argument("-f",                  required=True)
    parser.add_argument("-o",                  required=False, default=None)
    parser.add_argument("--input_score_meta",  required=True)
    parser.add_argument("--output_call_meta",  required=True)
    parser.add_argument("--method",            required=True, choices=["threshold", "top_n", "top_pct", "valley", "auto"])
    parser.add_argument("--threshold",         type=float, default=None)
    parser.add_argument("--n_doublets",        type=int,   default=None)
    parser.add_argument("--doublet_rate",      type=float, default=None)

    args = parser.parse_args()

    if args.method == "threshold" and args.threshold is None:
        ErrorJSON("--threshold is required when --method=threshold.")
    if args.method == "top_n" and args.n_doublets is None:
        ErrorJSON("--n_doublets is required when --method=top_n.")
    if args.method == "top_pct" and args.doublet_rate is None:
        ErrorJSON("--doublet_rate is required when --method=top_pct.")

    doublet_call(args)

if __name__ == "__main__":
    main()