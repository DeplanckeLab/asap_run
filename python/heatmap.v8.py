from __future__ import annotations
import argparse
import json
import sys
import struct
import h5py
import numpy as np
from pathlib import Path
from typing import Any, Optional

# ASAP v8 Heatmap compute step.
#
# Reads a genes x cells expression matrix from a LOOM file, subsets to a gene
# set (matched by symbol/accession) and a cell selection (categorical metadata),
# applies a value transform, computes seeded hierarchical clustering on both
# axes, and writes:
#   - heatmap_matrix.bin  : Float32 row-major (rows x cols), ordered by leaves
#   - heatmap_meta.json   : orders, dendrograms, labels, annotation tracks, range
#   - output.json         : run summary / warnings for the pipeline
#
# Everything needed is passed via a single --config JSON written by Rails before
# the container starts, which keeps the run fully specified and reproducible.

OUTPUT_JSON_NAME = "output.json"
META_JSON_NAME = "heatmap_meta.json"
MATRIX_BIN_NAME = "heatmap_matrix.bin"

_CELL_ID_PATH = "/col_attrs/CellID"
_GENE_ID_PATH = "/row_attrs/_StableID"
_GENE_NAME_PATH = "/row_attrs/Gene"
_GENE_ACCESSION_PATH = "/row_attrs/Accession"

sys.setrecursionlimit(1000000)


class ErrorJSON(Exception):
    def __init__(self, message: str, output_path: Optional[str] = None):
        super().__init__(message)
        payload = {"displayed_error": message}
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        else:
            print(json.dumps(payload, ensure_ascii=False), file=sys.stdout)
        sys.exit(1)


def _json_default(obj: Any):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _decode(v) -> str:
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode("utf-8", errors="replace")
    return str(v)


def _read_string_dataset(f: h5py.File, path: str) -> list[str]:
    raw = f[path][()]
    if isinstance(raw, (bytes, np.bytes_)):
        return [_decode(raw)]
    return [_decode(v) for v in np.asarray(raw).ravel()]


def _read_vector(f: h5py.File, path: str) -> Optional[list]:
    if path not in f:
        return None
    raw = f[path][()]
    arr = np.asarray(raw).ravel()
    return [_decode(v) if isinstance(v, (bytes, np.bytes_)) else v for v in arr]


def _open_loom_with_retry(path: str, mode: str, max_wait: int = 600):
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


def _matched_gene_rows(gene_identifiers, symbols, accessions):
    by_symbol = {}
    by_accession = {}
    for i, s in enumerate(symbols):
        key = str(s).strip().lower()
        if key and key not in by_symbol:
            by_symbol[key] = i
    for i, a in enumerate(accessions):
        key = str(a).strip().lower()
        if key and key not in by_accession:
            by_accession[key] = i

    rows = []
    seen = set()
    missing = 0
    for g in gene_identifiers:
        ens = str(g.get("ensembl_id", "")).strip().lower()
        sym = str(g.get("symbol", "")).strip().lower()
        idx = None
        if ens and ens in by_accession:
            idx = by_accession[ens]
        elif sym and sym in by_symbol:
            idx = by_symbol[sym]
        if idx is None:
            missing += 1
            continue
        if idx not in seen:
            seen.add(idx)
            rows.append(idx)
    return rows, missing


def _infer_track_type(values):
    non_empty = [v for v in values if v is not None and str(v).strip() != "" and str(v).lower() != "nan"]
    if not non_empty:
        return "categorical"
    floats = 0
    uniques = set()
    for v in non_empty:
        uniques.add(str(v))
        try:
            float(v)
            floats += 1
        except (ValueError, TypeError):
            pass
    if floats == len(non_empty) and len(uniques) > 12:
        return "numerical"
    return "categorical"


def _aggregate_categorical(values):
    counts = {}
    best, best_c = None, -1
    for v in values:
        k = str(v)
        counts[k] = counts.get(k, 0) + 1
        if counts[k] > best_c:
            best_c = counts[k]
            best = k
    return best


def _to_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return np.nan


def _linkage_and_tree(data, do_cluster, method, metric, seed):
    """Return (leaf_order, tree_json) for observations (rows of `data`).

    The tree is serialized as a FLAT linkage to keep the JSON shallow (deeply
    nested trees break JSON parsers' nesting limits and bloat the payload):
        {"n_leaves": N, "merges": [[left, right, height], ...]}
    Node ids: 0..N-1 are leaves (in display/matrix order); N..2N-2 are the
    internal merges, in order. The viewer rebuilds the tree iteratively.
    """
    n = data.shape[0]
    if not do_cluster or n < 3:
        return list(range(n)), None

    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist

    X = np.nan_to_num(np.asarray(data, dtype=np.float64), nan=0.0)
    np.random.seed(seed)

    if method == "ward":
        Z = linkage(X, method="ward", metric="euclidean")
    else:
        m = metric if metric in ("euclidean", "correlation", "cosine") else "euclidean"
        d = pdist(X, metric=m)
        d = np.nan_to_num(d, nan=0.0)
        Z = linkage(d, method=method)

    order = list(leaves_list(Z))
    pos_of = {int(obs_id): pos for pos, obs_id in enumerate(order)}

    merges = []
    for i in range(Z.shape[0]):
        a = int(Z[i, 0])
        b = int(Z[i, 1])
        h = float(Z[i, 2])
        # remap leaf cluster ids (< n) to display positions; internal ids (>= n) stay
        a = pos_of[a] if a < n else a
        b = pos_of[b] if b < n else b
        merges.append([a, b, h])

    return order, {"n_leaves": int(n), "merges": merges}


def run(args):
    out_dir = Path(args.o).resolve() if args.o else Path(".").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_json_path = str(out_dir / OUTPUT_JSON_NAME)

    input_path = Path(args.f).resolve()
    if not input_path.is_file():
        ErrorJSON(f"Input LOOM file not found: {args.f}", output_json_path)

    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        ErrorJSON(f"Heatmap config not found: {args.config}", output_json_path)
    with open(config_path, "r", encoding="utf-8") as fc:
        cfg = json.load(fc)

    warnings = list(cfg.get("warnings") or [])

    gene_identifiers = cfg.get("gene_identifiers") or []
    if not gene_identifiers:
        ErrorJSON("No genes were resolved from the selected gene set.", output_json_path)

    value_transform = cfg.get("value_transform", "zscore")
    column_mode = cfg.get("column_mode", "cells")
    cells_metadata = cfg.get("cells_metadata")
    cells_categories = [str(c) for c in (cfg.get("cells_categories") or [])]
    group_metadata = cfg.get("group_metadata") or cells_metadata
    column_tracks_cfg = cfg.get("column_tracks") or []
    row_tracks_cfg = cfg.get("row_tracks") or []
    max_cells = int(cfg.get("max_cells", 5000))
    seed = int(cfg.get("seed", 42))
    cluster_rows = bool(cfg.get("cluster_rows", True))
    cluster_cols = bool(cfg.get("cluster_cols", True))
    linkage_method = cfg.get("linkage_method", "ward")
    distance_metric = cfg.get("distance_metric", "euclidean")

    f_in, wt = _open_loom_with_retry(str(input_path), "r")
    with f_in:
        src = args.input_meta
        if src not in f_in:
            ErrorJSON(f"Input matrix path {src!r} not found in LOOM file.", output_json_path)
        node = f_in[src]
        if not isinstance(node, h5py.Dataset) or len(node.shape) != 2:
            ErrorJSON(f"Expected a 2-D matrix at {src}.", output_json_path)

        gene_ids = _read_string_dataset(f_in, _GENE_ID_PATH)
        cell_ids = _read_string_dataset(f_in, _CELL_ID_PATH)
        symbols = _read_vector(f_in, _GENE_NAME_PATH) or gene_ids
        accessions = _read_vector(f_in, _GENE_ACCESSION_PATH) or [""] * len(gene_ids)

        n_genes = len(gene_ids)
        n_cells = len(cell_ids)
        genes_first = int(node.shape[0]) == n_genes

        gene_rows, missing = _matched_gene_rows(gene_identifiers, symbols, accessions)
        if missing:
            warnings.append(f"{missing} of {len(gene_identifiers)} genes were not found in the dataset.")
        if not gene_rows:
            ErrorJSON("None of the selected genes are present in the dataset.", output_json_path)
        gene_rows = sorted(gene_rows)

        # Determine the cells to keep (columns for 'cells' mode / members for 'group' mode).
        cells_meta_values = _read_vector(f_in, cells_metadata) if cells_metadata else None
        if cells_meta_values is not None and cells_categories:
            base_cells = [i for i, v in enumerate(cells_meta_values) if str(v) in cells_categories]
        else:
            base_cells = list(range(n_cells))
        if not base_cells:
            ErrorJSON("The category filter selected no cells.", output_json_path)

        group_values = _read_vector(f_in, group_metadata) if (column_mode == "group" and group_metadata) else None
        if column_mode == "group" and group_values is None:
            ErrorJSON("Group-means mode requires a grouping metadata.", output_json_path)

        # Read the selected gene rows across all cells (subset columns later).
        if genes_first:
            sub = node[gene_rows, :].astype(np.float32)
        else:
            sub = node[:, gene_rows].astype(np.float32).T
        # sub: (n_selected_genes x n_cells)

        if column_mode == "group":
            groups = sorted({str(group_values[i]) for i in base_cells})
            col_labels = groups
            group_members = {g: [i for i in base_cells if str(group_values[i]) == g] for g in groups}
            cols_matrix = np.empty((sub.shape[0], len(groups)), dtype=np.float32)
            for gi, g in enumerate(groups):
                members = group_members[g]
                block = sub[:, members]
                cols_matrix[:, gi] = np.nanmean(block, axis=1) if block.shape[1] > 0 else np.nan
            column_units = [group_members[g] for g in groups]  # cell indices per column
        else:
            sel = base_cells
            if len(sel) > max_cells:
                rng = np.random.default_rng(seed)
                chosen = rng.choice(len(sel), size=max_cells, replace=False)
                chosen.sort()
                sel = [base_cells[i] for i in chosen]
                warnings.append(f"Subsampled {len(base_cells)} cells to {max_cells} (seed {seed}).")
            cols_matrix = sub[:, sel]
            col_labels = [cell_ids[i] for i in sel]
            column_units = [[i] for i in sel]

        row_labels = [symbols[i] if str(symbols[i]).strip() else gene_ids[i] for i in gene_rows]

    # Value transform (genes x cols).
    M = cols_matrix.astype(np.float64)
    diverging = False
    if value_transform == "zscore":
        mean = np.nanmean(M, axis=1, keepdims=True)
        std = np.nanstd(M, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            Z = (M - mean) / std
        Z[~np.isfinite(Z)] = 0.0
        M = Z
        diverging = True
    elif value_transform == "log1p":
        M = np.log1p(np.clip(M, 0, None))

    # Clustering / ordering.
    row_order, row_tree = _linkage_and_tree(M, cluster_rows, linkage_method, distance_metric, seed)
    col_order, col_tree = _linkage_and_tree(M.T, cluster_cols, linkage_method, distance_metric, seed)

    M = M[np.ix_(row_order, col_order)]
    row_labels = [row_labels[i] for i in row_order]
    col_labels = [col_labels[i] for i in col_order]
    col_cell_indices = None
    if column_mode == "group":
        col_cell_indices = [column_units[i] for i in col_order]

    # Value range for the colormap.
    if diverging:
        absvals = np.abs(M[np.isfinite(M)])
        vmax = float(np.percentile(absvals, 99)) if absvals.size else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = float(np.nanmax(np.abs(M))) if np.isfinite(M).any() else 1.0
        if vmax <= 0:
            vmax = 1.0
        vmin = -vmax
    else:
        finite = M[np.isfinite(M)]
        vmin = float(np.min(finite)) if finite.size else 0.0
        vmax = float(np.max(finite)) if finite.size else 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0

    n_rows, n_cols = M.shape

    # Write binary matrix (Float32, row-major). NaN preserved.
    matrix_path = out_dir / MATRIX_BIN_NAME
    M.astype("<f4").tofile(str(matrix_path))

    meta = {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "value_min": vmin,
        "value_max": vmax,
        "diverging": diverging,
        "value_transform": value_transform,
        "column_mode": column_mode,
        "row_labels": row_labels,
        "col_labels": col_labels,
        "row_tree": row_tree,
        "col_tree": col_tree,
        "warnings": warnings,
    }
    if col_cell_indices is not None:
        meta["col_cell_indices"] = col_cell_indices
    with open(out_dir / META_JSON_NAME, "w", encoding="utf-8") as fm:
        json.dump(meta, fm, default=_json_default, ensure_ascii=False)

    result = {
        "parameters": {
            "loom_path": str(input_path),
            "input_matrix": args.input_meta,
            "value_transform": value_transform,
            "column_mode": column_mode,
            "cluster_rows": cluster_rows,
            "cluster_cols": cluster_cols,
            "linkage_method": linkage_method,
            "distance_metric": distance_metric,
            "seed": seed,
        },
        "heatmap": {
            "n_genes": int(n_rows),
            "n_columns": int(n_cols),
            "matrix_file": MATRIX_BIN_NAME,
            "meta_file": META_JSON_NAME,
        },
        "metadata": [],
        "wait_time": wt,
    }
    if warnings:
        result["warnings"] = warnings

    with open(output_json_path, "w", encoding="utf-8") as fj:
        json.dump(result, fj, default=_json_default, ensure_ascii=False, separators=(",", ":"))


HELP_TEXT = """
Heatmap Script (ASAP v8)

Options:
  -f             Input LOOM file.                                     [required]
  --input_meta   Path to the expression matrix inside the LOOM.       [required]
  -o             Output folder (output.json, heatmap_matrix.bin, ...). [required]
  --config       Path to heatmap_config.json (written by Rails).      [required]
  --help         Show this message and exit.
"""


def main():
    if "--help" in sys.argv:
        print(HELP_TEXT)
        sys.exit(0)
    parser = argparse.ArgumentParser(description="Heatmap Script", add_help=False)
    parser.add_argument("-f", required=True)
    parser.add_argument("-o", required=True)
    parser.add_argument("--input_meta", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
