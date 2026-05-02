from __future__ import annotations

## Standard library
import argparse        # Argument parsing
import json            # For writing output files
import os              # For path/directory operations
import sys             # For ErrorJSON exit
import time            # For retry sleep
import warnings        # For non-fatal warnings
from contextlib import contextmanager, nullcontext
from pathlib import Path  # For path manipulation
from typing import Final

## Scientific stack
import h5py            # For reading/writing loom (HDF5) files
import numpy as np     # For array calculations
import scipy.sparse as sp  # For sparse matrix handling
from scipy import stats as scipy_stats
import pandas as pd # For matrix computation

## DE / stats (scanpy, anndata, pydeseq2 imported lazily where needed).
## t_test_approx: vectorized Welch t-test per gene (see _expression_matrix_for_test through run_t_test_approx_all_markers).
# scanpy, anndata  → run_scanpy_method() / run_scanpy_all_markers()
# pydeseq2         → run_deseq2() / run_deseq2_all_markers()

## Constants
AVAILABLE_METHODS: Final[list] = [
    "wilcoxon",             # Wilcoxon rank-sum test           (scanpy)
    "t_test",               # Student's t-test, pooled var     (scanpy)
    "t_test_overestim_var", # Student's t-test, per-gene var   (scanpy)
    # Welch unequal-variance t per gene; numpy BH-FDR; counts: norm 10k + log1p like scanpy t-test path.
    "t_test_approx",
    "deseq2",               # Negative-binomial GLM            (pydeseq2, counts only)
]
METHOD_NAME_MAP: Final[dict[str, str]] = {
    "wilcoxon": "wilcoxon",
    "t_test": "t-test",
    "t_test_overestim_var": "t-test_overestim_var",
    "t_test_approx": "t_test_approx",
    "deseq2": "deseq2",
}
COUNT_ONLY_METHODS: Final[set] = {"deseq2"}
BATCH_SUPPORT:      Final[set] = {"deseq2"}
DE_HEADERS: Final[list] = ["log Fold-Change", "p-value", "FDR", "Avg. Exp. Group 1", "Avg. Exp. Group 2"]
DE_NB_COLS: Final[int] = len(DE_HEADERS)  # Number of DE statistic columns per comparison

# FindAllMarkers (each group vs. rest): 8 columns — tested group label, ids, gene name, then five stats
FIND_ALL_MARKERS_GROUP_HEADER: Final[str] = "Compared group"
DE_HEADERS_FIND_ALL: Final[list[str]] = [
    "log Fold-Change",
    "p-value",
    "FDR",
    "Avg. exp. (tested group)",
    "Avg. exp. (other cells)",
]

# Column layout written to /attrs (must match nber_cols / headers in output.json)
DE_METADATA_HEADERS_PAIRWISE: Final[list[str]] = ["ensembl_id", "gene_name"] + DE_HEADERS
DE_METADATA_HEADERS_ALL_MARKERS: Final[list[str]] = [
    FIND_ALL_MARKERS_GROUP_HEADER,
    "ensembl_id",
    "gene_name",
] + DE_HEADERS_FIND_ALL

## Error handling
class ErrorJSON(Exception):
    """
    Prints a JSON error payload to stdout and exits immediately.
    """
    def __init__(self, message: str):
        super().__init__(message)
        print(json.dumps({"displayed_error": message}, ensure_ascii=False))
        os._exit(1)

## Loom File I/O class
class LoomHandler:
    """
    Wraps h5py access to a loom (HDF5) file with:
      - persistent open/close lifecycle (avoids reopening for every operation)
      - automatic retry on locked file (up to MAX_RETRIES attempts, RETRY_DELAY s apart)
      - clear ErrorJSON on file-not-found or unrecoverable I/O errors
      - context-manager support (with LoomHandler(...) as loom:)
    """

    MAX_RETRIES: int   = 50  # Maximum number of open attempts when file is locked
    RETRY_DELAY: float = 2.0  # Seconds to wait between retry attempts

    def __init__(self, loom_path: str, mode: str = "r"):
        """
        Parameters
        ----------
        loom_path : str
            Path to the .loom (HDF5) file.
        mode : str
            h5py open mode. Use "r" for read-only (default), "r+" for read-write.
        """
        self._path  = loom_path
        self._mode  = mode
        self._file: h5py.File | None = None

    def open(self) -> None:
        """
        Open the loom file, retrying on BlockingIOError (file locked by another process).
        Raises ErrorJSON immediately on file-not-found or any other OSError.
        """
        if not Path(self._path).is_file():
            ErrorJSON(f"Loom file not found: {self._path}")

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                self._file = h5py.File(self._path, self._mode)
                return
            except BlockingIOError:
                # File is locked — wait and retry
                if attempt == self.MAX_RETRIES:
                    ErrorJSON(f"Loom file is still locked after {self.MAX_RETRIES} attempts: {self._path}")
                warnings.warn(f"Loom file locked (attempt {attempt}/{self.MAX_RETRIES}), retrying in {self.RETRY_DELAY:.0f}s ...")
                time.sleep(self.RETRY_DELAY)
            except OSError as exc:
                ErrorJSON(f"Cannot open loom file {self._path!r}: {exc}")

    def close(self) -> None:
        """Close the loom file if it is currently open."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "LoomHandler":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _require_open(self) -> h5py.File:
        """Return the open file handle, or raise if the file has not been opened."""
        if self._file is None:
            raise RuntimeError("LoomHandler: file is not open — call open() first.")
        return self._file

    @staticmethod
    def _decode_str_array(arr: np.ndarray) -> np.ndarray:
        """Decode a bytes/object array to a plain Python-str array."""
        if arr.dtype.kind in ("S", "O"):
            return np.array([v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v) for v in arr])
        return arr.astype(str)

    def read_vector(self, dataset_path: str) -> np.ndarray | None:
        """
        Read a 1-D dataset from the open loom file.
        Returns None if the path is absent or does not point to a Dataset.
        """
        f = self._require_open()
        if dataset_path not in f:
            return None
        node = f[dataset_path]
        if not isinstance(node, h5py.Dataset):
            return None
        return node[:]

    def read_matrix_shape(self, dataset_path: str) -> tuple[int, int] | None:
        """
        Shape of the matrix at dataset_path without loading values (for npy cache checks).
        None if missing, unsupported node, or sparse matrix without a stored shape attribute.
        """
        f = self._require_open()
        if dataset_path not in f:
            return None
        node = f[dataset_path]
        if isinstance(node, h5py.Dataset):
            return tuple(int(x) for x in node.shape)
        if isinstance(node, h5py.Group) and all(k in node for k in ("data", "indices", "indptr")):
            shape_attr = node.attrs.get("shape", None)
            if shape_attr is not None:
                return tuple(int(x) for x in shape_attr)
            return None
        return None

    def read_matrix(self, dataset_path: str) -> np.ndarray | None:
        """
        Read an expression matrix from the open loom file.
        Returns a dense float64 array of shape (n_genes, n_cells), or None if absent.
        Handles dense datasets and CSR/CSC sparse groups.
        """
        f = self._require_open()
        if dataset_path not in f:
            return None
        node = f[dataset_path]

        # Sparse group (CSR / CSC)
        if isinstance(node, h5py.Group) and all(k in node for k in ("data", "indices", "indptr")):
            enc = node.attrs.get("encoding-type", b"csr_matrix")
            if isinstance(enc, bytes):
                enc = enc.decode()

            shape_attr = node.attrs.get("shape", None)
            if shape_attr is not None:
                shape = tuple(int(x) for x in shape_attr)
            else:
                indptr_len = int(node["indptr"].shape[0]) - 1
                max_idx    = int(np.max(node["indices"][:])) + 1
                shape = (indptr_len, max_idx) if enc == "csr_matrix" else (max_idx, indptr_len)

            mat_data = node["data"][:]
            indices  = node["indices"][:]
            indptr   = node["indptr"][:]
            if enc == "csr_matrix":
                return sp.csr_matrix((mat_data, indices, indptr), shape=shape).toarray().astype(np.float64)
            else:
                return sp.csc_matrix((mat_data, indices, indptr), shape=shape).toarray().astype(np.float64)

        # Dense dataset
        if isinstance(node, h5py.Dataset):
            return node[:].astype(np.float64)

        return None

    def write_dataset(self, dataset_path: str, data: np.ndarray) -> None:
        """Write (or overwrite) a dataset inside the open loom file."""
        f = self._require_open()
        if dataset_path in f:
            del f[dataset_path]
        parent = dataset_path.rsplit("/", 1)[0]
        if parent and parent not in f:
            f.require_group(parent)
        dtype = None
        if isinstance(data, np.ndarray) and data.dtype.kind in ("U", "S", "O"):
            dtype = h5py.string_dtype(encoding="utf-8")
            data = data.astype(object)
        f.create_dataset(dataset_path, data=data, dtype=dtype, compression="gzip", compression_opts=4)


def _expression_npy_cache_path(loom_path: str, input_dataset: str) -> Path:
    """Sidecar cache next to the loom: loom path segments with '/' replaced by '_', plus '.npy'."""
    name = input_dataset.replace("/", "_") + ".npy"
    return Path(loom_path).resolve().parent / name


def _write_expression_npy_cache(cache_path: Path, matrix: np.ndarray) -> None:
    """Write float32 matrix atomically for smaller disk use; readers promote to float64."""
    tmp_path = cache_path.parent / f"{cache_path.stem}._tmp_.npy"
    np.save(tmp_path, np.asarray(matrix, dtype=np.float32, order="C"))
    os.replace(str(tmp_path), str(cache_path))


def _load_expression_matrix_with_optional_npy_cache(
    loom: LoomHandler,
    loom_path: str,
    input_dataset: str,
) -> np.ndarray | None:
    """
    Load dense expression matrix from a .npy sidecar next to the loom when that file exists.
    The loom file mtime is not used (the loom is often updated elsewhere, e.g. attrs). When the
    loom exposes a matrix shape without loading data, the cache must match that shape; otherwise
    the cache is trusted as-is. If there is no sidecar, read from the loom and write the cache.
    """
    cache_path = _expression_npy_cache_path(loom_path, input_dataset)
    expected_shape = loom.read_matrix_shape(input_dataset)

    if cache_path.is_file():
        cached = np.load(cache_path, allow_pickle=False)
        if expected_shape is not None:
            if tuple(cached.shape) == expected_shape:
                return np.asarray(cached, dtype=np.float32, order="C")
            warnings.warn(
                f"Expression npy cache {str(cache_path)!r} shape {cached.shape} does not match "
                f"loom {expected_shape}; loading from loom."
            )
        else:
            return np.asarray(cached, dtype=np.float32, order="C")

    data_matrix = loom.read_matrix(input_dataset)
    if data_matrix is None:
        return None
    try:
        _write_expression_npy_cache(cache_path, data_matrix)
    except OSError as exc:
        warnings.warn(f"Could not write expression npy cache {str(cache_path)!r}: {exc}")
    return data_matrix


def _preview_parse_args(args: argparse.Namespace) -> tuple[bool, float | None, int, int, int]:
    """Returns (active, cell_fraction_or_none, seed, min_cells, max_cells). Preview is on only when --preview-cell-fraction is set."""
    fr = getattr(args, "preview_cell_fraction", None)
    seed = int(getattr(args, "preview_seed", 42))
    min_cells = int(getattr(args, "preview_min_cells", 1000))
    max_cells = int(getattr(args, "preview_max_cells", 10000))
    if min_cells < 1:
        ErrorJSON("--preview-min-cells must be >= 1")
    if max_cells < 1:
        ErrorJSON("--preview-max-cells must be >= 1")
    if max_cells < min_cells:
        ErrorJSON("--preview-max-cells must be >= --preview-min-cells")
    if fr is None:
        return False, None, seed, min_cells, max_cells
    if fr <= 0.0 or fr > 1.0:
        ErrorJSON("--preview-cell-fraction must be in (0, 1] when set")
    return True, fr, seed, min_cells, max_cells


def _preview_cap_total_cells(cell_idx: np.ndarray, max_cells: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly subsample cell_idx without replacement when len(cell_idx) exceeds max_cells."""
    cell_idx = np.asarray(cell_idx, dtype=np.intp)
    n = int(cell_idx.size)
    if n <= max_cells:
        return cell_idx
    return np.sort(rng.choice(cell_idx, size=max_cells, replace=False))


def _preview_stratified_cell_indices(
    labels: np.ndarray,
    fraction: float | None,
    rng: np.random.Generator,
    min_cells: int,
    max_cells: int,
) -> np.ndarray:
    """
    Per-label random subsample when fraction is set and in (0, 1).

    For each stratum with ng cells:
      - If ng < min_cells: keep all ng (cannot reach min_cells).
      - Else let k_frac = min(ng, round(fraction * ng)). Target count is k_frac clamped to
        [min_cells, max_cells], then capped by ng:
        k = min(ng, max(min_cells, min(max_cells, k_frac))).
      Random sample without replacement when k < ng.

    If fraction is None or >= 1, returns arange(n) (no per-stratum subsampling); caller may apply
    a global cap with _preview_cap_total_cells (e.g. when fraction is 1.0).
    """
    labels = np.asarray(labels)
    n = int(labels.shape[0])
    if fraction is None or fraction >= 1.0:
        return np.arange(n, dtype=np.intp)
    picked: list[np.ndarray] = []
    for g in np.unique(labels):
        ix = np.flatnonzero(labels == g)
        ng = int(ix.size)
        if ng < min_cells:
            picked.append(ix)
            continue
        k_frac = int(min(ng, round(fraction * float(ng))))
        k = int(min(ng, max(min_cells, min(max_cells, k_frac))))
        if k >= ng:
            picked.append(ix)
        else:
            picked.append(rng.choice(ix, size=k, replace=False))
    return np.sort(np.concatenate(picked))


def _de_table_gene_order_pairwise(
    lfc: np.ndarray,
    pvals: np.ndarray,
    pool: np.ndarray,
    n_top_per_direction: int | None,
) -> np.ndarray:
    """
    Row order for pairwise DE tables: gene indices into full vectors.

    If n_top_per_direction is None: all genes in pool, sorted by ascending p-value then descending LFC.

    If set: up to n_top genes with LFC > 0 ranked by ascending p then descending LFC, then up to n_top genes
    with LFC < 0 ranked by ascending p then ascending LFC (strongest down first). Genes need finite p-values.
    """
    pool = np.asarray(pool, dtype=np.intp).ravel()
    pool = pool[(pool >= 0) & (pool < int(lfc.shape[0]))]
    if pool.size == 0:
        return np.empty(0, dtype=np.intp)
    if n_top_per_direction is None:
        pv = pvals[pool]
        lf = lfc[pool]
        return pool[np.lexsort((-lf, np.where(np.isnan(pv), np.inf, pv)))]
    n_top = int(n_top_per_direction)
    up = pool[(lfc[pool] > 0.0) & np.isfinite(pvals[pool])]
    down = pool[(lfc[pool] < 0.0) & np.isfinite(pvals[pool])]
    parts: list[np.ndarray] = []
    if int(up.size) > 0:
        pv = pvals[up]
        lf = lfc[up]
        ord_up = up[np.lexsort((-lf, np.where(np.isnan(pv), np.inf, pv)))]
        parts.append(ord_up[:n_top])
    if int(down.size) > 0:
        pv = pvals[down]
        lf = lfc[down]
        ord_dn = down[np.lexsort((lf, np.where(np.isnan(pv), np.inf, pv)))]
        parts.append(ord_dn[:n_top])
    if not parts:
        return np.empty(0, dtype=np.intp)
    return np.concatenate(parts)


def _fam_table_row_count(
    unique_groups: list[str],
    all_results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    pool: np.ndarray,
    n_top_per_direction: int | None,
) -> int:
    n = 0
    for grp in unique_groups:
        pvals, _padj, lfc, _a1, _a2 = all_results[grp]
        n += int(_de_table_gene_order_pairwise(lfc, pvals, pool, n_top_per_direction).size)
    return n


## Output helpers
def write_tsv(
    out_path: str,
    ens_ids: list[str],
    gene_names: list[str],
    rows: list[tuple],
    table_top_de_per_direction: int | None = None,
) -> str:
    """
    Write DE results to a TSV file.
    rows is a list of (group_label | None, lfc, pvals, fdr, ave_g1, ave_g2).
    If group_label is not None for the first entry, a leading 'group' column is included.
    table_top_de_per_direction: if set, only top N up- and top N down-regulated genes by p-value (see _de_table_gene_order_pairwise).
    """
    has_group = rows[0][0] is not None
    headers   = list(DE_METADATA_HEADERS_ALL_MARKERS) if has_group else list(DE_METADATA_HEADERS_PAIRWISE)
    n_genes   = len(gene_names)
    sub = np.arange(n_genes, dtype=np.intp)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(headers) + "\n")
        for grp, lfc, pvals, fdr, ave_g1, ave_g2 in rows:
            sort_idx = _de_table_gene_order_pairwise(lfc, pvals, sub, table_top_de_per_direction)
            for i in sort_idx:
                row = []
                if has_group:
                    row.append(grp)
                row += [
                    ens_ids[i],
                    gene_names[i],
                    f"{lfc[i]}"   if not np.isnan(lfc[i])   else "NA",
                    f"{pvals[i]}" if not np.isnan(pvals[i]) else "NA",
                    f"{fdr[i]}"   if not np.isnan(fdr[i])   else "NA",
                    f"{ave_g1[i]}",
                    f"{ave_g2[i]}",
                ]
                fh.write("\t".join(row) + "\n")
    return out_path

def write_volcano_json(logfc: np.ndarray, pvals: np.ndarray, gene_names: list[str], ens_ids: list[str], output_dir: str) -> str | None:
    """Writes a plotly volcano plot to output.plot.json."""
    try:
        import plotly.graph_objects as go

        not_na = ~(np.isnan(logfc) | np.isnan(pvals) | (pvals <= 0))
        x      = logfc[not_na]
        y      = -np.log10(pvals[not_na])
        na_idx = np.where(not_na)[0]
        text   = [
            f"log FC: {lfc:.3f}<br>p-value: {pv:.3e}"
            f"<br>Ensembl: {ens_ids[i]}<br>Gene: {gene_names[i]}"
            for lfc, pv, i in zip(x, pvals[not_na], na_idx)
        ]
        fig = go.Figure(go.Scatter(x=x, y=y, mode="markers", text=text, hoverinfo="text", marker=dict(size=4, opacity=0.7)))
        fig.update_layout(title="Volcano plot", xaxis_title="log Fold-Change", yaxis_title="-log10(p-value)")
        out_path = os.path.join(output_dir, "output.plot.json")
        fig.write_json(out_path)
        return out_path
    except Exception as exc:
        warnings.warn(f"Could not generate volcano plot: {exc}")
        return None

def _unique_categories_sorted(groups: np.ndarray) -> list[str]:
    """
    Unique category labels from a 1-D loom metadata vector, sorted alphanumerically
    (same ordering as Ruby Array#sort on strings).
    """
    labels = [str(x) for x in groups.tolist()]
    return sorted(set(labels))

def _de_table_for_json(metadata_data: np.ndarray, headers: list[str]) -> dict[str, object]:
    """Same row order and string formatting as TSV / metadata matrix; JSON-serializable (no numpy scalars)."""
    n_rows, n_cols = int(metadata_data.shape[0]), int(metadata_data.shape[1])
    hdr = list(headers)
    rows: list[list[str]] = []
    for i in range(n_rows):
        rows.append([str(metadata_data[i, j]) for j in range(n_cols)])
    return {"headers": hdr, "rows": rows}


def build_result_json(
    output_dataset: str | None,
    nber_rows: int,
    unique_groups: list[str] | None = None,
) -> dict:
    """
    Build the output.json metadata dict for the /attrs dataset.
    nber_cols and headers match the matrix from build_metadata_matrix_* (same order as
    column_names on the HDF5 node). Uses key "headers" (plural) for Rails finish_run.
    """
    if unique_groups is not None:
        # FindAllMarkers: 8 columns — compared group, ids, gene name, then five vs-rest statistics
        return {
            "metadata": [
                {
                    "name":      output_dataset,
                    "on":        "GLOBAL",
                    "type":      "NUMERIC",
                    "nber_cols": len(DE_METADATA_HEADERS_ALL_MARKERS),
                    "nber_rows": nber_rows,
                    "headers":  list(DE_METADATA_HEADERS_ALL_MARKERS),
                    "groups":   unique_groups,
                }
            ]
        }
    # Single-marker or two-group: one row per gene after sorting — ensembl_id, gene_name, then DE_HEADERS
    return {
        "metadata": [
            {
                "name":      output_dataset,
                "on":        "GLOBAL",
                "type":      "NUMERIC",
                "nber_cols": len(DE_METADATA_HEADERS_PAIRWISE),
                "nber_rows": nber_rows,
                "headers":  list(DE_METADATA_HEADERS_PAIRWISE),
            }
        ]
    }

def build_metadata_matrix_single(
    ens_ids: list[str],
    gene_names: list[str],
    lfc: np.ndarray,
    pvals: np.ndarray,
    fdr: np.ndarray,
    ave_g1: np.ndarray,
    ave_g2: np.ndarray,
    table_top_de_per_direction: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    headers = list(DE_METADATA_HEADERS_PAIRWISE)
    n_genes = len(ens_ids)
    sub = np.arange(n_genes, dtype=np.intp)
    sort_idx = _de_table_gene_order_pairwise(lfc, pvals, sub, table_top_de_per_direction)
    ens_ids_s   = [ens_ids[i]   for i in sort_idx]
    gene_names_s = [gene_names[i] for i in sort_idx]
    lfc_s   = lfc[sort_idx];   pvals_s = pvals[sort_idx]
    fdr_s   = fdr[sort_idx];   ave_g1_s = ave_g1[sort_idx];  ave_g2_s = ave_g2[sort_idx]
    data = np.column_stack([
        np.array(ens_ids_s, dtype=object),
        np.array(gene_names_s, dtype=object),
        np.array([f"{v}" if not np.isnan(v) else "NA" for v in lfc_s], dtype=object),
        np.array([f"{v}" if not np.isnan(v) else "NA" for v in pvals_s], dtype=object),
        np.array([f"{v}" if not np.isnan(v) else "NA" for v in fdr_s], dtype=object),
        np.array([f"{v}" for v in ave_g1_s], dtype=object),
        np.array([f"{v}" for v in ave_g2_s], dtype=object),
    ])
    return data, headers

def build_metadata_matrix_all_markers(
    ens_ids: list[str],
    gene_names: list[str],
    unique_groups: list[str],
    all_results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    table_top_de_per_direction: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Builds the metadata matrix for FindAllMarkers in the same tall format as the TSV:
    one row per gene per group, sorted by ascending p-value then descending lfc,
    with a leading 'group' column. Matches write_tsv() output exactly.
    """
    headers = list(DE_METADATA_HEADERS_ALL_MARKERS)
    n_genes = len(gene_names)
    sub = np.arange(n_genes, dtype=np.intp)
    rows = []
    for grp in unique_groups:
        pvals, padj, lfc, ave_g1, ave_g2 = all_results[grp]
        sort_idx = _de_table_gene_order_pairwise(lfc, pvals, sub, table_top_de_per_direction)
        for i in sort_idx:
            rows.append([
                str(grp),
                str(ens_ids[i]),
                str(gene_names[i]),
                f"{lfc[i]}"   if not np.isnan(lfc[i])   else "NA",
                f"{pvals[i]}" if not np.isnan(pvals[i]) else "NA",
                f"{padj[i]}"  if not np.isnan(padj[i])  else "NA",
                f"{ave_g1[i]}",
                f"{ave_g2[i]}",
            ])
    return np.array(rows, dtype=object), headers

def write_metadata_dataset(loom_path: str, dataset_path: str, data: np.ndarray, headers: list[str]) -> str:
    """Writes metadata dataset under /attrs and stores the column names as an attribute."""
    if not dataset_path.startswith("/attrs/"):
        ErrorJSON("--write-metadata path should refer to the /attrs/ path (e.g. /attrs/_de_2_wilcoxon)")
    if data.ndim != 2:
        ErrorJSON(f"Metadata matrix must be 2-D, got shape {data.shape!r}")
    if data.shape[1] != len(headers):
        ErrorJSON(
            f"Metadata column mismatch: dataset has {data.shape[1]} columns but {len(headers)} headers {headers!r}"
        )
    # Object tables (FindAllMarkers + pairwise): force plain str per cell so HDF5 vlen UTF-8 write
    # does not coerce or drop the leading category column.
    if data.dtype == object and data.ndim == 2:
        r, c = data.shape
        norm = np.empty((r, c), dtype=object)
        for i in range(r):
            for j in range(c):
                v = data[i, j]
                norm[i, j] = "" if v is None else str(v)
        data = norm

    with LoomHandler(loom_path, mode="r+") as loom:
        loom.write_dataset(dataset_path, data)
        node = loom._require_open()[dataset_path]
        if "column_names" in node.attrs:
            del node.attrs["column_names"]
        node.attrs["column_names"] = np.array(headers, dtype=h5py.string_dtype(encoding="utf-8"))
    return dataset_path

def _count_tested(pvals: np.ndarray) -> int:
    return int(np.sum(~np.isnan(pvals)))

def _count_significant(fdr: np.ndarray, cutoff: float = 0.05) -> int:
    return int(np.sum((~np.isnan(fdr)) & (fdr <= cutoff)))

## Performance logging (stderr). Set ASAP_DE_PERF_LOG=1 to enable timings for t_test_approx and related steps.
def _perf_enabled() -> bool:
    v = os.environ.get("ASAP_DE_PERF_LOG", "").strip().lower()
    return v in ("1", "true", "yes")

def _perf_log(msg: str) -> None:
    if _perf_enabled():
        print(msg, file=sys.stderr, flush=True)

@contextmanager
def _perf_span(label: str):
    if not _perf_enabled():
        yield
        return
    t0 = time.perf_counter()
    yield
    _perf_log(f"[de_perf] {label}: {time.perf_counter() - t0:.3f}s")

## DE method runners
def run_scanpy_method(
    data_matrix:  np.ndarray,   # (n_genes, n_cells) — already subset to G1 + G2 cells
    group_labels: np.ndarray,   # 1-D int array: 1 or 2 per cell
    gene_names:   list[str],
    method:       str,
    is_count:     bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Wraps scanpy.tl.rank_genes_groups for any native scanpy method.
    Returns (pvals, padj, lfc, ave_g1, ave_g2) all shape (n_genes,), NaN for genes not returned.
    """
    import anndata as ad # Lazy loading
    import scanpy as sc # Lazy loading

    n_genes, n_cells = data_matrix.shape

    adata = ad.AnnData(X=data_matrix.T.astype(np.float32))
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = gene_names
    adata.obs["group"] = pd.Categorical([str(g) for g in group_labels], categories=["1", "2"])

    sc.settings.verbosity = 0
    if is_count:
        # Normalize to 10 000 counts per cell, then log1p-transform. This ensures rank_genes_groups computes LFC on a comparable log scale.
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    # If data is already normalized+logged, scanpy's LFC = mean(expr_g1) − mean(expr_g2) in log space  =  valid log fold-change.
    with warnings.catch_warnings(): # PerformanceWarning: DataFrame is highly fragmented. [...]
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning, module="scanpy")
        sc.tl.rank_genes_groups(adata, groupby="group", groups=["1"], reference="2", method=method, use_raw=False, n_genes=n_genes)

    res          = adata.uns["rank_genes_groups"]
    result_genes = np.array(res["names"]["1"])
    result_pvals = np.array(res["pvals"]["1"])
    result_padj  = np.array(res["pvals_adj"]["1"])
    result_lfc   = np.array(res["logfoldchanges"]["1"])

    # Build an index mapping: gene name → original position
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # Map result_genes (scanpy order) → original indices in one shot
    original_idx = np.array([gene_to_idx[g] for g in result_genes])

    pvals = np.full(n_genes, np.nan)
    lfc   = np.full(n_genes, np.nan)
    padj  = np.full(n_genes, np.nan)
    pvals[original_idx] = result_pvals
    lfc[original_idx]   = result_lfc
    padj[original_idx]  = result_padj

    # Averages computed on the original input matrix (--input-dataset values, before any normalization/log-transform).
    # np.asarray + ravel guards against adata.X being sparse after preprocessing (sparse .mean() returns a matrix, not a 1-D array).
    mask_1 = group_labels == 1
    mask_2 = group_labels == 2
    ave_g1 = data_matrix[:, mask_1].mean(axis=1)  # shape (n_genes,)
    ave_g2 = data_matrix[:, mask_2].mean(axis=1)

    return pvals, padj, lfc, ave_g1, ave_g2

def run_scanpy_all_markers(
    data_matrix:  np.ndarray,   # (n_genes, n_cells) — all cells
    group_labels: np.ndarray,   # 1-D str array: group name per cell
    gene_names:   list[str],
    method:       str,
    is_count:     bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Runs scanpy.tl.rank_genes_groups for ALL groups simultaneously vs. rest
    (equivalent to Seurat's FindAllMarkers). This is a single-pass operation —
    much faster than looping over groups individually.

    Returns a dict keyed by group name:
        group_name → (pvals, padj, lfc, ave_g1, ave_g2)  each shape (n_genes,)
    where ave_g1 is the mean expression of that group and ave_g2 is the mean
    of all remaining cells.
    """
    import anndata as ad
    import scanpy as sc

    n_genes, n_cells = data_matrix.shape
    unique_groups = sorted(set(group_labels))

    adata = ad.AnnData(X=data_matrix.T.astype(np.float32))
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = gene_names
    adata.obs["group"] = pd.Categorical(group_labels.tolist(), categories=unique_groups)

    sc.settings.verbosity = 0
    if is_count:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # Single pass: all groups vs. rest simultaneously
    with warnings.catch_warnings(): # PerformanceWarning: DataFrame is highly fragmented. [...]
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning, module="scanpy")
        sc.tl.rank_genes_groups(adata, groupby="group", reference="rest", method=method, use_raw=False, n_genes=n_genes)

    res = adata.uns["rank_genes_groups"]
    # Build an index mapping: gene name → original position
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    results: dict = {}
    for grp in unique_groups:
        result_genes = np.array(res["names"][grp])
        result_pvals = np.array(res["pvals"][grp])
        result_padj  = np.array(res["pvals_adj"][grp])
        result_lfc   = np.array(res["logfoldchanges"][grp])

        # Map result_genes (scanpy order) → original indices in one shot
        original_idx = np.array([gene_to_idx[g] for g in result_genes])

        pvals = np.full(n_genes, np.nan)
        lfc   = np.full(n_genes, np.nan)
        padj  = np.full(n_genes, np.nan)
        pvals[original_idx] = result_pvals
        lfc[original_idx]   = result_lfc
        padj[original_idx]  = result_padj

        # np.asarray + ravel guards against adata.X being sparse after preprocessing (sparse .mean() returns a matrix, not a 1-D array).
        mask_grp  = np.array(group_labels) == grp
        mask_rest = ~mask_grp
        ave_g1 = data_matrix[:, mask_grp].mean(axis=1)    # shape (n_genes,), original input-dataset values
        ave_g2 = data_matrix[:, mask_rest].mean(axis=1)

        results[grp] = (pvals, padj, lfc, ave_g1, ave_g2)

    return results

# -----------------------------------------------------------------------------
# t_test_approx implementation
#
# Statistical model: for each gene independently, treat cell expression values
# (after optional normalize_total + log1p for counts) as two samples G1 vs G2
# and test H0: equal means using Welch's t-test (unequal variances). Multiple
# testing across genes uses Benjamini-Hochberg FDR within each contrast.
#
# Computation: sufficient statistics sum_j x_j and sum_j x_j^2 per group are
# obtained as dense matrix-vector products (GEMV) along cells, never looping
# genes in Python. FindAllMarkers reuses full-cohort row sums and subtracts the
# current group to get the "rest" sums without a second full GEMV on the rest.
#
# Entry points: run_t_test_approx_pairwise (two masks), run_t_test_approx_all_markers (labels).
# -----------------------------------------------------------------------------

def _normalize_total_log1p_inplace(X: np.ndarray) -> None:
    """
    Count preprocessing aligned with scanpy rank_genes_groups when is_count=True.

    X is (n_genes, n_cells), column j = one cell. Steps (in place, float64):
      1) Divide each column by its sum so each cell sums to 1 (library size normalization).
      2) Scale by 1e4 so each cell targets 10k total counts (scanpy normalize_total default).
      3) log1p so downstream means are on a log-like scale comparable to log-normalized scRNA.

    Welch t-tests and LFC are then computed on this transformed X; reported ave_g1/ave_g2 for
    counts use the raw matrix separately (see run_t_test_approx_* raw_means branches).
    """
    col_sums = X.sum(axis=0, dtype=np.float64)
    col_sums[col_sums <= 0] = 1.0
    X /= col_sums
    X *= 1e4
    np.log1p(X, out=X)

def _expression_matrix_for_test(data_matrix: np.ndarray, is_count: bool) -> np.ndarray:
    """
    Build the matrix on which t_test_approx computes sums, squared sums, means, variances, and t-stats.

    Count data (is_count=True):
      Copy to float64 C-contiguous, apply _normalize_total_log1p_inplace. All DE statistics
      (Welch, LFC as mean difference on this scale) use this matrix.

    Non-count (is_count=False):
      Use the input as the "expression" scale for the test (already normalized/logged in the pipeline).
      float32 input stays float32 to speed BLAS GEMV (X @ w) and X*X; mixing float32 X with float64
      weights can force slow promotion paths, so mask weights w match X dtype in the runners.

    Returns: (n_genes, n_cells) C-contiguous; not necessarily a copy when non-count float32 path.
    """
    if is_count:
        X = np.asarray(data_matrix, dtype=np.float64, order="C")
        X = X.copy()
        _normalize_total_log1p_inplace(X)
        return X
    if data_matrix.dtype == np.float32:
        return np.asarray(data_matrix, dtype=np.float32, order="C")
    return np.asarray(data_matrix, dtype=np.float64, order="C")


def _mean_var_from_sums(sum_x: np.ndarray, sumsq_x: np.ndarray, n: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-gene sample mean and unbiased sample variance from sufficient statistics over one cell subset.

    For each gene g, given sum_j x_gj and sum_j x_gj^2 over n cells in that subset (j indexes cells
    in the subset; sums come from GEMV: X @ w with w binary or 0/1 membership):

      mean_g = (1/n) * sum_x

      For sample variance we use the standard one-pass stable form with Bessel correction:
      var_g = ( sumsq_x - (sum_x^2)/n ) / (n - 1)
      which equals (1/(n-1)) * sum_j (x_j - mean)^2.

    If n <= 1, mean and variance are undefined for inference; return NaN vectors.

    Negative variance from floating point noise is clamped to 0 before Welch (Welch uses var/n terms).

    All algebra in float64 to match stable statistics after float32 GEMV accumulation.
    """
    sum_x = np.asarray(sum_x, dtype=np.float64)
    sumsq_x = np.asarray(sumsq_x, dtype=np.float64)
    if n <= 1.0:
        nanv = np.full(sum_x.shape[0], np.nan, dtype=np.float64)
        return nanv, nanv
    mean = sum_x / n
    with np.errstate(divide="ignore", invalid="ignore"):
        v = (sumsq_x - (sum_x * sum_x) / n) / (n - 1.0)
    v = np.where(np.isfinite(v), np.maximum(v, 0.0), np.nan)
    return mean, v

def _welch_ttest_from_mean_var(
    mean1: np.ndarray,
    var1: np.ndarray,
    n1: float,
    mean2: np.ndarray,
    var2: np.ndarray,
    n2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized Welch (unequal variance) two-sample t-test, two-sided p-value, one row per gene.

    Notation: group 1 has n1 cells, per-gene sample mean mean1 and unbiased variance var1; same for group 2.

    Welch t statistic (per gene):
      se^2 = var1/n1 + var2/n2
      t = (mean1 - mean2) / sqrt(se^2)

    Welch-Satterthwaite approximate degrees of freedom:
      dof = se^4 / ( (var1/n1)^2/(n1-1) + (var2/n2)^2/(n2-1) )

    Two-sided p-value: 2 * P(T_dof > |t|) = 2 * scipy.stats.t.sf(abs(t), dof).

    Log fold change on the working expression scale (same matrix as means):
      lfc = mean1 - mean2  (difference of means; for log1p counts this is not log2 ratio of raw counts).

    Undefined / NaN: if n1 or n2 <= 1, or se^2 is non-positive or non-finite after sanitizing t.

    CellxGene-style diffexp uses the same var/n structure; this matches the usual Welch formula.
    """
    mean1 = np.asarray(mean1, dtype=np.float64)
    var1 = np.asarray(var1, dtype=np.float64)
    mean2 = np.asarray(mean2, dtype=np.float64)
    var2 = np.asarray(var2, dtype=np.float64)
    if n1 <= 1.0 or n2 <= 1.0:
        n_genes = mean1.shape[0]
        nanv = np.full(n_genes, np.nan, dtype=np.float64)
        return nanv, nanv
    # Sampling variance of the mean: Var(mean_k) = var_k / n_k (per gene).
    vn1 = var1 / n1
    vn2 = var2 / n2
    sum_vn = vn1 + vn2
    with np.errstate(divide="ignore", invalid="ignore"):
        # Welch-Satterthwaite dof; degenerate dof falls back to 1.0 so t.sf remains defined.
        dof = sum_vn**2 / (vn1**2 / (n1 - 1.0) + vn2**2 / (n2 - 1.0))
        dof = np.where(np.isfinite(dof) & (dof > 0), dof, 1.0)
        t = (mean1 - mean2) / np.sqrt(sum_vn)
    # Non-finite t (e.g. 0/0) mapped to 0 before sf so we can mask pvals afterward.
    t = np.where(np.isfinite(t), t, 0.0)
    # Two-sided: upper tail of |t| doubled (symmetric t distribution).
    pvals = scipy_stats.t.sf(np.abs(t), dof) * 2.0
    pvals = np.where(np.isfinite(pvals) & (sum_vn > 0), pvals, np.nan)
    lfc = mean1 - mean2
    return pvals, lfc

def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR adjustment for independent (or positively dependent) tests; per gene output padj.

    Genes with NaN or out-of-range p-values remain NaN in padj (excluded from adjustment).

    Algorithm (m = number of valid p-values):
      1) Sort valid p-values ascending: p_(1) <= ... <= p_(m).
      2) Provisional q_(i) = min(1, p_(i) * m / i)  (BH step-up multiplier m/i by rank).
      3) Enforce monotonicity: from largest i downward, q_(i) := min(q_(i), q_(i+1)) so the adjusted
         sequence is non-decreasing when mapped back to sorted order.
      4) Scatter adjusted values back to original gene order.

    mergesort keeps ties stable. Pure numpy (no statsmodels) because this runs once per group in FindAllMarkers.

    padj[g] is the smallest FDR threshold at which gene g would be rejected in the BH procedure.
    """
    padj = np.full_like(pvals, np.nan, dtype=np.float64)
    valid = np.isfinite(pvals) & (pvals >= 0.0) & (pvals <= 1.0)
    if not np.any(valid):
        return padj
    idx = np.nonzero(valid)[0]
    p_sub = pvals[idx]
    m = int(p_sub.size)
    order = np.argsort(p_sub, kind="mergesort")
    p_sorted = p_sub[order]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    q_sorted = np.minimum(1.0, p_sorted * m / ranks)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    out_sub = np.empty(m, dtype=np.float64)
    out_sub[order] = q_sorted
    padj[idx] = out_sub
    return padj

def run_t_test_approx_pairwise(
    data_matrix: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    is_count: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Two-group differential expression: Welch t-test per gene without iterating genes in Python.

    Pipeline:
      1) Build working matrix X = _expression_matrix_for_test(...) — same scale as scanpy t-test on counts
         when is_count (normalize to 10k + log1p).
      2) Binary masks mask1, mask2 length n_cells (1 where cell belongs to group 1 or 2). They need not
         be disjoint in pathological inputs; n1,n2 are sums of w (here callers pass disjoint G1/G2).
      3) Encode membership as nonnegative weights w1,w2 (0/1). Dtype matches X (float32 or float64)
         so GEMV uses one BLAS dtype and stays fast on float32 X.
      4) Per gene, sum of expression in group 1 is (X @ w1)_g = sum_{cells in G1} X[g,cell]. Similarly
         sumsq uses elementwise X2 = X*X and X2 @ w1 gives sum of squares in G1.
      5) _mean_var_from_sums converts (sum, sumsq, n) to mean and unbiased variance per gene per group.
      6) _welch_ttest_from_mean_var gives p-values and mean1-mean2 (reported as lfc).
      7) _bh_fdr adjusts p-values across genes (FDR within this contrast only).
      8) ave_g1, ave_g2: for counts, means on raw integer/float counts (R @ w / n) for JSON parity with
         scanpy path; for non-count, means on the same X used for the test.

    Returns: (pvals, padj, lfc, ave_g1, ave_g2) each shape (n_genes,).
    """
    n_genes, n_cells = data_matrix.shape
    if mask1.shape[0] != n_cells or mask2.shape[0] != n_cells:
        ErrorJSON("Internal error: t_test_approx mask length does not match number of cells.")

    with _perf_span("t_test_approx_pairwise prep_X"):
        X = _expression_matrix_for_test(data_matrix, is_count)
    # GEMV (X @ w) dtype: match w to X so BLAS does not upcast the whole multiply to float64 on float32 X.
    w_dtype = np.float32 if X.dtype == np.float32 else np.float64
    w1 = np.asarray(mask1, dtype=w_dtype).ravel()
    w2 = np.asarray(mask2, dtype=w_dtype).ravel()
    n1 = float(np.sum(w1))
    n2 = float(np.sum(w2))

    with _perf_span("t_test_approx_pairwise X_squared + gemv"):
        # Row g of X @ w is dot(X[g,:], w) = weighted sum over cells; w 0/1 => sum over group.
        X2 = np.multiply(X, X)
        sum1 = X @ w1
        sumsq1 = X2 @ w1
        sum2 = X @ w2
        sumsq2 = X2 @ w2

    with _perf_span("t_test_approx_pairwise mean_var_welch_bh"):
        mean1, var1 = _mean_var_from_sums(sum1, sumsq1, n1)
        mean2, var2 = _mean_var_from_sums(sum2, sumsq2, n2)
        pvals, lfc = _welch_ttest_from_mean_var(mean1, var1, n1, mean2, var2, n2)
        padj = _bh_fdr(pvals)

    with _perf_span("t_test_approx_pairwise raw_means"):
        if is_count:
            # Report average raw counts per group; DE stats still used log-normalized X above.
            R = np.asarray(data_matrix, dtype=np.float64, order="C")
            w1r = w1.astype(np.float64, copy=False)
            w2r = w2.astype(np.float64, copy=False)
            ave_g1 = (R @ w1r) / n1
            ave_g2 = (R @ w2r) / n2
        else:
            ave_g1 = (X @ w1) / n1
            ave_g2 = (X @ w2) / n2
    return pvals, padj, lfc, ave_g1, ave_g2

def run_t_test_approx_all_markers(
    data_matrix: np.ndarray,
    group_labels: np.ndarray,
    is_count: bool,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    FindAllMarkers (each biological group vs all other cells), same Welch+BHFDR per gene as pairwise.

    Optimization vs naive repeated pairwise:
      - X and X2 = X*X are built once.
      - For each group g, membership vector w has 1 on cells in g, 0 elsewhere. Only two GEMVs per group:
        sum_g = X @ w, sumsq_g = X2 @ w (O(n_genes * n_cells) each, BLAS).
      - "Rest" is the complement of g. Instead of GEMV with (1-w), use global totals:
        sum_all[gene] = sum over all cells = X @ 1 = row sums; sum_r = sum_all - sum_g.
        Similarly sumsq_r = sumsq_all - sumsq_g. Cell counts: n_r = n_tot - n_g.
        This is exact because sums decompose over disjoint partitions (G vs rest).

    Per group: Welch(mean_g, var_g, n_g, mean_r, var_r, n_r), BH-FDR on that group's p-values,
    ave_g1 = mean raw (or X) in g, ave_g2 = mean in rest via (total - sum_g)/n_r for counts raw totals.

    Returns dict keyed by string group name -> (pvals, padj, lfc, ave_g1, ave_g2).
    """
    n_genes, n_cells = data_matrix.shape
    gl = np.asarray(group_labels)
    unique_groups, inv = np.unique(gl, return_inverse=True)
    n_groups = int(unique_groups.size)
    n_tot = float(n_cells)

    with _perf_span("t_test_approx_all_markers prep_X_X2"):
        X = _expression_matrix_for_test(data_matrix, is_count)
        X2 = np.multiply(X, X)
        # Full-cohort row sums = X @ 1; used with per-group GEMV to derive "rest" without second GEMV on X.
        sum_all = np.sum(X, axis=1, dtype=np.float64)
        sumsq_all = np.sum(X2, axis=1, dtype=np.float64)
        if is_count:
            R = np.asarray(data_matrix, dtype=np.float64, order="C")
            raw_sum_all = np.sum(R, axis=1)
        else:
            R = None
            raw_sum_all = sum_all

    w_dtype = np.float32 if X.dtype == np.float32 else np.float64
    results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    loop_gemv = 0.0
    loop_welch = 0.0
    loop_bh = 0.0
    loop_raw = 0.0
    t_loop0 = time.perf_counter() if _perf_enabled() else 0.0

    for gi in range(n_groups):
        grp = unique_groups[gi]
        w = (inv == gi).astype(w_dtype)
        n_g = float(np.sum(w))
        t_g0 = time.perf_counter() if _perf_enabled() else 0.0
        sum_g = X @ w
        sumsq_g = X2 @ w
        if _perf_enabled():
            loop_gemv += time.perf_counter() - t_g0
        # Rest = all cells minus group g: additive sums avoid X @ (1-w).
        sum_r = sum_all - sum_g
        sumsq_r = sumsq_all - sumsq_g
        n_r = n_tot - n_g
        t_w0 = time.perf_counter() if _perf_enabled() else 0.0
        # Group g vs its complement: same Welch formulas as pairwise with (mean_g, n_g) and (mean_r, n_r).
        mean_g, var_g = _mean_var_from_sums(sum_g, sumsq_g, n_g)
        mean_r, var_r = _mean_var_from_sums(sum_r, sumsq_r, n_r)
        pvals, lfc = _welch_ttest_from_mean_var(mean_g, var_g, n_g, mean_r, var_r, n_r)
        if _perf_enabled():
            loop_welch += time.perf_counter() - t_w0
        t_b0 = time.perf_counter() if _perf_enabled() else 0.0
        padj = _bh_fdr(pvals)
        if _perf_enabled():
            loop_bh += time.perf_counter() - t_b0
        t_r0 = time.perf_counter() if _perf_enabled() else 0.0
        if R is not None:
            # Raw count means: one GEMV for group, rest from precomputed row total minus group sum.
            wr = w.astype(np.float64, copy=False)
            sum_raw_g = R @ wr
            ave_g1 = sum_raw_g / n_g
            ave_g2 = (raw_sum_all - sum_raw_g) / n_r
        else:
            # Non-count: working X is the expression layer; means match sum_g/n_g and complement.
            ave_g1 = sum_g / n_g
            ave_g2 = sum_r / n_r
        if _perf_enabled():
            loop_raw += time.perf_counter() - t_r0
        results[str(grp)] = (pvals, padj, lfc, ave_g1, ave_g2)

    if _perf_enabled():
        _perf_log(
            f"[de_perf] t_test_approx_all_markers per_group_loop({n_groups}): "
            f"gemv {loop_gemv:.3f}s, welch {loop_welch:.3f}s, bh_fdr {loop_bh:.3f}s, raw_means {loop_raw:.3f}s, "
            f"total_loop {time.perf_counter() - t_loop0:.3f}s"
        )
        _perf_log(
            f"[de_perf] t_test_approx_all_markers shape genes={n_genes} cells={n_cells} groups={n_groups}"
        )

    return results

def run_deseq2(
    data_matrix:  np.ndarray,         # (n_genes, n_cells), integer counts
    group_labels: np.ndarray,         # 1-D int array: 1 or 2 per cell
    batch_data:   np.ndarray | None,  # (n_cells, n_covariates) or None
    batch_names:  list[str],          # This method allows for batch as covariate in the model
    gene_names:   list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs DESeq2 via pydeseq2 (counts required).
    Returns (pvals, padj, lfc, ave_g1, ave_g2) all shape (n_genes,), NaN for untested genes.
    """
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds  import DeseqStats

    n_genes, n_cells = data_matrix.shape
    counts   = np.round(data_matrix.T).astype(int)  # cells x genes
    cell_ids = [f"cell_{i}" for i in range(n_cells)]

    meta = pd.DataFrame({"group": [str(g) for g in group_labels]}, index=cell_ids)
    if batch_data is not None:
        for j, bname in enumerate(batch_names):
            meta[bname] = batch_data[:, j].astype(str)

    counts_df      = pd.DataFrame(counts, index=cell_ids, columns=gene_names)
    design_factors = (batch_names + ["group"]) if batch_names else ["group"]
    design = "~ " + " + ".join(design_factors)

    dds = DeseqDataSet(counts=counts_df, metadata=meta, design=design, quiet=True, n_cpus = 1) # n_cpus intentionally set to 1: pydeseq2 multiprocessing triggers a Python 3.12 ResourceTracker bug (ChildProcessError on worker cleanup). Single-threaded is also typically faster for per-group DESeq2 runs due to joblib spawn overhead.
    with warnings.catch_warnings(): # UserWarning: Every gene contains at least one zero, cannot compute log geometric means. Switching to iterative mode.
        warnings.filterwarnings("ignore", category=UserWarning, message="Every gene contains at least one zero")
        dds.deseq2()

    stat_res = DeseqStats(dds, contrast=("group", "1", "2"), quiet=True)
    stat_res.summary()
    df = stat_res.results_df

    # Reindex by gene_names in one vectorised step; genes absent from results become NaN.
    df_reindexed = df.reindex(gene_names)
    pvals = df_reindexed["pvalue"].to_numpy()
    padj  = df_reindexed["padj"].to_numpy()
    lfc   = df_reindexed["log2FoldChange"].to_numpy()

    # Averages computed on the original input count matrix (--input-dataset values).
    mask_1 = np.array(meta["group"] == "1")
    mask_2 = np.array(meta["group"] == "2")
    ave_g1 = data_matrix[:, mask_1].mean(axis=1)   # shape (n_genes,)
    ave_g2 = data_matrix[:, mask_2].mean(axis=1)

    return pvals, padj, lfc, ave_g1, ave_g2

def run_deseq2_all_markers(
    data_matrix:  np.ndarray,         # (n_genes, n_cells), integer counts
    group_labels: np.ndarray,         # 1-D str array: group name per cell
    batch_data:   np.ndarray | None,
    batch_names:  list[str],
    gene_names:   list[str],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Runs DESeq2 for ALL groups vs. rest (FindAllMarkers equivalent for DESeq2).
    Loops once per group — unavoidable since DESeq2 is inherently pairwise.

    Returns a dict keyed by group name:
        group_name → (pvals, padj, lfc, ave_g1, ave_g2)  each shape (n_genes,)
    """
    unique_groups = sorted(set(group_labels.tolist()))
    results: dict = {}

    for grp in unique_groups:
        binary_labels = np.where(group_labels == grp, 1, 2).astype(int)
        pvals, padj, lfc, ave_g1, ave_g2 = run_deseq2(data_matrix, binary_labels, batch_data, batch_names, gene_names)
        results[grp] = (pvals, padj, lfc, ave_g1, ave_g2)

    return results

## Core logic
def run(args: argparse.Namespace) -> None:
    """Core DE logic, called after argument validation."""

    # Resolve output directory (only create when -o is set; TSV/volcano/JSON file output require it).
    input_path = Path(args.f).resolve()
    if args.o:
        output_dir = str(Path(args.o).resolve())
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    else:
        output_dir = str(input_path.parent)

    # Parse scalar arguments
    method_arg = args.method.strip()
    method = METHOD_NAME_MAP[method_arg]

    is_count_table = _parse_bool_string(args.is_count, "--is-count")
    write_tsv_output      = args.write_tsv
    write_volcano_output  = args.write_volcano
    write_metadata_output = args.write_metadata is not None
    split_files = args.split_files

    if (write_tsv_output or write_volcano_output) and not args.o:
        ErrorJSON("--write-tsv and --write-volcano require -o to be set (output folder is needed to write the files).")

    if method in COUNT_ONLY_METHODS and not is_count_table:
        ErrorJSON(f"Data should be a count table in order to run {method_arg}")

    if write_metadata_output:
        if not args.write_metadata.startswith("/attrs/"):
            ErrorJSON("--write-metadata path should refer to the /attrs/ path (e.g. /attrs/_de_2_wilcoxon)")

    tt_raw = getattr(args, "table_top_de_per_direction", None)
    if tt_raw is not None and int(tt_raw) < 1:
        ErrorJSON("--table-top-de-per-direction must be >= 1 when set")
    table_top_de_per_direction: int | None = None if tt_raw is None else int(tt_raw)

    data_warnings: list = []

    if not is_count_table:
        data_warnings.append("Data is assumed to be logged. If it's not the case, logFC could be wrong.")

    if split_files and not args.o:
        data_warnings.append(
            "--split-files was ignored because -o is not set; per-category cat_N.tsv files require an output folder."
        )
    split_files = bool(split_files and args.o)

    if split_files and not write_tsv_output:
        ErrorJSON("--split-files requires --write-tsv to be set.")
    if split_files and _null(args.group_dataset):
        ErrorJSON("--split-files requires --group-dataset (loom path to the primary annotation column).")
    if split_files and not _null(args.group1):
        ErrorJSON("--split-files is only valid in FindAllMarkers mode (omit --group).")

    preview_active, preview_fr, preview_seed, preview_min_cells, preview_max_cells = _preview_parse_args(args)
    rng = np.random.default_rng(preview_seed)

    if _null(args.group_dataset):
        ErrorJSON("--group-dataset is required (loom HDF5 path to the cell metadata column, e.g. /col_attrs/cell_type).")

    # ── Resolve group dataset paths (loom only; category lists come from the vectors) ──
    group1_raw: str | None = None if _null(args.group1) else args.group1.strip()
    group2_raw: str | None = None if _null(args.group2) else args.group2.strip()

    group_dataset_path:   str = args.group_dataset.strip()
    group_dataset_2_path: str | None = None if _null(args.group_dataset_2) else args.group_dataset_2.strip()
    group1: str | None = group1_raw
    group2: str | None = group2_raw
    all_categories:   list[str] | None = None  # sorted unique categories for --split-files (FindAllMarkers)
    list_cats_json_1: list[str] | None = None  # filled after reading loom; exposed in output JSON only in FindAllMarkers

    # ── Determine execution mode ─────────────────────────────────────────────────
    # Mode A — FindAllMarkers: --group omitted
    # Mode B — Single marker:  --group set, --group-2 / second dataset absent
    # Mode C — Two-group DE:   --group and --group-2 both set (second dataset optional)

    if group1 is None:
        # Mode A — FindAllMarkers
        if group2 is not None:
            ErrorJSON("When --group is omitted (FindAllMarkers mode), --group-2 must also be omitted.")
        mode = "all_markers"
    elif group2 is None and group_dataset_2_path is None:
        # Mode B — Single-group marker
        mode = "single_marker"
    else:
        # Mode C — Standard two-group DE
        if group2 is None:
            ErrorJSON("When a second group dataset is provided, --group-2 must also be provided.")
        if group_dataset_2_path is None:
            # Default: group 2 lives in the same annotation column as group 1
            group_dataset_2_path = group_dataset_path
        mode = "two_group"

    # Open the loom file for reading (close it afterwards to avoid locking during DE)
    with LoomHandler(str(input_path), mode="r") as loom:
        # Read count matrix from Loom file
        _read_cm = _perf_span("expression matrix load (loom or npy cache)") if _perf_enabled() else nullcontext()
        with _read_cm:
            data_matrix = _load_expression_matrix_with_optional_npy_cache(
                loom, str(input_path), args.input_dataset
            )
        if data_matrix is None:
            ErrorJSON(f"Input dataset {args.input_dataset!r} does not exist in the Loom file")
        n_genes, n_cells = data_matrix.shape

        # Gene annotations (used for volcano plot and output labeling)
        ens_raw  = loom.read_vector("/row_attrs/Accession")
        gene_raw = loom.read_vector("/row_attrs/Gene")
        ens_ids    = LoomHandler._decode_str_array(ens_raw).tolist()   if ens_raw  is not None else [f"Gene_{i}" for i in range(n_genes)]
        gene_names = LoomHandler._decode_str_array(gene_raw).tolist()  if gene_raw is not None else [f"Gene_{i}" for i in range(n_genes)]

        # Read primary group metadata (always required)
        groups_raw_1 = loom.read_vector(group_dataset_path)
        if groups_raw_1 is None:
            ErrorJSON(f"Group dataset {group_dataset_path!r} does not exist in the Loom file")
        groups_1 = LoomHandler._decode_str_array(groups_raw_1)

        # Second metadata column: two-group DE, or FindAllMarkers with optional --group-dataset-2 (union category list)
        need_groups_2 = mode == "two_group" or (
            mode == "all_markers"
            and group_dataset_2_path is not None
            and group_dataset_2_path.strip() != group_dataset_path.strip()
        )
        if need_groups_2:
            if group_dataset_2_path.strip() == group_dataset_path.strip():
                groups_2 = groups_1
            else:
                groups_raw_2 = loom.read_vector(group_dataset_2_path)
                if groups_raw_2 is None:
                    ErrorJSON(f"Group dataset 2 {group_dataset_2_path!r} does not exist in the Loom file")
                groups_2 = LoomHandler._decode_str_array(groups_raw_2)
                if int(groups_2.shape[0]) != n_cells:
                    ErrorJSON(
                        f"Group dataset 2 {group_dataset_2_path!r} length {int(groups_2.shape[0])} does not match "
                        f"number of cells ({n_cells}) in the loom."
                    )
        else:
            groups_2 = None

        # Read batch / covariate metadata
        batch_names: list[str] = []
        batch_data:  np.ndarray | None = None
        if not _null(args.batch):
            batch_paths = [p.strip() for p in args.batch.split(",") if p.strip()]
            if batch_paths:
                cols = []
                for bp in batch_paths:
                    bvec = loom.read_vector(bp)
                    if bvec is None:
                        ErrorJSON(f"Batch dataset {bp!r} does not exist in the Loom file")
                    cols.append(bvec.reshape(-1))
                    leaf = bp.split("/")[-1].lstrip("_")
                    batch_names.append(leaf)
                batch_data = np.column_stack(cols)  # (n_cells, n_batch_cols)
        if batch_data is not None and method not in BATCH_SUPPORT: # Warn if batch was requested for a method that does not support it
            data_warnings.append(f"Method '{method_arg}' does not support covariates — batch column(s) ignored. Covariate-aware DE is currently only supported by: {', '.join(sorted(BATCH_SUPPORT))}.")
            batch_data  = None
            batch_names = []

    # Loom file is now closed

    list_cats_json_1 = _unique_categories_sorted(groups_1)
    if mode == "all_markers" and groups_2 is not None:
        list_cats_json_1 = sorted(set(list_cats_json_1) | set(_unique_categories_sorted(groups_2)))

    if split_files:
        all_categories = list_cats_json_1

    result: dict = {}
    tsv_path: str | None = None
    split_tsv_paths: list[str] = []
    volcano_path: str | None = None
    metadata_path: str | None = None

    # ------------------------------------------------------------------ #
    # Mode A — FindAllMarkers                                             #
    # ------------------------------------------------------------------ #
    if mode == "all_markers":
        present_primary = {str(x) for x in groups_1.tolist()}
        unique_groups = [c for c in list_cats_json_1 if c in present_primary]
        if len(unique_groups) < 2:
            ErrorJSON("FindAllMarkers mode requires at least 2 distinct groups in --group-dataset.")

        # Validate minimum cell count per group
        group_sizes = {}
        for grp in unique_groups:
            cnt = int((groups_1 == grp).sum())
            group_sizes[grp] = cnt
            if cnt < 3:
                ErrorJSON(f"Group {grp!r} contains only {cnt} cell(s); at least 3 are required.")

        n_cells_full_am = int(n_cells)
        if preview_active:
            cell_idx = _preview_stratified_cell_indices(groups_1, preview_fr, rng, preview_min_cells, preview_max_cells)
            if preview_fr is None or preview_fr >= 1.0:
                cell_idx = _preview_cap_total_cells(cell_idx, preview_max_cells, rng)
            g1_sub = groups_1[cell_idx]
            for grp in unique_groups:
                if int((g1_sub == grp).sum()) < 3:
                    ErrorJSON(
                        f"After preview subsampling, group {grp!r} has fewer than 3 cells; "
                        "increase --preview-cell-fraction or disable preview."
                    )
            data_matrix = data_matrix[:, cell_idx]
            groups_1 = groups_1[cell_idx]
            if groups_2 is not None:
                groups_2 = groups_2[cell_idx]
            if batch_data is not None:
                batch_data = batch_data[cell_idx]
            n_genes, n_cells = data_matrix.shape
            data_warnings.append(
                f"Preview DE: cells {n_cells}/{n_cells_full_am} (stratified). cell_fraction={preview_fr!s} "
                f"min_cells={preview_min_cells} max_cells={preview_max_cells} seed={preview_seed}"
            )
            group_sizes = {grp: int((groups_1 == grp).sum()) for grp in unique_groups}

        if method == "deseq2":
            all_results = run_deseq2_all_markers(data_matrix, groups_1, batch_data, batch_names, gene_names)
        elif method == "t_test_approx":
            # One Welch+BHFDR run per category vs all other cells (vectorized; see module block above).
            with _perf_span("t_test_approx FindAllMarkers total (includes inner timings)"):
                all_results = run_t_test_approx_all_markers(data_matrix, groups_1, is_count=is_count_table)
        else:
            all_results = run_scanpy_all_markers(data_matrix, groups_1, gene_names, method, is_count=is_count_table)

        # Stack results as (5 * n_groups, n_genes) in DE_HEADERS order [lfc, pvals, fdr, ave_g1, ave_g2] per group.
        blocks = []
        n_genes_tested = 0
        n_genes_fdr_le_5pct = 0
        for grp in unique_groups:
            pvals, padj, lfc, ave_g1, ave_g2 = all_results[grp]
            blocks.append(np.vstack([lfc, pvals, padj, ave_g1, ave_g2]))
            n_genes_tested += _count_tested(pvals)
            n_genes_fdr_le_5pct += _count_significant(padj)
        data_out = np.vstack(blocks)  # shape: (5 * n_groups, n_genes)

        # TSV output — Extra leading column "group" to identify which group each row belongs to.
        if write_tsv_output:
            if split_files:
                # One file per group; file name is cat_{1-based index in sorted unique categories}.tsv
                if all_categories is None:
                    ErrorJSON("--split-files requires a category list; internal error (all_categories unset).")
                cat_index_map = {cat: i + 1 for i, cat in enumerate(all_categories)}
                for grp in unique_groups:
                    pvals_g, padj_g, lfc_g, ave_g1_g, ave_g2_g = all_results[grp]
                    grp_idx = cat_index_map.get(grp)
                    if grp_idx is None:
                        ErrorJSON(
                            f"Group {grp!r} was not found in the sorted unique category list for --group-dataset. "
                            "Cannot determine output file index for --split-files."
                        )
                    out_tsv = os.path.join(output_dir, f"cat_{grp_idx}.tsv")
                    write_tsv(
                        out_tsv,
                        ens_ids,
                        gene_names,
                        [(None, lfc_g, pvals_g, padj_g, ave_g1_g, ave_g2_g)],
                        table_top_de_per_direction=table_top_de_per_direction,
                    )
                    split_tsv_paths.append(out_tsv)
            else:
                tsv_rows = [(grp, all_results[grp][2], all_results[grp][0], all_results[grp][1], all_results[grp][3], all_results[grp][4]) for grp in unique_groups]  # (grp, lfc, pvals, fdr, ave_g1, ave_g2)
                out_tsv = os.path.join(output_dir, "output.tsv")
                tsv_path = write_tsv(
                    out_tsv,
                    ens_ids,
                    gene_names,
                    tsv_rows,
                    table_top_de_per_direction=table_top_de_per_direction,
                )

        # Volcano plot — not supported in FindAllMarkers mode (no single pairwise comparison to plot)
        if write_volcano_output:
            data_warnings.append("--write-volcano is not supported in FindAllMarkers mode and was skipped.")

        # Metadata output
        if write_metadata_output:
            metadata_data, metadata_headers = build_metadata_matrix_all_markers(
                ens_ids,
                gene_names,
                unique_groups,
                all_results,
                table_top_de_per_direction=table_top_de_per_direction,
            )
            metadata_path = write_metadata_dataset(str(input_path), args.write_metadata, metadata_data, metadata_headers)

        # JSON metadata (row count matches TSV / de_table)
        pool_m = np.arange(n_genes, dtype=np.intp)
        nber_rows_meta = _fam_table_row_count(unique_groups, all_results, pool_m, table_top_de_per_direction)
        result = build_result_json(
            args.write_metadata if write_metadata_output else None,
            nber_rows_meta,
            unique_groups,
        )
        result["mode"] = "FindAllMarkers"
        result["number_of_groups"] = len(unique_groups)
        result["group_sizes"] = group_sizes
        result["number_of_genes_tested"] = n_genes_tested
        result["number_of_genes_with_fdr_le_5pct"] = n_genes_fdr_le_5pct
        result["overlapping_cells"] = 0
        if preview_active:
            result["preview"] = {
                "applied": True,
                "cell_fraction": preview_fr,
                "seed": preview_seed,
                "min_cells": preview_min_cells,
                "max_cells": preview_max_cells,
                "n_cells_full": n_cells_full_am,
            }

    # ------------------------------------------------------------------ #
    # Mode B — Single-group marker (group1 vs. all other cells)           #
    # ------------------------------------------------------------------ #
    elif mode == "single_marker":
        candidates_1 = np.where(groups_1 == group1)[0]
        if len(candidates_1) == 0:
            ErrorJSON(f"Group 1 label {group1!r} was not found in {group_dataset_path!r}.")

        cell_group = np.zeros(n_cells, dtype=int)
        cell_group[candidates_1] = 1
        cell_group[cell_group == 0] = 2  # all other cells become group 2

        group1_size = int((cell_group == 1).sum())
        group2_size = int((cell_group == 2).sum())

        if group1_size < 3:
            ErrorJSON(f"Group 1 ({group1!r}) contains fewer than 3 cells.")
        if group2_size < 3:
            ErrorJSON("Group 2 (all other cells) contains fewer than 3 cells.")

        n_cells_full_sm = int(n_cells)
        if preview_active:
            cell_idx = _preview_stratified_cell_indices(cell_group, preview_fr, rng, preview_min_cells, preview_max_cells)
            if preview_fr is None or preview_fr >= 1.0:
                cell_idx = _preview_cap_total_cells(cell_idx, preview_max_cells, rng)
            cg_sub = cell_group[cell_idx]
            for lab in (1, 2):
                if int((cg_sub == lab).sum()) < 3:
                    ErrorJSON(
                        "After preview subsampling, one of the cell groups has fewer than 3 cells; "
                        "increase --preview-cell-fraction or disable preview."
                    )
            data_matrix = data_matrix[:, cell_idx]
            cell_group = cg_sub
            if batch_data is not None:
                batch_data = batch_data[cell_idx]
            n_genes, n_cells = data_matrix.shape
            group1_size = int((cell_group == 1).sum())
            group2_size = int((cell_group == 2).sum())
            data_warnings.append(
                f"Preview DE: cells {n_cells}/{n_cells_full_sm} (stratified by marker vs rest). "
                f"cell_fraction={preview_fr!s} min_cells={preview_min_cells} max_cells={preview_max_cells} seed={preview_seed}"
            )

        if method == "deseq2":
            out_pvals, out_fdr, out_lfc, ave_g1, ave_g2 = run_deseq2(data_matrix, cell_group, batch_data, batch_names, gene_names)
        elif method == "t_test_approx":
            # Marker cells vs rest: same two-sample Welch as two_group (boolean masks along cell_group).
            m1 = cell_group == 1
            m2 = cell_group == 2
            with _perf_span("t_test_approx single_marker total"):
                out_pvals, out_fdr, out_lfc, ave_g1, ave_g2 = run_t_test_approx_pairwise(
                    data_matrix, m1, m2, is_count=is_count_table
                )
        else:
            out_pvals, out_fdr, out_lfc, ave_g1, ave_g2 = run_scanpy_method(data_matrix, cell_group, gene_names, method, is_count=is_count_table)

        data_out = np.vstack([out_lfc, out_pvals, out_fdr, ave_g1, ave_g2])  # (5, n_genes)

        # TSV — no extra "group" column
        if write_tsv_output:
            out_tsv = os.path.join(output_dir, "output.tsv")
            tsv_path = write_tsv(
                out_tsv,
                ens_ids,
                gene_names,
                [(None, out_lfc, out_pvals, out_fdr, ave_g1, ave_g2)],
                table_top_de_per_direction=table_top_de_per_direction,
            )

        # VOLCANO plot - write as JSON
        if write_volcano_output:
            volcano_path = write_volcano_json(out_lfc, out_pvals, gene_names, ens_ids, output_dir)

        # Metadata output
        if write_metadata_output:
            metadata_data, metadata_headers = build_metadata_matrix_single(
                ens_ids,
                gene_names,
                out_lfc,
                out_pvals,
                out_fdr,
                ave_g1,
                ave_g2,
                table_top_de_per_direction=table_top_de_per_direction,
            )
            metadata_path = write_metadata_dataset(str(input_path), args.write_metadata, metadata_data, metadata_headers)

        # JSON for output JSON file
        pool_sm = np.arange(n_genes, dtype=np.intp)
        nber_rows_sm = int(_de_table_gene_order_pairwise(out_lfc, out_pvals, pool_sm, table_top_de_per_direction).size)
        result = build_result_json(args.write_metadata if write_metadata_output else None, nber_rows_sm)
        result["mode"] = "Single marker"
        result["group_name"] = group1
        result["group2_name"] = None
        result["group_size"] = group1_size
        result["group2_size"] = group2_size
        result["number_of_genes_tested"] = _count_tested(out_pvals)
        result["number_of_genes_with_fdr_le_5pct"] = _count_significant(out_fdr)
        result["overlapping_cells"] = 0
        if preview_active:
            result["preview"] = {
                "applied": True,
                "cell_fraction": preview_fr,
                "seed": preview_seed,
                "n_cells_full": n_cells_full_sm,
                "min_cells": preview_min_cells,
                "max_cells": preview_max_cells,
            }

    # ------------------------------------------------------------------ #
    # Mode C — Standard two-group DE                                      #
    # ------------------------------------------------------------------ #
    else:  # mode == "two_group"
        candidates_1 = np.where(groups_1 == group1)[0]
        candidates_2 = np.where(groups_2 == group2)[0]

        if len(candidates_1) == 0:
            ErrorJSON(f"Group 1 label {group1!r} was not found in {group_dataset_path!r}.")
        if len(candidates_2) == 0:
            ErrorJSON(f"Group 2 label {group2!r} was not found in {group_dataset_2_path!r}.")

        # Remove cells that appear in both groups from group 2
        overlap = np.intersect1d(candidates_1, candidates_2)
        if len(overlap) > 0:
            data_warnings.append(f"{len(overlap)} cell(s) appear in both group 1 ({group1!r}) and group 2 ({group2!r}). These cells were removed from group 2 to avoid overlap. Affected indices (first 20): {overlap[:20].tolist()} {' ...' if len(overlap) > 20 else ''}")
        non_overlap_2 = np.setdiff1d(candidates_2, candidates_1)

        cell_group = np.zeros(n_cells, dtype=int)
        cell_group[candidates_1]  = 1
        cell_group[non_overlap_2] = 2

        # Keep only assigned cells
        keep_idx        = np.where(cell_group > 0)[0]
        data_matrix_sub = data_matrix[:, keep_idx]
        cell_group_sub  = cell_group[keep_idx]
        batch_data_sub  = batch_data[keep_idx] if batch_data is not None else None

        group1_size = int((cell_group_sub == 1).sum())
        group2_size = int((cell_group_sub == 2).sum())

        if group1_size < 3:
            ErrorJSON(f"Group 1 ({group1!r}) contains fewer than 3 cells.")
        if group2_size < 3:
            ErrorJSON(f"Group 2 ({group2!r}) contains fewer than 3 cells after removing overlapping cells.")

        n_cells_sub_full = int(data_matrix_sub.shape[1])
        if preview_active:
            cell_idx = _preview_stratified_cell_indices(cell_group_sub, preview_fr, rng, preview_min_cells, preview_max_cells)
            if preview_fr is None or preview_fr >= 1.0:
                cell_idx = _preview_cap_total_cells(cell_idx, preview_max_cells, rng)
            cg2 = cell_group_sub[cell_idx]
            for lab in (1, 2):
                if int((cg2 == lab).sum()) < 3:
                    ErrorJSON(
                        "After preview subsampling, one of the cell groups has fewer than 3 cells; "
                        "increase --preview-cell-fraction or disable preview."
                    )
            data_matrix_sub = data_matrix_sub[:, cell_idx]
            cell_group_sub = cg2
            if batch_data_sub is not None:
                batch_data_sub = batch_data_sub[cell_idx]
            group1_size = int((cell_group_sub == 1).sum())
            group2_size = int((cell_group_sub == 2).sum())
            data_warnings.append(
                f"Preview DE: cells {data_matrix_sub.shape[1]}/{n_cells_sub_full} (stratified by two groups). "
                f"cell_fraction={preview_fr!s} min_cells={preview_min_cells} max_cells={preview_max_cells} seed={preview_seed}"
            )

        if method == "deseq2":
            out_pvals, out_fdr, out_lfc, ave_g1, ave_g2 = run_deseq2(data_matrix_sub, cell_group_sub, batch_data_sub, batch_names, gene_names)
        elif method == "t_test_approx":
            # Explicit two labels: subset matrix already contains only G1 and G2 cells; masks pick columns.
            m1 = cell_group_sub == 1
            m2 = cell_group_sub == 2
            with _perf_span("t_test_approx two_group total"):
                out_pvals, out_fdr, out_lfc, ave_g1, ave_g2 = run_t_test_approx_pairwise(
                    data_matrix_sub, m1, m2, is_count=is_count_table
                )
        else:
            out_pvals, out_fdr, out_lfc, ave_g1, ave_g2 = run_scanpy_method(data_matrix_sub, cell_group_sub, gene_names, method, is_count=is_count_table)

        data_out = np.vstack([out_lfc, out_pvals, out_fdr, ave_g1, ave_g2])  # (5, n_genes)

        if write_tsv_output:
            out_tsv = os.path.join(output_dir, "output.tsv")
            tsv_path = write_tsv(
                out_tsv,
                ens_ids,
                gene_names,
                [(None, out_lfc, out_pvals, out_fdr, ave_g1, ave_g2)],
                table_top_de_per_direction=table_top_de_per_direction,
            )

        if write_volcano_output:
            volcano_path = write_volcano_json(out_lfc, out_pvals, gene_names, ens_ids, output_dir)

        if write_metadata_output:
            metadata_data, metadata_headers = build_metadata_matrix_single(
                ens_ids,
                gene_names,
                out_lfc,
                out_pvals,
                out_fdr,
                ave_g1,
                ave_g2,
                table_top_de_per_direction=table_top_de_per_direction,
            )
            metadata_path = write_metadata_dataset(str(input_path), args.write_metadata, metadata_data, metadata_headers)

        pool_tg = np.arange(n_genes, dtype=np.intp)
        nber_rows_tg = int(_de_table_gene_order_pairwise(out_lfc, out_pvals, pool_tg, table_top_de_per_direction).size)
        result = build_result_json(args.write_metadata if write_metadata_output else None, nber_rows_tg)
        result["mode"] = "Two-group DE"
        result["group_name"] = group1
        result["group2_name"] = group2
        result["group_size"] = group1_size
        result["group2_size"] = group2_size
        result["number_of_genes_tested"] = _count_tested(out_pvals)
        result["number_of_genes_with_fdr_le_5pct"] = _count_significant(out_fdr)
        result["overlapping_cells"] = len(overlap)
        if preview_active:
            result["preview"] = {
                "applied": True,
                "cell_fraction": preview_fr,
                "seed": preview_seed,
                "n_cells_in_two_group_subset": n_cells_sub_full,
                "min_cells": preview_min_cells,
                "max_cells": preview_max_cells,
            }

    # JSON output (ordered category lists are only attached for FindAllMarkers / all-vs-rest)
    if mode == "all_markers":
        result["list_cats_json"] = list_cats_json_1
    if tsv_path is not None:
        result["tsv_path"] = tsv_path
    if split_tsv_paths:
        result["tsv_paths"] = split_tsv_paths
    if volcano_path is not None:
        result["volcano_path"] = volcano_path
    if data_warnings:
        result["warnings"] = data_warnings

    if not args.o:
        if mode == "all_markers":
            md, mh = build_metadata_matrix_all_markers(
                ens_ids,
                gene_names,
                unique_groups,
                all_results,
                table_top_de_per_direction=table_top_de_per_direction,
            )
        else:
            md, mh = build_metadata_matrix_single(
                ens_ids,
                gene_names,
                out_lfc,
                out_pvals,
                out_fdr,
                ave_g1,
                ave_g2,
                table_top_de_per_direction=table_top_de_per_direction,
            )
        result["de_table"] = _de_table_for_json(md, mh)

    if table_top_de_per_direction is not None:
        result["table_top_de_per_direction"] = table_top_de_per_direction

    if args.o:
        out_json = os.path.join(output_dir, "output.json")
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, allow_nan=True)
    else:
        print(json.dumps(result, ensure_ascii=False, allow_nan=True), flush=True)

## Helpers
def _null(s: str | None) -> bool:
    """Returns True if a string argument represents an absent/null value."""
    return s is None or s.strip().lower() in ("", "null", "na", "none")

def _parse_bool_string(value: str, parameter_name: str) -> bool:
    """Parses a string boolean argument."""
    raw_value = value.strip().lower()
    if raw_value == "true":
        return True
    if raw_value == "false":
        return False
    ErrorJSON(f"{parameter_name} should be 'true' or 'false'")

## Help text
custom_help = """
Differential Expression (Python / Scanpy)

Execution modes
───────────────
  FindAllMarkers  Omit --group and --group-2. Every group in --group-dataset is tested vs. rest.
                  Optional --group-dataset-2 merges unique values from both columns for
                  list_cats_json (still DE on --group-dataset only). Categories are sorted
                  alphanumerically (Ruby-like string order). list_cats_json is in output.json
                  and drives --split-files cat_N.tsv indices.

  Single marker   Provide --group only (omit --group-2 and --group-dataset-2).
                  Runs group vs. all other cells.

  Two-group DE    Provide --group and --group-2 with the exact category names in the loom columns.
                  Standard pairwise DE; overlapping cells are removed from group 2.

Annotation columns (loom HDF5 paths only, no database):
  --group-dataset            Required. Path to the obs-level metadata column (e.g. /col_attrs/cell_type).
  --group-dataset-2          Optional second column for two-group DE when group 1 and group 2
                             use different metadata. Defaults to --group-dataset when omitted.

Options:
  -f                         Path to the input .loom file.
  -o                         Output folder. When omitted, no output.json is written; the full JSON (including
                             a de_table with the same columns and row order as output.tsv) is printed to stdout.
                             --write-tsv and --write-volcano require -o.
  --method                   DE method: wilcoxon | t_test | t_test_overestim_var | t_test_approx | deseq2
                             t_test_approx: per-gene Welch two-sample t-test on working expression (GEMV sums
                             of X and X^2 over cells), Benjamini-Hochberg across genes per contrast. Counts:
                             normalize each cell to 10k then log1p for the test; Avg. Exp. columns use raw counts.
                             Non-count: test and Avg. Exp. use the loaded matrix values (e.g. normalized from loom).
  --input-dataset            Loom path to the expression matrix (e.g. /layers/norm_data). On first use,
                             a float32 .npy sidecar is written next to the loom (slashes in the path
                             replaced by underscores, e.g. _layers_norm_1_seurat.npy). If that file
                             already exists it is used for loads (loom mtime is ignored because the
                             loom may be updated without changing this matrix). Delete the sidecar to
                             force a refresh from the loom.
  --batch                    Comma-separated loom paths for batch/covariate columns, or "null".
                             Only used by deseq2.

  --group-dataset            Loom path to the primary cell annotation column (required).
  --group-dataset-2          Loom path to the second annotation column when needed.

  --group                    Category name in the primary column, or null for FindAllMarkers.
  --group-2                  Category name in the second column for two-group DE.

  --is-count                 true | false (default false). Required true for deseq2.
  --write-tsv                Write output.tsv (requires -o).
  --write-volcano            Write output.plot.json volcano plot (requires -o).
  --write-metadata           Write metadata dataset in the loom at the given /attrs/ path.
  --split-files              FindAllMarkers + -o + --write-tsv: one cat_N.tsv per category order
                             (N is 1-based index in sorted unique categories from the loom). Ignored without -o
                             (a warning is added to the JSON warnings list).

  --table-top-de-per-direction  Optional. Limits TSV, stdout de_table, and --write-metadata rows to at most N
                             genes with log FC > 0 and N with log FC < 0, each side ranked by ascending p-value
                             (then by LFC). FindAllMarkers applies this per group vs rest. Does not change which
                             genes are tested.

  --preview-cell-fraction    (0, 1]: enables preview mode: per category, keep about this fraction of cells
                             (stratified random sample). All genes are still tested on the cell subset.
                             Categories with fewer than --preview-min-cells cells keep all of them. Otherwise at
                             least preview-min-cells are kept when subsampling. Omit to disable preview.
  --preview-min-cells        Minimum cells per stratum when subsampling (default 1000). Strata smaller than this
                             use every cell in that stratum.
  --preview-max-cells        Per-stratum cap when fraction is in (0, 1): target count is clamped to at most this
                             value (default 10000). With --preview-cell-fraction 1, all cells are kept per stratum
                             then total cells are capped to this many by a uniform subsample.
  --preview-seed             RNG seed for preview subsampling (default 42).

Environment:
  ASAP_DE_PERF_LOG          If set to 1, true, or yes: print [de_perf] timings on stderr for
                             t_test_approx (prep_X, X squared + GEMV, mean/var + Welch + BH-FDR, raw means;
                             FindAllMarkers also logs per-group loop breakdown).

  --help                     Show this help message and exit.
"""

## Entry point
def main() -> None:
    if "--help" in sys.argv:
        print(custom_help)
        sys.exit(0)

    parser = argparse.ArgumentParser(description="Differential Expression Script", add_help=False)
    parser.add_argument("-f",                    metavar="Input loom file",                                              required=True)
    parser.add_argument("-o",                    metavar="Output folder",                                                required=False)
    parser.add_argument("--method",              metavar="DE method",                       choices=AVAILABLE_METHODS,   required=True)
    parser.add_argument("--input-dataset",       metavar="Input matrix dataset",                                         required=True,  dest="input_dataset")
    parser.add_argument("--batch",               metavar="Batch/covariate datasets",        default="null",              required=False)

    parser.add_argument("--group-dataset",       metavar="Group 1 annotation path",                                      required=False, dest="group_dataset",   default=None)
    parser.add_argument("--group-dataset-2",     metavar="Group 2 annotation path",         default=None,                required=False, dest="group_dataset_2")

    # Group labels
    parser.add_argument("--group",             metavar="Group 1 label (null = all)",      default="null",              required=False, dest="group1")
    parser.add_argument("--group-2",           metavar="Group 2 label",                   default="null",              required=False, dest="group2")

    # Flags
    parser.add_argument("--is-count",          metavar="Is count table [true|false]",     default="false",             required=False, dest="is_count")
    parser.add_argument("--write-tsv",         action="store_true",                       required=False,              dest="write_tsv")
    parser.add_argument("--write-volcano",     action="store_true",                       required=False,              dest="write_volcano")
    parser.add_argument("--write-metadata",    metavar="Output metadata dataset path",    required=False,              dest="write_metadata", default=None)
    parser.add_argument("--split-files",       action="store_true",                       required=False,              dest="split_files")
    parser.add_argument(
        "--table-top-de-per-direction",
        type=int,
        default=None,
        required=False,
        dest="table_top_de_per_direction",
        metavar="N",
    )
    parser.add_argument("--preview-cell-fraction", type=float, default=None, required=False, dest="preview_cell_fraction", metavar="F")
    parser.add_argument("--preview-min-cells", type=int, default=1000, required=False, dest="preview_min_cells", metavar="M")
    parser.add_argument("--preview-max-cells", type=int, default=10000, required=False, dest="preview_max_cells", metavar="C")
    parser.add_argument("--preview-seed",      type=int,   default=42,   required=False, dest="preview_seed", metavar="SEED")

    args = parser.parse_args()

    run(args)

if __name__ == "__main__":
    main()