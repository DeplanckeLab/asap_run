from __future__ import annotations

## Standard library
import argparse        # Argument parsing
import json            # For writing output files
import os              # For path/directory operations
import sys             # For ErrorJSON exit
import time            # For retry sleep
import warnings        # For non-fatal warnings
from pathlib import Path  # For path manipulation
from typing import Final

## Database
from contextlib import contextmanager
from typing import Any, Optional
from urllib.parse import urlparse
import psycopg2
from psycopg2.extensions import connection as PGConnection
from psycopg2.extras import RealDictCursor

## Scientific stack
import h5py            # For reading/writing loom (HDF5) files
import numpy as np     # For array calculations
import scipy.sparse as sp  # For sparse matrix handling
import pandas as pd # For matrix computation

## DE / stats (scanpy, anndata, statsmodels, pydeseq2 imported lazily where needed)
# scanpy, anndata  → run_scanpy_method() / run_scanpy_all_markers()
# pydeseq2         → run_deseq2() / run_deseq2_all_markers()

## Constants
AVAILABLE_METHODS: Final[list] = [
    "wilcoxon",             # Wilcoxon rank-sum test           (scanpy)
    "t_test",               # Student's t-test, pooled var     (scanpy)
    "t_test_overestim_var", # Student's t-test, per-gene var   (scanpy)
    "deseq2",               # Negative-binomial GLM            (pydeseq2, counts only)
]
METHOD_NAME_MAP: Final[dict[str, str]] = {
    "wilcoxon": "wilcoxon",
    "t_test": "t-test",
    "t_test_overestim_var": "t-test_overestim_var",
    "deseq2": "deseq2",
}
COUNT_ONLY_METHODS: Final[set] = {"deseq2"}
BATCH_SUPPORT:      Final[set] = {"deseq2"}
DE_HEADERS: Final[list] = ["log Fold-Change", "p-value", "FDR", "Avg. Exp. Group 1", "Avg. Exp. Group 2"]
DE_NB_COLS: Final[int] = len(DE_HEADERS)  # Number of DE output columns stored per gene
DEFAULT_DB_PORT: Final[int] = 5432
DEFAULT_CONNECT_TIMEOUT: Final[int] = 10

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

## Output helpers
def write_tsv(out_path: str, ens_ids: list[str], gene_names: list[str], rows: list[tuple]) -> str:
    """
    Write DE results to a TSV file.
    rows is a list of (group_label | None, lfc, pvals, fdr, ave_g1, ave_g2).
    If group_label is not None for the first entry, a leading 'group' column is included.
    """
    has_group = rows[0][0] is not None
    headers   = (["group", "ensembl_id", "gene_name"] if has_group else ["ensembl_id", "gene_name"]) + DE_HEADERS
    n_genes   = len(gene_names)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(headers) + "\n")
        for grp, lfc, pvals, fdr, ave_g1, ave_g2 in rows:
            pvals_for_sort = np.where(np.isnan(pvals), np.inf, pvals)  # NaN → inf so they sort last
            sort_idx = np.lexsort((-lfc, pvals_for_sort))  # primary: ascending p-value; secondary: descending lfc
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

def build_result_json(output_dataset: str | None, n_genes: int, unique_groups:  list[str] | None = None) -> dict:
    """
    Build the output.json metadata dict.
    If unique_groups is provided (Mode A), produces multi-group metadata with flat headers labelled by group name.
    Otherwise (Modes B/C), produces single-comparison metadata.
    """
    if unique_groups is not None:
        all_headers_flat = [f"{h} ({grp})" for grp in unique_groups for h in DE_HEADERS]
        return {
            "metadata": [
                {
                    "name":      output_dataset,
                    "on":        "GLOBAL",
                    "type":      "NUMERIC",
                    "nber_cols": len(all_headers_flat),
                    "nber_rows": n_genes,
                    "header":   all_headers_flat,
                    "groups":    unique_groups,
                }
            ]
        }
    return {
        "metadata": [
            {
                "name":      output_dataset,
                "on":        "GLOBAL",
                "type":      "NUMERIC",
                "nber_cols": DE_NB_COLS,
                "nber_rows": n_genes,
                "header":   DE_HEADERS,
            }
        ]
    }

def build_metadata_matrix_single(ens_ids: list[str], gene_names: list[str], lfc: np.ndarray, pvals: np.ndarray, fdr: np.ndarray, ave_g1: np.ndarray, ave_g2: np.ndarray) -> tuple[np.ndarray, list[str]]:
    headers = ["ensembl_id", "gene_name"] + DE_HEADERS
    # Sort: ascending p-value (NaN last), secondary: descending lfc
    pvals_for_sort = np.where(np.isnan(pvals), np.inf, pvals)
    sort_idx = np.lexsort((-lfc, pvals_for_sort))
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

def build_metadata_matrix_all_markers(ens_ids: list[str], gene_names: list[str], unique_groups: list[str], all_results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, list[str]]:
    """
    Builds the metadata matrix for FindAllMarkers in the same tall format as the TSV:
    one row per gene per group, sorted by ascending p-value then descending lfc,
    with a leading 'group' column. Matches write_tsv() output exactly.
    """
    headers = ["group", "ensembl_id", "gene_name"] + DE_HEADERS
    rows = []
    for grp in unique_groups:
        pvals, padj, lfc, ave_g1, ave_g2 = all_results[grp]
        pvals_for_sort = np.where(np.isnan(pvals), np.inf, pvals)
        sort_idx = np.lexsort((-lfc, pvals_for_sort))
        for i in sort_idx:
            rows.append([
                grp,
                ens_ids[i],
                gene_names[i],
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

    # Resolve output directory
    input_path = Path(args.f).resolve()
    output_dir = Path(args.o).resolve() if args.o else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = str(output_dir)

    # Parse scalar arguments
    method_arg = args.method.strip()
    method = METHOD_NAME_MAP[method_arg]

    is_count_table = _parse_bool_string(args.is_count, "--is-count")
    write_tsv_output      = args.write_tsv
    write_volcano_output  = args.write_volcano
    write_metadata_output = args.write_metadata is not None
    split_files           = args.split_files

    if (write_tsv_output or write_volcano_output) and not args.o:
        ErrorJSON("--write-tsv and --write-volcano require -o to be set (output folder is needed to write the files).")

    if split_files and not write_tsv_output:
        ErrorJSON("--split-files requires --write-tsv to be set.")
    if split_files and args.group_dataset_id is None:
        ErrorJSON("--split-files requires --group-dataset-id to be set (the file index is the 1-based position of the group in annots.list_cat_json).")

    if method in COUNT_ONLY_METHODS and not is_count_table:
        ErrorJSON(f"Data should be a count table in order to run {method_arg}")

    if write_metadata_output:
        if not args.write_metadata.startswith("/attrs/"):
            ErrorJSON("--write-metadata path should refer to the /attrs/ path (e.g. /attrs/_de_2_wilcoxon)")

    # Init warning list early — needed before DB resolution block below
    data_warnings: list = []

    if not is_count_table:
        data_warnings.append("Data is assumed to be logged. If it's not the case, logFC could be wrong.")

    # ── Determine dataset/id resolution style ───────────────────────────────────
    # Two mutually exclusive styles:
    #   DB-based:   --group-dataset-id / --group-dataset-id-2   (loom path + category names fetched from DB)
    #   Path-based: --group-dataset    / --group-dataset-2      (loom paths provided directly)
    # Mixing styles between group 1 and group 2 is not allowed.

    use_db_g1 = args.group_dataset_id  is not None   # --group-dataset-id was supplied
    use_db_g2 = args.group_dataset_id2 is not None   # --group-dataset-id-2 was supplied

    # ── Consistency checks ───────────────────────────────────────────────────────
    # Rule: cannot use --group-dataset-id-2 without --group-dataset-id
    if use_db_g2 and not use_db_g1:
        ErrorJSON("--group-dataset-id-2 requires --group-dataset-id to also be set.")

    # Rule: if DB-based for group 1, the second group must also be DB-based (or absent),
    #       i.e. --group-dataset-2 (path-based) is forbidden when --group-dataset-id is set.
    if use_db_g1 and not _null(args.group_dataset_2):
        ErrorJSON("Cannot mix --group-dataset-id with --group-dataset-2. When --group-dataset-id is used, specify the second group annotation via --group-dataset-id-2 instead.")

    # Rule: if path-based for group 1, the second group must also be path-based,
    #       i.e. --group-dataset-id-2 is forbidden when --group-dataset is set (and --group-dataset-id is absent).
    if not use_db_g1 and use_db_g2:
        # Already caught by "use_db_g2 and not use_db_g1" above, but kept for clarity.
        ErrorJSON("--group-dataset-id-2 requires --group-dataset-id to also be set.")

    # ── --dburl validation ───────────────────────────────────────────────────────
    if (use_db_g1 or use_db_g2) and _null(args.dburl):
        ErrorJSON("--dburl is required when --group-dataset-id or --group-dataset-id-2 are set.")

    if not use_db_g1 and not use_db_g2 and not _null(args.dburl):
        data_warnings.append("--dburl is set but not used (neither --group-dataset-id nor --group-dataset-id-2 are set).")

    # ── Resolve group dataset paths and group labels ─────────────────────────────
    group1_raw: str | None = None if _null(args.group1) else args.group1.strip()
    group2_raw: str | None = None if _null(args.group2) else args.group2.strip()

    group_dataset_path:   str       # resolved loom path for group 1's annotation column
    group_dataset_2_path: str | None = None  # resolved loom path for group 2's annotation column (None → defaults to group_dataset_path in Mode C)
    group1: str | None = group1_raw
    group2: str | None = group2_raw
    all_categories:   list[str] | None = None  # full category list from DB, populated when --split-files is set
    list_cats_json_1: list[str] | None = None  # ordered category list for annotation 1 (for output JSON)
    list_cats_json_2: list[str] | None = None  # ordered category list for annotation 2 (for output JSON, only when different from annotation 1)

    if use_db_g1:
        # ── DB-based resolution ─────────────────────────────────────────────────
        user_env = get_env("POSTGRES_USER", required=True)
        pass_env = get_env("POSTGRES_PASSWORD", required=True)
        host, port, dbname = parse_host_string(args.dburl)
        db = DBManager(host=host, dbname=dbname, port=port, user=user_env, password=pass_env)
        db.connect()
        try:
            # --- Group 1 dataset path ---
            db_g1_path = db.get_annotation_col_attr(args.group_dataset_id)
            if not _null(args.group_dataset):
                provided = args.group_dataset.strip()
                if provided != db_g1_path:
                    ErrorJSON(f"--group-dataset {provided!r} does not match the loom path in the DB for annotation id {args.group_dataset_id} ({db_g1_path!r}). Remove --group-dataset or correct the value.")
                data_warnings.append(f"--group-dataset is redundant when --group-dataset-id is set (DB resolved path: {db_g1_path!r}). It can be omitted.")
            group_dataset_path = db_g1_path

            # Always fetch the category list for annotation 1 (used for output JSON and --split-files)
            list_cats_json_1 = db.get_list_cat_json(args.group_dataset_id)
            if split_files:
                all_categories = list_cats_json_1

            # Resolve group1 numeric label → display name (if applicable)
            group1 = _resolve_group_name_from_annotation_id(args.group_dataset_id, group1_raw, db)

            # --- Group 2 dataset path ---
            if use_db_g2:
                db_g2_path = db.get_annotation_col_attr(args.group_dataset_id2)
                if not _null(args.group_dataset_2):
                    provided2 = args.group_dataset_2.strip()
                    if provided2 != db_g2_path:
                        ErrorJSON(f"--group-dataset-2 {provided2!r} does not match the loom path in the DB for annotation id {args.group_dataset_id2} ({db_g2_path!r}). Remove --group-dataset-2 or correct the value.")
                    data_warnings.append(f"--group-dataset-2 is redundant when --group-dataset-id-2 is set (DB resolved path: {db_g2_path!r}). It can be omitted.")
                group_dataset_2_path = db_g2_path
                # Resolve group2 numeric label → display name using its own annotation id
                group2 = _resolve_group_name_from_annotation_id(args.group_dataset_id2, group2_raw, db)
                # Fetch category list for annotation 2 (different from annotation 1)
                list_cats_json_2 = db.get_list_cat_json(args.group_dataset_id2)
            else:
                # Both groups share the same annotation column — resolve group2 with group 1's id
                group2 = _resolve_group_name_from_annotation_id(args.group_dataset_id, group2_raw, db)
                # group_dataset_2_path stays None here; defaults to group_dataset_path in Mode C below
        finally:
            db.disconnect()

    else:
        # ── Path-based resolution ───────────────────────────────────────────────
        if _null(args.group_dataset):
            ErrorJSON("--group-dataset is required when --group-dataset-id is not set.")
        group_dataset_path   = args.group_dataset.strip()
        group_dataset_2_path = None if _null(args.group_dataset_2) else args.group_dataset_2.strip()

    # ── Determine execution mode ─────────────────────────────────────────────────
    # Mode A — FindAllMarkers: --group omitted
    # Mode B — Single marker:  --group set, --group-2 / second dataset absent
    # Mode C — Two-group DE:   --group and --group-2 both set (second dataset optional)

    if group1 is None:
        # Mode A — FindAllMarkers
        if group2 is not None or group_dataset_2_path is not None:
            ErrorJSON("When --group is omitted (FindAllMarkers mode), --group-2 and --group-dataset-2 / --group-dataset-id-2 must also be omitted.")
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
        data_matrix = loom.read_matrix(args.input_dataset)
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

        # Group 2 metadata (only for Mode C, may come from a different column)
        if mode == "two_group":
            if group_dataset_2_path != group_dataset_path:
                groups_raw_2 = loom.read_vector(group_dataset_2_path)
                if groups_raw_2 is None:
                    ErrorJSON(f"Group dataset 2 {group_dataset_2_path!r} does not exist in the Loom file")
                groups_2 = LoomHandler._decode_str_array(groups_raw_2)
            else:
                groups_2 = groups_1  # same column
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

    # Derive category lists from loom data when DB was not used (path-based resolution)
    if list_cats_json_1 is None:
        list_cats_json_1 = sorted(set(groups_1.tolist()))
    if list_cats_json_2 is None and groups_2 is not None and group_dataset_2_path != group_dataset_path:
        list_cats_json_2 = sorted(set(groups_2.tolist()))

    result: dict = {}
    tsv_path: str | None = None
    split_tsv_paths: list[str] = []
    volcano_path: str | None = None
    metadata_path: str | None = None

    # ------------------------------------------------------------------ #
    # Mode A — FindAllMarkers                                             #
    # ------------------------------------------------------------------ #
    if mode == "all_markers":
        unique_groups = sorted(set(groups_1))
        if len(unique_groups) < 2:
            ErrorJSON("FindAllMarkers mode requires at least 2 distinct groups in --group-dataset.")

        # Validate minimum cell count per group
        group_sizes = {}
        for grp in unique_groups:
            cnt = int((groups_1 == grp).sum())
            group_sizes[grp] = cnt
            if cnt < 3:
                ErrorJSON(f"Group {grp!r} contains only {cnt} cell(s); at least 3 are required.")

        if method == "deseq2":
            all_results = run_deseq2_all_markers(data_matrix, groups_1, batch_data, batch_names, gene_names)
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
                # One file per group; file name is cat_{1-based DB index}.tsv
                if all_categories is None:
                    # Should not happen (validated earlier), but guard defensively
                    ErrorJSON("--split-files requires --group-dataset-id so that group order in the DB is known.")
                cat_index_map = {cat: i + 1 for i, cat in enumerate(all_categories)}
                for grp in unique_groups:
                    pvals_g, padj_g, lfc_g, ave_g1_g, ave_g2_g = all_results[grp]
                    grp_idx = cat_index_map.get(grp)
                    if grp_idx is None:
                        ErrorJSON(f"Group {grp!r} was not found in list_cat_json for annotation id {args.group_dataset_id}. Cannot determine output file index for --split-files.")
                    out_tsv = os.path.join(output_dir, f"cat_{grp_idx}.tsv")
                    write_tsv(out_tsv, ens_ids, gene_names, [(None, lfc_g, pvals_g, padj_g, ave_g1_g, ave_g2_g)])
                    split_tsv_paths.append(out_tsv)
            else:
                tsv_rows = [(grp, all_results[grp][2], all_results[grp][0], all_results[grp][1], all_results[grp][3], all_results[grp][4]) for grp in unique_groups]  # (grp, lfc, pvals, fdr, ave_g1, ave_g2)
                out_tsv = os.path.join(output_dir, "output.tsv")
                tsv_path = write_tsv(out_tsv, ens_ids, gene_names, tsv_rows)

        # Volcano plot — not supported in FindAllMarkers mode (no single pairwise comparison to plot)
        if write_volcano_output:
            data_warnings.append("--write-volcano is not supported in FindAllMarkers mode and was skipped.")

        # Metadata output
        if write_metadata_output:
            metadata_data, metadata_headers = build_metadata_matrix_all_markers(ens_ids, gene_names, unique_groups, all_results)
            metadata_path = write_metadata_dataset(str(input_path), args.write_metadata, metadata_data, metadata_headers)

        # JSON metadata
        result = build_result_json(args.write_metadata if write_metadata_output else None, n_genes * len(unique_groups))
        result["mode"] = "FindAllMarkers"
        result["number_of_groups"] = len(unique_groups)
        result["group_sizes"] = group_sizes
        result["number_of_genes_tested"] = n_genes_tested
        result["number_of_genes_with_fdr_le_5pct"] = n_genes_fdr_le_5pct
        result["overlapping_cells"] = 0

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

        if method == "deseq2":
            out_pvals, out_fdr, out_lfc, ave_g1, ave_g2 = run_deseq2(data_matrix, cell_group, batch_data, batch_names, gene_names)
        else:
            out_pvals, out_fdr, out_lfc, ave_g1, ave_g2 = run_scanpy_method(data_matrix, cell_group, gene_names, method, is_count=is_count_table)

        data_out = np.vstack([out_lfc, out_pvals, out_fdr, ave_g1, ave_g2])  # (5, n_genes)

        # TSV — no extra "group" column
        if write_tsv_output:
            out_tsv = os.path.join(output_dir, "output.tsv")
            tsv_path = write_tsv(out_tsv, ens_ids, gene_names, [(None, out_lfc, out_pvals, out_fdr, ave_g1, ave_g2)])

        # VOLCANO plot - write as JSON
        if write_volcano_output:
            volcano_path = write_volcano_json(out_lfc, out_pvals, gene_names, ens_ids, output_dir)

        # Metadata output
        if write_metadata_output:
            metadata_data, metadata_headers = build_metadata_matrix_single(ens_ids, gene_names, out_lfc, out_pvals, out_fdr, ave_g1, ave_g2)
            metadata_path = write_metadata_dataset(str(input_path), args.write_metadata, metadata_data, metadata_headers)

        # JSON for output JSON file
        result = build_result_json(args.write_metadata if write_metadata_output else None, n_genes)
        result["mode"] = "Single marker"
        result["group_name"] = group1
        result["group2_name"] = None
        result["group_size"] = group1_size
        result["group2_size"] = group2_size
        result["number_of_genes_tested"] = _count_tested(out_pvals)
        result["number_of_genes_with_fdr_le_5pct"] = _count_significant(out_fdr)
        result["overlapping_cells"] = 0

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

        if method == "deseq2":
            out_pvals, out_fdr, out_lfc, ave_g1, ave_g2 = run_deseq2(data_matrix_sub, cell_group_sub, batch_data_sub, batch_names, gene_names)
        else:
            out_pvals, out_fdr, out_lfc, ave_g1, ave_g2 = run_scanpy_method(data_matrix_sub, cell_group_sub, gene_names, method, is_count=is_count_table)

        data_out = np.vstack([out_lfc, out_pvals, out_fdr, ave_g1, ave_g2])  # (5, n_genes)

        if write_tsv_output:
            out_tsv = os.path.join(output_dir, "output.tsv")
            tsv_path = write_tsv(out_tsv, ens_ids, gene_names, [(None, out_lfc, out_pvals, out_fdr, ave_g1, ave_g2)])

        if write_volcano_output:
            volcano_path = write_volcano_json(out_lfc, out_pvals, gene_names, ens_ids, output_dir)

        if write_metadata_output:
            metadata_data, metadata_headers = build_metadata_matrix_single(ens_ids, gene_names, out_lfc, out_pvals, out_fdr, ave_g1, ave_g2)
            metadata_path = write_metadata_dataset(str(input_path), args.write_metadata, metadata_data, metadata_headers)

        result = build_result_json(args.write_metadata if write_metadata_output else None, n_genes)
        result["mode"] = "Two-group DE"
        result["group_name"] = group1
        result["group2_name"] = group2
        result["group_size"] = group1_size
        result["group2_size"] = group2_size
        result["number_of_genes_tested"] = _count_tested(out_pvals)
        result["number_of_genes_with_fdr_le_5pct"] = _count_significant(out_fdr)
        result["overlapping_cells"] = len(overlap)

    # JSON output
    result["list_cats_json"] = list_cats_json_1
    if list_cats_json_2 is not None:
        result["list_cats_json_2"] = list_cats_json_2
    if tsv_path is not None:
        result["tsv_path"] = tsv_path
    if split_tsv_paths:
        result["tsv_paths"] = split_tsv_paths
    if volcano_path is not None:
        result["volcano_path"] = volcano_path
    if data_warnings:
        result["warnings"] = data_warnings

    if args.o:
        out_json = os.path.join(output_dir, "output.json")
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, allow_nan=True)
    else:
        print(json.dumps(result, ensure_ascii=False, allow_nan=True))

## Helpers
def _null(s: str | None) -> bool:
    """Returns True if a string argument represents an absent/null value."""
    return s is None or s.strip().lower() in ("", "null", "na", "none")

def _parse_positive_int(value: str) -> int:
    """argparse type: positive integer (> 0)."""
    try:
        parsed = int(value)
    except ValueError:
        ErrorJSON(f"Expected a positive integer, got {value!r}")
    if parsed <= 0:
        ErrorJSON(f"Expected a positive integer, got {value!r}")
    return parsed

def _parse_bool_string(value: str, parameter_name: str) -> bool:
    """Parses a string boolean argument."""
    raw_value = value.strip().lower()
    if raw_value == "true":
        return True
    if raw_value == "false":
        return False
    ErrorJSON(f"{parameter_name} should be 'true' or 'false'")

def get_env(name: str, required: bool = False, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if required and not value:
        ErrorJSON(f"Missing required environment variable: {name}")
    return value

def parse_host_string(s: str):
    if not s:
        ErrorJSON("Missing dburl")
    if "://" not in s:
        s = "postgresql://" + s
    u = urlparse(s)

    if not u.hostname:
        ErrorJSON("Missing host")
    host = u.hostname

    port = None
    if u.netloc:
        hostport = u.netloc.rsplit("@", 1)[-1]
        parts = hostport.split(":")
        if len(parts) == 2:
            if parts[1].isdigit():
                port = int(parts[1])
            else:
                ErrorJSON(f"Invalid port: {parts[1]!r}")
    if port is None:
        port = DEFAULT_DB_PORT

    dbname = u.path.lstrip("/")
    if not dbname:
        ErrorJSON("Missing dbname")

    return host, port, dbname

## Database stuff
class DBManager:
    def __init__(self, *, dbname: str, user: str, password: str, host: str, port: int, connect_timeout: int = DEFAULT_CONNECT_TIMEOUT, sslmode: Optional[str] = None) -> None:
        self._base_conn_kwargs: dict[str, Any] = {
            "dbname": dbname,
            "user": user,
            "password": password,
            "port": port,
            "connect_timeout": connect_timeout,
            "host": host,
        }
        if sslmode is not None:
            self._base_conn_kwargs["sslmode"] = sslmode
        self._conn: PGConnection | None = None

    def connect(self) -> None:
        if self._conn is not None:
            try:
                self._conn.cursor().execute("SELECT 1")
                return
            except (psycopg2.OperationalError, psycopg2.InterfaceError):
                self._conn = None

        try:
            self._conn = psycopg2.connect(**self._base_conn_kwargs)
        except Exception as e:
            ErrorJSON(f"Failed to connect to PostgreSQL: {e}")

    def disconnect(self) -> None:
        if self._conn is None:
            return
        try:
            self._conn.close()
        finally:
            self._conn = None

    @contextmanager
    def _cursor(self):
        if self._conn is None or self._conn.closed:
            ErrorJSON("Not connected. Call connect() first.")
        cur = self._conn.cursor(cursor_factory=RealDictCursor)
        try:
            yield cur
        finally:
            cur.close()

    def get_annotation_col_attr(self, annotation_id: int) -> str:
        """
        Fetch the loom dataset path for a given annotation id.

        NOTE: This query assumes a column named 'name' in the 'annots' table
        that stores the full HDF5 path to the annotation column in the loom file
        (e.g. '/col_attrs/CellType'). Adjust the column name to match your DB schema.
        """
        sql = "SELECT name FROM annots WHERE id = %s"
        with self._cursor() as cur:
            cur.execute(sql, (annotation_id,))
            rows = cur.fetchall()

        if not rows:
            ErrorJSON(f"No annotation found in annots table with id = {annotation_id}")
        if len(rows) > 1:
            ErrorJSON(f"Multiple rows found in annots table with id = {annotation_id}")

        path = rows[0].get("name")
        if path is None or str(path).strip() == "":
            ErrorJSON(f"name is empty or NULL for annotation id = {annotation_id}")

        return str(path).strip()

    def get_list_cat_json(self, annotation_id: int) -> list[str]:
        sql = "SELECT list_cat_json FROM annots WHERE id = %s"
        with self._cursor() as cur:
            cur.execute(sql, (annotation_id,))
            rows = cur.fetchall()

        if not rows:
            ErrorJSON(f"No list_cat_json found in annots table with id = {annotation_id}")
        if len(rows) > 1:
            ErrorJSON(f"Too many list_cat_json rows found in annots table with id = {annotation_id}")

        raw_json = rows[0].get("list_cat_json")
        if raw_json is None:
            ErrorJSON(f"No list_cat_json found in annots table with id = {annotation_id}")

        try:
            categories = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            ErrorJSON(f"Invalid JSON in annots.list_cat_json for id = {annotation_id}: {exc}")

        if not isinstance(categories, list):
            ErrorJSON(f"annots.list_cat_json should contain a JSON array for id = {annotation_id}")

        return [str(cat) if cat is not None else "" for cat in categories]

def _resolve_group_name_from_annotation_id(annotation_id: int, group_value: str | None, db: DBManager) -> str | None:
    """Resolve a numeric group/category id to its display name using annots.list_cat_json."""
    if _null(group_value):
        return None

    raw_group = group_value.strip()
    if not raw_group.isdigit():
        return raw_group

    group_id = int(raw_group)
    if group_id <= 0:
        ErrorJSON(f"--group value should be a positive integer when used with --group-dataset-id. You entered {group_value!r}")

    categories = db.get_list_cat_json(annotation_id)
    if group_id > len(categories):
        ErrorJSON(f"Group id {group_id} is out of range for annotation id = {annotation_id}. Available ids: 1..{len(categories)}")

    group_name = categories[group_id - 1]
    if str(group_name).strip() == "":
        ErrorJSON(f"Group id {group_id} resolved to an empty group name for annotation id = {annotation_id}")
    return str(group_name)

## Help text
custom_help = """
Differential Expression (Python / Scanpy)

Execution modes
───────────────
  FindAllMarkers  Omit --group (and --group-2 / --group-dataset-id-2 / --group-dataset-2).
                  Every group in the annotation column is tested vs. all other cells.
                  Output TSV contains an extra leading "group" column.

  Single marker   Provide --group only (omit --group-2 and second-dataset arguments).
                  Runs group vs. all other cells.

  Two-group DE    Provide --group and --group-2.
                  Standard pairwise DE; overlapping cells are removed from group 2.

Dataset / annotation resolution (two mutually exclusive styles — do not mix):
───────────────────────────────────────────────────────────────────────────────
  DB-based (--group-dataset-id / --group-dataset-id-2):
    The loom column path and category list are fetched from the database.
    Requires --dburl. Group labels can be provided as numeric category ids.
    If --group-dataset or --group-dataset-2 are also set, they must match
    the DB-resolved path or an error is raised (a warning is emitted if they match).

  Path-based (--group-dataset / --group-dataset-2):
    Loom column paths are provided directly. No DB connection needed.

Options:
  -f %s                      Path to the input .loom file.
  -o %s                      Output folder (default: same directory as -f).
  --method %s                DE method to use. One of:
                               wilcoxon | t_test | t_test_overestim_var | deseq2
  --input-dataset %s         Loom path to the expression matrix (e.g. /layers/norm_data).
  --batch %s                 Comma-separated loom paths for batch/covariate columns, or "null".
                               Only used by deseq2.

  -- DB-based dataset resolution --
  --group-dataset-id %s      Annotation id in the DB annots table for group 1.
                               Fetches the loom column path and category names from DB.
                               Requires --dburl. Group labels may be numeric category ids.
  --group-dataset-id-2 %s    Annotation id in the DB annots table for group 2.
                               Required when group 2 uses a different annotation than group 1.
                               Requires --group-dataset-id to also be set.
  --dburl %s                 PostgreSQL DB URL (format HOST:PORT/DB). Required when
                               --group-dataset-id or --group-dataset-id-2 are set.
                               Credentials read from POSTGRES_USER / POSTGRES_PASSWORD env vars.

  -- Path-based dataset resolution --
  --group-dataset %s         Loom path to the obs-level annotation column for group 1.
                               Required when --group-dataset-id is not set.
                               If set together with --group-dataset-id, must match the DB path
                               (a warning is emitted; the DB path takes precedence).
  --group-dataset-2 %s       Loom path to the obs-level annotation column for group 2.
                               Defaults to --group-dataset when omitted.
                               Cannot be combined with --group-dataset-id.

  -- Group labels --
  --group %s                 Value in the group-1 annotation that labels group-1 cells.
                               If omitted, runs FindAllMarkers across all groups.
                               When --group-dataset-id is set, a positive integer is resolved
                               to its category name via annots.list_cat_json.
  --group-2 %s               Value in the group-2 annotation that labels group-2 cells.
                               Required for two-group DE.
                               When --group-dataset-id-2 (or --group-dataset-id) is set,
                               a positive integer is resolved via annots.list_cat_json.

  -- Flags --
  --is-count %s              Whether the matrix contains raw counts: true | false (default: false).
                               Required true for deseq2.
  --write-tsv                Write output.tsv (requires -o).
  --write-volcano            Write output.plot.json volcano plot (requires -o).
  --write-metadata %s        Write the TSV-like metadata dataset in the loom at the given /attrs/ path
                               (e.g. --write-metadata /attrs/_de_2_wilcoxon). Default: disabled.
  --split-files              FindAllMarkers + --write-tsv only: write one TSV per group
                               (cat_1.tsv, cat_2.tsv, ...) instead of a single output.tsv.
                               Requires --group-dataset-id (the file index is the 1-based
                               position of the group in annots.list_cat_json).
                               Does not affect --write-metadata (the single metadata dataset
                               with its extra "group" column is always written as usual).
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

    # DB-based dataset resolution
    parser.add_argument("--group-dataset-id",    metavar="Annotation id (group 1)",         type=_parse_positive_int,    required=False, dest="group_dataset_id",  default=None)
    parser.add_argument("--group-dataset-id-2",  metavar="Annotation id (group 2)",         type=_parse_positive_int,    required=False, dest="group_dataset_id2", default=None)
    parser.add_argument("--dburl",               metavar="PostgreSQL DB URL",               default=None,                required=False)

    # Path-based dataset resolution
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

    args = parser.parse_args()

    run(args)

if __name__ == "__main__":
    main()