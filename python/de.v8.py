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
    headers   = list(DE_METADATA_HEADERS_ALL_MARKERS) if has_group else list(DE_METADATA_HEADERS_PAIRWISE)
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

def _unique_categories_sorted(groups: np.ndarray) -> list[str]:
    """
    Unique category labels from a 1-D loom metadata vector, sorted alphanumerically
    (same ordering as Ruby Array#sort on strings).
    """
    labels = [str(x) for x in groups.tolist()]
    return sorted(set(labels))

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

def build_metadata_matrix_single(ens_ids: list[str], gene_names: list[str], lfc: np.ndarray, pvals: np.ndarray, fdr: np.ndarray, ave_g1: np.ndarray, ave_g2: np.ndarray) -> tuple[np.ndarray, list[str]]:
    headers = list(DE_METADATA_HEADERS_PAIRWISE)
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
    headers = list(DE_METADATA_HEADERS_ALL_MARKERS)
    rows = []
    for grp in unique_groups:
        pvals, padj, lfc, ave_g1, ave_g2 = all_results[grp]
        pvals_for_sort = np.where(np.isnan(pvals), np.inf, pvals)
        sort_idx = np.lexsort((-lfc, pvals_for_sort))
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
    if split_files and _null(args.group_dataset):
        ErrorJSON("--split-files requires --group-dataset (loom path to the primary annotation column).")
    if split_files and not _null(args.group1):
        ErrorJSON("--split-files is only valid in FindAllMarkers mode (omit --group).")

    if method in COUNT_ONLY_METHODS and not is_count_table:
        ErrorJSON(f"Data should be a count table in order to run {method_arg}")

    if write_metadata_output:
        if not args.write_metadata.startswith("/attrs/"):
            ErrorJSON("--write-metadata path should refer to the /attrs/ path (e.g. /attrs/_de_2_wilcoxon)")

    data_warnings: list = []

    if not is_count_table:
        data_warnings.append("Data is assumed to be logged. If it's not the case, logFC could be wrong.")

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
        result = build_result_json(
            args.write_metadata if write_metadata_output else None,
            n_genes * len(unique_groups),
            unique_groups,
        )
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
  -o                         Output folder (default: same directory as -f).
  --method                   DE method: wilcoxon | t_test | t_test_overestim_var | deseq2
  --input-dataset            Loom path to the expression matrix (e.g. /layers/norm_data).
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
  --split-files              FindAllMarkers + --write-tsv: one cat_N.tsv per category order
                             (N is 1-based index in sorted unique categories from the loom).
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

    args = parser.parse_args()

    run(args)

if __name__ == "__main__":
    main()