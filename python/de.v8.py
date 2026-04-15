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
    "wilcoxon",              # Wilcoxon rank-sum test           (scanpy)
    "t-test",                # Student's t-test, pooled var     (scanpy)
    "t-test_overestim_var",  # Student's t-test, per-gene var   (scanpy)
    "deseq2",                # Negative-binomial GLM            (pydeseq2, counts only)
]
COUNT_ONLY_METHODS: Final[set] = {"deseq2"}
BATCH_SUPPORT:      Final[set] = {"deseq2"}
DE_HEADERS: Final[list] = ["log Fold-Change", "p-value", "FDR", "Avg. Exp. Group 1", "Avg. Exp. Group 2"]
DE_NB_COLS: Final[int] = 5  # Number of DE output columns stored per gene

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
        f.create_dataset(dataset_path, data=data, compression="gzip", compression_opts=4)

## Output helpers
def write_tsv(out_path: str, ens_ids: list[str], gene_names: list[str], rows: list[tuple]) -> None:
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
            for i in range(n_genes):
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

def write_volcano_json(logfc: np.ndarray, pvals: np.ndarray, gene_names: list[str], ens_ids: list[str], output_dir: str) -> None:
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
        fig.write_json(os.path.join(output_dir, "output.plot.json"))
    except Exception as exc:
        warnings.warn(f"Could not generate volcano plot: {exc}")

def build_result_json(output_dataset: str, n_genes: int, unique_groups:  list[str] | None = None) -> dict:
    """
    Build the output.json metadata dict.
    If unique_groups is provided (Mode A), produces multi-group metadata with flat headers labelled by group name.
    Otherwise (Modes B/C), produces single-comparison metadata.
    """
    if unique_groups is not None:
        n_groups         = len(unique_groups)
        all_headers_flat = [f"{h} ({grp})" for grp in unique_groups for h in DE_HEADERS]
        return {
            "metadata": [
                {
                    "name":      output_dataset,
                    "on":        "GENE",
                    "type":      "NUMERIC",
                    "nber_cols": DE_NB_COLS * n_groups,
                    "nber_rows": n_genes,
                    "headers":   all_headers_flat,
                    "groups":    unique_groups,
                }
            ]
        }
    return {
        "metadata": [
            {
                "name":      output_dataset,
                "on":        "GENE",
                "type":      "NUMERIC",
                "nber_cols": DE_NB_COLS,
                "nber_rows": n_genes,
                "headers":   DE_HEADERS,
            }
        ]
    }

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

    # Averages computed on adata.X — already normalized+logged if is_count=True, or the original scale if is_count=False. Either way, consistent with the LFC.
    # np.asarray + ravel guards against adata.X being sparse after preprocessing (sparse .mean() returns a matrix, not a 1-D array).
    mask_1 = group_labels == 1
    mask_2 = group_labels == 2
    ave_g1 = np.asarray(adata.X[mask_1, :].mean(axis=0)).ravel()  # shape (n_genes,)
    ave_g2 = np.asarray(adata.X[mask_2, :].mean(axis=0)).ravel()

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
        ave_g1 = np.asarray(adata.X[mask_grp, :].mean(axis=0)).ravel()   # shape (n_genes,)
        ave_g2 = np.asarray(adata.X[mask_rest, :].mean(axis=0)).ravel()

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

    # Use DESeq2's own size-factor-normalized counts (stored by pydeseq2 after dds.deseq2()).
    # This is the natural scale for DESeq2 averages — comparable to its LFC.
    normed = dds.layers["normed_counts"]  # (n_cells, n_genes), numpy array
    mask_1 = np.array(meta["group"] == "1")
    mask_2 = np.array(meta["group"] == "2")
    ave_g1 = normed[mask_1, :].mean(axis=0)
    ave_g2 = normed[mask_2, :].mean(axis=0)

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
        print(f"  DESeq2: running group {grp!r} vs. rest ...")
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
    method = args.method.strip()

    is_count_raw = args.is_count.strip().lower()
    if is_count_raw == "true":
        is_count_table = True
    elif is_count_raw == "false":
        is_count_table = False
    else:
        ErrorJSON("--is-count should be 'true' or 'false'")

    if method in COUNT_ONLY_METHODS and not is_count_table:
        ErrorJSON(f"Data should be a count table in order to run {method}")

    # Determine the execution mode (3 modes) from the supplied group arguments
    group1           = None if _null(args.group1)           else args.group1.strip()
    group2           = None if _null(args.group2)           else args.group2.strip()
    group_dataset_2  = None if _null(args.group_dataset_2)  else args.group_dataset_2.strip()
    if group1 is None:
        # Mode A — FindAllMarkers
        if group2 is not None or group_dataset_2 is not None:
            ErrorJSON("When --group1 is omitted (FindAllMarkers mode), --group2 and --group-dataset-2 must also be omitted.")
        mode = "all_markers"
    elif group2 is None and group_dataset_2 is None:
        # Mode B — Single-group marker
        mode = "single_marker"
    else:
        # Mode C — Standard two-group DE: require both group2 and group_dataset_2
        if group2 is None:
            ErrorJSON("When --group-dataset-2 is provided, --group2 must also be provided.")
        if group_dataset_2 is None:
            # group_dataset_2 defaults to group_dataset when only group2 is given
            group_dataset_2 = args.group_dataset.strip()
        mode = "two_group"

    # Init warning list
    data_warnings: list = []

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
        groups_raw_1 = loom.read_vector(args.group_dataset)
        if groups_raw_1 is None:
            ErrorJSON(f"Group dataset {args.group_dataset!r} does not exist in the Loom file")
        groups_1 = LoomHandler._decode_str_array(groups_raw_1)

        # Group 2 metadata (only for Mode C, may come from a different column)
        if mode == "two_group":
            if group_dataset_2 != args.group_dataset:
                groups_raw_2 = loom.read_vector(group_dataset_2)
                if groups_raw_2 is None:
                    ErrorJSON(f"Group dataset 2 {group_dataset_2!r} does not exist in the Loom file")
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
                print(f"{len(batch_paths)} covariate(s) detected: {', '.join(batch_names)}")
        if batch_data is not None and method not in BATCH_SUPPORT: # Warn if batch was requested for a method that does not support it
            data_warnings.append(f"Method '{method}' does not support covariates — batch column(s) ignored. Covariate-aware DE is currently only supported by: {', '.join(sorted(BATCH_SUPPORT))}.")
            batch_data  = None
            batch_names = []

    # Loom file is now closed

    # ------------------------------------------------------------------ #
    # Mode A — FindAllMarkers                                             #
    # ------------------------------------------------------------------ #
    if mode == "all_markers":
        unique_groups = sorted(set(groups_1))
        if len(unique_groups) < 2:
            ErrorJSON("FindAllMarkers mode requires at least 2 distinct groups in --group-dataset.")

        # Validate minimum cell count per group
        for grp in unique_groups:
            cnt = (groups_1 == grp).sum()
            if cnt < 3:
                ErrorJSON(f"Group {grp!r} contains only {cnt} cell(s); at least 3 are required.")

        print(f"FindAllMarkers mode: {len(unique_groups)} groups detected — {', '.join(unique_groups)}")
        print(f"Running DE method: {method} on {n_genes} genes ...")

        if method == "deseq2":
            all_results = run_deseq2_all_markers(data_matrix, groups_1, batch_data, batch_names, gene_names)
        else:
            all_results = run_scanpy_all_markers(data_matrix, groups_1, gene_names, method, is_count=is_count_table)

        # Stack results as (5 * n_groups, n_genes) in DE_HEADERS order [lfc, pvals, fdr, ave_g1, ave_g2] per group.
        blocks = []
        for grp in unique_groups:
            pvals, padj, lfc, ave_g1, ave_g2 = all_results[grp]
            blocks.append(np.vstack([lfc, pvals, padj, ave_g1, ave_g2]))
        data_out = np.vstack(blocks)  # shape: (5 * n_groups, n_genes)

        # Loom output
        with LoomHandler(str(input_path), mode="r+") as loom:
            loom.write_dataset(args.output_dataset, data_out)
            print(f"DE results written to: {args.output_dataset}")

        # TSV output — Extra leading column "group" to identify which group each row belongs to.
        tsv_rows = [(grp, *all_results[grp][2:3], *all_results[grp][0:2], *all_results[grp][3:5]) for grp in unique_groups]  # (grp, lfc, pvals, fdr, ave_g1, ave_g2)
        out_tsv = os.path.join(output_dir, "output.tsv")
        write_tsv(out_tsv, ens_ids, gene_names, tsv_rows)
        print(f"TSV written to: {out_tsv}")

        # Volcano plot — use the first group as a representative
        first_grp = unique_groups[0]
        write_volcano_json(all_results[first_grp][2], all_results[first_grp][0], gene_names, ens_ids, output_dir)

        # JSON metadata
        result = build_result_json(args.output_dataset, n_genes, unique_groups=unique_groups)

    # ------------------------------------------------------------------ #
    # Mode B — Single-group marker (group1 vs. all other cells)           #
    # ------------------------------------------------------------------ #
    elif mode == "single_marker":
        candidates_1 = np.where(groups_1 == group1)[0]
        if len(candidates_1) == 0:
            ErrorJSON(f"Group 1 label {group1!r} was not found in {args.group_dataset!r}.")

        cell_group = np.zeros(n_cells, dtype=int)
        cell_group[candidates_1] = 1
        cell_group[cell_group == 0] = 2  # all other cells become group 2

        if (cell_group == 1).sum() < 3:
            ErrorJSON(f"Group 1 ({group1!r}) contains fewer than 3 cells.")
        if (cell_group == 2).sum() < 3:
            ErrorJSON("Group 2 (all other cells) contains fewer than 3 cells.")

        print(f"Marker gene mode: {group1!r} vs. all other cells.")
        print(f"Group 1: {(cell_group == 1).sum()} cells | Group 2: {(cell_group == 2).sum()} cells")
        print(f"Running DE method: {method} on {n_genes} genes ...")

        if method == "deseq2":
            out_pvals, out_fdr, out_lfc, ave_g1, ave_g2 = run_deseq2(data_matrix, cell_group, batch_data, batch_names, gene_names)
        else:
            out_pvals, out_fdr, out_lfc, ave_g1, ave_g2 = run_scanpy_method(data_matrix, cell_group, gene_names, method, is_count=is_count_table)

        data_out = np.vstack([out_lfc, out_pvals, out_fdr, ave_g1, ave_g2])  # (5, n_genes)

        # LOOM - write dataset in Loom
        with LoomHandler(str(input_path), mode="r+") as loom:
            loom.write_dataset(args.output_dataset, data_out)
            print(f"DE results written to: {args.output_dataset}")

        # TSV — no extra "group" column
        out_tsv = os.path.join(output_dir, "output.tsv")
        write_tsv(out_tsv, ens_ids, gene_names, [(None, out_lfc, out_pvals, out_fdr, ave_g1, ave_g2)])
        print(f"TSV written to: {out_tsv}")

        # VOLCANO plot - write as JSON
        write_volcano_json(out_lfc, out_pvals, gene_names, ens_ids, output_dir)

        # JSON for output JSON file
        result = build_result_json(args.output_dataset, n_genes)

    # ------------------------------------------------------------------ #
    # Mode C — Standard two-group DE                                      #
    # ------------------------------------------------------------------ #
    else:  # mode == "two_group"
        candidates_1 = np.where(groups_1 == group1)[0]
        candidates_2 = np.where(groups_2 == group2)[0]

        if len(candidates_1) == 0:
            ErrorJSON(f"Group 1 label {group1!r} was not found in {args.group_dataset!r}.")
        if len(candidates_2) == 0:
            ErrorJSON(f"Group 2 label {group2!r} was not found in {group_dataset_2!r}.")

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

        if (cell_group_sub == 1).sum() < 3:
            ErrorJSON(f"Group 1 ({group1!r}) contains fewer than 3 cells.")
        if (cell_group_sub == 2).sum() < 3:
            ErrorJSON(f"Group 2 ({group2!r}) contains fewer than 3 cells after removing overlapping cells.")

        print(f"Two-group DE: {group1!r} vs {group2!r} — discarding unassigned cells.")
        print(f"Group 1: {(cell_group_sub == 1).sum()} cells | Group 2: {(cell_group_sub == 2).sum()} cells")
        print(f"Running DE method: {method} on {n_genes} genes ...")

        if method == "deseq2":
            out_pvals, out_fdr, out_lfc, ave_g1, ave_g2 = run_deseq2(data_matrix_sub, cell_group_sub, batch_data_sub, batch_names, gene_names)
        else:
            out_pvals, out_fdr, out_lfc, ave_g1, ave_g2 = run_scanpy_method(data_matrix_sub, cell_group_sub, gene_names, method, is_count=is_count_table)

        data_out = np.vstack([out_lfc, out_pvals, out_fdr, ave_g1, ave_g2])  # (5, n_genes)

        with LoomHandler(str(input_path), mode="r+") as loom:
            loom.write_dataset(args.output_dataset, data_out)
            print(f"DE results written to: {args.output_dataset}")

        out_tsv = os.path.join(output_dir, "output.tsv")
        write_tsv(out_tsv, ens_ids, gene_names, [(None, out_lfc, out_pvals, out_fdr, ave_g1, ave_g2)])
        print(f"TSV written to: {out_tsv}")

        write_volcano_json(out_lfc, out_pvals, gene_names, ens_ids, output_dir)

        result = build_result_json(args.output_dataset, n_genes)

    # JSON output
    if data_warnings:
        result["warnings"] = data_warnings

    out_json = os.path.join(output_dir, "output.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, allow_nan=True)
    print(f"JSON written to: {out_json}")

## Argument helpers
def _null(s: str | None) -> bool:
    """Returns True if a string argument represents an absent/null value."""
    return s is None or s.strip().lower() in ("", "null", "na", "none")

## Help text
custom_help = """
Differential Expression (Python / Scanpy)

Execution modes
───────────────
  FindAllMarkers  Omit --group1 (and --group2 / --group-dataset-2).
                  Every group in --group-dataset is tested vs. all other cells.
                  Output TSV contains an extra leading "group" column.

  Single marker   Provide --group1 only (omit --group2 / --group-dataset-2).
                  Runs group1 vs. all other cells.

  Two-group DE    Provide all four group arguments.
                  Standard pairwise DE; overlapping cells are removed from group 2.

Options:
  -f %s                  Path to the input .loom file.
  -o %s                  Output folder (default: same directory as -f).
  --method %s            DE method to use. One of:
                           wilcoxon | t-test | t-test_overestim_var | deseq2
  --input-dataset %s     Loom path to the expression matrix to use (e.g. /layers/norm_data).
  --output-dataset %s    Loom path where DE results will be written (e.g. /row_attrs/_de_1_wilcoxon).
  --batch %s             Comma-separated loom paths for batch/covariate columns, or "null".
                           Only used by deseq2.
  --group-dataset %s     Loom path to the obs-level metadata column (always required).
  --group-dataset-2 %s   Loom path to the obs-level metadata defining group 2 cells (two-group DE only).
                           If omitted, defaults to --group-dataset.
  --group1 %s            Value in --group-dataset that labels group-1 cells.
                           If omitted, runs FindAllMarkers across all groups.
  --group2 %s            Value in --group-dataset-2 (or --group-dataset) for group-2 cells.
                           Required when --group-dataset-2 is provided.
  --is-count %s          Whether the matrix contains raw counts: true | false (default: false).
                           Required true for deseq2.
  --help                 Show this help message and exit.
"""

## Entry point
def main() -> None:
    if "--help" in sys.argv:
        print(custom_help)
        sys.exit(0)

    parser = argparse.ArgumentParser(description="Differential Expression Script", add_help=False)
    parser.add_argument("-f",                metavar="Input loom file",                                                  required=True)
    parser.add_argument("-o",                metavar="Output folder",                                                    required=False)
    parser.add_argument("--method",          metavar="DE method",                       choices=AVAILABLE_METHODS,       required=True)
    parser.add_argument("--input-dataset",   metavar="Input matrix dataset",                                             required=True,  dest="input_dataset")
    parser.add_argument("--output-dataset",  metavar="Output DE dataset",                                                required=True,  dest="output_dataset")
    parser.add_argument("--batch",           metavar="Batch/covariate datasets",        default="null",                  required=False)
    parser.add_argument("--group-dataset",   metavar="Group metadata path",                                              required=True,  dest="group_dataset")
    parser.add_argument("--group-dataset-2", metavar="Group 2 metadata path",           default=None,                    required=False, dest="group_dataset_2")
    parser.add_argument("--group1",          metavar="Group 1 label (null = all)",      default="null",                  required=False)
    parser.add_argument("--group2",          metavar="Group 2 label",                   default="null",                  required=False)
    parser.add_argument("--is-count",        metavar="Is count table [true|false]",     default="false",                 required=False, dest="is_count")

    args = parser.parse_args()

    run(args)

if __name__ == "__main__":
    main()
