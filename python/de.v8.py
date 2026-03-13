from __future__ import annotations

## Standard library
import argparse        # Argument parsing
import json            # For writing output files
import os              # For path/directory operations
import sys             # For ErrorJSON exit
import warnings        # For non-fatal warnings
from pathlib import Path  # For path manipulation
from typing import Final

## Scientific stack
import h5py            # For reading/writing loom (HDF5) files
import numpy as np     # For array calculations
import scipy.sparse as sp  # For sparse matrix handling

## DE / stats (scanpy, anndata, statsmodels, pydeseq2 imported lazily where needed)
# scanpy, anndata  → run_scanpy_method()
# pydeseq2         → run_deseq2()
# statsmodels      → bh_fdr()

## Constants
AVAILABLE_METHODS: Final[list] = [
    "wilcoxon",              # Wilcoxon rank-sum test           (scanpy)
    "t-test",                # Student's t-test, pooled var     (scanpy)
    "t-test_overestim_var",  # Student's t-test, per-gene var   (scanpy)
    "logreg",                # Logistic regression              (scanpy)
    "deseq2",                # Negative-binomial GLM            (pydeseq2, counts only)
]
COUNT_ONLY_METHODS: Final[set] = {"deseq2"}
BATCH_SUPPORT:      Final[set] = {"deseq2"}

DE_HEADERS: Final[list] = [
    "log Fold-Change",
    "p-value",
    "FDR",
    "Avg. Exp. Group 1",
    "Avg. Exp. Group 2",
]

DE_NB_COLS: Final[int] = 5  # Number of DE output columns stored per gene


## Error handling

class ErrorJSON(Exception):
    """
    Prints a JSON error payload to stdout and exits immediately.
    Mirrors error.json() from de.R.
    """
    def __init__(self, message: str):
        super().__init__(message)
        print(json.dumps({"displayed_error": message}, ensure_ascii=False))
        sys.exit(1)


## Loom I/O helpers

def _decode_str_array(arr: np.ndarray) -> np.ndarray:
    """Decode a bytes/object array to a plain Python-str array."""
    if arr.dtype.kind in ("S", "O"):
        return np.array(
            [v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v) for v in arr]
        )
    return arr.astype(str)


def read_loom_vector(loom_path: str, dataset_path: str) -> np.ndarray | None:
    """Read a 1-D dataset from a loom file. Returns None if the path is absent."""
    with h5py.File(loom_path, "r") as f:
        if dataset_path not in f:
            return None
        node = f[dataset_path]
        if not isinstance(node, h5py.Dataset):
            return None
        return node[:]


def read_loom_matrix(loom_path: str, dataset_path: str) -> np.ndarray | None:
    """
    Read an expression matrix from a loom file.
    Returns a dense float64 array of shape (n_genes, n_cells), or None if absent.
    Handles dense datasets and CSR/CSC sparse groups.
    """
    with h5py.File(loom_path, "r") as f:
        if dataset_path not in f:
            return None
        node = f[dataset_path]

        # Sparse group (CSR / CSC)
        if isinstance(node, h5py.Group) and all(
            k in node for k in ("data", "indices", "indptr")
        ):
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


def write_loom_dataset(loom_path: str, dataset_path: str, data: np.ndarray) -> None:
    """Write (or overwrite) a dataset inside an existing loom file."""
    with h5py.File(loom_path, "r+") as f:
        if dataset_path in f:
            del f[dataset_path]
        parent = dataset_path.rsplit("/", 1)[0]
        if parent and parent not in f:
            f.require_group(parent)
        f.create_dataset(dataset_path, data=data, compression="gzip", compression_opts=4)


## Statistical helpers

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction, matching R's p.adjust(method = 'fdr').
    NaN p-values are preserved as NaN in the output.
    """
    from statsmodels.stats.multitest import multipletests

    out   = np.full_like(pvals, np.nan, dtype=float)
    valid = ~np.isnan(pvals)
    if valid.any():
        _, adj, _, _ = multipletests(pvals[valid], method="fdr_bh")
        out[valid] = adj
    return out


def log2_pseudo_mean(x: np.ndarray, is_log2: bool) -> np.ndarray:
    """
    Per-gene log2 pseudo-mean, replicating the FC logic from de.R.
    x shape: (n_genes, n_cells_subset)
    """
    if is_log2:
        # log2(mean(2^x - 1) + 1)  [ExpMean from Seurat]
        return np.log2(np.mean(np.exp2(x) - 1, axis=1) + 1)
    else:
        # sign(m) * log2(|m| + 1)
        m = np.mean(x, axis=1)
        return np.sign(m) * np.log2(np.abs(m) + 1)


## DE method runners

def run_scanpy_method(
    data_matrix:  np.ndarray,   # (n_genes, n_cells) — already subset to G1 + G2 cells
    group_labels: np.ndarray,   # 1-D int array: 1 or 2 per cell
    gene_names:   list[str],
    method:       str,
    n_cores:      int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Wraps scanpy.tl.rank_genes_groups for any native scanpy method.
    Returns (pvals, logFCs) both shape (n_genes,), NaN for genes not returned.
    """
    import anndata as ad
    import scanpy as sc
    import pandas as pd

    n_genes, n_cells = data_matrix.shape

    adata = ad.AnnData(X=data_matrix.T.astype(np.float32))
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = gene_names
    adata.obs["group"] = pd.Categorical(
        [str(g) for g in group_labels], categories=["1", "2"]
    )

    sc.settings.verbosity = 0
    sc.tl.rank_genes_groups(
        adata,
        groupby="group",
        groups=["1"],
        reference="2",
        method=method,
        use_raw=False,
        n_genes=n_genes,
    )

    res          = adata.uns["rank_genes_groups"]
    result_genes = np.array(res["names"]["1"])
    result_pvals = np.array(res["pvals"]["1"])
    result_lfc   = np.array(res["logfoldchanges"]["1"])

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    pvals = np.full(n_genes, np.nan)
    lfcs  = np.full(n_genes, np.nan)
    for rg, rp, rl in zip(result_genes, result_pvals, result_lfc):
        idx = gene_to_idx.get(rg)
        if idx is not None:
            pvals[idx] = rp
            lfcs[idx]  = rl

    return pvals, lfcs


def run_deseq2(
    data_matrix:  np.ndarray,         # (n_genes, n_cells), integer counts
    group_labels: np.ndarray,         # 1-D int array: 1 or 2 per cell
    batch_data:   np.ndarray | None,  # (n_cells, n_covariates) or None
    batch_names:  list[str],
    gene_names:   list[str],
    n_cores:      int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs DESeq2 via pydeseq2 (counts required).
    Returns (pvals, padj, log2FoldChange) all shape (n_genes,), NaN for untested genes.
    """
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds  import DeseqStats
    except ImportError:
        ErrorJSON(
            "pydeseq2 is required for the 'deseq2' method. "
            "Install it with: pip install pydeseq2"
        )

    import pandas as pd

    n_genes, n_cells = data_matrix.shape
    counts   = np.round(data_matrix.T).astype(int)  # cells x genes
    cell_ids = [f"cell_{i}" for i in range(n_cells)]

    meta = pd.DataFrame({"group": [str(g) for g in group_labels]}, index=cell_ids)
    if batch_data is not None:
        for j, bname in enumerate(batch_names):
            meta[bname] = batch_data[:, j].astype(str)

    counts_df      = pd.DataFrame(counts, index=cell_ids, columns=gene_names)
    design_factors = (batch_names + ["group"]) if batch_names else ["group"]

    dds = DeseqDataSet(counts=counts_df, metadata=meta, design_factors=design_factors, quiet=True)
    dds.deseq2()

    stat_res = DeseqStats(dds, contrast=("group", "1", "2"), quiet=True)
    stat_res.summary()
    df = stat_res.results_df

    pvals = np.full(n_genes, np.nan)
    padj  = np.full(n_genes, np.nan)
    lfc   = np.full(n_genes, np.nan)
    for i, gname in enumerate(gene_names):
        if gname in df.index:
            pvals[i] = df.loc[gname, "pvalue"]
            padj[i]  = df.loc[gname, "padj"]
            lfc[i]   = df.loc[gname, "log2FoldChange"]

    return pvals, padj, lfc


## Volcano plot

def write_volcano_json(
    logfc:      np.ndarray,
    pvals:      np.ndarray,
    gene_names: list[str],
    ens_ids:    list[str],
    output_dir: str,
) -> None:
    """Writes a plotly volcano plot to output.plot.json, mirroring de.R."""
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
        fig = go.Figure(go.Scatter(
            x=x, y=y, mode="markers", text=text, hoverinfo="text",
            marker=dict(size=4, opacity=0.7),
        ))
        fig.update_layout(
            title="Volcano plot",
            xaxis_title="log Fold-Change",
            yaxis_title="-log10(p-value)",
        )
        fig.write_json(os.path.join(output_dir, "output.plot.json"))
    except Exception as exc:
        warnings.warn(f"Could not generate volcano plot: {exc}")


## Argument helpers

def _null(s: str | None) -> bool:
    """Returns True if a string argument represents an absent/null value."""
    return s is None or s.strip().lower() in ("", "null", "na", "none")


def positive_int(value: str) -> int:
    try:
        ivalue = int(value)
    except ValueError:
        ErrorJSON(f"--nb-cores must be a positive integer and {value!r} is not an integer")
    if ivalue <= 0:
        ErrorJSON("--nb-cores must be a positive integer")
    return ivalue


## Help text

custom_help = """
Differential Expression Mode (Python / Scanpy)

Options:
  -f %s                  Path to the input .loom file.
  -o %s                  Output folder (default: same directory as -f).
  --method %s            DE method to use. One of:
                           wilcoxon | t-test | t-test_overestim_var | logreg | deseq2
  --input-dataset %s     Loom path to the expression matrix to use (e.g. /layers/norm_data).
  --output-dataset %s    Loom path where DE results will be written (e.g. /row_attrs/_de_1_wilcoxon).
  --batch %s             Comma-separated loom paths for batch/covariate columns, or "null".
                           Only used by deseq2.
  --group-dataset %s     Loom path to the obs-level metadata defining group 1 cells.
  --group-dataset-2 %s   Loom path to the obs-level metadata defining group 2 cells (optional).
                           If omitted, group 2 = all cells not in group 1 (marker genes mode).
  --group1 %s            Value in --group-dataset that labels group-1 cells.
  --group2 %s            Value in --group-dataset-2 (or --group-dataset) for group-2 cells.
                           If omitted, runs group-1 vs. all others (marker gene mode).
  --is-count %s          Whether the matrix contains raw counts: true | false (default: false).
                           Required true for deseq2.
  --nb-cores %i          Number of cores for parallelisable steps (default: 8).
  --help                 Show this help message and exit.
"""


## Core logic

def run(args: argparse.Namespace) -> None:
    """Core DE logic, called after argument validation."""

    # Resolve output directory (mirrors parse_v8.py)
    input_path = Path(args.f).resolve()
    output_dir = Path(args.o).resolve() if args.o else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = str(output_dir)

    # ── Validate method ────────────────────────────────────────────────────
    method = args.method.strip()
    if method not in AVAILABLE_METHODS:
        ErrorJSON(
            f"{method!r} is not a supported method. "
            f"Available: {', '.join(AVAILABLE_METHODS)}"
        )

    # ── Parse scalar arguments ─────────────────────────────────────────────
    is_count_raw = args.is_count.strip().lower()
    if is_count_raw == "true":
        is_count_table = True
    elif is_count_raw == "false":
        is_count_table = False
    else:
        ErrorJSON("--is-count should be 'true' or 'false'")

    nb_cores = args.nb_cores  # already validated as positive_int

    if method in COUNT_ONLY_METHODS and not is_count_table:
        ErrorJSON(f"Data should be a count table in order to run {method}")

    group_1 = args.group1.strip()
    group_2 = args.group2.strip() if not _null(args.group2) else None

    if (
        group_2 is not None
        and (_null(args.group_dataset_2) or args.group_dataset_2 == args.group_dataset)
        and group_1 == group_2
    ):
        ErrorJSON("Cannot compute DE from the same group.")

    data_warnings: list[dict] = []

    # ── Read loom ──────────────────────────────────────────────────────────
    loom_path = str(input_path)

    data_matrix = read_loom_matrix(loom_path, args.input_dataset)
    if data_matrix is None:
        ErrorJSON(f"Input dataset {args.input_dataset!r} does not exist in the Loom file")

    n_genes, n_cells = data_matrix.shape

    # Gene annotations (used for volcano plot and output labeling)
    ens_raw  = read_loom_vector(loom_path, "/row_attrs/Accession")
    gene_raw = read_loom_vector(loom_path, "/row_attrs/Gene")
    ens_ids    = _decode_str_array(ens_raw).tolist()   if ens_raw  is not None else [f"gene_{i}" for i in range(n_genes)]
    gene_names = _decode_str_array(gene_raw).tolist()  if gene_raw is not None else [f"gene_{i}" for i in range(n_genes)]

    # ── Read group metadata ────────────────────────────────────────────────
    groups_raw_1 = read_loom_vector(loom_path, args.group_dataset)
    if groups_raw_1 is None:
        ErrorJSON(f"Group dataset {args.group_dataset!r} does not exist in the Loom file")
    groups_1 = _decode_str_array(groups_raw_1)

    # Group 2 may come from a different metadata column (new feature vs. de.R)
    use_separate_g2_col = (
        not _null(args.group_dataset_2)
        and args.group_dataset_2 != args.group_dataset
    )
    if use_separate_g2_col:
        groups_raw_2 = read_loom_vector(loom_path, args.group_dataset_2)
        if groups_raw_2 is None:
            ErrorJSON(f"Group dataset 2 {args.group_dataset_2!r} does not exist in the Loom file")
        groups_2 = _decode_str_array(groups_raw_2)
    else:
        groups_2 = groups_1  # same column, same semantics

    # ── Read batch / covariate metadata ────────────────────────────────────
    batch_names: list[str] = []
    batch_data:  np.ndarray | None = None

    if not _null(args.batch):
        batch_paths = [p.strip() for p in args.batch.split(",") if p.strip()]
        if batch_paths:
            cols = []
            for bp in batch_paths:
                bvec = read_loom_vector(loom_path, bp)
                if bvec is None:
                    ErrorJSON(f"Batch dataset {bp!r} does not exist in the Loom file")
                cols.append(bvec.reshape(-1))
                leaf = bp.split("/")[-1].lstrip("_")
                batch_names.append(leaf)
            batch_data = np.column_stack(cols)  # (n_cells, n_batch_cols)
            print(f"{len(batch_paths)} covariate(s) detected: {', '.join(batch_names)}")

    # Warn if batch was requested for a method that does not support it
    if batch_data is not None and method not in BATCH_SUPPORT:
        data_warnings.append({
            "name": f"Method '{method}' does not support covariates — batch column(s) ignored.",
            "description": (
                f"Covariate-aware DE is currently only supported by: "
                f"{', '.join(sorted(BATCH_SUPPORT))}."
            ),
        })
        batch_data  = None
        batch_names = []

    # ── Assign cells to groups ─────────────────────────────────────────────
    cell_group = np.zeros(n_cells, dtype=int)  # 0 = unassigned
    cell_group[groups_1 == group_1] = 1

    if group_2 is not None:
        # Explicit two-group comparison
        candidates_1 = np.where(groups_1 == group_1)[0]
        candidates_2 = np.where(groups_2 == group_2)[0]
        overlap = np.intersect1d(candidates_1, candidates_2)
        if len(overlap) > 0:
            data_warnings.append({
                "name": (
                    f"{len(overlap)} cell(s) appear in both group 1 ({group_1!r}) "
                    f"and group 2 ({group_2!r})."
                ),
                "description": (
                    "These cells were removed from group 2 to avoid overlap. "
                    f"Affected indices (first 20): {overlap[:20].tolist()}"
                    + (" ..." if len(overlap) > 20 else "")
                ),
            })
        non_overlap_2 = np.setdiff1d(candidates_2, candidates_1)
        cell_group[non_overlap_2] = 2
        print(f"Two-group DE: {group_1!r} vs {group_2!r} — discarding unassigned cells.")
    else:
        # Marker gene mode: group 1 vs. all remaining cells
        cell_group[cell_group == 0] = 2
        print(f"Marker gene mode: {group_1!r} vs. all other cells.")

    # ── Filter to assigned cells ───────────────────────────────────────────
    keep_idx    = np.where(cell_group > 0)[0]
    data_matrix = data_matrix[:, keep_idx]
    cell_group  = cell_group[keep_idx]
    if batch_data is not None:
        batch_data = batch_data[keep_idx]

    cells_1 = cell_group == 1
    cells_2 = cell_group == 2

    if cells_1.sum() < 3:
        ErrorJSON("Group 1 should contain at least 3 cells")
    if cells_2.sum() < 3:
        ErrorJSON("Group 2 should contain at least 3 cells")

    print(f"Group 1: {cells_1.sum()} cells | Group 2: {cells_2.sum()} cells")

    # ── Detect data type ───────────────────────────────────────────────────
    is_log2    = True
    compute_FC = True

    if is_count_table or data_matrix.max() > 35:
        is_log2 = False
    if data_matrix.min() < 0:
        compute_FC     = False
        is_log2        = False
        is_count_table = False
        data_warnings.append({
            "name": "Data has negative values — no FC will be computed.",
            "description": "Select another dataset if you want to compute Fold-Change.",
        })

    # ── Compute log2 Fold-Change ───────────────────────────────────────────
    total_diff = np.full(n_genes, np.nan)
    if compute_FC:
        total_diff = (
            log2_pseudo_mean(data_matrix[:, cells_1], is_log2)
            - log2_pseudo_mean(data_matrix[:, cells_2], is_log2)
        )

    # ── Run DE ─────────────────────────────────────────────────────────────
    print(f"Running DE method: {method} on {n_genes} genes ...")

    out_pvals = np.full(n_genes, np.nan)
    out_fdr   = np.full(n_genes, np.nan)
    out_lfc   = total_diff.copy()

    if method == "deseq2":
        pvals_all, fdr_all, lfc_all = run_deseq2(
            data_matrix, cell_group, batch_data, batch_names, gene_names, nb_cores
        )
        out_pvals = pvals_all
        out_fdr   = fdr_all
        # DESeq2 provides its own shrunken LFC; prefer it over the naive mean difference
        out_lfc   = lfc_all

    else:
        pvals_all, lfc_scanpy = run_scanpy_method(
            data_matrix, cell_group, gene_names, method, nb_cores
        )
        out_pvals = pvals_all
        out_fdr   = bh_fdr(pvals_all)
        # Use naive log2 FC when available (consistent with de.R for all non-DESeq2 methods)
        if not compute_FC:
            out_lfc = lfc_scanpy

    # ── FC sanity check (mirrors de.R) ────────────────────────────────────
    ave_g1 = data_matrix[:, cells_1].mean(axis=1)
    ave_g2 = data_matrix[:, cells_2].mean(axis=1)
    not_na = ~np.isnan(out_lfc)
    if not_na.any():
        correct   = int(np.sum(np.sign(ave_g1[not_na] - ave_g2[not_na]) == np.sign(out_lfc[not_na])))
        incorrect = int(not_na.sum()) - correct
        if correct >= incorrect:
            print("SANITY CHECK: OK")
        else:
            print("SANITY CHECK: flipping logFC sign so that positive = higher in group 1")
            out_lfc = -out_lfc

    # ── Assemble output matrix (5 x n_genes) ─────────────────────────────
    # Layout mirrors de.R: t(data.out) stored as (5, n_genes) in loom
    data_out = np.vstack([
        out_lfc,    # row 0: logFC
        out_pvals,  # row 1: pval
        out_fdr,    # row 2: FDR
        ave_g1,     # row 3: AveG1
        ave_g2,     # row 4: AveG2
    ])

    # ── Write DE results to loom ──────────────────────────────────────────
    write_loom_dataset(loom_path, args.output_dataset, data_out)
    print(f"DE results written to: {args.output_dataset}")

    # ── Volcano plot ───────────────────────────────────────────────────────
    if compute_FC:
        write_volcano_json(out_lfc, out_pvals, gene_names, ens_ids, output_dir)

    # ── Write output.json ──────────────────────────────────────────────────
    result: dict = {
        "metadata": [
            {
                "name":      args.output_dataset,
                "on":        "GENE",
                "type":      "NUMERIC",
                "nber_cols": DE_NB_COLS,
                "nber_rows": n_genes,
                "headers":   DE_HEADERS,
            }
        ]
    }
    if data_warnings:
        result["warnings"] = data_warnings

    out_json = os.path.join(output_dir, "output.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, allow_nan=True)
    print(f"JSON written to: {out_json}")


## Entry point

def main() -> None:
    if "--help" in sys.argv:
        print(custom_help)
        sys.exit(0)

    parser = argparse.ArgumentParser(description="Differential Expression Script", add_help=False)
    parser.add_argument("-f",                metavar="Input loom file",                                                  required=True)
    parser.add_argument("-o",                metavar="Output folder",                                                    required=False)
    parser.add_argument("--method",          metavar="DE method",           choices=AVAILABLE_METHODS,                   required=True)
    parser.add_argument("--input-dataset",   metavar="Input matrix dataset",                                             required=True,  dest="input_dataset")
    parser.add_argument("--output-dataset",  metavar="Output DE dataset",                                                required=True,  dest="output_dataset")
    parser.add_argument("--batch",           metavar="Batch/covariate datasets",        default="null",                  required=False)
    parser.add_argument("--group-dataset",   metavar="Group 1 metadata path",                                            required=True,  dest="group_dataset")
    parser.add_argument("--group-dataset-2", metavar="Group 2 metadata path",           default=None,                    required=False, dest="group_dataset_2")
    parser.add_argument("--group1",          metavar="Group 1 label",                                                    required=True)
    parser.add_argument("--group2",          metavar="Group 2 label (null = 1-vs-rest)", default="null",                 required=False)
    parser.add_argument("--is-count",        metavar="Is count table [true|false]",     default="false",                 required=False, dest="is_count")
    parser.add_argument("--nb-cores",        metavar="Number of cores",                 default=8, type=positive_int,    required=False, dest="nb_cores")

    args = parser.parse_args()

    # Validate input file (mirrors parse_v8.py main())
    input_path = Path(args.f).resolve()
    if not input_path.is_file():
        ErrorJSON(f"Input file not found: {args.f}")

    run(args)


if __name__ == "__main__":
    main()
