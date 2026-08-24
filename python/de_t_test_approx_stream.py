"""
Out-of-core Welch t_test_approx for de_approx.v8.py.

Fast path: sidecar loom next to the input loom
  ``foo.loom`` -> ``foo.chunked_by_gene.loom``
with a compressed, gene-oriented copy of the expression matrix under
  ``/layers/matrix_chunked_by_gene``           (from ``/matrix``)
  ``/layers/<layer>_chunked_by_gene``          (from ``/layers/<layer>``)

No Annot / output.json registration — cache file only. Safe to delete before archive.

Concurrent builds of the same sidecar are serialized with an exclusive flock on
``*.chunked_by_gene.loom.lock``; waiters re-check and reuse the finished cache.

Gene-block reads + BLAS GEMM across groups. Count tables: library sizes from
/col_attrs/_Depth when aligned; else a matrix pass. Working expression =
normalize_total(1e4) + log1p; Avg. Exp. uses raw.
"""
from __future__ import annotations

import fcntl
import os
import shutil
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import h5py
import numpy as np

# Target bytes for one float32 (gene_block x n_cells) slab during accumulate.
_GENE_BLOCK_TARGET_BYTES = 512 * 1024 * 1024
_STREAM_IF_DENSE_BYTES = 2 * 1024 ** 3
_CHUNKED_BY_GENE_SUFFIX = ".chunked_by_gene.loom"
_ASAP_CACHE_ATTR = "asap_cache"
_ASAP_CACHE_SOURCE_ATTR = "asap_cache_source_dataset"
# HDF5 chunk target (~64 MiB uncompressed float32) for gene-friendly layout.
_HDF5_CHUNK_TARGET_BYTES = 64 * 1024 * 1024


def gene_block_size(n_cells: int, target_bytes: int = _GENE_BLOCK_TARGET_BYTES) -> int:
    per_gene = max(int(n_cells), 1) * 4
    return int(max(32, min(2048, target_bytes // per_gene)))


def dense_matrix_nbytes(n_genes: int, n_cells: int, itemsize: int = 4) -> int:
    return int(n_genes) * int(n_cells) * int(itemsize)


def should_stream_dense(n_genes: int, n_cells: int, itemsize: int = 4) -> bool:
    if os.environ.get("ASAP_DE_FORCE_STREAM", "").strip().lower() in ("1", "true", "yes"):
        return True
    return dense_matrix_nbytes(n_genes, n_cells, itemsize) >= _STREAM_IF_DENSE_BYTES


def loom_path_is_dense_dataset(loom_file: h5py.File, dataset_path: str) -> bool:
    if dataset_path not in loom_file:
        return False
    return isinstance(loom_file[dataset_path], h5py.Dataset)


def chunked_by_gene_loom_path(loom_path: str) -> Path:
    """``dir/foo.loom`` -> ``dir/foo.chunked_by_gene.loom`` (sibling cache file)."""
    p = Path(loom_path).resolve()
    if p.suffix == ".loom":
        return p.with_name(f"{p.stem}{_CHUNKED_BY_GENE_SUFFIX}")
    return p.with_name(f"{p.name}{_CHUNKED_BY_GENE_SUFFIX}")


def chunked_by_gene_dataset_path(input_dataset: str) -> str:
    """
    Path inside the sidecar loom for the gene-chunked copy.

    /matrix              -> /layers/matrix_chunked_by_gene
    /layers/normalized   -> /layers/normalized_chunked_by_gene
    """
    ds = (input_dataset or "").strip()
    if ds in ("/matrix", "matrix"):
        return "/layers/matrix_chunked_by_gene"
    if ds.startswith("/layers/"):
        name = ds[len("/layers/") :].strip("/")
        if not name:
            raise ValueError(f"invalid input_dataset {input_dataset!r}")
        return f"/layers/{name}_chunked_by_gene"
    name = ds.strip("/").replace("/", "_")
    if not name:
        raise ValueError(f"invalid input_dataset {input_dataset!r}")
    return f"/layers/{name}_chunked_by_gene"


def _hdf5_gene_chunks(n_genes: int, n_cells: int) -> tuple[int, int]:
    """Gene-oriented chunks: prefer many cells per chunk so gene-block reads stay efficient."""
    g = int(min(64, max(1, n_genes)))
    c = int(max(1, n_cells))
    max_bytes = _HDF5_CHUNK_TARGET_BYTES
    while g * c * 4 > max_bytes and c > 4096:
        c = max(4096, c // 2)
    while g * c * 4 > max_bytes and g > 1:
        g = max(1, g // 2)
    return (g, min(c, n_cells))


def _perf_enabled() -> bool:
    return os.environ.get("ASAP_DE_PERF_LOG", "").strip().lower() in ("1", "true", "yes")


def _perf_log(msg: str) -> None:
    if _perf_enabled():
        print(msg, flush=True)


def _cache_dataset_valid(dset: h5py.Dataset, shape: tuple[int, int], source_dataset: str) -> bool:
    if tuple(dset.shape) != shape:
        return False
    if dset.dtype != np.float32:
        return False
    if dset.attrs.get(_ASAP_CACHE_ATTR) not in (1, True, b"1", "1"):
        return False
    src = dset.attrs.get(_ASAP_CACHE_SOURCE_ATTR)
    if src is None:
        return False
    if isinstance(src, bytes):
        src = src.decode("utf-8", errors="replace")
    return str(src) == str(source_dataset)


def _try_chunked_by_gene_cache_hit(
    cache_loom: Path,
    dest_ds_path: str,
    shape: tuple[int, int],
    input_dataset: str,
) -> bool:
    if not cache_loom.is_file():
        return False
    try:
        with h5py.File(cache_loom, "r") as cached:
            if dest_ds_path not in cached:
                return False
            if not _cache_dataset_valid(cached[dest_ds_path], shape, input_dataset):
                warnings.warn(
                    f"Expression chunked_by_gene cache {str(cache_loom)!r}::{dest_ds_path} "
                    f"shape/dtype/attrs do not match source; rebuilding that dataset."
                )
                return False
            _perf_log(
                f"[de_perf] chunked_by_gene cache hit {cache_loom}::{dest_ds_path} shape={shape}"
            )
            return True
    except Exception as exc:
        warnings.warn(
            f"Could not open chunked_by_gene cache {str(cache_loom)!r}: {exc}; rebuilding."
        )
        return False


@contextmanager
def _exclusive_file_lock(lock_path: Path) -> Iterator[None]:
    """
    Cross-process exclusive lock for sidecar builds.
    Waiters block in fcntl.flock until the holder finishes (lock released on close/crash).
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fh = open(lock_path, "a+", encoding="utf-8")
    t0 = time.perf_counter()
    _perf_log(f"[de_perf] waiting for chunked_by_gene lock {lock_path}")
    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        waited = time.perf_counter() - t0
        _perf_log(f"[de_perf] acquired chunked_by_gene lock {lock_path} (waited {waited:.1f}s)")
        yield
    finally:
        try:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        finally:
            lock_fh.close()


def _build_chunked_by_gene_dataset(
    loom_path: str,
    input_dataset: str,
    cache_loom: Path,
    dest_ds_path: str,
    shape: tuple[int, int],
) -> None:
    n_genes, n_cells = shape
    tmp_path = cache_loom.parent / f"{cache_loom.name}._tmp_.{os.getpid()}"
    if tmp_path.exists():
        if tmp_path.is_file():
            tmp_path.unlink()
        else:
            shutil.rmtree(tmp_path)

    if cache_loom.is_file():
        shutil.copy2(cache_loom, tmp_path)
        mode = "a"
    else:
        mode = "w"

    gblock = gene_block_size(n_cells)
    chunks = _hdf5_gene_chunks(n_genes, n_cells)
    t0 = time.perf_counter()
    _perf_log(
        f"[de_perf] building chunked_by_gene {tmp_path}::{dest_ds_path} "
        f"shape={shape} chunks={chunks} gene_block={gblock}"
    )
    try:
        with h5py.File(loom_path, "r") as src:
            if not loom_path_is_dense_dataset(src, input_dataset):
                raise ValueError(f"chunked_by_gene cache requires a dense dataset at {input_dataset!r}")
            dataset = src[input_dataset]
            with h5py.File(tmp_path, mode) as dst:
                if "layers" not in dst:
                    dst.create_group("layers")
                if dest_ds_path in dst:
                    del dst[dest_ds_path]
                out = dst.create_dataset(
                    dest_ds_path,
                    shape=shape,
                    dtype=np.float32,
                    chunks=chunks,
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
                out.attrs[_ASAP_CACHE_ATTR] = 1
                out.attrs[_ASAP_CACHE_SOURCE_ATTR] = str(input_dataset)
                for g0 in range(0, n_genes, gblock):
                    g1 = min(n_genes, g0 + gblock)
                    out[g0:g1, :] = np.asarray(dataset[g0:g1, :], dtype=np.float32)
                    if _perf_enabled() and (g0 // gblock) % 20 == 0:
                        _perf_log(f"[de_perf] chunked_by_gene write genes {g0}:{g1}/{n_genes}")
        os.replace(str(tmp_path), str(cache_loom))
    except Exception:
        if tmp_path.exists():
            try:
                if tmp_path.is_file():
                    tmp_path.unlink()
                else:
                    shutil.rmtree(tmp_path)
            except OSError:
                pass
        raise

    size_gib = cache_loom.stat().st_size / (1024 ** 3)
    _perf_log(
        f"[de_perf] chunked_by_gene ready {cache_loom}::{dest_ds_path} "
        f"in {time.perf_counter() - t0:.1f}s (file {size_gib:.2f} GiB on disk; "
        f"dense float32 would be {dense_matrix_nbytes(*shape) / (1024**3):.1f} GiB)"
    )


def ensure_expression_chunked_by_gene_cache(
    loom_path: str,
    input_dataset: str,
    *,
    cache_loom_path: Path | None = None,
) -> tuple[Path, str]:
    """
    Ensure a compressed gene-chunked float32 copy exists in the sidecar loom.
    Builds by streaming gene blocks from the source loom (no full densify in RAM).
    Concurrent callers share one build via an exclusive flock; waiters reuse the result.
    Returns (sidecar_loom_path, dataset_path_inside_sidecar).
    """
    cache_loom = Path(cache_loom_path) if cache_loom_path is not None else chunked_by_gene_loom_path(loom_path)
    dest_ds_path = chunked_by_gene_dataset_path(input_dataset)

    with h5py.File(loom_path, "r") as src:
        if not loom_path_is_dense_dataset(src, input_dataset):
            raise ValueError(f"chunked_by_gene cache requires a dense dataset at {input_dataset!r}")
        shape = (int(src[input_dataset].shape[0]), int(src[input_dataset].shape[1]))

    if _try_chunked_by_gene_cache_hit(cache_loom, dest_ds_path, shape, input_dataset):
        return cache_loom, dest_ds_path

    lock_path = cache_loom.with_name(f"{cache_loom.name}.lock")
    with _exclusive_file_lock(lock_path):
        # Another process may have finished while we waited.
        if _try_chunked_by_gene_cache_hit(cache_loom, dest_ds_path, shape, input_dataset):
            return cache_loom, dest_ds_path
        _build_chunked_by_gene_dataset(loom_path, input_dataset, cache_loom, dest_ds_path, shape)

    if not _try_chunked_by_gene_cache_hit(cache_loom, dest_ds_path, shape, input_dataset):
        raise RuntimeError(
            f"chunked_by_gene cache {str(cache_loom)!r}::{dest_ds_path} missing after build"
        )
    return cache_loom, dest_ds_path


# Backward-compatible aliases (old .npy API names used by callers/tests).
def expression_npy_cache_path(loom_path: str, input_dataset: str) -> Path:
    """Deprecated alias: returns the chunked_by_gene sidecar loom path."""
    return chunked_by_gene_loom_path(loom_path)


def ensure_expression_npy_cache(
    loom_path: str,
    input_dataset: str,
    *,
    cache_path: Path | None = None,
) -> Path:
    """Deprecated alias: builds chunked_by_gene sidecar; returns its path."""
    cache_loom, _ = ensure_expression_chunked_by_gene_cache(
        loom_path, input_dataset, cache_loom_path=cache_path
    )
    return cache_loom


@contextmanager
def open_chunked_by_gene_matrix(
    loom_path: str,
    input_dataset: str,
) -> Iterator[h5py.Dataset]:
    """Build/reuse sidecar and yield the float32 gene-chunked dataset (file stays open)."""
    cache_loom, dest_ds = ensure_expression_chunked_by_gene_cache(loom_path, input_dataset)
    f = h5py.File(cache_loom, "r")
    try:
        yield f[dest_ds]
    finally:
        f.close()


def _library_sizes_from_loom(loom_file: h5py.File, n_cells: int) -> np.ndarray | None:
    if "/col_attrs/_Depth" not in loom_file:
        return None
    depth = loom_file["/col_attrs/_Depth"][:]
    if int(depth.shape[0]) != n_cells:
        warnings.warn(
            f"/col_attrs/_Depth length {depth.shape[0]} != n_cells {n_cells}; "
            "will derive library sizes from the matrix."
        )
        return None
    return np.asarray(depth, dtype=np.float64).ravel()


def _library_sizes_from_matrix(matrix, gblock: int) -> np.ndarray:
    n_genes, n_cells = int(matrix.shape[0]), int(matrix.shape[1])
    lib = np.zeros(n_cells, dtype=np.float64)
    t0 = time.perf_counter() if _perf_enabled() else 0.0
    for g0 in range(0, n_genes, gblock):
        g1 = min(n_genes, g0 + gblock)
        lib += np.asarray(matrix[g0:g1, :], dtype=np.float64).sum(axis=0)
    _perf_log(f"[de_perf] library_sizes from matrix: {time.perf_counter() - t0:.3f}s")
    return lib


def _count_scale(lib_sizes: np.ndarray) -> np.ndarray:
    scale = np.empty(lib_sizes.shape[0], dtype=np.float32)
    np.divide(1e4, lib_sizes, out=scale, where=lib_sizes > 0)
    scale[lib_sizes <= 0] = 0.0
    return scale


def _one_hot_weights(inv: np.ndarray, n_groups: int, dtype=np.float32) -> np.ndarray:
    n_cells = int(inv.shape[0])
    W = np.zeros((n_cells, n_groups), dtype=dtype)
    for gi in range(n_groups):
        W[inv == gi, gi] = 1
    return W


def _accumulate_gene_blocks(
    matrix,
    *,
    W: np.ndarray,
    is_count: bool,
    scale: np.ndarray | None,
    gblock: int,
):
    n_genes, n_cells = (int(matrix.shape[0]), int(matrix.shape[1]))
    n_groups = int(W.shape[1])
    if W.shape[0] != n_cells:
        raise ValueError("W rows must equal n_cells")

    sum_g = np.zeros((n_groups, n_genes), dtype=np.float64)
    sumsq_g = np.zeros((n_groups, n_genes), dtype=np.float64)
    sum_all = np.zeros(n_genes, dtype=np.float64)
    sumsq_all = np.zeros(n_genes, dtype=np.float64)
    if is_count:
        raw_sum_g = np.zeros((n_groups, n_genes), dtype=np.float64)
        raw_sum_all = np.zeros(n_genes, dtype=np.float64)
        if scale is None or scale.shape[0] != n_cells:
            raise ValueError("count path requires per-cell scale")
        ones = np.ones(n_cells, dtype=np.float32)
    else:
        raw_sum_g = None
        raw_sum_all = None
        ones = None

    kept = (W.sum(axis=1) > 0).astype(np.float32, copy=False)
    W64 = W.astype(np.float64, copy=False)
    kept64 = kept.astype(np.float64, copy=False)
    scale64 = scale.astype(np.float64, copy=False) if scale is not None else None

    t0 = time.perf_counter() if _perf_enabled() else 0.0
    n_blocks = 0
    for g0 in range(0, n_genes, gblock):
        g1 = min(n_genes, g0 + gblock)
        raw32 = np.asarray(matrix[g0:g1, :], dtype=np.float32, order="C")
        if is_count:
            ones64 = ones.astype(np.float64, copy=False)
            raw_sum_all[g0:g1] = raw32.astype(np.float64, copy=False) @ ones64
            raw_sum_g[:, g0:g1] = (raw32.astype(np.float64, copy=False) @ W64).T
            work = raw32.astype(np.float64, copy=False)
            work *= scale64
            np.log1p(work, out=work)
            sum_all[g0:g1] = work @ kept64
            work_sq = np.multiply(work, work)
            sumsq_all[g0:g1] = work_sq @ kept64
            sum_g[:, g0:g1] = (work @ W64).T
            sumsq_g[:, g0:g1] = (work_sq @ W64).T
        else:
            work = raw32
            sum_all[g0:g1] = work @ kept
            work_sq = np.multiply(work, work)
            sumsq_all[g0:g1] = work_sq @ kept
            sum_g[:, g0:g1] = (work @ W).T
            sumsq_g[:, g0:g1] = (work_sq @ W).T
        n_blocks += 1

    _perf_log(
        f"[de_perf] gene_block accumulate: {time.perf_counter() - t0:.3f}s "
        f"blocks={n_blocks} genes={n_genes} cells={n_cells} groups={n_groups} gene_block={gblock}"
    )
    return sum_g, sumsq_g, sum_all, sumsq_all, raw_sum_g, raw_sum_all


def _finalize_all_markers(
    unique_groups: np.ndarray,
    n_g: np.ndarray,
    sum_g: np.ndarray,
    sumsq_g: np.ndarray,
    sum_all: np.ndarray,
    sumsq_all: np.ndarray,
    raw_sum_g: np.ndarray | None,
    raw_sum_all: np.ndarray | None,
    is_count: bool,
    mean_var_from_sums: Callable,
    welch_ttest_from_mean_var: Callable,
    bh_fdr: Callable,
):
    n_tot = float(n_g.sum())
    results = {}
    for gi in range(int(unique_groups.size)):
        n_gi = float(n_g[gi])
        n_r = n_tot - n_gi
        sum_r = sum_all - sum_g[gi]
        sumsq_r = sumsq_all - sumsq_g[gi]
        mean_g, var_g = mean_var_from_sums(sum_g[gi], sumsq_g[gi], n_gi)
        mean_r, var_r = mean_var_from_sums(sum_r, sumsq_r, n_r)
        pvals, lfc = welch_ttest_from_mean_var(mean_g, var_g, n_gi, mean_r, var_r, n_r)
        padj = bh_fdr(pvals)
        if is_count:
            ave_g1 = raw_sum_g[gi] / n_gi
            ave_g2 = (raw_sum_all - raw_sum_g[gi]) / n_r
        else:
            ave_g1 = sum_g[gi] / n_gi
            ave_g2 = sum_r / n_r
        results[str(unique_groups[gi])] = (pvals, padj, lfc, ave_g1, ave_g2)
    return results


def run_t_test_approx_all_markers_stream(
    loom_path: str,
    input_dataset: str,
    group_labels: np.ndarray,
    is_count: bool,
    *,
    cell_keep: np.ndarray | None = None,
    mean_var_from_sums: Callable,
    welch_ttest_from_mean_var: Callable,
    bh_fdr: Callable,
):
    gl = np.asarray(group_labels).ravel()
    with open_chunked_by_gene_matrix(loom_path, input_dataset) as matrix:
        n_genes, n_cells = (int(matrix.shape[0]), int(matrix.shape[1]))
        if gl.shape[0] != n_cells:
            raise ValueError(f"group_labels length {gl.shape[0]} does not match n_cells {n_cells}")

        if cell_keep is not None:
            present_mask = np.asarray(cell_keep, dtype=bool).ravel()
            if present_mask.shape[0] != n_cells:
                raise ValueError("cell_keep length must match n_cells")
        else:
            present_mask = np.ones(n_cells, dtype=bool)

        gl_str = gl.astype(str) if gl.dtype.kind in ("U", "S", "O") else np.array([str(x) for x in gl], dtype=object)
        present_labels = gl_str[present_mask]
        unique_groups = np.array(sorted(set(present_labels.tolist())), dtype=object)
        n_groups = int(unique_groups.size)
        if n_groups < 2:
            raise ValueError("FindAllMarkers streaming requires at least 2 groups among kept cells")

        inv = np.full(n_cells, -1, dtype=np.int32)
        for gi, g in enumerate(unique_groups):
            inv[(gl_str == str(g)) & present_mask] = gi

        n_g = np.array([(inv == gi).sum() for gi in range(n_groups)], dtype=np.float64)
        W = _one_hot_weights(inv, n_groups, dtype=np.float32)
        gblock = gene_block_size(n_cells)

        scale = None
        if is_count:
            with h5py.File(loom_path, "r") as f:
                lib = _library_sizes_from_loom(f, n_cells)
            if lib is None:
                lib = _library_sizes_from_matrix(matrix, gblock)
            scale = _count_scale(lib)

        sum_g, sumsq_g, sum_all, sumsq_all, raw_sum_g, raw_sum_all = _accumulate_gene_blocks(
            matrix, W=W, is_count=is_count, scale=scale, gblock=gblock
        )
        return _finalize_all_markers(
            unique_groups, n_g, sum_g, sumsq_g, sum_all, sumsq_all,
            raw_sum_g, raw_sum_all, is_count,
            mean_var_from_sums, welch_ttest_from_mean_var, bh_fdr,
        )


def run_t_test_approx_pairwise_stream(
    loom_path: str,
    input_dataset: str,
    mask1: np.ndarray,
    mask2: np.ndarray,
    is_count: bool,
    *,
    mean_var_from_sums: Callable,
    welch_ttest_from_mean_var: Callable,
    bh_fdr: Callable,
):
    m1 = np.asarray(mask1, dtype=bool).ravel()
    m2 = np.asarray(mask2, dtype=bool).ravel()
    with open_chunked_by_gene_matrix(loom_path, input_dataset) as matrix:
        n_genes, n_cells = (int(matrix.shape[0]), int(matrix.shape[1]))
        if m1.shape[0] != n_cells or m2.shape[0] != n_cells:
            raise ValueError("mask length must match n_cells")

        W = np.column_stack([m1.astype(np.float32), m2.astype(np.float32)])
        n1 = float(m1.sum())
        n2 = float(m2.sum())
        gblock = gene_block_size(n_cells)

        scale = None
        if is_count:
            with h5py.File(loom_path, "r") as f:
                lib = _library_sizes_from_loom(f, n_cells)
            if lib is None:
                lib = _library_sizes_from_matrix(matrix, gblock)
            scale = _count_scale(lib)

        sum_g, sumsq_g, sum_all, sumsq_all, raw_sum_g, raw_sum_all = _accumulate_gene_blocks(
            matrix, W=W, is_count=is_count, scale=scale, gblock=gblock
        )
        sum1, sum2 = sum_g[0], sum_g[1]
        sumsq1, sumsq2 = sumsq_g[0], sumsq_g[1]
        mean1, var1 = mean_var_from_sums(sum1, sumsq1, n1)
        mean2, var2 = mean_var_from_sums(sum2, sumsq2, n2)
        pvals, lfc = welch_ttest_from_mean_var(mean1, var1, n1, mean2, var2, n2)
        padj = bh_fdr(pvals)
        if is_count:
            ave_g1 = raw_sum_g[0] / n1
            ave_g2 = raw_sum_g[1] / n2
        else:
            ave_g1 = sum1 / n1
            ave_g2 = sum2 / n2
        return pvals, padj, lfc, ave_g1, ave_g2
