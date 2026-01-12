from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from contextlib import contextmanager
import psycopg2 # Postgresql adapter
from psycopg2.extensions import connection as PGConnection # Postgresql adapter
from psycopg2.extras import RealDictCursor # Postgresql adapter
from urllib.parse import urlparse # For parsing host string
import argparse # Needed to parse arguments
import sys # Needed for ErrorJSON
import os # Needed for pathjoin and path/create dirs
import h5py # Needed for parsing hdf5 files
import json # For writing output files
import numpy as np # For diverse array calculations
import re # Regex for gene to db comparison
from pathlib import Path # Needed for file extension manipulation
from collections import Counter # To count categories occurrences
import scipy.sparse as sp

custom_help = """
Parsing Mode

Options:
  -f %s             File to parse.
  -o %s             Output folder.
  --filetype %s     File type [RAW_TEXT, LOOM, H5_10x, H5AD, RDS, MTX]
  --header %s       The file has a header [true, false] (default: true).
  --col %s          Name Column [none, first, last] (default: first).
  --sel %s          Name of entry to load from archive or HDF5 (if multiple groups).
  --delim %s        Delimiter (default: tab).
  --organism %i     ID of the organism.
  --dburl %s        DB URL (format HOST:PORT/DB)
  --help            Show this help message and exit.
"""
       
class H5ADHandler:
    @staticmethod
    def _infer_meta_dims(arr: np.ndarray, orientation: str) -> tuple[int, int]:
        if arr.ndim == 0:
            return (1, 1)
        if arr.ndim == 1:
            n = int(arr.shape[0])
            if orientation == "CELL":
                return (1, n)
            if orientation == "GENE":
                return (n, 1)
            return (n, 1)  # GLOBAL: vector as column
        # arr.ndim >= 2
        n_items = int(arr.shape[0])
        n_components = int(arr.shape[1])
        if orientation == "CELL":
            return (n_components, n_items)
        if orientation == "GENE":
            return (n_items, n_components)
        return (n_items, n_components)  # GLOBAL: keep shape
    
    @staticmethod
    def _is_compound_dataset(node) -> bool:
        return isinstance(node, h5py.Dataset) and node.dtype.fields is not None

    @staticmethod
    def _write_no_compound_and_register(*, src_obj: h5py.Dataset, loom, result: dict, existing_paths: set, loom_path: str, orientation: str, expected_len: int | None = None) -> None:
        """
        Writes src_obj to loom_path, guaranteeing NO compound dtype in output.
        Reuses your existing Metadata patterns:

        - If compound with 1 numeric field -> write 1D vector to loom_path and create
          metadata like transfer_metadata (distinct_values, missing_values, categories, finalize_type).
        - Otherwise -> write one dataset per field at loom_path__<field> (best-effort),
          and register each (1D fields get categorical stats; multidim fields get numeric-only meta).
        """
        if loom_path in existing_paths:
            return

        # Load structured array
        data = src_obj[()]
        arr = np.asarray(data)

        # Not compound => just write directly (caller should have handled, but safe)
        if arr.dtype.fields is None:
            # length check if requested
            if expected_len is not None and arr.ndim >= 1 and int(arr.shape[0]) != int(expected_len):
                result.setdefault("warning", []).append(
                    f"Skipping {loom_path}: length mismatch. Expected {expected_len}, found {arr.shape[0]}."
                )
                return

            loom.write(arr, loom_path)

            # register minimal meta (caller usually does its own, but safe)
            meta = Metadata(
                name=loom_path,
                on=orientation,
                imported=1,
                dataset_size=loom.get_dataset_size(loom_path),
            )

            # Compute dims according to our rules
            nber_rows, nber_cols = H5ADHandler._infer_meta_dims(np.asarray(arr), orientation)
            meta.nber_rows = int(nber_rows)
            meta.nber_cols = int(nber_cols)
            
            # if 1D, do full categorical logic like transfer_metadata
            if arr.ndim <= 1:
                vals = arr
                if vals.dtype.kind in ('S', 'U', 'O'):
                    vals = np.array([v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v) for v in vals], dtype=object)
                counts = Counter(vals.flatten() if hasattr(vals, 'flatten') else vals)
                meta.distinct_values = len(counts)
                meta.categories = dict(counts)
                meta.missing_values = count_missing(vals)
                is_numeric = np.issubdtype(np.array(vals).dtype, np.number)
                meta.finalize_type(is_numeric)
            else:
                meta.type = "NUMERIC" if np.issubdtype(np.array(arr).dtype, np.number) else "STRING"

            result["metadata"].append(meta)
            existing_paths.add(loom_path)
            return

        # Compound case
        fields = list(arr.dtype.names or [])
        if not fields:
            result.setdefault("warning", []).append(f"Skipping {loom_path}: compound dataset has no fields.")
            return

        arr_flat = arr.reshape(-1) if arr.ndim > 1 else arr
        n_rows = int(arr_flat.shape[0])

        if expected_len is not None and n_rows != int(expected_len):
            result.setdefault("warning", []).append(
                f"Skipping {loom_path}: length mismatch. Expected {expected_len}, found {n_rows}."
            )
            return

        def field_is_numeric(fname: str) -> bool:
            dt = arr.dtype.fields[fname][0]
            return np.issubdtype(dt, np.number) or np.issubdtype(dt, np.bool_)

        all_numeric = all(field_is_numeric(f) for f in fields)

        # ---- 1 field => keep as 1D dataset ----
        if len(fields) == 1:
            col = np.asarray(arr_flat[fields[0]]).reshape(-1)
            # decode to string if needed
            if col.dtype.kind in ("S", "O", "U"):
                col = np.array(
                    [v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v) for v in col],
                    dtype=object,
                )

            loom.write(col, loom_path)

            meta = Metadata(
                name=loom_path,
                on=orientation,
                imported=1,
                dataset_size=loom.get_dataset_size(loom_path),
            )

            # 1D: keep your categorical logic
            vals = col
            counts = Counter(vals)
            meta.distinct_values = len(counts)
            meta.categories = dict(counts)
            meta.missing_values = count_missing(vals)
            is_numeric = np.issubdtype(np.array(vals).dtype, np.number)
            meta.finalize_type(is_numeric)
            meta.nber_rows = len(vals) if orientation == "GENE" else 1
            meta.nber_cols = len(vals) if orientation == "CELL" else 1

            result["metadata"].append(meta)
            existing_paths.add(loom_path)
            return

        # ---- >1 field => keep as single 2D dataset at SAME path ----
        if all_numeric:
            mat = np.empty((n_rows, len(fields)), dtype=np.float32)
            for j, f_ in enumerate(fields):
                mat[:, j] = np.asarray(arr_flat[f_], dtype=np.float32)
        else:
            # store as strings to avoid splitting
            mat = np.empty((n_rows, len(fields)), dtype=object)
            for j, f_ in enumerate(fields):
                col = np.asarray(arr_flat[f_]).reshape(-1)
                if col.dtype.kind in ("S", "O", "U"):
                    col = np.array(
                        [v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v) for v in col],
                        dtype=object,
                    )
                else:
                    col = np.array([str(v) for v in col], dtype=object)
                mat[:, j] = col

        loom.write(mat, loom_path)

        # Compute dims according to our rules
        nber_rows, nber_cols = H5ADHandler._infer_meta_dims(np.asarray(mat), orientation)
        
        meta = Metadata(
            name=loom_path,
            on=orientation,
            type="NUMERIC" if all_numeric else "STRING",
            nber_rows=int(nber_rows),
            nber_cols=int(nber_cols),
            dataset_size=loom.get_dataset_size(loom_path),
            imported=1,
        )
        result["metadata"].append(meta)
        existing_paths.add(loom_path)
        return
    
    @staticmethod
    def _table_columns(f, table_path: str):
        node = f[table_path]
        if isinstance(node, h5py.Group):
            cols = node.attrs.get('_column_names', list(node.keys()))
            if isinstance(cols, np.ndarray):
                cols = cols.tolist()
            return cols
        elif H5ADHandler._is_compound_dataset(node):
            return list(node.dtype.names or [])
        else:
            return []

    @staticmethod
    def _table_get_column(f, table_path: str, col: str):
        node = f[table_path]
        if isinstance(node, h5py.Group):
            if col not in node:
                raise KeyError(col)
            return node[col]  # may be Dataset or Group (categorical)
        elif H5ADHandler._is_compound_dataset(node):
            # returns a numpy array for that field
            return node[col]
        else:
            raise TypeError(f"Unsupported table node at {table_path}")
    
    @staticmethod
    def get_size(f, path):
        node = f[path]
    
        # 1) Standard attrs (anndata / h5sparse)
        shape = node.attrs.get("shape", None)
        if shape is None:
            shape = node.attrs.get("h5sparse_shape", None)
        if shape is not None:
            return tuple(int(x) for x in shape)
    
        # 2) Dense dataset fallback
        if isinstance(node, h5py.Dataset):
            return tuple(int(x) for x in node.shape)
    
        # 3) Sparse group inference fallback (best-effort)
        if isinstance(node, h5py.Group) and all(k in node for k in ("data", "indices", "indptr")):
            # infer rows/cols depending on encoding
            enc = node.attrs.get("encoding-type", "csr_matrix")
            if isinstance(enc, bytes):
                enc = enc.decode()
            enc = str(enc)
    
            # One dimension is length(indptr)-1
            major = int(node["indptr"].shape[0] - 1)
    
            # Other dimension: max(indices)+1 (expensive but usually ok, still best-effort)
            # Use chunks if huge? For now keep simple.
            idx = node["indices"][:]
            minor = int(idx.max() + 1) if idx.size else 0
    
            if enc == "csr_matrix":
                return (major, minor)   # (cells, genes)
            elif enc == "csc_matrix":
                return (minor, major)   # (cells, genes)
            else:
                ErrorJSON(f"Unsupported encoding-type={enc!r} at {path}")
    
        ErrorJSON(f"Could not determine shape for {path}")
    
    @staticmethod
    def extract_index(f, path):
        try:
            node = f[path]
    
            # CASE A: Group-based obs/var (current behavior)
            if isinstance(node, h5py.Group):
                raw = node.attrs.get('_index', 'index')
                index_col_name = raw.decode() if isinstance(raw, bytes) else raw
                full_path = f"{path.rstrip('/')}/{index_col_name.lstrip('/')}"
                if full_path not in f:
                    full_path = f"{path.rstrip('/')}/_index"
                return [v.decode() if isinstance(v, bytes) else str(v) for v in f[full_path][:]]
    
            # CASE B: Compound dataset obs/var
            if H5ADHandler._is_compound_dataset(node):
                raw = node.attrs.get('_index', None)
                index_field = raw.decode() if isinstance(raw, bytes) else raw
    
                # try attr-provided field first, otherwise common fallbacks
                candidates = []
                if index_field:
                    candidates.append(index_field)
                candidates += ["_index", "index", "obs_names", "var_names"]
    
                for c in candidates:
                    if c in (node.dtype.names or ()):
                        vals = node[c]
                        # decode bytes if needed
                        if getattr(vals, "dtype", None) is not None and vals.dtype.kind in ("S", "O", "U"):
                            return [v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v) for v in vals]
                        return [str(v) for v in vals]
    
                ErrorJSON(f"Could not find index field in compound table {path}. Fields: {node.dtype.names}")
    
            ErrorJSON(f"Unsupported obs/var node at {path}")
    
        except Exception:
            ErrorJSON(f"Could not find index for {path}. Ensure the H5AD is valid.")

    @staticmethod
    def _find_categories_for_column(f, table_path: str, col: str):
        """
        Returns list[str] categories if found, else None.
        Looks in old-style __categories, cat, and raw/cat locations.
        """
        candidates = []
    
        # categories attached to table
        candidates.append(f"{table_path.rstrip('/')}/__categories/{col}")
        candidates.append(f"{table_path.rstrip('/')}/cat/{col}")
    
        # raw-specific category store you mentioned
        # if table_path is raw/obs or raw/var, raw/cat is a likely store
        if table_path.startswith("raw/") or table_path.startswith("/raw/"):
            candidates.append("raw/cat/" + col)
            candidates.append("/raw/cat/" + col)
    
        # also sometimes categories live at top-level /cat
        candidates.append("cat/" + col)
        candidates.append("/cat/" + col)
    
        for p in candidates:
            if p in f and isinstance(f[p], h5py.Dataset):
                cats = f[p][:]
                cats = [v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v) for v in cats]
                return cats

        # Fallback: categories stored in uns as "<col>_categories"
        for p in (f"uns/{col}_categories", f"/uns/{col}_categories"):
            if p in f and isinstance(f[p], h5py.Dataset):
                cats = f[p][:]
                return [v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v) for v in cats]
    
        return None

    @staticmethod
    def transfer_metadata(f, group_path, loom, result, orientation, existing_paths):
        if group_path not in f:
            return
    
        cols = H5ADHandler._table_columns(f, group_path)
        if not cols:
            return
    
        # what to skip
        skip_cols = {"_index", "categories", "codes", "__categories", "cat"}
    
        # Identify index column name if available (group attr)
        node = f[group_path]
        index_col = None
        if isinstance(node, h5py.Group):
            index_col = node.attrs.get('_index', '_index')
            if isinstance(index_col, bytes):
                index_col = index_col.decode()
    
        for col in cols:
            if col in skip_cols or (index_col and col == index_col):
                continue
    
            loom_path = f"/{'col' if orientation == 'CELL' else 'row'}_attrs/{col}"
            if loom_path in existing_paths:
                continue
    
            values = None
    
            # ---- CASE 1: group-based column (could be categorical group or dataset) ----
            if isinstance(node, h5py.Group):
                item = node[col]
    
                # new-style categorical group: {categories, codes}
                if isinstance(item, h5py.Group) and 'categories' in item and 'codes' in item:
                    cats = [v.decode() if isinstance(v, bytes) else str(v) for v in item['categories'][:]]
                    codes = item['codes'][:]
                    values = np.array([cats[c] if c != -1 else "nan" for c in codes], dtype=object)
    
                # old-style categorical: codes in obs/<col>, categories in obs/__categories/<col>
                elif isinstance(item, h5py.Dataset) and np.issubdtype(item.dtype, np.integer):
                    cats = H5ADHandler._find_categories_for_column(f, group_path, col)
                    if cats is not None:
                        codes = item[:]
                        def map_code(c):
                            if c == -1: return "nan"
                            c = int(c)
                            return cats[c] if 0 <= c < len(cats) else "nan"
                        values = np.array([map_code(c) for c in codes], dtype=object)
                    else:
                        values = item[:]
    
                else:
                    raw_values = item[:]
                    if raw_values.dtype.kind in ('S', 'U', 'O'):
                        values = np.array([v.decode() if isinstance(v, bytes) else str(v) for v in raw_values], dtype=object)
                    else:
                        values = raw_values
    
            # ---- CASE 2: compound table (values are already arrays) ----
            else:
                raw_values = node[col]
    
                # If integer codes and categories exist (raw.cat etc.), decode them
                if np.issubdtype(np.array(raw_values).dtype, np.integer):
                    cats = H5ADHandler._find_categories_for_column(f, group_path, col)
                    if cats is not None:
                        def map_code(c):
                            if c == -1: return "nan"
                            c = int(c)
                            return cats[c] if 0 <= c < len(cats) else "nan"
                        values = np.array([map_code(c) for c in raw_values], dtype=object)
                    else:
                        values = raw_values
                else:
                    if np.array(raw_values).dtype.kind in ('S', 'U', 'O'):
                        values = np.array([v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v) for v in raw_values], dtype=object)
                    else:
                        values = raw_values
    
            if values is None:
                continue

            # Check dimension consistency
            expected_len = result["nber_cols"] if orientation == "CELL" else result["nber_rows"]
            if len(values) != expected_len:
                entity = "cells" if orientation == "CELL" else "genes"
                result.setdefault("warning", []).append(f"Skipping {col} from {group_path}: length mismatch. Expected {expected_len} {entity}, but found {len(values)}.")
                continue

            # Compute dimensions according to our rules
            arr = np.asarray(values)
            nber_rows, nber_cols = H5ADHandler._infer_meta_dims(arr, orientation)
            
            meta = Metadata(
                name=loom_path,
                on=orientation,
                nber_rows=int(nber_rows),
                nber_cols=int(nber_cols),
                missing_values=count_missing(values),
                imported=1
            )
    
            counts = Counter(values.flatten() if hasattr(values, 'flatten') else values)
            meta.distinct_values = len(counts)
            meta.categories = dict(counts)
    
            is_numeric = np.issubdtype(np.array(values).dtype, np.number)
            meta.finalize_type(is_numeric)
    
            loom.write(values, loom_path)
            meta.dataset_size = loom.get_dataset_size(loom_path)
    
            result["metadata"].append(meta)
            existing_paths.add(loom_path)


    @staticmethod
    def transfer_multidimensional_metadata(f, group_path, loom, result, orientation, existing_paths):
        """
        Handles transferring obsm (CELL) and varm (GENE) multidimensional matrices.
        """
        if group_path not in f: return
        group = f[group_path]
        prefix = "col" if orientation == "CELL" else "row"
    
        for key in group.keys():
            loom_path = f"/{prefix}_attrs/{key}"
            
            # 1. Check for duplicates
            if loom_path in existing_paths:
                result.setdefault("warning", []).append(f"Skipping {group_path}/{key}: already exists.")
                continue
    
            # 2. Extract and Write
            # These are usually stored as datasets (e.g., PCA, UMAP)
            try:
                # Check dimensions before loading full data (if possible)
                expected_n = result["nber_cols"] if orientation == "CELL" else result["nber_rows"]
                actual_shape = group[key].shape
                if actual_shape[0] != expected_n:
                    result.setdefault("warning", []).append(f"Skipping {group_path}/{key}: dimension mismatch. Expected {expected_n} {orientation.lower()}s, but found {actual_shape[0]}.")
                    continue

                # Continue if dimension is ok
                item = group[key]
                if not isinstance(item, h5py.Dataset):
                    result.setdefault("warning", []).append(f"Skipping {group_path}/{key}: not a dataset.")
                    continue
              
                # Handle compound datasets without writing compound dtypes
                if H5ADHandler._is_compound_dataset(item):
                    H5ADHandler._write_no_compound_and_register(
                        src_obj=item,
                        loom=loom,
                        result=result,
                        existing_paths=existing_paths,
                        loom_path=loom_path,
                        orientation=orientation,
                        expected_len=expected_n
                    )
                    continue
                
                # Normal multidim write path (your existing behavior)
                data = item[:]
                loom.write(data, loom_path)

                # Compute dims according to our rules
                nber_rows, nber_cols = H5ADHandler._infer_meta_dims(np.asarray(data), orientation)
                
                meta = Metadata(
                    name=loom_path,
                    on=orientation,
                    type="NUMERIC",
                    nber_rows=int(nber_rows),
                    nber_cols=int(nber_cols),
                    dataset_size=loom.get_dataset_size(loom_path),
                    imported=1
                )
                result["metadata"].append(meta)
                existing_paths.add(loom_path)
                
            except Exception as e:
                result.setdefault("warning", []).append(f"Could not transfer {group_path}/{key}: {str(e)}")

    @staticmethod
    def transfer_layers(f, loom, result, n_cells, n_genes, existing_paths):
        """
        Transfers additional matrices from H5AD /layers to Loom /layers.
        Supports dense, CSR, and CSC encoded sparse layers.
        Uses chunked processing to remain RAM-efficient.
        """
        if "layers" not in f:
            return
    
        def _encoding_type(h5_group) -> str:
            et = h5_group.attrs.get("encoding-type", "csr_matrix")
            if isinstance(et, bytes):
                et = et.decode()
            return str(et)
    
        for layer_name in f["layers"].keys():
            loom_path = f"/layers/{layer_name}"
    
            # 1) Duplicate check
            if loom_path in existing_paths:
                result.setdefault("warning", []).append(f"Skipping layer {layer_name}: already exists.")
                continue
    
            try:
                # 2) Dimension check (Cells, Genes in H5AD)
                l_shape = H5ADHandler.get_size(f, f"layers/{layer_name}")
                if l_shape[0] != n_cells or l_shape[1] != n_genes:
                    result.setdefault("warning", []).append(
                        f"Skipping layer {layer_name}: dimension mismatch. "
                        f"Expected ({n_cells}, {n_genes}), found {l_shape}."
                    )
                    continue
    
                # 3) Prepare Loom dataset (Genes x Cells)
                dset = loom.handle.create_dataset(
                    loom_path,
                    shape=(n_genes, n_cells),
                    dtype="float32",
                    chunks=(min(n_genes, 1024), min(n_cells, 1024)),
                    compression="gzip",
                )
    
                layer_node = f["layers"][layer_name]
    
                is_sparse = (
                    isinstance(layer_node, h5py.Group)
                    and all(k in layer_node for k in ("data", "indices", "indptr"))
                )
                is_dense = isinstance(layer_node, h5py.Dataset)
    
                if not (is_sparse or is_dense):
                    result.setdefault("warning", []).append(
                        f"Skipping layer {layer_name}: unsupported layout (not dense dataset or sparse group)."
                    )
                    continue
    
                # 4) If CSC, build CSR view once
                enc = None
                csr_view = None
                if is_sparse:
                    enc = _encoding_type(layer_node)
                    if enc == "csc_matrix":
                        X_csc = sp.csc_matrix(
                            (layer_node["data"][:], layer_node["indices"][:], layer_node["indptr"][:]),
                            shape=(n_cells, n_genes),
                        )
                        csr_view = X_csc.tocsr()
                    elif enc != "csr_matrix":
                        result.setdefault("warning", []).append(
                            f"Skipping layer {layer_name}: unsupported encoding-type={enc!r}."
                        )
                        continue
    
                # 5) Chunked write (cells x genes -> genes x cells)
                for start in range(0, n_cells, 1024):
                    end = min(start + 1024, n_cells)
    
                    if is_sparse:
                        if enc == "csr_matrix":
                            ptr = layer_node["indptr"][start:end + 1]
                            chunk = sp.csr_matrix(
                                (layer_node["data"][ptr[0]:ptr[-1]],
                                 layer_node["indices"][ptr[0]:ptr[-1]],
                                 ptr - ptr[0]),
                                shape=(end - start, n_genes),
                            ).toarray()
                        else:
                            # enc == "csc_matrix"
                            chunk = csr_view[start:end, :].toarray()
                    else:
                        chunk = layer_node[start:end, :]
    
                    dset[:, start:end] = chunk.T
    
                # 6) Metadata entry
                meta = Metadata(
                    name=loom_path,
                    on="EXPRESSION_MATRIX",
                    type="NUMERIC",
                    nber_rows=n_genes,
                    nber_cols=n_cells,
                    dataset_size=loom.get_dataset_size(loom_path),
                    imported=1,
                )
                result["metadata"].append(meta)
                existing_paths.add(loom_path)
    
            except Exception as e:
                result.setdefault("warning", []).append(f"Failed to transfer layer {layer_name}: {str(e)}")

    @staticmethod
    def transfer_unstructured_metadata(f, loom, result, existing_paths):
        if "uns" not in f:
            return
    
        # Keys in uns that are really categorical dictionaries / lookup tables
        def should_skip_uns_key(key: str) -> bool:
            k = key.lower()
            # the one you hit
            if k.endswith("_categories"):
                return True
            # optional: also skip other common ones if you don't want them
            # if k.endswith("_levels") or k.endswith("_codes"):
            #     return True
            return False
    
        def walk(group, prefix=""):
            for key in group.keys():
                item = group[key]
                attr_name = f"{prefix}{key}"
    
                # --- NEW: skip *_categories (and friends, if you enable them) anywhere in uns ---
                if should_skip_uns_key(attr_name):
                    continue
    
                loom_path = f"/attrs/{attr_name}"
    
                if isinstance(item, h5py.Group):
                    walk(item, prefix=f"{attr_name}_")
                    continue
    
                if not isinstance(item, h5py.Dataset):
                    continue
    
                if loom_path in existing_paths:
                    continue

                if H5ADHandler._is_compound_dataset(item):
                    # We typically do NOT want multidim treatment for uns; decompose to 1D or per-field
                    H5ADHandler._write_no_compound_and_register(
                        src_obj=item,
                        loom=loom,
                        result=result,
                        existing_paths=existing_paths,
                        loom_path=loom_path,
                        orientation="GLOBAL",
                        expected_len=None
                    )
                    continue
                
                try:
                    val = item[()]
                    if isinstance(val, (bytes, np.bytes_)):
                        val = val.decode("utf-8")
                    elif isinstance(val, np.ndarray) and val.dtype.kind in ("S", "U"):
                        val = np.array(
                            [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in val.flatten()]
                        ).reshape(val.shape)
    
                    loom.write(val, loom_path)

                    # Compute dims according to our rules
                    val_arr = val if isinstance(val, np.ndarray) else np.array(val)
                    nber_rows, nber_cols = H5ADHandler._infer_meta_dims(val_arr, "GLOBAL")
                    
                    is_num = False
                    try:
                        is_num = np.issubdtype(np.array(val).dtype, np.number)
                    except:
                        pass
    
                    meta = Metadata(
                        name=loom_path,
                        on="GLOBAL",
                        type="NUMERIC" if is_num else "STRING",
                        nber_rows=int(nber_rows),
                        nber_cols=int(nber_cols),
                        dataset_size=loom.get_dataset_size(loom_path),
                        imported=1
                    )
                    result["metadata"].append(meta)
                    existing_paths.add(loom_path)
    
                except Exception as e:
                    result.setdefault("warning", []).append(f"Skipping uns item {attr_name}: {str(e)}")
    
        walk(f["uns"])

    @staticmethod
    def parse(args, file_path, gene_db):
        result = {
            "detected_format": "H5AD",
            "input_path": file_path,
            "input_group": "",
            "output_path": args.output_path,
            "nber_rows":-1,
            "nber_cols":-1,
            "nber_not_found_genes":-1,
            "nber_zeros":-1,
            "is_count_table":-1,
            "empty_cols": -1,
            "empty_rows": -1,
            "dataset_size": -1,
            "metadata": []
        }
        # Deal with previous warnings
        if getattr(args, "warnings", None):
            result["warning"] = list(args.warnings)

        # I. Read the h5ad file and find the correct matrix to load
        with h5py.File(file_path, 'r') as f:
            ## 1. Check if --sel was provided
            if args.sel is not None:
                # Case 1: group explicitly provided
                if args.sel not in f:
                    ErrorJSON(f"Path '{args.sel}' (provided with --sel) not found in file")
                result["input_group"] = args.sel
            else:
                # Case 2: group is None → find if there is indeed exactly one candidate
                CANDIDATES = ['/X', '/raw/X', '/raw.X']
                matches = [g for g in CANDIDATES if g in f]
                if len(matches) == 0:
                    ErrorJSON(f"None of the candidate path were found: {CANDIDATES}. Please provide a path with --sel")
                elif len(matches) > 1:
                    ErrorJSON(f"Multiple candidate groups found: {CANDIDATES}. Please provide a path with --sel")
                result["input_group"] = matches[0]
                
            ## 2. Get matrix dimensions
            result["nber_cols"], result["nber_rows"] = H5ADHandler.get_size(f, result["input_group"])
            
            ## 3. Create Loom file
            loom = LoomFile(args.output_path)
            loom.ribo_protein_gene_names = getattr(gene_db, "ribo_protein_gene_names", set())
            
            ## 4. Handle the gene annotation
            ### 4.a. Load gene/cell names
            var_path = "/var" # default
            obs_path = "/obs" # default
            if result["input_group"] == '/raw.X':
                if '/raw.var' in f: var_path = "/raw.var"
                if '/raw.obs' in f: obs_path = "/raw.obs"
            elif result["input_group"] == '/raw/X':
                if '/raw/var' in f: var_path = "/raw/var"
                if '/raw/obs' in f: obs_path = "/raw/obs"
            var_genes = H5ADHandler.extract_index(f, var_path)
            obs_cells = H5ADHandler.extract_index(f, obs_path)
            
            ### 4.b. Write Cell names and Original gene names + fill associated metadata for output JSON
            loom.write(obs_cells, "/col_attrs/CellID") # Add cells to Loom           
            result["metadata"].append(Metadata(name="/col_attrs/CellID", on="CELL", type="STRING", nber_cols=len(obs_cells), nber_rows=1, distinct_values=len(set(obs_cells)), missing_values=count_missing(obs_cells), dataset_size=loom.get_dataset_size("/col_attrs/CellID"), imported=0))
            loom.write(var_genes, "/row_attrs/Original_Gene") # Add original gene names to Loom
            result["metadata"].append(Metadata(name="/row_attrs/Original_Gene", on="GENE", type="STRING", nber_cols=1, nber_rows=len(var_genes), distinct_values=len(set(var_genes)), missing_values=count_missing(var_genes), dataset_size=loom.get_dataset_size("/row_attrs/Original_Gene"), imported=0))
            ### 4.c. Add informations from Gene database
            parsed_genes, result["nber_not_found_genes"] = gene_db.parse_genes_list(var_genes) # Compare genes to database
            # Extract the vectors for Loom writing and write to loom
            #### 4.c.1. Ensembl IDS
            ens_ids = [g.ensembl_id for g in parsed_genes]
            loom.write(ens_ids, "/row_attrs/Accession")
            result["metadata"].append(Metadata(name="/row_attrs/Accession", on="GENE", type="STRING", nber_cols=1, nber_rows=len(ens_ids), distinct_values=len(set(ens_ids)), missing_values=count_missing(ens_ids), dataset_size=loom.get_dataset_size("/row_attrs/Accession"), imported=0))
            #### 4.c.2. Gene symbols       
            gene_names = [g.name for g in parsed_genes]
            loom.write(gene_names, "/row_attrs/Gene") # Backward compatibility. New path is "Name"
            result["metadata"].append(Metadata(name="/row_attrs/Gene", on="GENE", type="STRING", nber_cols=1, nber_rows=len(gene_names), distinct_values=len(set(gene_names)), missing_values=count_missing(gene_names), dataset_size=loom.get_dataset_size("/row_attrs/Gene"), imported=0))
            #### 4.c.3. Gene symbols        
            loom.write(gene_names, "/row_attrs/Name")
            result["metadata"].append(Metadata(name="/row_attrs/Name", on="GENE", type="STRING", nber_cols=1, nber_rows=len(gene_names), distinct_values=len(set(gene_names)), missing_values=count_missing(gene_names), dataset_size=loom.get_dataset_size("/row_attrs/Name"), imported=0))
            #### 4.c.3. Biotypes     
            biotypes = [g.biotype for g in parsed_genes]
            loom.write(biotypes, "/row_attrs/_Biotypes")
            result["metadata"].append(Metadata(name="/row_attrs/_Biotypes", on="GENE", type="DISCRETE", nber_cols=1, nber_rows=len(biotypes), distinct_values=len(set(biotypes)), missing_values=count_missing(biotypes), dataset_size=loom.get_dataset_size("/row_attrs/_Biotypes"), imported=0, categories=dict(Counter(biotypes))))
            #### 4.c.4. Chromosomes     
            chroms = [g.chr for g in parsed_genes]
            loom.write(chroms, "/row_attrs/_Chromosomes")
            result["metadata"].append(Metadata(name="/row_attrs/_Chromosomes", on="GENE", type="DISCRETE", nber_cols=1, nber_rows=len(chroms), distinct_values=len(set(chroms)), missing_values=count_missing(chroms), dataset_size=loom.get_dataset_size("/row_attrs/_Chromosomes"), imported=0, categories=dict(Counter(chroms))))
            #### 4.c.5. Sum Exon Length     
            sum_exon_length = [g.sum_exon_length for g in parsed_genes]
            loom.write(sum_exon_length, "/row_attrs/_SumExonLength")
            result["metadata"].append(Metadata(name="/row_attrs/_SumExonLength", on="GENE", type="NUMERIC", nber_cols=1, nber_rows=len(sum_exon_length), distinct_values=len(set(sum_exon_length)), missing_values=count_missing(sum_exon_length), dataset_size=loom.get_dataset_size("/row_attrs/_SumExonLength"), imported=0))

            ## 5. Handle the main matrix
            ### 5.a. Write matrix to Loom and compute stats on-the-fly
            stats = loom.write_matrix(f, result["input_group"], "/matrix", result["nber_cols"], result["nber_rows"], parsed_genes)
            
            ### 5.b. Fill result dictionary
            result["dataset_size"] = loom.get_dataset_size("/matrix")
            result["nber_zeros"] = int(stats["total_zeros"])
            result["empty_cols"] = int(stats["empty_cells"])
            result["empty_rows"] = int(stats["empty_genes"])
            result["is_count_table"] = int(stats["is_count_table"])
            ### 5.c. Write the calculated vectors to Loom
            ### 5.c.1. Depth (sum counts per cell)
            loom.write(stats["cell_depth"], "/col_attrs/_Depth")
            result["metadata"].append(Metadata(name="/col_attrs/_Depth", on="CELL", type="NUMERIC", nber_cols=len(stats["cell_depth"]), nber_rows=1, distinct_values=len(set(stats["cell_depth"])), missing_values=count_missing(stats["cell_depth"]), dataset_size=loom.get_dataset_size("/col_attrs/_Depth"), imported=0))
            ### 5.c.2. Detected genes
            loom.write(stats["cell_detected"], "/col_attrs/_Detected_Genes")
            result["metadata"].append(Metadata(name="/col_attrs/_Detected_Genes", on="CELL", type="NUMERIC", nber_cols=len(stats["cell_detected"]), nber_rows=1, distinct_values=len(set(stats["cell_detected"])), missing_values=count_missing(stats["cell_detected"]), dataset_size=loom.get_dataset_size("/col_attrs/_Detected_Genes"), imported=0))
            ### 5.c.3. Mitochondrial content
            loom.write(stats["cell_mt"], "/col_attrs/_Mitochondrial_Content")
            result["metadata"].append(Metadata(name="/col_attrs/_Mitochondrial_Content", on="CELL", type="NUMERIC", nber_cols=len(stats["cell_mt"]), nber_rows=1, distinct_values=len(set(stats["cell_mt"])), missing_values=count_missing(stats["cell_mt"]), dataset_size=loom.get_dataset_size("/col_attrs/_Mitochondrial_Content"), imported=0))
            ### 5.c.4. Ribosomal content
            loom.write(stats["cell_rrna"], "/col_attrs/_Ribosomal_Content")
            result["metadata"].append(Metadata(name="/col_attrs/_Ribosomal_Content", on="CELL", type="NUMERIC", nber_cols=len(stats["cell_rrna"]), nber_rows=1, distinct_values=len(set(stats["cell_rrna"])), missing_values=count_missing(stats["cell_rrna"]), dataset_size=loom.get_dataset_size("/col_attrs/_Ribosomal_Content"), imported=0))
            ### 5.c.5. Protein coding content
            loom.write(stats["cell_prot"], "/col_attrs/_Protein_Coding_Content")
            result["metadata"].append(Metadata(name="/col_attrs/_Protein_Coding_Content", on="CELL", type="NUMERIC", nber_cols=len(stats["cell_prot"]), nber_rows=1, distinct_values=len(set(stats["cell_prot"])), missing_values=count_missing(stats["cell_prot"]), dataset_size=loom.get_dataset_size("/col_attrs/_Protein_Coding_Content"), imported=0))
            ### 5.c.6. Sum counts per gene
            loom.write(stats["gene_sum"], "/row_attrs/_Sum")
            result["metadata"].append(Metadata(name="/row_attrs/_Sum", on="GENE", type="NUMERIC", nber_cols=1, nber_rows=len(stats["gene_sum"]), distinct_values=len(set(stats["gene_sum"])), missing_values=count_missing(stats["gene_sum"]), dataset_size=loom.get_dataset_size("/row_attrs/_Sum"), imported=0))

            ## 6. Add the Stable IDs
            stable_ids_rows = list(range(result["nber_rows"]))
            loom.write(stable_ids_rows, "/row_attrs/_StableID")
            result["metadata"].append(Metadata(name="/row_attrs/_StableID", on="GENE", type="NUMERIC", nber_cols=1, nber_rows=len(stable_ids_rows), distinct_values=len(set(stable_ids_rows)), missing_values=count_missing(stable_ids_rows), dataset_size=loom.get_dataset_size("/row_attrs/_StableID"), imported=0))

            stable_ids_cols = list(range(result["nber_cols"]))
            loom.write(stable_ids_cols, "/col_attrs/_StableID")
            result["metadata"].append(Metadata(name="/col_attrs/_StableID", on="CELL", type="NUMERIC", nber_cols=len(stable_ids_cols), nber_rows=1, distinct_values=len(set(stable_ids_cols)), missing_values=count_missing(stable_ids_cols), dataset_size=loom.get_dataset_size("/col_attrs/_StableID"), imported=0))

            ## 7. Transfer the existing metadata from h5ad to Loom
            # To avoid overwriting fields I've already generated
            existing_paths = {m.name for m in result["metadata"]}
            existing_paths.add("/attrs/LOOM_SPEC_VERSION")

            ### 7.a. Transfer obs (Cell unidimensional metadata)
            if result["input_group"] == '/raw.X' and '/raw.obs' in f: H5ADHandler.transfer_metadata(f, "raw.obs", loom, result, "CELL", existing_paths)
            if result["input_group"] == '/raw/X' and '/raw/obs' in f: H5ADHandler.transfer_metadata(f, "raw/obs", loom, result, "CELL", existing_paths)
            H5ADHandler.transfer_metadata(f, "obs", loom, result, "CELL", existing_paths)
            
            ### 7.b. Transfer var (Gene unidimensional metadata)
            if result["input_group"] == '/raw.X' and '/raw.var' in f: H5ADHandler.transfer_metadata(f, "raw.var", loom, result, "GENE", existing_paths)
            if result["input_group"] == '/raw/X' and '/raw/var' in f: H5ADHandler.transfer_metadata(f, "raw/var", loom, result, "GENE", existing_paths)
            H5ADHandler.transfer_metadata(f, "var", loom, result, "GENE", existing_paths)
            
            ### 7.c. Transfer obsm (Cell multidimensional metadata)
            if result["input_group"] == '/raw.X' and '/raw.obsm' in f: H5ADHandler.transfer_multidimensional_metadata(f, "raw.obsm", loom, result, "CELL", existing_paths)
            if result["input_group"] == '/raw/X' and '/raw/obsm' in f: H5ADHandler.transfer_multidimensional_metadata(f, "raw/obsm", loom, result, "CELL", existing_paths)
            H5ADHandler.transfer_multidimensional_metadata(f, "obsm", loom, result, "CELL", existing_paths)
            
            ### 7.d. Transfer varm (Gene multidimensional metadata)
            if result["input_group"] == '/raw.X' and '/raw.varm' in f: H5ADHandler.transfer_multidimensional_metadata(f, "raw.varm", loom, result, "GENE", existing_paths)
            if result["input_group"] == '/raw/X' and '/raw.varm' in f: H5ADHandler.transfer_multidimensional_metadata(f, "raw/varm", loom, result, "GENE", existing_paths)
            H5ADHandler.transfer_multidimensional_metadata(f, "varm", loom, result, "GENE", existing_paths)

            ### 7.e. Layers (Alternative matrices like 'spliced'/'unspliced' or 'transformed')
            H5ADHandler.transfer_layers(f, loom, result, result["nber_cols"], result["nber_rows"], existing_paths)

            ### 7.f: uns (Unstructured metadata like colors or global parameters)
            H5ADHandler.transfer_unstructured_metadata(f, loom, result, existing_paths)

            ### 7.g. Transfer alternative main matrices to layers
            # We check the candidates and see if they were the one picked as 'input_group'
            for candidate in ['/X', '/raw/X', '/raw.X']:
                if candidate in f and candidate != result["input_group"]:
                    try:
                        # 1. Check dimensions in H5AD (Cells, Genes)
                        c_count, r_count = H5ADHandler.get_size(f, candidate)
                        if c_count == result["nber_cols"] and r_count == result["nber_rows"]:
                            layer_name = candidate.strip("/").replace("/", "_")
                            loom_layer_path = f"/layers/{layer_name}"
                            if loom_layer_path not in existing_paths:
                                # 2. Write matrix and get stats (passing None for gene_metadata to skip heavy stats)
                                layer_stats = loom.write_matrix(f, candidate, loom_layer_path, result["nber_cols"], result["nber_rows"], None)
                                # 3. Register in Metadata with layer-specific stats
                                meta = Metadata(
                                    name=loom_layer_path,
                                    on="EXPRESSION_MATRIX",
                                    type="NUMERIC",
                                    nber_rows=result["nber_rows"],
                                    nber_cols=result["nber_cols"],
                                    dataset_size=loom.get_dataset_size(loom_layer_path),
                                    imported=1
                                )
                                # Manually attach layer-specific stats since Metadata constructor might not handle them as positional arguments
                                meta.nber_zeros = layer_stats["total_zeros"]
                                meta.is_count_table = layer_stats["is_count_table"]
                                result["metadata"].append(meta)
                                existing_paths.add(loom_layer_path)
                        else:
                            # Log dimension mismatch as a warning
                            result.setdefault("warning", []).append( f"Skipping {candidate}: Dimensions ({r_count}x{c_count}) don't match primary matrix ({result['nber_rows']}x{result['nber_cols']})")
                    except Exception as e:
                        result.setdefault("warning", []).append(f"Could not transfer alternative matrix {candidate}: {e}")
            
            ### 8. end
            loom.close()

        # II. Write final JSON
        json_str = json.dumps(result, default=dataclass_to_json, ensure_ascii=False)  # to deal with class Metadata serialization
        
        if args.o:
            with open(os.path.join(args.o, "output.json"), "w", encoding="utf-8") as out:
                out.write(json_str)
        else:
            print(json_str)

class LoomHandler:
    """
    Parses an input .loom and writes a *new* .loom (Loom v3 layout) to args.output_path,
    reproducing the H5ADHandler.parse behavior:
      - detect/select main matrix
      - compute on-the-fly stats (depth, detected genes, mt/ribo/protein-coding %)
      - write gene annotations from the DB (Accession/Name/Biotypes/Chromosomes/etc.)
      - copy all other matrices + metadata from the source loom to the new loom
      - emit JSON summary with Metadata entries
    """

    @staticmethod
    def _infer_meta_dims(arr: np.ndarray, on: str, *, n_cells: int, n_genes: int) -> tuple[int, int]:
        """
        Enforce Loom conventions:
          - CELL: nber_cols == n_cells
          - GENE: nber_rows == n_genes
          - EXPRESSION_MATRIX: (n_genes, n_cells)
          - GLOBAL: use natural shape rules
        """
        if arr.ndim == 0:
            return (1, 1)

        if on == "CELL":
            if arr.ndim == 1:
                return (1, n_cells)
            r, c = int(arr.shape[0]), int(arr.shape[1])
            if r == n_cells:
                return (c, n_cells)
            if c == n_cells:
                return (r, n_cells)
            # fallback: force n_cells
            return (r, n_cells)

        if on == "GENE":
            if arr.ndim == 1:
                return (n_genes, 1)
            r, c = int(arr.shape[0]), int(arr.shape[1])
            if r == n_genes:
                return (n_genes, c)
            if c == n_genes:
                return (n_genes, r)
            # fallback: force n_genes
            return (n_genes, c)

        if on == "EXPRESSION_MATRIX":
            return (n_genes, n_cells)

        # GLOBAL (or unknown): natural shape
        if arr.ndim == 1:
            return (int(arr.shape[0]), 1)
        return (int(arr.shape[0]), int(arr.shape[1]))
    
    @staticmethod
    def _is_compound_dataset(obj) -> bool:
        return isinstance(obj, h5py.Dataset) and obj.dtype.fields is not None

    @staticmethod
    def _compound_field_names(obj: h5py.Dataset) -> list[str]:
        return list(obj.dtype.names or [])

    @staticmethod
    def _compound_all_numeric(obj: h5py.Dataset) -> bool:
        # All fields numeric? (int/float/bool)
        for name in LoomHandler._compound_field_names(obj):
            dt = obj.dtype.fields[name][0]
            if not np.issubdtype(dt, np.number) and not np.issubdtype(dt, np.bool_):
                return False
        return True

    @staticmethod
    def _write_compound_as_regular( src_obj: h5py.Dataset, loom_out: LoomFile, result: dict, existing_paths: set[str], full_path: str) -> None:
        """
        Writes a compound dataset without using compound dtypes in the output.
        Strategy:
          - If all fields numeric => write 2D float32 dataset to SAME path (N x K).
          - Else => write one dataset per field: f"{path}__{field}".
        """
        fields = LoomHandler._compound_field_names(src_obj)
        if not fields:
            result.setdefault("warning", []).append(f"Skipping compound dataset {full_path}: no fields found.")
            return

        # Read once
        data = src_obj[:]  # structured array shape (N,) or (N,...) typically

        # Most loom attrs like Embedding are (N,) structured arrays.
        # If it's higher-rank structured, we still try to flatten leading dims into rows.
        # We'll reshape to (N_rows, ) where possible.
        if isinstance(data, np.ndarray) and data.dtype.fields is not None:
            # reshape structured array to 1D rows if possible
            if data.ndim > 1:
                # flatten all but last? safest: flatten completely
                data_flat = data.reshape(-1)
            else:
                data_flat = data
        else:
            # shouldn't happen, but guard
            result.setdefault("warning", []).append(f"Skipping {full_path}: expected structured ndarray.")
            return

        n_rows = int(data_flat.shape[0])

        # Case 1: all numeric -> write (n_rows x n_fields) to SAME path
        if LoomHandler._compound_all_numeric(src_obj):
            # If only 1 field, write a 1D vector (NOT a (n,1) matrix)
            if len(fields) == 1:
                name = fields[0]
                vec = np.asarray(data_flat[name])  # keep native numeric dtype (int/float)
                # ensure it is 1D
                vec = vec.reshape(-1)
        
                loom_out.write(vec, full_path)
                meta = LoomHandler._make_meta_for_written_dataset(
                    loom_out, full_path, vec,
                    n_cells=result["nber_cols"],
                    n_genes=result["nber_rows"],
                )
                result["metadata"].append(meta)
                existing_paths.add(full_path)
                return
        
            # otherwise: write a true 2D numeric matrix (n_rows x n_fields)
            mat = np.empty((n_rows, len(fields)), dtype=np.float32)
            for j, name in enumerate(fields):
                col = data_flat[name]
                mat[:, j] = np.asarray(col, dtype=np.float32)
        
            loom_out.write(mat, full_path)
            meta = LoomHandler._make_meta_for_written_dataset(
                loom_out, full_path, mat,
                n_cells=result["nber_cols"],
                n_genes=result["nber_rows"],
            )
            result["metadata"].append(meta)
            existing_paths.add(full_path)
            return

        # Case 2: mixed types -> write single 2D string matrix to SAME path
        if len(fields) == 1:
            col = np.asarray(data_flat[fields[0]]).reshape(-1)
            if col.dtype.kind in ("S", "O", "U"):
                col = np.array(
                    [v.decode(errors="ignore") if isinstance(v, (bytes, np.bytes_)) else str(v) for v in col],
                    dtype=object,
                )
            loom_out.write(col, full_path)
            meta = LoomHandler._make_meta_for_written_dataset(
                loom_out, full_path, col, n_cells=result["nber_cols"], n_genes=result["nber_rows"]
            )
            result["metadata"].append(meta)
            existing_paths.add(full_path)
            return

        mat = np.empty((n_rows, len(fields)), dtype=object)
        for j, name in enumerate(fields):
            col = np.asarray(data_flat[name]).reshape(-1)
            mat[:, j] = [
                v.decode(errors="ignore") if isinstance(v, (bytes, np.bytes_)) else str(v)
                for v in col
            ]

        loom_out.write(mat, full_path)
        meta = LoomHandler._make_meta_for_written_dataset(
            loom_out, full_path, mat, n_cells=result["nber_cols"], n_genes=result["nber_rows"]
        )
        result["metadata"].append(meta)
        existing_paths.add(full_path)
        return
    
    @staticmethod
    def _as_str_list(arr) -> list[str]:
        if isinstance(arr, np.ndarray):
            flat = arr.ravel()
        else:
            flat = arr
        out = []
        for v in flat:
            if isinstance(v, (bytes, np.bytes_)):
                out.append(v.decode(errors="ignore"))
            else:
                out.append(str(v))
        return out

    @staticmethod
    def _dataset_shape(node) -> tuple[int, ...]:
        if isinstance(node, h5py.Dataset):
            return tuple(int(x) for x in node.shape)
        return ()

    @staticmethod
    def _find_first_existing(f: h5py.File, paths: list[str]) -> str | None:
        for p in paths:
            if p in f:
                return p
        return None

    @staticmethod
    def _infer_cell_ids(f: h5py.File, n_cells: int) -> tuple[list[str], str]:
        """
        Returns (cell_ids, source_path_or_reason).
        Prefers /col_attrs/CellID, then common alternatives, otherwise creates Cell_1..Cell_n.
        """
        candidates = ["/col_attrs/CellID", "/col_attrs/cell_id", "/col_attrs/cell_ids", "/col_attrs/barcode", "/col_attrs/barcodes", "/col_attrs/Barcode", "/col_attrs/obs_names", "/col_attrs/index", "/col_attrs/_index" ]
        p = LoomHandler._find_first_existing(f, candidates)
        if p and isinstance(f[p], h5py.Dataset):
            vals = f[p][:]
            cell_ids = LoomHandler._as_str_list(vals)
            if len(cell_ids) == n_cells:
                return cell_ids, p

        # Fallback
        return [f"Cell_{i+1}" for i in range(n_cells)], "__generated__"

    @staticmethod
    def _infer_original_gene_names(f: h5py.File, n_genes: int) -> tuple[list[str], str]:
        """
        Returns (gene_names, source_path_or_reason).
        Prefers /row_attrs/Original_Gene then Name/Gene/Accession etc.
        """
        candidates = [ "/row_attrs/Original_Gene", "/row_attrs/Name", "/row_attrs/Gene", "/row_attrs/Accession", "/row_attrs/gene", "/row_attrs/gene_name", "/row_attrs/var_names", "/row_attrs/index", "/row_attrs/_index" ]
        p = LoomHandler._find_first_existing(f, candidates)
        if p and isinstance(f[p], h5py.Dataset):
            vals = f[p][:]
            genes = LoomHandler._as_str_list(vals)
            if len(genes) == n_genes:
                return genes, p

        # Fallback
        return [f"Gene_{i+1}" for i in range(n_genes)], "__generated__"

    @staticmethod
    def _path_orientation(path: str) -> str:
        if path.startswith("/col_attrs/"): return "CELL"
        if path.startswith("/row_attrs/"): return "GENE"
        if path == "/matrix" or path.startswith("/layers/"): return "EXPRESSION_MATRIX"
        if path.startswith("/attrs/"): return "GLOBAL"
        if path.startswith("/row_graphs/"): return "GENE"
        if path.startswith("/col_graphs/"): return "CELL"
        return "GLOBAL"

    @staticmethod
    def _make_meta_for_written_dataset(loom: LoomFile, path: str, value, *, n_cells: int, n_genes: int) -> "Metadata":
        """
        Builds a Metadata object for a just-written dataset at `path`.
        Uses `loom.get_dataset_size(path)` and tries to infer shape/orientation/type.
            Conventions enforced:
              - /col_attrs/* (CELL): nber_cols MUST equal n_cells
                  * 1D (n_cells,) => nber_rows=1, nber_cols=n_cells
                  * 2D (n_cells, k) => nber_rows=k, nber_cols=n_cells
                  * 2D (k, n_cells) => nber_rows=k, nber_cols=n_cells  (also accepted)
              - /row_attrs/* (GENE) (SYMMETRIC): nber_rows MUST equal n_genes
                  * 1D (n_genes,)         -> nber_rows=n_genes, nber_cols=1
                  * 2D (n_genes, k)       -> nber_rows=n_genes, nber_cols=k
                  * 2D (k, n_genes)       -> nber_rows=n_genes, nber_cols=k
              - /matrix and /layers (EXPRESSION_MATRIX): nber_rows=n_genes, nber_cols=n_cells
              - Scalars: 1x1
        """
        on = LoomHandler._path_orientation(path)

        # Infer shape
        arr = np.array(value) if not isinstance(value, np.ndarray) else value
        missing = None # only compute this for 1-D arrays
        if arr.ndim <= 1: missing = int(count_missing(arr))

        is_numeric = False
        try:
            is_numeric = np.issubdtype(np.array(arr).dtype, np.number)
        except Exception:
            is_numeric = False
    
        # Compute dims according to our rules
        nber_rows, nber_cols = LoomHandler._infer_meta_dims(arr, on, n_cells=n_cells, n_genes=n_genes)
    
        m = Metadata(
            name=path,
            on=on if on in ("CELL", "GENE", "GLOBAL") else "EXPRESSION_MATRIX",
            type=None,
            nber_rows=int(nber_rows),
            nber_cols=int(nber_cols),
            missing_values=missing,
            dataset_size=loom.get_dataset_size(path),
            imported=1,
        )
    
        # Categoricals only for 1D-like attrs
        if on in ("CELL", "GENE", "GLOBAL") and arr.ndim <= 1:
            vals = LoomHandler._as_str_list(arr)
            counts = Counter(vals)
            m.distinct_values = len(counts)
            m.categories = dict(counts)
            m.finalize_type(is_numeric=is_numeric)
        else:
            m.type = "NUMERIC" if is_numeric else "STRING"
    
        return m

    # -----------------------------------------
    # Core: copy matrix + compute stats on fly
    # -----------------------------------------
    @staticmethod
    def _copy_matrix_and_compute_stats( src_f: h5py.File, src_path: str, loom_out: LoomFile, dest_path: str, n_cells: int, n_genes: int, gene_metadata: list["Gene"] | None, block_cells: int = 1024 ) -> dict:
        """
        Copies a dense Loom matrix (Genes x Cells) from src_path to dest_path.
        Computes stats like LoomFile.write_matrix() did for H5AD (but input already is Genes x Cells).
        """
        if src_path not in src_f: ErrorJSON(f"Matrix path not found in Loom: {src_path}")
        node = src_f[src_path]
        if not isinstance(node, h5py.Dataset): ErrorJSON(f"Unsupported Loom matrix layout at {src_path}: expected a dataset")
        shape = LoomHandler._dataset_shape(node)
        if len(shape) != 2: ErrorJSON(f"Unsupported Loom matrix rank at {src_path}: expected 2D, got shape={shape}")
        if shape[0] != n_genes or shape[1] != n_cells:  ErrorJSON( f"Matrix dimension mismatch at {src_path}: expected ({n_genes}, {n_cells}) but found {shape}")

        # Create destination dataset (Genes x Cells)
        dset = loom_out.handle.create_dataset( dest_path, shape=(n_genes, n_cells), dtype="float32", chunks=(min(n_genes, 1024), min(n_cells, 1024)), compression="gzip")

        total_zeros = 0
        is_integer = True

        cell_depth = np.zeros(n_cells) if gene_metadata else None
        cell_detected = np.zeros(n_cells) if gene_metadata else None
        cell_mt = np.zeros(n_cells) if gene_metadata else None
        cell_rrna = np.zeros(n_cells) if gene_metadata else None
        cell_prot = np.zeros(n_cells) if gene_metadata else None
        gene_sum = np.zeros(n_genes) if gene_metadata else None

        mt_idx = ribo_idx = prot_idx = []
        if gene_metadata is not None:
            mt_idx = [i for i, g in enumerate(gene_metadata) if loom_out.is_mito(g.chr)]
            ribo_idx = [i for i, g in enumerate(gene_metadata) if loom_out.is_ribo(g.name)]
            prot_idx = [i for i, g in enumerate(gene_metadata) if loom_out.is_protein_coding(g.biotype)]

        # Process by cell blocks (columns)
        for start in range(0, n_cells, block_cells):
            end = min(start + block_cells, n_cells)

            # src block is genes x cells; convert to cells x genes for stat computation
            block_gxc = node[:, start:end]  # (n_genes, block_cells)
            # Convert to numpy array (h5py returns ndarray already)
            block_gxc = np.array(block_gxc, copy=False)

            # write as float32
            dset[:, start:end] = block_gxc.astype("float32", copy=False)

            # stats work on cells x genes
            chunk = block_gxc.T  # (block_cells, n_genes)

            total_zeros += int((chunk == 0).sum())
            if is_integer and not np.all(np.floor(chunk) == chunk): is_integer = False

            if gene_metadata is not None:
                depth = chunk.sum(axis=1)
                detected = (chunk > 0).sum(axis=1)

                cell_depth[start:end] = depth
                cell_detected[start:end] = detected

                # gene sums in original gene order (n_genes)
                gene_sum += chunk.sum(axis=0)

                if mt_idx: cell_mt[start:end] = chunk[:, mt_idx].sum(axis=1)
                if ribo_idx: cell_rrna[start:end] = chunk[:, ribo_idx].sum(axis=1)
                if prot_idx: cell_prot[start:end] = chunk[:, prot_idx].sum(axis=1)

        def to_percentage(subset_sum, total_sum):
            if subset_sum is None or total_sum is None: return None
            return np.divide(subset_sum, total_sum, out=np.zeros_like(subset_sum), where=total_sum != 0) * 100

        stats = {
            "cell_depth": cell_depth,
            "cell_detected": cell_detected,
            "cell_mt": to_percentage(cell_mt, cell_depth),
            "cell_rrna": to_percentage(cell_rrna, cell_depth),
            "cell_prot": to_percentage(cell_prot, cell_depth),
            "gene_sum": gene_sum,
            "total_zeros": int(total_zeros),
            "is_count_table": int(is_integer),
            "empty_cells": int(np.sum(cell_depth == 0)) if cell_depth is not None else None,
            "empty_genes": int(np.sum(gene_sum == 0)) if gene_sum is not None else None,
        }
        return stats

    # -----------------------------------------
    # Core: copy everything else (metadata/layers)
    # -----------------------------------------
    @staticmethod
    def _copy_all_other_datasets( src_f: h5py.File, loom_out: LoomFile, result: dict, existing_paths: set[str], skip_prefixes: tuple[str, ...] = ()) -> None:
        """
        Copies all datasets from src loom to dest loom, except those in existing_paths
        and except those under skip_prefixes (e.g. skip_prefixes=("/matrix",) if matrix handled).
        Adds Metadata entries for copied datasets.
        """
        def should_skip(full_path: str) -> bool:
            if full_path in existing_paths: return True
            for sp in skip_prefixes:
                if full_path == sp or full_path.startswith(sp + "/"): return True
            return False

        def visitor(name: str, obj):
            # h5py visititems provides names WITHOUT leading slash typically.
            full_path = "/" + name if not name.startswith("/") else name

            if isinstance(obj, h5py.Dataset):
                if should_skip(full_path): return

                # Read and write
                try:
                    if LoomHandler._is_compound_dataset(obj):
                        LoomHandler._write_compound_as_regular(src_obj=obj, loom_out=loom_out, result=result, existing_paths=existing_paths, full_path=full_path)
                        return

                    # Non compound
                    val = obj[()]
                    loom_out.write(val, full_path)

                    meta = LoomHandler._make_meta_for_written_dataset(loom_out, full_path, val, n_cells=result["nber_cols"], n_genes=result["nber_rows"])
                    result["metadata"].append(meta)
                    existing_paths.add(full_path)

                except Exception as e:
                    result.setdefault("warning", []).append(f"Could not copy dataset {full_path}: {e}")
                    
        src_f.visititems(visitor)

    # -----------------------------
    # Public entry point
    # -----------------------------
    @staticmethod
    def parse(args, file_path, gene_db: "MapGene"):
        result = {
            "detected_format": "LOOM",
            "input_path": file_path,
            "input_group": "",
            "output_path": args.output_path,
            "nber_rows": -1,             # genes
            "nber_cols": -1,             # cells
            "nber_not_found_genes": -1,
            "nber_zeros": -1,
            "is_count_table": -1,
            "empty_cols": -1,            # empty cells
            "empty_rows": -1,            # empty genes
            "dataset_size": -1,
            "metadata": [],
        }
        # Deal with previous warnings
        if getattr(args, "warnings", None):
            result["warning"] = list(args.warnings)

        with h5py.File(file_path, "r") as f:
            # 1) Pick main matrix
            if args.sel is not None:
                if args.sel not in f: ErrorJSON(f"Path '{args.sel}' (provided with --sel) not found in file")
                matrix_path = args.sel
            else:
                # Loom convention: /matrix
                candidates = ["/matrix"]
                matrix_path = LoomHandler._find_first_existing(f, candidates)
                if matrix_path is None: ErrorJSON(f"None of the candidate path were found: {candidates}. Please provide a path with --sel")

            result["input_group"] = matrix_path

            # 2) Get dimensions (genes x cells)
            node = f[matrix_path]
            if not isinstance(node, h5py.Dataset): ErrorJSON(f"Unsupported Loom main matrix at {matrix_path}: expected dataset")
            shape = LoomHandler._dataset_shape(node)
            if len(shape) != 2: ErrorJSON(f"Unsupported Loom main matrix rank at {matrix_path}: expected 2D, got shape={shape}")

            n_genes, n_cells = int(shape[0]), int(shape[1])
            result["nber_rows"] = n_genes
            result["nber_cols"] = n_cells

            # 3) Create output Loom
            loom = LoomFile(args.output_path)
            loom.ribo_protein_gene_names = getattr(gene_db, "ribo_protein_gene_names", set())

            # 4) Read cell + gene names from input loom (best-effort)
            obs_cells, obs_src = LoomHandler._infer_cell_ids(f, n_cells)
            var_genes, var_src = LoomHandler._infer_original_gene_names(f, n_genes)

            # 5) Write CellID and Original_Gene to output loom (+ metadata)
            loom.write(obs_cells, "/col_attrs/CellID")
            result["metadata"].append(Metadata(name="/col_attrs/CellID", on="CELL", type="STRING", nber_cols=len(obs_cells), nber_rows=1, distinct_values=len(set(obs_cells)), missing_values=count_missing(obs_cells), dataset_size=loom.get_dataset_size("/col_attrs/CellID"), imported=0))
            loom.write(var_genes, "/row_attrs/Original_Gene")
            result["metadata"].append(Metadata(name="/row_attrs/Original_Gene", on="GENE", type="STRING", nber_cols=1, nber_rows=len(var_genes), distinct_values=len(set(var_genes)), missing_values=count_missing(var_genes), dataset_size=loom.get_dataset_size("/row_attrs/Original_Gene"), imported=0))

            # 6) Map genes through DB, write gene annotations (same as H5ADHandler.parse)
            parsed_genes, result["nber_not_found_genes"] = gene_db.parse_genes_list(var_genes)

            ens_ids = [g.ensembl_id for g in parsed_genes]
            loom.write(ens_ids, "/row_attrs/Accession")
            result["metadata"].append(Metadata(name="/row_attrs/Accession", on="GENE", type="STRING", nber_cols=1, nber_rows=len(ens_ids), distinct_values=len(set(ens_ids)), missing_values=count_missing(ens_ids), dataset_size=loom.get_dataset_size("/row_attrs/Accession"), imported=0))

            gene_names = [g.name for g in parsed_genes]
            loom.write(gene_names, "/row_attrs/Gene")
            result["metadata"].append(Metadata(name="/row_attrs/Gene", on="GENE", type="STRING", nber_cols=1, nber_rows=len(gene_names), distinct_values=len(set(gene_names)), missing_values=count_missing(gene_names), dataset_size=loom.get_dataset_size("/row_attrs/Gene"), imported=0))

            loom.write(gene_names, "/row_attrs/Name")
            result["metadata"].append(Metadata(name="/row_attrs/Name", on="GENE", type="STRING", nber_cols=1, nber_rows=len(gene_names), distinct_values=len(set(gene_names)), missing_values=count_missing(gene_names), dataset_size=loom.get_dataset_size("/row_attrs/Name"), imported=0))

            biotypes = [g.biotype for g in parsed_genes]
            loom.write(biotypes, "/row_attrs/_Biotypes")
            result["metadata"].append(Metadata(name="/row_attrs/_Biotypes", on="GENE", type="DISCRETE", nber_cols=1, nber_rows=len(biotypes), distinct_values=len(set(biotypes)), missing_values=count_missing(biotypes), dataset_size=loom.get_dataset_size("/row_attrs/_Biotypes"), imported=0, categories=dict(Counter(biotypes))))

            chroms = [g.chr for g in parsed_genes]
            loom.write(chroms, "/row_attrs/_Chromosomes")
            result["metadata"].append(Metadata(name="/row_attrs/_Chromosomes", on="GENE", type="DISCRETE", nber_cols=1, nber_rows=len(chroms), distinct_values=len(set(chroms)), missing_values=count_missing(chroms), dataset_size=loom.get_dataset_size("/row_attrs/_Chromosomes"), imported=0, categories=dict(Counter(chroms))))

            sum_exon_length = [g.sum_exon_length for g in parsed_genes]
            loom.write(sum_exon_length, "/row_attrs/_SumExonLength")
            result["metadata"].append(Metadata(name="/row_attrs/_SumExonLength", on="GENE", type="NUMERIC", nber_cols=1, nber_rows=len(sum_exon_length), distinct_values=len(set(sum_exon_length)), missing_values=count_missing(sum_exon_length), dataset_size=loom.get_dataset_size("/row_attrs/_SumExonLength"), imported=0))

            # 7) Copy main matrix + compute stats (on the fly)
            stats = LoomHandler._copy_matrix_and_compute_stats(
                src_f=f,
                src_path=matrix_path,
                loom_out=loom,
                dest_path="/matrix",
                n_cells=n_cells,
                n_genes=n_genes,
                gene_metadata=parsed_genes,
            )

            result["dataset_size"] = loom.get_dataset_size("/matrix")
            result["nber_zeros"] = int(stats["total_zeros"])
            result["empty_cols"] = int(stats["empty_cells"])
            result["empty_rows"] = int(stats["empty_genes"])
            result["is_count_table"] = int(stats["is_count_table"])

            # 8) Write computed vectors (same paths as H5ADHandler.parse)
            loom.write(stats["cell_depth"], "/col_attrs/_Depth")
            result["metadata"].append(Metadata(name="/col_attrs/_Depth", on="CELL", type="NUMERIC", nber_cols=len(stats["cell_depth"]), nber_rows=1, distinct_values=len(set(stats["cell_depth"])), missing_values=count_missing(stats["cell_depth"]), dataset_size=loom.get_dataset_size("/col_attrs/_Depth"), imported=0))

            loom.write(stats["cell_detected"], "/col_attrs/_Detected_Genes")
            result["metadata"].append(Metadata(name="/col_attrs/_Detected_Genes", on="CELL", type="NUMERIC", nber_cols=len(stats["cell_detected"]), nber_rows=1, distinct_values=len(set(stats["cell_detected"])), missing_values=count_missing(stats["cell_detected"]), dataset_size=loom.get_dataset_size("/col_attrs/_Detected_Genes"), imported=0))

            loom.write(stats["cell_mt"], "/col_attrs/_Mitochondrial_Content")
            result["metadata"].append(Metadata(name="/col_attrs/_Mitochondrial_Content", on="CELL", type="NUMERIC", nber_cols=len(stats["cell_mt"]), nber_rows=1, distinct_values=len(set(stats["cell_mt"])), missing_values=count_missing(stats["cell_mt"]), dataset_size=loom.get_dataset_size("/col_attrs/_Mitochondrial_Content"), imported=0))

            loom.write(stats["cell_rrna"], "/col_attrs/_Ribosomal_Content")
            result["metadata"].append(Metadata(name="/col_attrs/_Ribosomal_Content", on="CELL", type="NUMERIC", nber_cols=len(stats["cell_rrna"]), nber_rows=1, distinct_values=len(set(stats["cell_rrna"])), missing_values=count_missing(stats["cell_rrna"]), dataset_size=loom.get_dataset_size("/col_attrs/_Ribosomal_Content"), imported=0))

            loom.write(stats["cell_prot"], "/col_attrs/_Protein_Coding_Content")
            result["metadata"].append(Metadata(name="/col_attrs/_Protein_Coding_Content", on="CELL", type="NUMERIC", nber_cols=len(stats["cell_prot"]), nber_rows=1, distinct_values=len(set(stats["cell_prot"])), missing_values=count_missing(stats["cell_prot"]), dataset_size=loom.get_dataset_size("/col_attrs/_Protein_Coding_Content"), imported=0))

            loom.write(stats["gene_sum"], "/row_attrs/_Sum")
            result["metadata"].append(Metadata(name="/row_attrs/_Sum", on="GENE", type="NUMERIC", nber_cols=1, nber_rows=len(stats["gene_sum"]), distinct_values=len(set(stats["gene_sum"])), missing_values=count_missing(stats["gene_sum"]), dataset_size=loom.get_dataset_size("/row_attrs/_Sum"), imported=0))

            # 9) Stable IDs
            stable_ids_rows = list(range(n_genes))
            loom.write(stable_ids_rows, "/row_attrs/_StableID")
            result["metadata"].append(Metadata(name="/row_attrs/_StableID", on="GENE", type="NUMERIC", nber_cols=1, nber_rows=len(stable_ids_rows), distinct_values=len(set(stable_ids_rows)), missing_values=count_missing(stable_ids_rows), dataset_size=loom.get_dataset_size("/row_attrs/_StableID"), imported=0))

            stable_ids_cols = list(range(n_cells))
            loom.write(stable_ids_cols, "/col_attrs/_StableID")
            result["metadata"].append(Metadata(name="/col_attrs/_StableID", on="CELL", type="NUMERIC", nber_cols=len(stable_ids_cols), nber_rows=1, distinct_values=len(set(stable_ids_cols)), missing_values=count_missing(stable_ids_cols), dataset_size=loom.get_dataset_size("/col_attrs/_StableID"), imported=0))

            # 10) Copy all remaining content from source loom into new loom (matrices + metadata)
            existing_paths = {m.name for m in result["metadata"]}
            existing_paths.add("/attrs/LOOM_SPEC_VERSION")  # created by LoomFile

            # We already handled /matrix explicitly.
            # We also intentionally overwrite (with DB-derived values) several standard row/col attrs,
            # so keep them in existing_paths.
            LoomHandler._copy_all_other_datasets(
                src_f=f,
                loom_out=loom,
                result=result,
                existing_paths=existing_paths,
                skip_prefixes=("/matrix",),  # do not duplicate
            )

            loom.close()

        # 11) Emit JSON
        json_str = json.dumps(result, default=dataclass_to_json, ensure_ascii=False)
        if args.o:
            with open(os.path.join(args.o, "output.json"), "w", encoding="utf-8") as out:
                out.write(json_str)
        else:
            print(json_str)

class H510xHandler:
    """
    Parses a 10x Genomics HDF5 Feature-Barcode Matrix (.h5) and writes a Loom v3 file to args.output_path.

    Mirrors H5ADHandler.parse + LoomHandler.parse behavior:
      - select input group (root-level genome group) via --sel or auto if unique
      - write /col_attrs/CellID and /row_attrs/Original_Gene
      - map genes via DB and write Accession/Name/Biotypes/Chromosomes/etc.
      - write /matrix (Genes x Cells) with chunked processing and compute stats on the fly:
          _Depth, _Detected_Genes, _Mitochondrial_Content, _Ribosomal_Content, _Protein_Coding_Content, _Sum
      - write _StableID vectors for rows/cols
      - transfer extra 10x feature annotations (feature_type, genome, etc.) into /row_attrs/*
      - emit JSON summary with Metadata entries (same structure as other handlers)
    """

    @staticmethod
    def _decode_list(arr) -> list[str]:
        out = []
        for v in arr:
            if isinstance(v, (bytes, np.bytes_, np.void)):
                try:
                    out.append(v.decode(errors="ignore"))
                except Exception:
                    out.append(str(v))
            else:
                out.append(str(v))
        return out

    @staticmethod
    def _find_10x_groups(f: h5py.File) -> list[str]:
        """
        Returns list of root-level group names that look like 10x matrices.
        Your preparse detection: {'barcodes','data','indices','indptr','shape'} subset.
        """
        out = []
        for key in f.keys():
            if isinstance(f[key], h5py.Group):
                grp = f[key]
                if {'barcodes', 'data', 'indices', 'indptr', 'shape'}.issubset(grp.keys()):
                    out.append(key)
        return out

    @staticmethod
    def _pick_group(f: h5py.File, sel: str | None) -> str:
        candidates = H510xHandler._find_10x_groups(f)
        if sel is not None:
            if sel in f and isinstance(f[sel], h5py.Group):
                # accept exact match even if not in candidates (but validate)
                grp = f[sel]
                if {'barcodes', 'data', 'indices', 'indptr', 'shape'}.issubset(grp.keys()):
                    return sel
                ErrorJSON(f"Path '{sel}' (provided with --sel) is not a valid 10x matrix group (missing required datasets).")
            ErrorJSON(f"Path '{sel}' (provided with --sel) not found in file")
        # auto-pick
        if len(candidates) == 0:
            ErrorJSON("No valid 10x matrix group found at root level. Please provide --sel with the correct group name.")
        if len(candidates) > 1:
            ErrorJSON(f"Multiple 10x matrix groups found: {candidates}. Please provide a group with --sel")
        return candidates[0]

    @staticmethod
    def _read_barcodes(grp: h5py.Group) -> list[str]:
        if "barcodes" not in grp:
            ErrorJSON("10x group missing 'barcodes'")
        return H510xHandler._decode_list(grp["barcodes"][:])

    @staticmethod
    def _read_feature_ids_and_names(grp: h5py.Group, n_genes: int) -> tuple[list[str], dict[str, np.ndarray]]:
        """
        Returns:
          - var_genes: preferred original gene names (for /row_attrs/Original_Gene)
          - extra_fields: dict of other feature-level vectors to optionally transfer to /row_attrs/*
        Handles both older (gene_names / genes) and v3 (features/*) layouts.
        """
        extra_fields: dict[str, np.ndarray] = {}

        # ---- Older 10x layouts ----
        if "gene_names" in grp:
            names = grp["gene_names"][:]
            var_genes = H510xHandler._decode_list(names)
            if len(var_genes) != n_genes:
                # best-effort truncate/pad
                var_genes = (var_genes + [f"Gene_{i+1}" for i in range(len(var_genes), n_genes)])[:n_genes]
            return var_genes, extra_fields

        if "genes" in grp:
            # sometimes it's Ensembl IDs or symbols depending on producer
            ids = grp["genes"][:]
            var_genes = H510xHandler._decode_list(ids)
            if len(var_genes) != n_genes:
                var_genes = (var_genes + [f"Gene_{i+1}" for i in range(len(var_genes), n_genes)])[:n_genes]
            return var_genes, extra_fields

        # ---- 10x v3+ 'features' group ----
        if "features" in grp and isinstance(grp["features"], h5py.Group):
            fg = grp["features"]

            # Candidate datasets for "original gene label"
            # Prefer features/name if available; fall back to features/id.
            if "name" in fg:
                var_genes = H510xHandler._decode_list(fg["name"][:])
            elif "id" in fg:
                var_genes = H510xHandler._decode_list(fg["id"][:])
            else:
                var_genes = [f"Gene_{i+1}" for i in range(n_genes)]

            # Collect extra feature annotations for transfer (length == n_genes)
            for k in fg.keys():
                if k in {"name", "id"}:
                    continue
                node = fg[k]
                if isinstance(node, h5py.Dataset):
                    try:
                        arr = node[:]
                        if isinstance(arr, np.ndarray) and arr.shape[0] == n_genes:
                            # decode bytes arrays to str arrays if needed
                            if arr.dtype.kind in ("S", "O", "U"):
                                arr = np.array(
                                    [v.decode(errors="ignore") if isinstance(v, (bytes, np.bytes_)) else str(v) for v in arr],
                                    dtype=object,
                                )
                            extra_fields[k] = arr
                    except Exception:
                        pass

            # Ensure length
            if len(var_genes) != n_genes:
                var_genes = (var_genes + [f"Gene_{i+1}" for i in range(len(var_genes), n_genes)])[:n_genes]

            return var_genes, extra_fields

        # ---- Fallback ----
        return [f"Gene_{i+1}" for i in range(n_genes)], extra_fields

    @staticmethod
    def _write_10x_matrix_and_compute_stats(grp: h5py.Group, loom: LoomFile, dest_path: str, n_cells: int, n_genes: int, gene_metadata: list["Gene"] | None) -> dict:
        """
        10x H5 stores a CSC matrix with:
          - shape = (n_genes, n_cells)
          - indptr length = n_cells + 1  (columns = cells)
          - indices are row indices (genes)
        We write /matrix as Genes x Cells and compute stats using chunk over cells.
        """
        # Validate
        for k in ("data", "indices", "indptr", "shape"):
            if k not in grp:
                ErrorJSON(f"10x group missing '{k}'")

        shape = tuple(int(x) for x in grp["shape"][:])
        if len(shape) != 2:
            ErrorJSON(f"10x 'shape' is not 2D: {shape}")
        # shape is (genes, cells)
        if shape[0] != n_genes or shape[1] != n_cells:
            ErrorJSON(f"10x matrix dimension mismatch: expected ({n_genes},{n_cells}) found {shape}")

        indptr = grp["indptr"]
        indices = grp["indices"]
        data = grp["data"]

        if int(indptr.shape[0]) != n_cells + 1:
            # Some exotic producers might transpose; we keep strict to avoid silent corruption
            ErrorJSON(f"Unexpected 10x indptr length={int(indptr.shape[0])}. Expected n_cells+1={n_cells+1} (CSC with columns=cells).")

        # Prepare Loom dataset (Genes x Cells)
        dset = loom.handle.create_dataset(
            dest_path,
            shape=(n_genes, n_cells),
            dtype="float32",
            chunks=(min(n_genes, 1024), min(n_cells, 1024)),
            compression="gzip",
        )

        total_zeros = 0
        is_integer = True

        cell_depth = np.zeros(n_cells) if gene_metadata else None
        cell_detected = np.zeros(n_cells) if gene_metadata else None
        cell_mt = np.zeros(n_cells) if gene_metadata else None
        cell_rrna = np.zeros(n_cells) if gene_metadata else None
        cell_prot = np.zeros(n_cells) if gene_metadata else None
        gene_sum = np.zeros(n_genes) if gene_metadata else None

        mt_idx = ribo_idx = prot_idx = []
        if gene_metadata is not None:
            mt_idx = [i for i, g in enumerate(gene_metadata) if loom.is_mito(g.chr)]
            ribo_idx = [i for i, g in enumerate(gene_metadata) if loom.is_ribo(g.name)]
            prot_idx = [i for i, g in enumerate(gene_metadata) if loom.is_protein_coding(g.biotype)]

        # chunk over cells
        block_cells = 1024
        for start in range(0, n_cells, block_cells):
            end = min(start + block_cells, n_cells)

            ptr = indptr[start:end + 1]
            # ptr is a numpy array (or array-like)
            ptr0 = int(ptr[0])
            ptrN = int(ptr[-1])

            # Slice CSC pieces
            data_slice = data[ptr0:ptrN]
            idx_slice = indices[ptr0:ptrN]
            # Local indptr for this block
            local_indptr = (ptr - ptr0).astype(np.int64, copy=False)

            # Build CSC (genes x block_cells)
            block = sp.csc_matrix(
                (data_slice, idx_slice, local_indptr),
                shape=(n_genes, end - start),
            )

            # Convert to dense for writing/stats
            block_gxc = block.toarray().astype(np.float32, copy=False)  # (genes, block_cells)
            dset[:, start:end] = block_gxc

            # stats need cells x genes
            chunk = block_gxc.T  # (block_cells, genes)

            # zeros: count over dense chunk
            total_zeros += int((chunk == 0).sum())

            if is_integer and not np.all(np.floor(chunk) == chunk):
                is_integer = False

            if gene_metadata is not None:
                depth = chunk.sum(axis=1)
                detected = (chunk > 0).sum(axis=1)

                cell_depth[start:end] = depth
                cell_detected[start:end] = detected

                gene_sum += chunk.sum(axis=0)

                if mt_idx:
                    cell_mt[start:end] = chunk[:, mt_idx].sum(axis=1)
                if ribo_idx:
                    cell_rrna[start:end] = chunk[:, ribo_idx].sum(axis=1)
                if prot_idx:
                    cell_prot[start:end] = chunk[:, prot_idx].sum(axis=1)

        def to_percentage(subset_sum, total_sum):
            if subset_sum is None or total_sum is None:
                return None
            return np.divide(subset_sum, total_sum, out=np.zeros_like(subset_sum), where=total_sum != 0) * 100

        stats = {
            "cell_depth": cell_depth,
            "cell_detected": cell_detected,
            "cell_mt": to_percentage(cell_mt, cell_depth),
            "cell_rrna": to_percentage(cell_rrna, cell_depth),
            "cell_prot": to_percentage(cell_prot, cell_depth),
            "gene_sum": gene_sum,
            "total_zeros": int(total_zeros),
            "is_count_table": int(is_integer),
            "empty_cells": int(np.sum(cell_depth == 0)) if cell_depth is not None else None,
            "empty_genes": int(np.sum(gene_sum == 0)) if gene_sum is not None else None,
        }
        return stats

    @staticmethod
    def parse(args, file_path, gene_db: "MapGene"):
        result = {
            "detected_format": "H5_10x",
            "input_path": file_path,
            "input_group": "",
            "output_path": args.output_path,
            "nber_rows": -1,             # genes
            "nber_cols": -1,             # cells
            "nber_not_found_genes": -1,
            "nber_zeros": -1,
            "is_count_table": -1,
            "empty_cols": -1,            # empty cells
            "empty_rows": -1,            # empty genes
            "dataset_size": -1,
            "metadata": [],
        }
        # Deal with previous warnings
        if getattr(args, "warnings", None):
            result["warning"] = list(args.warnings)

        with h5py.File(file_path, "r") as f:
            # 1) Select group
            group_name = H510xHandler._pick_group(f, args.sel)
            result["input_group"] = group_name
            grp = f[group_name]

            # 2) Dimensions
            n_genes = int(grp["shape"][0])
            n_cells = int(grp["shape"][1])
            result["nber_rows"] = n_genes
            result["nber_cols"] = n_cells

            # 3) Create output Loom
            loom = LoomFile(args.output_path)
            loom.ribo_protein_gene_names = getattr(gene_db, "ribo_protein_gene_names", set())

            # 4) Read barcodes + feature names
            obs_cells = H510xHandler._read_barcodes(grp)
            if len(obs_cells) != n_cells:
                obs_cells = (obs_cells + [f"Cell_{i+1}" for i in range(len(obs_cells), n_cells)])[:n_cells]

            var_genes, feature_extra = H510xHandler._read_feature_ids_and_names(grp, n_genes)

            # 5) Write CellID and Original_Gene
            loom.write(obs_cells, "/col_attrs/CellID")
            result["metadata"].append(Metadata(name="/col_attrs/CellID", on="CELL", type="STRING", nber_cols=len(obs_cells), nber_rows=1, distinct_values=len(set(obs_cells)), missing_values=count_missing(obs_cells), dataset_size=loom.get_dataset_size("/col_attrs/CellID"), imported=0))

            loom.write(var_genes, "/row_attrs/Original_Gene")
            result["metadata"].append(Metadata(name="/row_attrs/Original_Gene", on="GENE", type="STRING", nber_cols=1, nber_rows=len(var_genes), distinct_values=len(set(var_genes)), missing_values=count_missing(var_genes), dataset_size=loom.get_dataset_size("/row_attrs/Original_Gene"), imported=0))

            # 6) Map genes through DB, write gene annotations (same as H5AD + Loom handlers)
            parsed_genes, result["nber_not_found_genes"] = gene_db.parse_genes_list(var_genes)

            ens_ids = [g.ensembl_id for g in parsed_genes]
            loom.write(ens_ids, "/row_attrs/Accession")
            result["metadata"].append(Metadata(name="/row_attrs/Accession", on="GENE", type="STRING", nber_cols=1, nber_rows=len(ens_ids), distinct_values=len(set(ens_ids)), missing_values=count_missing(ens_ids), dataset_size=loom.get_dataset_size("/row_attrs/Accession"), imported=0))

            gene_names = [g.name for g in parsed_genes]
            loom.write(gene_names, "/row_attrs/Gene")  # backward compatibility
            result["metadata"].append(Metadata(name="/row_attrs/Gene", on="GENE", type="STRING", nber_cols=1, nber_rows=len(gene_names), distinct_values=len(set(gene_names)), missing_values=count_missing(gene_names), dataset_size=loom.get_dataset_size("/row_attrs/Gene"), imported=0))

            loom.write(gene_names, "/row_attrs/Name")
            result["metadata"].append(Metadata(name="/row_attrs/Name", on="GENE", type="STRING", nber_cols=1, nber_rows=len(gene_names), distinct_values=len(set(gene_names)), missing_values=count_missing(gene_names), dataset_size=loom.get_dataset_size("/row_attrs/Name"), imported=0))

            biotypes = [g.biotype for g in parsed_genes]
            loom.write(biotypes, "/row_attrs/_Biotypes")
            result["metadata"].append(Metadata(name="/row_attrs/_Biotypes", on="GENE", type="DISCRETE", nber_cols=1, nber_rows=len(biotypes), distinct_values=len(set(biotypes)), missing_values=count_missing(biotypes), dataset_size=loom.get_dataset_size("/row_attrs/_Biotypes"), imported=0, categories=dict(Counter(biotypes)), )
            )

            chroms = [g.chr for g in parsed_genes]
            loom.write(chroms, "/row_attrs/_Chromosomes")
            result["metadata"].append(Metadata(name="/row_attrs/_Chromosomes", on="GENE", type="DISCRETE", nber_cols=1, nber_rows=len(chroms), distinct_values=len(set(chroms)), missing_values=count_missing(chroms), dataset_size=loom.get_dataset_size("/row_attrs/_Chromosomes"), imported=0, categories=dict(Counter(chroms)), )
            )

            sum_exon_length = [g.sum_exon_length for g in parsed_genes]
            loom.write(sum_exon_length, "/row_attrs/_SumExonLength")
            result["metadata"].append(Metadata(name="/row_attrs/_SumExonLength", on="GENE", type="NUMERIC", nber_cols=1, nber_rows=len(sum_exon_length), distinct_values=len(set(sum_exon_length)), missing_values=count_missing(sum_exon_length), dataset_size=loom.get_dataset_size("/row_attrs/_SumExonLength"), imported=0))

            # 7) Write main matrix + compute stats
            stats = H510xHandler._write_10x_matrix_and_compute_stats(grp=grp, loom=loom, dest_path="/matrix", n_cells=n_cells, n_genes=n_genes, gene_metadata=parsed_genes)

            result["dataset_size"] = loom.get_dataset_size("/matrix")
            result["nber_zeros"] = int(stats["total_zeros"])
            result["empty_cols"] = int(stats["empty_cells"])
            result["empty_rows"] = int(stats["empty_genes"])
            result["is_count_table"] = int(stats["is_count_table"])

            # 8) Write computed vectors
            loom.write(stats["cell_depth"], "/col_attrs/_Depth")
            result["metadata"].append(Metadata(name="/col_attrs/_Depth", on="CELL", type="NUMERIC", nber_cols=len(stats["cell_depth"]), nber_rows=1, distinct_values=len(set(stats["cell_depth"])), missing_values=count_missing(stats["cell_depth"]), dataset_size=loom.get_dataset_size("/col_attrs/_Depth"), imported=0))

            loom.write(stats["cell_detected"], "/col_attrs/_Detected_Genes")
            result["metadata"].append(Metadata(name="/col_attrs/_Detected_Genes", on="CELL", type="NUMERIC", nber_cols=len(stats["cell_detected"]), nber_rows=1, distinct_values=len(set(stats["cell_detected"])), missing_values=count_missing(stats["cell_detected"]), dataset_size=loom.get_dataset_size("/col_attrs/_Detected_Genes"), imported=0))

            loom.write(stats["cell_mt"], "/col_attrs/_Mitochondrial_Content")
            result["metadata"].append(Metadata(name="/col_attrs/_Mitochondrial_Content", on="CELL", type="NUMERIC", nber_cols=len(stats["cell_mt"]), nber_rows=1, distinct_values=len(set(stats["cell_mt"])), missing_values=count_missing(stats["cell_mt"]), dataset_size=loom.get_dataset_size("/col_attrs/_Mitochondrial_Content"), imported=0))

            loom.write(stats["cell_rrna"], "/col_attrs/_Ribosomal_Content")
            result["metadata"].append(Metadata(name="/col_attrs/_Ribosomal_Content", on="CELL", type="NUMERIC", nber_cols=len(stats["cell_rrna"]), nber_rows=1, distinct_values=len(set(stats["cell_rrna"])), missing_values=count_missing(stats["cell_rrna"]), dataset_size=loom.get_dataset_size("/col_attrs/_Ribosomal_Content"), imported=0))

            loom.write(stats["cell_prot"], "/col_attrs/_Protein_Coding_Content")
            result["metadata"].append(Metadata(name="/col_attrs/_Protein_Coding_Content", on="CELL", type="NUMERIC", nber_cols=len(stats["cell_prot"]), nber_rows=1, distinct_values=len(set(stats["cell_prot"])), missing_values=count_missing(stats["cell_prot"]), dataset_size=loom.get_dataset_size("/col_attrs/_Protein_Coding_Content"), imported=0))

            loom.write(stats["gene_sum"], "/row_attrs/_Sum")
            result["metadata"].append(Metadata(name="/row_attrs/_Sum", on="GENE", type="NUMERIC", nber_cols=1, nber_rows=len(stats["gene_sum"]), distinct_values=len(set(stats["gene_sum"])), missing_values=count_missing(stats["gene_sum"]), dataset_size=loom.get_dataset_size("/row_attrs/_Sum"), imported=0))

            # 9) Stable IDs
            stable_ids_rows = list(range(n_genes))
            loom.write(stable_ids_rows, "/row_attrs/_StableID")
            result["metadata"].append(Metadata(name="/row_attrs/_StableID", on="GENE", type="NUMERIC", nber_cols=1, nber_rows=len(stable_ids_rows), distinct_values=len(set(stable_ids_rows)), missing_values=count_missing(stable_ids_rows), dataset_size=loom.get_dataset_size("/row_attrs/_StableID"), imported=0))

            stable_ids_cols = list(range(n_cells))
            loom.write(stable_ids_cols, "/col_attrs/_StableID")
            result["metadata"].append(Metadata(name="/col_attrs/_StableID", on="CELL", type="NUMERIC", nber_cols=len(stable_ids_cols), nber_rows=1, distinct_values=len(set(stable_ids_cols)), missing_values=count_missing(stable_ids_cols), dataset_size=loom.get_dataset_size("/col_attrs/_StableID"), imported=0))

            # 10) Transfer extra feature-level annotations (10x v3 features/*) as imported metadata
            existing_paths = {m.name for m in result["metadata"]}
            for k, arr in feature_extra.items():
                loom_path = f"/row_attrs/{k}"
                if loom_path in existing_paths:
                    continue
                try:
                    loom.write(arr, loom_path)

                    # Build metadata similarly to transfer_metadata logic
                    meta = Metadata(name=loom_path, on="GENE", nber_rows=n_genes, nber_cols=1, missing_values=count_missing(arr), imported=1, dataset_size=loom.get_dataset_size(loom_path))

                    # categorical counts only for 1D
                    if np.asarray(arr).ndim <= 1:
                        # Convert to strings for counting
                        vals = arr
                        if np.asarray(vals).dtype.kind in ("S", "U", "O"):
                            vals = np.array([v.decode(errors="ignore") if isinstance(v, (bytes, np.bytes_)) else str(v) for v in vals], dtype=object)
                        counts = Counter(list(vals))
                        meta.distinct_values = len(counts)
                        meta.categories = dict(counts)
                        is_numeric = np.issubdtype(np.array(vals).dtype, np.number)
                        meta.finalize_type(is_numeric)
                    else:
                        meta.type = "NUMERIC" if np.issubdtype(np.array(arr).dtype, np.number) else "STRING"

                    result["metadata"].append(meta)
                    existing_paths.add(loom_path)
                except Exception as e:
                    result.setdefault("warning", []).append(f"Could not transfer 10x feature field '{k}': {e}")

            loom.close()

        # 11) Emit JSON
        json_str = json.dumps(result, default=dataclass_to_json, ensure_ascii=False)
        if args.o:
            with open(os.path.join(args.o, "output.json"), "w", encoding="utf-8") as out:
                out.write(json_str)
        else:
            print(json_str)

class RdsHandler:
    @staticmethod
    def parse(args, file_path, gene_db):
        # build the JSON-able structure
        result = {
            "detected_format": "RDS",
            "input_path": file_path,
            "output_path": args.output_path,
        }

        # Serialize to JSON
        json_str = json.dumps(result, ensure_ascii=False)
        
        # Either write to file or print
        if args.o:
            with open(os.path.join(args.o, "output.json"), "w", encoding="utf-8") as out:
                out.write(json_str)
        else:
            print(json_str)

class MtxHandler:
    @staticmethod
    def parse(args, file_path, gene_db):
        # build the JSON-able structure
        result = {
            "detected_format": "MTX",
            "input_path": file_path,
            "output_path": args.output_path,
        }

        # Serialize to JSON
        json_str = json.dumps(result, ensure_ascii=False)
        
        # Either write to file or print
        if args.o:
            with open(os.path.join(args.o, "output.json"), "w", encoding="utf-8") as out:
                out.write(json_str)
        else:
            print(json_str)

class TextHandler:
    @staticmethod
    def parse(args, file_path, gene_db):
        # build the JSON-able structure
        result = {
            "detected_format": "RAW_TEXT",
            "input_path": file_path,
            "output_path": args.output_path,
        }

        # Serialize to JSON
        json_str = json.dumps(result, ensure_ascii=False)
        
        # Either write to file or print
        if args.o:
            with open(os.path.join(args.o, "output.json"), "w", encoding="utf-8") as out:
                out.write(json_str)
        else:
            print(json_str)

### Error handling ###

class ErrorJSON(Exception):
    def __init__(self, message: str, output: str = None):
        # still call Exception init so tracebacks, etc. work if you ever raise()
        super().__init__(message)
        payload = {"displayed_error": message}

        if output:
            # write JSON to the given file
            with open(output, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        else:
            # or print it to stdout
            print(json.dumps(payload, ensure_ascii=False), file=sys.stdout)

        # then immediately exit with an error code
        sys.exit(1)

### File handling ###
class LoomFile:
    handle = None
    
    def __init__(self, fileName):
        self.loomPath = Path(fileName)
        if self.loomPath.exists():
            self.loomPath.unlink() # Overwrite
        self.handle = h5py.File(self.loomPath, "w")
        self._ensure_loom_v3_layout()
        self.ribo_protein_gene_names: set[str] = set()
        
    def _ensure_loom_v3_layout(self) -> None:
        """Create required Loom v3 groups/datasets if missing."""
        # Required groups (Loom spec expects these paths)
        self.handle.require_group("attrs")       # /attrs
        self.handle.require_group("col_attrs")   # /col_attrs
        self.handle.require_group("row_attrs")   # /row_attrs
        self.handle.require_group("layers")      # /layers (optional in spec, but you want it)
        self.handle.require_group("row_graphs")  # /row_graphs (required group, can be empty)
        self.handle.require_group("col_graphs")  # /col_graphs (required group, can be empty)
        self.handle["attrs"].create_dataset("LOOM_SPEC_VERSION", data="3.0.0", dtype=h5py.string_dtype(encoding="utf-8")) # Use variable-length UTF-8 strings

    def get_dataset_size(self, path: str) -> int:
        """ Returns the storage size of the dataset at 'path' in bytes. """
        if path in self.handle:
            return self.handle[path].id.get_storage_size()
        return 0

    @staticmethod
    def _detect_encoding(node, n_cells, n_genes) -> str:
        # 1) Try attr
        enc = node.attrs.get("encoding-type", None)
        if isinstance(enc, bytes):
            enc = enc.decode()
        enc = str(enc) if enc is not None else None
    
        # 2) Validate / infer from indptr length
        indptr_len = int(node["indptr"].shape[0])
        if indptr_len == n_cells + 1:
            inferred = "csr_matrix"
        elif indptr_len == n_genes + 1:
            inferred = "csc_matrix"
        else:
            ErrorJSON( f"Cannot infer sparse encoding: len(indptr)={indptr_len}, expected {n_cells+1} (CSR) or {n_genes+1} (CSC)" )
    
        # If attr exists but disagrees, prefer inferred (and warn)
        if enc and enc != inferred:
            # optional: log warning instead of error
            # print(f"WARNING: encoding-type attr={enc} but inferred={inferred}; using inferred")
            return inferred
    
        return inferred

    @staticmethod
    def is_mito(chr_name: str | None) -> bool:
        if chr_name is None: return False
        s = str(chr_name).strip()
        if not s: return False
    
        s_up = s.upper()
        if s_up in {"MT", "M", "MITOCHONDRION_GENOME", "DMEL_MITOCHONDRION_GENOME"}: return True
        s_up = s_up.replace("CHROMOSOME", "").replace("CHR", "").replace("_", "").replace("-", "").strip()
        if s_up in {"MT", "M"}: return True
        if "MITO" in s_up: return True
    
        return False

    @staticmethod
    def _norm_gene_name(x: str | None) -> str:
        if x is None:
            return ""
        if isinstance(x, (bytes, np.bytes_)):
            x = x.decode(errors="ignore")
        return str(x).strip().upper()
    
    def is_ribo(self, gene_name: str | None) -> bool:
        """
        Ribosomal *protein* genes only, defined by GO:0003735-derived set.
        If the set is empty, returns False (no ribo-content computed).
        """
        if not self.ribo_protein_gene_names:
            return False
        return self._norm_gene_name(gene_name) in self.ribo_protein_gene_names

    @staticmethod
    def is_protein_coding(biotype: str | None) -> bool:
        if biotype is None: return False
        s = str(biotype).strip()
        if not s: return False
    
        s_up = s.upper()
        if s_up in {"PROTEIN_CODING"}: return True
        if s_up.startswith("PROTEIN_CODING"): return True
        
        return False

    def write(self, data, path: str) -> None:
        try:
            # 1. Convert to numpy and handle strings
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            
            if data.dtype.kind in ('U', 'S', 'O'):
                data = data.astype('O') 
                dtype = h5py.string_dtype(encoding='utf-8')
            else:
                dtype = None 

            # 2. Delete existing path to allow overwriting
            if path in self.handle:
                del self.handle[path]
            
            # 3. Handle Scalar vs Dataset
            # If data.ndim is 0, it's a scalar (e.g., a single float or string)
            if data.ndim == 0:
                self.handle.create_dataset(path, data=data, dtype=dtype)
            else:
                # Only use compression for arrays with 1 or more dimensions
                self.handle.create_dataset(
                    path, 
                    data=data, 
                    dtype=dtype, 
                    compression="gzip"
                )
        except Exception as e:
            ErrorJSON(f"Error writing to {path} with dtype {data.dtype}: {e}")

    def write_matrix(self, f, src_path, dest_path, n_cells, n_genes, gene_metadata):
        """
        Writes the matrix from H5AD to Loom /matrix.
        Computes statistics on the fly to save RAM.
        """
        
        # 1. Prepare Loom Dataset (Genes x Cells)
        # Using compression helps with disk space and doesn't hurt speed much for sparse data
        dset = self.handle.create_dataset(
            dest_path, 
            shape=(n_genes, n_cells), 
            dtype='float32', 
            chunks=(min(n_genes, 1024), min(n_cells, 1024)),
            compression="gzip"
        )

        # 2. Identify Matrix Type
        node = f[src_path]
        is_sparse = isinstance(node, h5py.Group) and all(k in node for k in ("data", "indices", "indptr"))
        is_dense  = isinstance(node, h5py.Dataset)
        if not (is_sparse or is_dense):
            ErrorJSON(f"Unsupported matrix layout at {src_path}: expected CSR group or dense dataset")

        ## Prepare a CSR view once if CSC
        csr_view = None
        enc = None
        if is_sparse:
            enc = self._detect_encoding(node, n_cells, n_genes)
            if enc == "csc_matrix":
                # build once, convert once
                X_csc = sp.csc_matrix(
                    (node["data"][:], node["indices"][:], node["indptr"][:]),
                    shape=(n_cells, n_genes),
                )
                csr_view = X_csc.tocsr()
            elif enc != "csr_matrix":
                ErrorJSON(f"Unsupported sparse encoding-type={enc!r} at {src_path}")
        
        # Initialize stats
        total_zeros = 0
        is_integer = True
        
        # Initialize stats vectors ONLY if gene_metadata is provided
        cell_depth = np.zeros(n_cells) if gene_metadata else None
        cell_detected = np.zeros(n_cells) if gene_metadata else None
        cell_mt = np.zeros(n_cells) if gene_metadata else None
        cell_rrna = np.zeros(n_cells) if gene_metadata else None
        cell_prot = np.zeros(n_cells) if gene_metadata else None
        gene_sum = np.zeros(n_genes) if gene_metadata else None
        
        # Pre-calculate indices to avoid doing it inside the loop
        mt_idx = ribo_idx = prot_idx = []
        if gene_metadata is not None:
            mt_idx = [i for i, g in enumerate(gene_metadata) if self.is_mito(g.chr)]
            ribo_idx = [i for i, g in enumerate(gene_metadata) if self.is_ribo(g.name)]
            prot_idx = [i for i, g in enumerate(gene_metadata) if self.is_protein_coding(g.biotype)]
        
        # 3. Chunked Processing (Process by Cell blocks)
        for start in range(0, n_cells, 1024):
            end = min(start + 1024, n_cells)
            
            # Extract chunk
            if is_sparse:
                if enc == "csr_matrix":
                    ptr = node["indptr"][start:end + 1]
                    chunk = sp.csr_matrix( (node["data"][ptr[0]:ptr[-1]], node["indices"][ptr[0]:ptr[-1]], ptr - ptr[0]), shape=(end - start, n_genes) ).toarray()
                else:
                    # enc == "csc_matrix": use csr_view slicing
                    chunk = csr_view[start:end, :].toarray()
            else:
                # dense dataset
                chunk = node[start:end, :]

            # --- Compute Stats ---
            # Check if all values are integers
            total_zeros += (chunk == 0).sum()
            if is_integer and not np.all(np.equal(np.mod(chunk, 1), 0)):
                is_integer = False

            if gene_metadata is not None: # We only compute it in this case
                # Cell stats
                cell_depth[start:end] = chunk.sum(axis=1)
                cell_detected[start:end] = (chunk > 0).sum(axis=1)
                
                # Gene stats (Accumulate)
                gene_sum += chunk.sum(axis=0)
                
                # Content based on gene biotypes/chrom          
                if mt_idx: cell_mt[start:end] = chunk[:, mt_idx].sum(axis=1)
                if ribo_idx: cell_rrna[start:end] = chunk[:, ribo_idx].sum(axis=1)
                if prot_idx: cell_prot[start:end] = chunk[:, prot_idx].sum(axis=1)

            # --- Write to Loom (Transposed chunk) ---
            dset[:, start:end] = chunk.T

        # Function to calculate Percentages (0 to 100)
        # We use np.divide with 'where' to avoid division by zero errors for empty cells
        def to_percentage(subset_sum, total_sum):
            if subset_sum is None or total_sum is None: return None
            return np.divide(subset_sum, total_sum, out=np.zeros_like(subset_sum), where=total_sum != 0) * 100
        
        stats = {
            "cell_depth": cell_depth,
            "cell_detected": cell_detected,
            "cell_mt": to_percentage(cell_mt, cell_depth),
            "cell_rrna": to_percentage(cell_rrna, cell_depth),
            "cell_prot": to_percentage(cell_prot, cell_depth),
            "gene_sum": gene_sum,
            "total_zeros": int(total_zeros),
            "is_count_table": int(is_integer),
            "empty_cells": np.sum(cell_depth == 0) if cell_depth is not None else None,
            "empty_genes": np.sum(gene_sum == 0) if gene_sum is not None else None,
        }

        return stats
    
    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None

### DB Stuff ###

def count_missing(obj):
    if isinstance(obj, np.ndarray):
        flat = obj.ravel()
    else:
        flat = obj

    missing = 0
    for o in flat:
        if o is None:
            missing += 1
            continue
        if isinstance(o, (bytes, np.bytes_)):
            o = o.decode(errors="ignore")
        s = str(o)
        if s == "" or s.lower() in ("nan", "null"):
            missing += 1
    return missing

def dataclass_to_json(obj):
    if dataclasses.is_dataclass(obj):
        d = dataclasses.asdict(obj)
        return {
            k: dataclass_to_json(v) for k, v in d.items() 
            if v is not None and not (isinstance(v, (set, list, dict)) and len(v) == 0)
        }
    if isinstance(obj, dict):
        # Clean keys: Convert numpy/bool keys to strings
        return {str(k): dataclass_to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        # Recursively clean lists/sets
        return [dataclass_to_json(x) for x in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.str_, np.bytes_)):
        return str(obj)
    return obj

@dataclass
class Metadata:
    name: str = None
    on: str = None
    type: str = None
    nber_cols: int = 0
    nber_rows: int = 0
    dataset_size: int = 0
    distinct_values: int = None
    missing_values: int = None
    nber_zeros: int = None
    is_count_table: int = None
    imported: int = 0
    categories: Dict[str, int] = field(default_factory=dict) # Changed from Set to Dict
    values: Set[str] = field(default_factory=set)
    
    def is_categorical(self) -> bool:
        n_unique = len(self.categories)
        if n_unique == 0: return False
        if n_unique > 500: return False
        if n_unique < 10: return True
            
        # Determine the denominator based on the orientation
        if self.on == "CELL":
            denominator = self.nber_cols
        elif self.on == "GENE":
            denominator = self.nber_rows
        else: 
            # For GLOBAL, we don't really have a denominator to compare density, 
            # so we rely on the n_unique < 10 or n_unique > 500 rules.
            return n_unique < 10

        if denominator == 0: return False
        
        # Threshold: unique values < 10% of the relevant dimension
        return n_unique <= (denominator * 0.10)

    def finalize_type(self, is_numeric: bool):
        """Sets the final type and clears categories if not categorical."""
        if self.is_categorical():
            self.type = "DISCRETE"
        else:
            self.categories = {} # Clear counts if it's just a long list of unique strings
            self.type = "NUMERIC" if is_numeric else "STRING"


@dataclass
class Gene:
    ensembl_id: Optional[str] = None
    name: Optional[str] = None
    biotype: Optional[str] = None
    chr: Optional[str] = None
    gene_length: int = 0
    sum_exon_length: int = 0
    latest_ensembl_release: int = 0
    alt_names: Set[str] = field(default_factory=set)
    obsolete_alt_names: Set[str] = field(default_factory=set)

@dataclass
class MapGene:
    ensembl_db: Dict[str, List[Gene]] = field(default_factory=dict)
    gene_db: Dict[str, List[Gene]] = field(default_factory=dict)
    alt_db: Dict[str, List[Gene]] = field(default_factory=dict)
    obsolete_db: Dict[str, List[Gene]] = field(default_factory=dict)
    
    def init(self) -> None:
        self.ensembl_db.clear()
        self.gene_db.clear()
        self.alt_db.clear()
        self.obsolete_db.clear()

    @staticmethod
    def retrieve_latest(db_hits: List[Gene]) -> Gene:
        """ Finds the gene with the highest Ensembl release version. """
        if not db_hits:
            return None
        # Sorts by latest_ensembl_release descending and takes the first
        return sorted(db_hits, key=lambda g: g.latest_ensembl_release, reverse=True)[0]

    def parse_gene(self, query: str, index: int) -> Gene:
        """ Translates a query string into a Gene object using the MapGene DB. """
        # 1. Clean the input
        if not query or query.strip() == "":
            query = None
        
        db_hit = None
        q_up = query.upper() if query else None
    
        if q_up:
            # Waterfall search using self databases
            # 1. Ensembl DB
            db_hit = self.ensembl_db.get(q_up)
            
            # 2. Ensembl DB without version suffix (e.g., ENSG000.1 -> ENSG000)
            if db_hit is None:
                clean_ens = re.sub(r"\.\d+$", "", q_up)
                db_hit = self.ensembl_db.get(clean_ens)
                
            # 3. Gene Name DB
            if db_hit is None:
                db_hit = self.gene_db.get(q_up)
                
            # 4. Alt Names DB
            if db_hit is None:
                db_hit = self.alt_db.get(q_up)
                
            # 5. Obsolete Names DB
            if db_hit is None:
                db_hit = self.obsolete_db.get(q_up)
    
        # If found in DB
        if db_hit:
            g_hit = self.retrieve_latest(db_hit)
            # Ensure name is not null
            if not g_hit.name:
                g_hit.name = g_hit.ensembl_id
            return g_hit
    
        # If NOT found (Create fallback)
        fallback_name = query if query else f"Gene_{index + 1}"
        
        return Gene(
            ensembl_id=fallback_name,
            name=fallback_name,
            biotype="__unknown",
            chr="__unknown",
            sum_exon_length=0,
            alt_names=set()
        )

    def parse_genes_list(self, queries: List[str]):
        """ Process a list of gene identifiers (e.g. var_genes)  and returns a list of Gene objects. """
        results = []
        not_found_count = 0
        for i, q in enumerate(queries):
            gene_obj = self.parse_gene(q, i)
            if gene_obj.biotype == "__unknown":
                not_found_count += 1
            results.append(gene_obj)
        return results, not_found_count

class DBManager:
    def __init__(self, *, dbname: str, user: str, password: str, host: str, port:int, connect_timeout: int = 10, sslmode: Optional[str] = None) -> None:
        self._base_conn_kwargs: Dict[str, Any] = {
            "dbname": dbname,
            "user": user,
            "password": password,
            "port": port,
            "connect_timeout": connect_timeout,
            "host": host
        }
        if sslmode is not None:
            self._base_conn_kwargs["sslmode"] = sslmode
        self._conn: Optional[PGConnection] = None
        self.map_gene = MapGene()
    
    def connect(self) -> None:
        if self._conn is not None and not self._conn.closed:
            return
        conn_kwargs = dict(self._base_conn_kwargs)
        try:
            self._conn = psycopg2.connect(**conn_kwargs)
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
            ErrorJSON("Not connected. Call connect() first (or use get_genes_in_db_once).")
        cur = self._conn.cursor(cursor_factory=RealDictCursor)
        try:
            yield cur
        finally:
            cur.close()

    def get_genes_in_db(self, organism_id: int) -> MapGene:
        self.map_gene.init()

        sql = """
            SELECT ensembl_id, name, biotype, chr, gene_length, sum_exon_length, latest_ensembl_release, alt_names, obsolete_alt_names
            FROM genes
            WHERE organism_id = %s
        """

        with self._cursor() as cur:
            cur.execute(sql, (organism_id,))
            rows = cur.fetchall()

        for r in rows:
            g = Gene(
                ensembl_id=r.get("ensembl_id"),
                name=r.get("name"),
                biotype=r.get("biotype"),
                chr=r.get("chr"),
                gene_length=int(r.get("gene_length") or 0),
                sum_exon_length=int(r.get("sum_exon_length") or 0),
                latest_ensembl_release=int(r.get("latest_ensembl_release") or 0),
            )

            # By ensembl_id
            if g.ensembl_id:
                key = g.ensembl_id.upper()
                self.map_gene.ensembl_db.setdefault(key, []).append(g)

            # By gene name
            if g.name:
                key = g.name.upper()
                self.map_gene.gene_db.setdefault(key, []).append(g)

            # alt_names (comma-separated string)
            alt_names_raw = r.get("alt_names") or ""
            for token in (t.strip() for t in alt_names_raw.split(",") if t.strip()):
                token_up = token.upper()
                g.alt_names.add(token_up)
                self.map_gene.alt_db.setdefault(token_up, []).append(g)

            # obsolete_alt_names (comma-separated string)
            obsolete_raw = r.get("obsolete_alt_names") or ""
            for token in (t.strip() for t in obsolete_raw.split(",") if t.strip()):
                token_up = token.upper()
                g.obsolete_alt_names.add(token_up)
                self.map_gene.obsolete_db.setdefault(token_up, []).append(g)

        return self.map_gene

    def get_genes_in_db_once(self, organism_id: int) -> MapGene:
        self.connect()
        try:
            return self.get_genes_in_db(organism_id)
        finally:
            self.disconnect()

    def get_ribo_protein_gene_names(self, organism_id: int, go_term: str = "GO:0003735") -> set[str]:
        """
        Returns a set of gene symbols (genes.name) for the GO term.
        Uses gene_set_items.identifier (NOT gene_sets.identifier).
        """
        sql = """
        WITH target_set AS (
          SELECT
            gs.organism_id,
            gsi.gene_set_id,
            gsi.content
          FROM gene_set_items gsi
          JOIN gene_sets gs ON gs.id = gsi.gene_set_id
          WHERE gsi.identifier = %s
            AND gs.organism_id = %s
        )
        SELECT
          string_agg(DISTINCT g.name, ',' ORDER BY g.name) AS gene_names
        FROM target_set ts
        CROSS JOIN LATERAL unnest(string_to_array(ts.content, ',')) AS gene_id_text
        JOIN genes g ON g.id = (gene_id_text)::integer;
        """
    
        with self._cursor() as cur:
            cur.execute(sql, (go_term, organism_id))
            row = cur.fetchone() or {}
    
        names_csv = row.get("gene_names") or ""
        names = {n.strip() for n in names_csv.split(",") if n and n.strip()}
        
        return {n.upper() for n in names}

    def get_ribo_protein_gene_names_once(self, organism_id: int, go_term: str = "GO:0003735") -> set[str]:
        self.connect()
        try:
            return self.get_ribo_protein_gene_names(organism_id, go_term=go_term)
        finally:
            self.disconnect()


### MAIN functions ###
        
# Dispatch table to run appropriate parsing function depending on filetype
dispatch = {
    'H5_10X': H510xHandler.parse,
    'H5AD': H5ADHandler.parse,
    'LOOM': LoomHandler.parse,
    'RDS': RdsHandler.parse,
    'MTX': MtxHandler.parse,
    'RAW_TEXT': TextHandler.parse,
}

def get_env(name: str, required: bool = False, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if required and not value:
        ErrorJSON(f"Missing required environment variable: {name}")
    return value

def parse_host_string(s: str):
    if "://" not in s: s = "dummy://" + s
    u = urlparse(s)
    # host
    if not u.hostname:
        ErrorJSON("Missing host")
    host = u.hostname

    # port
    port = None
    if u.netloc:
        parts = u.netloc.split(":")
        if len(parts) == 2:
            if parts[1].isdigit():
                port = int(parts[1])
            else:
               ErrorJSON(f"Invalid port: {parts[1]!r}")
    if port is None:
        #ErrorJSON("Missing port in --dburl (expected HOST:PORT/DB)")
        port = 5432 # default
    
    # dbname
    dbname = u.path.lstrip("/")
    if not dbname:
        ErrorJSON("Missing dbname")

    return host, port, dbname


# Main parsing method, called by "Main" and redispatching to correct format parsing method
def parse(args):
    # Find user/password from environment variables
    user_env = get_env("POSTGRES_USER", required=True)
    pass_env = get_env("POSTGRES_PASSWORD", required=True)
   
    # Extract informations from args.dburl
    host, port, dbname = parse_host_string(args.dburl)
    
    # Connect to DB
    db = DBManager(host=host, dbname=dbname, port=port, user=user_env, password=pass_env)
    
    # Extract genes from species
    gene_db = db.get_genes_in_db_once(args.organism)

    # Fetch ribosomal protein gene symbols via GO and attach to gene_db object
    ribo_set = db.get_ribo_protein_gene_names_once(args.organism, go_term="GO:0003735")
    gene_db.ribo_protein_gene_names = ribo_set

    args.warnings = getattr(args, "warnings", [])
    if not ribo_set:
        args.warnings.append(f"No ribosomal protein genes found for GO:0003735 (organism_id={args.organism}). Ribosomal protein content will be reported as 0 for all cells.")
    
    # Call the appropriate parsing function
    handler = dispatch.get(args.filetype)
    if handler:
        handler(args, args.f, gene_db)
    else:
        ErrorJSON(f"Unknown file type: {args.filetype}")

# Validation method for organism argument
def positive_int(value):
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer")
    if ivalue < 0:
        raise argparse.ArgumentTypeError("organism must be a positive integer")
    return ivalue

def main():
    if '--help' in sys.argv:
        print(custom_help)
        sys.exit(0)

    parser = argparse.ArgumentParser(description='Parsing Mode Script', add_help=False)
    parser.add_argument('-f', metavar='File to parse', required=True)
    parser.add_argument('-o', metavar='Output folder', required=False)
    parser.add_argument('--filetype', metavar='File type', choices=['H5_10X', 'H5AD', 'LOOM', 'RDS', 'MTX', 'RAW_TEXT'], required=True)
    parser.add_argument('--header', metavar='[RAW_TEXT] Is there a header', choices=['true', 'false'], default='true', required=False)
    parser.add_argument('--col', metavar='[RAW_TEXT] Which column contains row names', choices=['none', 'first', 'last'], default='first', required=False)
    parser.add_argument('--sel', metavar='In case of multiple matrices, which one to use', default=None, required=False)
    parser.add_argument('--delim', metavar='[RAW_TEXT] Delimiter to parse columns', default='\t', required=False)
    parser.add_argument('--organism', metavar='Organism', type=positive_int, required=True)
    parser.add_argument('--dburl', metavar='Host URL for DB (format HOST:PORT/DB)', default=None, required=True)

    args = parser.parse_args()

    input_path = Path(args.f).resolve()
    if not input_path.is_file():
        ErrorJSON(f"Input file not found: {args.f}")
    # Determine the output directory
    output_dir = Path(args.o).resolve() if args.o else input_path.parent
    # Ensure the directory exists (handles both cases)
    output_dir.mkdir(parents=True, exist_ok=True) 
    # Set the final file path
    args.output_path = str(output_dir / "output.loom")
    
    parse(args)

if __name__ == '__main__':
    main()
