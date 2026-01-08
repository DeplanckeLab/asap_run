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
    def _is_compound_table(node) -> bool:
        return isinstance(node, h5py.Dataset) and node.dtype.fields is not None

    @staticmethod
    def _table_columns(f, table_path: str):
        node = f[table_path]
        if isinstance(node, h5py.Group):
            cols = node.attrs.get('_column_names', list(node.keys()))
            if isinstance(cols, np.ndarray):
                cols = cols.tolist()
            return cols
        elif H5ADHandler._is_compound_table(node):
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
        elif H5ADHandler._is_compound_table(node):
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
                raise ErrorJSON(f"Unsupported encoding-type={enc!r} at {path}")
    
        raise ErrorJSON(f"Could not determine shape for {path}")
    
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
            if H5ADHandler._is_compound_table(node):
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
    
            expected_len = result["nber_cols"] if orientation == "CELL" else result["nber_rows"]
            if len(values) != expected_len:
                entity = "cells" if orientation == "CELL" else "genes"
                result.setdefault("warning", []).append(f"Skipping {col} from {group_path}: length mismatch. Expected {expected_len} {entity}, but found {len(values)}.")
                continue
            
            # --- your existing Metadata build/write logic ---
            meta = Metadata(
                name=loom_path,
                on=orientation,
                nber_rows=len(values) if orientation == "GENE" else 1,
                nber_cols=len(values) if orientation == "CELL" else 1,
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
                data = group[key][:]
                loom.write(data, loom_path)
                
                # Determine dimensions
                n_items, n_components = data.shape[0], (data.shape[1] if len(data.shape) > 1 else 1)
    
                # 3. Create Metadata entry
                meta = Metadata(
                    name=loom_path,
                    on=orientation,
                    type="NUMERIC",
                    nber_rows=n_items if orientation == "GENE" else n_components,
                    nber_cols=n_items if orientation == "CELL" else n_components,
                    dataset_size=loom.get_dataset_size(loom_path),
                    imported=1
                )
                               
                # 4. Update result object
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
    
                try:
                    raw_shape = item.shape
                    if len(raw_shape) >= 2:
                        r, c = raw_shape[0], raw_shape[1]
                    elif len(raw_shape) == 1:
                        r, c = raw_shape[0], 1
                    else:
                        r, c = 1, 1
    
                    val = item[()]
                    if isinstance(val, (bytes, np.bytes_)):
                        val = val.decode("utf-8")
                    elif isinstance(val, np.ndarray) and val.dtype.kind in ("S", "U"):
                        val = np.array(
                            [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in val.flatten()]
                        ).reshape(val.shape)
    
                    loom.write(val, loom_path)
    
                    is_num = False
                    try:
                        is_num = np.issubdtype(np.array(val).dtype, np.number)
                    except:
                        pass
    
                    meta = Metadata(
                        name=loom_path,
                        on="GLOBAL",
                        type="NUMERIC" if is_num else "STRING",
                        nber_rows=int(r),
                        nber_cols=int(c),
                        dataset_size=loom.get_dataset_size(loom_path),
                        imported=1
                    )
                    result["metadata"].append(meta)
    
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
        # Optional: "message":"bla"

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

class H510xHandler:
    @staticmethod
    def parse(args, file_path, gene_db):
        # build the JSON-able structure
        result = {
            "detected_format": "H5_10x",
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

class LoomHandler:
    @staticmethod
    def parse(args, file_path, gene_db):
        # build the JSON-able structure
        result = {
            "detected_format": "LOOM",
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
            raise ErrorJSON( f"Cannot infer sparse encoding: len(indptr)={indptr_len}, expected {n_cells+1} (CSR) or {n_genes+1} (CSC)" )
    
        # If attr exists but disagrees, prefer inferred (and warn)
        if enc and enc != inferred:
            # optional: log warning instead of error
            # print(f"WARNING: encoding-type attr={enc} but inferred={inferred}; using inferred")
            return inferred
    
        return inferred
    
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
        mt_idx = rrna_idx = prot_idx = []
        if gene_metadata is not None:
            mt_idx = [i for i, g in enumerate(gene_metadata) if g.chr == "MT"]
            rrna_idx = [i for i, g in enumerate(gene_metadata) if g.biotype == "rRNA"]
            prot_idx = [i for i, g in enumerate(gene_metadata) if g.biotype == "protein_coding"]
        
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
                if rrna_idx: cell_rrna[start:end] = chunk[:, rrna_idx].sum(axis=1)
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
            raise ErrorJSON(f"Failed to connect to PostgreSQL: {e}") from e

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
            raise ErrorJSON("Not connected. Call connect() first (or use get_genes_in_db_once).")
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


### MAIN functions ###
        
# Dispatch table to run appropriate parsing function depending on filetype
dispatch = {
    'H5_10X': H510xHandler.parse,
    'H5AD': H5ADHandler.parse,
    'LOOM': LoomHandler.parse,
    'RDS': LoomHandler.parse,
    'MTX': MtxHandler.parse,
    'RAW_TEXT': TextHandler.parse,
}

def get_env(name: str, required: bool = False, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if required and not value:
        raise ErrorJSON(f"Missing required environment variable: {name}")
    return value

def parse_host_string(s: str):
    from urllib.parse import urlparse
    if "://" not in s:
        s = "dummy://" + s
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
