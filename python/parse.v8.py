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
from scipy.sparse import isspmatrix_csr, isspmatrix_csc # For handling sparse matrices

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
    def get_size(f, path):
        # Get matrix dimensions
        shape = f[path].attrs.get('shape', None)
        if shape is None:
            shape = f[path].attrs.get('h5sparse_shape', None)
        if shape is None: # Dense matrix?
            shape = f[path].shape  # fallback: dense matrix or explicitly stored shape
        return tuple(int(x) for x in shape)
    
    @staticmethod
    def extract_index(f, path):
        try:
            # AnnData stores the name of the index column in the '_index' attribute
            raw = f[path].attrs.get('_index', 'index') 
            index_col_name = raw.decode() if isinstance(raw, bytes) else raw
            full_path = f"{path.rstrip('/')}/{index_col_name.lstrip('/')}"
            
            if full_path not in f:
                # Fallback if the attribute pointed to a non-existent path
                full_path = f"{path.rstrip('/')}/_index"
                
            return [v.decode() if isinstance(v, bytes) else str(v)
                    for v in f[full_path][:]]    
        except Exception:
            ErrorJSON(f"Could not find index for {path}. Ensure the H5AD is valid.")

    @staticmethod
    def transfer_metadata(f, group_path, loom, result, orientation, existing_paths):
        """
        f: H5AD file handle
        group_path: e.g., 'obs' or 'var'
        orientation: 'CELL' (col_attrs) or 'GENE' (row_attrs)
        existing_paths: set of paths already written (to avoid duplicates)
        """
        if group_path not in f: return
        group = f[group_path]
        columns = group.attrs.get('_column_names', group.keys())
    
        for col in columns:
            if col in ['_index', 'categories', 'codes']: continue # Skip internal AnnData keys
            loom_path = f"/{'col' if orientation == 'CELL' else 'row'}_attrs/{col}"
            if loom_path in existing_paths:
                result.setdefault("warning", []).append(f"Skipping {col}: already exists.")
                continue
    
            data = group[col]

            # Dimension check
            expected_len = result["nber_cols"] if orientation == "CELL" else result["nber_rows"]
            if len(data) != expected_len:
                result.setdefault("warning", []).append(f"Skipping {group_path}/{col}: length mismatch. Expected {expected_len}, found {len(data)}.")
                continue
            
            # Handle Categorical
            if isinstance(data, h5py.Group) and 'categories' in data:
                categories = [v.decode() if isinstance(v, bytes) else str(v) for v in data['categories'][:]]
                codes = data['codes'][:]
                values = [categories[c] if c != -1 else "nan" for c in codes]
            else:
                raw_values = data[:]
                values = [v.decode() if isinstance(v, bytes) else str(v) for v in raw_values]
    
            meta = Metadata(
                name=loom_path,
                on=orientation,
                nber_rows=len(values) if orientation == "GENE" else 1,
                nber_cols=len(values) if orientation == "CELL" else 1,
                missing_values = count_missing(values),
                distinct_values = len(set(values)),
                imported=1
            )
            
            counts = Counter(values)
            meta.categories = dict(counts)
            
            # Numeric Check
            is_numeric = True
            for v in values[:1000]: # Sample for speed, or check all if strict
                if v in [None, "", "nan", "NaN"]: continue
                try: float(v.replace(',', '.'))
                except: 
                    is_numeric = False
                    break
            
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
                # data.shape is (N, M). 
                # If orientation is CELL: N = cells, M = components (cols)
                # If orientation is GENE: N = genes, M = components (rows)
                n_items, n_components = data.shape[0], (data.shape[1] if len(data.shape) > 1 else 1)
    
                # 3. Create Metadata entry
                meta = Metadata(
                    name=loom_path,
                    on=orientation,
                    type="NUMERIC",
                    nber_rows=n_items if orientation == "GENE" else 1,
                    nber_cols=n_components if orientation == "CELL" else 1,
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
        Uses chunked processing to remain RAM-efficient.
        """
        if "layers" not in f:  return
        for layer_name in f["layers"].keys():
            loom_path = f"/layers/{layer_name}"
            
            # 1. Duplicate Check
            if loom_path in existing_paths:
                result.setdefault("warning", []).append(f"Skipping layer {layer_name}: already exists.")
                continue
    
            try:
                # Check dimension
                l_shape = H5ADHandler.get_size(f, f"layers/{layer_name}")
                if l_shape[0] != n_cells or l_shape[1] != n_genes:
                    result.setdefault("warning", []).append(f"Skipping layer {layer_name}: dimension mismatch. Expected ({n_cells}, {n_genes}), found {l_shape}.")
                    continue
                
                # 2. Prepare Loom Dataset (Genes x Cells)
                # Loom expects layers to be transposed relative to H5AD, just like the main matrix
                dset = loom.handle.create_dataset(
                    loom_path, 
                    shape=(n_genes, n_cells), 
                    dtype='float32', 
                    chunks=(min(n_genes, 1024), min(n_cells, 1024)),
                    compression="gzip"
                )
    
                # 3. Detect Sparse vs Dense
                layer_data = f["layers"][layer_name]
                is_sparse = 'data' in layer_data and 'indices' in layer_data and 'indptr' in layer_data
    
                # 4. Chunked Write
                for start in range(0, n_cells, 1024):
                    end = min(start + 1024, n_cells)
                    
                    if is_sparse:
                        import scipy.sparse as sp
                        ptr = layer_data['indptr'][start:end+1]
                        chunk = sp.csr_matrix(
                            (layer_data['data'][ptr[0]:ptr[-1]], 
                             layer_data['indices'][ptr[0]:ptr[-1]], 
                             ptr - ptr[0]), 
                            shape=(end-start, n_genes)
                        ).toarray()
                    else:
                        chunk = layer_data[start:end, :]
    
                    # Write Transposed
                    dset[:, start:end] = chunk.T
    
                # 5. Metadata Entry
                meta = Metadata(
                    name=loom_path,
                    on="EXPRESSION_MATRIX", # Layers are global to the file
                    type="NUMERIC",
                    nber_rows=n_genes,
                    nber_cols=n_cells,
                    dataset_size=loom.get_dataset_size(loom_path),
                    imported=1
                )
                result["metadata"].append(meta)
                existing_paths.add(loom_path)
    
            except Exception as e:
                result.setdefault("warning", []).append(f"Failed to transfer layer {layer_name}: {str(e)}")

    @staticmethod
    def transfer_unstructured_metadata(f, loom, result, existing_paths):
        if "uns" not in f: return

        # Function for recursive check of uns/ arborescence
        def walk(group, prefix=""):
            for key in group.keys():
                item = group[key]
                attr_name = f"{prefix}{key}"
                loom_path = f"/attrs/{attr_name}"
                
                if isinstance(item, h5py.Group):
                    # Recursively walk groups
                    walk(item, prefix=f"{attr_name}_")
                elif isinstance(item, h5py.Dataset):
                    # 0. Protection Check
                    if loom_path in existing_paths:
                        continue
                    
                    try:
                        # 1. Size constraint: uns is for small metadata. 
                        # Large arrays should be in obsm/varm/layers.
                        if item.size > 1000:
                            continue
                        
                        val = item[()]
                        
                        # 2. Advanced String/Byte Decoding
                        if isinstance(val, (bytes, np.bytes_)):
                            val = val.decode('utf-8')
                        elif isinstance(val, np.ndarray):
                            if val.dtype.kind in ['S', 'U']: # Bytes or Unicode
                                val = [v.decode('utf-8') if isinstance(v, bytes) else str(v) for v in val]
                            elif val.ndim == 0: # Handle scalar arrays
                                val = val.item()
                                if isinstance(val, bytes): val = val.decode('utf-8')
                        
                        # 3. Write as a dataset in attrs
                        loom.write(val, loom_path)
                        
                        # 4. Determine Numeric Type (We use NUMERIC if it's a number/array of numbers, else STRING)
                        is_num = False
                        try:
                            is_num = np.issubdtype(np.array(val).dtype, np.number)
                        except:
                            pass
                        
                        # Add to result["metadata"]
                        meta = Metadata(
                            name=loom_path, # Global attrs in Loom are at root level
                            on="GLOBAL",
                            type="NUMERIC" if is_num else "STRING",
                            nber_cols=len(val) if isinstance(val, (list, np.ndarray)) else 1, # If it's a list/array, we can specify the length in nber_cols
                            nber_rows=1,
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
            if result["input_group"] == '/raw.X':
                if '/raw.var' in f:   
                    var_genes = H5ADHandler.extract_index(f, "/raw.var")
                else:
                    var_genes = H5ADHandler.extract_index(f, "/var")
                if '/raw.obs' in f:   
                    obs_cells = H5ADHandler.extract_index(f, "/raw.obs")
                else:
                    obs_cells = H5ADHandler.extract_index(f, "/obs")
            elif result["input_group"] == '/raw/X':
                if '/raw/var' in f:   
                    var_genes = H5ADHandler.extract_index(f, "/raw/var")
                else:
                    var_genes = H5ADHandler.extract_index(f, "/var")
                if '/raw/obs' in f:   
                    obs_cells = H5ADHandler.extract_index(f, "/raw/obs")
                else:
                    obs_cells = H5ADHandler.extract_index(f, "/obs")
            else:
                var_genes = H5ADHandler.extract_index(f, "/var")
                obs_cells = H5ADHandler.extract_index(f, "/obs")

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

            ## 6. Transfer the existing metadata from h5ad to Loom
            # To avoid overwriting fields I've already generated
            existing_paths = {m.name for m in result["metadata"]}
            existing_paths.add("/attrs/LOOM_SPEC_VERSION")

            ### 6.a. Transfer obs (Cell unidimensional metadata)
            H5ADHandler.transfer_metadata(f, "obs", loom, result, "CELL", existing_paths)

            ### 6.b. Transfer var (Gene unidimensional metadata)
            H5ADHandler.transfer_metadata(f, "var", loom, result, "GENE", existing_paths)

            ### 6.c. Transfer obsm (Cell multidimensional metadata)
            H5ADHandler.transfer_multidimensional_metadata(f, "obsm", loom, result, "CELL", existing_paths)
            
            ### 6.d. Transfer varm (Gene multidimensional metadata)
            H5ADHandler.transfer_multidimensional_metadata(f, "varm", loom, result, "GENE", existing_paths)

            ### 6.e. Layers (Alternative matrices like 'spliced'/'unspliced' or 'transformed')
            H5ADHandler.transfer_layers(f, loom, result, result["nber_cols"], result["nber_rows"], existing_paths)

            ### 6.f: uns (Unstructured metadata like colors or global parameters)
            H5ADHandler.transfer_unstructured_metadata(f, loom, result, existing_paths)

            ### 7. end
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
        group = f[src_path]
        is_sparse = 'data' in group and 'indices' in group and 'indptr' in group
        
        # Initialize stats vectors
        cell_depth = np.zeros(n_cells)
        cell_detected = np.zeros(n_cells)
        cell_mt = np.zeros(n_cells)
        cell_rrna = np.zeros(n_cells)
        cell_prot = np.zeros(n_cells)
        
        gene_sum = np.zeros(n_genes)
        total_zeros = 0
        is_integer = True

        # 3. Chunked Processing (Process by Cell blocks) 
        for start in range(0, n_cells, 1024):
            end = min(start + 1024, n_cells)
            
            # Extract chunk
            if is_sparse:
                # Load sparse chunk and convert to dense for calculation
                import scipy.sparse as sp
                data = group['data']
                indices = group['indices']
                indptr = group['indptr']
                
                # Create a CSR slice
                ptr_slice = indptr[start:end+1]
                data_slice = data[ptr_slice[0]:ptr_slice[-1]]
                indices_slice = indices[ptr_slice[0]:ptr_slice[-1]]
                
                # Adjust indptr to start at 0
                ptr_slice = ptr_slice - ptr_slice[0]
                
                chunk = sp.csr_matrix((data_slice, indices_slice, ptr_slice), 
                                     shape=(end-start, n_genes)).toarray()
            else:
                chunk = group[start:end, :] # Dense slice

            # --- Compute Stats ---
            # Check if all values are integers
            if is_integer and not np.all(np.equal(np.mod(chunk, 1), 0)):
                is_integer = False
            
            # Cell stats
            cell_depth[start:end] = chunk.sum(axis=1)
            cell_detected[start:end] = (chunk > 0).sum(axis=1)
            
            # Content based on gene biotypes/chrom
            # (Assuming gene_metadata is a list of Gene objects indexed 0..n_genes)
            mt_idx = [i for i, g in enumerate(gene_metadata) if g.chr == "MT"]
            rrna_idx = [i for i, g in enumerate(gene_metadata) if g.biotype == "rRNA"]
            prot_idx = [i for i, g in enumerate(gene_metadata) if g.biotype == "protein_coding"]
            
            if mt_idx: cell_mt[start:end] = chunk[:, mt_idx].sum(axis=1)
            if rrna_idx: cell_rrna[start:end] = chunk[:, rrna_idx].sum(axis=1)
            if prot_idx: cell_prot[start:end] = chunk[:, prot_idx].sum(axis=1)

            # Gene stats (Accumulate)
            gene_sum += chunk.sum(axis=0)
            total_zeros += (chunk == 0).sum()

            # --- Write to Loom (Transposed) ---
            dset[:, start:end] = chunk.T

        # 5. Final Aggregates
        # Function to calculate Percentages (0 to 100)
        # We use np.divide with 'where' to avoid division by zero errors for empty cells
        def to_percentage(subset_sum, total_sum):
            return np.divide(subset_sum, total_sum, out=np.zeros_like(subset_sum), where=total_sum != 0) * 100
        
        stats = {
            "cell_depth": cell_depth,
            "cell_detected": cell_detected,
            "cell_mt": to_percentage(cell_mt, cell_depth),
            "cell_rrna": to_percentage(cell_rrna, cell_depth),
            "cell_prot": to_percentage(cell_prot, cell_depth),
            "gene_sum": gene_sum,
            "total_zeros": total_zeros,
            "is_count_table": is_integer,
            "empty_cells": np.sum(cell_depth == 0),
            "empty_genes": np.sum(gene_sum == 0)
        }
        return stats
    
    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None

### DB Stuff ###

def count_missing(obj):
    return sum(1 for o in obj if o in [None, "", "nan", "NaN", "null"])

def dataclass_to_json(obj):
    if dataclasses.is_dataclass(obj):
        d = dataclasses.asdict(obj)
        return {
            k: v for k, v in d.items() 
            if v is not None and not (isinstance(v, (set, list, dict)) and len(v) == 0)
        }
    if isinstance(obj, set):
        return sorted(list(obj))
    # Dictionaries are natively supported by JSON, so no extra logic needed here
    return str(obj)

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
    imported: int = 0
    categories: Dict[str, int] = field(default_factory=dict) # Changed from Set to Dict
    values: Set[str] = field(default_factory=set)
    
    def is_categorical(self) -> bool:
        n_unique = len(self.categories)
        if n_unique == 0: return False
        if n_unique > 500: return False
        if n_unique < 10: return True
        # Threshold: unique values < 10% of total rows
        return n_unique <= (self.nber_rows * 0.10)

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
