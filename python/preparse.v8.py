import argparse # Needed to parse arguments
import sys # Needed for ErrorJSON
import os # Needed for pathjoin and path/create dirs
import h5py # Needed for parsing hdf5 files
import json # For writing output files
import numpy as np # For diverse array calculations
import tarfile # To check if file is a TAR archive
from pathlib import Path # Needed for file extension manipulation

custom_help = """
Preparsing Mode

Options:
  -f %s             File to preparse.
  -o %s             Output folder.
  --header %s       The file has a header [true, false] (default: true).
  --col %s          Name Column [none, first, last] (default: first).
  --sel %s          Name of entry to load from archive or HDF5 (if multiple groups).
  --delim %s        Delimiter (default: tab).
  --help            Show this help message and exit.
"""

#   --organism %i     ID of the organism.
#   --row-names %s    Metadata column for row names of the main matrix.
#   --col-names %s    Metadata column for column names of the main matrix.
#   --host %s         Host override (default: postgres:5434).

class FileType:
    H5_10X = 'H5_10X'
    H5AD = 'H5AD'
    LOOM = 'LOOM'
    RDS = 'RDS'
    MTX = 'MTX'
    ARCHIVE = 'ARCHIVE'
    RAW_TEXT = 'RAW_TEXT'
    UNKNOWN = 'UNKNOWN'

def extract_from_archive(file_path, sel):
    with open(file_path, 'rb') as f:
        magic = f.read(264)
    
    # zip (can contain multiple files)
    if magic.startswith(b'PK\x03\x04'):
        import zipfile, shutil
        with zipfile.ZipFile(file_path, 'r') as z:
            # Filter only files (exclude directories)
            file_paths = [f for f in z.namelist() if not f.endswith('/')]
            if sel in file_paths:
                # Extract the single file
                extracted_path = z.extract(sel, path='.')
                # Move the file to current dir, stripping folders
                final_path = Path('.').resolve() / Path(sel).name
                shutil.move(extracted_path, final_path)
                # Optionally, remove the now-empty folder structure
                file_path, archive_path = decompress_if_needed(str(final_path), sel)
                return file_path[0]
            else:
                return None

    # tar (can contain multiple files)
    if tarfile.is_tarfile(file_path):
        import shutil
        with tarfile.open(file_path, mode='r:') as t:
            # Filter only files (exclude directories)
            file_paths = [m.name for m in t.getmembers() if m.isfile()]
            if sel in file_paths:
                # Extract the single file
                t.extract(sel, path='.', filter='data')
                extracted_path = Path('.') / sel
                # Move the file to current dir, stripping folders
                final_path = Path('.').resolve() / Path(sel).name
                shutil.move(extracted_path, final_path)
                # Optionally, remove the now-empty folder structure
                file_path, archive_path = decompress_if_needed(str(final_path), sel)
                return file_path[0]
            else:
                return None

def decompress_if_needed(file_path, sel):
    with open(file_path, 'rb') as f:
        magic = f.read(264)

    # gzip (cannot contain multiple files)
    if magic.startswith(b'\x1f\x8b'):
        import subprocess
        output_path = Path(file_path).with_suffix('')
        if output_path == Path(file_path):
            output_path = output_path.with_name(output_path.name + '.out')
        with open(output_path, 'wb') as f_out:
            subprocess.run(["pigz", "-d", '-k', '-f', "-c", file_path], stdout=f_out, check=True)
        return decompress_if_needed(str(output_path), sel)

    # bzip2 (cannot contain multiple files)
    if magic.startswith(b'BZh'):
        import subprocess
        output_path = Path(file_path).with_suffix('')
        if output_path == Path(file_path):
            output_path = output_path.with_name(output_path.name + '.out')
        with open(output_path, 'wb') as f_out:
            subprocess.run(["pbzip2", "-d", "-k", "-f", "-c", file_path], stdout=f_out, check=True)
        return decompress_if_needed(str(output_path), sel)
        
    # xz (cannot contain multiple files)
    if magic.startswith(b'\xfd7zXZ\x00'):
        import subprocess
        output_path = Path(file_path).with_suffix('')
        if output_path == Path(file_path):
            output_path = output_path.with_name(output_path.name + '.out')
        with open(output_path, 'wb') as f_out:
            subprocess.run(["pixz", "-d", "-k", "-c", file_path], stdout=f_out, check=True)
        return decompress_if_needed(str(output_path), sel)

    # zip (can contain multiple files)
    if magic.startswith(b'PK\x03\x04'):
        import zipfile, shutil
        with zipfile.ZipFile(file_path, 'r') as z:
            # Filter only files (exclude directories)
            file_paths = [f for f in z.namelist() if not f.endswith('/')]
            if len(file_paths) == 1:
                # Extract the single file
                extracted_path = z.extract(file_paths[0], path='.')
                # Move the file to current dir, stripping folders
                final_path = Path('.').resolve() / Path(file_paths[0]).name
                shutil.move(extracted_path, final_path)
                # Optionally, remove the now-empty folder structure
                return decompress_if_needed(str(final_path), sel)
            else:
                # Multiple files
                if sel:
                    # If sel is defined, extract the required file
                    if sel in file_paths:
                        # Extract the single file
                        extracted_path = z.extract(sel, path='.')
                        # Move the file to current dir, stripping folders
                        final_path = Path('.').resolve() / Path(sel).name
                        shutil.move(extracted_path, final_path)
                        # Optionally, remove the now-empty folder structure
                        return decompress_if_needed(str(final_path), sel)
                    else:
                        ErrorJSON(f"Your selected file '{sel}' is not in the list of files of the archive: {file_paths}")
                else:
                    # MTX format in a zip archive. matrix.mtx barcodes.tsv and features.tsv || genes.tsv
                    if len(file_paths) == 3 and {{Path(p).name for p in file_paths} == {"matrix.mtx", "barcodes.tsv", "features.tsv"} or {Path(p).name for p in file_paths} == {"matrix.mtx", "barcodes.tsv", "genes.tsv"}}:
                        final_file_paths = []
                        # Extract all three files
                        for fname in file_paths:
                            # Extract the single file
                            extracted_path = z.extract(fname, path='.')
                            final_path = Path('.').resolve() / Path(fname).name
                            shutil.move(extracted_path, final_path)
                            final_file_paths.append(str(final_path))
                        # Return None for archive path (use this to check later that it is a MTX format)
                        return final_file_paths, None
                    else:
                        # If not, then return the list of files in the archive + the archive path
                        return file_paths, file_path

    # tar (can contain multiple files)
    if tarfile.is_tarfile(file_path):
        import shutil
        with tarfile.open(file_path, mode='r:') as t:
            # Filter only files (exclude directories)
            file_paths = [m.name for m in t.getmembers() if m.isfile()]
            if len(file_paths) == 1:
                # Extract the single file
                t.extract(file_paths[0], path='.', filter='data')
                extracted_path = Path('.') / file_paths[0]
                final_path = Path('.').resolve() / Path(file_paths[0]).name
                shutil.move(extracted_path, final_path)
                return decompress_if_needed(str(final_path), sel)
            else:
                # Multiple files
                if sel:
                    # If sel is defined, extract the required file
                    if sel in file_paths:
                        # Extract the single file
                        t.extract(sel, path='.', filter='data')
                        extracted_path = Path('.') / sel
                        # Move the file to current dir, stripping folders
                        final_path = Path('.').resolve() / Path(sel).name
                        shutil.move(extracted_path, final_path)
                        # Optionally, remove the now-empty folder structure
                        return decompress_if_needed(str(final_path), sel)
                    else:
                        ErrorJSON(f"Your selected file '{sel}' is not in the list of files of the archive: {file_paths}")
                else:
                    # MTX format in a tar archive. matrix.mtx barcodes.tsv and features.tsv || genes.tsv
                    if len(file_paths) == 3 and {{Path(p).name for p in file_paths} == {"matrix.mtx", "barcodes.tsv", "features.tsv"} or {Path(p).name for p in file_paths} == {"matrix.mtx", "barcodes.tsv", "genes.tsv"}}:
                        final_file_paths = []
                        # Extract all three files
                        for fname in file_paths:
                            t.extract(fname, path='.', filter='data')
                            extracted_path = Path('.') / fname
                            final_path = Path('.').resolve() / Path(fname).name
                            shutil.move(extracted_path, final_path)
                            final_file_paths.append(str(final_path))
                        # Return None for archive path (use this to check later that it is a MTX format)
                        return final_file_paths, None
                    else:
                        # If not, then return the list of files in the archive + the archive path
                        return file_paths, file_path

    # Uncompressed single file
    return [file_path], None

def check_file_type(file_path):   
    with open(file_path, 'rb') as f:
        magic = f.read(8)
        if magic != b'\x89HDF\r\n\x1a\n':
            return FileType.RAW_TEXT  # Not HDF5, no need to try h5py
    
    # It should be an HDF5
    try:
        with h5py.File(file_path, 'r') as f:
            # 10x Genomics HDF5 Feature-Barcode Matrix detection
            root_groups = [key for key in f.keys() if isinstance(f[key], h5py.Group)]
            for group_name in root_groups:
                if {'barcodes', 'data', 'indices', 'indptr', 'shape'}.issubset(f[group_name].keys()):
                    return FileType.H5_10X
            
            # Loom format detection
            if {'matrix', 'row_attrs', 'col_attrs'}.issubset(f.keys()):
                return FileType.LOOM

            # AnnData H5AD detection
            if {'X', 'obs', 'var'}.issubset(f.keys()):
                return FileType.H5AD
            
            # We don't know what it is, but it's a HDF5 file
            return FileType.UNKNOWN
    except (OSError, IOError, Exception):
        # Not a valid HDF5 or unable to read; fall through
        return FileType.UNKNOWN

def count_nonempty_lines(filepath):
    import subprocess
    result = subprocess.run(['wc', '-l', filepath], capture_output=True, text=True)
    return int(result.stdout.strip().split()[0])

def is_rds_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            magic = f.read(5)
            is_gzip = magic.startswith(b'\x1f\x8b')
        if is_gzip:
            import gzip
            with gzip.open(file_path, 'rb') as f:
                header = f.read(5)
        else:
            with open(file_path, 'rb') as f:
                header = f.read(5)
        return header.startswith(b'RDX') or header.startswith(b'X\n') or header.startswith(b'A\n')
    except Exception as e:
        return False

def preparse(args):
    if is_rds_file(args.f):
        RdsHandler.preparse(args, args.f)
    else:
        # Decompress unique zip files, or treat as archive
        list_files, archive_path = decompress_if_needed(args.f, args.sel)
        if len(list_files) == 0: # Empty archive
            ErrorJSON(f"Archive is empty or corrupted: {archive_path}")        
        elif len(list_files) == 3 and archive_path == None: # MTX (10x)
            MtxHandler.preparse(args, list_files)
        elif len(list_files) > 1: # Archive with multiple files
            ArchiveHandler.write_listing_json(args, archive_path, list_files)
        else: # Unique file, decompressed if needed
            file_path = list_files[0]
            # Detect file type from magic bytes or hdf5 reader
            file_type = check_file_type(file_path)
            # Call appropriate handler
            if file_type == FileType.H5_10X:
                H510xHandler.preparse(args, file_path)
            elif file_type == FileType.H5AD:
                H5ADHandler.preparse(args, file_path)
            elif file_type == FileType.LOOM:
                LoomHandler.preparse(args, file_path)
            elif file_type == FileType.RAW_TEXT:
                # Check if MTX format
                first_line = None
                try:
                    with open(file_path, 'r') as f:
                        first_line = f.readline().strip()
                except (Exception) as e:
                    ErrorJSON(str(e).strip() + '. Format is not handled?')
                if first_line.startswith("%%MatrixMarket matrix"):
                    # It's a unique file. Did we miss barcodes.tsv and genes/features.tsv ?
                    file_paths = [file_path]
                    if args.sel: # If it comes from an archive
                        p = Path(args.sel)
                        barcode_path = str(p.parent / "barcodes.tsv") if p.parent != Path('.') else "barcodes.tsv"
                        barcode_file = extract_from_archive(args.f, barcode_path)
                        if barcode_file: # If barcode file is found
                            file_paths.append(barcode_file)
                        feature_path = str(p.parent / "features.tsv") if p.parent != Path('.') else "features.tsv"
                        feature_file = extract_from_archive(args.f, feature_path)
                        if not feature_file:
                            feature_path = str(p.parent / "genes.tsv") if p.parent != Path('.') else "genes.tsv"
                            feature_file = extract_from_archive(args.f, feature_path)
                        if feature_file: # If feature file is found
                            file_paths.append(feature_file)
                    MtxHandler.preparse(args, file_paths)
                else:
                    TextHandler.preparse(args, file_path)
            else: # Not detected
                ErrorJSON(f"File format not detected.")

class H510xHandler:
    @staticmethod
    def preparse(args, file_path):
        # build the JSON-able structure
        result = {
            "detected_format": FileType.H5_10X,
            "file_path": file_path, # If it was compressed, it should be decompressed before
            "list_groups": []
        }
        
        # Detect groups
        with h5py.File(file_path, 'r') as f:
            root_groups = [key for key in f.keys() if isinstance(f[key], h5py.Group)]
            for group_name in root_groups:
                grp = f[group_name]
                if {'barcodes', 'data', 'indices', 'indptr', 'shape'}.issubset(grp.keys()):
                    # Matrix size
                    nGenes = int(grp["shape"][0]);
                    nCells = int(grp["shape"][1])

                    # Get genes
                    try:
                        genes = [g.decode() for g in grp["gene_names"][:min(10,nGenes)]] # scRNA-seq
                    except KeyError:
                        genes = [g.decode() for g in grp["features/id"][:min(10,nGenes)]] # scATAC-seq

                    # Extract 10x10 matrix
                    indptr = grp["indptr"][:min(10, nCells) + 1] # 10 first rows + beginning of 11th
                    data = grp["data"][:indptr[-1]] # All values for 10 first rows
                    indices = grp["indices"][:indptr[-1]] # All values for 10 first rows
                    matrix = [[0 for _ in range(min(10, nGenes))] for _ in range(min(10, nCells))]
                    for row in range(min(10, nCells)):
                       start, end = indptr[row], indptr[row+1]
                       for idx_pos in range(start, end):
                           col = indices[idx_pos]
                           if col < min(10, nGenes):
                               matrix[row][col] = int(data[idx_pos])
                  
                    # Valid group
                    entry = {
                        "group":    group_name,
                        "nber_cols": nCells,
                        "nber_rows": nGenes,
                        "is_count":  int((np.array(matrix) % 1 == 0).all()),
                        "genes": genes,
                        "cells": [b.decode() for b in grp["barcodes"][:min(10,nCells)]],
                        "matrix": matrix
                    }
                    result["list_groups"].append(entry)

        # Serialize to JSON
        json_str = json.dumps(result, ensure_ascii=False)
        
        # Either write to file or print
        if args.o:
            with open(os.path.join(args.o, "output.json"), "w", encoding="utf-8") as out:
                out.write(json_str)
        else:
            print(json_str)
        

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
        # Read gene names
        try:
            # Standard case: _index dataset
            # Check attribute _index if exists
            raw = f[path].attrs.get('_index', '_index')
            index_path = raw.decode() if isinstance(raw, bytes) else raw
            # Return 10 first values
            return [v.decode() if isinstance(v, bytes) else str(v)
                    for v in f[f"{path}/{index_path}"][:10]]
        except KeyError:
            # Fallback: compound dataset with named fields (e.g. f["/var"] is a structured array)
            dset = f[path]
            if isinstance(dset.dtype, np.dtype) and dset.dtype.names:
                first_col = dset.dtype.names[0]
                return [v[first_col].decode() if isinstance(v[first_col], bytes) else str(v[first_col])
                       for v in dset[:10]]
            else:
                return [f"{path}:invalid_structure"]
    
    @staticmethod
    def preparse(args, file_path):
        result = {
            "detected_format": FileType.H5AD,
            "file_path": file_path, # If it was compressed, it should be decompressed before
            "list_groups": []
        }

        with h5py.File(file_path, 'r') as f:
            for group in ['/X', '/raw/X', '/raw.X']:
                if group in f:                
                    # Get matrix dimensions
                    nCells, nGenes = H5ADHandler.get_size(f, group)

                    # Determine matrix format
                    if isinstance(f[group], h5py.Group) and "data" in f[group]:
                        matrix_format = f[group].attrs.get('h5sparse_format', None)
                        if matrix_format is None:
                            matrix_format = f[group].attrs.get('encoding-type', None)[:-7] # Other way to encore the type
                        max_cells = min(10, nCells)
                        max_genes = min(10, nGenes)
                        matrix = [[0 for _ in range(max_genes)] for _ in range(max_cells)]
                        if matrix_format == 'csr':
                            indptr = f[f"{group}/indptr"][:max_cells + 1]
                            indices = f[f"{group}/indices"][:indptr[-1]]
                            data = f[f"{group}/data"][:indptr[-1]]
                            for row in range(max_cells):
                                start, end = indptr[row], indptr[row + 1]
                                for idx in range(start, end):
                                    col = indices[idx]
                                    if col < max_genes:
                                        matrix[row][col] = float(data[idx])
                        elif matrix_format == 'csc':
                            indptr = f[f"{group}/indptr"][:max_genes + 1]
                            indices = f[f"{group}/indices"][:indptr[-1]]
                            data = f[f"{group}/data"][:indptr[-1]]
                            for col in range(max_genes):
                                start, end = indptr[col], indptr[col + 1]
                                for idx in range(start, end):
                                    row = indices[idx]
                                    if row < max_cells:
                                        matrix[row][col] = float(data[idx])
                        else:
                            ErrorJSON("Unsupported or missing matrix format")
                    else:
                        # Dense matrix
                        matrix = f[group][:min(10, nCells), :min(10, nGenes)].astype(float).tolist()

                    # Load gene/cell names
                    if group == '/raw.X':
                        if '/raw.var' in f:   
                            genes = H5ADHandler.extract_index(f, "/raw.var")
                        else:
                            genes = H5ADHandler.extract_index(f, "/var")
                        if '/raw.obs' in f:   
                            cells = H5ADHandler.extract_index(f, "/raw.obs")
                        else:
                            cells = H5ADHandler.extract_index(f, "/obs")
                    elif group == '/raw/X':
                        if '/raw/var' in f:   
                            genes = H5ADHandler.extract_index(f, "/raw/var")
                        else:
                            genes = H5ADHandler.extract_index(f, "/var")
                        if '/raw/obs' in f:   
                            cells = H5ADHandler.extract_index(f, "/raw/obs")
                        else:
                            cells = H5ADHandler.extract_index(f, "/obs")
                    else:
                        genes = H5ADHandler.extract_index(f, "/var")
                        cells = H5ADHandler.extract_index(f, "/obs")
                    
                    entry = {
                        "group": group,
                        "nber_cols": nCells,
                        "nber_rows": nGenes,
                        "is_count": int((np.array(matrix) % 1 == 0).all()),
                        "genes": genes,
                        "cells": cells,
                        "matrix": matrix
                    }
                    result["list_groups"].append(entry)

        json_str = json.dumps(result, ensure_ascii=False)

        if args.o:
            with open(os.path.join(args.o, "output.json"), "w", encoding="utf-8") as out:
                out.write(json_str)
        else:
            print(json_str)

class LoomHandler:
    @staticmethod
    def decode_list(arr):
        return [v.decode() if isinstance(v, bytes) else str(v) for v in arr]
    
    @staticmethod
    def preparse(args, file_path):
        result = {
            "detected_format": FileType.LOOM,
            "loom_version": "unknown",
            "file_path": file_path, # If it was compressed, it should be decompressed before
            "list_groups": []
        }

        with h5py.File(file_path, 'r') as f:
            if '/matrix' in f:
                shape = f['/matrix'].shape
                nGenes, nCells = int(shape[0]), int(shape[1])

                max_genes = min(10, nGenes)
                max_cells = min(10, nCells)
                
                # Extract 10x10 matrix (Dense)
                matrix = f['/matrix'][:max_genes, :max_cells].astype(float).tolist()

                # Loom version
                if '/attrs/LOOM_SPEC_VERSION' in f:
                    ds = f['/attrs/LOOM_SPEC_VERSION']
                    val = ds[()]  # read scalar dataset
                    result["loom_version"] = val.decode() if isinstance(val, bytes) else str(val)
                elif 'LOOM_SPEC_VERSION' in f.attrs: # e.g. 2.0.0
                    version = f.attrs['LOOM_SPEC_VERSION']
                    result["loom_version"] = version.decode() if isinstance(version, bytes) else str(version)
                elif 'version' in f.attrs: # e.g. 0.2.0
                    version = f.attrs['version']
                    result["loom_version"] = version.decode() if isinstance(version, bytes) else str(version)
                
                # Load gene names
                if '/row_attrs/Gene' in f:   
                    genes = LoomHandler.decode_list(f['/row_attrs/Gene'][:max_genes])
                elif '/row_attrs/Accession' in f:   
                    genes = LoomHandler.decode_list(f['/row_attrs/Accession'][:max_genes])
                elif '/row_attrs/id' in f: # Very old
                    genes = LoomHandler.decode_list(f['/row_attrs/id'][:max_genes])
                elif '/row_attrs/GeneName' in f: # Seen in few datasets
                    genes = LoomHandler.decode_list(f['/row_attrs/GeneName'][:max_genes])
                else:
                    genes = [f"Gene_{i+1}" for i in range(max_genes)]
                
                # Load cell names
                if '/col_attrs/CellID' in f:   
                    cells = LoomHandler.decode_list(f['/col_attrs/CellID'][:max_cells])
                elif '/col_attrs/id' in f:
                    cells = LoomHandler.decode_list(f['/col_attrs/id'][:max_cells])
                else:
                    cells = [f"Cell_{i+1}" for i in range(max_cells)]
  
                entry = {
                    "group": "/matrix",
                    "nber_cols": nCells,
                    "nber_rows": nGenes,
                    "is_count": int((np.array(matrix) % 1 == 0).all()),
                    "genes": genes,
                    "cells": cells,
                    "matrix": matrix
                }
                result["list_groups"].append(entry)

        json_str = json.dumps(result, ensure_ascii=False)

        if args.o:
            with open(os.path.join(args.o, "output.json"), "w", encoding="utf-8") as out:
                out.write(json_str)
        else:
            print(json_str)

class RdsHandler:
    @staticmethod
    def preparse(args, file_path):
        result = {
            "detected_format": FileType.RDS,
            "file_path": file_path, # Does not change for RDS files
            "list_groups": []
        }
        
        # Import rpy2 for reading R objects in Python
        import rpy2.rinterface_lib.callbacks
        rpy2.rinterface_lib.callbacks.consolewrite_warnerror = lambda *args: None # Suppress console output from R
        from rpy2.robjects.packages import importr
        from rpy2.robjects import r
        
        # Import base package for reading RDS files
        base = importr('base')

        # Load R object
        obj = base.readRDS(file_path)
        obj_class = r['class'](obj)[0]
        
        # Check its type and apply conversion
        if obj_class == "Seurat":
            Matrix = importr('Matrix')
            Seurat = importr('Seurat')
            
            # 10x10 matrix
            matrix = np.array(Matrix.as_matrix(r('function(obj) GetAssayData(obj, slot = "count")[1:10, 1:10]')(obj))).tolist()
            
            # Global dims
            nCells = int(list(r['ncol'](obj))[0])
            nGenes = int(list(r['nrow'](obj))[0])
            
            # Gene and cell names
            genes = list(r['rownames'](obj)[1:10])
            cells = list(r['colnames'](obj)[1:10])
                        
            entry = {
                "group": "seurat_object",
                "nber_cols": nCells,
                "nber_rows": nGenes,
                "is_count": int((np.array(matrix) % 1 == 0).all()),
                "genes": genes,
                "cells": cells,
                "matrix": matrix
            }
            result["list_groups"].append(entry)
        elif obj_class == "data.frame":
            print("Data Frame")
            from rpy2.robjects import pandas2ri
            pandas2ri.activate()
            df = pandas2ri.rpy2py(obj)
            n_rows = len(df)
            entry = {
                "group": "data_frame",
                "nber_cols": nCells,
                "nber_rows": nGenes,
                "is_count": int((np.array(matrix) % 1 == 0).all()),
                "genes": genes,
                "cells": cells,
                "matrix": matrix
            }
            result["list_groups"].append(entry)
        
        json_str = json.dumps(result, ensure_ascii=False)

        if args.o:
            with open(os.path.join(args.o, "output.json"), "w", encoding="utf-8") as out:
                out.write(json_str)
        else:
            print(json_str)
 
class ArchiveHandler:
    @staticmethod
    def write_listing_json(args, file_path, list_files):
        result = {
            "detected_format": FileType.ARCHIVE,
            "file_path": file_path, # Path of the archive (should be decompressed already)
            "list_files": [{"filename": f} for f in list_files]
        }
        json_str = json.dumps(result, ensure_ascii=False)
        if args.o:
            with open(os.path.join(args.o, "output.json"), "w", encoding="utf-8") as out:
                out.write(json_str)
        else:
            print(json_str)

class MtxHandler:
    @staticmethod
    def preparse(args, file_paths):
        result = {
            "detected_format": FileType.MTX,
            "file_path": file_paths, # Could have been unzipped/unarchived
            "list_groups": []
        }
        
        # Cell names
        cells = [f"Cell_{i+1}" for i in range(10)]
        barcode_path = next((p for p in file_paths if Path(p).name == "barcodes.tsv"), None)
        print(file_paths)
        if barcode_path:
            with open(barcode_path, "r") as f:
                cells = [line.strip().split('\t')[0] for _, line in zip(range(10), f)]
                
        # Gene names
        genes = [f"Gene_{i+1}" for i in range(10)]
        feature_path = next((p for p in file_paths if Path(p).name == "features.tsv"), None)
        if not feature_path:
            feature_path = next((p for p in file_paths if Path(p).name == "genes.tsv"), None)
        if feature_path:
            with open(feature_path, "r") as f:
                genes = [line.strip().split('\t')[0] for _, line in zip(range(10), f)]
        
        # MTX file
        matrix_path = next((p for p in file_paths if Path(p).name not in {"barcodes.tsv", "features.tsv", "genes.tsv"}), None)
        if not matrix_path: # This should not happen
            ErrorJSON("No matrix found?")

        # Read first 10 rows
        nGenes = -1
        nCells = -1
        # Check MTX format
        if "pattern" in open(matrix_path).readline():
            pattern = True
        else:
            pattern = False
        # Parse file
        with open(matrix_path, 'r') as f:
            # Skip comments and header
            for line in f:
                if line.startswith('%'):
                    continue
                else:
                    nGenes, nCells, n_nz = map(int, line.strip().split())
                    break

            matrix = np.zeros((min(nGenes, 10), min(nCells, 10)))
            for line in f:
                parts = line.strip().split()
                if pattern:
                    if len(parts) != 2:
                        continue
                    i, j = map(int, parts)
                    val = 1.0
                else:
                    if len(parts) != 3:
                        continue
                    i, j = map(int, parts[:2])
                    val = float(parts[2])

                i -= 1
                j -= 1
                if i < 10 and j < 10:
                    matrix[i, j] = val
            
        # Restrict genes and cells to correct values
        cells = cells[:min(nCells, 10)]
        genes = genes[:min(nGenes, 10)]

        entry = {
                "group": Path(matrix_path).name,
                "nber_cols": nCells,
                "nber_rows": nGenes,
                "is_count": int((np.array(matrix) % 1 == 0).all()),
                "genes": genes,
                "cells": cells,
                "matrix": matrix.tolist()
            }
        result["list_groups"].append(entry)
        
        json_str = json.dumps(result, ensure_ascii=False)

        if args.o:
            with open(os.path.join(args.o, "output.json"), "w", encoding="utf-8") as out:
                out.write(json_str)
        else:
            print(json_str)


class TextHandler:
    @staticmethod
    def preparse(args, file_path):
        result = {
            "warnings":[],
            "detected_format": FileType.RAW_TEXT,
            "file_path": file_path, # Could have been unzipped/unarchived
            "list_groups": []
        }
        import pandas as pd
        import warnings
        
        # Count n_rows
        nGenes = count_nonempty_lines(file_path)

        # Header
        param_header = None
        if args.header == 'true':
            param_header = 0
            nGenes = nGenes - 1
        
        # Row indexes
        if args.col == 'first':
            param_col = 0
        else:
            param_col = False
            
        # Decode escape characters in separator
        param_delim = bytes(args.delim, "utf-8").decode("unicode_escape")
        
        # Read first 10 rows
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", pd.errors.ParserWarning)
            try:
                matrix = pd.read_csv(file_path, sep=param_delim, header=param_header, index_col=param_col, nrows=10, engine='c')
            except pd.errors.ParserError as e:
                error_message = str(e).strip()
                if 'Error tokenizing data' in error_message:
                    error_message += '. Did you forget the header?'
                ErrorJSON(error_message)
            except ValueError as e:
                error_message = str(e).strip()
                if 'separators > 1 char' in error_message:
                    error_message += '. Did you use multiple separator characters?'
                ErrorJSON(error_message)
            
            for warning in w:
                if issubclass(warning.category, pd.errors.ParserWarning):
                    result["warnings"].append(f"WARNING: {str(warning.message)}")

        if args.col == 'last':
            matrix.index = matrix.iloc[:, -1]      # Set last column as row index
            matrix = matrix.iloc[:, :-1]           # Remove last column

        if args.col == 'none':
            matrix.index = [f"Gene_{i+1}" for i in range(len(matrix))]
            
        if param_header == None:
            matrix.columns = [f"Cell_{i+1}" for i in range(len(matrix.columns))]

        if matrix.shape[1] == 0:
            result["warnings"].append("No column is detected? Did you set the 'delimiter' parameter correctly?")
            
        if matrix.shape[1] == 1:
            result["warnings"].append("Only one column is detected? Did you set the 'delimiter' parameter correctly?")

        if not pd.api.types.is_numeric_dtype(matrix.values):
            result["warnings"].append("Matrix is not numeric, is header / column name set correctly?")
            is_count = int(0) # Or NA?
        else:
            is_count = int((np.array(matrix) % 1 == 0).all())

        entry = {
            "group": "text_file",
            "nber_cols": int(matrix.shape[1]),
            "nber_rows": int(nGenes),
            "is_count": is_count,
            "genes": matrix.index[0:10].tolist(),
            "cells": matrix.columns[0:10].tolist(),
            "matrix": np.array(matrix.iloc[0:10,0:10]).tolist()
        }
        result["list_groups"].append(entry)

        if not matrix.index.is_unique:
            result["warnings"].append("Gene names are not unique. Did you set the 'column index' parameter correctly?")

        # Remove warnings of results if empty
        if not result["warnings"]:
            del result["warnings"]
        
        # Write JSON output
        json_str = json.dumps(result, ensure_ascii=False)
        if args.o:
            with open(os.path.join(args.o, "output.json"), "w", encoding="utf-8") as out:
                out.write(json_str)
        else:
            #print("toto")
            print(json_str)

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

def main():
    if '--help' in sys.argv:
        print(custom_help)
        sys.exit(0)

    parser = argparse.ArgumentParser(description='Preparsing Mode Script', add_help=False)
    parser.add_argument('-f', metavar='INPUT_FILE', required=True)
    parser.add_argument('-o', metavar='OUTPUT_FOLDER', required=False)
    #parser.add_argument('--organism', type=int, required=False)
    parser.add_argument('--header', choices=['true', 'false'], default='true', required=False)
    parser.add_argument('--col', choices=['none', 'first', 'last'], default='first', required=False)
    parser.add_argument('--sel', default=None)
    parser.add_argument('--delim', default='\t')
    #parser.add_argument('--row-names', default=None)
    #parser.add_argument('--col-names', default=None)
    #parser.add_argument('--host', default='postgres:5434')

    args = parser.parse_args()

    if not os.path.isfile(args.f):
        ErrorJSON(f"Input file not found: {args.f}")
    if args.o and not os.path.isdir(args.o):
        os.makedirs(args.o, exist_ok=True)

    preparse(args)

if __name__ == '__main__':
    main()

