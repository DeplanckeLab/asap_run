import argparse
import sys
import os
import h5py
#import loompy
import zipfile
import tarfile
import gzip
import bz2
import csv
import json
import subprocess
import io
import numpy as np
from pathlib import Path
from dataclasses import dataclass

custom_help = """
Preparsing Mode

Options:
  -f %s             File to preparse.
  -o %s             Output folder.
  --organism %i     ID of the organism.
  --header %s       The file has a header [true, false].
  --col %s          Name Column [none, first, last].
  --sel %s          Name of entry to load from archive or HDF5 (if multiple groups).
  --delim %s        Delimiter (default: tab).
  --row-names %s    Metadata column for row names of the main matrix.
  --col-names %s    Metadata column for column names of the main matrix.
  --host %s         Host override (default: postgres:5434).
  --help            Show this help message and exit.
"""

class FileType:
    H5_10X = 'H5_10X'
    H5AD = 'H5AD'
    LOOM = 'LOOM'
    ARCHIVE = 'ARCHIVE'
    RAW_TEXT = 'RAW_TEXT'
    UNKNOWN = 'UNKNOWN'

def decompress_if_needed(file_path):
    suffix = Path(file_path).suffix.lower()
    if suffix == '.gz':
        p = subprocess.Popen(['pigz', '-dc', file_path], stdout=subprocess.PIPE, bufsize=10**8)
        decompressed_data = p.stdout.read()
        p.stdout.close()
        p.wait()
        return io.BytesIO(decompressed_data)
    elif suffix in ['.bz2', '.bzip2']:
        return bz2.open(file_path, 'rb')
    elif suffix == '.zip':
        with zipfile.ZipFile(file_path, 'r') as z:
            names = z.namelist()
            if len(names) != 1:
                raise ValueError(f"ZIP archive must contain exactly one file, found: {names}")
            return io.BytesIO(z.read(names[0]))
    return open(file_path, 'rb')

def check_file_type_from_stream(stream):
    # Ensure stream at start
    stream.seek(0)
    # Try reading as HDF5
    try:
        with h5py.File(stream, 'r') as f:
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
        pass

    # Default to raw text
    stream.seek(0)
    return FileType.RAW_TEXT

def list_archive_contents(file_path):
    if zipfile.is_zipfile(file_path):
        with zipfile.ZipFile(file_path, 'r') as z:
            return z.namelist()
    elif tarfile.is_tarfile(file_path):
        with tarfile.open(file_path, 'r') as t:
            return t.getnames()
    return None

def count_nonempty_lines(buffered_reader):
    return sum(1 for line in buffered_reader if line.strip())

def preparse(args):
    with decompress_if_needed(args.f) as stream:
        try:
            file_type = check_file_type_from_stream(stream)
        except Exception:
            file_type = FileType.RAW_TEXT
        stream.seek(0)

        if file_type == FileType.H5_10X:
            H510xHandler.preparse(args, stream)
        elif file_type == FileType.H5AD:
            H5ADHandler.preparse(args, stream)
        elif file_type == FileType.LOOM:
            LoomHandler.preparse(args, stream)
        else:
            # Archive and plain text case
            if not args.sel:
                files = list_archive_contents(args.f)
                if files is None: # Try plain text
                    try:
                        text_stream = io.TextIOWrapper(stream)
                        n_rows = count_nonempty_lines(text_stream)
                        stream.seek(0)
                        TextHandler.preparse(args, stream)
                        #preparse_text(n_rows, io.TextIOWrapper(stream))
                    except ValueError:
                        ErrorJSON("Non-numeric values detected. Did you forget a header?")
                    except IOError:
                        ErrorJSON("Failed to read file as plain text.")
                elif len(files) == 0: # Empty archive
                    ErrorJSON("Archive is empty or corrupted.")
                elif len(files) == 1: # Archive with only one file
                    args.sel = files[0]
                    with open(args.f, 'rb') as archive:
                        reader = ArchiveHandler.get_reader(archive, args.sel)
                        n_rows = count_nonempty_lines(io.TextIOWrapper(reader))
                        reader.seek(0)
                        #preparse_text(n_rows, io.TextIOWrapper(reader))
                        TextHandler.preparse(args, stream)
                else: # Archive with multiple files
                    ArchiveHandler.write_listing_json(files, FileType.ARCHIVE, args)
            else:
                with open(args.f, 'rb') as archive:
                    reader = ArchiveHandler.get_reader(archive, args.sel)
                    n_rows = count_nonempty_lines(io.TextIOWrapper(reader))
                    reader.seek(0)
                    preparse_text(n_rows, io.TextIOWrapper(reader))

class H510xHandler:
    @staticmethod
    def preparse(args, stream):
        # build the JSON-able structure
        result = {
            "detected_format": FileType.H5_10X,
            "list_groups": []
        }
        
        # Detect groups
        with h5py.File(stream, 'r') as f:
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
    def preparse(args, stream):
        result = {
            "detected_format": FileType.H5AD,
            "list_groups": []
        }

        with h5py.File(stream, 'r') as f:
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
    def preparse(args, stream):
        print("LOOM")
        pass

class ArchiveHandler:
    @staticmethod
    def get_reader(archive, member):
        if zipfile.is_zipfile(archive.name):
            with zipfile.ZipFile(archive) as z:
                return z.open(member, 'r')
        elif tarfile.is_tarfile(archive.name):
            with tarfile.open(fileobj=archive) as t:
                return t.extractfile(member)
        ErrorJSON("Unsupported archive format")
    
    @staticmethod
    def write_listing_json(files, file_type, args, loom_version=None):
        data = {
            "detected_format": file_type,
            "list_files": [{"filename": f} for f in files]
        }
        if file_type == "LOOM" and loom_version is not None:
            data["loom_version"] = loom_version

        output_path = os.path.join(args.o, "output.json")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except IOError as ioe:
           ErrorJSON(f"JSON writing error: {ioe}")

class TextHandler:
    @staticmethod
    def preparse(args, stream):
        print("TEXT")
        reader = csv.reader(br, delimiter=Parameters.delimiter)
        header = next(reader, None)
        if not header:
            ErrorJSON(f"Empty file or invalid encoding. Format: {Parameters.fileType}")

        g = GroupPreparse(Parameters.selection or Parameters.fileName.split("/")[-1])
        tokens = None

        if Parameters.has_header:
            data_row = next(reader, None)
            if not data_row:
                ErrorJSON(f"No data rows in file. Format: {Parameters.fileType}")
            tokens = data_row
            g.nbGenes = n_rows - 1
        else:
            tokens = header
            g.nbGenes = n_rows
            header = None

        # Determine number of cells
        if Parameters.name_column in ("FIRST", "LAST"):
            base_len = len(header) if header else len(tokens) + 1
            g.nbCells = base_len - 1
        else:  # NONE
            g.nbCells = len(header) if header else len(tokens)

        # Assign cellNames if header exists
        if header:
            if Parameters.name_column == "FIRST":
                start = 1
                expected_len = g.nbCells + 1
            elif Parameters.name_column == "LAST":
                start = 0
                expected_len = g.nbCells + 1
            else:  # NONE
                start = 0
                expected_len = g.nbCells
            if len(header) < start + min(10, g.nbCells):
                ErrorJSON("Header length mismatch.")
            g.cellNames = [header[i].replace('"', "")
                           for i in range(start, start + min(10, g.nbCells))]

        # Initialize storage limited to first 10 × 10
        G = min(10, g.nbGenes)
        C = min(10, g.nbCells)
        if Parameters.name_column != "NONE":
            g.geneNames = ["" for _ in range(G)]
        g.matrix = [[0.0]*C for _ in range(G)]

        # Process rows
        for i in range(G):
            if not tokens or len(tokens) != (g.nbCells + (1 if Parameters.name_column!="NONE" else 0)):
                ErrorJSON(f"Row {i + (2 if header else 1)} element count mismatch.")
            values = tokens.copy()
            if Parameters.name_column == "FIRST":
                g.geneNames[i] = values.pop(0).replace('"', "")
            elif Parameters.name_column == "LAST":
                g.geneNames[i] = values.pop(-1).replace('"', "")
            # NONE means no geneNames
            
            for j in range(C):
                val = float(values[j].replace(",", "."))
                g.matrix[i][j] = val
                g.isCount &= val.is_integer()

            tokens = next(reader, None)

        # Write JSON output
        ArchiveHandler.write_output_json([g])

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
    parser.add_argument('--organism', type=int, required=False)
    parser.add_argument('--header', choices=['true', 'false'], required=False)
    parser.add_argument('--col', choices=['none', 'first', 'last'], required=False)
    parser.add_argument('--sel', default=None)
    parser.add_argument('--delim', default='\t')
    parser.add_argument('--row-names', default=None)
    parser.add_argument('--col-names', default=None)
    parser.add_argument('--host', default='postgres:5434')

    args = parser.parse_args()

    if not os.path.isfile(args.f):
        ErrorJSON(f"Input file not found: {args.f}")
    if args.o and not os.path.isdir(args.o):
        os.makedirs(args.o, exist_ok=True)

    preparse(args)

if __name__ == '__main__':
    main()

