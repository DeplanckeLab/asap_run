import argparse # Needed to parse arguments
import sys # Needed for ErrorJSON
import os # Needed for pathjoin and path/create dirs
import h5py # Needed for parsing hdf5 files
import json # For writing output files
import numpy as np # For diverse array calculations
import tarfile # To check if file is a TAR archive
from pathlib import Path # Needed for file extension manipulation
import shutil # For moving files and removing directories

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

#  --metadata        Export metadata fields (absent: false). Only working with certain formats.

class FileType:
    H5_10X = 'H5_10X'
    H5AD = 'H5AD'
    LOOM = 'LOOM'
    RDS = 'RDS'
    MTX = 'MTX'
    ARCHIVE = 'ARCHIVE'
    RAW_TEXT = 'RAW_TEXT'
    UNKNOWN = 'UNKNOWN'

def cleanup_empty_dirs(base_path):
    """Remove empty directories recursively."""
    base_path = Path(base_path)
    if not base_path.exists() or not base_path.is_dir():
        return
    
    for item in sorted(base_path.rglob('*'), reverse=True):
        if item.is_dir() and not any(item.iterdir()):
            try:
                item.rmdir()
            except:
                pass

def is_mtx_triplet(file_list):
    """
    Check if file list contains exactly ONE MTX triplet.
    Returns True only if there's a single complete triplet (3 files).
    Multiple triplets should be treated as a regular archive requiring --sel.
    
    Supported patterns:
    1. Standard: matrix.mtx, barcodes.tsv, features.tsv/genes.tsv
    2. Prefixed: PREFIX-matrix.mtx, PREFIX-barcodes.tsv, PREFIX-features.tsv
    3. Alternative: *counts*.mtx, *cell_metadata*.csv, *gene_names*.csv
    """
    def base_name(path):
        """Get base name without compression extensions."""
        p = Path(path).name
        suffix = Path(p).suffix
        if suffix in {".gz", ".bz2", ".xz"}:
            p = p[: -len(suffix)]
        return p
    
    # Standard triplet check (exact names) - must be exactly 3 files
    if len(file_list) == 3:
        base_names = {base_name(p) for p in file_list}
        allowed_sets = [
            {"matrix.mtx", "barcodes.tsv", "features.tsv"},
            {"matrix.mtx", "barcodes.tsv", "genes.tsv"}
        ]
        if base_names in allowed_sets:
            return True
    
    # If not exactly 3 files, check if it's a single prefixed/variant triplet
    # Group files by prefix
    from collections import defaultdict
    groups = defaultdict(lambda: {'matrix': None, 'barcodes': None, 'features': None})
    
    for fpath in file_list:
        fname = base_name(fpath)
        fname_lower = fname.lower()
        
        # Check for matrix file (.mtx extension)
        if fname.endswith('.mtx'):
            # Pattern 1: Standard "matrix.mtx"
            if fname_lower == 'matrix.mtx':
                groups['']['matrix'] = fpath
            # Pattern 2: Prefixed "PREFIX-matrix.mtx" or "PREFIX_matrix.mtx"
            elif 'matrix' in fname_lower:
                for sep in ['-matrix', '_matrix']:
                    if sep in fname_lower:
                        prefix = fname_lower.split(sep)[0]
                        groups[prefix]['matrix'] = fpath
                        break
            # Pattern 3: "counts" variant (*counts*.mtx)
            elif 'count' in fname_lower:
                # Extract common prefix (everything before 'count')
                prefix = fname_lower.split('count')[0].rstrip('_-')
                groups[prefix]['matrix'] = fpath
        
        # Check for barcodes/cell metadata file (.tsv or .csv extension)
        elif fname.endswith('.tsv') or fname.endswith('.csv'):
            # Pattern 1: Standard "barcodes.tsv"
            if fname_lower in ('barcodes.tsv', 'barcode.tsv'):
                groups['']['barcodes'] = fpath
            # Pattern 2: Prefixed barcodes
            elif 'barcode' in fname_lower:
                for sep in ['-barcode', '_barcode']:
                    if sep in fname_lower:
                        prefix = fname_lower.split(sep)[0]
                        groups[prefix]['barcodes'] = fpath
                        break
            # Pattern 3: "cell_metadata" variant
            elif 'cell' in fname_lower and 'metadata' in fname_lower:
                # Extract common prefix (everything before 'cell')
                prefix = fname_lower.split('cell')[0].rstrip('_-')
                groups[prefix]['barcodes'] = fpath
            # Pattern 4: Features/genes
            elif 'feature' in fname_lower:
                if fname_lower in ('features.tsv', 'features.csv'):
                    groups['']['features'] = fpath
                else:
                    for sep in ['-feature', '_feature']:
                        if sep in fname_lower:
                            prefix = fname_lower.split(sep)[0]
                            groups[prefix]['features'] = fpath
                            break
            elif 'gene' in fname_lower:
                if fname_lower in ('genes.tsv', 'genes.csv', 'gene_names.csv'):
                    groups['']['features'] = fpath
                else:
                    for sep in ['-gene', '_gene']:
                        if sep in fname_lower:
                            prefix = fname_lower.split(sep)[0]
                            groups[prefix]['features'] = fpath
                            break
                    # Pattern 3: "gene_names" variant
                    if groups[fname_lower.split('gene')[0].rstrip('_-')]['features'] is None:
                        prefix = fname_lower.split('gene')[0].rstrip('_-')
                        groups[prefix]['features'] = fpath
    
    # Count complete triplets
    complete_triplets = [prefix for prefix, files in groups.items() 
                         if files['matrix'] and files['barcodes'] and files['features']]
    
    # Only return True if there's exactly ONE complete triplet
    return len(complete_triplets) == 1 and len(file_list) == 3

def find_mtx_triplet_for_file(selected_file, all_files):
    """
    Given a selected file (e.g., matrix.mtx), find its companion triplet files.
    Returns list of [matrix, barcodes, features] if found, else None.
    
    Supports multiple naming patterns:
    1. Standard: matrix.mtx, barcodes.tsv, features.tsv/genes.tsv
    2. Prefixed: PREFIX-matrix.mtx, PREFIX-barcodes.tsv, PREFIX-features.tsv
    3. Alternative: *counts*.mtx, *cell_metadata*.csv, *gene_names*.csv
    """
    def base_name(path):
        """Get base name without compression extensions."""
        p = Path(path).name
        suffix = Path(p).suffix
        if suffix in {".gz", ".bz2", ".xz"}:
            p = p[: -len(suffix)]
        return p
    
    selected_base = base_name(selected_file)
    selected_lower = selected_base.lower()
    
    # Detect what type of file was selected and extract prefix
    prefix = None
    file_type = None
    
    # Check for matrix file
    if selected_base.endswith('.mtx'):
        file_type = 'matrix'
        if selected_lower == 'matrix.mtx':
            prefix = ''
        elif 'matrix' in selected_lower:
            for sep in ['-matrix', '_matrix']:
                if sep in selected_lower:
                    prefix = selected_lower.split(sep)[0]
                    break
        elif 'count' in selected_lower:
            prefix = selected_lower.split('count')[0].rstrip('_-')
    
    # Check for barcodes/cell metadata file
    elif selected_base.endswith('.tsv') or selected_base.endswith('.csv'):
        if selected_lower in ('barcodes.tsv', 'barcode.tsv'):
            file_type = 'barcodes'
            prefix = ''
        elif 'barcode' in selected_lower:
            file_type = 'barcodes'
            for sep in ['-barcode', '_barcode']:
                if sep in selected_lower:
                    prefix = selected_lower.split(sep)[0]
                    break
        elif 'cell' in selected_lower and 'metadata' in selected_lower:
            file_type = 'barcodes'
            prefix = selected_lower.split('cell')[0].rstrip('_-')
        elif selected_lower in ('features.tsv', 'features.csv', 'genes.tsv', 'genes.csv', 'gene_names.csv'):
            file_type = 'features'
            prefix = ''
        elif 'feature' in selected_lower:
            file_type = 'features'
            for sep in ['-feature', '_feature']:
                if sep in selected_lower:
                    prefix = selected_lower.split(sep)[0]
                    break
        elif 'gene' in selected_lower:
            file_type = 'features'
            for sep in ['-gene', '_gene']:
                if sep in selected_lower:
                    prefix = selected_lower.split(sep)[0]
                    break
            if prefix is None:
                prefix = selected_lower.split('gene')[0].rstrip('_-')
    
    if prefix is None or file_type is None:
        return None
    
    # Now find the companion files with the same prefix
    triplet = {'matrix': None, 'barcodes': None, 'features': None}
    
    for fpath in all_files:
        fname = base_name(fpath)
        fname_lower = fname.lower()
        
        # Check for matrix file
        if fname.endswith('.mtx'):
            if fname_lower == 'matrix.mtx' and prefix == '':
                triplet['matrix'] = fpath
            elif 'matrix' in fname_lower:
                for sep in ['-matrix', '_matrix']:
                    if sep in fname_lower:
                        file_prefix = fname_lower.split(sep)[0]
                        if file_prefix == prefix:
                            triplet['matrix'] = fpath
                        break
            elif 'count' in fname_lower:
                file_prefix = fname_lower.split('count')[0].rstrip('_-')
                if file_prefix == prefix:
                    triplet['matrix'] = fpath
        
        # Check for barcodes/cell metadata file
        elif fname.endswith('.tsv') or fname.endswith('.csv'):
            if fname_lower in ('barcodes.tsv', 'barcode.tsv') and prefix == '':
                triplet['barcodes'] = fpath
            elif 'barcode' in fname_lower:
                for sep in ['-barcode', '_barcode']:
                    if sep in fname_lower:
                        file_prefix = fname_lower.split(sep)[0]
                        if file_prefix == prefix:
                            triplet['barcodes'] = fpath
                        break
            elif 'cell' in fname_lower and 'metadata' in fname_lower:
                file_prefix = fname_lower.split('cell')[0].rstrip('_-')
                if file_prefix == prefix:
                    triplet['barcodes'] = fpath
            # Check for features/genes
            elif fname_lower in ('features.tsv', 'features.csv', 'genes.tsv', 'genes.csv', 'gene_names.csv') and prefix == '':
                triplet['features'] = fpath
            elif 'feature' in fname_lower:
                for sep in ['-feature', '_feature']:
                    if sep in fname_lower:
                        file_prefix = fname_lower.split(sep)[0]
                        if file_prefix == prefix:
                            triplet['features'] = fpath
                        break
            elif 'gene' in fname_lower:
                for sep in ['-gene', '_gene']:
                    if sep in fname_lower:
                        file_prefix = fname_lower.split(sep)[0]
                        if file_prefix == prefix:
                            triplet['features'] = fpath
                        break
                if triplet['features'] is None:
                    file_prefix = fname_lower.split('gene')[0].rstrip('_-')
                    if file_prefix == prefix:
                        triplet['features'] = fpath
    
    # Return triplet if complete
    if triplet['matrix'] and triplet['barcodes'] and triplet['features']:
        return [triplet['matrix'], triplet['barcodes'], triplet['features']]
    
    return None

def extract_and_process_mtx_triplet(archive_obj, file_paths, extract_dir, original_input=None):
    """Extract and process MTX triplet files."""
    canonical = {
        "matrix.mtx": "matrix.mtx", "matrix": "matrix.mtx",
        "barcodes.tsv": "barcodes.tsv", "barcodes": "barcodes.tsv",
        "features.tsv": "features.tsv", "features": "features.tsv",
        "genes.tsv": "genes.tsv", "genes": "genes.tsv"
    }
    
    def base_name(path):
        p = Path(path).name
        suffix = Path(p).suffix
        if suffix in {".gz", ".bz2", ".xz"}:
            p = p[: -len(suffix)]
        return p
    
    def get_canonical_name(fname):
        """Determine canonical name for MTX triplet file."""
        base = base_name(fname)
        base_lower = base.lower()
        
        # Check if it's already a canonical name
        if base in canonical:
            return canonical[base]
        
        # Check for matrix files (.mtx extension)
        if base.endswith('.mtx'):
            return 'matrix.mtx'
        
        # Check for barcodes/cell metadata files (.tsv or .csv)
        elif base.endswith('.tsv') or base.endswith('.csv'):
            # Preserve the original extension
            extension = Path(base).suffix  # .tsv or .csv
            if 'barcode' in base_lower or ('cell' in base_lower and 'metadata' in base_lower):
                return f'barcodes{extension}'
            elif 'feature' in base_lower or 'gene' in base_lower:
                return f'features{extension}'
        
        # Fallback - should not happen if triplet detection worked correctly
        return base
    
    final_file_paths = []
    for fname in file_paths:
        # Extract to the designated directory
        if isinstance(archive_obj, tarfile.TarFile):
            archive_obj.extract(fname, path=extract_dir, filter='data')
        else:  # zipfile
            archive_obj.extract(fname, path=extract_dir)
        
        extracted_path = Path(extract_dir) / fname
        # Decompress if needed
        decompressed_path, _ = decompress_if_needed(str(extracted_path), None, original_input)
        
        # Delete the original compressed file if decompression happened
        if str(extracted_path) != decompressed_path[0]:
            try:
                extracted_path.unlink()
            except:
                pass
        
        # Move to target location with canonical name (flatten structure)
        target_base = get_canonical_name(fname)
        target_path = Path(extract_dir) / target_base
        if Path(decompressed_path[0]) != target_path:
            shutil.move(decompressed_path[0], target_path)
        final_file_paths.append(str(target_path))
    
    # Clean up any subdirectories that were created during extraction
    for item in extract_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
    
    return final_file_paths

def decompress_if_needed(file_path, sel, original_input=None):
    """
    Decompress files as needed. Returns (list_of_files, archive_path).
    - For single-file compression (gz, bz2, xz): decompress and recurse
    - For archives (zip, tar): handle according to content
    - original_input: The original -f argument file path (never delete this)
    """
    # Track original input if this is the first call
    if original_input is None:
        original_input = file_path
    
    with open(file_path, 'rb') as f:
        magic = f.read(264)

    # gzip (single file compression)
    if magic.startswith(b'\x1f\x8b'):
        import subprocess
        output_path = Path(file_path).with_suffix('')
        if output_path == Path(file_path):
            output_path = output_path.with_name(output_path.name + '.out')
        with open(output_path, 'wb') as f_out:
            subprocess.run(["pigz", "-d", '-k', '-f', "-c", file_path], stdout=f_out, check=True)
        return decompress_if_needed(str(output_path), sel, original_input)

    # bzip2 (single file compression)
    if magic.startswith(b'BZh'):
        import subprocess
        output_path = Path(file_path).with_suffix('')
        if output_path == Path(file_path):
            output_path = output_path.with_name(output_path.name + '.out')
        with open(output_path, 'wb') as f_out:
            subprocess.run(["pbzip2", "-d", "-k", "-f", "-c", file_path], stdout=f_out, check=True)
        return decompress_if_needed(str(output_path), sel, original_input)
        
    # xz (single file compression)
    if magic.startswith(b'\xfd7zXZ\x00'):
        import subprocess
        output_path = Path(file_path).with_suffix('')
        if output_path == Path(file_path):
            output_path = output_path.with_name(output_path.name + '.out')
        with open(output_path, 'wb') as f_out:
            subprocess.run(["pixz", "-d", "-k", "-c", file_path], stdout=f_out, check=True)
        return decompress_if_needed(str(output_path), sel, original_input)

    # zip (multi-file archive)
    if magic.startswith(b'PK\x03\x04'):
        import zipfile
        with zipfile.ZipFile(file_path, 'r') as z:
            file_paths = [f for f in z.namelist() if not f.endswith('/')]
            
            # Single file in archive
            if len(file_paths) == 1:
                extract_dir = Path(file_path).parent / Path(file_paths[0]).name.replace('.', '_')
                extract_dir.mkdir(exist_ok=True)
                z.extract(file_paths[0], path=extract_dir)
                extracted_path = extract_dir / file_paths[0]
                final_path = extract_dir / Path(file_paths[0]).name
                if extracted_path != final_path:
                    shutil.move(extracted_path, final_path)
                cleanup_empty_dirs(extract_dir)
                return decompress_if_needed(str(final_path), sel, original_input)
            
            # MTX triplet without --sel
            if not sel and is_mtx_triplet(file_paths):
                archive_stem = Path(file_path).stem
                if archive_stem.endswith('.tar'):
                    archive_stem = archive_stem[:-4]
                extract_dir = Path(file_path).parent / archive_stem
                extract_dir.mkdir(exist_ok=True)
                final_paths = extract_and_process_mtx_triplet(z, file_paths, extract_dir, original_input)
                return final_paths, None
            
            # Multiple files with --sel
            if sel:
                if sel not in file_paths:
                    ErrorJSON(f"Your selected file '{sel}' is not in the list of files of the archive: {file_paths}")
                
                # Check if selected file is part of an MTX triplet
                triplet_files = find_mtx_triplet_for_file(sel, file_paths)
                
                if triplet_files:
                    # Extract the entire triplet
                    archive_stem = Path(file_path).stem
                    if archive_stem.endswith('.tar'):
                        archive_stem = archive_stem[:-4]
                    extract_dir = Path(file_path).parent / archive_stem
                    extract_dir.mkdir(exist_ok=True)
                    final_paths = extract_and_process_mtx_triplet(z, triplet_files, extract_dir, original_input)
                    return final_paths, None
                else:
                    # Extract single file
                    archive_stem = Path(file_path).stem
                    if archive_stem.endswith('.tar'):
                        archive_stem = archive_stem[:-4]
                    extract_dir = Path(file_path).parent / archive_stem
                    extract_dir.mkdir(exist_ok=True)
                    
                    # Extract only the selected file
                    z.extract(sel, path=extract_dir)
                    extracted_path = extract_dir / sel
                    final_path = extract_dir / Path(sel).name
                    if extracted_path != final_path:
                        shutil.move(extracted_path, final_path)
                    
                    # Clean up any subdirectories created during extraction
                    for item in extract_dir.iterdir():
                        if item.is_dir():
                            shutil.rmtree(item)
                    
                    return decompress_if_needed(str(final_path), sel, original_input)
            
            # Multiple files without --sel: return listing
            return file_paths, file_path

    # tar (multi-file archive)
    if tarfile.is_tarfile(file_path):
        with tarfile.open(file_path, mode='r:') as t:
            file_paths = [m.name for m in t.getmembers() if m.isfile()]
            
            # Single file in archive
            if len(file_paths) == 1:
                extract_dir = Path(file_path).parent / Path(file_paths[0]).name.replace('.', '_')
                extract_dir.mkdir(exist_ok=True)
                t.extract(file_paths[0], path=extract_dir, filter='data')
                extracted_path = extract_dir / file_paths[0]
                final_path = extract_dir / Path(file_paths[0]).name
                if extracted_path != final_path:
                    shutil.move(extracted_path, final_path)
                cleanup_empty_dirs(extract_dir)
                return decompress_if_needed(str(final_path), sel, original_input)
            
            # MTX triplet without --sel
            if not sel and is_mtx_triplet(file_paths):
                archive_stem = Path(file_path).stem
                if archive_stem.endswith('.tar'):
                    archive_stem = archive_stem[:-4]
                extract_dir = Path(file_path).parent / archive_stem
                extract_dir.mkdir(exist_ok=True)
                final_paths = extract_and_process_mtx_triplet(t, file_paths, extract_dir, original_input)
                
                # Delete the tar after MTX extraction ONLY if it's not the original input
                tar_to_delete = Path(file_path)
                original_input_path = Path(original_input).resolve() if original_input else None
                if (tar_to_delete.exists() and 
                    tar_to_delete.suffix in {'.tar', ''} and 
                    original_input_path and 
                    tar_to_delete.resolve() != original_input_path):
                    try:
                        tar_to_delete.unlink()
                    except:
                        pass
                
                return final_paths, None
            
            # Multiple files with --sel
            if sel:
                if sel not in file_paths:
                    ErrorJSON(f"Your selected file '{sel}' is not in the list of files of the archive: {file_paths}")
                
                # Check if selected file is part of an MTX triplet
                triplet_files = find_mtx_triplet_for_file(sel, file_paths)
                
                if triplet_files:
                    # Extract the entire triplet
                    archive_stem = Path(file_path).stem
                    if archive_stem.endswith('.tar'):
                        archive_stem = archive_stem[:-4]
                    extract_dir = Path(file_path).parent / archive_stem
                    extract_dir.mkdir(exist_ok=True)
                    final_paths = extract_and_process_mtx_triplet(t, triplet_files, extract_dir, original_input)
                    
                    # Delete the tar after extraction ONLY if it's not the original input
                    tar_to_delete = Path(file_path)
                    original_input_path = Path(original_input).resolve() if original_input else None
                    if (tar_to_delete.exists() and 
                        tar_to_delete.suffix in {'.tar', ''} and 
                        original_input_path and 
                        tar_to_delete.resolve() != original_input_path):
                        try:
                            tar_to_delete.unlink()
                        except:
                            pass
                    
                    return final_paths, None
                else:
                    # Extract single file
                    archive_stem = Path(file_path).stem
                    if archive_stem.endswith('.tar'):
                        archive_stem = archive_stem[:-4]
                    extract_dir = Path(file_path).parent / archive_stem
                    extract_dir.mkdir(exist_ok=True)
                    
                    # Extract only the selected file
                    t.extract(sel, path=extract_dir, filter='data')
                    extracted_path = extract_dir / sel
                    final_path = extract_dir / Path(sel).name
                    if extracted_path != final_path:
                        shutil.move(extracted_path, final_path)
                    
                    # Clean up any subdirectories created during extraction
                    for item in extract_dir.iterdir():
                        if item.is_dir():
                            shutil.rmtree(item)
                    
                    # Delete the tar after extraction ONLY if it's not the original input
                    tar_to_delete = Path(file_path)
                    original_input_path = Path(original_input).resolve() if original_input else None
                    if (tar_to_delete.exists() and 
                        tar_to_delete.suffix in {'.tar', ''} and 
                        original_input_path and 
                        tar_to_delete.resolve() != original_input_path):
                        try:
                            tar_to_delete.unlink()
                        except:
                            pass
                    
                    return decompress_if_needed(str(final_path), sel, original_input)
            
            # Multiple files without --sel: cache uncompressed tar if compressed
            original_path = Path(file_path)
            if original_path.suffix in {'.gz', '.bz2', '.xz'}:
                # This is a compressed tar - cache the uncompressed version
                cached_tar = original_path.with_suffix('')
                if not cached_tar.exists() or cached_tar == original_path:
                    # Already uncompressed or need different name
                    pass
                # Return listing with original compressed path
                return file_paths, str(original_path)
            else:
                # Already uncompressed tar
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
                    MtxHandler.preparse(args, [file_path])
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
        if '_index' in f[path]:
            # AnnData >= 0.7 style
            # The index is typically stored in obs/_index or var/_index as a dataset
            idx_data = f[path]['_index'][:]
            if isinstance(idx_data[0], bytes):
                return [x.decode('utf-8') for x in idx_data[:10]]
            else:
                return idx_data[:10].tolist()
        else:
            # Older style or fallback
            return [f"Item_{i+1}" for i in range(10)]
    
    @staticmethod
    def preparse(args, file_path):
        result = {
            "detected_format": FileType.H5AD,
            "file_path": file_path,
            "list_groups": []
        }
        
        with h5py.File(file_path, 'r') as f:
            # Get dimensions
            nGenes, nCells = H5ADHandler.get_size(f, 'X')
            
            # Get gene and cell names
            genes = H5ADHandler.extract_index(f, 'var')
            cells = H5ADHandler.extract_index(f, 'obs')
            
            # Extract 10x10 matrix
            x_group = f['X']
            if 'data' in x_group and 'indices' in x_group and 'indptr' in x_group:
                # Sparse CSR format
                indptr = x_group['indptr'][:min(10, nCells) + 1]
                data = x_group['data'][:indptr[-1]]
                indices = x_group['indices'][:indptr[-1]]
                matrix = [[0 for _ in range(min(10, nGenes))] for _ in range(min(10, nCells))]
                for row in range(min(10, nCells)):
                    start, end = indptr[row], indptr[row+1]
                    for idx_pos in range(start, end):
                        col = indices[idx_pos]
                        if col < min(10, nGenes):
                            matrix[row][col] = float(data[idx_pos])
            else:
                # Dense format
                matrix = x_group[:min(10, nCells), :min(10, nGenes)].tolist()
            
            entry = {
                "group": "X",
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
    def preparse(args, file_path):
        result = {
            "detected_format": FileType.LOOM,
            "file_path": file_path,
            "list_groups": []
        }
        
        with h5py.File(file_path, 'r') as f:
            # Get dimensions
            matrix_shape = f['matrix'].shape
            nGenes, nCells = matrix_shape
            
            # Get gene and cell names
            genes = [g.decode('utf-8') if isinstance(g, bytes) else str(g) 
                    for g in f['row_attrs']['Gene'][:min(10, nGenes)]]
            cells = [c.decode('utf-8') if isinstance(c, bytes) else str(c)
                    for c in f['col_attrs']['CellID'][:min(10, nCells)]]
            
            # Extract 10x10 matrix
            matrix = f['matrix'][:min(10, nGenes), :min(10, nCells)].tolist()
            
            entry = {
                "group": "matrix",
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
            "file_path": file_path,
            "list_groups": [],
            "warnings": []
        }

        # Try to import rpy2
        try:
            import rpy2.robjects as ro
            from rpy2.robjects.packages import importr
        except ImportError:
            ErrorJSON("rpy2 is not installed. Cannot read RDS files.")
        
        # Initialize R
        r = ro.r
        
        # Import base package for reading RDS files
        base = importr('base')

        # Load R object
        obj = base.readRDS(file_path)
        obj_class = r['class'](obj)[0]
        
        # Check its type and apply conversion
        if obj_class == "Seurat":
            Seurat = importr('Seurat')
            
            # Check available assays
            seurat_assays = [str(a) for a in r["Assays"](obj)]
            
            # Iterate through each assay
            for assay in seurat_assays:
                try:
                    layers_r = r(f'''
                    function(obj) {{
                      assay <- "{assay}"
                      Layers(obj[[assay]])
                    }}
                    ''')(obj)
                    layers = [str(x) for x in layers_r]
                    if "counts" in layers:
                        # 10x10 matrix        
                        matrix_r = r(f'function(obj) as(GetAssayData(obj, layer="counts", assay="{assay}")[1:10, 1:10], "matrix")')(obj)
                        matrix = np.array(matrix_r).tolist()
                        
                        # Global dims
                        nCells = int(list(r['ncol'](obj))[0])
                        nGenes = int(list(r['nrow'](obj))[0])
                        
                        # Gene and cell names
                        genes = list(r['rownames'](obj)[1:10])
                        cells = list(r['colnames'](obj)[1:10])
                                    
                        entry = {
                            "group": assay,
                            "nber_cols": nCells,
                            "nber_rows": nGenes,
                            "is_count": int((np.array(matrix) % 1 == 0).all()),
                            "genes": genes,
                            "cells": cells,
                            "matrix": matrix
                        }
                        result["list_groups"].append(entry)
                    else:
                        result["warnings"].append(f"WARNING: Skipping assay '{assay}', because it's missing the layer 'counts'. Available layers for assay '{assay}': {str(layers)}")
                except Exception as e:
                    result["warnings"].append(f"WARNING: Skipping assay '{assay}' due to error: {e}")
        elif obj_class == "data.frame":
            from rpy2.robjects import pandas2ri
            pandas2ri.activate()
            df = pandas2ri.rpy2py(obj)
            n_rows = len(df)
            entry = {
                "group": "data_frame",
                "nber_cols": len(df.columns),
                "nber_rows": n_rows,
                "is_count": 0,
                "genes": df.index[:10].tolist() if hasattr(df, 'index') else [f"Row_{i+1}" for i in range(10)],
                "cells": df.columns[:10].tolist(),
                "matrix": df.iloc[:10, :10].values.tolist()
            }
            result["list_groups"].append(entry)
        else:
            result["warnings"].append(f"WARNING: RDS object type not handled [{obj_class}]. Only handled types are [Seurat, data.frame]")
        
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
        
        # Cell names - try to find barcodes file (TSV or CSV)
        cells = [f"Cell_{i+1}" for i in range(10)]
        # Look for barcodes.tsv or barcodes.csv
        barcode_path = next((p for p in file_paths if Path(p).name in ["barcodes.tsv", "barcodes.csv"]), None)
        
        if barcode_path:
            # Standard TSV format (one barcode per line)
            with open(barcode_path, "r") as f:
                lines = [line.strip() for _, line in zip(range(11), f)]  # Read 11 lines
                
                # Check if first line is a header (contains commas or many tabs suggesting metadata columns)
                if lines and (',' in lines[0] or lines[0].count('\t') > 2):
                    # Skip header row, extract first column from remaining lines
                    cells = [line.split('\t')[0].split(',')[0].strip('"') for line in lines[1:] if line]
                else:
                    # No header, just barcode per line
                    cells = [line.split('\t')[0] for line in lines if line]
                
                cells = cells[:10]  # Keep only first 10
        else:
            # Try CSV format with metadata (cell_metadata*.csv)
            barcode_path = next((p for p in file_paths 
                                if 'cell' in Path(p).name.lower() and 
                                   'metadata' in Path(p).name.lower() and 
                                   Path(p).suffix == '.csv'), None)
            if barcode_path:
                # CSV with metadata - extract first column (row names) or barcode column
                import csv
                with open(barcode_path, "r") as f:
                    reader = csv.DictReader(f)
                    cells = []
                    for i, row in enumerate(reader):
                        if i >= 10:
                            break
                        # Try barcode column first, then first column (row names)
                        if 'barcode' in row:
                            cells.append(row['barcode'])
                        else:
                            # First column is typically the row index/names
                            cells.append(list(row.values())[0])

        # Gene names - try TSV or CSV format
        genes = [f"Gene_{i+1}" for i in range(10)]
        # Look for features.tsv, features.csv, genes.tsv, or genes.csv
        feature_path = next((p for p in file_paths if Path(p).name in ["features.tsv", "features.csv"]), None)
        if not feature_path:
            feature_path = next((p for p in file_paths if Path(p).name in ["genes.tsv", "genes.csv"]), None)
        if not feature_path:
            # Try CSV format (gene_names*.csv)
            feature_path = next((p for p in file_paths 
                                if 'gene' in Path(p).name.lower() and 
                                   Path(p).suffix == '.csv'), None)
        
        if feature_path:
            if feature_path.endswith('.tsv'):
                # TSV format
                with open(feature_path, "r") as f:
                    genes = [line.strip().split('\t')[0] for _, line in zip(range(10), f)]
            elif feature_path.endswith('.csv'):
                # CSV format - one gene per line, no header
                with open(feature_path, "r") as f:
                    genes = [line.strip() for _, line in zip(range(10), f)]
        
        # MTX file
        matrix_path = next((p for p in file_paths if Path(p).name not in {"barcodes.tsv", "barcodes.csv", "features.tsv", "features.csv", "genes.tsv", "genes.csv"}), None)
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
    parser.add_argument('--header', choices=['true', 'false'], default='true', required=False)
    parser.add_argument('--col', choices=['none', 'first', 'last'], default='first', required=False)
    parser.add_argument('--sel', default=None)
    parser.add_argument('--delim', default='\t')
    #parser.add_argument('--metadata', action='store_true')

    args = parser.parse_args()

    if not os.path.isfile(args.f):
        ErrorJSON(f"Input file not found: {args.f}")
    if args.o and not os.path.isdir(args.o):
        os.makedirs(args.o, exist_ok=True)

    preparse(args)

if __name__ == '__main__':
    main()
    