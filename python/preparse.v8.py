import argparse # Needed to parse arguments
import sys # Needed for ErrorJSON
import os # Needed for pathjoin and path/create dirs
import h5py # Needed for parsing hdf5 files
import json # For writing output files
import math
import re
import numpy as np # For diverse array calculations
import tarfile # To check if file is a TAR archive
from pathlib import Path # Needed for file extension manipulation
import shutil # For moving files and removing directories

_OUTPUT_DIR = None

def truncated_h5ad_error_message(exc, file_path=None):
    text = str(exc)
    match = re.search(r"eof = (\d+).*stored_eof = (\d+)", text, re.DOTALL)
    size_bit = ""
    if match:
        size_bit = (
            f" on-disk size is {match.group(1)} bytes, "
            f"HDF5 header expects {match.group(2)} bytes."
        )
    name = f" {file_path}" if file_path else ""
    return (
        f"The H5AD file{name} is incomplete.{size_bit} "
        "The download was interrupted or is still running. "
        "Wait until the download finishes, then run preparsing again."
    )

def corrupted_h5ad_error_message(file_path=None, detail=None):
    name = f" {file_path}" if file_path else ""
    extra = f" ({detail})" if detail else ""
    return (
        f"The H5AD file{name} is corrupted{extra}. "
        "HDF5 gzip/B-tree data cannot be read. This happens when a download was "
        "interrupted or two downloads wrote the same file. "
        "Delete the upload and download the file again."
    )

def hdf5_io_error_message(exc, file_path=None):
    text = str(exc).lower()
    if "truncated file" in text:
        return truncated_h5ad_error_message(exc, file_path)
    if "inflate() failed" in text or "wrong b-tree signature" in text:
        return corrupted_h5ad_error_message(file_path, detail=str(exc).strip())
    return None

def scrub_json_value(value):
    """Convert values to strict JSON types. Python allows NaN/Infinity in dumps; Ruby does not."""
    if isinstance(value, dict):
        return {key: scrub_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return scrub_json_value(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    return value

def dumps_output_json(result):
    return json.dumps(scrub_json_value(result), ensure_ascii=False, allow_nan=False)

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
    H5_10x = 'H5_10x'
    H5AD = 'H5AD'
    LOOM = 'LOOM'
    RDS = 'RDS'
    MEX = 'MEX'
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

def is_mex_triplet(file_list):
    """
    Check if file list contains exactly ONE MEX triplet.
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

def find_mex_triplet_for_file(selected_file, all_files):
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

def extract_and_process_mex_triplet(archive_obj, file_paths, extract_dir, original_input=None):
    """Extract and process MEX triplet files."""
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
        """Determine canonical name for MEX triplet file."""
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
            
            # MEX triplet without --sel
            if not sel and is_mex_triplet(file_paths):
                archive_stem = Path(file_path).stem
                if archive_stem.endswith('.tar'):
                    archive_stem = archive_stem[:-4]
                extract_dir = Path(file_path).parent / archive_stem
                extract_dir.mkdir(exist_ok=True)
                final_paths = extract_and_process_mex_triplet(z, file_paths, extract_dir, original_input)
                return final_paths, None
            
            # Multiple files with --sel
            if sel:
                if sel not in file_paths:
                    ErrorJSON(f"Your selected file '{sel}' is not in the list of files of the archive: {file_paths}")
                
                # Check if selected file is part of an MEX triplet
                triplet_files = find_mex_triplet_for_file(sel, file_paths)
                
                if triplet_files:
                    # Extract the entire triplet
                    archive_stem = Path(file_path).stem
                    if archive_stem.endswith('.tar'):
                        archive_stem = archive_stem[:-4]
                    extract_dir = Path(file_path).parent / archive_stem
                    extract_dir.mkdir(exist_ok=True)
                    final_paths = extract_and_process_mex_triplet(z, triplet_files, extract_dir, original_input)
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
            
            # MEX triplet without --sel
            if not sel and is_mex_triplet(file_paths):
                archive_stem = Path(file_path).stem
                if archive_stem.endswith('.tar'):
                    archive_stem = archive_stem[:-4]
                extract_dir = Path(file_path).parent / archive_stem
                extract_dir.mkdir(exist_ok=True)
                final_paths = extract_and_process_mex_triplet(t, file_paths, extract_dir, original_input)
                
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
                
                # Check if selected file is part of an MEX triplet
                triplet_files = find_mex_triplet_for_file(sel, file_paths)
                
                if triplet_files:
                    # Extract the entire triplet
                    archive_stem = Path(file_path).stem
                    if archive_stem.endswith('.tar'):
                        archive_stem = archive_stem[:-4]
                    extract_dir = Path(file_path).parent / archive_stem
                    extract_dir.mkdir(exist_ok=True)
                    final_paths = extract_and_process_mex_triplet(t, triplet_files, extract_dir, original_input)
                    
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

    # File has a compression-like extension but magic bytes didn't match any known
    # compressed format. Strip the extension so downstream tools (e.g. pandas)
    # don't try to decompress a plain file.
    p = Path(file_path)
    if p.suffix in {'.gz', '.bz2', '.xz'}:
        new_path = p.with_suffix('')
        if str(new_path) == str(p):  # edge-case: no stem change
            new_path = p.with_name(p.stem + '.out')
        shutil.copy2(str(p), str(new_path))
        return decompress_if_needed(str(new_path), sel, original_input)

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
                    return FileType.H5_10x
            
            # Loom format detection
            if {'matrix', 'row_attrs', 'col_attrs'}.issubset(f.keys()):
                return FileType.LOOM

            # AnnData H5AD detection
            if {'X', 'obs', 'var'}.issubset(f.keys()):
                return FileType.H5AD
            
            # We don't know what it is, but it's a HDF5 file
            return FileType.UNKNOWN
    except OSError as e:
        if "truncated file" in str(e).lower():
            ErrorJSON(truncated_h5ad_error_message(e, file_path))
        return FileType.UNKNOWN
    except (IOError, Exception):
        return FileType.UNKNOWN

def count_nonempty_lines(filepath):
    import subprocess
    result = subprocess.run(['wc', '-l', filepath], capture_output=True, text=True)
    return int(result.stdout.strip().split()[0])

def is_rds_file(file_path):
    # saveRDS() compresses with gzip (default), bzip2 or xz; readRDS() reads all three directly
    try:
        with open(file_path, 'rb') as f:
            magic = f.read(6)
        if magic.startswith(b'\x1f\x8b'):
            import gzip
            opener = gzip.open
        elif magic.startswith(b'BZh'):
            import bz2
            opener = bz2.open
        elif magic.startswith(b'\xfd7zXZ\x00'):
            import lzma
            opener = lzma.open
        else:
            opener = open
        with opener(file_path, 'rb') as f:
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
        elif len(list_files) == 3 and archive_path == None: # MEX (10x)
            MexHandler.preparse(args, list_files)
        elif len(list_files) > 1: # Archive with multiple files
            ArchiveHandler.write_listing_json(args, archive_path, list_files)
        else: # Unique file, decompressed if needed
            file_path = list_files[0]
            # An RDS extracted from a zip/tar archive is not HDF5 and would otherwise be read as text
            if is_rds_file(file_path):
                RdsHandler.preparse(args, file_path)
                return
            # Detect file type from magic bytes or hdf5 reader
            file_type = check_file_type(file_path)
            # Call appropriate handler
            if file_type == FileType.H5_10x:
                H510xHandler.preparse(args, file_path)
            elif file_type == FileType.H5AD:
                H5ADHandler.preparse(args, file_path)
            elif file_type == FileType.LOOM:
                LoomHandler.preparse(args, file_path)
            elif file_type == FileType.RAW_TEXT:
                # Check if MEX format
                first_line = None
                try:
                    with open(file_path, 'r') as f:
                        first_line = f.readline().strip()
                except (Exception) as e:
                    ErrorJSON(str(e).strip() + '. Format is not handled?')
                if first_line.startswith("%%MatrixMarket matrix"):
                    MexHandler.preparse(args, [file_path])
                else:
                    TextHandler.preparse(args, file_path)
            else: # Not detected
                ErrorJSON(f"File format not detected.")

class H510xHandler:
    @staticmethod
    def preparse(args, file_path):
        # build the JSON-able structure
        result = {
            "detected_format": FileType.H5_10x,
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

                    # Get genes: feature IDs (what parse maps and stores as Original_Gene), else symbols
                    def _dec(v): return v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)
                    gene_ds = next((p for p in ("genes", "features/id", "gene_names", "features/name") if p in grp), None)
                    genes = [_dec(g) for g in grp[gene_ds][:min(10, nGenes)]] if gene_ds else [f"Gene_{i+1}" for i in range(min(10, nGenes))]

                    # Extract 10x10 matrix, genes x cells like the other formats (10x is CSC: one indptr entry per cell)
                    indptr = grp["indptr"][:min(10, nCells) + 1] # 10 first cells + beginning of 11th
                    data = grp["data"][:indptr[-1]] # All values for 10 first cells
                    indices = grp["indices"][:indptr[-1]] # Gene indices for 10 first cells
                    matrix = [[0.0 for _ in range(min(10, nCells))] for _ in range(min(10, nGenes))]
                    for cell in range(min(10, nCells)):
                       start, end = indptr[cell], indptr[cell+1]
                       for idx_pos in range(start, end):
                           gene = indices[idx_pos]
                           if gene < min(10, nGenes):
                               matrix[gene][cell] = float(data[idx_pos]) # keep decimals so is_count is meaningful
                  
                    # Valid group
                    entry = {
                        "group": f'/{group_name}',
                        "nber_cols": nCells,
                        "nber_rows": nGenes,
                        "is_count": int((np.array(matrix) % 1 == 0).all()),
                        "genes": genes,
                        "cells": [_dec(b) for b in grp["barcodes"][:min(10,nCells)]],
                        "matrix": matrix
                    }
                    result["list_groups"].append(entry)

        # Serialize to JSON
        json_str = dumps_output_json(result)
        
        # Either write to file or print
        if args.o:
            with open(os.path.join(args.o, "output.json"), "w", encoding="utf-8") as out:
                out.write(json_str)
        else:
            print(json_str)
        
class H5ADHandler:
    @staticmethod
    def get_size(f, path):
        shape = f[path].attrs.get('shape', None)
        if shape is None:
            shape = f[path].attrs.get('h5sparse_shape', None)
        if shape is None:
            shape = f[path].shape
        return tuple(int(x) for x in shape)

    @staticmethod
    def get_table_size(grp, default=0):
        """
        Get the number of rows (cells for obs, genes for var) in an obs/var table.
        Handles three storage layouts:
          - Compound dataset (older AnnData)
          - Group with column datasets and optional _index (current AnnData)
          - Group containing categorical columns stored as sub-groups with {categories, codes}
            (newer AnnData, e.g. CELLxGENE Visium H5ADs)
        """
        # Compound dataset
        if isinstance(grp, h5py.Dataset):
            return int(grp.shape[0]) if len(grp.shape) >= 1 else default

        if not isinstance(grp, h5py.Group):
            return default

        # Prefer the _index dataset if declared
        idx_attr = grp.attrs.get('_index', None)
        if isinstance(idx_attr, bytes):
            idx_attr = idx_attr.decode()
        if idx_attr and idx_attr in grp and isinstance(grp[idx_attr], h5py.Dataset):
            return int(grp[idx_attr].shape[0])
        if '_index' in grp and isinstance(grp['_index'], h5py.Dataset):
            return int(grp['_index'].shape[0])

        # Otherwise scan columns until we find one with a recoverable length
        for key in grp.keys():
            item = grp[key]
            if isinstance(item, h5py.Dataset) and len(item.shape) >= 1:
                return int(item.shape[0])
            # New-style categorical: Group with 'codes' dataset (1D, length == n_rows)
            if isinstance(item, h5py.Group) and 'codes' in item and isinstance(item['codes'], h5py.Dataset):
                return int(item['codes'].shape[0])

        return default

    @staticmethod
    def detect_spatial(f):
        """
        Detect whether the H5AD is a spatial transcriptomics dataset (Visium / Slide-seqV2)
        following the scFAIR v7.1.0 spatial schema. Returns (is_spatial, library_id_or_None).
        """
        if 'uns' not in f or 'spatial' not in f['uns']:
            return False, None
        spatial_grp = f['uns/spatial']
        library_id = None
        for k in spatial_grp.keys():
            if k == 'is_single':
                continue
            if isinstance(spatial_grp[k], h5py.Group):
                library_id = k
                break
        return True, library_id

    @staticmethod
    def verify_sparse_readable(f, path, file_path):
        if path not in f:
            return
        node = f[path]
        if not (isinstance(node, h5py.Group) and "indptr" in node):
            return
        ds = node["indptr"]
        n = int(ds.shape[0]) if ds.shape else 0
        if n <= 0:
            return
        try:
            indptr = ds[:] # n_cells+1 (or n_genes+1) integers: small next to data/indices
        except OSError as e:
            msg = hdf5_io_error_message(e, file_path)
            ErrorJSON(msg if msg else f"Cannot read {path}/indptr: {e}")
        # A valid indptr starts at 0, never decreases, and ends at the number of stored values
        nnz = int(node["data"].shape[0]) if "data" in node and node["data"].shape else None
        if int(indptr[0]) != 0 or (nnz is not None and int(indptr[-1]) != nnz) or np.any(np.diff(indptr) < 0):
            ErrorJSON(f"The sparse matrix at {path} is corrupted: invalid indptr (first={int(indptr[0])}, last={int(indptr[-1])}, "
                      f"stored values={nnz}, non-decreasing={bool(np.all(np.diff(indptr) >= 0))}).")

    @staticmethod
    def extract_index(f, path, orientation="Item", n=10):
        if path not in f:
            return [f"{orientation}_{i+1}" for i in range(n)]
        grp = f[path]
        try:
            # _index is an ATTRIBUTE naming which column holds the index
            if '_index' in grp.attrs:
                idx_col = grp.attrs['_index']
                if isinstance(idx_col, bytes):
                    idx_col = idx_col.decode()
                item = grp[idx_col]
            elif '_index' in grp:
                # fallback: stored as a dataset/group directly
                item = grp['_index']
            else:
                return [f"{orientation}_{i+1}" for i in range(n)]

            if isinstance(item, h5py.Dataset):
                idx_data = item[:]
            elif isinstance(item, h5py.Group) and 'values' in item and 'mask' in item:
                # AnnData nullable-string-array (and related): {values, mask}
                values = item['values'][:]
                mask = np.asarray(item['mask'][:], dtype=bool)
                idx_data = []
                for v, m in zip(values.flat, mask.flat):
                    if m:
                        idx_data.append('nan')
                    else:
                        idx_data.append(v.decode('utf-8') if isinstance(v, (bytes, np.bytes_)) else str(v))
            elif isinstance(item, h5py.Group) and 'categories' in item and 'codes' in item:
                cats = [v.decode('utf-8') if isinstance(v, (bytes, np.bytes_)) else str(v) for v in item['categories'][:]]
                codes = item['codes'][:]
                idx_data = [cats[c] if 0 <= c < len(cats) else 'nan' for c in codes] # -1 = missing; out-of-range = corrupted
            else:
                return [f"{orientation}_{i+1}" for i in range(n)]

            if len(idx_data) > 0 and isinstance(idx_data[0], (bytes, np.bytes_)):
                return [x.decode('utf-8') for x in idx_data[:n]]
            return [str(x) for x in idx_data[:n]]
        except Exception:
            return [f"{orientation}_{i+1}" for i in range(n)]

    @staticmethod
    def sample_value_stats(f, path, max_values=200000):
        """Tell counts from normalized data using up to max_values stored values: the non-zeros of a sparse matrix, or
        the first rows of a dense one. The 10x10 preview corner is often all zeros, which looks like counts.
        Returns (is_count, dtype): 1 = non-negative integers, 0 = not, -1 = unknown (no non-zero value read)."""
        node = f[path]
        if isinstance(node, h5py.Group):
            if "data" not in node:
                return -1, None
            ds = node["data"]
            n = int(ds.shape[0]) if ds.shape else 0
            vals = ds[:min(n, max_values)] if n > 0 else np.empty(0, dtype=ds.dtype)
        else:
            ds = node
            if len(ds.shape) != 2 or ds.shape[0] == 0 or ds.shape[1] == 0:
                return -1, str(ds.dtype)
            rows = max(1, min(int(ds.shape[0]), max_values // int(ds.shape[1])))
            vals = ds[:rows]
        dtype = str(ds.dtype)
        vals = np.asarray(vals).ravel()
        vals = vals[vals != 0]  # zeros say nothing about integer-ness
        if vals.size == 0:
            return -1, dtype
        if np.issubdtype(vals.dtype, np.integer) or np.issubdtype(vals.dtype, np.bool_):
            return int(vals.min() >= 0), dtype
        if not np.issubdtype(vals.dtype, np.floating):
            return -1, dtype
        if not np.all(np.isfinite(vals)):
            return 0, dtype
        return int(bool(vals.min() >= 0 and np.all(np.floor(vals) == vals))), dtype

    @staticmethod
    def extract_matrix(f, path, nCells, nGenes, n=10):
        """Extract the top-left n x n submatrix from a sparse or dense dataset, as genes x cells (like the other formats)."""
        nr, nc = min(n, nCells), min(n, nGenes)
        grp = f[path]
        encoding = grp.attrs.get('encoding-type', None)
        if encoding is None:
            encoding = grp.attrs.get('h5sparse_format', '')
        encoding = encoding.decode() if isinstance(encoding, bytes) else str(encoding)
        matrix = [[0.0] * nc for _ in range(nr)]

        if isinstance(grp, h5py.Dataset):
            # Dense
            matrix = grp[:nr, :nc].tolist()
        elif 'data' in grp and 'indices' in grp and 'indptr' in grp:
            if 'csc' in encoding:
                # CSC: columns are compressed
                indptr = grp['indptr'][:nc + 1]
                data = grp['data'][:int(indptr[-1])]
                indices = grp['indices'][:int(indptr[-1])]
                for col in range(nc):
                    start, end = int(indptr[col]), int(indptr[col + 1])
                    for k in range(start, end):
                        row = int(indices[k])
                        if row < nr:
                            matrix[row][col] = float(data[k])
            else:
                # CSR (default)
                indptr = grp['indptr'][:nr + 1]
                data = grp['data'][:int(indptr[-1])]
                indices = grp['indices'][:int(indptr[-1])]
                for row in range(nr):
                    start, end = int(indptr[row]), int(indptr[row + 1])
                    for k in range(start, end):
                        col = int(indices[k])
                        if col < nc:
                            matrix[row][col] = float(data[k])
        else:
            raise ValueError(f"Unrecognized matrix format at {path}")
        # H5AD stores cells x genes; the preview contract (Loom, MEX, text, RDS, legacy Java) is genes x cells
        return [[matrix[r][c] for r in range(nr)] for c in range(nc)]

    @staticmethod
    def preparse(args, file_path):
        result = {
            "detected_format": FileType.H5AD,
            "file_path": file_path,
            "list_groups": []
        }

        with h5py.File(file_path, 'r') as f:
            # X shape is (n_cells, n_genes)
            nCells, nGenes = H5ADHandler.get_size(f, '/X')

            genes = H5ADHandler.extract_index(f, '/var', orientation='Gene')
            cells = H5ADHandler.extract_index(f, '/obs', orientation='Cell')

            # Detect spatial transcriptomics datasets (Visium / Slide-seqV2). The presence
            # of /uns/spatial is the scFAIR v7.1.0 marker. is_spatial is always
            # emitted as 0 or 1 so downstream consumers don't need to test for
            # key existence (matches the is_count / is_count_table convention).
            is_spatial, spatial_library_id = H5ADHandler.detect_spatial(f)
            result["is_spatial"] = 1 if is_spatial else 0
            if is_spatial and spatial_library_id is not None:
                result["spatial_library_id"] = spatial_library_id

            # Collect all matrix paths to report
            matrix_paths = []

            if '/X' in f:
                matrix_paths.append(('/X', nCells, nGenes))

            if '/raw' in f:
                raw_grp = f['raw']
                # raw/X uses raw's own var (may differ in gene count)
                raw_nGenes = nGenes
                if 'var' in raw_grp:
                    # raw can have more genes than filtered X. The first child of raw/var
                    # may be a Group (categorical encoded as {categories, codes}), not a
                    # Dataset, so we use the robust get_table_size helper instead of
                    # blindly grabbing the first key's .shape.
                    raw_var_size = H5ADHandler.get_table_size(raw_grp['var'], default=nGenes)
                    raw_nGenes = raw_var_size
                if 'X' in raw_grp:
                    raw_nCells, raw_nGenes = H5ADHandler.get_size(f, '/raw/X')
                    matrix_paths.append(('/raw/X', raw_nCells, raw_nGenes))

            if '/layers' in f:
                for layer_name in f['/layers']:
                    layer_path = f'/layers/{layer_name}'
                    lCells, lGenes = H5ADHandler.get_size(f, layer_path)
                    matrix_paths.append((layer_path, lCells, lGenes))

            for path, nC, nG in matrix_paths:
                H5ADHandler.verify_sparse_readable(f, path, file_path)
                # For raw, genes may differ — re-extract from raw/var
                if path.startswith('/raw/'):
                    g = H5ADHandler.extract_index(f, 'raw/var', orientation='Gene')
                    c = cells
                else:
                    g, c = genes, cells

                try:
                    matrix = H5ADHandler.extract_matrix(f, path, nC, nG)
                    arr = np.array(matrix)
                    is_count = int((arr % 1 == 0).all())
                except OSError as e:
                    msg = hdf5_io_error_message(e, file_path)
                    if msg:
                        ErrorJSON(msg)
                    raise
                except Exception as e:
                    matrix = []
                    is_count = -1

                dtype = None
                try:
                    sampled, dtype = H5ADHandler.sample_value_stats(f, path)
                    if sampled != -1:
                        is_count = sampled
                except Exception:
                    pass # keep the preview-based guess

                entry = {
                    "group": path,
                    "nber_cols": nC,   # cells
                    "nber_rows": nG,   # genes
                    "is_count": is_count,
                    "dtype": dtype,
                    "genes": g,
                    "cells": c,
                    "matrix": matrix
                }
                result["list_groups"].append(entry)

        json_str = dumps_output_json(result)
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
            matrix_shape = f['/matrix'].shape
            nGenes, nCells = matrix_shape

            # Detect spatial Loom (round-trip from a scFAIR-spatial H5AD parse).
            # Our parser writes /attrs/spatial/... in this case. is_spatial is
            # always emitted as 0 or 1 (matches the H5AD preparse convention).
            result["is_spatial"] = 0
            if '/attrs/spatial' in f and isinstance(f['/attrs/spatial'], h5py.Group):
                result["is_spatial"] = 1
                spatial_grp = f['/attrs/spatial']
                for k in spatial_grp.keys():
                    if k == 'is_single':
                        continue
                    if isinstance(spatial_grp[k], h5py.Group):
                        result["spatial_library_id"] = k
                        break
            
            # Get gene and cell names: ASAP / loompy attributes first, then the same fallbacks parse uses
            # (looms written by anndata/scanpy have var_names/obs_names); generic names if none exist
            def first_names(candidates, n, prefix):
                for p in candidates:
                    if p in f and isinstance(f[p], h5py.Dataset) and f[p].ndim == 1 and f[p].shape[0] == n:
                        return [v.decode('utf-8', errors='ignore') if isinstance(v, (bytes, np.bytes_)) else str(v) for v in f[p][:min(10, n)]]
                return [f"{prefix}_{i+1}" for i in range(min(10, n))]
            genes = first_names(["/row_attrs/Gene", "/row_attrs/Accession", "/row_attrs/Name", "/row_attrs/Original_Gene",
                                 "/row_attrs/gene", "/row_attrs/gene_name", "/row_attrs/var_names", "/row_attrs/index", "/row_attrs/_index"], nGenes, "Gene")
            cells = first_names(["/col_attrs/CellID", "/col_attrs/cell_id", "/col_attrs/cell_ids", "/col_attrs/barcode", "/col_attrs/barcodes",
                                 "/col_attrs/Barcode", "/col_attrs/obs_names", "/col_attrs/index", "/col_attrs/_index"], nCells, "Cell")
            
            # Extract 10x10 matrix
            matrix = f['/matrix'][:min(10, nGenes), :min(10, nCells)].tolist()
            
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
        
        json_str = dumps_output_json(result)
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
            
            # Preview of one assay's counts. The assay name is passed as an R argument (never pasted into R source),
            # and dims/names come from that assay's counts matrix (nrow()/rownames() of the object = default assay).
            _assay_preview = r('''
                function(obj, assay) {
                    layers <- as.character(Layers(obj[[assay]]))
                    if (!("counts" %in% layers)) return(list(layers = layers))
                    m  <- GetAssayData(obj, assay = assay, layer = "counts")
                    nr <- min(10L, nrow(m))
                    nc <- min(10L, ncol(m))
                    list(layers  = layers,
                         n_genes = as.integer(nrow(m)),
                         n_cells = as.integer(ncol(m)),
                         genes   = as.character(head(rownames(m), nr)),
                         cells   = as.character(head(colnames(m), nc)),
                         matrix  = as.matrix(m[seq_len(nr), seq_len(nc), drop = FALSE]))
                }
            ''')

            # Iterate through each assay
            for assay in seurat_assays:
                try:
                    preview = _assay_preview(obj, assay)
                    layers = [str(x) for x in preview.rx2("layers")]
                    if "counts" in layers:
                        # Up to 10x10 matrix (genes x cells). R matrices are column-major: rebuild the 2-D shape explicitly
                        matrix_r = preview.rx2("matrix")
                        n_r, n_c = (int(x) for x in matrix_r.dim)
                        matrix = np.array(list(matrix_r), dtype=float).reshape((n_r, n_c), order="F").tolist()

                        # Global dims
                        nCells = int(preview.rx2("n_cells")[0])
                        nGenes = int(preview.rx2("n_genes")[0])

                        # Gene and cell names (same first rows/columns as the matrix preview)
                        genes = list(preview.rx2("genes"))
                        cells = list(preview.rx2("cells"))

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
            from rpy2.robjects.conversion import localconverter
            # rpy2 >= 3.6 removed pandas2ri.activate(): convert through an explicit local context (as parse does)
            _cv = ro.default_converter + pandas2ri.converter
            with localconverter(_cv):
                df = _cv.rpy2py(obj)
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
        
        json_str = dumps_output_json(result)

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
        json_str = dumps_output_json(result)
        if args.o:
            with open(os.path.join(args.o, "output.json"), "w", encoding="utf-8") as out:
                out.write(json_str)
        else:
            print(json_str)

class MexHandler:
    @staticmethod
    def preparse(args, file_paths):
        result = {
            "detected_format": FileType.MEX,
            "file_path": file_paths, # Could have been unzipped/unarchived
            "list_groups": []
        }
        
        # Cell names - try to find barcodes file (TSV or CSV)
        cells = [f"Cell_{i+1}" for i in range(10)]
        # Look for barcodes.tsv or barcodes.csv
        barcode_path = next((p for p in file_paths if Path(p).name in ["barcodes.tsv", "barcodes.csv"]), None)

        # Number of cells declared by the MTX dimension line (decides whether the barcodes file has a header)
        n_cells_mtx = None
        mtx_path = next((p for p in file_paths if Path(p).name.lower().endswith(".mtx")), None)
        if mtx_path:
            try:
                with open(mtx_path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if not line.startswith('%'):
                            parts = line.split()
                            if len(parts) == 3 and parts[1].isdigit():
                                n_cells_mtx = int(parts[1])
                            break
            except OSError:
                n_cells_mtx = None # fall back to the first-line heuristic below

        if barcode_path:
            # Standard TSV format (one barcode per line)
            with open(barcode_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]

                # Header detection, same rule as parse: the line count vs the matrix decides when it can,
                # otherwise a first line with commas or many tabs (metadata columns) is a header
                if n_cells_mtx is not None and len(lines) in (n_cells_mtx, n_cells_mtx + 1):
                    has_header = len(lines) == n_cells_mtx + 1
                else:
                    has_header = bool(lines) and (',' in lines[0] or lines[0].count('\t') > 2)
                # First field of each line (as parse reads it: tab-delimited for .tsv, comma-delimited for .csv)
                is_csv = barcode_path.endswith('.csv')
                cells = [(line.split(',')[0] if is_csv else line.split('\t')[0]).strip('"') for line in lines[(1 if has_header else 0):]]

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
        
        json_str = dumps_output_json(result)

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
            "group": Path(file_path).name,
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
        json_str = dumps_output_json(result)
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

        out_path = output
        if out_path is None and _OUTPUT_DIR:
            out_path = os.path.join(_OUTPUT_DIR, "output.json")

        if out_path:
            # write JSON to the given file
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        else:
            # or print it to stdout
            print(json.dumps(payload, ensure_ascii=False), file=sys.stdout)

        # then immediately exit with an error code
        sys.exit(1)

def main():
    global _OUTPUT_DIR
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

    _OUTPUT_DIR = args.o
    if not os.path.isfile(args.f):
        ErrorJSON(f"Input file not found: {args.f}")
    if args.o and not os.path.isdir(args.o):
        os.makedirs(args.o, exist_ok=True)

    try:
        preparse(args)
    except Exception as e:
        # Catch-all: unexpected errors still produce a displayed_error (the traceback goes to stderr for debugging)
        import traceback
        traceback.print_exc(file=sys.stderr)
        ErrorJSON(f"Unexpected error while reading the file ({type(e).__name__}: {e})")

if __name__ == '__main__':
    main()