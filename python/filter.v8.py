"""
Row / column filtering of a Loom file (Python port of ASAP.jar -T FilterCols / -T FilterRows).

Drop-in replacement for the two Java modes (java/src/filtering/FileFilter.java): same flags, same
output files (<o>/output.loom, <o>/output.json), same output.json keys and error messages. Callers
swap "java -jar ASAP.jar" for "python3 /srv/filter.v8.py" and keep the rest of the command line.

  python3 filter.v8.py -T FilterCols -loom in.loom -o out_dir -col_indexes_file discarded.json
  python3 filter.v8.py -T FilterCols -loom in.loom -o out_dir -col_names_file filtered_cells.json
  python3 filter.v8.py -T FilterRows -loom in.loom -o out_dir -m keep -row_names_file to_keep.json
  python3 filter.v8.py -T FilterRows -loom in.loom -o out_dir -m basic
  python3 filter.v8.py -T FilterRows -loom in.loom -o out_dir -m cpm <nbCountsPerCell> <nbCellsDetected>

FilterCols removes the empty cells (/col_attrs/_Depth == 0) plus the cells listed in the JSON
({"discarded_cols": [indexes]} or {"filtered_cells": [CellIDs]}), recomputes /row_attrs/_Sum.
FilterRows removes the empty genes (/row_attrs/_Sum == 0) plus, depending on -m, the genes not in
{"kept_genes": [indexes]} (keep) or detected (CPM > nbCountsPerCell) in fewer than nbCellsDetected
cells (cpm); it recomputes /col_attrs/_Depth and /col_attrs/_Detected_Genes.
/matrix, every /layers/* matrix and every /row_attrs, /col_attrs dataset is filtered on the
matching axis (kept rows/cols stay in their original order, _StableID values travel with them);
/attrs is copied unchanged, except the result tables of other runs (DE, enrichment: 2D
/attrs/_<step>_<run>_<method>), which describe the unfiltered data and stay in the loom of that run;
/row_graphs and /col_graphs edge lists (a, b, w) keep the edges between kept rows / cols, renumbered.
The output is a valid Loom file: every /layers matrix is n_genes x n_cells like /matrix, every
/row_attrs (/col_attrs) dataset keeps its orientation and is filtered on its gene (cell) axis.

Differences with the Java (all deliberate, see the report of the port):
  - output.json nber_rows/nber_cols follow the ASAP convention (CELL 1 x n_cells or k x n_cells,
    GENE n_genes x 1 or n_genes x k, layers n_genes x n_cells, GLOBAL as parse.v8.py) instead of
    the values stamped by LoomFile.copyMetadata (e.g. n_filtered x n_original).
  - datasets keep their HDF5 type (no float64 -> float32 / int64 -> int32 conversion), their shape,
    their HDF5 attributes; booleans, 1D/2D /attrs datasets and nested /attrs groups are copied
    instead of being dropped (or failing); unchunked /matrix and layers are supported.
  - errors always land in <o>/output.json when -o is given (JSON-escaped), not on stdout.
  - boolean /row_attrs, /col_attrs datasets (not allowed in Loom) are written as uint8 0/1.
  - graphs are filtered instead of being dropped; DE / enrichment tables of /attrs are not copied
    (the Java failed on them, the web stripped them before cell filtering); a layer that does not
    have the /matrix shape is skipped with a warning instead of failing the run.
"""
from __future__ import annotations
import json # For reading the index files and writing output.json
import os # Needed for paths / temporary output file
import re # For Java-like integer parsing
import sys # Needed for ErrorJSON
import time # For waiting on locked files
from pathlib import Path # For output paths
from typing import Any, Callable, Final, Optional

import h5py # Needed for reading/writing the Loom files
import numpy as np # For diverse array calculations

## Constants
MODES: Final[tuple] = ("FilterCols", "FilterRows")
LOOM_FILENAME: Final[str] = "output.loom" # Output LOOM file name (as ASAP.jar)
OUTPUT_JSON_NAME: Final[str] = "output.json" # Output JSON file name (as ASAP.jar)
LOOM_VERSION: Final[str] = "3.0.0" # ASAP.jar reports Parameters.loomVersion, never read from the input in these modes
DEFAULT_MATRIX_CHUNK: Final[int] = 1024 # Chunk side used when an input expression matrix is not chunked
DEFAULT_GZIP_LEVEL: Final[int] = 2 # ASAP.jar deflate level for expression matrices
TILE_BYTES: Final[int] = 64 * 1024**2 # Target size of one block read from the input (bounded RAM)
STRING_ITEM_BYTES: Final[int] = 64 # Estimated size of a variable-length string element (block sizing only)
OUTPUT_CHUNK_CACHE: Final[int] = 512 * 1024**2 # Output chunk cache: keeps partially written chunks between blocks
OPEN_MAX_WAIT: Final[int] = 3600 # Seconds to wait for a locked input file (ASAP.jar: 3600 x 1 s)
IS_INTEGER_TOL: Final[float] = 1e-5 # tools.Utils.isInteger tolerance

# Result tables of other runs in /attrs: de.v8.py / DESeq2 / enrichment --write-metadata paths
# /attrs/_<step tag>_<run num>_<method> (e.g. /attrs/_de_3_wilcoxon) and /attrs/de_<run id>_<k>.
RUN_RESULT_TABLE_RE: Final = re.compile(r"\A(?:_[A-Za-z]+_\d+_[A-Za-z0-9_.-]+|de_\d+_\d+)\Z")
GRAPH_GROUPS: Final[tuple] = (("row_graphs", "GENE"), ("col_graphs", "CELL"))

_OUTPUT_DIR: Optional[str] = None
_INT_RE = re.compile(r"[+-]?\d+")

## Error handling (same pattern as parse.v8.py; message strings are the ASAP.jar ones)
class ErrorJSON(Exception):
    def __init__(self, message: str, output_dir: str = None):
        super().__init__(message)
        payload = json.dumps({"displayed_error": message}, ensure_ascii=False, separators=(",", ":"))
        out_dir = output_dir or _OUTPUT_DIR
        if out_dir:
            with open(os.path.join(out_dir, OUTPUT_JSON_NAME), "w", encoding="utf-8") as f:
                f.write(payload)
        else:
            print(payload, file=sys.stdout)
        sys.exit(1)

## Help (texts of Parameters.printHelp)
HELP: Final[dict] = {
    "FilterCols": "FilterCols Mode\n\nOptions:\n"
                  "-o %s \t[Required] Output folder for generated Loom and JSON files.\n"
                  "-loom %s \t[Required] Loom file to read.\n"
                  "-col_names_file %s \t[Required one of both] JSON file containing Cells to filter out.\n"
                  "-col_indexes_file %s \t[Required one of both] JSON file containing Cells to filter out.",
    "FilterRows": "FilterRows Mode\n\nOptions:\n"
                  "-o %s \t\tOutput folder.\n"
                  "-loom %s \tFile to parse.\n"
                  "-m %s \t\tModel to use for filtering. It should be one of the following: [basic, cpm, keep]\n"
                  "-row_names_file %s \t[Required] JSON file containing Genes to Keep.",
}

def print_help(mode: str) -> None:
    print(HELP[mode])

## Command line (same flags and checks as ASAP.readMode + Parameters.loadFilterCells / loadFilterGenes)
def _java_path(value: str) -> str:
    return value.replace("\\", "/")

def _next_value(args: list[str], i: int, opt: str) -> str:
    if i >= len(args):
        ErrorJSON(f"'{opt}' should be followed by a value")
    return args[i]

def read_mode(argv: list[str]) -> tuple[str, list[str]]:
    args = [a for a in argv if a != "--debug"] # ASAP.jar strips --debug before reading the mode
    if "-T" not in args or len(args) < 2:
        ErrorJSON("Argument -T is mandatory:\n-T %s \t\tMode to run ASAP. Please select in [" + ", ".join(MODES) + "]")
    rest, mode, i = [], None, 0
    while i < len(args):
        if args[i] == "-T":
            i += 1
            mode = args[i] if i < len(args) else ""
            if mode not in MODES:
                ErrorJSON(f"This mode (-T) '{mode}' does not exist. Please select in [" + ", ".join(MODES) + "]")
        else:
            rest.append(args[i])
        i += 1
    return mode, rest

def set_output_dir(mode: str, args: list[str]) -> Optional[str]:
    """-o is read before the other options so that every later error is written to <o>/output.json."""
    global _OUTPUT_DIR
    out = None
    for i, a in enumerate(args):
        if a == "-o":
            out = _java_path(_next_value(args, i + 1, "-o"))
    if out is None:
        return None
    if os.path.exists(out) and not os.path.isdir(out):
        ErrorJSON(f"{out} is not a folder.")
    try:
        os.makedirs(out, exist_ok=True)
    except OSError as e:
        ErrorJSON(f"Cannot create the output folder {out}: {e}")
    _OUTPUT_DIR = out
    return out

def _java_int(value: str) -> Optional[int]:
    """Integer.parseInt: optional sign + digits, 32-bit range."""
    if value is None or not _INT_RE.fullmatch(value):
        return None
    v = int(value)
    return v if -2**31 <= v < 2**31 else None

def load_filter_cols(args: list[str]) -> dict:
    p = {"loom": None, "json": None, "is_index": False}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("-"):
            if arg == "-o":
                i += 1
            elif arg == "-loom":
                i += 1
                if i >= len(args):
                    ErrorJSON("The '-loom' option should be followed by Loom file path.")
                p["loom"] = _java_path(args[i])
                if not os.path.exists(p["loom"]):
                    ErrorJSON(f"No file at path {p['loom']}")
                if not os.path.isfile(p["loom"]):
                    ErrorJSON(f"{p['loom']} is not a file")
            elif arg in ("-col_indexes_file", "-col_names_file"):
                if arg == "-col_indexes_file":
                    p["is_index"] = True # sticky, as the Java switch fall-through
                i += 1
                p["json"] = _java_path(_next_value(args, i, arg))
                if os.path.isdir(p["json"]):
                    ErrorJSON("'-col_names_file' should be followed by a JSON file name containing cell names, not a folder name.")
            else:
                print(f"Unused argument: {arg}", file=sys.stderr)
        i += 1
    if _OUTPUT_DIR is None:
        print_help("FilterCols")
        ErrorJSON("Please specify an output folder using -o option")
    if p["loom"] is None:
        print_help("FilterCols")
        ErrorJSON("Please specify a loom file to filter using -loom option")
    if p["json"] is None:
        print_help("FilterCols")
        ErrorJSON("Please specify a JSON file containing Cells to filter out using -col_names_file option")
    return p

def load_filter_rows(args: list[str]) -> dict:
    p = {"loom": None, "json": None, "model": None, "nb_counts_per_cell": -1, "nb_cells_detected": -1}
    cpm_error = "The '-m cpm' model should be followed by two Integers: 'nbCountsPerCell nbCellsDetected'."
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("-"):
            if arg == "-o":
                i += 1
            elif arg == "-loom":
                i += 1
                p["loom"] = _java_path(_next_value(args, i, arg))
            elif arg == "-row_names_file":
                i += 1
                p["json"] = _java_path(_next_value(args, i, arg))
                if os.path.isdir(p["json"]):
                    ErrorJSON("'-row_names_file' should be followed by a JSON file name containing gene IDs to keep, not a folder name.")
            elif arg == "-m":
                i += 1
                model = _next_value(args, i, arg)
                if model in ("basic", "keep"):
                    p["model"] = model
                elif model == "cpm":
                    p["model"] = model
                    for key, which in (("nb_counts_per_cell", "first"), ("nb_cells_detected", "second")):
                        i += 1
                        if i >= len(args):
                            ErrorJSON(f"{cpm_error[:-1]}. The {which} one is missing!")
                        value = _java_int(args[i])
                        if value is None:
                            ErrorJSON(f"{cpm_error[:-1]}. The {which} one you entered is not an Integer: {args[i]}")
                        p[key] = value
                    if p["nb_counts_per_cell"] < 0 or p["nb_cells_detected"] < 0:
                        ErrorJSON(f"The '-m cpm' model should be followed by two positive Integers. You entered '{p['nb_counts_per_cell']} {p['nb_cells_detected']}'.")
                else:
                    ErrorJSON(f"The entered model, {model}, does not exist!\nIt should be one of the following: [basic, cpm, keep]")
            else:
                print(f"Unused argument: {arg}", file=sys.stderr)
        i += 1
    if p["model"] is None or _OUTPUT_DIR is None or p["loom"] is None:
        print_help("FilterRows")
        error = "Filtering cannot be run because parameters are missing:\n"
        if p["model"] is None: error += "No model is specified, please choose a model by using the '-m' option.\n"
        if p["loom"] is None: error += "No file is specified, please choose a data file by using the '-f' option.\n"
        if _OUTPUT_DIR is None: error += "No output folder is specified, please choose an output file by using the '-o' option.\n"
        ErrorJSON(error)
    if p["model"] == "keep" and p["json"] is None:
        print_help("FilterRows")
        ErrorJSON("Please specify a JSON file containing Cells to filter out after filtering using -row_names_file option")
    return p

## JSON index files (json.FilterCellsJSON)
def read_filter_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        ErrorJSON(f"The JSON cell list was not found at the given path: {path}")
    except Exception as e:
        ErrorJSON(f"Problem detected when reading the JSON cell list: {type(e).__name__}: {e}")
    if not isinstance(data, dict):
        ErrorJSON(f"Problem detected when reading the JSON cell list: a JSON object was expected, found {type(data).__name__}")
    return data

def _as_list(value: Any) -> list:
    # R's jsonlite toJSON(auto_unbox=TRUE) writes a single kept index as a scalar instead of [x]
    return value if isinstance(value, list) else [value]

def json_int_list(value: Any, field: str) -> np.ndarray:
    """Integers as Gson reads them for int[]/long[] (numbers with no fraction, or numeric strings)."""
    out = []
    for v in _as_list(value):
        if isinstance(v, bool):
            iv = None
        elif isinstance(v, int):
            iv = v
        elif isinstance(v, float) and v.is_integer():
            iv = int(v)
        elif isinstance(v, str) and _INT_RE.fullmatch(v.strip()):
            iv = int(v.strip())
        else:
            iv = None
        if iv is None:
            ErrorJSON(f"Problem detected when reading the JSON cell list: '{field}' should contain integers, found {v!r}")
        out.append(iv)
    return np.asarray(out, dtype=np.int64)

## HDF5 helpers
def _is_lock_error(e: Exception) -> bool:
    """HDF5 lock held by another process (EAGAIN), as opposed to a file that cannot be read at all."""
    text = str(e).lower()
    return "errno = 11" in text or "temporarily unavailable" in text

def open_input_loom(path: str) -> h5py.File:
    """LoomFile(\"r\", path): waits for a locked file (other errors fail at once), then checks the Loom layout."""
    if not os.path.exists(path):
        ErrorJSON(f"This Loom file does not exists: {path}")
    waited = 0
    while True:
        try:
            f = h5py.File(path, "r")
            break
        except OSError as e:
            if not _is_lock_error(e):
                ErrorJSON(f"Cannot open the Loom file {path}: {e}")
            if waited >= OPEN_MAX_WAIT:
                ErrorJSON("The file cannot be unlocked")
            print("The file is locked.")
            time.sleep(1)
            waited += 1
    # LoomFile.checkLoomFormat (same message concatenation)
    msg = ""
    if not isinstance(f.get("matrix"), h5py.Dataset): msg += "Loom format is not correct, there is no /matrix dataset\n"
    for g in ("col_attrs", "col_graphs", "row_attrs", "row_graphs"):
        if not isinstance(f.get(g), h5py.Group): msg += f"Loom format is not correct, there is no /{g} group"
    if msg:
        f.close()
        ErrorJSON(msg)
    if len(f["matrix"].shape) != 2:
        f.close()
        ErrorJSON("Loom format is not correct, /matrix should be a 2D dataset")
    return f

def read_vector(f: h5py.File, path: str, expected_len: int, what: str) -> np.ndarray:
    if not isinstance(f.get(path), h5py.Dataset):
        ErrorJSON(f"Error in the Loom file. Path {path} should exist!")
    values = np.asarray(f[path][()]).reshape(-1)
    if values.shape[0] != expected_len:
        ErrorJSON(f"Error in the Loom file. Path {path} has {values.shape[0]} values but /matrix has {expected_len} {what}")
    return values

def read_strings(ds: h5py.Dataset) -> np.ndarray:
    raw = np.asarray(ds[()]).reshape(-1)
    return np.array([v.decode("utf-8", errors="replace") if isinstance(v, (bytes, np.bytes_)) else str(v) for v in raw], dtype=object)

def dataset_size(ds: h5py.Dataset) -> int:
    """LoomFile.getSizeInBytes: number of elements (min 1) x element size; variable-length strings count 16 bytes."""
    nb = 1 if ds.shape is None else int(np.prod(ds.shape, dtype=np.int64))
    if nb == 0: nb = 1
    dt = ds.dtype
    sinfo = h5py.check_string_dtype(dt)
    if (sinfo is not None and sinfo.length is None) or h5py.check_vlen_dtype(dt) is not None:
        esize = 16
    else:
        esize = dt.itemsize
    return int(nb * esize)

def _item_bytes(dt: np.dtype) -> int:
    return STRING_ITEM_BYTES if dt.kind == "O" else max(1, dt.itemsize)

def _clip_chunks(chunks: Optional[tuple], shape: tuple) -> Optional[tuple]:
    if chunks is None: return None
    return tuple(max(1, min(int(c), int(s))) for c, s in zip(chunks, shape))

def copy_attrs(src: h5py.HLObject, dst: h5py.HLObject) -> None:
    """HDF5 attributes (e.g. column_names of a DE table); best effort, ASAP.jar copied none."""
    for name in src.attrs:
        try:
            dst.attrs.create(name, src.attrs[name], dtype=src.attrs.get_id(name).dtype)
        except Exception:
            try:
                dst.attrs[name] = src.attrs[name]
            except Exception as e:
                print(f"Could not copy attribute {name} of {src.name}: {e}", file=sys.stderr)

def create_like(dst: h5py.File, path: str, src: h5py.Dataset, shape: tuple, expression_matrix: bool = False) -> h5py.Dataset:
    """New dataset with the type, chunking and compression of src (expression matrices are always chunked + gzip)."""
    kwargs: dict = {}
    if 0 not in shape:
        if src.chunks is not None:
            kwargs = {"chunks": _clip_chunks(src.chunks, shape), "compression": src.compression,
                      "compression_opts": src.compression_opts, "shuffle": src.shuffle}
        elif expression_matrix:
            kwargs = {"chunks": _clip_chunks((DEFAULT_MATRIX_CHUNK,) * len(shape), shape),
                      "compression": "gzip", "compression_opts": DEFAULT_GZIP_LEVEL}
    out = dst.create_dataset(path, shape=shape, dtype=src.dtype, **kwargs)
    copy_attrs(src, out)
    return out

def _block_lengths(src: h5py.Dataset) -> tuple[int, int]:
    """Block sides along axes 0 and 1: multiples of the input chunks (no chunk decompressed twice), ~TILE_BYTES."""
    shape = src.shape
    n0, n1 = shape[0], (shape[1] if len(shape) > 1 else 1)
    rest = int(np.prod(shape[2:], dtype=np.int64)) if len(shape) > 2 else 1
    chunks = src.chunks or tuple(min(s, DEFAULT_MATRIX_CHUNK) for s in shape)
    c0 = max(1, chunks[0])
    c1 = max(1, chunks[1]) if len(shape) > 1 else 1
    item = _item_bytes(src.dtype) * rest
    t1 = n1 if n1 * c0 * item <= TILE_BYTES else max(c1, (TILE_BYTES // (c0 * item)) // c1 * c1)
    t0 = max(c0, (TILE_BYTES // max(1, t1 * item)) // c0 * c0)
    return max(1, min(t0, n0)), max(1, min(t1, n1))

def copy_filtered(src: h5py.Dataset, dst: h5py.Dataset, keep0: Optional[np.ndarray], keep1: Optional[np.ndarray],
                  on_block: Optional[Callable[[np.ndarray, int, int], None]] = None) -> None:
    """dst = src[keep0, keep1, ...] block by block. keep0/keep1: sorted kept indexes on axes 0/1 (None = all).
    on_block(sub, out_row, out_col) sees every written block (used for the /matrix statistics)."""
    shape = src.shape
    if 0 in dst.shape:
        return
    ndim = len(shape)
    k0 = np.arange(shape[0]) if keep0 is None else keep0
    k1 = None if ndim < 2 else (np.arange(shape[1]) if keep1 is None else keep1)
    t0, t1 = _block_lengths(src)
    for r0 in range(0, shape[0], t0):
        r1 = min(shape[0], r0 + t0)
        a, b = np.searchsorted(k0, r0), np.searchsorted(k0, r1)
        if a == b: continue
        rsel = None if (b - a) == (r1 - r0) else k0[a:b] - r0
        if ndim == 1:
            block = src[r0:r1]
            dst[a:b] = block if rsel is None else block[rsel]
            continue
        for c0 in range(0, shape[1], t1):
            c1 = min(shape[1], c0 + t1)
            ca, cb = np.searchsorted(k1, c0), np.searchsorted(k1, c1)
            if ca == cb: continue
            csel = None if (cb - ca) == (c1 - c0) else k1[ca:cb] - c0
            sub = src[r0:r1, c0:c1]
            if rsel is not None: sub = sub[rsel]
            if csel is not None: sub = sub[:, csel]
            dst[a:b, ca:cb] = sub
            if on_block is not None: on_block(sub, a, ca)

## Statistics of the filtered /matrix (what LoomFile.copyFloatMatrixByGene/ByCell accumulate in LoomData)
class MatrixStats:
    def __init__(self, n_genes: int, n_cells: int, gene_stats: bool, cell_stats: bool):
        self.nber_zeros = 0
        self.is_count_table = True
        self.gene_stats, self.cell_stats = gene_stats, cell_stats
        self.gene_sum = np.zeros(n_genes, dtype=np.float64) # /row_attrs/_Sum (FilterCols)
        self.depth = np.zeros(n_cells, dtype=np.float64) # /col_attrs/_Depth (FilterRows)
        self.detected = np.zeros(n_cells, dtype=np.int64) # /col_attrs/_Detected_Genes (FilterRows)

    def __call__(self, sub: np.ndarray, row: int, col: int) -> None:
        nonzero = np.count_nonzero(sub, axis=0) # value != 0 (NaN counts as detected, as in the Java)
        self.nber_zeros += int(sub.size - nonzero.sum())
        if self.is_count_table:
            v = sub.astype(np.float32, copy=False) # Utils.isInteger(float): |f - round(f)| <= 1e-5 (NaN passes, Inf does not)
            with np.errstate(invalid="ignore"):
                frac = np.abs(v - np.rint(v)).astype(np.float64) > IS_INTEGER_TOL
            if frac.any() or np.isinf(v).any():
                self.is_count_table = False
        v64 = sub.astype(np.float64, copy=False) # the Java sums the float values in doubles
        if self.gene_stats:
            self.gene_sum[row:row + sub.shape[0]] += v64.sum(axis=1)
        if self.cell_stats:
            self.depth[col:col + sub.shape[1]] += v64.sum(axis=0)
            self.detected[col:col + sub.shape[1]] += nonzero

## Metadata (output.json "metadata" entries)
def meta_dims(on: str, shape: Optional[tuple], n_axis: int) -> tuple[int, int]:
    """(nber_rows, nber_cols) in the ASAP Annot convention (the one of parse.v8.py / LoomFile.listMetadata)."""
    if shape is None or len(shape) == 0: return 1, 1
    if on == "EXPRESSION_MATRIX": return int(shape[0]), int(shape[1])
    if on == "CELL":
        if len(shape) == 1: return 1, int(shape[0])
        if shape[0] != n_axis and shape[1] == n_axis: return int(shape[0]), int(shape[1]) # stored k x n_cells
        return int(shape[1]), int(shape[0])
    if on == "GENE":
        if len(shape) == 1: return int(shape[0]), 1
        if shape[0] != n_axis and shape[1] == n_axis: return int(shape[1]), int(shape[0]) # stored k x n_genes
        return int(shape[0]), int(shape[1])
    if len(shape) == 1: return int(shape[0]), 1 # GLOBAL
    return int(shape[0]), int(shape[1])

def meta_entry(path: str, on: str, ds: h5py.Dataset, n_axis: int, mtype: Optional[str] = None) -> dict:
    nber_rows, nber_cols = meta_dims(on, ds.shape, n_axis)
    m = {"name": path, "on": on}
    if mtype is not None: m["type"] = mtype
    m["nber_cols"] = nber_cols
    m["nber_rows"] = nber_rows
    m["dataset_size"] = dataset_size(ds)
    return m

class LoomFilter:
    """Writes output.loom from src, keeping rows keep_rows and cols keep_cols (sorted index arrays)."""
    GROUPS: Final[tuple] = (("row_attrs", "GENE"), ("col_attrs", "CELL"), ("attrs", "GLOBAL"), ("layers", "EXPRESSION_MATRIX")) # LoomFile.listMetadata order

    def __init__(self, src: h5py.File, keep_rows: np.ndarray, keep_cols: np.ndarray, skip: set[str], gene_stats: bool):
        self.src = src
        self.n_genes, self.n_cells = (int(x) for x in src["matrix"].shape)
        self.keep_rows = keep_rows
        self.keep_cols = keep_cols
        self.filter_rows = keep_rows.shape[0] != self.n_genes
        self.filter_cols = keep_cols.shape[0] != self.n_cells
        self.skip = skip
        self.meta: list[dict] = []
        self.warnings: list[str] = []
        self.stats = MatrixStats(keep_rows.shape[0], keep_cols.shape[0], gene_stats=gene_stats, cell_stats=not gene_stats)

    def write(self, out_path: str) -> None:
        with h5py.File(out_path, "w", rdcc_nbytes=OUTPUT_CHUNK_CACHE, rdcc_nslots=100003) as dst:
            # LoomFile.createEmptyLoomFile
            dst.require_group("attrs")
            dst["attrs"].create_dataset("LOOM_SPEC_VERSION", data=LOOM_VERSION, dtype=h5py.string_dtype("utf-8"))
            for g in ("col_attrs", "row_attrs", "layers", "row_graphs", "col_graphs"):
                dst.require_group(g)
            # Main matrix, with the statistics
            self.copy_expression_matrix("/matrix", self.src["matrix"], dst, self.stats)
            # Metadata
            for group, on in self.GROUPS:
                if isinstance(self.src.get(group), h5py.Group):
                    self.copy_group(self.src[group], dst, on)
            self.copy_graphs(dst)

    def copy_graphs(self, dst: h5py.File) -> None:
        """Loom graphs are edge lists a -> b (weight w) of row (gene) or col (cell) indexes: keep the edges whose
        two ends are kept, renumbered to the output indexes. Not listed in output.json (as parse.v8.py)."""
        for gname, on in GRAPH_GROUPS:
            grp = self.src.get(gname)
            if not isinstance(grp, h5py.Group):
                continue
            filtered = self.filter_rows if on == "GENE" else self.filter_cols
            keep = self.keep_rows if on == "GENE" else self.keep_cols
            n_axis = self.n_genes if on == "GENE" else self.n_cells
            new_index = np.full(n_axis, -1, dtype=np.int64)
            new_index[keep] = np.arange(keep.shape[0], dtype=np.int64)
            for name in sorted(grp.keys()):
                path = f"/{gname}/{name}"
                try:
                    obj = grp[name]
                except (KeyError, OSError):
                    continue
                if not filtered:
                    self.src.copy(obj, dst[gname], name=name)
                    continue
                edges = self._graph_edges(obj)
                if edges is None:
                    self.warnings.append(f"{path} was not copied: only graphs stored as Loom edge lists (a, b, w) can be filtered")
                    continue
                a, b = edges
                ok = (a >= 0) & (a < n_axis) & (b >= 0) & (b < n_axis)
                na = np.full(a.shape, -1, dtype=np.int64)
                nb = np.full(b.shape, -1, dtype=np.int64)
                na[ok], nb[ok] = new_index[a[ok]], new_index[b[ok]]
                kept = (na >= 0) & (nb >= 0)
                out = dst[gname].create_group(name)
                copy_attrs(obj, out)
                for key in sorted(obj.keys()):
                    ds = obj[key]
                    if key == "a": values = na[kept]
                    elif key == "b": values = nb[kept]
                    elif isinstance(ds, h5py.Dataset) and ds.shape == (a.shape[0],): values = ds[()][kept] # w and any per-edge value
                    else:
                        self.src.copy(ds, out, name=key)
                        continue
                    out.create_dataset(key, data=values.astype(ds.dtype, copy=False))
                    copy_attrs(ds, out[key])

    @staticmethod
    def _graph_edges(obj: Any) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """(a, b) as int64 arrays when obj is a Loom graph group with 1D integer a and b of the same length."""
        if not isinstance(obj, h5py.Group): return None
        a, b = obj.get("a"), obj.get("b")
        if not (isinstance(a, h5py.Dataset) and isinstance(b, h5py.Dataset)): return None
        if a.ndim != 1 or a.shape != b.shape or a.dtype.kind not in "iu" or b.dtype.kind not in "iu": return None
        return a[()].astype(np.int64), b[()].astype(np.int64)

    def copy_group(self, grp: h5py.Group, dst: h5py.File, on: str) -> None:
        for name in sorted(grp.keys()):
            path = f"{grp.name}/{name}"
            try:
                obj = grp[name]
            except (KeyError, OSError): # dangling soft/external link
                continue
            if isinstance(obj, h5py.Group):
                copy_attrs(obj, dst.require_group(path))
                self.copy_group(obj, dst, on)
            elif isinstance(obj, h5py.Dataset) and path not in self.skip:
                if on == "GLOBAL" and grp.name == "/attrs" and obj.ndim >= 2 and RUN_RESULT_TABLE_RE.match(name):
                    self.warnings.append(f"{path} was not copied: it holds the results of another run (e.g. a DE), "
                                         "computed on the unfiltered data; they stay available in that run")
                    continue
                self.copy_dataset(path, obj, dst, on)

    def copy_dataset(self, path: str, ds: h5py.Dataset, dst: h5py.File, on: str) -> None:
        if path in dst:
            del dst[path] # /attrs/LOOM_SPEC_VERSION is replaced by the input one (LoomFile.copyMetadata)
        if ds.dtype.kind == "O" and h5py.check_string_dtype(ds.dtype) is None and h5py.check_vlen_dtype(ds.dtype) is None:
            self.warnings.append(f"{path} was not copied: HDF5 references are not supported")
            return
        if on == "EXPRESSION_MATRIX":
            if ds.shape is None or tuple(ds.shape) != (self.n_genes, self.n_cells):
                # A Loom layer has the /matrix shape; anything else cannot be filtered consistently
                self.warnings.append(f"{path} was not copied: its shape {tuple(ds.shape or ())} is not the /matrix shape "
                                     f"({self.n_genes} x {self.n_cells})")
                return
            self.copy_expression_matrix(path, ds, dst, None)
            self.meta.append(meta_entry(path, on, dst[path], self.n_genes))
            return
        if on in ("CELL", "GENE") and ds.dtype.kind == "b":
            self.copy_boolean_attribute(path, ds, dst, on)
            return
        axis, keep, n_axis = None, None, (self.n_cells if on == "CELL" else self.n_genes)
        if on in ("CELL", "GENE") and ds.shape is not None and len(ds.shape) > 0:
            if (on == "CELL" and self.filter_cols) or (on == "GENE" and self.filter_rows):
                keep = self.keep_cols if on == "CELL" else self.keep_rows
                if ds.shape[0] == n_axis: axis = 0
                elif len(ds.shape) == 2 and ds.shape[1] == n_axis: axis = 1
                else: self.warnings.append(f"{path} has shape {tuple(ds.shape)}, which does not match the {n_axis} {'cells' if on == 'CELL' else 'genes'}: it was copied unfiltered")
        if axis is None:
            parent, name = path.rsplit("/", 1)
            self.src.copy(ds, dst.require_group(parent), name=name) # unchanged: exact HDF5 copy (type, layout, attributes)
        else:
            shape = list(ds.shape)
            shape[axis] = keep.shape[0]
            out = create_like(dst, path, ds, tuple(shape))
            copy_filtered(ds, out, keep if axis == 0 else None, keep if axis == 1 else None)
        n_axis_out = (self.keep_cols if on == "CELL" else self.keep_rows).shape[0] if on in ("CELL", "GENE") else n_axis
        self.meta.append(meta_entry(path, on, dst[path], n_axis_out))

    def copy_boolean_attribute(self, path: str, ds: h5py.Dataset, dst: h5py.File, on: str) -> None:
        """Loom attributes cannot be booleans (loompy rejects the file): stored as uint8 0/1, filtered like the others."""
        values = np.asarray(ds[()]).astype(np.uint8)
        n_axis = self.n_cells if on == "CELL" else self.n_genes
        keep = self.keep_cols if on == "CELL" else self.keep_rows
        if (self.filter_cols if on == "CELL" else self.filter_rows) and values.ndim > 0:
            if values.shape[0] == n_axis: values = values[keep]
            elif values.ndim == 2 and values.shape[1] == n_axis: values = values[:, keep]
            else: self.warnings.append(f"{path} has shape {tuple(ds.shape)}, which does not match the {n_axis} {'cells' if on == 'CELL' else 'genes'}: it was copied unfiltered")
        out = dst.create_dataset(path, data=values)
        copy_attrs(ds, out)
        self.meta.append(meta_entry(path, on, out, keep.shape[0]))

    def copy_expression_matrix(self, path: str, ds: h5py.Dataset, dst: h5py.File, stats: Optional[MatrixStats]) -> None:
        if tuple(ds.shape) != (self.n_genes, self.n_cells):
            ErrorJSON(f"{path} has shape {tuple(ds.shape)}, but /matrix is {self.n_genes} x {self.n_cells}")
        out = create_like(dst, path, ds, (self.keep_rows.shape[0], self.keep_cols.shape[0]), expression_matrix=True)
        copy_filtered(ds, out, self.keep_rows if self.filter_rows else None, self.keep_cols if self.filter_cols else None, stats)

    def write_vector(self, dst: h5py.File, path: str, values: np.ndarray, on: str) -> None:
        if path in dst:
            del dst[path]
        ds = dst.create_dataset(path, data=values)
        n_axis = self.keep_cols.shape[0] if on == "CELL" else self.keep_rows.shape[0]
        self.meta.append(meta_entry(path, on, ds, n_axis, mtype="NUMERIC"))

def write_output_json(out_dir: str, n_rows: int, n_cols: int, stats: MatrixStats, meta: list[dict], warnings: list[str]) -> None:
    """json.FilterCellsJSON.writeOutputJSON"""
    res = {"detected_format": "LOOM", "loom_version": LOOM_VERSION, "nber_rows": int(n_rows), "nber_cols": int(n_cols),
           "nber_zeros": int(stats.nber_zeros), "is_count_table": 1 if stats.is_count_table else 0, "metadata": meta}
    if warnings:
        res["warnings"] = warnings
    with open(os.path.join(out_dir, OUTPUT_JSON_NAME), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, separators=(",", ":"))

def run_filter(src: h5py.File, keep_rows: np.ndarray, keep_cols: np.ndarray, skip: set[str], gene_stats: bool,
               recompute: Callable[["LoomFilter", h5py.File], None]) -> None:
    """gene_stats: accumulate per-gene sums (FilterCols) instead of per-cell depth / detected genes (FilterRows)."""
    out_dir = _OUTPUT_DIR
    final_path = os.path.join(out_dir, LOOM_FILENAME)
    tmp_path = os.path.join(out_dir, f".{LOOM_FILENAME}.{os.getpid()}.tmp") # input may be <o>/output.loom itself
    lf = LoomFilter(src, keep_rows, keep_cols, skip, gene_stats)
    try:
        print("Now processing the Metadata...")
        lf.write(tmp_path)
        with h5py.File(tmp_path, "r+") as dst:
            recompute(lf, dst)
        src.close()
        os.replace(tmp_path, final_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    write_output_json(out_dir, keep_rows.shape[0], keep_cols.shape[0], lf.stats, lf.meta, lf.warnings)

## FilterCols (FileFilter.filterCells)
def filter_cols(p: dict) -> None:
    print(f"Filtering Cells of file : {p['loom']}")
    src = open_input_loom(p["loom"])
    n_genes, n_cells = (int(x) for x in src["matrix"].shape)
    depth = read_vector(src, "/col_attrs/_Depth", n_cells, "columns")
    drop = depth == 0
    print(f"{int(drop.sum())} empty cells to filter.")
    print("Reading JSON file...", end="")
    data = read_filter_json(p["json"])
    if not p["is_index"]:
        names = data.get("filtered_cells")
        if names is None:
            ErrorJSON("Field 'filtered_cells' is missing from JSON")
        names = {str(v) for v in _as_list(names) if v is not None}
        print(f" OK! {len(names)} cell names in the JSON to filter.")
        if not isinstance(src.get("/col_attrs/CellID"), h5py.Dataset):
            ErrorJSON("Error in the Loom file. Path /col_attrs/CellID should exist!")
        cell_names = read_strings(src["/col_attrs/CellID"])
        if cell_names.shape[0] != n_cells:
            ErrorJSON(f"Error in the Loom file. Path /col_attrs/CellID has {cell_names.shape[0]} values but /matrix has {n_cells} columns")
        drop |= np.fromiter((c in names for c in cell_names), dtype=bool, count=n_cells)
    else:
        if data.get("discarded_cols") is None:
            ErrorJSON("Field 'discarded_cols' is missing from JSON")
        idx = json_int_list(data["discarded_cols"], "discarded_cols")
        bad = idx[(idx < 0) | (idx >= n_cells)]
        if bad.size:
            ErrorJSON(f"'discarded_cols' contains indexes outside of the {n_cells} columns of /matrix (e.g. {int(bad[0])})")
        drop[idx] = True
    print(f"{int(drop.sum())} cells were found and will be filtered.")
    print(f"Current /matrix size = {n_genes} x {n_cells}")
    keep_cols = np.flatnonzero(~drop)
    if keep_cols.size == 0:
        ErrorJSON("No more cells after filtering...")
    print(f"Creating New Loom file with /matrix size = {n_genes} x {keep_cols.size}")

    def recompute(lf: LoomFilter, dst: h5py.File) -> None:
        lf.write_vector(dst, "/row_attrs/_Sum", lf.stats.gene_sum, "GENE")

    run_filter(src, np.arange(n_genes), keep_cols, {"/row_attrs/_Sum"}, True, recompute)

## FilterRows (FileFilter.filterGenes / filterBASIC / filterKEEP / filterCPM)
def cpm_detected(src: h5py.File, rows: np.ndarray, nb_counts_per_cell: int) -> np.ndarray:
    """Per gene, number of cells with (value / (float)depth) * 1000000 > nbCountsPerCell, in float32 as the Java."""
    mat = src["matrix"]
    n_genes, n_cells = mat.shape
    depth = read_vector(src, "/col_attrs/_Depth", n_cells, "columns").astype(np.float32)
    detected = np.zeros(n_genes, dtype=np.int64)
    t0, t1 = _block_lengths(mat)
    thr = np.float32(nb_counts_per_cell)
    print(f"Reading {int(np.ceil(n_genes / t0))} independent blocks...")
    for r0 in range(0, n_genes, t0):
        r1 = min(n_genes, r0 + t0)
        if not rows[r0:r1].any(): continue
        for c0 in range(0, n_cells, t1):
            c1 = min(n_cells, c0 + t1)
            block = mat[r0:r1, c0:c1].astype(np.float32, copy=False)
            with np.errstate(divide="ignore", invalid="ignore"):
                cpm = (block / depth[c0:c1]) * np.float32(1000000)
            detected[r0:r1] += np.count_nonzero(cpm > thr, axis=1)
    return detected

def filter_rows(p: dict) -> None:
    src = open_input_loom(p["loom"])
    n_genes, n_cells = (int(x) for x in src["matrix"].shape)
    gene_sum = read_vector(src, "/row_attrs/_Sum", n_genes, "rows")
    drop = gene_sum == 0
    print(f"{int(drop.sum())} empty genes to filter.")
    if p["model"] == "keep":
        data = read_filter_json(p["json"])
        if data.get("kept_genes") is None:
            ErrorJSON("JSON file does not have 'kept_genes' field")
        kept = json_int_list(data["kept_genes"], "kept_genes")
        keep_mask = np.zeros(n_genes, dtype=bool)
        keep_mask[kept[(kept >= 0) & (kept < n_genes)]] = True # indexes outside of /matrix are ignored (as the Java)
        drop |= ~keep_mask
        print(f"{int(drop.sum())} genes to filter.")
    elif p["model"] == "cpm":
        detected = cpm_detected(src, ~drop, p["nb_counts_per_cell"])
        drop |= detected < p["nb_cells_detected"]
    print(f"Current /matrix size = {n_genes} x {n_cells}")
    keep_rows = np.flatnonzero(~drop)
    if keep_rows.size == 0:
        ErrorJSON("No more genes after filtering...")
    print(f"Creating New Loom file with /matrix size = {keep_rows.size} x {n_cells}")

    def recompute(lf: LoomFilter, dst: h5py.File) -> None:
        lf.write_vector(dst, "/col_attrs/_Depth", lf.stats.depth, "CELL")
        lf.write_vector(dst, "/col_attrs/_Detected_Genes", lf.stats.detected.astype(np.int32), "CELL")

    run_filter(src, keep_rows, np.arange(n_cells), {"/col_attrs/_Depth", "/col_attrs/_Detected_Genes"}, False, recompute)

def main() -> None:
    mode, args = read_mode(sys.argv[1:])
    if len(args) == 0: # Parameters.load: no option => help
        print_help(mode)
        sys.exit(0)
    set_output_dir(mode, args)
    try:
        if mode == "FilterCols":
            filter_cols(load_filter_cols(args))
        else:
            filter_rows(load_filter_rows(args))
    except SystemExit:
        raise
    except Exception as e:
        # Catch-all: unexpected errors still produce a displayed_error (the traceback goes to stderr for debugging)
        import traceback
        traceback.print_exc(file=sys.stderr)
        ErrorJSON(f"Unexpected error while filtering the Loom file ({type(e).__name__}: {e})")

if __name__ == "__main__":
    main()
