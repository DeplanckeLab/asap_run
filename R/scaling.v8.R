#!/usr/bin/env Rscript
# scaling.R — Scale scRNA-seq data from a LOOM file using Seurat v5.
#
# Reads a LOOM file (from parse.py or normalize.py), applies Seurat's ScaleData,
# appends the scaled layer directly into the input LOOM, and writes output.json.
#
# Required packages: hdf5r, Seurat (>= 5.0), jsonlite, Matrix, optparse

## ── Phase 0: HELP (no library needed, runs instantly) ────────────────────────

HELP_TEXT <- "
Scaling Script — Seurat v5 backend

Reads a LOOM file, applies Seurat v5 ScaleData, appends the scaled layer
directly into the input LOOM file, and writes output.json.

Options:
  -f / --file        Input LOOM file.                                       [required]
  --input_meta       Path to the matrix to scale inside the LOOM file.      [required]
                       Examples: /layers/normalized   /layers/normalized_ln
  --output_meta      Path where the scaled matrix will be appended in        [required]
                     the LOOM file. Must be under /layers/.
                       Examples: /layers/scaled   /layers/scaled_1_toto
  --method           Scaling method: ScaleData                               [required]
  -o / --output_dir  Output folder for output.json. [optional, default: input dir]
  --assay            Seurat assay name.              [default: RNA]

  ── ScaleData parameters (defaults match Seurat v5 ScaleData) ────────────────────
  --do_scale         Divide each gene by its standard deviation.            [default: TRUE]
  --do_center        Center each gene to mean zero.                         [default: TRUE]
  --scale_max        Max value after scaling (clips values above this).     [default: 10]
  --vars_to_regress  Comma-separated /col_attrs/ LOOM paths to regress out
                       (e.g. '/col_attrs/percent_mt,/col_attrs/nCount_RNA'). [default: NULL]
  --features         Path to a LOOM /row_attrs/ dataset with 0/1 flags
                       identifying which genes to scale (NULL = all genes).  [default: NULL]
                       Example: /row_attrs/hvg_norm_ln_vst
  --block_size       Number of genes processed per block (memory control).  [default: 1000]

  --help             Show this message and exit.

Output JSON scaling block:
  tool          str  — always 'Seurat'
  tool_version  str  — installed Seurat version
  do_scale      bool — whether standard-deviation scaling was applied
  do_center     bool — whether mean-centering was applied
  scale_max     num  — clip threshold (null if do_scale is FALSE)

Output JSON metadata entry (mirrors parse_v8.py Metadata dataclass):
  name            str  — LOOM-internal path of the scaled layer (= --output_meta)
  on              str  — always 'EXPRESSION_MATRIX'
  type            str  — always 'NUMERIC'
  nber_cols       int  — number of cells
  nber_rows       int  — number of genes
  dataset_size    int  — on-disk compressed size in bytes
  is_count_table  int  — 1 if all values are integers, 0 otherwise
  imported        int  — always 0 (generated, not imported from source)

Mandatory parameters: -f/--file, --input_meta, --output_meta, --method
"

argv <- commandArgs(trailingOnly = TRUE)
if (length(argv) == 0 || "--help" %in% argv || "-h" %in% argv) { cat(HELP_TEXT); quit(save = "no", status = 0) }

## ── Phase 1: minimal imports for argument parsing and error handling ──────────

suppressPackageStartupMessages({
  library(optparse)
  library(jsonlite)
})

## Error handling (same pattern as parse_v8.py / normalize.py)
ErrorJSON <- function(message, output_path = NULL) {
  payload  <- list(displayed_error = message)
  json_str <- toJSON(payload, auto_unbox = TRUE)
  if (!is.null(output_path)) writeLines(json_str, con = output_path) else cat(json_str, "\n")
  quit(save = "no", status = 1)
}

## Constants (matching parse_v8 / normalize conventions)
LOOM_CHUNK_GENES <- 64L
LOOM_CHUNK_CELLS <- 64L
LOOM_GZIP_LEVEL  <- 2L
LOOM_DTYPE       <- "float64"  # on-disk precision: "float32" or "float64"
OUTPUT_JSON_NAME <- "output.json"

CELL_ID_PATH <- "col_attrs/CellID"
GENE_ID_PATH <- "row_attrs/_StableID"

## CLI option list
option_list <- list(
  # Core I/O — -f, --input_meta, --output_meta, --method are mandatory (validated below)
  make_option(c("-f", "--file"),         type = "character", default = NULL,   help = "Input LOOM file. [required]"),
  make_option(c("-o", "--output_dir"),   type = "character", default = NULL,   help = "Output folder. [optional, default: input dir]"),
  make_option("--input_meta",            type = "character", default = NULL,   help = "Path to scale inside the LOOM (e.g. /layers/normalized). [required]"),
  make_option("--output_meta",           type = "character", default = NULL,   help = "Path for scaled output inside the LOOM (must be under /layers/). [required]"),
  make_option("--method",                type = "character", default = NULL,   help = "Scaling method: ScaleData. [required]"),
  make_option("--assay",                 type = "character", default = "RNA",  help = "Seurat assay name. [default: RNA]"),

  # ── ScaleData parameters (defaults match Seurat v5 ScaleData) ────────────────────────
  make_option("--do_scale",        type = "logical",   default = TRUE,   help = "[ScaleData] Divide by standard deviation. [default: TRUE]"),
  make_option("--do_center",       type = "logical",   default = TRUE,   help = "[ScaleData] Center to mean zero. [default: TRUE]"),
  make_option("--scale_max",       type = "double",    default = 10.0,   help = "[ScaleData] Clip values above this threshold after scaling. [default: 10]"),
  make_option("--vars_to_regress", type = "character", default = NULL,   help = "[ScaleData] Comma-separated /col_attrs/ LOOM paths to regress out. [default: NULL]"),
  make_option("--features",        type = "character", default = NULL,   help = "[ScaleData] /row_attrs/ LOOM path with 0/1 flags (NULL = all genes). [default: NULL]"),
  make_option("--block_size",      type = "integer",   default = 1000L,  help = "[ScaleData] Genes processed per block. [default: 1000]")
)

parser <- OptionParser(option_list = option_list, add_help_option = FALSE, description = "Scale a scRNA-seq LOOM file using Seurat v5.")
args   <- parse_args(parser)

## Validate mandatory arguments
if (is.null(args$file))        ErrorJSON("Missing required argument -f (input LOOM file).")
if (is.null(args$input_meta))  ErrorJSON("Missing required argument --input_meta (LOOM-internal path, e.g. /layers/normalized).")
if (is.null(args$output_meta)) ErrorJSON("Missing required argument --output_meta (LOOM-internal output path, e.g. /layers/scaled_1_toto).")
if (is.null(args$method))      ErrorJSON("Missing required argument --method.")

valid_methods <- c("ScaleData")
if (!args$method %in% valid_methods) ErrorJSON(paste0("Unknown method '", args$method, "'. Valid: ", paste(valid_methods, collapse = ", "), "."))
if (!startsWith(args$output_meta, "/layers/")) ErrorJSON(paste0("--output_meta must be a path under /layers/ (e.g. /layers/scaled_1_toto), got: '", args$output_meta, "'."))

input_path <- normalizePath(args$file, mustWork = FALSE)
if (!file.exists(input_path)) ErrorJSON(paste0("Input LOOM file not found: ", args$file))

out_dir          <- if (!is.null(args$output_dir)) normalizePath(args$output_dir, mustWork = FALSE) else dirname(input_path)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
output_json_path <- file.path(out_dir, OUTPUT_JSON_NAME)

## ── Phase 2: heavy imports (only reached when arguments are valid) ────────────

suppressPackageStartupMessages({
  library(hdf5r)
  library(Matrix)
  library(Seurat)
})

## Validate Seurat version
seurat_version <- tryCatch(as.character(packageVersion("Seurat")), error = function(e) "0.0.0")
if (numeric_version(seurat_version) < numeric_version("5.0.0")) ErrorJSON(paste0("Seurat v5 required (found v", seurat_version, "). Update with: install.packages('Seurat')"))

## ── LOOM helpers (need hdf5r) ─────────────────────────────────────────────────

write_loom_dataset <- function(h5, path, mat_gxc, n_genes, n_cells) {
  # Write a (n_genes × n_cells) matrix to h5[path] with chunked gzip compression (precision = LOOM_DTYPE).
  # Uses hdf5r's native chunk_dims + gzip_level parameters (no custom DCPL) to guarantee
  # exactly one GZIP filter is applied.  Returns on-disk size in bytes.
  chunk_g <- min(n_genes, LOOM_CHUNK_GENES)
  chunk_c <- min(n_cells, LOOM_CHUNK_CELLS)
  parts   <- Filter(nzchar, strsplit(path, "/")[[1]])
  if (length(parts) > 1) {
    grp_path <- paste(parts[-length(parts)], collapse = "/")
    if (!h5$exists(grp_path)) h5$create_group(grp_path)
  }
  if (h5$exists(path)) {
    parent_grp <- if (length(parts) > 1) h5[[paste(parts[-length(parts)], collapse = "/")]] else h5
    parent_grp$link_delete(parts[length(parts)])
  }
  # Transpose before writing: hdf5r stores R matrices in Fortran column-major order,
  # so writing mat_gxc (n_genes × n_cells) would produce a transposed layout in HDF5.
  # Writing t(mat_gxc) with dims = c(n_cells, n_genes) produces the same on-disk layout
  # as Python/h5py, ensuring consistent orientation across both tools.
  ds <- h5$create_dataset(
    name       = path,
    dtype      = if (LOOM_DTYPE == "float64") h5types$H5T_IEEE_F64LE else h5types$H5T_IEEE_F32LE,
    dims       = c(n_cells, n_genes),
    chunk_dims = c(chunk_c, chunk_g),
    gzip_level = LOOM_GZIP_LEVEL
  )
  ds[1:n_cells, 1:n_genes] <- matrix(as.numeric(t(mat_gxc)), nrow = n_cells, ncol = n_genes)
  invisible(ds$get_storage_size())
}

## ── Shared warnings environment ───────────────────────────────────────────────

warn_env <- new.env(parent = emptyenv())
warn_env$w <- character(0)

## ── LOOM file lock retry helper ─────────────────────────────────────────────

open_loom_with_retry <- function(path, mode, max_wait = 600L) {
  # Try to open the LOOM file in the requested mode.
  # If the file is locked by another process, retry every second for up to max_wait seconds.
  # Returns a list: h5 = H5File object, wait_time = seconds waited.
  elapsed <- 0L
  repeat {
    h5 <- tryCatch(H5File$new(path, mode = mode), error = function(e) e)
    if (!inherits(h5, "error")) return(list(h5 = h5, wait_time = elapsed))
    if (elapsed >= max_wait) ErrorJSON(paste0("Could not open '", path, "' in mode '", mode, "' after ", max_wait, "s: ", conditionMessage(h5)))
    Sys.sleep(1)
    elapsed <- elapsed + 1L
  }
}

## ── Result skeleton ───────────────────────────────────────────────────────────

result <- list(
  parameters = list(),
  scaling    = list(tool = "Seurat", tool_version = seurat_version, do_scale = args$do_scale, do_center = args$do_center, scale_max = if (args$do_scale) args$scale_max else NULL),
  metadata   = list(),
  wait_time  = 0L
)

## ── Read the LOOM file ────────────────────────────────────────────────────────

ret   <- open_loom_with_retry(input_path, mode = "r")
h5_in <- ret$h5
result$wait_time <- result$wait_time + ret$wait_time

# --input_meta is the full LOOM-internal path; hdf5r does not use a leading slash
src_path <- sub("^/", "", args$input_meta)
if (!h5_in$exists(src_path)) ErrorJSON(paste0("Source path '", args$input_meta, "' not found in LOOM file."))

# Determine orientation: hdf5r reverses dims on read for Python-written datasets only.
# Use gene/cell ID lengths (single scalar reads) to detect which axis is which.
raw_mat  <- h5_in[[src_path]][,]
ra_len   <- h5_in[[GENE_ID_PATH]]$dims
ca_len   <- h5_in[[CELL_ID_PATH]]$dims
if (!is.na(ra_len) && !is.na(ca_len) && nrow(raw_mat) == ca_len && ncol(raw_mat) == ra_len) {
  matrix_gxc <- t(raw_mat)   # Python-written: hdf5r reversed dims, transpose back
} else {
  matrix_gxc <- raw_mat      # R-written: double reversal already cancelled out
}
n_genes    <- nrow(matrix_gxc)
n_cells    <- ncol(matrix_gxc)

cell_ids <- as.character(h5_in[[CELL_ID_PATH]][])
gene_ids <- as.character(h5_in[[GENE_ID_PATH]][])

rownames(matrix_gxc) <- gene_ids
colnames(matrix_gxc) <- cell_ids

# Read vars_to_regress metadata from LOOM (comma-separated /col_attrs/ paths)
vars_meta <- NULL
if (!is.null(args$vars_to_regress)) {
  vtr_paths <- trimws(strsplit(args$vars_to_regress, ",")[[1]])
  vars_meta <- list()
  for (vp in vtr_paths) {
    vp_h5 <- sub("^/", "", vp)
    if (!h5_in$exists(vp_h5)) ErrorJSON(paste0("--vars_to_regress path '", vp, "' not found in LOOM file."))
    vname <- basename(vp_h5)
    vars_meta[[vname]] <- as.numeric(h5_in[[vp_h5]][])
  }
}

# Read features flags from LOOM (/row_attrs/ 0/1 path, same interface as PCA)
features_from_loom <- NULL
if (!is.null(args$features)) {
  feat_h5 <- sub("^/", "", args$features)
  if (!h5_in$exists(feat_h5)) ErrorJSON(paste0("--features path '", args$features, "' not found in LOOM file."))
  feat_flags <- as.integer(h5_in[[feat_h5]][])
  if (length(feat_flags) != n_genes) ErrorJSON(paste0("--features dataset length (", length(feat_flags), ") does not match number of genes (", n_genes, ")."))
  features_from_loom <- gene_ids[feat_flags == 1L]
  if (length(features_from_loom) == 0L) ErrorJSON("No genes selected by --features (all flags are 0).")
}

h5_in$close_all()

## ── Create Seurat object ──────────────────────────────────────────────────────

# Create Seurat object with a zero sparse matrix as counts (counts are not used by ScaleData).
# This avoids the costly dense→sparse conversion Seurat would otherwise do internally.
fake_counts <- Matrix::Matrix(0, nrow = n_genes, ncol = n_cells, sparse = TRUE, dimnames = list(gene_ids, cell_ids))
seurat_obj  <- suppressWarnings(CreateSeuratObject(counts = fake_counts, assay = args$assay, project = "scaling_r"))
# Set the data layer to the actual input matrix using the Seurat v5 assay-level API.
seurat_obj[[args$assay]] <- SetAssayData(seurat_obj[[args$assay]], layer = "data", new.data = matrix_gxc)

# Inject regression variables as cell metadata
if (!is.null(vars_meta)) {
  for (vname in names(vars_meta)) {
    seurat_obj[[vname]] <- vars_meta[[vname]]
  }
}

## ── Apply scaling (capture Seurat warnings into warn_env) ────────────────────

capture_warnings <- function(expr) {
  withCallingHandlers(expr, warning = function(w) {
    warn_env$w <- c(warn_env$w, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
}

vars_to_regress_vec <- if (!is.null(vars_meta)) names(vars_meta) else NULL
features_vec        <- if (!is.null(features_from_loom)) features_from_loom else rownames(seurat_obj)

result$parameters <- list(
  loom_path        = input_path,
  input_loom_path  = args$input_meta,
  output_loom_path = args$output_meta,
  method           = args$method,
  do_scale         = args$do_scale,
  do_center        = args$do_center,
  scale_max        = if (args$do_scale) args$scale_max else NULL,
  vars_to_regress  = args$vars_to_regress,
  features         = args$features,
  block_size       = args$block_size
)

seurat_obj <- capture_warnings(suppressMessages(ScaleData(
  seurat_obj,
  assay           = args$assay,
  features        = features_vec,
  vars.to.regress = vars_to_regress_vec,
  do.scale        = args$do_scale,
  do.center       = args$do_center,
  scale.max       = args$scale_max,
  block.size      = args$block_size,
  verbose         = FALSE
)))

scaled_data <- GetAssayData(seurat_obj, assay = args$assay, layer = "scale.data")

## ── Align dimensions (ScaleData may return partial gene set if features != all) ──

if (nrow(scaled_data) != n_genes) {
  found_genes <- rownames(scaled_data)
  warn_env$w  <- c(warn_env$w, paste0("ScaleData returned scaled values for ", nrow(scaled_data), "/", n_genes, " genes. Missing genes filled with 0 in ", args$output_meta, "."))
  full_mat              <- matrix(0, nrow = n_genes, ncol = n_cells, dimnames = list(gene_ids, cell_ids))
  idx                   <- match(found_genes, gene_ids)
  valid                 <- !is.na(idx)
  full_mat[idx[valid],] <- as.matrix(scaled_data[found_genes[valid], ])
  scaled_data           <- full_mat
}

if (nrow(scaled_data) != n_genes || ncol(scaled_data) != n_cells) ErrorJSON(paste0("Scaled matrix has unexpected dimensions: ", nrow(scaled_data), " x ", ncol(scaled_data), " (expected ", n_genes, " x ", n_cells, ")."))

# Force dense only once, right before writing — suppress the sparse→dense allocation warning
scaled_gxc <- suppressWarnings(as.matrix(scaled_data))

## ── Append scaled layer directly into the input LOOM ─────────────────────────

ret     <- open_loom_with_retry(input_path, mode = "a")
h5_loom <- ret$h5
result$wait_time <- result$wait_time + ret$wait_time

out_path_h5 <- sub("^/", "", args$output_meta)   # hdf5r does not use a leading slash
if (h5_loom$exists(out_path_h5)) {
  warn_env$w <- c(warn_env$w, paste0("Path '", args$output_meta, "' already exists in the LOOM file and will be overwritten."))
  parts      <- Filter(nzchar, strsplit(out_path_h5, "/")[[1]])
  parent_grp <- if (length(parts) > 1) h5_loom[[paste(parts[-length(parts)], collapse = "/")]] else h5_loom
  parent_grp$link_delete(parts[length(parts)])
}
if (!h5_loom$exists("layers")) h5_loom$create_group("layers")

scaled_size <- write_loom_dataset(h5_loom, out_path_h5, scaled_gxc, n_genes, n_cells)
h5_loom$close_all()

# Metadata entry — mirrors the Metadata dataclass from parse_v8.py
is_count_scaled <- as.integer(all(scaled_gxc %% 1 == 0))
result$metadata <- c(result$metadata, list(list(
  name           = args$output_meta,
  on             = "EXPRESSION_MATRIX",
  type           = "NUMERIC",
  nber_cols      = as.integer(n_cells),
  nber_rows      = as.integer(n_genes),
  dataset_size   = as.integer(scaled_size),
  is_count_table = is_count_scaled,
  imported       = 0L
)))

## ── Emit output.json ─────────────────────────────────────────────────────────

if (length(warn_env$w) > 0) result$warnings <- as.list(warn_env$w)

json_str <- toJSON(result, auto_unbox = TRUE, null = "null", na = "null", pretty = FALSE, digits = 7)
if (!is.null(args$output_dir)) writeLines(json_str, con = output_json_path) else cat(json_str, "\n")