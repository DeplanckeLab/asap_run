#!/usr/bin/env Rscript
# normalize.R — Normalize scRNA-seq data from a LOOM file using Seurat v5.
#
# Reads a LOOM file (from parse.py), applies the selected Seurat v5 normalization
# method, appends the normalized layer directly into the input LOOM, and writes output.json.
#
# Required packages: hdf5r, Seurat (>= 5.0), jsonlite, Matrix, optparse

## ── Phase 0: HELP (no library needed, runs instantly) ────────────────────────

HELP_TEXT <- "
Normalization Script — Seurat v5 backend

Reads a LOOM file (from parse.py), applies Seurat v5 normalization, appends the
normalized layer directly into the input LOOM file, and writes output.json.

Options:
  -f / --file        Input LOOM file (from parse.py).                       [required]
  --input_meta       Path to the matrix to normalize inside the LOOM file.  [required]
                       Examples: /matrix   /layers/spliced   /layers/toto
  --output_meta      Path where the normalized matrix will be appended in   [required]
                     the LOOM file. Must be under /layers/.
                       Examples: /layers/normalized   /layers/normalized_1_toto
  --method           Normalization method: LogNormalize | CLR | RC | SCTransform
                                                                             [required]
  -o / --output_dir  Output folder for output.json. [optional, default: input dir]
  --assay            Seurat assay name.             [default: RNA]

  -- NormalizeData parameters (defaults match Seurat v5 NormalizeData) ---------------
  --scale_factor     Scale factor applied before log1p.           [default: 10000]
  --margin           For CLR: 1 = across features, 2 = across cells. [default: 1]

  -- SCTransform parameters (defaults match Seurat v5 SCTransform) --------------------
  --vst_flavor              VST flavour: v1 | v2.                        [default: v2]
  --vars_to_regress         Comma-separated variables to regress
                              (e.g. 'percent.mt,nCount_RNA').            [default: NULL]
  --n_cells                 Cells used for parameter estimation.         [default: NULL (all)]
  --variable_features_n     Top variable features to select.             [default: 3000]
  --variable_features_rv_th Residual-variance threshold for variable
                              feature selection.                          [default: 1.3]
  --residual_features       Comma-separated genes to compute residuals
                              for (NULL = all variable genes).            [default: NULL]
  --do_correct_umi          Correct UMI counts before modelling.         [default: TRUE]
  --do_scale                Scale Pearson residuals.                      [default: FALSE]
  --do_center               Center Pearson residuals.                     [default: TRUE]
  --conserve_memory         Reduce memory usage (slower).                 [default: FALSE]
  --return_only_var_genes   Return residuals only for variable genes.    [default: TRUE]
  --new_assay_name          Name for the new SCT assay.                  [default: SCT]
  --seed_use                Random seed.                                  [default: 1448145]

  --help             Show this message and exit.

Output JSON normalization block:
  is_log_transformed  bool — whether log was applied
  log_type            str  — e.g. 'ln (natural log, log1p)' or 'ln (natural log) — inside CLR'
  log_base            always null for Seurat (natural log only)

Output JSON metadata entry (mirrors parse_v8.py Metadata dataclass):
  name            str  — LOOM-internal path of the normalized layer (= --output_meta)
  on              str  — always 'EXPRESSION_MATRIX'
  type            str  — always 'NUMERIC'
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

## Constants (matching parse_v8 / normalize.py conventions)
LOOM_CHUNK_GENES <- 64L
LOOM_CHUNK_CELLS <- 64L
LOOM_GZIP_LEVEL  <- 2L
OUTPUT_JSON_NAME <- "output.json"

CELL_ID_CANDIDATES <- c("col_attrs/CellID", "col_attrs/cell_id", "col_attrs/barcode", "col_attrs/Barcode", "col_attrs/barcodes", "col_attrs/obs_names")
GENE_ID_CANDIDATES <- c("row_attrs/Gene", "row_attrs/Accession", "row_attrs/Original_Gene", "row_attrs/gene_name", "row_attrs/gene", "row_attrs/var_names")

## CLI option list
option_list <- list(
  # Core I/O — -f, --input_meta, --output_meta, --method are mandatory (validated below)
  make_option(c("-f", "--file"),            type = "character", default = NULL, help = "Input LOOM file (from parse.py). [required]"),
  make_option(c("-o", "--output_dir"),          type = "character", default = NULL, help = "Output folder. [optional, default: input dir]"),
  make_option("--input_meta",               type = "character", default = NULL, help = "Path to normalize inside the LOOM (e.g. /matrix or /layers/toto). [required]"),
  make_option("--output_meta",              type = "character", default = NULL, help = "Path for normalized output inside the LOOM (must be under /layers/). [required]"),
  make_option("--method",                   type = "character", default = NULL, help = "Normalization method: LogNormalize | CLR | RC | SCTransform. [required]"),
  make_option("--assay",                    type = "character", default = "RNA", help = "Seurat assay name. [default: RNA]"),

  # ── NormalizeData parameters (defaults match Seurat v5 NormalizeData) ─────────────────
  make_option("--scale_factor",             type = "double",    default = 10000,     help = "[NormalizeData] Scale factor. [default: 10000]"),
  make_option("--margin",                   type = "integer",   default = 1L,        help = "[NormalizeData/CLR] 1 = across features, 2 = across cells. [default: 1]"),

  # ── SCTransform parameters (defaults match Seurat v5 SCTransform) ────────────────────
  make_option("--vst_flavor",               type = "character", default = "v2",      help = "[SCTransform] VST flavour: v1 | v2. [default: v2]"),
  make_option("--vars_to_regress",          type = "character", default = NULL,      help = "[SCTransform] Comma-separated variables to regress (e.g. percent.mt). [default: NULL]"),
  make_option("--n_cells",                  type = "integer",   default = NULL,      help = "[SCTransform] Cells used for parameter estimation (NULL = all). [default: NULL]"),
  make_option("--variable_features_n",      type = "integer",   default = 3000L,     help = "[SCTransform] Top variable features to select. [default: 3000]"),
  make_option("--variable_features_rv_th",  type = "double",    default = 1.3,       help = "[SCTransform] Residual-variance threshold for variable feature selection. [default: 1.3]"),
  make_option("--residual_features",        type = "character", default = NULL,      help = "[SCTransform] Comma-separated genes to compute residuals for (NULL = all variable). [default: NULL]"),
  make_option("--do_correct_umi",           type = "logical",   default = TRUE,      help = "[SCTransform] Correct UMI counts before modelling. [default: TRUE]"),
  make_option("--do_scale",                 type = "logical",   default = FALSE,     help = "[SCTransform] Scale Pearson residuals. [default: FALSE]"),
  make_option("--do_center",                type = "logical",   default = TRUE,      help = "[SCTransform] Center Pearson residuals. [default: TRUE]"),
  make_option("--conserve_memory",          type = "logical",   default = FALSE,     help = "[SCTransform] Reduce memory usage (slower). [default: FALSE]"),
  make_option("--return_only_var_genes",    type = "logical",   default = TRUE,      help = "[SCTransform] Return residuals only for variable genes. [default: TRUE]"),
  make_option("--new_assay_name",           type = "character", default = "SCT",     help = "[SCTransform] Name for the new SCT assay. [default: SCT]"),
  make_option("--seed_use",                 type = "integer",   default = 1448145L,  help = "[SCTransform] Random seed. [default: 1448145]")
)

parser <- OptionParser(option_list = option_list, add_help_option = FALSE, description = "Normalize a scRNA-seq LOOM file using Seurat v5.")
args   <- parse_args(parser)

## Validate mandatory arguments
if (is.null(args$file))        ErrorJSON("Missing required argument -f (input LOOM file).")
if (is.null(args$input_meta))  ErrorJSON("Missing required argument --input_meta (LOOM-internal path, e.g. /matrix or /layers/toto).")
if (is.null(args$output_meta)) ErrorJSON("Missing required argument --output_meta (LOOM-internal output path, e.g. /layers/normalized_1_toto).")
if (is.null(args$method))      ErrorJSON("Missing required argument --method.")

valid_methods <- c("LogNormalize", "CLR", "RC", "SCTransform")
if (!args$method %in% valid_methods) ErrorJSON(paste0("Unknown method '", args$method, "'. Valid: ", paste(valid_methods, collapse = ", "), "."))
if (!startsWith(args$output_meta, "/layers/")) ErrorJSON(paste0("--output_meta must be a path under /layers/ (e.g. /layers/normalized_1_toto), got: '", args$output_meta, "'."))

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

read_string_dataset <- function(h5, candidates, expected_len) {
  # Try each candidate path; return the first string vector of the right length.
  for (path in candidates) {
    tryCatch({
      if (h5$exists(path)) {
        vals <- h5[[path]][]
        if (length(vals) == expected_len) return(as.character(vals))
      }
    }, error = function(e) NULL)
  }
  NULL
}

write_loom_dataset <- function(h5, path, mat_gxc, n_genes, n_cells) {
  # Write a (n_genes × n_cells) float32 matrix to h5[path] with chunked gzip compression.
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
  ds <- h5$create_dataset(
    name       = path,
    dtype      = h5types$H5T_IEEE_F32LE,
    dims       = c(n_genes, n_cells),
    chunk_dims = c(chunk_g, chunk_c),
    gzip_level = LOOM_GZIP_LEVEL
  )
  ds[1:n_genes, 1:n_cells] <- matrix(as.numeric(mat_gxc), nrow = n_genes, ncol = n_cells)
  invisible(ds$get_storage_size())
}

## Log-type label helpers
seurat_log_info <- function(method) {
  # Returns list(is_log_transformed, log_type, log_base) for each Seurat method.
  # NormalizeData (LogNormalize, CLR) and SCTransform all use natural log internally.
  switch(method,
    LogNormalize = list(is_log_transformed = TRUE,  log_type = "ln (natural log, log1p) — applied by LogNormalize", log_base = NULL),
    CLR          = list(is_log_transformed = TRUE,  log_type = "ln (natural log) — used inside Centered Log-Ratio (CLR)", log_base = NULL),
    RC           = list(is_log_transformed = FALSE, log_type = NULL, log_base = NULL),
    SCTransform  = list(is_log_transformed = TRUE,  log_type = "ln (natural log) — Pearson residuals via regularised NB regression (SCTransform)", log_base = NULL),
    list(is_log_transformed = FALSE, log_type = NULL, log_base = NULL)
  )
}

## ── Shared warnings environment ───────────────────────────────────────────────

warn_env <- new.env(parent = emptyenv())
warn_env$w <- character(0)

## ── Result skeleton ───────────────────────────────────────────────────────────

result <- list(
  parameters       = list(),
  normalization    = list(tool = "Seurat", tool_version = seurat_version, is_log_transformed = FALSE, log_type = NULL, log_base = NULL),
  metadata         = list()
)

## ── Read the LOOM file ────────────────────────────────────────────────────────

h5_in <- tryCatch(
  H5File$new(input_path, mode = "r"),
  error = function(e) ErrorJSON(paste0("Could not open '", args$file, "' as an HDF5/LOOM file. Is it a valid LOOM file?"))
)

# --input_meta is the full LOOM-internal path; hdf5r does not use a leading slash
src_path <- sub("^/", "", args$input_meta)
if (!h5_in$exists(src_path)) ErrorJSON(paste0("Source path '", args$input_meta, "' not found in LOOM file."))

matrix_gxc <- t(h5_in[[src_path]][,])  # hdf5r reverses HDF5 dims (C↔Fortran); transpose restores genes×cells
n_genes    <- nrow(matrix_gxc)
n_cells    <- ncol(matrix_gxc)

cell_ids <- read_string_dataset(h5_in, CELL_ID_CANDIDATES, n_cells)
if (is.null(cell_ids)) cell_ids <- paste0("Cell_", seq_len(n_cells))

gene_ids <- read_string_dataset(h5_in, GENE_ID_CANDIDATES, n_genes)
if (is.null(gene_ids)) gene_ids <- paste0("Gene_", seq_len(n_genes))

if (any(duplicated(gene_ids))) {
  gene_ids   <- make.unique(gene_ids)
  warn_env$w <- c(warn_env$w, "Duplicate gene names detected; made unique with make.unique().")
}

rownames(matrix_gxc) <- gene_ids
colnames(matrix_gxc) <- cell_ids
h5_in$close_all()


## ── Create Seurat object ──────────────────────────────────────────────────────

sparse_counts <- suppressWarnings(Matrix(matrix_gxc, sparse = TRUE))
seurat_obj    <- suppressWarnings(CreateSeuratObject(counts = sparse_counts, assay = args$assay, project = "normalize_r"))

## ── Apply normalization (capture Seurat warnings into warn_env) ───────────────

capture_warnings <- function(expr) {
  withCallingHandlers(expr, warning = function(w) {
    warn_env$w <- c(warn_env$w, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
}

if (args$method %in% c("LogNormalize", "CLR", "RC")) {
  if (args$method %in% c("LogNormalize", "RC")) {
    result$parameters <- list(loom_path = input_path, input_loom_path = args$input_meta, output_loom_path = args$output_meta, method = args$method, scale_factor = args$scale_factor)
  } else { # CLR
    result$parameters <- list(loom_path = input_path, input_loom_path = args$input_meta, output_loom_path = args$output_meta, method = args$method, margin = args$margin)
  }
  seurat_obj <- capture_warnings(suppressMessages(NormalizeData(seurat_obj, normalization.method = args$method, scale.factor = args$scale_factor, margin = args$margin, verbose = FALSE)))
  normalized_data <- GetAssayData(seurat_obj, assay = args$assay, layer = "data")

} else if (args$method == "SCTransform") {
  vars_to_regress_vec   <- if (!is.null(args$vars_to_regress))   trimws(strsplit(args$vars_to_regress,   ",")[[1]]) else NULL
  residual_features_vec <- if (!is.null(args$residual_features)) trimws(strsplit(args$residual_features, ",")[[1]]) else NULL

  # SCTransform requires all cells to have at least 1 count (log_umi covariate = log(total) must be finite).
  # Drop zero-count cells before creating the Seurat object, fill them back with 0 afterwards.
  cell_totals   <- colSums(sparse_counts)
  zero_cells    <- which(cell_totals == 0)
  if (length(zero_cells) > 0) {
    warn_env$w <- c(warn_env$w, paste0(length(zero_cells), " cell(s) with zero total counts excluded before SCTransform and filled with 0 in ", args$output_meta, "."))
    sparse_counts_sct <- sparse_counts[, -zero_cells, drop = FALSE]
    seurat_obj <- suppressWarnings(CreateSeuratObject(counts = sparse_counts_sct, assay = args$assay, project = "normalize_r"))
  } else {
    sparse_counts_sct <- sparse_counts
  }

  result$parameters <- list(loom_path = input_path, input_loom_path = args$input_meta, output_loom_path = args$output_meta, method = args$method, vst_flavor = args$vst_flavor, new_assay_name = args$new_assay_name, vars_to_regress = vars_to_regress_vec, n_cells = args$n_cells, variable_features_n = args$variable_features_n, variable_features_rv_th = args$variable_features_rv_th, residual_features = residual_features_vec, do_correct_umi = args$do_correct_umi, do_scale = args$do_scale, do_center = args$do_center, conserve_memory = args$conserve_memory, return_only_var_genes = args$return_only_var_genes, seed_use = args$seed_use)
  options(future.globals.maxSize = Inf)
  seurat_obj <- capture_warnings(suppressMessages(SCTransform(seurat_obj, assay = args$assay, new.assay.name = args$new_assay_name, vst.flavor = args$vst_flavor, vars.to.regress = vars_to_regress_vec, ncells = args$n_cells, variable.features.n = args$variable_features_n, variable.features.rv.th = args$variable_features_rv_th, residual.features = residual_features_vec, do.correct.umi = args$do_correct_umi, do.scale = args$do_scale, do.center = args$do_center, conserve.memory = args$conserve_memory, return.only.var.genes = args$return_only_var_genes, seed.use = args$seed_use, verbose = FALSE)))
  sct_assay <- args$new_assay_name
  if (!sct_assay %in% Assays(seurat_obj)) ErrorJSON(paste0("SCTransform assay '", sct_assay, "' not found after SCTransform."))
  normalized_data <- GetAssayData(seurat_obj, assay = sct_assay, layer = "scale.data")
  if (nrow(normalized_data) == 0) normalized_data <- GetAssayData(seurat_obj, assay = sct_assay, layer = "data")
}

## ── Align dimensions (SCTransform may return partial gene/cell set) ───────────────

# Re-insert zero-count cells dropped before SCTransform (filled with 0)
if (args$method == "SCTransform" && length(zero_cells) > 0) {
  sct_cols    <- colnames(normalized_data)
  full_mat_c  <- matrix(0, nrow = nrow(normalized_data), ncol = n_cells, dimnames = list(rownames(normalized_data), cell_ids))
  kept_idx    <- match(sct_cols, cell_ids)
  full_mat_c[, kept_idx[!is.na(kept_idx)]] <- as.matrix(normalized_data[, sct_cols[!is.na(kept_idx)], drop = FALSE])
  normalized_data <- full_mat_c
}

if (nrow(normalized_data) != n_genes) {
  found_genes <- rownames(normalized_data)
  warn_env$w  <- c(warn_env$w, paste0("SCTransform returned residuals for ", nrow(normalized_data), "/", n_genes, " genes. Missing genes filled with 0 in ", args$output_meta, "."))
  full_mat              <- matrix(0, nrow = n_genes, ncol = n_cells, dimnames = list(gene_ids, cell_ids))
  idx                   <- match(found_genes, gene_ids)
  valid                 <- !is.na(idx)
  full_mat[idx[valid],] <- as.matrix(normalized_data[found_genes[valid], ])
  normalized_data       <- full_mat
}

if (nrow(normalized_data) != n_genes || ncol(normalized_data) != n_cells) ErrorJSON(paste0("Normalized matrix has unexpected dimensions: ", nrow(normalized_data), " x ", ncol(normalized_data), " (expected ", n_genes, " x ", n_cells, ")."))

# Force dense only once, right before writing — suppress the sparse→dense allocation warning
normalized_gxc <- suppressWarnings(as.matrix(normalized_data))

## ── Log transformation info ───────────────────────────────────────────────────

log_info <- seurat_log_info(args$method)
result$normalization$is_log_transformed <- log_info$is_log_transformed
result$normalization$log_type           <- log_info$log_type
result$normalization$log_base           <- log_info$log_base


## ── Append normalized layer directly into the input LOOM ─────────────────────

h5_loom <- tryCatch(
  H5File$new(input_path, mode = "a"),
  error = function(e) ErrorJSON(paste0("Could not open '", args$file, "' in append mode: ", conditionMessage(e)))
)

out_path_h5 <- sub("^/", "", args$output_meta)   # hdf5r does not use a leading slash
if (h5_loom$exists(out_path_h5)) {
  warn_env$w <- c(warn_env$w, paste0("Path '", args$output_meta, "' already exists in the LOOM file and will be overwritten."))
  # Split into parent group + dataset name and delete via the parent
  parts      <- Filter(nzchar, strsplit(out_path_h5, "/")[[1]])
  parent_grp <- if (length(parts) > 1) h5_loom[[paste(parts[-length(parts)], collapse = "/")]] else h5_loom
  parent_grp$link_delete(parts[length(parts)])
}
if (!h5_loom$exists("layers")) h5_loom$create_group("layers")

norm_size   <- write_loom_dataset(h5_loom, out_path_h5, normalized_gxc, n_genes, n_cells)
h5_loom$close_all()

# Metadata entry — mirrors the Metadata dataclass from parse_v8.py
is_count_norm <- as.integer(all(normalized_gxc %% 1 == 0))
result$metadata   <- c(result$metadata, list(list(
  name            = args$output_meta,
  on              = "EXPRESSION_MATRIX",
  type            = "NUMERIC",
  nber_cols       = as.integer(n_cells),
  nber_rows       = as.integer(n_genes),
  dataset_size    = as.integer(norm_size),
  is_count_table  = is_count_norm,
  imported        = 0L
)))

## ── Emit output.json ─────────────────────────────────────────────────────────

if (length(warn_env$w) > 0) result$warnings <- as.list(warn_env$w)

json_str <- toJSON(result, auto_unbox = TRUE, null = "null", na = "null", pretty = FALSE, digits = 7)
if (!is.null(args$output_dir)) writeLines(json_str, con = output_json_path) else cat(json_str, "\n")