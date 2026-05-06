#!/usr/bin/env Rscript
# normalize.R — Normalize scRNA-seq data from a LOOM file using Seurat v5.
#
# Reads a LOOM file (from parse.py), applies the selected Seurat v5 normalization
# method, writes a new LOOM (/layers/normalized = normalized data, /matrix = raw
# counts preserved) and output.json.
#
# Required packages: hdf5r, Seurat (>= 5.0), jsonlite, Matrix, optparse

suppressPackageStartupMessages({
  library(optparse)
  library(jsonlite)
  library(hdf5r)
  library(Matrix)
  library(Seurat)
})

## Constants (matching parse_v8 / normalize.py conventions)
LOOM_CHUNK_GENES    <- 1024L
LOOM_CHUNK_CELLS    <- 1024L
LOOM_GZIP_LEVEL     <- 4L
NORMALIZED_LAYER    <- "layers/normalized"
OUTPUT_LOOM_NAME    <- "normalized.loom"
OUTPUT_JSON_NAME    <- "output.json"

CELL_ID_CANDIDATES <- c("col_attrs/CellID", "col_attrs/cell_id", "col_attrs/barcode", "col_attrs/Barcode", "col_attrs/barcodes", "col_attrs/obs_names")
GENE_ID_CANDIDATES <- c("row_attrs/Gene", "row_attrs/Accession", "row_attrs/Original_Gene", "row_attrs/gene_name", "row_attrs/gene", "row_attrs/var_names")

## Error handling (same pattern as parse_v8.py / normalize.py)
ErrorJSON <- function(message, output_path = NULL) {
  payload  <- list(displayed_error = message)
  json_str <- toJSON(payload, auto_unbox = TRUE)
  if (!is.null(output_path)) writeLines(json_str, con = output_path) else cat(json_str, "\n")
  quit(save = "no", status = 1)
}

## LOOM helpers
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
  # Write a (n_genes × n_cells) matrix to h5[path] with gzip compression.
  # Returns on-disk size in bytes.
  chunk_g <- min(n_genes, LOOM_CHUNK_GENES)
  chunk_c <- min(n_cells, LOOM_CHUNK_CELLS)
  dtype   <- h5types$H5T_IEEE_F32LE
  space   <- H5S$new("simple", dims = c(n_genes, n_cells), maxdims = c(n_genes, n_cells))
  dcpl    <- H5P_DATASET_CREATE$new()
  dcpl$set_chunk(c(chunk_g, chunk_c))
  dcpl$set_deflate(LOOM_GZIP_LEVEL)
  parts <- Filter(nzchar, strsplit(path, "/")[[1]])
  if (length(parts) > 1) {
    grp_path <- paste(parts[-length(parts)], collapse = "/")
    if (!h5$exists(grp_path)) h5$create_group(grp_path)
  }
  if (h5$exists(path)) h5[[path]]$`__unlink`()
  ds <- h5$create_dataset(name = path, dtype = dtype, space = space, dataset_create_pl = dcpl)
  ds[1:n_genes, 1:n_cells] <- matrix(as.numeric(mat_gxc), nrow = n_genes, ncol = n_cells)
  invisible(ds$get_storage_size())
}

copy_h5_object <- function(src_obj, dst_h5, dst_path, warnings_env) {
  # Recursively copy an HDF5 object (group or dataset) into dst_h5.
  if (inherits(src_obj, "H5Group")) {
    if (!dst_h5$exists(dst_path)) dst_h5$create_group(dst_path)
    dst_grp <- dst_h5[[dst_path]]
    for (child_key in names(src_obj)) copy_h5_object(src_obj[[child_key]], dst_grp, child_key, warnings_env)
  } else if (inherits(src_obj, "H5D")) {
    if (!dst_h5$exists(dst_path)) {
      tryCatch(dst_h5[[dst_path]] <- src_obj[], error = function(e) {
        warnings_env$w <- c(warnings_env$w, paste0("Could not copy '", dst_path, "': ", conditionMessage(e)))
      })
    }
  }
}

copy_loom_metadata <- function(src_h5, dst_h5, skip_top_keys, warnings_env) {
  # Copy all top-level HDF5 groups/datasets except those in skip_top_keys.
  for (key in names(src_h5)) {
    if (key %in% skip_top_keys) next
    tryCatch(copy_h5_object(src_h5[[key]], dst_h5, key, warnings_env), error = function(e) {
      warnings_env$w <- c(warnings_env$w, paste0("Could not copy '", key, "': ", conditionMessage(e)))
    })
  }
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

## CLI option list
option_list <- list(
  # Core I/O — -f, --input_meta, --method, --layer are mandatory (validated below)
  make_option("-f",                         type = "character", default = NULL, help = "Input LOOM file (from parse.py). [required]"),
  make_option("-o",                         type = "character", default = NULL, help = "Output folder. [optional, default: input dir]"),
  make_option("--input_meta",               type = "character", default = NULL, help = "Input metadata JSON (output.json from parse.py). [required]"),
  make_option("--method",                   type = "character", default = NULL, help = "Normalization method: LogNormalize | CLR | RC | SCTransform. [required]"),
  make_option("--assay",                    type = "character", default = "RNA", help = "Seurat assay name. [default: RNA]"),
  make_option("--layer",                    type = "character", default = NULL, help = 'LOOM layer to normalize. Use "matrix" for /matrix. [required]'),

  # ── NormalizeData parameters (defaults match Seurat v5 NormalizeData) ─────────────────
  make_option("--scale_factor",             type = "double",    default = 10000, help = "[NormalizeData] Scale factor. [default: 10000]"),
  make_option("--margin",                   type = "integer",   default = 1L,    help = "[NormalizeData/CLR] 1 = across features, 2 = across cells. [default: 1]"),

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

parser <- OptionParser(option_list = option_list, description = "Normalize a scRNA-seq LOOM file using Seurat v5.\nWrites a new LOOM (/layers/normalized) and output.json.")
args   <- parse_args(parser)

## Validate mandatory arguments
if (is.null(args$f))           ErrorJSON("Missing required argument -f (input LOOM file).")
if (is.null(args$input_meta))  ErrorJSON("Missing required argument --input_meta.")
if (is.null(args$method))      ErrorJSON("Missing required argument --method.")
if (is.null(args$layer))       ErrorJSON("Missing required argument --layer.")

valid_methods <- c("LogNormalize", "CLR", "RC", "SCTransform")
if (!args$method %in% valid_methods) ErrorJSON(paste0("Unknown method '", args$method, "'. Valid: ", paste(valid_methods, collapse = ", "), "."))

input_path <- normalizePath(args$f, mustWork = FALSE)
if (!file.exists(input_path)) ErrorJSON(paste0("Input LOOM file not found: ", args$f))
if (!file.exists(args$input_meta)) ErrorJSON(paste0("Input metadata JSON not found: ", args$input_meta))

out_dir <- if (!is.null(args$o)) normalizePath(args$o, mustWork = FALSE) else dirname(input_path)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
output_loom_path <- file.path(out_dir, OUTPUT_LOOM_NAME)
output_json_path <- file.path(out_dir, OUTPUT_JSON_NAME)

## Validate Seurat version
seurat_version <- tryCatch(as.character(packageVersion("Seurat")), error = function(e) "0.0.0")
if (numeric_version(seurat_version) < numeric_version("5.0.0")) ErrorJSON(paste0("Seurat v5 required (found v", seurat_version, "). Update with: install.packages('Seurat')"))

## Load input metadata
input_meta_list <- tryCatch({
  loaded <- fromJSON(args$input_meta, simplifyVector = FALSE)
  if (!is.null(loaded$metadata)) loaded$metadata else list()
}, error = function(e) ErrorJSON(paste0("Could not parse input metadata JSON: ", conditionMessage(e))))

## Shared warnings environment (append-only, passed by reference)
warn_env <- new.env(parent = emptyenv())
warn_env$w <- character(0)

## Result skeleton (mirrors parse_v8.py / normalize.py structure)
result <- list(
  input_path   = input_path,
  output_path  = output_loom_path,
  tool         = "Seurat",
  tool_version = seurat_version,
  method       = args$method,
  nber_rows    = -1L,
  nber_cols    = -1L,
  layer_path   = paste0("/", NORMALIZED_LAYER),
  parameters   = list(),
  normalization = list(is_log_transformed = FALSE, log_type = NULL, log_base = NULL),
  statistics   = list(),
  metadata     = input_meta_list
)

## Read the LOOM file
h5_in <- H5File$new(input_path, mode = "r")

src_path <- if (args$layer == "matrix") "matrix" else paste0("layers/", args$layer)
if (!h5_in$exists(src_path)) ErrorJSON(paste0("Source path '", src_path, "' not found in LOOM file."))

matrix_gxc <- h5_in[[src_path]][,]   # LOOM convention: rows = genes, cols = cells
n_genes    <- nrow(matrix_gxc)
n_cells    <- ncol(matrix_gxc)
result$nber_rows <- n_genes
result$nber_cols <- n_cells

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

## Pre-normalization statistics
result$statistics$min_value_before    <- min(matrix_gxc)
result$statistics$max_value_before    <- max(matrix_gxc)
result$statistics$nber_zeros_before   <- as.integer(sum(matrix_gxc == 0))
result$statistics$median_depth_before <- median(colSums(matrix_gxc))

## Create Seurat object
sparse_counts <- suppressWarnings(as(matrix_gxc, "dgCMatrix"))
seurat_obj    <- suppressWarnings(CreateSeuratObject(counts = sparse_counts, assay = args$assay, project = "normalize_r"))

## Apply normalization (capture Seurat warnings into warn_env)
capture_warnings <- function(expr) {
  withCallingHandlers(expr, warning = function(w) {
    warn_env$w <- c(warn_env$w, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
}

if (args$method %in% c("LogNormalize", "CLR", "RC")) {
  result$parameters <- list(
    source_layer         = args$layer,
    normalization_method = args$method,
    scale_factor         = args$scale_factor,
    margin               = args$margin
  )
  seurat_obj <- capture_warnings(suppressMessages(NormalizeData(seurat_obj, normalization.method = args$method, scale.factor = args$scale_factor, margin = args$margin, verbose = FALSE)))
  normalized_data <- GetAssayData(seurat_obj, assay = args$assay, layer = "data")

} else if (args$method == "SCTransform") {
  vars_to_regress_vec   <- if (!is.null(args$vars_to_regress))   trimws(strsplit(args$vars_to_regress,   ",")[[1]]) else NULL
  residual_features_vec <- if (!is.null(args$residual_features)) trimws(strsplit(args$residual_features, ",")[[1]]) else NULL
  result$parameters <- list(
    source_layer            = args$layer,
    vst_flavor              = args$vst_flavor,
    new_assay_name          = args$new_assay_name,
    vars_to_regress         = vars_to_regress_vec,
    n_cells                 = args$n_cells,
    variable_features_n     = args$variable_features_n,
    variable_features_rv_th = args$variable_features_rv_th,
    residual_features       = residual_features_vec,
    do_correct_umi          = args$do_correct_umi,
    do_scale                = args$do_scale,
    do_center               = args$do_center,
    conserve_memory         = args$conserve_memory,
    return_only_var_genes   = args$return_only_var_genes,
    seed_use                = args$seed_use
  )
  seurat_obj <- capture_warnings(suppressMessages(SCTransform(seurat_obj, assay = args$assay, new.assay.name = args$new_assay_name, vst.flavor = args$vst_flavor, vars.to.regress = vars_to_regress_vec, ncells = args$n_cells, variable.features.n = args$variable_features_n, variable.features.rv.th = args$variable_features_rv_th, residual.features = residual_features_vec, do.correct.umi = args$do_correct_umi, do.scale = args$do_scale, do.center = args$do_center, conserve.memory = args$conserve_memory, return.only.var.genes = args$return_only_var_genes, seed.use = args$seed_use, verbose = FALSE)))
  sct_assay <- args$new_assay_name
  if (!sct_assay %in% Assays(seurat_obj)) ErrorJSON(paste0("SCTransform assay '", sct_assay, "' not found after SCTransform."))
  normalized_data <- GetAssayData(seurat_obj, assay = sct_assay, layer = "scale.data")
  if (nrow(normalized_data) == 0) normalized_data <- GetAssayData(seurat_obj, assay = sct_assay, layer = "data")
}

## Convert normalized data to dense (n_genes × n_cells), padding zeros for SCTransform partial output
if (inherits(normalized_data, "sparseMatrix")) normalized_data <- as.matrix(normalized_data)

if (nrow(normalized_data) != n_genes) {
  found_genes <- rownames(normalized_data)
  warn_env$w  <- c(warn_env$w, paste0("SCTransform returned residuals for ", nrow(normalized_data), "/", n_genes, " genes. Missing genes filled with 0 in /layers/normalized."))
  full_mat              <- matrix(0, nrow = n_genes, ncol = n_cells, dimnames = list(gene_ids, cell_ids))
  idx                   <- match(found_genes, gene_ids)
  valid                 <- !is.na(idx)
  full_mat[idx[valid],] <- normalized_data[found_genes[valid], ]
  normalized_data       <- full_mat
}

if (nrow(normalized_data) != n_genes || ncol(normalized_data) != n_cells) ErrorJSON(paste0("Normalized matrix has unexpected dimensions: ", nrow(normalized_data), " x ", ncol(normalized_data), " (expected ", n_genes, " x ", n_cells, ")."))

normalized_gxc <- as.matrix(normalized_data)

## Log transformation info
log_info <- seurat_log_info(args$method)
result$normalization$is_log_transformed <- log_info$is_log_transformed
result$normalization$log_type           <- log_info$log_type
result$normalization$log_base           <- log_info$log_base

## Post-normalization statistics
result$statistics$min_value_after    <- min(normalized_gxc)
result$statistics$max_value_after    <- max(normalized_gxc)
result$statistics$nber_zeros_after   <- as.integer(sum(normalized_gxc == 0))
result$statistics$median_depth_after <- median(colSums(normalized_gxc))

## Write output LOOM
h5_out <- H5File$new(output_loom_path, mode = "w")

for (grp in c("attrs", "col_attrs", "row_attrs", "layers", "row_graphs", "col_graphs")) {
  if (!h5_out$exists(grp)) h5_out$create_group(grp)
}
h5_out[["attrs/LOOM_SPEC_VERSION"]] <- "3.0.0"

write_loom_dataset(h5_out, "matrix",          matrix_gxc,     n_genes, n_cells)
norm_size <- write_loom_dataset(h5_out, NORMALIZED_LAYER, normalized_gxc, n_genes, n_cells)
result$statistics$normalized_layer_disk_size_bytes <- as.integer(norm_size)

# Copy metadata (col_attrs, row_attrs, graphs) and any pre-existing layers from input LOOM
h5_in2 <- H5File$new(input_path, mode = "r")
copy_loom_metadata(h5_in2, h5_out, skip_top_keys = c("matrix", "attrs", "layers"), warn_env)
if (h5_in2$exists("layers")) {
  for (lyr in names(h5_in2[["layers"]])) {
    dst_path <- paste0("layers/", lyr)
    if (!h5_out$exists(dst_path)) {
      tryCatch({
        lyr_data <- h5_in2[[paste0("layers/", lyr)]][,]
        write_loom_dataset(h5_out, dst_path, lyr_data, n_genes, n_cells)
      }, error = function(e) warn_env$w <- c(warn_env$w, paste0("Could not copy layer '", lyr, "': ", conditionMessage(e))))
    }
  }
}
h5_in2$close_all()
h5_out$close_all()

## Finalise and emit output.json
if (length(warn_env$w) > 0) result$warnings <- as.list(warn_env$w)

json_str <- toJSON(result, auto_unbox = TRUE, null = "null", na = "null", pretty = TRUE, digits = 7)
if (!is.null(args$o)) writeLines(json_str, con = output_json_path) else cat(json_str, "\n")
