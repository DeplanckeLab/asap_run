#!/usr/bin/env Rscript
# pca.R — PCA on scRNA-seq data from a LOOM file using Seurat v5.
#
# Reads a LOOM file, runs PCA, appends cell embeddings into the input LOOM,
# and writes output.json.
#
# Required packages: hdf5r, Seurat (>= 5.0), jsonlite, Matrix, optparse

## ── Phase 0: HELP (no library needed, runs instantly) ────────────────────────

HELP_TEXT <- "
PCA Script — Seurat v5 backend

Reads a LOOM file, runs Seurat v5 RunPCA, appends cell embeddings directly
into the input LOOM file, and writes output.json.

Options:
  -f / --file        Input LOOM file.                                       [required]
  --input_meta       Path to the scaled/normalized matrix inside the LOOM.  [required]
                       Examples: /layers/scaled   /layers/normalized_ln
  --output_meta      Path where cell embeddings (cells x n_pcs) are stored  [required]
                     in the LOOM file.
                       Examples: /col_attrs/pca   /col_attrs/pca_1_scaled_ln
  --method           PCA method: RunPCA                                     [required]
  -o / --output_dir  Output folder for output.json. [optional, default: stdout]
  --assay            Seurat assay name.              [default: RNA]

  -- RunPCA parameters (defaults match Seurat v5 RunPCA) --------------------------
  --n_pcs            Number of principal components to compute.  [default: 50]
  --features         Path to a LOOM /row_attrs/ dataset with 0/1 flags
                       identifying which genes to use for PCA
                       (e.g. /row_attrs/highly_variable).
                       NULL = use all genes.                      [default: NULL]
  --weight_by_var    Weight cell embeddings by variance.         [default: TRUE]
  --seed_use         Random seed.                                 [default: 42]

  --help             Show this message and exit.

Output JSON pca block:
  tool             str    — always 'Seurat'
  tool_version     str    — installed Seurat version
  n_pcs            int    — number of PCs computed
  n_features       int    — number of genes used for PCA
  variance         array  — variance explained per PC
  variance_ratio   array  — fraction of variance explained per PC

Output JSON metadata entry:
  name          str  — LOOM-internal path of cell embeddings (= --output_meta)
  on            str  — always 'CELL'
  type          str  — always 'NUMERIC'
  nber_rows     int  — number of cells
  nber_cols     int  — number of PCs
  dataset_size  int  — on-disk compressed size in bytes
  imported      int  — always 0

Mandatory parameters: -f/--file, --input_meta, --output_meta, --method
"

argv <- commandArgs(trailingOnly = TRUE)
if (length(argv) == 0 || "--help" %in% argv || "-h" %in% argv) { cat(HELP_TEXT); quit(save = "no", status = 0) }

## ── Phase 1: minimal imports for argument parsing and error handling ──────────

suppressPackageStartupMessages({
  library(optparse)
  library(jsonlite)
})

ErrorJSON <- function(message, output_path = NULL) {
  payload  <- list(displayed_error = message)
  json_str <- toJSON(payload, auto_unbox = TRUE)
  if (!is.null(output_path)) writeLines(json_str, con = output_path) else cat(json_str, "\n")
  quit(save = "no", status = 1)
}

## Constants
LOOM_CHUNK_GENES <- 64L
LOOM_CHUNK_CELLS <- 64L
LOOM_GZIP_LEVEL  <- 2L
LOOM_DTYPE       <- "float64"  # on-disk precision: "float32" or "float64"
OUTPUT_JSON_NAME <- "output.json"

CELL_ID_PATH <- "col_attrs/CellID"
GENE_ID_PATH <- "row_attrs/_StableID"

## CLI
option_list <- list(
  make_option(c("-f", "--file"),       type = "character", default = NULL,  help = "Input LOOM file. [required]"),
  make_option(c("-o", "--output_dir"), type = "character", default = NULL,  help = "Output folder for output.json. [optional, default: stdout]"),
  make_option("--input_meta",          type = "character", default = NULL,  help = "Path to the matrix to run PCA on. [required]"),
  make_option("--output_meta",         type = "character", default = NULL,  help = "Path for cell embeddings in the LOOM (e.g. /col_attrs/pca). [required]"),
  make_option("--method",              type = "character", default = NULL,  help = "PCA method: RunPCA. [required]"),
  make_option("--assay",               type = "character", default = "RNA", help = "Seurat assay name. [default: RNA]"),
  make_option("--n_pcs",               type = "integer",   default = 50L,   help = "[RunPCA] Number of PCs. [default: 50]"),
  make_option("--features",            type = "character", default = NULL,  help = "[RunPCA] LOOM /row_attrs/ path with 0/1 HVG flags (NULL = all genes). [default: NULL]"),
  make_option("--weight_by_var",       type = "logical",   default = TRUE,  help = "[RunPCA] Weight embeddings by variance. [default: TRUE]"),
  make_option("--seed_use",            type = "integer",   default = 42L,   help = "[RunPCA] Random seed. [default: 42]")
)

parser <- OptionParser(option_list = option_list, add_help_option = FALSE)
args   <- parse_args(parser)

if (is.null(args$file))        ErrorJSON("Missing required argument -f.")
if (is.null(args$input_meta))  ErrorJSON("Missing required argument --input_meta.")
if (is.null(args$output_meta)) ErrorJSON("Missing required argument --output_meta.")
if (is.null(args$method))      ErrorJSON("Missing required argument --method.")

valid_methods <- c("RunPCA")
if (!args$method %in% valid_methods) ErrorJSON(paste0("Unknown method '", args$method, "'. Valid: ", paste(valid_methods, collapse = ", "), "."))
if (!startsWith(args$output_meta, "/")) ErrorJSON(paste0("--output_meta must start with / (e.g. /col_attrs/pca), got: '", args$output_meta, "'."))

input_path <- normalizePath(args$file, mustWork = FALSE)
if (!file.exists(input_path)) ErrorJSON(paste0("Input LOOM file not found: ", args$file))

out_dir          <- if (!is.null(args$output_dir)) normalizePath(args$output_dir, mustWork = FALSE) else NULL
if (!is.null(out_dir)) dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
output_json_path <- if (!is.null(out_dir)) file.path(out_dir, OUTPUT_JSON_NAME) else NULL

## ── Phase 2: heavy imports ────────────────────────────────────────────────────

suppressPackageStartupMessages({
  library(hdf5r)
  library(Matrix)
  library(Seurat)
})

seurat_version <- tryCatch(as.character(packageVersion("Seurat")), error = function(e) "0.0.0")
if (numeric_version(seurat_version) < numeric_version("5.0.0")) ErrorJSON(paste0("Seurat v5 required (found v", seurat_version, ")."))

## ── LOOM helpers ──────────────────────────────────────────────────────────────

write_2d_dataset <- function(h5, path, mat, dim1, dim2) {
  # Write a (dim1 x dim2) matrix. Same transpose trick as normalize/scaling:
  # write t(mat) with dims = c(dim2, dim1) to match Python/h5py on-disk layout.
  chunk1 <- min(dim1, LOOM_CHUNK_CELLS)
  chunk2 <- min(dim2, LOOM_CHUNK_GENES)
  dtype  <- if (LOOM_DTYPE == "float64") h5types$H5T_IEEE_F64LE else h5types$H5T_IEEE_F32LE
  parts  <- Filter(nzchar, strsplit(path, "/")[[1]])
  if (length(parts) > 1) {
    grp <- paste(parts[-length(parts)], collapse = "/")
    if (!h5$exists(grp)) h5$create_group(grp)
  }
  if (h5$exists(path)) {
    pg <- if (length(parts) > 1) h5[[paste(parts[-length(parts)], collapse = "/")]] else h5
    pg$link_delete(parts[length(parts)])
  }
  ds <- h5$create_dataset(name = path, dtype = dtype, dims = c(dim2, dim1),
                           chunk_dims = c(chunk2, chunk1), gzip_level = LOOM_GZIP_LEVEL)
  ds[1:dim2, 1:dim1] <- matrix(as.numeric(t(mat)), nrow = dim2, ncol = dim1)
  invisible(ds$get_storage_size())
}

## ── Shared warnings + retry helper ───────────────────────────────────────────

warn_env <- new.env(parent = emptyenv())
warn_env$w <- character(0)

open_loom_with_retry <- function(path, mode, max_wait = 600L) {
  elapsed <- 0L
  repeat {
    h5 <- tryCatch(H5File$new(path, mode = mode), error = function(e) e)
    if (!inherits(h5, "error")) return(list(h5 = h5, wait_time = elapsed))
    if (elapsed >= max_wait) ErrorJSON(paste0("Could not open '", path, "' after ", max_wait, "s: ", conditionMessage(h5)))
    Sys.sleep(1)
    elapsed <- elapsed + 1L
  }
}

capture_warnings <- function(expr) {
  withCallingHandlers(expr, warning = function(w) {
    warn_env$w <- c(warn_env$w, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
}

## ── Result skeleton ───────────────────────────────────────────────────────────

result <- list(
  parameters = list(),
  pca        = list(tool = "Seurat", tool_version = seurat_version,
                    n_pcs = args$n_pcs, n_features = -1L,
                    variance = NULL, variance_ratio = NULL),
  metadata   = list(),
  wait_time  = 0L
)

## ── Read the LOOM file ────────────────────────────────────────────────────────

ret   <- open_loom_with_retry(input_path, mode = "r")
h5_in <- ret$h5
result$wait_time <- result$wait_time + ret$wait_time

src_path <- sub("^/", "", args$input_meta)
if (!h5_in$exists(src_path)) ErrorJSON(paste0("Source path '", args$input_meta, "' not found in LOOM file."))

raw_mat <- h5_in[[src_path]][,]
ra_len  <- h5_in[[GENE_ID_PATH]]$dims
ca_len  <- h5_in[[CELL_ID_PATH]]$dims
if (!is.na(ra_len) && !is.na(ca_len) && nrow(raw_mat) == ca_len && ncol(raw_mat) == ra_len) {
  matrix_gxc <- t(raw_mat)   # Python-written: transpose back to genes x cells
} else {
  matrix_gxc <- raw_mat      # R-written: double reversal already cancelled out
}
n_genes <- nrow(matrix_gxc)
n_cells <- ncol(matrix_gxc)

cell_ids <- as.character(h5_in[[CELL_ID_PATH]][])
gene_ids <- as.character(h5_in[[GENE_ID_PATH]][])

# Read HVG flags if --features is provided
if (!is.null(args$features)) {
  hvg_path <- sub("^/", "", args$features)
  if (!h5_in$exists(hvg_path)) ErrorJSON(paste0("--features path '", args$features, "' not found in LOOM file."))
  hvg_flags    <- as.integer(h5_in[[hvg_path]][])
  if (length(hvg_flags) != n_genes) ErrorJSON(paste0("--features dataset length (", length(hvg_flags), ") does not match number of genes (", n_genes, ")."))
  features_vec <- gene_ids[hvg_flags == 1L]
  if (length(features_vec) == 0) ErrorJSON("No genes selected by --features (all flags are 0).")
} else {
  features_vec <- gene_ids
}

rownames(matrix_gxc) <- gene_ids
colnames(matrix_gxc) <- cell_ids
h5_in$close_all()

## ── Cap n_pcs ─────────────────────────────────────────────────────────────────

n_pcs_actual <- min(args$n_pcs, length(features_vec) - 1L, n_cells - 1L)
if (n_pcs_actual < args$n_pcs) {
  warn_env$w <- c(warn_env$w, paste0("n_pcs capped from ", args$n_pcs, " to ", n_pcs_actual, " (min(n_features, n_cells) - 1)."))
}

result$parameters <- list(
  loom_path        = input_path,
  input_loom_path  = args$input_meta,
  output_loom_path = args$output_meta,
  method           = args$method,
  n_pcs            = n_pcs_actual,
  features         = args$features,
  weight_by_var    = args$weight_by_var,
  seed_use         = args$seed_use
)
result$pca$n_pcs      <- n_pcs_actual
result$pca$n_features <- length(features_vec)

## ── Create Seurat object and run PCA ─────────────────────────────────────────

# Use zero sparse counts; RunPCA reads from scale.data.
fake_counts <- Matrix::Matrix(0, nrow = n_genes, ncol = n_cells, sparse = TRUE,
                               dimnames = list(gene_ids, cell_ids))
seurat_obj  <- suppressWarnings(CreateSeuratObject(counts = fake_counts, assay = args$assay, project = "pca_r"))
seurat_obj[[args$assay]] <- SetAssayData(seurat_obj[[args$assay]], layer = "scale.data", new.data = matrix_gxc)

seurat_obj <- capture_warnings(suppressMessages(RunPCA(
  seurat_obj,
  assay         = args$assay,
  features      = features_vec,
  npcs          = n_pcs_actual,
  weight.by.var = args$weight_by_var,
  seed.use      = args$seed_use,
  verbose       = FALSE
)))

cell_embeddings <- Embeddings(seurat_obj, reduction = "pca")   # (n_cells x n_pcs)
stdev           <- seurat_obj[["pca"]]@stdev
variance        <- stdev^2
variance_ratio  <- variance / sum(variance)

result$pca$variance       <- as.numeric(variance)
result$pca$variance_ratio <- as.numeric(variance_ratio)

## ── Append cell embeddings into the LOOM ─────────────────────────────────────

ret     <- open_loom_with_retry(input_path, mode = "a")
h5_loom <- ret$h5
result$wait_time <- result$wait_time + ret$wait_time

out_path_h5 <- sub("^/", "", args$output_meta)
if (h5_loom$exists(out_path_h5)) {
  warn_env$w <- c(warn_env$w, paste0("Path '", args$output_meta, "' already exists and will be overwritten."))
}
size_ce <- write_2d_dataset(h5_loom, out_path_h5, cell_embeddings, n_cells, n_pcs_actual)
h5_loom$close_all()

## ── Metadata + JSON ──────────────────────────────────────────────────────────

result$metadata <- list(list(
  name         = args$output_meta,
  on           = "CELL",
  type         = "NUMERIC",
  nber_rows    = as.integer(n_cells),
  nber_cols    = as.integer(n_pcs_actual),
  dataset_size = as.integer(size_ce),
  imported     = 0L
))

if (length(warn_env$w) > 0) result$warnings <- as.list(warn_env$w)

json_str <- toJSON(result, auto_unbox = TRUE, null = "null", na = "null", pretty = FALSE, digits = 7)
if (!is.null(output_json_path)) writeLines(json_str, con = output_json_path) else cat(json_str, "\n")