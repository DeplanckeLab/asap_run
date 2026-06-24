#!/usr/bin/env Rscript
# viz_bulk.R — PCA on normalized bulk RNA-seq data from a LOOM file.
#
# Reads a normalized matrix, computes PCA on the top variable genes,
# appends PC coordinates as col attributes in the input LOOM, and writes output.json.
#
# Required packages: hdf5r, jsonlite, optparse

## ── Phase 0: HELP (no library needed, runs instantly) ────────────────────────

HELP_TEXT <- "
PCA — bulk RNA-seq

Reads a normalized matrix from a LOOM file, computes PCA on the most variable
genes, and appends per-sample PC coordinates as col attributes in the LOOM.

Options:
  -f / --file        Input LOOM file.                                        [required]
  --input_meta       LOOM-internal path of the normalized matrix.            [required]
                       Examples: /layers/normalized_vst   /layers/normalized_edgeR
  --output_meta      LOOM col_attr prefix for PC coordinates.                [required]
                       Must be under /col_attrs/.
                       Writes: <prefix>_1, <prefix>_2, ..., <prefix>_N
                       Example: /col_attrs/PCA
  -o / --output_dir  Output folder for output.json. [optional, default: input dir]

  -- PCA parameters ----------------------------------------------------------------
  --top_var_genes  Number of most variable genes used for PCA (0 = all genes).
                     [default: 500]
  --filter_meta    LOOM row_attr path of filter mask from filter_bulk.R.
                     Filtered-out genes are excluded before PCA.             [optional]
                     Example: /row_attrs/filter_pass
  --n_pcs          Number of PCs to compute and store.         [default: 10]
  --scale_data     Scale genes to unit variance before PCA.    [default: TRUE]

  --help        Show this message and exit.

Output LOOM col attrs (written to input LOOM):
  <prefix>_1  ...  <prefix>_N   PC coordinate per sample

Mandatory parameters: -f/--file, --input_meta, --output_meta
"

argv <- commandArgs(trailingOnly = TRUE)
if (length(argv) == 0 || "--help" %in% argv || "-h" %in% argv) { cat(HELP_TEXT); quit(save = "no", status = 0) }

## ── Phase 1: minimal imports ─────────────────────────────────────────────────

suppressPackageStartupMessages({ library(optparse); library(jsonlite) })

ErrorJSON <- function(message, output_path = NULL) {
  payload  <- list(displayed_error = message)
  json_str <- toJSON(payload, auto_unbox = TRUE)
  if (!is.null(output_path)) writeLines(json_str, con = output_path) else cat(json_str, "\n")
  quit(save = "no", status = 1)
}

LOOM_CHUNK_CELLS <- 1024L
LOOM_GZIP_LEVEL  <- 4L
OUTPUT_JSON_NAME <- "output.json"

option_list <- list(
  make_option(c("-f", "--file"),       type = "character", default = NULL,   help = "Input LOOM file. [required]"),
  make_option(c("-o", "--output_dir"), type = "character", default = NULL,   help = "Output folder for output.json. [optional, default: input dir]"),
  make_option("--input_meta",          type = "character", default = NULL,   help = "LOOM-internal path of the normalized matrix. [required]"),
  make_option("--output_meta",         type = "character", default = NULL,   help = "LOOM col_attr prefix for PC coordinates (e.g. /col_attrs/PCA). [required]"),
  make_option("--top_var_genes",       type = "integer",   default = 500L,   help = "Top variable genes for PCA (0 = all). [default: 500]"),
  make_option("--filter_meta",         type = "character", default = NULL,   help = "LOOM row_attr path of filter mask from filter_bulk.R (e.g. /row_attrs/filter_pass). [optional]"),
  make_option("--n_pcs",               type = "integer",   default = 10L,    help = "Number of PCs to compute and store. [default: 10]"),
  make_option("--scale_data",          type = "logical",   default = TRUE,   help = "Scale genes to unit variance before PCA. [default: TRUE]")
)

parser <- OptionParser(option_list = option_list, add_help_option = FALSE)
args   <- parse_args(parser)

if (is.null(args$file))        ErrorJSON("Missing required argument -f.")
if (is.null(args$input_meta))  ErrorJSON("Missing required argument --input_meta.")
if (is.null(args$output_meta)) ErrorJSON("Missing required argument --output_meta.")
if (!startsWith(args$output_meta, "/col_attrs/")) ErrorJSON(paste0("--output_meta must be under /col_attrs/ (e.g. /col_attrs/PCA), got: '", args$output_meta, "'."))

input_path <- normalizePath(args$file, mustWork = FALSE)
if (!file.exists(input_path)) ErrorJSON(paste0("Input LOOM file not found: ", args$file))

out_dir          <- if (!is.null(args$output_dir)) normalizePath(args$output_dir, mustWork = FALSE) else dirname(input_path)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
output_json_path <- file.path(out_dir, OUTPUT_JSON_NAME)

## ── Phase 2: heavy imports ────────────────────────────────────────────────────

suppressPackageStartupMessages(library(hdf5r))

warn_env <- new.env(parent = emptyenv())
warn_env$w <- character(0)

## ── Read normalized matrix from LOOM ─────────────────────────────────────────

h5_in <- tryCatch(
  H5File$new(input_path, mode = "r"),
  error = function(e) ErrorJSON(paste0("Could not open '", args$file, "' as HDF5/LOOM."))
)

src_path <- sub("^/", "", args$input_meta)
if (!h5_in$exists(src_path)) ErrorJSON(paste0("Source path '", args$input_meta, "' not found in LOOM."))

h5_dims   <- h5_in[[src_path]]$dims      # hdf5r reverses HDF5 dims: [1]=n_samples, [2]=n_genes
n_samples <- as.integer(h5_dims[1])
n_genes   <- as.integer(h5_dims[2])
matrix_gxc <- t(h5_in[[src_path]][,])   # [,] reads (n_samples × n_genes); t() gives (n_genes × n_samples)

# Read optional filter mask
filter_mask <- NULL
if (!is.null(args$filter_meta)) {
  fmask_path <- sub("^/", "", args$filter_meta)
  if (!h5_in$exists(fmask_path)) ErrorJSON(paste0("Filter mask '", args$filter_meta, "' not found in LOOM."))
  filter_mask <- as.logical(h5_in[[fmask_path]][])
  if (length(filter_mask) != n_genes) ErrorJSON(paste0("Filter mask length (", length(filter_mask), ") does not match number of genes (", n_genes, ")."))
}
h5_in$close_all()

# Apply filter mask before top-var-gene selection
if (!is.null(filter_mask)) {
  n_before   <- n_genes
  matrix_gxc <- matrix_gxc[filter_mask, ]
  n_genes    <- nrow(matrix_gxc)
  warn_env$w <- c(warn_env$w, paste0("Filter mask applied: ", n_genes, "/", n_before, " genes used for PCA."))
}

## ── Select top variable genes ─────────────────────────────────────────────────

rowVars_fast <- function(m) { rowMeans(m^2) - rowMeans(m)^2 }   # avoids loading matrixStats

n_top <- if (args$top_var_genes > 0 && args$top_var_genes < n_genes) as.integer(args$top_var_genes) else n_genes
if (n_top < n_genes) {
  gene_vars  <- rowVars_fast(matrix_gxc)
  top_idx    <- order(gene_vars, decreasing = TRUE)[seq_len(n_top)]
  matrix_sub <- matrix_gxc[top_idx, ]
} else {
  matrix_sub <- matrix_gxc
}

## ── Compute PCA ───────────────────────────────────────────────────────────────

n_pcs_req  <- min(args$n_pcs, n_samples - 1L, n_top)
pca_res    <- tryCatch(
  prcomp(t(matrix_sub), center = TRUE, scale. = args$scale_data, rank. = n_pcs_req),
  error = function(e) ErrorJSON(paste0("prcomp failed: ", conditionMessage(e)))
)

n_pcs_computed  <- ncol(pca_res$x)
var_explained   <- round(100 * pca_res$sdev^2 / sum(pca_res$sdev^2), 3)
pc_coords       <- pca_res$x   # (n_samples × n_pcs) matrix

## ── Append PC coordinates to LOOM as a single 2D col attr ───────────────────

# pc_coords is (n_samples × n_pcs) in R.
# hdf5r reverses dims on write: writing t(pc_coords) (n_pcs × n_samples) → HDF5 (n_samples × n_pcs) ✓
out_path_h5 <- sub("^/", "", args$output_meta)
h5_loom <- tryCatch(
  H5File$new(input_path, mode = "a"),
  error = function(e) ErrorJSON(paste0("Could not open '", args$file, "' in append mode: ", conditionMessage(e)))
)
if (!h5_loom$exists("col_attrs")) h5_loom$create_group("col_attrs")
if (h5_loom$exists(out_path_h5)) {
  warn_env$w <- c(warn_env$w, paste0("'", args$output_meta, "' already exists and will be overwritten."))
  parts      <- Filter(nzchar, strsplit(out_path_h5, "/")[[1]])
  parent_grp <- if (length(parts) > 1) h5_loom[[paste(parts[-length(parts)], collapse = "/")]] else h5_loom
  parent_grp$link_delete(parts[length(parts)])
}
dtype  <- h5types$H5T_IEEE_F64LE
space  <- H5S$new("simple", dims = c(n_pcs_computed, n_samples), maxdims = c(n_pcs_computed, n_samples))
sink(nullfile())
dcpl <- H5P_DATASET_CREATE$new()
dcpl$set_chunk(c(min(n_pcs_computed, LOOM_CHUNK_CELLS), min(n_samples, LOOM_CHUNK_CELLS)))
dcpl$set_deflate(LOOM_GZIP_LEVEL)
sink()
ds     <- h5_loom$create_dataset(name = out_path_h5, dtype = dtype, space = space, dataset_create_pl = dcpl)
ds[1:n_pcs_computed, 1:n_samples] <- matrix(as.numeric(t(pc_coords)), nrow = n_pcs_computed, ncol = n_samples)
h5_loom$close_all()

## ── Emit output.json ─────────────────────────────────────────────────────────

result <- list(
  loom_path        = input_path,
  tool             = "base R (prcomp)",
  tool_version     = as.character(getRversion()),
  method           = "PCA",
  nber_rows        = n_genes,
  nber_cols        = n_samples,
  input_loom_path  = args$input_meta,
  output_loom_path = args$output_meta,
  parameters       = list(
    input_loom_path = args$input_meta,
    output_loom_path = args$output_meta,
    top_var_genes   = n_top,
    filter_meta     = args$filter_meta,
    n_pcs           = n_pcs_computed,
    scale_data      = args$scale_data
  ),
  n_pcs_computed    = n_pcs_computed,
  variance_explained = var_explained[seq_len(n_pcs_computed)]
)
if (length(warn_env$w) > 0) result$warnings <- as.list(warn_env$w)

json_str <- toJSON(result, auto_unbox = TRUE, null = "null", na = "null", pretty = FALSE, digits = 7)
if (!is.null(args$output_dir)) writeLines(json_str, con = output_json_path) else cat(json_str, "\n")