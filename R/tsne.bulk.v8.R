#!/usr/bin/env Rscript
# tsne_bulk.R — t-SNE on bulk RNA-seq data from a LOOM file using Rtsne (Seurat-independent).
#
# Reads either a 2D embedding (PCA, /col_attrs/...) or an expression matrix
# (/matrix or /layers/..., genes x samples) from a LOOM file, runs t-SNE via Rtsne,
# appends the t-SNE sample embeddings into the input LOOM, and writes output.json.
#
# Required packages: hdf5r, Rtsne, jsonlite, optparse

## ── Phase 0: HELP (no library needed) ────────────────────────────────────────

HELP_TEXT <- "
t-SNE Script (bulk) — Rtsne backend, Seurat-independent

Reads a 2D embedding (PCA) OR an expression matrix from a LOOM file, runs t-SNE
via Rtsne, appends t-SNE sample embeddings into the input LOOM, writes output.json.

Options:
  -f / --file        Input LOOM file.                                       [required]
  --input_meta       LOOM path of the input matrix.                         [required]
                       - Embedding (samples x dims):   /col_attrs/PCA
                       - Expression (genes x samples):  /matrix  or  /layers/normalized
  --output_meta      LOOM /col_attrs/ path for t-SNE embeddings (samples x n_components).
                                                                            [required]
                       Example: /col_attrs/TSNE
  --method           t-SNE method: Rtsne.                                   [required]
  -o / --output_dir  Output folder for output.json. [optional, default: stdout]

  -- Input handling -----------------------------------------------------------
  --filter_meta      LOOM /row_attrs/ path of a boolean filter mask.        [optional]
                       Only applied when --input_meta is an expression matrix.
  --top_var_genes    Top variable genes used (0 = all).                     [default: 0]
                       Only applied when --input_meta is an expression matrix.
  --n_dims           Number of leading dimensions to use from an embedding.  [optional]
                       Only applied when --input_meta is an embedding (/col_attrs/).
                       [default: all available]

  -- t-SNE parameters (Rtsne) -------------------------------------------------
  --n_components     Number of t-SNE dimensions.        [default: 2]
  --perplexity       Perplexity.                         [default: 30]
  --pca              Run internal PCA before t-SNE (TRUE/FALSE). [default: FALSE]
  --seed_use         Random seed.                        [default: 42]

  --help             Show this message and exit.

Mandatory parameters: -f/--file, --input_meta, --output_meta, --method

Output stored in LOOM (/col_attrs/):
  A (n_samples x n_components) embedding at --output_meta.

Notes:
  - Input type is inferred from --input_meta: /col_attrs/ = embedding (samples x dims);
    /matrix and /layers/... = expression (genes x samples, transposed internally).
  - perplexity is automatically capped to floor((n_samples - 1) / 3).
  - For embedding input, --pca defaults to FALSE (embedding is already reduced);
    for expression input you may set --pca TRUE to let Rtsne reduce first.
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

LOOM_CHUNK_DIM   <- 64L
LOOM_GZIP_LEVEL  <- 2L
OUTPUT_JSON_NAME <- "output.json"
VALID_METHODS    <- c("Rtsne")

option_list <- list(
  make_option(c("-f", "--file"),       type = "character", default = NULL,   help = "Input LOOM file. [required]"),
  make_option(c("-o", "--output_dir"), type = "character", default = NULL,   help = "Output folder. [optional, default: stdout]"),
  make_option("--input_meta",          type = "character", default = NULL,   help = "LOOM path of input matrix (embedding or expression). [required]"),
  make_option("--output_meta",         type = "character", default = NULL,   help = "LOOM /col_attrs/ path for t-SNE embeddings. [required]"),
  make_option("--method",              type = "character", default = NULL,   help = "t-SNE method: Rtsne. [required]"),
  make_option("--filter_meta",         type = "character", default = NULL,   help = "LOOM /row_attrs/ filter mask (expression input only). [optional]"),
  make_option("--top_var_genes",       type = "integer",   default = 0L,     help = "Top variable genes (0 = all; expression input only). [default: 0]"),
  make_option("--n_dims",              type = "integer",   default = NULL,   help = "Leading dims to use from an embedding (NULL = all). [default: NULL]"),
  make_option("--n_components",        type = "integer",   default = 2L,     help = "[Rtsne] t-SNE dimensions. [default: 2]"),
  make_option("--perplexity",          type = "double",    default = 30.0,   help = "[Rtsne] Perplexity. [default: 30]"),
  make_option("--pca",                 type = "logical",   default = FALSE,  help = "[Rtsne] Internal PCA before t-SNE. [default: FALSE]"),
  make_option("--seed_use",            type = "integer",   default = 42L,    help = "[Rtsne] Random seed. [default: 42]")
)

parser <- OptionParser(option_list = option_list, add_help_option = FALSE)
args   <- parse_args(parser)

if (is.null(args$file))        ErrorJSON("Missing required argument -f.")
if (is.null(args$input_meta))  ErrorJSON("Missing required argument --input_meta.")
if (is.null(args$output_meta)) ErrorJSON("Missing required argument --output_meta.")
if (is.null(args$method))      ErrorJSON("Missing required argument --method.")
if (!args$method %in% VALID_METHODS) ErrorJSON(paste0("Unknown method '", args$method, "'. Valid: ", paste(VALID_METHODS, collapse = ", "), "."))
if (!startsWith(args$output_meta, "/col_attrs/")) ErrorJSON(paste0("--output_meta must be under /col_attrs/ (e.g. /col_attrs/TSNE), got: '", args$output_meta, "'."))
if (!is.null(args$filter_meta) && !startsWith(args$filter_meta, "/row_attrs/")) ErrorJSON(paste0("--filter_meta must be under /row_attrs/, got: '", args$filter_meta, "'."))

input_path <- normalizePath(args$file, mustWork = FALSE)
if (!file.exists(input_path)) ErrorJSON(paste0("Input LOOM file not found: ", args$file))

out_dir          <- if (!is.null(args$output_dir)) normalizePath(args$output_dir, mustWork = FALSE) else NULL
if (!is.null(out_dir)) dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
output_json_path <- if (!is.null(out_dir)) file.path(out_dir, OUTPUT_JSON_NAME) else NULL

## ── Phase 2: heavy imports ────────────────────────────────────────────────────

suppressPackageStartupMessages(library(hdf5r))
if (!requireNamespace("Rtsne", quietly = TRUE)) ErrorJSON("Rtsne not installed. Use: install.packages('Rtsne')")
suppressPackageStartupMessages(library(Rtsne))
rtsne_version <- tryCatch(as.character(packageVersion("Rtsne")), error = function(e) "unknown")

## ── Helpers ───────────────────────────────────────────────────────────────────

warn_env   <- new.env(parent = emptyenv()); warn_env$w <- character(0)
capture_warnings <- function(expr) withCallingHandlers(expr, warning = function(w) { warn_env$w <- c(warn_env$w, conditionMessage(w)); invokeRestart("muffleWarning") })
rowVars_fast <- function(m) { rowMeans(m^2) - rowMeans(m)^2 }

is_embedding_input <- startsWith(args$input_meta, "/col_attrs/")

## ── Read input matrix from LOOM ──────────────────────────────────────────────

h5_in <- tryCatch(H5File$new(input_path, mode = "r"), error = function(e) ErrorJSON(paste0("Could not open '", args$file, "' as HDF5/LOOM.")))
src_path <- sub("^/", "", args$input_meta)
if (!h5_in$exists(src_path)) ErrorJSON(paste0("Source path '", args$input_meta, "' not found in LOOM file."))

if (!h5_in$exists("matrix")) ErrorJSON("/matrix not found in LOOM; cannot determine sample count.")
mat_dims  <- h5_in[["matrix"]]$dims
n_samples <- as.integer(mat_dims[1])

if (is_embedding_input) {
  raw <- h5_in[[src_path]][,]
  if (is.null(dim(raw))) raw <- matrix(raw, ncol = 1L)
  if (nrow(raw) == n_samples)      data_obs <- raw
  else if (ncol(raw) == n_samples) data_obs <- t(raw)
  else ErrorJSON(paste0("Embedding dims (", nrow(raw), " x ", ncol(raw), ") do not match n_samples (", n_samples, ")."))
  n_feat_total <- ncol(data_obs)
  if (!is.null(args$n_dims)) {
    n_use <- min(args$n_dims, n_feat_total)
    if (args$n_dims > n_feat_total) warn_env$w <- c(warn_env$w, paste0("--n_dims (", args$n_dims, ") exceeds available dims (", n_feat_total, "); using all ", n_feat_total, "."))
    data_obs <- data_obs[, seq_len(n_use), drop = FALSE]
  }
} else {
  n_genes    <- as.integer(mat_dims[2])
  matrix_gxs <- t(h5_in[[src_path]][,])
  if (nrow(matrix_gxs) != n_genes || ncol(matrix_gxs) != n_samples)
    ErrorJSON(paste0("Expression matrix dims (", nrow(matrix_gxs), " x ", ncol(matrix_gxs), ") do not match expected (", n_genes, " x ", n_samples, ")."))

  filter_mask <- NULL
  if (!is.null(args$filter_meta)) {
    fmask_path <- sub("^/", "", args$filter_meta)
    if (!h5_in$exists(fmask_path)) ErrorJSON(paste0("Filter mask '", args$filter_meta, "' not found in LOOM."))
    filter_mask <- as.logical(h5_in[[fmask_path]][])
    if (length(filter_mask) != n_genes) ErrorJSON(paste0("Filter mask length (", length(filter_mask), ") != n_genes (", n_genes, ")."))
    matrix_gxs <- matrix_gxs[filter_mask, , drop = FALSE]
    warn_env$w <- c(warn_env$w, paste0("Filter mask applied: ", nrow(matrix_gxs), "/", n_genes, " genes retained."))
  }

  n_genes_now <- nrow(matrix_gxs)
  n_top <- if (args$top_var_genes > 0 && args$top_var_genes < n_genes_now) as.integer(args$top_var_genes) else n_genes_now
  if (n_top < n_genes_now) {
    gene_vars  <- rowVars_fast(matrix_gxs)
    top_idx    <- order(gene_vars, decreasing = TRUE)[seq_len(n_top)]
    matrix_gxs <- matrix_gxs[top_idx, , drop = FALSE]
    warn_env$w <- c(warn_env$w, paste0("Top ", n_top, " variable genes selected for t-SNE."))
  }

  data_obs <- t(matrix_gxs)   # samples x genes
}
h5_in$close_all()

if (nrow(data_obs) != n_samples) ErrorJSON(paste0("Internal error: observation matrix has ", nrow(data_obs), " rows, expected ", n_samples, " samples."))

## ── Cap perplexity (Rtsne requires 3 * perplexity < n_samples) ───────────────

eff_perplexity <- args$perplexity
max_perplexity <- floor((n_samples - 1) / 3)
if (max_perplexity < 1) ErrorJSON(paste0("Too few samples (", n_samples, ") to run t-SNE (need >= 4)."))
if (eff_perplexity > max_perplexity) {
  warn_env$w     <- c(warn_env$w, paste0("perplexity (", args$perplexity, ") capped to ", max_perplexity, " (must satisfy 3*perplexity < n_samples)."))
  eff_perplexity <- as.double(max_perplexity)
}

## ── Deduplicate rows (Rtsne errors on duplicates unless told otherwise) ──────

dup_rows <- duplicated(data_obs)
if (any(dup_rows)) warn_env$w <- c(warn_env$w, paste0(sum(dup_rows), " duplicate sample row(s) present; Rtsne run with check_duplicates = FALSE."))

## ── Run t-SNE via Rtsne ──────────────────────────────────────────────────────

set.seed(args$seed_use)
tsne_res <- capture_warnings(Rtsne::Rtsne(
  X                = as.matrix(data_obs),
  dims             = args$n_components,
  perplexity       = eff_perplexity,
  pca              = isTRUE(args$pca),
  check_duplicates = FALSE,
  verbose          = FALSE
))
tsne_embeddings <- tsne_res$Y                 # (n_samples x n_components)
rownames(tsne_embeddings) <- NULL
n_components_out <- ncol(tsne_embeddings)

## ── Append t-SNE embeddings into the LOOM (/col_attrs/) ──────────────────────

h5_loom <- tryCatch(H5File$new(input_path, mode = "a"), error = function(e) ErrorJSON(paste0("Could not open LOOM in append mode: ", conditionMessage(e))))
out_path_h5 <- sub("^/", "", args$output_meta)
if (!h5_loom$exists("col_attrs")) h5_loom$create_group("col_attrs")
if (h5_loom$exists(out_path_h5)) {
  warn_env$w <- c(warn_env$w, paste0("Path '", args$output_meta, "' already exists and will be overwritten."))
  parts      <- Filter(nzchar, strsplit(out_path_h5, "/")[[1]])
  parent_grp <- if (length(parts) > 1) h5_loom[[paste(parts[-length(parts)], collapse = "/")]] else h5_loom
  parent_grp$link_delete(parts[length(parts)])
}

# tsne_embeddings is (n_samples x n_components) in R.
# hdf5r reverses dims on write: write t(emb) (n_components x n_samples) → HDF5 (n_samples x n_components) ✓
dtype  <- h5types$H5T_IEEE_F64LE
space  <- H5S$new("simple", dims = c(n_components_out, n_samples), maxdims = c(n_components_out, n_samples))
sink(nullfile())
dcpl <- H5P_DATASET_CREATE$new()
dcpl$set_chunk(c(min(n_components_out, LOOM_CHUNK_DIM), min(n_samples, LOOM_CHUNK_DIM)))
dcpl$set_deflate(LOOM_GZIP_LEVEL)
sink()
ds <- h5_loom$create_dataset(name = out_path_h5, dtype = dtype, space = space, dataset_create_pl = dcpl)
ds[1:n_components_out, 1:n_samples] <- t(tsne_embeddings)
size_tsne <- ds$get_storage_size()
h5_loom$close_all()

## ── Result JSON ───────────────────────────────────────────────────────────────

result <- list(
  loom_path        = input_path,
  tool             = "Rtsne",
  tool_version     = rtsne_version,
  method           = args$method,
  input_loom_path  = args$input_meta,
  output_loom_path = args$output_meta,
  input_type       = if (is_embedding_input) "embedding" else "expression_matrix",
  parameters       = list(
    input_loom_path  = args$input_meta,
    output_loom_path = args$output_meta,
    input_type       = if (is_embedding_input) "embedding" else "expression_matrix",
    n_components     = n_components_out,
    perplexity       = eff_perplexity,
    pca              = isTRUE(args$pca),
    seed_use         = args$seed_use,
    n_dims           = if (is_embedding_input) (if (!is.null(args$n_dims)) args$n_dims else NULL) else NULL,
    top_var_genes    = if (!is_embedding_input) args$top_var_genes else NULL,
    filter_meta      = if (!is_embedding_input) args$filter_meta else NULL
  ),
  tsne = list(
    tool         = "Rtsne",
    tool_version = rtsne_version,
    n_components = n_components_out,
    perplexity   = eff_perplexity
  ),
  metadata = list(list(
    name         = args$output_meta,
    on           = "CELL",
    type         = "NUMERIC",
    nber_rows    = as.integer(n_samples),
    nber_cols    = as.integer(n_components_out),
    dataset_size = as.integer(size_tsne),
    imported     = 0L
  ))
)
if (length(warn_env$w) > 0) result$warnings <- as.list(warn_env$w)

json_str <- toJSON(result, auto_unbox = TRUE, null = "null", na = "null", pretty = FALSE, digits = 7)
if (!is.null(output_json_path)) writeLines(json_str, con = output_json_path) else cat(json_str, "\n")