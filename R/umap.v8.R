#!/usr/bin/env Rscript
# umap.R — UMAP on scRNA-seq data from a LOOM file using Seurat v5.
#
# Reads PCA cell embeddings from a LOOM file, builds a kNN graph, runs UMAP,
# appends the UMAP cell embeddings into the input LOOM, and writes output.json.
#
# Required packages: hdf5r, Seurat (>= 5.0), jsonlite, Matrix, optparse, ggplot2, plotly

## ── Phase 0: HELP (no library needed, runs instantly) ────────────────────────

HELP_TEXT <- "
UMAP Script — Seurat v5 backend

Reads PCA cell embeddings from a LOOM file, runs Seurat v5 FindNeighbors +
RunUMAP, appends UMAP embeddings directly into the input LOOM, and writes output.json.

Options:
  -f / --file        Input LOOM file.                                       [required]
  --input_meta       Path to the PCA cell embeddings inside the LOOM.       [required]
                       Example: /col_attrs/pca_R_scaled_ln
  --output_meta      Path where UMAP embeddings (cells x n_components) are  [required]
                     stored in the LOOM file.
                       Example: /col_attrs/umap_R_scaled_ln
  --method           UMAP method: RunUMAP                                   [required]
  -o / --output_dir  Output folder for output.json and plots.
                     [optional, default: stdout for JSON, same dir as LOOM for plots]
  --assay            Seurat assay name.              [default: RNA]

  -- FindNeighbors parameters ─────────────────────────────────────────────────────
  --n_dims           Number of PCA dimensions to use.   [default: all available]
  --n_neighbors      Number of neighbors (k) for kNN.   [default: 30]
  --metric           Distance metric for kNN.            [default: cosine]

  -- RunUMAP parameters (defaults match Seurat v5 RunUMAP) ────────────────────────
  --n_components     Number of UMAP dimensions.          [default: 2]
  --min_dist         Minimum distance between points.    [default: 0.3]
  --seed_use         Random seed.                        [default: 42]

  --help             Show this message and exit.

Output JSON umap block:
  tool          str  — always 'Seurat'
  tool_version  str  — installed Seurat version
  n_components  int  — number of UMAP dimensions
  n_dims        int  — number of PCA dimensions used
  n_neighbors   int  — k used for kNN graph
  min_dist      num  — min_dist parameter
  metric        str  — distance metric used

Output JSON metadata entry:
  name          str  — LOOM-internal path of UMAP embeddings (= --output_meta)
  on            str  — always 'CELL'
  type          str  — always 'NUMERIC'
  nber_rows     int  — number of cells
  nber_cols     int  — number of UMAP dimensions
  dataset_size  int  — on-disk compressed size in bytes
  imported      int  — always 0

Mandatory parameters: -f/--file, --input_meta, --output_meta, --method
"

argv <- commandArgs(trailingOnly = TRUE)
if (length(argv) == 0 || "--help" %in% argv || "-h" %in% argv) { cat(HELP_TEXT); quit(save = "no", status = 0) }

## ── Phase 1: minimal imports ──────────────────────────────────────────────────

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
  make_option(c("-f", "--file"),       type = "character", default = NULL,     help = "Input LOOM file. [required]"),
  make_option(c("-o", "--output_dir"), type = "character", default = NULL,     help = "Output folder. [optional, default: stdout / LOOM dir]"),
  make_option("--input_meta",          type = "character", default = NULL,     help = "Path to PCA cell embeddings (e.g. /col_attrs/pca_R_scaled_ln). [required]"),
  make_option("--output_meta",         type = "character", default = NULL,     help = "Path for UMAP embeddings (e.g. /col_attrs/umap_R_scaled_ln). [required]"),
  make_option("--method",              type = "character", default = NULL,     help = "UMAP method: RunUMAP. [required]"),
  make_option("--assay",               type = "character", default = "RNA",    help = "Seurat assay name. [default: RNA]"),
  # FindNeighbors
  make_option("--n_dims",              type = "integer",   default = NULL,     help = "[FindNeighbors] PCA dims to use (NULL = all). [default: NULL]"),
  make_option("--n_neighbors",         type = "integer",   default = 30L,      help = "[FindNeighbors] k for kNN. [default: 30]"),
  make_option("--metric",              type = "character", default = "cosine", help = "[FindNeighbors] Distance metric. [default: cosine]"),
  # RunUMAP
  make_option("--n_components",        type = "integer",   default = 2L,       help = "[RunUMAP] UMAP dimensions. [default: 2]"),
  make_option("--min_dist",            type = "double",    default = 0.3,      help = "[RunUMAP] Min distance. [default: 0.3]"),
  make_option("--seed_use",            type = "integer",   default = 42L,      help = "[RunUMAP] Random seed. [default: 42]")
)

parser <- OptionParser(option_list = option_list, add_help_option = FALSE)
args   <- parse_args(parser)

if (is.null(args$file))        ErrorJSON("Missing required argument -f.")
if (is.null(args$input_meta))  ErrorJSON("Missing required argument --input_meta.")
if (is.null(args$output_meta)) ErrorJSON("Missing required argument --output_meta.")
if (is.null(args$method))      ErrorJSON("Missing required argument --method.")

valid_methods <- c("RunUMAP")
if (!args$method %in% valid_methods) ErrorJSON(paste0("Unknown method '", args$method, "'. Valid: ", paste(valid_methods, collapse = ", "), "."))
if (!startsWith(args$output_meta, "/")) ErrorJSON(paste0("--output_meta must start with / (e.g. /col_attrs/umap), got: '", args$output_meta, "'."))

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
  # Write a (dim1 x dim2) matrix with the same transpose trick as normalize/scaling.
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
    msg <- conditionMessage(w)
    # Suppress the one-time uwot/reticulate informational message from RunUMAP
    if (!grepl("umap.method|umap-learn|reticulate|UWOT", msg, ignore.case = TRUE)) {
      warn_env$w <- c(warn_env$w, msg)
    }
    invokeRestart("muffleWarning")
  })
}

## ── Result skeleton ───────────────────────────────────────────────────────────

result <- list(
  parameters = list(),
  umap       = list(tool = "Seurat", tool_version = seurat_version,
                    n_components = args$n_components, n_dims = -1L,
                    n_neighbors = args$n_neighbors, min_dist = args$min_dist,
                    metric = args$metric),
  metadata   = list(),
  wait_time  = 0L
)

## ── Read PCA cell embeddings from LOOM ───────────────────────────────────────

ret   <- open_loom_with_retry(input_path, mode = "r")
h5_in <- ret$h5
result$wait_time <- result$wait_time + ret$wait_time

src_path <- sub("^/", "", args$input_meta)
if (!h5_in$exists(src_path)) ErrorJSON(paste0("Source path '", args$input_meta, "' not found in LOOM file."))

raw_emb  <- h5_in[[src_path]][,]
ca_len   <- h5_in[[CELL_ID_PATH]]$dims
cell_ids <- as.character(h5_in[[CELL_ID_PATH]][])
h5_in$close_all()

# Orientation: input is cells x n_pcs.
# R-written: on-disk shape = (n_pcs, n_cells), hdf5r reads as (n_cells, n_pcs) — correct.
# Python-written: on-disk shape = (n_cells, n_pcs), hdf5r reads as (n_pcs, n_cells) — transpose needed.
if (nrow(raw_emb) == ca_len) {
  emb_cxp <- raw_emb          # R-written: already cells x pcs
} else {
  emb_cxp <- t(raw_emb)       # Python-written: transpose to cells x pcs
}
n_cells <- nrow(emb_cxp)
n_pcs   <- ncol(emb_cxp)
rownames(emb_cxp) <- cell_ids
colnames(emb_cxp) <- paste0("PC_", seq_len(n_pcs))

# Resolve n_dims
n_dims <- if (!is.null(args$n_dims)) min(args$n_dims, n_pcs) else n_pcs
if (!is.null(args$n_dims) && args$n_dims > n_pcs) {
  warn_env$w <- c(warn_env$w, paste0("--n_dims (", args$n_dims, ") exceeds available PCs (", n_pcs, "); using all ", n_pcs, "."))
}

result$parameters <- list(
  loom_path        = input_path,
  input_loom_path  = args$input_meta,
  output_loom_path = args$output_meta,
  method           = args$method,
  n_dims           = n_dims,
  n_neighbors      = args$n_neighbors,
  metric           = args$metric,
  n_components     = args$n_components,
  min_dist         = args$min_dist,
  seed_use         = args$seed_use
)
result$umap$n_dims <- n_dims

## ── Create Seurat object with PCA reduction and run UMAP ─────────────────────

# Use a zero sparse 1×n_cells matrix just to satisfy CreateSeuratObject.
fake_counts <- Matrix::Matrix(0, nrow = 1L, ncol = n_cells, sparse = TRUE,
                               dimnames = list("dummy_gene", cell_ids))
seurat_obj  <- suppressWarnings(CreateSeuratObject(counts = fake_counts, assay = args$assay, project = "umap_r"))

# Inject PCA embeddings as a DimReduc object so FindNeighbors/RunUMAP can use them.
seurat_obj[["pca"]] <- CreateDimReducObject(
  embeddings = emb_cxp,
  key        = "PC_",
  assay      = args$assay
)

seurat_obj <- capture_warnings(suppressMessages(FindNeighbors(
  seurat_obj,
  reduction = "pca",
  dims      = seq_len(n_dims),
  k.param   = args$n_neighbors,
  annoy.metric = args$metric,
  verbose   = FALSE
)))

seurat_obj <- capture_warnings(suppressMessages(RunUMAP(
  seurat_obj,
  reduction     = "pca",
  dims          = seq_len(n_dims),
  umap.method   = "uwot",
  n.components  = args$n_components,
  min.dist      = args$min_dist,
  seed.use      = args$seed_use,
  verbose       = FALSE
)))

umap_embeddings <- Embeddings(seurat_obj, reduction = "umap")   # (n_cells x n_components)

## ── Append UMAP embeddings into the LOOM ─────────────────────────────────────

ret     <- open_loom_with_retry(input_path, mode = "a")
h5_loom <- ret$h5
result$wait_time <- result$wait_time + ret$wait_time

out_path_h5 <- sub("^/", "", args$output_meta)
if (h5_loom$exists(out_path_h5)) {
  warn_env$w <- c(warn_env$w, paste0("Path '", args$output_meta, "' already exists and will be overwritten."))
}
size_umap <- write_2d_dataset(h5_loom, out_path_h5, umap_embeddings, n_cells, args$n_components)
h5_loom$close_all()

## ── Metadata + JSON ──────────────────────────────────────────────────────────

result$metadata <- list(list(
  name         = args$output_meta,
  on           = "CELL",
  type         = "NUMERIC",
  nber_rows    = as.integer(n_cells),
  nber_cols    = as.integer(args$n_components),
  dataset_size = as.integer(size_umap),
  imported     = 0L
))

if (length(warn_env$w) > 0) result$warnings <- as.list(warn_env$w)

json_str <- toJSON(result, auto_unbox = TRUE, null = "null", na = "null", pretty = FALSE, digits = 7)
if (!is.null(output_json_path)) writeLines(json_str, con = output_json_path) else cat(json_str, "\n")