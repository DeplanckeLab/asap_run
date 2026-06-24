#!/usr/bin/env Rscript
# cluster_bulk.R — Cluster samples from a bulk RNA-seq normalized LOOM file.
#
# Computes pairwise sample distances / k-means and appends cluster labels as a
# col attribute in the input LOOM, then writes output.json.
#
# Required packages: hdf5r, jsonlite, optparse

## ── Phase 0: HELP (no library needed, runs instantly) ────────────────────────

HELP_TEXT <- "
Sample Clustering — bulk RNA-seq

Reads a normalized matrix from a LOOM file, clusters the samples, and appends
integer cluster labels as a col attribute in the LOOM.

Options:
  -f / --file        Input LOOM file.                                        [required]
  --input_meta       LOOM-internal path of the normalized matrix.            [required]
                       Examples: /layers/normalized_vst   /layers/normalized_edgeR
  --output_meta      LOOM col_attr path for cluster labels (1 integer per sample).
                       Must be under /col_attrs/.                            [required]
                       Example: /col_attrs/cluster_vst
  --method           Clustering method: hierarchical | kmeans                [required]
  -o / --output_dir  Output folder for output.json. [optional, default: input dir]

  -- Common parameters -------------------------------------------------------------
  -k                Number of clusters (required for both methods).          [required]
  --top_var_genes   Number of most variable genes used (0 = all genes).      [default: 500]
  --filter_meta     LOOM row_attr path of filter mask from filter_bulk.R.
                      Filtered-out genes are excluded before clustering.      [optional]
                      Example: /row_attrs/filter_pass

  -- hierarchical parameters (defaults match R hclust / dist) ----------------------
  --distance        Distance metric: euclidean | pearson | spearman.         [default: euclidean]
  --linkage         Linkage method: ward.D2 | complete | average | single |
                      mcquitty | median | centroid.                          [default: ward.D2]

  -- kmeans parameters (defaults match R kmeans) -----------------------------------
  --nstart          Number of random starts.                                 [default: 25]
  --iter_max        Maximum number of iterations.                            [default: 300]
  --seed            Random seed.                                             [default: 42]

  --help        Show this message and exit.

Output LOOM col attr:
  <output_meta>   Integer cluster label (1-based) per sample

Mandatory parameters: -f/--file, --input_meta, --output_meta, --method, -k
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

OUTPUT_JSON_NAME <- "output.json"

option_list <- list(
  make_option(c("-f", "--file"),       type = "character", default = NULL,     help = "Input LOOM file. [required]"),
  make_option(c("-o", "--output_dir"), type = "character", default = NULL,     help = "Output folder for output.json. [optional, default: input dir]"),
  make_option("--input_meta",          type = "character", default = NULL,     help = "LOOM-internal path of the normalized matrix. [required]"),
  make_option("--output_meta",         type = "character", default = NULL,     help = "LOOM col_attr path for cluster labels (e.g. /col_attrs/cluster_vst). [required]"),
  make_option("--method",              type = "character", default = NULL,     help = "Clustering method: hierarchical | kmeans. [required]"),
  make_option(c("-k", "--k"),          type = "integer",   default = NULL,     help = "Number of clusters. [required]"),
  make_option("--top_var_genes",       type = "integer",   default = 500L,     help = "Top variable genes (0 = all). [default: 500]"),
  make_option("--filter_meta",         type = "character", default = NULL,     help = "LOOM row_attr path of filter mask from filter_bulk.R (e.g. /row_attrs/filter_pass). [optional]"),
  # hierarchical
  make_option("--distance",            type = "character", default = "euclidean", help = "[hierarchical] Distance: euclidean | pearson | spearman. [default: euclidean]"),
  make_option("--linkage",             type = "character", default = "ward.D2",   help = "[hierarchical] Linkage: ward.D2 | complete | average | single | mcquitty | median | centroid. [default: ward.D2]"),
  # kmeans
  make_option("--nstart",              type = "integer",   default = 25L,      help = "[kmeans] Number of random starts. [default: 25]"),
  make_option("--iter_max",            type = "integer",   default = 300L,     help = "[kmeans] Max iterations. [default: 300]"),
  make_option("--seed",                type = "integer",   default = 42L,      help = "[kmeans] Random seed. [default: 42]")
)

parser <- OptionParser(option_list = option_list, add_help_option = FALSE)
args   <- parse_args(parser)

if (is.null(args$file))        ErrorJSON("Missing required argument -f.")
if (is.null(args$input_meta))  ErrorJSON("Missing required argument --input_meta.")
if (is.null(args$output_meta)) ErrorJSON("Missing required argument --output_meta.")
if (is.null(args$method))      ErrorJSON("Missing required argument --method.")
if (is.null(args$k))           ErrorJSON("Missing required argument -k.")
if (!args$method %in% c("hierarchical", "kmeans")) ErrorJSON(paste0("Unknown method '", args$method, "'. Valid: hierarchical, kmeans."))
if (!startsWith(args$output_meta, "/col_attrs/")) ErrorJSON(paste0("--output_meta must be under /col_attrs/ (e.g. /col_attrs/cluster_vst), got: '", args$output_meta, "'."))
if (args$k < 2L) ErrorJSON(paste0("-k must be >= 2, got: ", args$k))

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
matrix_gxc <- t(h5_in[[src_path]][,])   # [,] reads (n_samples × n_genes); t() → (n_genes × n_samples)

# Read optional filter mask
filter_mask <- NULL
if (!is.null(args$filter_meta)) {
  fmask_path <- sub("^/", "", args$filter_meta)
  if (!h5_in$exists(fmask_path)) ErrorJSON(paste0("Filter mask '", args$filter_meta, "' not found in LOOM."))
  filter_mask <- as.logical(h5_in[[fmask_path]][])
  if (length(filter_mask) != n_genes) ErrorJSON(paste0("Filter mask length (", length(filter_mask), ") does not match number of genes (", n_genes, ")."))
}
h5_in$close_all()

if (args$k >= n_samples) ErrorJSON(paste0("-k (", args$k, ") must be less than the number of samples (", n_samples, ")."))

# Apply filter mask before any gene selection
if (!is.null(filter_mask)) {
  n_before   <- n_genes
  matrix_gxc <- matrix_gxc[filter_mask, ]
  n_genes    <- nrow(matrix_gxc)
  warn_env$w <- c(warn_env$w, paste0("Filter mask applied: ", n_genes, "/", n_before, " genes used for clustering."))
}
## ── Select top variable genes ─────────────────────────────────────────────────

rowVars_fast <- function(m) { rowMeans(m^2) - rowMeans(m)^2 }

n_top <- if (args$top_var_genes > 0 && args$top_var_genes < n_genes) as.integer(args$top_var_genes) else n_genes
if (n_top < n_genes) {
  top_idx    <- order(rowVars_fast(matrix_gxc), decreasing = TRUE)[seq_len(n_top)]
  matrix_sub <- matrix_gxc[top_idx, ]
} else {
  matrix_sub <- matrix_gxc
}

## ── Cluster samples ───────────────────────────────────────────────────────────

if (args$method == "hierarchical") {

  valid_distances <- c("euclidean", "pearson", "spearman")
  valid_linkages  <- c("ward.D2", "complete", "average", "single", "mcquitty", "median", "centroid")
  if (!args$distance %in% valid_distances) ErrorJSON(paste0("Unknown distance '", args$distance, "'. Valid: ", paste(valid_distances, collapse = ", ")))
  if (!args$linkage  %in% valid_linkages)  ErrorJSON(paste0("Unknown linkage '",  args$linkage,  "'. Valid: ", paste(valid_linkages,  collapse = ", ")))

  # Compute distance matrix between samples
  dist_mat <- if (args$distance == "euclidean") {
    dist(t(matrix_sub))   # dist() works on rows; t() puts samples as rows
  } else {
    as.dist(1 - cor(matrix_sub, method = args$distance))   # correlation distance; cor() works on columns = samples
  }

  hc             <- hclust(dist_mat, method = args$linkage)
  cluster_labels <- as.integer(cutree(hc, k = args$k))

  clust_params <- list(distance = args$distance, linkage = args$linkage)

} else if (args$method == "kmeans") {

  set.seed(args$seed)
  km <- tryCatch(
    kmeans(t(matrix_sub), centers = args$k, nstart = args$nstart, iter.max = args$iter_max),
    error = function(e) ErrorJSON(paste0("kmeans failed: ", conditionMessage(e)))
  )
  cluster_labels <- as.integer(km$cluster)
  if (km$ifault != 0) warn_env$w <- c(warn_env$w, paste0("kmeans convergence issue (ifault=", km$ifault, "); consider increasing --iter_max."))

  clust_params <- list(nstart = args$nstart, iter_max = args$iter_max, seed = args$seed, tot_withinss = km$tot.withinss, betweenss = km$betweenss)
}

## ── Append cluster labels to LOOM as a col attr (1D integer vector) ──────────

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
h5_loom[[out_path_h5]] <- cluster_labels
h5_loom$close_all()

## ── Cluster size summary ─────────────────────────────────────────────────────

cluster_sizes <- as.list(table(cluster_labels))
names(cluster_sizes) <- paste0("cluster_", names(cluster_sizes))

## ── Emit output.json ─────────────────────────────────────────────────────────

result <- list(
  loom_path        = input_path,
  tool             = "base R",
  tool_version     = as.character(getRversion()),
  method           = args$method,
  nber_rows        = n_genes,
  nber_cols        = n_samples,
  input_loom_path  = args$input_meta,
  output_loom_path = args$output_meta,
  parameters       = c(list(input_loom_path = args$input_meta, output_loom_path = args$output_meta, k = args$k, top_var_genes = n_top, filter_meta = args$filter_meta), clust_params),
  cluster_sizes    = cluster_sizes
)
if (length(warn_env$w) > 0) result$warnings <- as.list(warn_env$w)

json_str <- toJSON(result, auto_unbox = TRUE, null = "null", na = "null", pretty = FALSE, digits = 7)
if (!is.null(args$output_dir)) writeLines(json_str, con = output_json_path) else cat(json_str, "\n")