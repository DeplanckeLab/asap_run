#!/usr/bin/env Rscript
# filter_bulk.R — Filter low-expressed genes from a bulk RNA-seq LOOM file.
#
# Appends a boolean filter mask directly into the input LOOM as a row attribute,
# and writes output.json.
#
# Required packages: hdf5r, jsonlite, optparse
# Optional:          edgeR (for cpm method)

## ── Phase 0: HELP (no library needed, runs instantly) ────────────────────────

HELP_TEXT <- "
Filtering Script — bulk RNA-seq

Reads a LOOM file, computes a gene-level filter mask, appends it as a row
attribute in the input LOOM, and writes output.json.
The mask is a logical vector (TRUE = gene kept, FALSE = gene removed).

Options:
  -f / --file        Input LOOM file.                                        [required]
  --input_meta       Path to the count matrix inside the LOOM file.          [required]
                       Examples: /matrix   /layers/counts
  --output_meta      Path where the boolean mask will be stored.             [required]
                       Must be under /row_attrs/.
                       Examples: /row_attrs/keep   /row_attrs/filter_pass
  --method           Filtering method:                                       [required]
                       min_counts   Keep genes where >= min_samples samples have
                                    >= min_counts raw counts.
                       cpm          Keep genes where >= min_samples samples have
                                    >= min_cpm CPM (requires edgeR).
  -o / --output_dir  Output folder for output.json. [optional, default: input dir]

  -- min_counts parameters (defaults match DESeq2 pre-filtering recommendation) -----
  --min_counts  Minimum raw count per sample.              [default: 10]
  --min_samples Minimum number of samples meeting the threshold.
                  Use 0 to apply the threshold to the sum across all samples.
                  [default: 1]

  -- cpm parameters ----------------------------------------------------------------
  --min_cpm     Minimum CPM per sample.                    [default: 1]
  --min_samples Minimum number of samples meeting the threshold. [default: 1]

  --help        Show this message and exit.

Output JSON fields:
  nber_genes_before   int  — total genes in input matrix
  nber_genes_kept     int  — genes passing the filter
  nber_genes_removed  int  — genes removed

Mandatory parameters: -f/--file, --input_meta, --output_meta, --method
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

LOOM_CHUNK_GENES <- 1024L
LOOM_CHUNK_CELLS <- 1024L
LOOM_GZIP_LEVEL  <- 4L
OUTPUT_JSON_NAME <- "output.json"

VALID_METHODS <- c("min_counts", "cpm")

option_list <- list(
  make_option(c("-f", "--file"),        type = "character", default = NULL,  help = "Input LOOM file. [required]"),
  make_option(c("-o", "--output_dir"),  type = "character", default = NULL,  help = "Output folder for output.json. [optional, default: input dir]"),
  make_option("--input_meta",           type = "character", default = NULL,  help = "LOOM-internal path of the count matrix (e.g. /matrix). [required]"),
  make_option("--output_meta",          type = "character", default = NULL,  help = "LOOM row_attr path for the boolean mask (e.g. /row_attrs/keep). [required]"),
  make_option("--method",               type = "character", default = NULL,  help = "Filtering method: min_counts | cpm. [required]"),
  make_option("--min_counts",           type = "integer",   default = 10L,   help = "[min_counts] Min raw count per sample. [default: 10]"),
  make_option("--min_cpm",              type = "double",    default = 1.0,   help = "[cpm] Min CPM per sample. [default: 1]"),
  make_option("--min_samples",          type = "integer",   default = 1L,    help = "Min samples meeting threshold. Use 0 for sum across all samples. [default: 1]")
)

parser <- OptionParser(option_list = option_list, add_help_option = FALSE)
args   <- parse_args(parser)

if (is.null(args$file))        ErrorJSON("Missing required argument -f (input LOOM file).")
if (is.null(args$input_meta))  ErrorJSON("Missing required argument --input_meta.")
if (is.null(args$output_meta)) ErrorJSON("Missing required argument --output_meta.")
if (is.null(args$method))      ErrorJSON("Missing required argument --method.")
if (!args$method %in% VALID_METHODS) ErrorJSON(paste0("Unknown method '", args$method, "'. Valid: ", paste(VALID_METHODS, collapse = ", "), "."))
if (!startsWith(args$output_meta, "/row_attrs/")) ErrorJSON(paste0("--output_meta must be under /row_attrs/ (e.g. /row_attrs/keep), got: '", args$output_meta, "'."))
if (args$method == "min_counts" && args$min_counts < 0)  ErrorJSON(paste0("--min_counts must be >= 0, got: ", args$min_counts))
if (args$method == "cpm"        && args$min_cpm    < 0)  ErrorJSON(paste0("--min_cpm must be >= 0, got: ", args$min_cpm))
if (args$min_samples < 0) ErrorJSON(paste0("--min_samples must be >= 0, got: ", args$min_samples))

input_path <- normalizePath(args$file, mustWork = FALSE)
if (!file.exists(input_path)) ErrorJSON(paste0("Input LOOM file not found: ", args$file))

out_dir          <- if (!is.null(args$output_dir)) normalizePath(args$output_dir, mustWork = FALSE) else dirname(input_path)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
output_json_path <- file.path(out_dir, OUTPUT_JSON_NAME)

## ── Phase 2: heavy imports ────────────────────────────────────────────────────

suppressPackageStartupMessages(library(hdf5r))
if (args$method == "cpm") {
  if (!requireNamespace("edgeR", quietly = TRUE)) ErrorJSON("edgeR is not installed (required for cpm method). Install with: BiocManager::install('edgeR')")
  suppressPackageStartupMessages(library(edgeR))
}

## ── LOOM helpers ─────────────────────────────────────────────────────────────

write_row_attr <- function(h5, path, vec) {
  # Write a 1D vector to a LOOM row_attr path (strip leading /).
  parts <- Filter(nzchar, strsplit(path, "/")[[1]])
  if (!h5$exists("row_attrs")) h5$create_group("row_attrs")
  if (h5$exists(path)) {
    parent_grp <- if (length(parts) > 1) h5[[paste(parts[-length(parts)], collapse = "/")]] else h5
    parent_grp$link_delete(parts[length(parts)])
  }
  h5[[path]] <- vec
}

warn_env <- new.env(parent = emptyenv())
warn_env$w <- character(0)

## ── Read the LOOM file ────────────────────────────────────────────────────────

h5_in <- tryCatch(
  H5File$new(input_path, mode = "r"),
  error = function(e) ErrorJSON(paste0("Could not open '", args$file, "' as an HDF5/LOOM file."))
)

src_path <- sub("^/", "", args$input_meta)
if (!h5_in$exists(src_path)) ErrorJSON(paste0("Source path '", args$input_meta, "' not found in LOOM file."))

h5_dims   <- h5_in[[src_path]]$dims      # hdf5r reverses HDF5 dims: [1]=n_samples, [2]=n_genes
n_samples <- as.integer(h5_dims[1])
n_genes   <- as.integer(h5_dims[2])
matrix_gxc <- t(h5_in[[src_path]][,])   # [,] reads (n_samples × n_genes); t() gives (n_genes × n_samples)

if (!h5_in$exists("row_attrs/_StableID")) ErrorJSON("row_attrs/_StableID not found in LOOM file. Was the file created with parse_v8.py?")
gene_ids <- as.character(h5_in[["row_attrs/_StableID"]][])
h5_in$close_all()

## ── Apply filter ─────────────────────────────────────────────────────────────
# matrix_gxc is (n_genes × n_samples); rowSums gives one value per gene

if (args$method == "min_counts") {
  if (args$min_samples == 0L) {
    keep <- rowSums(matrix_gxc) >= args$min_counts
  } else {
    keep <- rowSums(matrix_gxc >= args$min_counts) >= args$min_samples
  }
} else if (args$method == "cpm") {
  cpm_mat <- cpm(matrix_gxc)   # edgeR cpm() expects genes × samples ✓
  if (args$min_samples == 0L) {
    keep <- rowSums(cpm_mat) >= args$min_cpm
  } else {
    keep <- rowSums(cpm_mat >= args$min_cpm) >= args$min_samples
  }
}

nber_kept    <- as.integer(sum(keep))
nber_removed <- as.integer(n_genes - nber_kept)

## ── Write mask to LOOM ───────────────────────────────────────────────────────

h5_loom <- tryCatch(
  H5File$new(input_path, mode = "a"),
  error = function(e) ErrorJSON(paste0("Could not open '", args$file, "' in append mode: ", conditionMessage(e)))
)
out_path_h5 <- sub("^/", "", args$output_meta)
if (h5_loom$exists(out_path_h5)) {
  warn_env$w <- c(warn_env$w, paste0("Path '", args$output_meta, "' already exists and will be overwritten."))
  parts      <- Filter(nzchar, strsplit(out_path_h5, "/")[[1]])
  parent_grp <- if (length(parts) > 1) h5_loom[[paste(parts[-length(parts)], collapse = "/")]] else h5_loom
  parent_grp$link_delete(parts[length(parts)])
}
if (!h5_loom$exists("row_attrs")) h5_loom$create_group("row_attrs")
h5_loom[[out_path_h5]] <- as.integer(keep)   # 1 = kept, 0 = removed (hdf5r handles logical → integer cleanly)
h5_loom$close_all()

## ── Result and output.json ────────────────────────────────────────────────────

result <- list(
  loom_path        = input_path,
  tool             = if (args$method == "cpm") "edgeR" else "base R",
  tool_version     = if (args$method == "cpm") tryCatch(as.character(packageVersion("edgeR")), error = function(e) "unknown") else as.character(getRversion()),
  method           = args$method,
  nber_rows        = n_genes,     # parse_v8.py convention: nber_rows = n_genes
  nber_cols        = n_samples,   # nber_cols = n_samples
  input_loom_path  = args$input_meta,
  output_loom_path = args$output_meta,
  parameters       = list(
    input_loom_path = args$input_meta,
    output_loom_path = args$output_meta,
    min_counts      = if (args$method == "min_counts") args$min_counts else NULL,
    min_cpm         = if (args$method == "cpm") args$min_cpm else NULL,
    min_samples     = args$min_samples
  ),
  nber_genes_before  = n_genes,
  nber_genes_kept    = nber_kept,
  nber_genes_removed = nber_removed,
  pct_genes_kept     = round(100 * nber_kept / n_genes, 2)
)
if (length(warn_env$w) > 0) result$warnings <- as.list(warn_env$w)
json_str <- toJSON(result, auto_unbox = TRUE, null = "null", na = "null", pretty = FALSE, digits = 7)
if (!is.null(args$output_dir)) writeLines(json_str, con = output_json_path) else cat(json_str, "\n")