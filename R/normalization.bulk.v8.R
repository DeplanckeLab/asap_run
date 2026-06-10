#!/usr/bin/env Rscript
# normalize_bulk.R — Normalize bulk RNA-seq data from a LOOM file.
#
# Appends the normalized layer directly into the input LOOM file and writes output.json.
# Methods: DESeq2_VST, DESeq2_rlog, DESeq2_normcounts, edgeR_TMM, limma_voom
#
# Required packages: hdf5r, jsonlite, optparse
#   + DESeq2  (Bioconductor) for DESeq2_* methods
#   + edgeR   (Bioconductor) for edgeR_TMM
#   + limma   (Bioconductor) for limma_voom (also requires edgeR)

## ── Phase 0: HELP (no library needed, runs instantly) ────────────────────────

HELP_TEXT <- "
Normalization Script — bulk RNA-seq backend

Reads a LOOM file, applies the selected bulk RNA-seq normalization method,
appends the normalized layer directly into the input LOOM file, and writes output.json.
The LOOM convention is: rows = genes, columns = samples.

Options:
  -f / --file        Input LOOM file.                                        [required]
  --input_meta       Path to the count matrix inside the LOOM file.          [required]
                       Examples: /matrix   /layers/counts
  --output_meta      Path where the normalized matrix will be appended.       [required]
                       Must be under /layers/.
                       Examples: /layers/normalized   /layers/vst_blind
  --method           Normalization method:                                    [required]
                       DESeq2_VST       Variance Stabilizing Transformation (log2)
                       DESeq2_rlog      Regularized Log Transformation (log2)
                       DESeq2_normcounts  Size-factor normalized counts + log2(1+x)
                       edgeR_TMM        TMM normalization + log2 CPM
                       limma_voom       voom log2-CPM transformation
  -o / --output_dir  Output folder for output.json.  [optional, default: input dir]

  -- DESeq2 parameters (all three DESeq2_* methods) --------------------------------
  --blind       Blind dispersion estimation to sample condition.  [default: TRUE]
  --fit_type    Dispersion fit: parametric | local | mean | glmGamPoi.
                                                                  [default: parametric]
  --sf_type     Size factor estimation: ratio | poscounts | iterate.
                  'poscounts' is recommended when many zeros are present.
                                                                  [default: ratio]

  -- edgeR_TMM parameters (defaults match edgeR package defaults) -------------------------
  --ref_column  Reference sample column index for TMM (NULL = auto). [default: NULL]
  --prior_count Prior count added to each observation before log2 CPM (scaled to
                  library size, avoids log(0); this is edgeR's default behaviour).
                  [default: 2]

  -- limma_voom parameters ---------------------------------------------------------
  --normalize_method  Between-sample normalisation applied inside voom:
                        none | scale | quantile | cyclicloess.   [default: none]
  --voom_span         Smoothing span for the mean-variance trend. [default: 0.5]

  --help        Show this message and exit.

Output JSON normalization block:
  is_log_transformed  bool   — whether the output is on a log scale
  log_type            str    — e.g. 'log2 — VST' or 'log2 — voom log2 CPM'
  log_base            float  — 2.0 for all log methods, null otherwise

Mandatory parameters: -f/--file, --input_meta, --output_meta, --method
"

argv <- commandArgs(trailingOnly = TRUE)
if (length(argv) == 0 || "--help" %in% argv || "-h" %in% argv) { cat(HELP_TEXT); quit(save = "no", status = 0) }

## ── Phase 1: minimal imports for argument parsing and error handling ──────────

suppressPackageStartupMessages({ library(optparse); library(jsonlite) })

ErrorJSON <- function(message, output_path = NULL) {
  payload  <- list(displayed_error = message)
  json_str <- toJSON(payload, auto_unbox = TRUE)
  if (!is.null(output_path)) writeLines(json_str, con = output_path) else cat(json_str, "\n")
  quit(save = "no", status = 1)
}

## Constants (matching normalize.R conventions)
LOOM_CHUNK_GENES <- 1024L
LOOM_CHUNK_CELLS <- 1024L
LOOM_GZIP_LEVEL  <- 4L
OUTPUT_JSON_NAME <- "output.json"

SAMPLE_ID_CANDIDATES <- c("col_attrs/SampleID", "col_attrs/sample_id", "col_attrs/Sample", "col_attrs/sample", "col_attrs/CellID", "col_attrs/cell_id", "col_attrs/barcode", "col_attrs/obs_names")

VALID_METHODS <- c("DESeq2_VST", "DESeq2_rlog", "DESeq2_normcounts", "edgeR_TMM", "limma_voom")

## CLI option list
option_list <- list(
  # Core I/O — mandatory (validated below)
  make_option(c("-f", "--file"),            type = "character", default = NULL,          help = "Input LOOM file. [required]"),
  make_option(c("-o", "--output_dir"),      type = "character", default = NULL,          help = "Output folder for output.json. [optional, default: input dir]"),
  make_option("--input_meta",               type = "character", default = NULL,          help = "LOOM-internal path of the count matrix (e.g. /matrix). [required]"),
  make_option("--output_meta",              type = "character", default = NULL,          help = "LOOM-internal path for the normalized output (under /layers/). [required]"),
  make_option("--method",                   type = "character", default = NULL,          help = paste0("Normalization method: ", paste(VALID_METHODS, collapse = " | "), ". [required]")),

  # DESeq2 parameters (defaults match DESeq2 package defaults)
  make_option("--blind",                    type = "logical",   default = TRUE,          help = "[DESeq2] Blind dispersion estimation to condition. [default: TRUE]"),
  make_option("--fit_type",                 type = "character", default = "parametric",  help = "[DESeq2] Dispersion fit: parametric | local | mean | glmGamPoi. [default: parametric]"),
  make_option("--sf_type",                  type = "character", default = "ratio",       help = "[DESeq2] Size factor type: ratio | poscounts | iterate. [default: ratio]"),
  make_option("--nsub",                     type = "integer",   default = 1000L,         help = "[DESeq2_VST] Number of genes for dispersion estimation (only available in recent DESeq2 versions). [default: 1000]"),

  # edgeR parameters (defaults match edgeR package defaults)
  make_option("--ref_column",               type = "integer",   default = NULL,          help = "[edgeR_TMM] Reference sample column index (NULL = auto). [default: NULL]"),
  make_option("--prior_count",              type = "double",    default = 2.0,           help = "[edgeR_TMM] Prior count added before log2 CPM (edgeR default, not a flat +1). [default: 2]"),

  # limma/voom parameters (defaults match limma package defaults)
  make_option("--normalize_method",         type = "character", default = "none",        help = "[limma_voom] Between-sample normalisation: none | scale | quantile | cyclicloess. [default: none]"),
  make_option("--voom_span",                type = "double",    default = 0.5,           help = "[limma_voom] Smoothing span for mean-variance trend. [default: 0.5]")
)

parser <- OptionParser(option_list = option_list, add_help_option = FALSE, description = "Normalize bulk RNA-seq data in a LOOM file.")
args   <- parse_args(parser)

## Validate mandatory arguments
if (is.null(args$file))        ErrorJSON("Missing required argument -f (input LOOM file).")
if (is.null(args$input_meta))  ErrorJSON("Missing required argument --input_meta (LOOM-internal path, e.g. /matrix).")
if (is.null(args$output_meta)) ErrorJSON("Missing required argument --output_meta (LOOM-internal output path, e.g. /layers/normalized).")
if (is.null(args$method))      ErrorJSON("Missing required argument --method.")
if (!args$method %in% VALID_METHODS) ErrorJSON(paste0("Unknown method '", args$method, "'. Valid: ", paste(VALID_METHODS, collapse = ", "), "."))
if (!startsWith(args$output_meta, "/layers/")) ErrorJSON(paste0("--output_meta must be under /layers/ (e.g. /layers/normalized), got: '", args$output_meta, "'."))

input_path <- normalizePath(args$file, mustWork = FALSE)
if (!file.exists(input_path)) ErrorJSON(paste0("Input LOOM file not found: ", args$file))

out_dir          <- if (!is.null(args$output_dir)) normalizePath(args$output_dir, mustWork = FALSE) else dirname(input_path)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
output_json_path <- file.path(out_dir, OUTPUT_JSON_NAME)

## ── Phase 2: heavy imports (only reached when arguments are valid) ────────────

suppressPackageStartupMessages(library(hdf5r))

if (args$method %in% c("DESeq2_VST", "DESeq2_rlog", "DESeq2_normcounts")) {
  if (!requireNamespace("DESeq2", quietly = TRUE)) ErrorJSON("DESeq2 is not installed. Install with: BiocManager::install('DESeq2')")
  suppressPackageStartupMessages(library(DESeq2))
  tool <- "DESeq2"
  tool_version <- tryCatch(as.character(packageVersion("DESeq2")), error = function(e) "unknown")
} else if (args$method == "edgeR_TMM") {
  if (!requireNamespace("edgeR", quietly = TRUE)) ErrorJSON("edgeR is not installed. Install with: BiocManager::install('edgeR')")
  suppressPackageStartupMessages(library(edgeR))
  tool <- "edgeR"
  tool_version <- tryCatch(as.character(packageVersion("edgeR")), error = function(e) "unknown")
} else if (args$method == "limma_voom") {
  if (!requireNamespace("limma", quietly = TRUE)) ErrorJSON("limma is not installed. Install with: BiocManager::install('limma')")
  if (!requireNamespace("edgeR", quietly = TRUE)) ErrorJSON("edgeR is not installed (required by limma_voom). Install with: BiocManager::install('edgeR')")
  suppressPackageStartupMessages({ library(limma); library(edgeR) })
  tool <- "limma"
  tool_version <- tryCatch(as.character(packageVersion("limma")), error = function(e) "unknown")
}

## ── LOOM helpers ─────────────────────────────────────────────────────────────

read_string_dataset <- function(h5, candidates, expected_len) {
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

write_loom_dataset <- function(h5, path, mat_gxc, n_genes, n_cols) {
  chunk_g <- min(n_genes, LOOM_CHUNK_GENES)
  chunk_c <- min(n_cols,  LOOM_CHUNK_CELLS)
  dtype   <- h5types$H5T_IEEE_F32LE
  space   <- H5S$new("simple", dims = c(n_genes, n_cols), maxdims = c(n_genes, n_cols))
  sink(nullfile())
  dcpl <- H5P_DATASET_CREATE$new()
  dcpl$set_chunk(c(chunk_g, chunk_c))
  dcpl$set_deflate(LOOM_GZIP_LEVEL)
  sink()
  parts <- Filter(nzchar, strsplit(path, "/")[[1]])
  if (length(parts) > 1) {
    grp_path <- paste(parts[-length(parts)], collapse = "/")
    if (!h5$exists(grp_path)) h5$create_group(grp_path)
  }
  if (h5$exists(path)) {
    parent_grp <- if (length(parts) > 1) h5[[paste(parts[-length(parts)], collapse = "/")]] else h5
    parent_grp$link_delete(parts[length(parts)])
  }
  ds <- h5$create_dataset(name = path, dtype = dtype, space = space, dataset_create_pl = dcpl)
  ds[1:n_genes, 1:n_cols] <- matrix(as.numeric(mat_gxc), nrow = n_genes, ncol = n_cols)
  invisible(ds$get_storage_size())
}

## Log-type info per method
bulk_log_info <- function(method) {
  switch(method,
    DESeq2_VST        = list(is_log_transformed = TRUE,  log_type = "log2 — variance stabilizing transformation (VST)",        log_base = 2),
    DESeq2_rlog       = list(is_log_transformed = TRUE,  log_type = "log2 — regularized log transformation (rlog)",            log_base = 2),
    DESeq2_normcounts = list(is_log_transformed = TRUE,  log_type = "log2 (log2(1 + size-factor-normalized counts))",          log_base = 2),
    edgeR_TMM         = list(is_log_transformed = TRUE,  log_type = "log2 — log2 CPM after TMM normalization (edgeR prior.count)", log_base = 2),
    limma_voom        = list(is_log_transformed = TRUE,  log_type = "log2 — voom log2 CPM transformation",                    log_base = 2),
    list(is_log_transformed = FALSE, log_type = NULL, log_base = NULL)
  )
}

## ── Shared warnings environment ───────────────────────────────────────────────

warn_env <- new.env(parent = emptyenv())
warn_env$w <- character(0)

capture_warnings <- function(expr) {
  withCallingHandlers(expr, warning = function(w) {
    warn_env$w <- c(warn_env$w, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
}

## ── Result skeleton ───────────────────────────────────────────────────────────

result <- list(
  loom_path        = input_path,
  tool             = tool,
  tool_version     = tool_version,
  method           = args$method,
  nber_rows        = -1L,
  nber_cols        = -1L,
  input_loom_path  = args$input_meta,
  output_loom_path = args$output_meta,
  parameters       = list(),
  normalization    = list(is_log_transformed = FALSE, log_type = NULL, log_base = NULL),
  statistics       = list()
)

## ── Read the LOOM file ────────────────────────────────────────────────────────

h5_in <- tryCatch(
  H5File$new(input_path, mode = "r"),
  error = function(e) ErrorJSON(paste0("Could not open '", args$file, "' as an HDF5/LOOM file. Is it a valid LOOM file?"))
)

src_path <- sub("^/", "", args$input_meta)
if (!h5_in$exists(src_path)) ErrorJSON(paste0("Source path '", args$input_meta, "' not found in LOOM file."))

h5_dims   <- h5_in[[src_path]]$dims      # hdf5r reverses HDF5 dims: [1]=n_samples, [2]=n_genes
n_samples <- as.integer(h5_dims[1])
n_genes   <- as.integer(h5_dims[2])
result$nber_rows <- n_genes
result$nber_cols <- n_samples
matrix_gxc <- t(h5_in[[src_path]][,])   # [,] reads (n_samples × n_genes); t() gives (n_genes × n_samples)

sample_ids <- read_string_dataset(h5_in, SAMPLE_ID_CANDIDATES, n_samples)
if (is.null(sample_ids)) sample_ids <- paste0("Sample_", seq_len(n_samples))

if (!h5_in$exists("row_attrs/_StableID")) ErrorJSON("row_attrs/_StableID not found in LOOM file. Was the file created with parse_v8.py?")
gene_ids <- as.character(h5_in[["row_attrs/_StableID"]][])
h5_in$close_all()

if (any(duplicated(gene_ids))) warn_env$w <- c(warn_env$w, paste0(sum(duplicated(gene_ids)), " duplicate gene IDs after _StableID lookup; check parse_v8.py output."))
rownames(matrix_gxc) <- gene_ids
colnames(matrix_gxc) <- sample_ids

## ── Pre-normalization statistics ──────────────────────────────────────────────

result$statistics$min_value_before    <- min(matrix_gxc)
result$statistics$max_value_before    <- max(matrix_gxc)
result$statistics$nber_zeros_before   <- as.integer(sum(matrix_gxc == 0))
result$statistics$median_depth_before <- median(colSums(matrix_gxc))   # median total raw counts per sample

## ── Check integer counts for DESeq2 and edgeR ────────────────────────────────

needs_integers <- args$method %in% c("DESeq2_VST", "DESeq2_rlog", "DESeq2_normcounts", "edgeR_TMM", "limma_voom")
if (needs_integers && any(matrix_gxc != round(matrix_gxc))) {
  warn_env$w <- c(warn_env$w, "Input matrix contains non-integer values; rounding to integers for normalization.")
  matrix_gxc <- round(matrix_gxc)
}
if (needs_integers && any(matrix_gxc < 0)) ErrorJSON("Input matrix contains negative values; cannot apply count-based normalization.")

## ── Apply normalization ───────────────────────────────────────────────────────

if (args$method %in% c("DESeq2_VST", "DESeq2_rlog", "DESeq2_normcounts")) {

  result$parameters <- list(
    input_loom_path = args$input_meta,
    output_loom_path = args$output_meta,
    blind    = args$blind,
    fit_type = args$fit_type,
    sf_type  = args$sf_type
  )

  storage.mode(matrix_gxc) <- "integer"
  dds <- tryCatch(
    capture_warnings(suppressMessages(DESeqDataSetFromMatrix(
      countData = matrix_gxc,
      colData   = data.frame(row.names = sample_ids, condition = rep("A", n_samples)),
      design    = ~1
    ))),
    error = function(e) ErrorJSON(paste0("DESeqDataSetFromMatrix failed: ", conditionMessage(e)))
  )
  dds <- tryCatch(
    capture_warnings(suppressMessages(estimateSizeFactors(dds, type = args$sf_type))),
    error = function(e) ErrorJSON(paste0("estimateSizeFactors failed: ", conditionMessage(e)))
  )

  if (args$method == "DESeq2_VST") {
    out <- tryCatch(
      capture_warnings(suppressMessages(varianceStabilizingTransformation(dds, blind = args$blind, fitType = args$fit_type))),
      error = function(e) ErrorJSON(paste0("DESeq2 VST failed: ", conditionMessage(e)))
    )
    normalized_data <- assay(out)
  } else if (args$method == "DESeq2_rlog") {
    out <- tryCatch(
      capture_warnings(suppressMessages(rlog(dds, blind = args$blind, fitType = args$fit_type))),
      error = function(e) ErrorJSON(paste0("DESeq2 rlog failed: ", conditionMessage(e)))
    )
    normalized_data <- assay(out)
  } else {
    normalized_data <- counts(dds, normalized = TRUE)
    normalized_data <- log2(1 + normalized_data)   # log2(1+x) applied after size-factor normalization
  }

} else if (args$method == "edgeR_TMM") {

  result$parameters <- list(
    input_loom_path  = args$input_meta,
    output_loom_path = args$output_meta,
    ref_column       = args$ref_column,
    prior_count      = args$prior_count
  )
  storage.mode(matrix_gxc) <- "integer"
  dge <- tryCatch(DGEList(counts = matrix_gxc), error = function(e) ErrorJSON(paste0("DGEList failed: ", conditionMessage(e))))
  dge <- tryCatch(
    capture_warnings(calcNormFactors(dge, method = "TMM", refColumn = args$ref_column)),
    error = function(e) ErrorJSON(paste0("calcNormFactors TMM failed: ", conditionMessage(e)))
  )
  normalized_data <- cpm(dge, log = TRUE, prior.count = args$prior_count)

} else if (args$method == "limma_voom") {

  result$parameters <- list(
    input_loom_path    = args$input_meta,
    output_loom_path   = args$output_meta,
    normalize_method   = args$normalize_method,
    voom_span          = args$voom_span
  )
  storage.mode(matrix_gxc) <- "integer"
  dge <- tryCatch(DGEList(counts = matrix_gxc), error = function(e) ErrorJSON(paste0("DGEList failed: ", conditionMessage(e))))
  dge <- capture_warnings(calcNormFactors(dge, method = "TMM"))
  v   <- tryCatch(
    capture_warnings(suppressMessages(voom(dge, normalize.method = args$normalize_method, span = args$voom_span))),
    error = function(e) ErrorJSON(paste0("voom failed: ", conditionMessage(e)))
  )
  normalized_data <- v$E   # log2 CPM matrix (genes × samples)
}

normalized_gxc <- suppressWarnings(as.matrix(normalized_data))
if (nrow(normalized_gxc) != n_genes || ncol(normalized_gxc) != n_samples) ErrorJSON(paste0("Normalized matrix has unexpected dimensions: ", nrow(normalized_gxc), " x ", ncol(normalized_gxc), " (expected ", n_genes, " x ", n_samples, ")."))

## ── Log transformation info ───────────────────────────────────────────────────

log_info <- bulk_log_info(args$method)
result$normalization$is_log_transformed <- log_info$is_log_transformed
result$normalization$log_type           <- log_info$log_type
result$normalization$log_base           <- log_info$log_base

## ── Post-normalization statistics ─────────────────────────────────────────────

result$statistics$min_value_after         <- min(normalized_gxc)
result$statistics$max_value_after         <- max(normalized_gxc)
result$statistics$nber_zeros_after        <- as.integer(sum(normalized_gxc == 0))
result$statistics$median_expr_per_gene    <- median(rowMeans(normalized_gxc))   # median of per-gene mean expression

## ── Append normalized layer directly into the input LOOM ─────────────────────

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
if (!h5_loom$exists("layers")) h5_loom$create_group("layers")
# Transpose back to (n_samples × n_genes): hdf5r reverses dims on write,
# so R (n_samples × n_genes) → HDF5 (n_genes × n_samples) = correct LOOM layout
norm_size <- write_loom_dataset(h5_loom, out_path_h5, t(normalized_gxc), n_samples, n_genes)
result$statistics$normalized_layer_disk_size_bytes <- as.integer(norm_size)
h5_loom$close_all()

## ── Emit output.json ─────────────────────────────────────────────────────────

if (length(warn_env$w) > 0) result$warnings <- as.list(warn_env$w)
json_str <- toJSON(result, auto_unbox = TRUE, null = "null", na = "null", pretty = FALSE, digits = 7)
if (!is.null(args$output_dir)) writeLines(json_str, con = output_json_path) else cat(json_str, "\n")