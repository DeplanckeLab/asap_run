#!/usr/bin/env Rscript
# hvg.R — Highly Variable Gene selection on scRNA-seq data from a LOOM file using Seurat v5.
#
# Reads a LOOM file, identifies HVGs, writes a 0/1 flag per gene into /row_attrs/,
# and writes output.json.
#
# Required packages: hdf5r, Seurat (>= 5.0), jsonlite, Matrix, optparse

## ── Phase 0: HELP (no library needed, runs instantly) ────────────────────────

HELP_TEXT <- "
HVG Script — Seurat v5 backend

Reads a LOOM file, runs Seurat v5 FindVariableFeatures, writes a 0/1 flag
per gene into /row_attrs/ of the input LOOM file, and writes output.json.

Options:
  -f / --file        Input LOOM file.                                       [required]
  --input_meta       Path to the count/normalized matrix inside the LOOM.   [required]
                       Examples: /matrix   /layers/normalized_ln
  --output_meta      Path in /row_attrs/ where the 0/1 HVG flag is stored.  [required]
                       Example: /row_attrs/highly_variable
  --method           HVG method: vst | mvp | disp                           [required]
  -o / --output_dir  Output folder for output.json. [optional, default: stdout]
  --assay            Seurat assay name.              [default: RNA]

  -- FindVariableFeatures parameters (defaults match Seurat v5) -------------------
  --n_top_genes      Number of top variable genes to select.     [default: 2000]
  --loess_span       Span for loess smoothing (vst only).        [default: 0.3]
  --clip_max         After standardization, values above this
                       are clipped (vst only).                   [default: Inf]
  --mean_cutoff      Min and max mean expression cutoffs for
                       mvp/disp, as 'min,max' (e.g. '0.1,8').   [default: 0.1,8]
  --dispersion_cutoff  Min and max dispersion cutoffs for
                       mvp/disp, as 'min,max' (e.g. '1,Inf').   [default: 1,Inf]
  --num_bin          Number of bins for mvp.                     [default: 20]

  --help             Show this message and exit.

Output JSON hvg block:
  tool          str  — always 'Seurat'
  tool_version  str  — installed Seurat version
  method        str  — HVG method used
  n_top_genes   int  — number of genes requested
  n_hvg_found   int  — number of HVGs actually selected

Output JSON metadata entry:
  name          str  — LOOM-internal path of the HVG flag (= --output_meta)
  on            str  — always 'GENE'
  type          str  — always 'INTEGER'
  nber_rows     int  — number of genes
  nber_cols     int  — always 1
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
LOOM_GZIP_LEVEL  <- 2L
OUTPUT_JSON_NAME <- "output.json"

CELL_ID_PATH <- "col_attrs/CellID"
GENE_ID_PATH <- "row_attrs/_StableID"

## CLI
option_list <- list(
  make_option(c("-f", "--file"),       type = "character", default = NULL,    help = "Input LOOM file. [required]"),
  make_option(c("-o", "--output_dir"), type = "character", default = NULL,    help = "Output folder. [optional, default: stdout]"),
  make_option("--input_meta",          type = "character", default = NULL,    help = "Path to the count/normalized matrix. [required]"),
  make_option("--output_meta",         type = "character", default = NULL,    help = "Path in /row_attrs/ for the 0/1 HVG flag (e.g. /row_attrs/highly_variable). [required]"),
  make_option("--method",              type = "character", default = NULL,    help = "HVG method: vst | mvp | disp. [required]"),
  make_option("--assay",               type = "character", default = "RNA",   help = "Seurat assay name. [default: RNA]"),
  make_option("--n_top_genes",         type = "integer",   default = 2000L,   help = "[FindVariableFeatures] Top variable genes. [default: 2000]"),
  make_option("--loess_span",          type = "double",    default = 0.3,     help = "[FindVariableFeatures/vst] Loess span. [default: 0.3]"),
  make_option("--clip_max",            type = "double",    default = Inf,     help = "[FindVariableFeatures/vst] Clip max. [default: Inf]"),
  make_option("--mean_cutoff",         type = "character", default = "0.1,8", help = "[FindVariableFeatures/mvp|disp] Mean cutoff 'min,max'. [default: 0.1,8]"),
  make_option("--dispersion_cutoff",   type = "character", default = "1,Inf", help = "[FindVariableFeatures/mvp|disp] Dispersion cutoff 'min,max'. [default: 1,Inf]"),
  make_option("--num_bin",             type = "integer",   default = 20L,     help = "[FindVariableFeatures/mvp] Number of bins. [default: 20]")
)

parser <- OptionParser(option_list = option_list, add_help_option = FALSE)
args   <- parse_args(parser)

if (is.null(args$file))        ErrorJSON("Missing required argument -f.")
if (is.null(args$input_meta))  ErrorJSON("Missing required argument --input_meta.")
if (is.null(args$output_meta)) ErrorJSON("Missing required argument --output_meta.")
if (is.null(args$method))      ErrorJSON("Missing required argument --method.")

valid_methods <- c("vst", "mvp", "disp")
if (!args$method %in% valid_methods) ErrorJSON(paste0("Unknown method '", args$method, "'. Valid: ", paste(valid_methods, collapse = ", "), "."))
if (!startsWith(args$output_meta, "/row_attrs/")) ErrorJSON(paste0("--output_meta must be under /row_attrs/ (e.g. /row_attrs/highly_variable), got: '", args$output_meta, "'."))

input_path <- normalizePath(args$file, mustWork = FALSE)
if (!file.exists(input_path)) ErrorJSON(paste0("Input LOOM file not found: ", args$file))

# Parse cutoff pairs
parse_cutoff <- function(s, name) {
  parts <- trimws(strsplit(s, ",")[[1]])
  if (length(parts) != 2) ErrorJSON(paste0("--", name, " must be 'min,max', got: '", s, "'."))
  as.numeric(parts)
}
mean_cutoff        <- parse_cutoff(args$mean_cutoff,       "mean_cutoff")
dispersion_cutoff  <- parse_cutoff(args$dispersion_cutoff, "dispersion_cutoff")

out_dir          <- if (!is.null(args$output_dir)) normalizePath(args$output_dir, mustWork = FALSE) else NULL
if (!is.null(out_dir)) dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
output_json_path <- if (!is.null(out_dir)) file.path(out_dir, OUTPUT_JSON_NAME) else NULL
plot_dir         <- if (!is.null(out_dir)) out_dir else dirname(input_path)
plot_png_path    <- file.path(plot_dir, "plot.hvg.png")
plot_json_path   <- file.path(plot_dir, "plot.hvg.json")

## ── Phase 2: heavy imports ────────────────────────────────────────────────────

suppressPackageStartupMessages({
  library(hdf5r)
  library(Matrix)
  library(Seurat)
  library(ggplot2)
  library(plotly)
})

seurat_version <- tryCatch(as.character(packageVersion("Seurat")), error = function(e) "0.0.0")
if (numeric_version(seurat_version) < numeric_version("5.0.0")) ErrorJSON(paste0("Seurat v5 required (found v", seurat_version, ")."))

## ── LOOM helpers ──────────────────────────────────────────────────────────────

write_1d_integer <- function(h5, path, vec) {
  # Write a 1D integer vector (0/1 HVG flags).
  parts <- Filter(nzchar, strsplit(path, "/")[[1]])
  if (length(parts) > 1) {
    grp <- paste(parts[-length(parts)], collapse = "/")
    if (!h5$exists(grp)) h5$create_group(grp)
  }
  if (h5$exists(path)) {
    pg <- if (length(parts) > 1) h5[[paste(parts[-length(parts)], collapse = "/")]] else h5
    pg$link_delete(parts[length(parts)])
  }
  n  <- length(vec)
  ds <- h5$create_dataset(name = path, dtype = h5types$H5T_NATIVE_INT32, dims = n,
                           chunk_dims = min(n, 1024L), gzip_level = LOOM_GZIP_LEVEL)
  ds[1:n] <- as.integer(vec)
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
  hvg        = list(tool = "Seurat", tool_version = seurat_version,
                    method = args$method, n_top_genes = args$n_top_genes, n_hvg_found = -1L),
  metadata   = list(),
  plots      = list(),
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
  matrix_gxc <- t(raw_mat)
} else {
  matrix_gxc <- raw_mat
}
n_genes <- nrow(matrix_gxc)
n_cells <- ncol(matrix_gxc)

cell_ids <- as.character(h5_in[[CELL_ID_PATH]][])
gene_ids <- as.character(h5_in[[GENE_ID_PATH]][])
rownames(matrix_gxc) <- gene_ids
colnames(matrix_gxc) <- cell_ids
h5_in$close_all()

# Represent Inf as the string "Inf" so JSON doesn't serialize it as null
inf_to_str <- function(v) ifelse(is.infinite(v) & v > 0, "Inf", ifelse(is.infinite(v) & v < 0, "-Inf", v))

result$parameters <- list(
  loom_path        = input_path,
  input_loom_path  = args$input_meta,
  output_loom_path = args$output_meta,
  method           = args$method,
  n_top_genes      = args$n_top_genes,
  loess_span       = if (args$method == "vst")             args$loess_span              else NULL,
  clip_max         = if (args$method == "vst")             inf_to_str(args$clip_max)    else NULL,
  mean_cutoff      = if (args$method %in% c("mvp","disp")) mean_cutoff                  else NULL,
  dispersion_cutoff= if (args$method %in% c("mvp","disp")) inf_to_str(dispersion_cutoff) else NULL,
  num_bin          = if (args$method == "mvp")             args$num_bin                 else NULL
)

## ── Create Seurat object and find variable features ───────────────────────────

# vst reads from 'counts'; mvp/disp read from 'data'.
# For vst: load the real matrix as counts (vst needs raw-ish counts).
# For mvp/disp: use a zero sparse fake for counts (faster) and load the real matrix into data.
if (args$method == "vst") {
  sparse_counts <- suppressWarnings(Matrix(matrix_gxc, sparse = TRUE))
  seurat_obj    <- suppressWarnings(CreateSeuratObject(counts = sparse_counts, assay = args$assay, project = "hvg_r"))
} else {
  fake_counts <- Matrix::Matrix(0, nrow = n_genes, ncol = n_cells, sparse = TRUE, dimnames = list(gene_ids, cell_ids))
  seurat_obj  <- suppressWarnings(CreateSeuratObject(counts = fake_counts, assay = args$assay, project = "hvg_r"))
  seurat_obj[[args$assay]] <- SetAssayData(seurat_obj[[args$assay]], layer = "data", new.data = matrix_gxc)
}

seurat_obj <- capture_warnings(suppressMessages(FindVariableFeatures(
  seurat_obj,
  assay              = args$assay,
  selection.method   = args$method,
  nfeatures          = args$n_top_genes,
  loess.span         = args$loess_span,
  clip.max           = args$clip_max,
  mean.cutoff        = mean_cutoff,
  dispersion.cutoff  = dispersion_cutoff,
  num.bin            = args$num_bin,
  verbose            = FALSE
)))

hvg_names   <- VariableFeatures(seurat_obj, assay = args$assay)
hvg_flags   <- as.integer(gene_ids %in% hvg_names)
n_hvg_found <- sum(hvg_flags)

result$hvg$n_hvg_found <- as.integer(n_hvg_found)

## ── Write HVG flags into the LOOM ────────────────────────────────────────────

ret     <- open_loom_with_retry(input_path, mode = "a")
h5_loom <- ret$h5
result$wait_time <- result$wait_time + ret$wait_time

out_path_h5 <- sub("^/", "", args$output_meta)
if (h5_loom$exists(out_path_h5)) {
  warn_env$w <- c(warn_env$w, paste0("Path '", args$output_meta, "' already exists and will be overwritten."))
}
size_hvg <- write_1d_integer(h5_loom, out_path_h5, hvg_flags)
h5_loom$close_all()

## ── Metadata + JSON ──────────────────────────────────────────────────────────

result$metadata <- list(list(
  name         = args$output_meta,
  on           = "GENE",
  type         = "INTEGER",
  nber_rows    = as.integer(n_genes),
  nber_cols    = 1L,
  dataset_size = as.integer(size_hvg),
  imported     = 0L
))

if (length(warn_env$w) > 0) result$warnings <- as.list(warn_env$w)

## ── Generate plots ───────────────────────────────────────────────────────────

tryCatch({
  # Extract per-gene statistics from the Seurat object
  hvf      <- HVFInfo(seurat_obj, assay = args$assay, method = args$method)
  hvf$gene <- rownames(hvf)
  hvf$hvg  <- hvf$gene %in% hvg_names

  # Determine axes based on method — detect column names dynamically since
  # Seurat v5 StdAssay may prefix them (e.g. vst.variance.standardized, mvp.dispersion.scaled)
  hvf$x_val <- hvf[[names(hvf)[grepl("^mean$|[.]mean$", names(hvf))][1]]]
  x_label   <- "Mean expression"
  if (args$method == "vst") {
    y_col   <- names(hvf)[grepl("standardized", names(hvf))][1]
    y_label <- "Standardized variance"
  } else {
    y_col   <- names(hvf)[grepl("normalized|scaled", names(hvf))][1]
    y_label <- "Normalized dispersion"
  }
  if (is.na(y_col)) stop(paste0("Cannot find y-axis column in HVFInfo. Available: ", paste(names(hvf), collapse = ", ")))
  hvf$y_val <- hvf[[y_col]]

  # Build ggplot
  p <- ggplot(hvf, aes(x = x_val, y = y_val, text = gene)) +
    geom_point(data = hvf[!hvf$hvg, ], color = "#CCCCCC", size = 0.4, alpha = 0.6) +
    geom_point(data = hvf[ hvf$hvg, ], color = "#E87C3E", size = 0.8, alpha = 0.9) +
    labs(title   = paste0("Highly Variable Genes — ", args$method,
                          " (", n_hvg_found, "/", n_genes, " selected)"),
         x = x_label, y = y_label) +
    theme_bw(base_size = 11) +
    theme(panel.grid.minor = element_blank())

  # Save PNG
  ggsave(plot_png_path, plot = p, width = 8, height = 6, dpi = 150)

  # Save plotly JSON — plotly:::to_JSON is plotly's own serializer and correctly
  # handles all internal plotly classes (plotly_eval etc.) that jsonlite cannot.
  pl       <- ggplotly(p, tooltip = "text")
  fig_json <- plotly:::to_JSON(plotly::plotly_build(pl)$x)
  writeLines(fig_json, plot_json_path)

  result$plots <- list(
    list(path = plot_png_path,  type = "png"),
    list(path = plot_json_path, type = "plotly_json")
  )
}, error = function(e) {
  warn_env$w <- c(warn_env$w, paste0("Plot generation failed: ", conditionMessage(e)))
})

## ── Emit output.json ─────────────────────────────────────────────────────────

if (length(warn_env$w) > 0) result$warnings <- as.list(warn_env$w)

json_str <- toJSON(result, auto_unbox = TRUE, null = "null", na = "null", pretty = FALSE, digits = 7)
if (!is.null(output_json_path)) writeLines(json_str, con = output_json_path) else cat(json_str, "\n")