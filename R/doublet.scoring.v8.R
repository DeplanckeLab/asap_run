#!/usr/bin/env Rscript
# doublet_score.R — Doublet scoring from a LOOM file using DoubletFinder.

HELP_TEXT <- "
Doublet Score Script — DoubletFinder backend

Reads a LOOM file, runs DoubletFinder, writes the pANN doublet score and
binary doublet call (0=Singlet, 1=Doublet) into the input LOOM, and writes
output.json. When --pK is not set, paramSweep is run and the pK maximizing
BCmetric is selected (as recommended by the authors). A BCmetric plot is saved.

Options:
  -f / --file            Input LOOM file.                                   [required]
  --input_meta           Path to the raw count matrix.                      [required]
                           Example: /matrix
  --clustering_meta      Path to cluster labels for homotypic adjustment.   [optional]
                         If not set, nExp is used without homotypic adjustment.
                           Example: /col_attrs/clusters_R_leiden
  --output_score_meta    Path in /col_attrs/ for the pANN score (float).    [required]
                           Example: /col_attrs/doublet_score_df
  --output_call_meta     Path in /col_attrs/ for the binary call (0/1).     [required]
                           Example: /col_attrs/doublet_call_df
  --method               DoubletFinder                                      [required]
  -o / --output_dir      Output folder for output.json and plots.
                         [optional, default: stdout / LOOM dir for plots]
  --assay                Seurat assay name.            [default: RNA]

  -- DoubletFinder parameters ──────────────────────────────────────────────────
  --n_dims               Number of PCA dims to use.   [default: all]
  --pN                   Proportion of artificial doublets.   [default: 0.25]
  --pK                   Neighborhood size. If not set, auto-estimated via
                           paramSweep — WARNING: on large datasets (>10k cells)
                           this runs 5 pN x 33 pK = 165 full preprocessing
                           cycles and can take hours. Always provide --pK
                           for large datasets (e.g. --pK 0.09).  [default: NULL → auto]
  --n_features           Number of variable genes for internal PCA.  [default: 2000]
  --doublet_rate         Expected doublet rate (fraction of cells). Overrides
                           the 10x auto-formula.       [default: NULL → auto]
  --n_cores              Cores for paramSweep.         [default: 1]
  --seed_use             Random seed.                  [default: 42]

  --help Show this message and exit.

Output JSON doublet_score block:
  tool / tool_version, pN, pK, pK_auto, doublet_rate, nExp,
  n_doublets_called, n_singlets_called

Output plots (saved to -o dir, or same dir as LOOM file):
  plot.doublet_score.png / plot.doublet_score.json — BCmetric vs pK curve
  (only generated when --pK is NULL and paramSweep was run)

Mandatory: -f, --input_meta, --pca_meta, --clustering_meta,
           --output_score_meta, --output_call_meta, --method
"

argv <- commandArgs(trailingOnly = TRUE)
if (length(argv) == 0 || "--help" %in% argv || "-h" %in% argv) { cat(HELP_TEXT); quit(save = "no", status = 0) }

suppressPackageStartupMessages({ library(optparse); library(jsonlite) })

ErrorJSON <- function(message, output_path = NULL) {
  payload  <- list(displayed_error = message)
  json_str <- toJSON(payload, auto_unbox = TRUE)
  if (!is.null(output_path)) writeLines(json_str, con = output_path) else cat(json_str, "\n")
  quit(save = "no", status = 1)
}

LOOM_GZIP_LEVEL  <- 2L
LOOM_DTYPE       <- "float64"
OUTPUT_JSON_NAME <- "output.json"
CELL_ID_PATH     <- "col_attrs/CellID"
GENE_ID_PATH     <- "row_attrs/_StableID"

option_list <- list(
  make_option(c("-f", "--file"),       type = "character", default = NULL),
  make_option(c("-o", "--output_dir"), type = "character", default = NULL),
  make_option("--input_meta",          type = "character", default = NULL),
  make_option("--clustering_meta",     type = "character", default = NULL),
  make_option("--output_score_meta",   type = "character", default = NULL),
  make_option("--output_call_meta",    type = "character", default = NULL),
  make_option("--method",              type = "character", default = NULL),
  make_option("--assay",               type = "character", default = "RNA"),
  make_option("--n_dims",              type = "integer",   default = NULL),
  make_option("--pN",                  type = "double",    default = 0.25),
  make_option("--pK",                  type = "double",    default = NULL),
  make_option("--doublet_rate",        type = "double",    default = NULL),
  make_option("--n_features",          type = "integer",   default = 2000L),
  make_option("--seed_use",            type = "integer",   default = 42L)
)

parser <- OptionParser(option_list = option_list, add_help_option = FALSE)
args   <- parse_args(parser)

if (is.null(args$file))              ErrorJSON("Missing -f.")
if (is.null(args$input_meta))        ErrorJSON("Missing --input_meta.")
if (is.null(args$output_score_meta)) ErrorJSON("Missing --output_score_meta.")
if (is.null(args$output_call_meta))  ErrorJSON("Missing --output_call_meta.")
if (is.null(args$method))            ErrorJSON("Missing --method.")
if (args$method != "DoubletFinder")  ErrorJSON("Unknown method. Valid: DoubletFinder.")

input_path <- normalizePath(args$file, mustWork = FALSE)
if (!file.exists(input_path)) ErrorJSON(paste0("Input LOOM file not found: ", args$file))

out_dir          <- if (!is.null(args$output_dir)) normalizePath(args$output_dir, mustWork = FALSE) else NULL
if (!is.null(out_dir)) dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
output_json_path <- if (!is.null(out_dir)) file.path(out_dir, OUTPUT_JSON_NAME) else NULL
plot_dir         <- if (!is.null(out_dir)) out_dir else dirname(input_path)
plot_png_path    <- file.path(plot_dir, "plot.doublet_score.png")
plot_json_path   <- file.path(plot_dir, "plot.doublet_score.json")

suppressPackageStartupMessages({ library(hdf5r); library(Matrix); library(Seurat); library(DoubletFinder); library(ggplot2); library(plotly) })


seurat_version <- tryCatch(as.character(packageVersion("Seurat")),        error = function(e) "0.0.0")
df_version     <- tryCatch(as.character(packageVersion("DoubletFinder")), error = function(e) "unknown")
if (numeric_version(seurat_version) < numeric_version("5.0.0")) ErrorJSON(paste0("Seurat v5 required (found v", seurat_version, ")."))
if (df_version != "unknown" && numeric_version(df_version) < numeric_version("2.0.4"))
  ErrorJSON(paste0("DoubletFinder >= 2.0.4 required for Seurat v5 compatibility (found v", df_version, "). Update with: remotes::install_github('chris-mcginnis-ucsf/DoubletFinder', force = TRUE)"))

# Handle both old (v3) and new DoubletFinder API
.paramSweep    <- if (exists("paramSweep"))    paramSweep    else paramSweep_v3
.doubletFinder <- if (exists("doubletFinder")) doubletFinder else doubletFinder_v3

## ── LOOM helpers ──────────────────────────────────────────────────────────────

write_1d_float <- function(h5, path, vec) {
  dtype <- if (LOOM_DTYPE == "float64") h5types$H5T_IEEE_F64LE else h5types$H5T_IEEE_F32LE
  parts <- Filter(nzchar, strsplit(path, "/")[[1]])
  if (length(parts) > 1) { grp <- paste(parts[-length(parts)], collapse = "/"); if (!h5$exists(grp)) h5$create_group(grp) }
  if (h5$exists(path)) { pg <- if (length(parts) > 1) h5[[paste(parts[-length(parts)], collapse = "/")]] else h5; pg$link_delete(parts[length(parts)]) }
  n <- length(vec); ds <- h5$create_dataset(name = path, dtype = dtype, dims = n, chunk_dims = min(n, 1024L), gzip_level = LOOM_GZIP_LEVEL); ds[1:n] <- as.numeric(vec); invisible(ds$get_storage_size())
}

write_1d_integer <- function(h5, path, vec) {
  parts <- Filter(nzchar, strsplit(path, "/")[[1]])
  if (length(parts) > 1) { grp <- paste(parts[-length(parts)], collapse = "/"); if (!h5$exists(grp)) h5$create_group(grp) }
  if (h5$exists(path)) { pg <- if (length(parts) > 1) h5[[paste(parts[-length(parts)], collapse = "/")]] else h5; pg$link_delete(parts[length(parts)]) }
  n <- length(vec); ds <- h5$create_dataset(name = path, dtype = h5types$H5T_NATIVE_INT32, dims = n, chunk_dims = min(n, 1024L), gzip_level = LOOM_GZIP_LEVEL); ds[1:n] <- as.integer(vec); invisible(ds$get_storage_size())
}

warn_env <- new.env(parent = emptyenv()); warn_env$w <- character(0)

open_loom_with_retry <- function(path, mode, max_wait = 600L) {
  elapsed <- 0L
  repeat {
    h5 <- tryCatch(H5File$new(path, mode = mode), error = function(e) e)
    if (!inherits(h5, "error")) return(list(h5 = h5, wait_time = elapsed))
    if (elapsed >= max_wait) ErrorJSON(paste0("Could not open '", path, "' after ", max_wait, "s: ", conditionMessage(h5)))
    Sys.sleep(1); elapsed <- elapsed + 1L
  }
}

capture_warnings <- function(expr) {
  withCallingHandlers(expr, warning = function(w) { warn_env$w <- c(warn_env$w, conditionMessage(w)); invokeRestart("muffleWarning") })
}

## ── Result skeleton ───────────────────────────────────────────────────────────

result <- list(
  parameters    = list(),
  doublet_score = list(tool = "DoubletFinder", tool_version = df_version,
                       pN = args$pN, pK = NULL, pK_auto = FALSE,
                       nExp = -1L,
                       n_doublets_called = -1L, n_singlets_called = -1L),
  metadata      = list(),
  wait_time     = 0L
)

## ── Read LOOM ─────────────────────────────────────────────────────────────────

ret   <- open_loom_with_retry(input_path, mode = "r")
h5_in <- ret$h5
result$wait_time <- result$wait_time + ret$wait_time

# Count matrix
src_path <- sub("^/", "", args$input_meta)
if (!h5_in$exists(src_path)) ErrorJSON(paste0("Source path '", args$input_meta, "' not found."))
raw_mat  <- h5_in[[src_path]][,]
ra_len   <- h5_in[[GENE_ID_PATH]]$dims
ca_len   <- h5_in[[CELL_ID_PATH]]$dims
matrix_gxc <- if (nrow(raw_mat) == ca_len && ncol(raw_mat) == ra_len) t(raw_mat) else raw_mat
n_genes <- nrow(matrix_gxc); n_cells <- ncol(matrix_gxc)
cell_ids <- as.character(h5_in[[CELL_ID_PATH]][])
gene_ids <- as.character(h5_in[[GENE_ID_PATH]][])
rownames(matrix_gxc) <- gene_ids; colnames(matrix_gxc) <- cell_ids

# Cluster labels (optional)
cluster_labels <- NULL
if (!is.null(args$clustering_meta)) {
  cl_src <- sub("^/", "", args$clustering_meta)
  if (!h5_in$exists(cl_src)) ErrorJSON(paste0("Clustering path '", args$clustering_meta, "' not found."))
  cluster_labels <- as.character(h5_in[[cl_src]][])
}
h5_in$close_all()

# DoubletFinder runs its own internal PCA; n_dims = number of PCs to use internally
n_dims <- if (!is.null(args$n_dims)) args$n_dims else 50L

# Expected doublet rate
if (!is.null(args$doublet_rate)) {
  expected_dr <- args$doublet_rate
} else {
  expected_dr <- min((0.8 * n_cells) / 100000, 0.10)
  warn_env$w  <- c(warn_env$w, paste0("doublet_rate auto-computed (10x formula, capped at 10%): ", round(expected_dr * 100, 2), "%."))
}

## ── Seurat object ─────────────────────────────────────────────────────────────

sparse_counts <- suppressWarnings(Matrix(matrix_gxc, sparse = TRUE))
seurat_obj <- suppressWarnings(CreateSeuratObject(counts = sparse_counts, assay = args$assay, project = "doublet_df"))
if (!is.null(cluster_labels)) {
  seurat_obj$seurat_clusters <- factor(cluster_labels)
  Idents(seurat_obj) <- "seurat_clusters"
}

## ── OS-level stdout suppressor (silences C-level output that bypasses R's capture.output) ──

.silence <- function(expr) {
  # Redirect stdout to /dev/null using sink() which catches C-level progress bars
  # that bypass R's capture.output(). We also suppress messages.
  null_fd <- file("/dev/null", open = "w")
  sink(null_fd, type = "output")
  sink(null_fd, type = "message")
  result <- tryCatch(
    withCallingHandlers(suppressMessages(expr), warning = function(w) {
      warn_env$w <- c(warn_env$w, conditionMessage(w))
      invokeRestart("muffleWarning")
    }),
    finally = {
      sink(NULL, type = "message")
      sink(NULL, type = "output")
      close(null_fd)
    }
  )
  invisible(result)
}

## ── Pre-process Seurat object (required by DoubletFinder) ───────────────────────
# DoubletFinder requires a fully pre-processed object (NormalizeData, FindVariableFeatures,
# ScaleData, RunPCA) so that orig.commands is populated. It copies those params when
# re-processing its internal merged real+artificial dataset.
# We also force v3 assay globally so DoubletFinder's internal CreateSeuratObject
# calls also produce v3 assays, not v5 StdAssay objects.
old_assay_version <- getOption("Seurat.object.assay.version")
options(Seurat.object.assay.version = "v3")
on.exit(options(Seurat.object.assay.version = old_assay_version), add = TRUE)

seurat_obj <- .silence(suppressMessages({
  seurat_obj <- NormalizeData(seurat_obj, verbose = FALSE)
  seurat_obj <- FindVariableFeatures(seurat_obj, nfeatures = min(args$n_features, n_genes), verbose = FALSE)
  seurat_obj <- ScaleData(seurat_obj, features = VariableFeatures(seurat_obj), verbose = FALSE)
  RunPCA(seurat_obj, npcs = n_dims, verbose = FALSE)
}))

## ── pK estimation (auto) ──────────────────────────────────────────────────────

bcmvn   <- NULL
pK_auto <- FALSE
if (is.null(args$pK)) {
  pK_auto    <- TRUE
  warn_env$w <- c(warn_env$w, "Running paramSweep — may be slow on large datasets (tests 5 pN x 33 pK values). Provide --pK to skip.")
  RNGkind("L'Ecuyer-CMRG"); set.seed(args$seed_use)
  .silence({
    sweep.res   <<- suppressMessages(.paramSweep(seurat_obj, PCs = seq_len(n_dims), sct = FALSE, num.cores = 1L))
    sweep.stats <<- summarizeSweep(sweep.res, GT = FALSE)
    bcmvn       <<- find.pK(sweep.stats)
  })
  pK_used     <- as.double(as.character(bcmvn$pK[which.max(bcmvn$BCmetric)]))
  warn_env$w  <- c(warn_env$w, paste0("Auto-selected pK = ", pK_used, " (maximizes BCmetric)."))
  pK_vals <- as.numeric(as.character(bcmvn$pK))
  if (pK_used == max(pK_vals)) {
    warn_env$w <- c(warn_env$w, paste0("WARNING: selected pK (", pK_used, ") is the maximum tested value — the BCmetric curve may still be rising. Inspect the plot and consider setting --pK manually."))
  } else if (pK_used == min(pK_vals)) {
    warn_env$w <- c(warn_env$w, paste0("WARNING: selected pK (", pK_used, ") is the minimum tested value — the BCmetric curve may be monotonically decreasing (no clear peak). Inspect the plot and consider setting --pK manually."))
  }
} else {
  pK_used <- args$pK
}

## ── nExp ──────────────────────────────────────────────────────────────────────

nExp_poi <- round(expected_dr * n_cells)
if (!is.null(cluster_labels)) {
  homotypic.prop <- modelHomotypic(seurat_obj$seurat_clusters)
  nExp_poi_adj   <- round(nExp_poi * (1 - homotypic.prop))
  warn_env$w <- c(warn_env$w, paste0("nExp_poi.adj = ", nExp_poi_adj, " (homotypic-adjusted from nExp = ", nExp_poi, ")."))
} else {
  nExp_poi_adj <- nExp_poi
  warn_env$w <- c(warn_env$w, paste0("nExp = ", nExp_poi, " (no --clustering_meta; homotypic adjustment skipped)."))
}

result$parameters <- list(
  loom_path = input_path, input_loom_path = args$input_meta,
  clustering_loom_path = args$clustering_meta, output_score_loom_path = args$output_score_meta,
  output_call_loom_path = args$output_call_meta, method = args$method,
  n_dims = n_dims, n_features = as.integer(min(args$n_features, n_genes)), pN = args$pN, pK = pK_used, pK_auto = pK_auto,
  expected_doublet_rate = expected_dr, nExp = as.integer(nExp_poi_adj), seed_use = args$seed_use
)
result$doublet_score$pK          <- pK_used
result$doublet_score$pK_auto     <- pK_auto
result$doublet_score$doublet_rate <- expected_dr
result$doublet_score$nExp        <- as.integer(nExp_poi_adj)

set.seed(args$seed_use)
# Clean metadata columns to plain vectors before calling doubletFinder —
# Seurat v5 can leave some columns as data frames which causes xtfrm.data.frame errors
# when doubletFinder internally tries to sort the pANN values.
for (.col in names(seurat_obj@meta.data)) {
  if (is.data.frame(seurat_obj@meta.data[[.col]])) {
    seurat_obj@meta.data[[.col]] <- seurat_obj@meta.data[[.col]][[1]]
  }
}
seurat_obj <- .silence(suppressMessages(.doubletFinder(seurat_obj, PCs = seq_len(n_dims), pN = args$pN,
                                                        pK = pK_used, nExp = nExp_poi_adj, reuse.pANN = NULL, sct = FALSE)))

score_col <- grep("^pANN_",               colnames(seurat_obj@meta.data), value = TRUE)[1]
call_col  <- grep("^DF.classifications_", colnames(seurat_obj@meta.data), value = TRUE)[1]
if (is.na(score_col) || is.na(call_col)) ErrorJSON("DoubletFinder output columns not found in metadata.")

pann_scores  <- seurat_obj@meta.data[[score_col]]
df_calls     <- as.integer(seurat_obj@meta.data[[call_col]] == "Doublet")
n_doublets   <- sum(df_calls)
result$doublet_score$n_doublets_called <- as.integer(n_doublets)
result$doublet_score$n_singlets_called <- as.integer(n_cells - n_doublets)
result$doublet_score$doublet_rate      <- round(as.double(n_doublets) / as.double(n_cells), 6)

## ── BCmetric plot (only when pK was auto-estimated) ───────────────────────────

if (!is.null(bcmvn)) {
  tryCatch({
    bcmvn$pK <- as.numeric(as.character(bcmvn$pK))
    p <- ggplot(bcmvn, aes(x = pK, y = BCmetric, group = 1)) +
      geom_line(color = "#4A90D9") +
      geom_point(color = "#4A90D9", size = 2) +
      geom_vline(xintercept = pK_used, linetype = "dashed", color = "#E87C3E") +
      annotate("text", x = pK_used, y = min(bcmvn$BCmetric),
               label = paste0("pK = ", pK_used), hjust = -0.1, color = "#E87C3E", size = 3.5) +
      labs(title = "BCmetric distribution — optimal pK selection",
           x = "pK", y = "BCmetric") +
      theme_bw(base_size = 11) + theme(panel.grid.minor = element_blank())

    ggsave(plot_png_path, plot = p, width = 8, height = 4, dpi = 150)

    pl       <- ggplotly(p)
    fig_json <- plotly:::to_JSON(plotly::plotly_build(pl)$x)
    writeLines(fig_json, plot_json_path)

    result$plots <- list(
      list(path = plot_png_path,  type = "png"),
      list(path = plot_json_path, type = "plotly_json")
    )
  }, error = function(e) {
    warn_env$w <- c(warn_env$w, paste0("Plot generation failed: ", conditionMessage(e)))
  })
}

## ── Write to LOOM ─────────────────────────────────────────────────────────────

ret     <- open_loom_with_retry(input_path, mode = "a")
h5_loom <- ret$h5
result$wait_time <- result$wait_time + ret$wait_time
for (p in c(sub("^/", "", args$output_score_meta), sub("^/", "", args$output_call_meta))) {
  if (h5_loom$exists(p)) warn_env$w <- c(warn_env$w, paste0("Path '/", p, "' already exists and will be overwritten."))
}
size_score <- write_1d_float(h5_loom,   sub("^/", "", args$output_score_meta), pann_scores)
size_call  <- write_1d_integer(h5_loom, sub("^/", "", args$output_call_meta),  df_calls)
h5_loom$close_all()

result$metadata <- list(
  list(name = args$output_score_meta, on = "CELL", type = "NUMERIC",  nber_rows = as.integer(n_cells), nber_cols = 1L, dataset_size = as.integer(size_score), imported = 0L),
  list(name = args$output_call_meta,  on = "CELL", type = "INTEGER",  nber_rows = as.integer(n_cells), nber_cols = 1L, dataset_size = as.integer(size_call),  imported = 0L)
)

if (length(warn_env$w) > 0) result$warnings <- as.list(warn_env$w)
json_str <- toJSON(result, auto_unbox = TRUE, null = "null", na = "null", pretty = FALSE, digits = 7)
if (!is.null(output_json_path)) writeLines(json_str, con = output_json_path) else cat(json_str, "\n")