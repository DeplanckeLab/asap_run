#!/usr/bin/env Rscript
# dea_bulk.R — Differential Expression Analysis for bulk RNA-seq from a LOOM file.
#
# Reads raw counts + condition labels from the LOOM, runs DEA, stores per-gene
# statistics as row attributes in the same LOOM, writes a TSV table, and outputs
# a JSON summary.
#
# Required packages: hdf5r, jsonlite, optparse
#   + DESeq2  (Bioconductor) for DESeq2 method
#   + edgeR   (Bioconductor) for edgeR method
#   + limma   (Bioconductor) for limma method (also requires edgeR)

## ── Phase 0: HELP (no library needed, runs instantly) ────────────────────────

HELP_TEXT <- "
Differential Expression Analysis — bulk RNA-seq

Reads a LOOM file, performs DEA between two conditions, appends per-gene
statistics in the input LOOM, writes a TSV results table,
and outputs output.json.

Options:
  -f / --file        Input LOOM file.                                        [required]
  --input_meta       LOOM-internal path of the raw count matrix.             [required]
                       Examples: /matrix   /layers/counts
  --output_meta      LOOM /uns/ path for the DEA result table.               [required]
                       Must be under /uns/.
                       Stores a (n_genes × 4) matrix: log2FC, stat, pvalue, padj
                       Example: /uns/dea_treated_vs_control
  --method           DEA method: DESeq2 | edgeR | limma                     [required]
  --condition        Full LOOM path of the col_attr containing condition labels. [required]
                       Example: /col_attrs/condition   /col_attrs/cluster_hc_vst
  --reference        Reference group label (denominator in log2FC).          [required]
                       Example: --reference control
  --test_group       Test group label (numerator in log2FC).
                       [optional — omit for one-vs-rest: reference vs all other samples]
                       Example: --test_group treated
  -o / --output_dir  Output folder for TSV and output.json.
                       [optional, default: input dir]

  -- Optional filter mask ---------------------------------------------------------
  --filter_meta      LOOM row_attr path of a boolean/integer mask from
                     filter_bulk.R. Genes with mask = 0 are excluded from DEA.
                     [default: NULL, all genes used]

  -- DESeq2 parameters (defaults match DESeq2 package defaults) -------------------
  --fit_type         Dispersion fit: parametric | local | mean | glmGamPoi.
                                                                  [default: parametric]
  --sf_type          Size factor type: ratio | poscounts | iterate. [default: ratio]
  --lfc_threshold    Log2FC threshold for lfcShrink / results.       [default: 0]
  --shrinkage        Shrinkage estimator: none | apeglm | ashr | normal.
                                                                  [default: apeglm]
  --alpha            Adjusted p-value threshold passed to results(). [default: 0.05]

  -- edgeR parameters (defaults match edgeR package defaults) ---------------------
  --test_type        exactTest (two-group) | glmLRT (generalized linear model).
                                                                  [default: exactTest]
  --dispersion_type  common | trended | tagwise.                   [default: tagwise]

  -- limma parameters (defaults match limma package defaults) ---------------------
  --voom_normalize   Between-sample normalisation in voom:
                     none | scale | quantile | cyclicloess.        [default: none]
  --trend_var        Trend the prior variance (limma-trend).        [default: FALSE]
  --robust_var       Robust prior variance estimation.              [default: FALSE]

  -- Significance thresholds (used for JSON summary n_de/n_up/n_down) ------------
  --padj_threshold   Adjusted p-value threshold for n_de summary.  [default: 0.05]
  --log2fc_threshold Absolute log2FC threshold for n_de summary.    [default: 1.0]

  --help        Show this message and exit.

Output stored in LOOM (/attrs/):
  A (n_genes × 7) string matrix at --output_meta. Identical content to output.tsv.
  Column order: ensembl_id, gene_name, log Fold-Change, p-value, FDR, Avg. Exp. Group 1, Avg. Exp. Group 2
  NaN values stored as \"NA\". Sorted: ascending p-value, then descending log Fold-Change.

Output files (written to output folder):
  output.tsv        ensembl_id, gene_name, log Fold-Change, p-value, FDR, Avg. Exp. Group 1, Avg. Exp. Group 2
                    Sorted: ascending p-value (NA last), then descending log Fold-Change
  output.plot.json  Plotly volcano plot (interactive JSON)

Mandatory parameters: -f/--file, --input_meta, --output_meta, --method,
                      --condition, --reference
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
OUTPUT_TSV_NAME  <- "output.tsv"
OUTPUT_PLOT_NAME <- "output.plot.json"

LOOM_COLUMNS <- c("ensembl_id", "gene_name", "log Fold-Change", "p-value", "FDR", "Avg. Exp. Group 1", "Avg. Exp. Group 2")

SAMPLE_ID_CANDIDATES <- c("col_attrs/SampleID", "col_attrs/sample_id", "col_attrs/Sample", "col_attrs/sample", "col_attrs/CellID", "col_attrs/cell_id", "col_attrs/barcode", "col_attrs/obs_names")

VALID_METHODS <- c("DESeq2", "edgeR", "limma")

option_list <- list(
  make_option(c("-f", "--file"),       type = "character", default = NULL,          help = "Input LOOM file. [required]"),
  make_option(c("-o", "--output_dir"), type = "character", default = NULL,          help = "Output folder for TSV and output.json. [optional, default: input dir]"),
  make_option("--input_meta",          type = "character", default = NULL,          help = "LOOM-internal path of the count matrix. [required]"),
  make_option("--output_meta",         type = "character", default = NULL,          help = "LOOM row_attr prefix for results (e.g. /row_attrs/dea_treated_vs_control). [required]"),
  make_option("--method",              type = "character", default = NULL,          help = paste0("DEA method: ", paste(VALID_METHODS, collapse = " | "), ". [required]")),
  make_option("--condition",           type = "character", default = NULL,          help = "Name of the LOOM col_attr containing condition labels. [required]"),
  make_option("--reference",           type = "character", default = NULL,          help = "Reference group label (denominator). [required]"),
  make_option("--test_group",          type = "character", default = NULL,          help = "Test group label (numerator). [optional — omit for one-vs-rest mode]"),
  make_option("--filter_meta",         type = "character", default = NULL,          help = "LOOM row_attr path of a filter mask from filter_bulk.R. [default: NULL]"),

  # DESeq2 parameters
  make_option("--fit_type",            type = "character", default = "parametric",  help = "[DESeq2] Dispersion fit: parametric | local | mean | glmGamPoi. [default: parametric]"),
  make_option("--sf_type",             type = "character", default = "ratio",       help = "[DESeq2] Size factor type: ratio | poscounts | iterate. [default: ratio]"),
  make_option("--lfc_threshold",       type = "double",    default = 0,             help = "[DESeq2] Log2FC threshold for results(). [default: 0]"),
  make_option("--shrinkage",           type = "character", default = "apeglm",      help = "[DESeq2] LFC shrinkage: none | apeglm | ashr | normal. [default: apeglm]"),
  make_option("--alpha",               type = "double",    default = 0.05,          help = "[DESeq2] Adjusted p-value threshold for results(). [default: 0.05]"),

  # edgeR parameters
  make_option("--test_type",           type = "character", default = "exactTest",   help = "[edgeR] Test type: exactTest | glmLRT. [default: exactTest]"),
  make_option("--dispersion_type",     type = "character", default = "tagwise",     help = "[edgeR] Dispersion: common | trended | tagwise. [default: tagwise]"),

  # limma parameters
  make_option("--voom_normalize",      type = "character", default = "none",        help = "[limma] voom normalisation: none | scale | quantile | cyclicloess. [default: none]"),
  make_option("--trend_var",           type = "logical",   default = FALSE,         help = "[limma] Trend prior variance. [default: FALSE]"),
  make_option("--robust_var",          type = "logical",   default = FALSE,         help = "[limma] Robust variance estimation. [default: FALSE]"),

  # Significance thresholds
  make_option("--padj_threshold",      type = "double",    default = 0.05,          help = "Adjusted p-value threshold for is_de. [default: 0.05]"),
  make_option("--log2fc_threshold",    type = "double",    default = 1.0,           help = "Absolute log2FC threshold for is_de. [default: 1.0]")
)

parser <- OptionParser(option_list = option_list, add_help_option = FALSE)
args   <- parse_args(parser)

if (is.null(args$file))        ErrorJSON("Missing required argument -f.")
if (is.null(args$input_meta))  ErrorJSON("Missing required argument --input_meta.")
if (is.null(args$output_meta)) ErrorJSON("Missing required argument --output_meta.")
if (is.null(args$method))      ErrorJSON("Missing required argument --method.")
if (is.null(args$condition))   ErrorJSON("Missing required argument --condition.")
if (is.null(args$reference))   ErrorJSON("Missing required argument --reference.")
if (!args$method %in% VALID_METHODS) ErrorJSON(paste0("Unknown method '", args$method, "'. Valid: ", paste(VALID_METHODS, collapse = ", "), "."))
if (!startsWith(args$output_meta, "/attrs/"))       ErrorJSON(paste0("--output_meta must be under /attrs/ (e.g. /attrs/_de_2_wilcoxon), got: '", args$output_meta, "'."))
if (!startsWith(args$condition,   "/col_attrs/")) ErrorJSON(paste0("--condition must be a full LOOM col_attr path (e.g. /col_attrs/condition), got: '", args$condition, "'."))

input_path <- normalizePath(args$file, mustWork = FALSE)
if (!file.exists(input_path)) ErrorJSON(paste0("Input LOOM file not found: ", args$file))

out_dir          <- if (!is.null(args$output_dir)) normalizePath(args$output_dir, mustWork = FALSE) else dirname(input_path)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
output_json_path <- file.path(out_dir, OUTPUT_JSON_NAME)
output_tsv_path  <- file.path(out_dir, OUTPUT_TSV_NAME)
output_plot_path <- file.path(out_dir, OUTPUT_PLOT_NAME)

## ── Phase 2: heavy imports ────────────────────────────────────────────────────

suppressPackageStartupMessages(library(hdf5r))
if (!requireNamespace("plotly", quietly = TRUE)) ErrorJSON("plotly not installed. Use: install.packages('plotly')")
suppressPackageStartupMessages(library(plotly))

if (args$method == "DESeq2") {
  if (!requireNamespace("DESeq2", quietly = TRUE)) ErrorJSON("DESeq2 not installed. Use: BiocManager::install('DESeq2')")
  suppressPackageStartupMessages(library(DESeq2))
  if (args$shrinkage == "apeglm" && !requireNamespace("apeglm", quietly = TRUE)) ErrorJSON("apeglm not installed. Use: BiocManager::install('apeglm') or set --shrinkage ashr")
  tool <- "DESeq2"; tool_version <- tryCatch(as.character(packageVersion("DESeq2")), error = function(e) "unknown")
} else if (args$method == "edgeR") {
  if (!requireNamespace("edgeR", quietly = TRUE)) ErrorJSON("edgeR not installed. Use: BiocManager::install('edgeR')")
  suppressPackageStartupMessages(library(edgeR))
  tool <- "edgeR"; tool_version <- tryCatch(as.character(packageVersion("edgeR")), error = function(e) "unknown")
} else if (args$method == "limma") {
  if (!requireNamespace("limma", quietly = TRUE)) ErrorJSON("limma not installed. Use: BiocManager::install('limma')")
  if (!requireNamespace("edgeR", quietly = TRUE)) ErrorJSON("edgeR not installed (required by limma). Use: BiocManager::install('edgeR')")
  suppressPackageStartupMessages({ library(limma); library(edgeR) })
  tool <- "limma"; tool_version <- tryCatch(as.character(packageVersion("limma")), error = function(e) "unknown")
}

## ── LOOM helpers ─────────────────────────────────────────────────────────────

read_string_dataset <- function(h5, candidates, expected_len) {
  for (path in candidates) {
    tryCatch({ if (h5$exists(path)) { vals <- h5[[path]][]; if (length(vals) == expected_len) return(as.character(vals)) } }, error = function(e) NULL)
  }
  NULL
}

warn_env <- new.env(parent = emptyenv())
warn_env$w <- character(0)
capture_warnings <- function(expr) withCallingHandlers(expr, warning = function(w) { warn_env$w <- c(warn_env$w, conditionMessage(w)); invokeRestart("muffleWarning") })

## ── Read the LOOM file ────────────────────────────────────────────────────────

h5_in <- tryCatch(H5File$new(input_path, mode = "r"), error = function(e) ErrorJSON(paste0("Could not open '", args$file, "' as HDF5/LOOM.")))

src_path <- sub("^/", "", args$input_meta)
if (!h5_in$exists(src_path)) ErrorJSON(paste0("Source path '", args$input_meta, "' not found in LOOM."))

h5_dims   <- h5_in[[src_path]]$dims      # hdf5r reverses HDF5 dims: [1]=n_samples, [2]=n_genes
n_samples <- as.integer(h5_dims[1])
n_genes   <- as.integer(h5_dims[2])
matrix_gxc <- t(h5_in[[src_path]][,])   # [,] reads (n_samples × n_genes); t() gives (n_genes × n_samples)

if (!h5_in$exists("row_attrs/_StableID")) ErrorJSON("row_attrs/_StableID not found. Was the file created with parse_v8.py?")
gene_ids         <- as.character(h5_in[["row_attrs/_StableID"]][])
gene_accessions  <- if (h5_in$exists("row_attrs/Accession"))  as.character(h5_in[["row_attrs/Accession"]][])  else rep(NA_character_, n_genes)
gene_names       <- if (h5_in$exists("row_attrs/Gene"))       as.character(h5_in[["row_attrs/Gene"]][])       else rep(NA_character_, n_genes)

sample_ids <- read_string_dataset(h5_in, SAMPLE_ID_CANDIDATES, n_samples)
if (is.null(sample_ids)) sample_ids <- paste0("Sample_", seq_len(n_samples))

# Read condition labels — args$condition is a full LOOM path e.g. /col_attrs/condition
cond_path <- sub("^/", "", args$condition)
if (!h5_in$exists(cond_path)) ErrorJSON(paste0("Condition path '", args$condition, "' not found in LOOM."))
condition_vec <- as.character(h5_in[[cond_path]][])
if (length(condition_vec) != n_samples) ErrorJSON(paste0("Condition vector length (", length(condition_vec), ") does not match number of samples (", n_samples, ")."))

# Validate groups
if (!args$reference %in% condition_vec) ErrorJSON(paste0("Reference group '", args$reference, "' not found. Found: ", paste(unique(condition_vec), collapse = ", ")))
if (!is.null(args$test_group) && !args$test_group %in% condition_vec) ErrorJSON(paste0("Test group '", args$test_group, "' not found. Found: ", paste(unique(condition_vec), collapse = ", ")))

# Resolve effective test label and condition vector
# -- Pairwise mode:  --test supplied → compare reference vs test only
# -- One-vs-rest:    --test omitted  → compare reference vs all other samples (merged)
if (is.null(args$test_group)) {
  if (!any(condition_vec != args$reference)) ErrorJSON(paste0("No samples found outside reference group '", args$reference, "'. Cannot run one-vs-rest DE."))
  effective_test        <- "all_others"
  condition_vec_for_dea <- ifelse(condition_vec == args$reference, args$reference, effective_test)
} else {
  effective_test        <- args$test_group
  condition_vec_for_dea <- condition_vec
}

# Read optional filter mask (length = n_genes = ncol)
filter_mask <- NULL
if (!is.null(args$filter_meta)) {
  fmask_path <- sub("^/", "", args$filter_meta)
  if (!h5_in$exists(fmask_path)) ErrorJSON(paste0("Filter mask '", args$filter_meta, "' not found in LOOM."))
  filter_mask <- as.logical(h5_in[[fmask_path]][])
  if (length(filter_mask) != n_genes) ErrorJSON(paste0("Filter mask length (", length(filter_mask), ") does not match number of genes (", n_genes, ")."))
}

h5_in$close_all()

if (any(duplicated(gene_ids))) warn_env$w <- c(warn_env$w, paste0(sum(duplicated(gene_ids)), " duplicate gene IDs after _StableID lookup; check parse_v8.py output."))
rownames(matrix_gxc) <- gene_ids
colnames(matrix_gxc) <- sample_ids

# Subset matrix to the two groups (all samples in one-vs-rest mode)
keep_samples <- condition_vec_for_dea %in% c(args$reference, effective_test)
matrix_sub   <- matrix_gxc[, keep_samples]
cond_sub      <- factor(condition_vec_for_dea[keep_samples], levels = c(args$reference, effective_test))
n_ref  <- sum(cond_sub == args$reference)
n_test <- sum(cond_sub == effective_test)

if (n_ref < 2)  warn_env$w <- c(warn_env$w, paste0("Reference group '", args$reference, "' has only ", n_ref, " sample(s). Results may be unreliable."))
if (n_test < 2) warn_env$w <- c(warn_env$w, paste0("Test group '", effective_test, "' has only ", n_test, " sample(s). Results may be unreliable."))

# Average expression per group (all genes, before filter) — written to TSV
avg_test_full <- rowMeans(matrix_sub[, cond_sub == effective_test, drop = FALSE])
avg_ref_full  <- rowMeans(matrix_sub[, cond_sub == args$reference,  drop = FALSE])

# Apply filter mask if provided
n_genes_before_filter <- nrow(matrix_sub)
if (!is.null(filter_mask)) {
  matrix_sub <- matrix_sub[filter_mask, ]
  warn_env$w <- c(warn_env$w, paste0("Filter mask applied: ", nrow(matrix_sub), "/", n_genes_before_filter, " genes retained for DEA."))
}
if (nrow(matrix_sub) == 0) ErrorJSON("No genes remain after applying the filter mask.")

# Round to integers
if (any(matrix_sub != round(matrix_sub))) { warn_env$w <- c(warn_env$w, "Non-integer counts detected; rounding."); matrix_sub <- round(matrix_sub) }
if (any(matrix_sub < 0)) ErrorJSON("Negative counts detected; cannot run DEA.")
storage.mode(matrix_sub) <- "integer"

## ── Run DEA ──────────────────────────────────────────────────────────────────

if (args$method == "DESeq2") {
  col_data <- data.frame(row.names = colnames(matrix_sub), condition = cond_sub)
  dds <- tryCatch(capture_warnings(suppressMessages(DESeqDataSetFromMatrix(countData = matrix_sub, colData = col_data, design = ~condition))),
    error = function(e) ErrorJSON(paste0("DESeqDataSetFromMatrix failed: ", conditionMessage(e))))
  dds <- tryCatch(capture_warnings(suppressMessages(DESeq(dds, fitType = args$fit_type, sfType = args$sf_type, quiet = TRUE))),
    error = function(e) ErrorJSON(paste0("DESeq() failed: ", conditionMessage(e))))

  coef_name <- paste0("condition_", effective_test, "_vs_", args$reference)
  if (!coef_name %in% resultsNames(dds)) coef_name <- resultsNames(dds)[length(resultsNames(dds))]

  # Always use results() for stat/pvalue/padj (lfcShrink drops stat for apeglm/ashr)
  res_df <- as.data.frame(suppressMessages(results(dds, name = coef_name, alpha = args$alpha, lfcThreshold = args$lfc_threshold)))

  # Overlay shrunken LFC if requested (only replaces log2FoldChange column)
  if (args$shrinkage != "none") {
    shrunken <- tryCatch(
      capture_warnings(suppressMessages(lfcShrink(dds, coef = coef_name, type = args$shrinkage))),
      error = function(e) { warn_env$w <- c(warn_env$w, paste0("lfcShrink failed (", conditionMessage(e), "); using unshrunken LFC.")); NULL }
    )
    if (!is.null(shrunken)) res_df$log2FoldChange <- as.data.frame(shrunken)$log2FoldChange
    else warn_env$w <- c(warn_env$w, "apeglm LFC shrinkage failed for all genes; using unshrunken LFC. Consider --shrinkage ashr or --shrinkage none.")
  }
  log2fc_col <- "log2FoldChange"; stat_col <- "stat"; pval_col <- "pvalue"; padj_col <- "padj"

} else if (args$method == "edgeR") {
  dge <- DGEList(counts = matrix_sub, group = cond_sub)
  dge <- capture_warnings(calcNormFactors(dge, method = "TMM"))
  if (args$dispersion_type == "common") {
    dge <- suppressMessages(capture_warnings(estimateCommonDisp(dge)))
  } else if (args$dispersion_type == "trended") {
    dge <- suppressMessages(capture_warnings(estimateTrendedDisp(dge)))
  } else {
    dge <- suppressMessages(capture_warnings(estimateDisp(dge)))
  }
  if (args$test_type == "exactTest") {
    et     <- tryCatch(exactTest(dge, pair = c(args$reference, effective_test)), error = function(e) ErrorJSON(paste0("exactTest failed: ", conditionMessage(e))))
    res_df <- topTags(et, n = Inf, sort.by = "none")$table
    res_df$stat <- suppressWarnings(res_df$logFC / sqrt(res_df$logCPM))  # approximate; store LR if available
  } else {
    design <- model.matrix(~cond_sub)
    fit    <- tryCatch(glmFit(dge, design), error = function(e) ErrorJSON(paste0("glmFit failed: ", conditionMessage(e))))
    lrt    <- glmLRT(fit, coef = ncol(design))
    res_df <- topTags(lrt, n = Inf, sort.by = "none")$table
    names(res_df)[names(res_df) == "LR"] <- "stat"
  }
  names(res_df)[names(res_df) == "logFC"]  <- "log2FoldChange"
  names(res_df)[names(res_df) == "PValue"] <- "pvalue"
  names(res_df)[names(res_df) == "FDR"]    <- "padj"
  if (!"stat" %in% names(res_df)) res_df$stat <- NA_real_
  log2fc_col <- "log2FoldChange"; stat_col <- "stat"; pval_col <- "pvalue"; padj_col <- "padj"

} else if (args$method == "limma") {
  dge    <- DGEList(counts = matrix_sub, group = cond_sub)
  dge    <- capture_warnings(calcNormFactors(dge, method = "TMM"))
  design <- model.matrix(~cond_sub)
  v      <- tryCatch(capture_warnings(suppressMessages(voom(dge, design, normalize.method = args$voom_normalize))),
    error = function(e) ErrorJSON(paste0("voom failed: ", conditionMessage(e))))
  fit    <- lmFit(v, design)
  fit    <- tryCatch(capture_warnings(eBayes(fit, trend = args$trend_var, robust = args$robust_var)),
    error = function(e) ErrorJSON(paste0("eBayes failed: ", conditionMessage(e))))
  res_df <- topTable(fit, coef = ncol(design), number = Inf, sort.by = "none")
  names(res_df)[names(res_df) == "logFC"]   <- "log2FoldChange"
  names(res_df)[names(res_df) == "t"]       <- "stat"
  names(res_df)[names(res_df) == "P.Value"] <- "pvalue"
  names(res_df)[names(res_df) == "adj.P.Val"] <- "padj"
  log2fc_col <- "log2FoldChange"; stat_col <- "stat"; pval_col <- "pvalue"; padj_col <- "padj"
}

## ── Build per-gene result vectors (full gene list, NA for filtered-out genes) ─

log2fc_full <- rep(NA_real_, n_genes); names(log2fc_full) <- gene_ids
pval_full   <- rep(NA_real_, n_genes); names(pval_full)   <- gene_ids
padj_full   <- rep(NA_real_, n_genes); names(padj_full)   <- gene_ids
stat_full   <- rep(NA_real_, n_genes); names(stat_full)   <- gene_ids

tested_genes <- rownames(res_df)
log2fc_full[tested_genes] <- res_df[[log2fc_col]]
pval_full[tested_genes]   <- res_df[[pval_col]]
padj_full[tested_genes]   <- res_df[[padj_col]]
stat_full[tested_genes]   <- res_df[[stat_col]]

## ── Build sorted string matrix (identical content for both TSV and LOOM /attrs/) ─
# Sort — two blocks:
# Block 1 (FDR <= padj_threshold): significant — descending |LFC|, ties by ascending p-value
# Block 2 (FDR >  padj_threshold): non-significant — ascending p-value (NA last)

is_sig_de  <- !is.na(padj_full) & padj_full <= args$padj_threshold
block_key  <- as.integer(!is_sig_de)
abs_lfc    <- abs(log2fc_full)
pv_safe    <- ifelse(is.na(pval_full), Inf, pval_full)
sort_idx   <- order(block_key, ifelse(is_sig_de, -abs_lfc, pv_safe), ifelse(is_sig_de, pv_safe, 0L), na.last = TRUE)

fmt_num <- function(x) ifelse(is.na(x), "NA", as.character(x))  # matches Python: f"{v}" if not nan else "NA"

attrs_matrix <- cbind(
  gene_accessions[sort_idx],
  gene_names[sort_idx],
  fmt_num(log2fc_full[sort_idx]),
  fmt_num(pval_full[sort_idx]),
  fmt_num(padj_full[sort_idx]),
  as.character(avg_test_full[sort_idx]),
  as.character(avg_ref_full[sort_idx])
)  # (n_genes × 7) character matrix

## ── Write DEA results as a string matrix in LOOM (/attrs/) ───────────────────
# Python stores (n_genes × 7) in HDF5; hdf5r reverses dims on write, so we
# write t(attrs_matrix) = (7 × n_genes) → hdf5r → HDF5 (n_genes × 7) ✓

out_path_h5 <- sub("^/", "", args$output_meta)
h5_loom  <- tryCatch(H5File$new(input_path, mode = "a"), error = function(e) ErrorJSON(paste0("Could not open '", args$file, "' in append mode: ", conditionMessage(e))))
if (!h5_loom$exists("attrs")) h5_loom$create_group("attrs")
if (h5_loom$exists(out_path_h5)) {
  warn_env$w <- c(warn_env$w, paste0("'", args$output_meta, "' already exists and will be overwritten."))
  parts      <- Filter(nzchar, strsplit(out_path_h5, "/")[[1]])
  parent_grp <- if (length(parts) > 1) h5_loom[[paste(parts[-length(parts)], collapse = "/")]] else h5_loom
  parent_grp$link_delete(parts[length(parts)])
}
h5_loom[[out_path_h5]] <- t(attrs_matrix)
h5attr(h5_loom[[out_path_h5]], "column_names") <- LOOM_COLUMNS
h5_loom$close_all()

## ── Write TSV (identical to LOOM /attrs/ matrix) ─────────────────────────────

tsv_df <- as.data.frame(attrs_matrix, stringsAsFactors = FALSE)
names(tsv_df) <- LOOM_COLUMNS
write.table(tsv_df, file = output_tsv_path, sep = "\t", row.names = FALSE, quote = FALSE)

## ── Summary counts ────────────────────────────────────────────────────────────

is_de_full <- as.integer(!is.na(padj_full) & padj_full <= args$padj_threshold & abs(log2fc_full) >= args$log2fc_threshold)
n_de   <- sum(is_de_full, na.rm = TRUE)
n_up   <- sum(!is.na(log2fc_full) & !is.na(padj_full) & padj_full <= args$padj_threshold & log2fc_full >= args$log2fc_threshold)
n_down <- sum(!is.na(log2fc_full) & !is.na(padj_full) & padj_full <= args$padj_threshold & log2fc_full <= -args$log2fc_threshold)

## ── Volcano plot (output.plot.json, matches de.v8.py write_volcano_json) ──────

volcano_path <- tryCatch({
  not_na    <- !is.na(log2fc_full) & !is.na(pval_full) & pval_full > 0
  x         <- log2fc_full[not_na]
  y         <- -log10(pval_full[not_na])
  hover_txt <- paste0("log FC: ", round(x, 3), "<br>p-value: ", formatC(pval_full[not_na], format = "e", digits = 2), "<br>Ensembl: ", gene_accessions[not_na], "<br>Gene: ", gene_names[not_na])
  fig <- plot_ly(x = x, y = y, type = "scatter", mode = "markers", text = hover_txt, hoverinfo = "text", marker = list(size = 4, opacity = 0.7)) %>%
    layout(title = "Volcano plot", xaxis = list(title = "log Fold-Change"), yaxis = list(title = "-log10(p-value)"))
  writeLines(plotly_json(fig, pretty = FALSE), con = output_plot_path)
  output_plot_path
}, error = function(e) {
  warn_env$w <<- c(warn_env$w, paste0("Volcano plot failed: ", conditionMessage(e)))
  NULL
})

## ── Result and output.json ────────────────────────────────────────────────────

# Collapse duplicate warnings (e.g. repeated apeglm optimiser messages)
if (length(warn_env$w) > 0) {
  warn_tbl   <- table(warn_env$w)
  warn_env$w <- mapply(function(msg, n) {
    collapsed <- if (n > 1) paste0(msg, " [x", n, "]") else msg
    if (grepl("line search routine failed", msg)) paste0(collapsed, " — apeglm numerical issue; consider --shrinkage ashr") else collapsed
  }, names(warn_tbl), as.integer(warn_tbl), SIMPLIFY = TRUE, USE.NAMES = FALSE)
}

result <- list(
  loom_path        = input_path,
  tool             = tool,
  tool_version     = tool_version,
  method           = args$method,
  nber_rows        = n_genes,
  nber_cols        = n_samples,
  input_loom_path  = args$input_meta,
  output_loom_path = args$output_meta,
  table_columns    = LOOM_COLUMNS,
  comparison       = list(test = args$test_group, reference = args$reference, condition_loom_path = args$condition, mode = if (is.null(args$test_group)) "one_vs_rest" else "pairwise"),
  parameters       = list(input_loom_path = args$input_meta, output_loom_path = args$output_meta, fit_type = if (args$method == "DESeq2") args$fit_type else NULL, sf_type = if (args$method == "DESeq2") args$sf_type else NULL, shrinkage = if (args$method == "DESeq2") args$shrinkage else NULL, lfc_threshold = if (args$method == "DESeq2") args$lfc_threshold else NULL, alpha = if (args$method == "DESeq2") args$alpha else NULL, test_type = if (args$method == "edgeR") args$test_type else NULL, dispersion_type = if (args$method == "edgeR") args$dispersion_type else NULL, voom_normalize = if (args$method == "limma") args$voom_normalize else NULL, trend_var = if (args$method == "limma") args$trend_var else NULL, robust_var = if (args$method == "limma") args$robust_var else NULL, padj_threshold = args$padj_threshold, log2fc_threshold = args$log2fc_threshold),
  n_samples_reference = n_ref,
  n_samples_test      = n_test,
  n_genes_tested      = length(tested_genes),
  n_de                = n_de,
  n_up                = n_up,
  n_down              = n_down,
  tsv_path            = output_tsv_path,
  volcano_path        = volcano_path
)
if (length(warn_env$w) > 0) result$warnings <- as.list(warn_env$w)
json_str <- toJSON(result, auto_unbox = TRUE, null = "null", na = "null", pretty = FALSE, digits = 7)
if (!is.null(args$output_dir)) writeLines(json_str, con = output_json_path) else cat(json_str, "\n")