#!/usr/bin/env Rscript
# enrich_bulk.R — Gene Set Enrichment/ORA Analysis for bulk RNA-seq from a LOOM file.
#
# Reads a DEA result matrix from /attrs/ in a LOOM file, fetches gene sets from
# the ASAP PostgreSQL database, runs enrichment (Fisher ORA, clusterProfiler ORA,
# or fgsea GSEA), stores the result matrix in /attrs/, and writes output.tsv +
# output.json.
#
# Required packages: hdf5r, optparse, jsonlite, DBI, RPostgres
#   + fgsea (Bioconductor) for the fgsea method
#   + clusterProfiler (Bioconductor) for the clusterProfiler method
#
# DB credentials: POSTGRES_USER and POSTGRES_PASSWORD environment variables.

## ── Phase 0: HELP (no library needed) ────────────────────────────────────────

HELP_TEXT <- "
Gene Set Enrichment Analysis — bulk RNA-seq

Reads DEA results from a /attrs/ LOOM matrix, fetches gene sets from the ASAP
PostgreSQL database, and runs Fisher ORA, clusterProfiler ORA, or fgsea GSEA.

Options:
  -f / --file            Input LOOM file.                                [required]
  --input_meta           LOOM /attrs/ path of the DEA result matrix.    [required]
                           (Written by dea_bulk.R, e.g. /attrs/_de_2_DESeq2)
  --output_meta          LOOM /attrs/ path for the enrichment result.   [required]
  --method               Enrichment method: fisher | clusterProfiler | fgsea
                                                                         [required]
  --geneset_db_id        ID of the gene_sets entry in the ASAP DB.      [required]
  --dburl                DB connection string: HOST[:PORT]/DB            [required]
  -o / --output_dir      Output folder (default: input file directory).

  -- Background ----------------------------------------------------------------
  --filter_meta          LOOM /row_attrs/ path of a boolean filter mask. [optional]
                           If provided, background = filter-passing genes that
                           map to the DB.  If omitted, background = genes with
                           non-NA p-value in the DEA matrix (genes actually tested).
                           Example: /row_attrs/filter_pass

  -- Gene selection (Fisher / clusterProfiler foreground only) -----------------
  Up-regulated (LFC > 0) and down-regulated (LFC < 0) genes are always processed
  separately and merged into a single output table. The effect_size (OR) is
  positive for up-regulated enrichment and negative for down-regulated enrichment.

  --top_n                Use top N genes by p-value per direction as foreground.
                           [optional; if omitted uses --fdr_threshold + --lfc_threshold]
                           E.g. --top_n 100: top 100 up + top 100 down genes.
                           Ignored with a warning for fgsea.
  --fdr_threshold        FDR threshold for foreground (ORA, if --top_n unset). [default: 0.05]
  --lfc_threshold        Log FC magnitude threshold.                      [default: 1.0]
                           up:   LFC >= +lfc_threshold
                           down: LFC <= -lfc_threshold

  -- Gene set filters ----------------------------------------------------------
  --min_geneset_size     Min gene set size (restricted to background).   [default: 5]
  --max_geneset_size     Max gene set size (restricted to background).   [default: 500]

  -- Multiple testing correction -----------------------------------------------
  --padj_method          Adjustment: BH | bonferroni | none.             [default: BH]

  --help  Show this message and exit.

DB connection:
  Reads POSTGRES_USER and POSTGRES_PASSWORD from environment variables.

Output stored in LOOM (/attrs/):
  A (n_genesets x 7) string matrix identical to output.tsv.
  Columns: name, description, p-value, adjusted p-value, effect_size, size, overlap
  Sorted: FDR<=5% block first (descending effect_size, ties ascending p-value),
          then FDR>5% block (ascending p-value).

Output files: output.tsv, output.json

Mandatory parameters: -f/--file, --input_meta, --output_meta, --method,
                      --geneset_db_id, --dburl
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

OUTPUT_JSON_NAME   <- "output.json"
OUTPUT_TSV_NAME    <- "output.tsv"
ENRICHMENT_COLUMNS <- c("name", "description", "p-value", "adjusted p-value", "effect_size", "size", "overlap")
VALID_METHODS      <- c("fisher", "clusterProfiler", "fgsea")
VALID_PADJ         <- c("BH", "bonferroni", "none")
ORA_METHODS        <- c("fisher", "clusterProfiler")  # methods that take a foreground set
GSEA_METHODS       <- c("fgsea")                       # methods that use a full ranked list

option_list <- list(
  make_option(c("-f", "--file"),         type = "character", default = NULL,   help = "Input LOOM file. [required]"),
  make_option(c("-o", "--output_dir"),   type = "character", default = NULL,   help = "Output folder. [optional]"),
  make_option("--input_meta",            type = "character", default = NULL,   help = "LOOM /attrs/ path of DEA results. [required]"),
  make_option("--output_meta",           type = "character", default = NULL,   help = "LOOM /attrs/ path for enrichment results. [required]"),
  make_option("--method",                type = "character", default = NULL,   help = paste0("Enrichment method: ", paste(VALID_METHODS, collapse = " | "), ". [required]")),
  make_option("--geneset_db_id",         type = "integer",   default = NULL,   help = "Gene_set id in ASAP DB. [required]"),
  make_option("--dburl",                 type = "character", default = NULL,   help = "DB connection HOST[:PORT]/DB. [required]"),
  make_option("--filter_meta",           type = "character", default = NULL,   help = "LOOM /row_attrs/ path of boolean filter mask. [optional]"),
  make_option("--top_n",                 type = "integer",   default = NULL,   help = "Top N ranked genes per direction as foreground (ORA only). [optional]"),
  make_option("--fdr_threshold",         type = "double",    default = 0.05,   help = "FDR threshold for foreground (ORA, if --top_n unset). [default: 0.05]"),
  make_option("--lfc_threshold",         type = "double",    default = 1.0,    help = "Abs log FC threshold for foreground (ORA, if --top_n unset). [default: 1.0]"),
  make_option("--min_geneset_size",      type = "integer",   default = 5L,     help = "Min gene set size (in background). [default: 5]"),
  make_option("--max_geneset_size",      type = "integer",   default = 500L,   help = "Max gene set size (in background). [default: 500]"),
  make_option("--padj_method",           type = "character", default = "BH",   help = "P-value adjustment: BH | bonferroni | none. [default: BH]")
)

parser <- OptionParser(option_list = option_list, add_help_option = FALSE)
args   <- parse_args(parser)

if (is.null(args$file))          ErrorJSON("Missing required argument -f.")
if (is.null(args$input_meta))    ErrorJSON("Missing required argument --input_meta.")
if (is.null(args$output_meta))   ErrorJSON("Missing required argument --output_meta.")
if (is.null(args$method))        ErrorJSON("Missing required argument --method.")
if (is.null(args$geneset_db_id)) ErrorJSON("Missing required argument --geneset_db_id.")
if (is.null(args$dburl))         ErrorJSON("Missing required argument --dburl.")
if (!args$method      %in% VALID_METHODS) ErrorJSON(paste0("Unknown --method '",      args$method,      "'. Valid: ", paste(VALID_METHODS, collapse = ", "), "."))
if (!args$padj_method %in% VALID_PADJ)   ErrorJSON(paste0("Unknown --padj_method '",  args$padj_method, "'. Valid: ", paste(VALID_PADJ,   collapse = ", "), "."))
if (!startsWith(args$input_meta,  "/attrs/"))     ErrorJSON(paste0("--input_meta must be under /attrs/, got: '",     args$input_meta,  "'."))
if (!startsWith(args$output_meta, "/attrs/"))     ErrorJSON(paste0("--output_meta must be under /attrs/, got: '",    args$output_meta, "'."))
if (!is.null(args$filter_meta) && !startsWith(args$filter_meta, "/row_attrs/"))
  ErrorJSON(paste0("--filter_meta must be under /row_attrs/, got: '", args$filter_meta, "'."))

input_path <- normalizePath(args$file, mustWork = FALSE)
if (!file.exists(input_path)) ErrorJSON(paste0("Input LOOM file not found: ", args$file))

out_dir          <- if (!is.null(args$output_dir)) normalizePath(args$output_dir, mustWork = FALSE) else dirname(input_path)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
output_json_path <- file.path(out_dir, OUTPUT_JSON_NAME)
output_tsv_path  <- file.path(out_dir, OUTPUT_TSV_NAME)

## ── Phase 2: heavy imports ────────────────────────────────────────────────────

suppressPackageStartupMessages(library(hdf5r))
if (!requireNamespace("DBI",       quietly = TRUE)) ErrorJSON("DBI not installed. Use: install.packages('DBI')")
if (!requireNamespace("RPostgres", quietly = TRUE)) ErrorJSON("RPostgres not installed. Use: install.packages('RPostgres')")
suppressPackageStartupMessages({ library(DBI); library(RPostgres) })

if (args$method == "fgsea") {
  if (!requireNamespace("fgsea", quietly = TRUE)) ErrorJSON("fgsea not installed. Use: BiocManager::install('fgsea')")
  suppressPackageStartupMessages(library(fgsea))
  tool <- "fgsea"; tool_version <- tryCatch(as.character(packageVersion("fgsea")), error = function(e) "unknown")
} else if (args$method == "clusterProfiler") {
  # Suppress stdout + stderr during the ENTIRE loading block (requireNamespace AND
  # library) — the blank line comes from .onLoad() hooks firing during
  # requireNamespace(), which runs before any subsequent local sink can be set up.
  null_msg <- file(nullfile(), "w")
  sink(null_msg, type = "message")
  cp_available <- FALSE
  invisible(capture.output(cp_available <- requireNamespace("clusterProfiler", quietly = TRUE)))
  if (!isTRUE(cp_available)) { sink(type="message"); close(null_msg); ErrorJSON("clusterProfiler not installed. Use: BiocManager::install('clusterProfiler')") }
  invisible(capture.output(suppressPackageStartupMessages(library(clusterProfiler))))
  sink(type = "message"); close(null_msg)
  tool <- "clusterProfiler"; tool_version <- tryCatch(as.character(packageVersion("clusterProfiler")), error = function(e) "unknown")
} else {
  tool <- "fisher.test"; tool_version <- tryCatch(as.character(packageVersion("base")),  error = function(e) "unknown")
}

## ── Helpers ───────────────────────────────────────────────────────────────────

warn_env         <- new.env(parent = emptyenv()); warn_env$w <- character(0)
capture_warnings <- function(expr) withCallingHandlers(expr, warning = function(w) { warn_env$w <- c(warn_env$w, conditionMessage(w)); invokeRestart("muffleWarning") })
fmt_val          <- function(x) { if (is.null(x) || (length(x)==1 && is.na(x))) return("NA"); if (is.infinite(x) && x>0) return("Inf"); if (is.infinite(x) && x<0) return("-Inf"); as.character(x) }
parse_numvec     <- function(x) suppressWarnings(as.numeric(ifelse(x == "NA", NA_character_, x)))

## ── DB helpers ────────────────────────────────────────────────────────────────

connect_db <- function(dburl) {
  pg_user <- Sys.getenv("POSTGRES_USER");  if (!nzchar(pg_user)) ErrorJSON("POSTGRES_USER environment variable is not set.")
  pg_pass <- Sys.getenv("POSTGRES_PASSWORD"); if (!nzchar(pg_pass)) ErrorJSON("POSTGRES_PASSWORD environment variable is not set.")
  m <- regmatches(dburl, regexec("^([^/:]+)(?::([0-9]+))?/(.+)$", dburl))[[1]]
  if (length(m) < 4 || !nzchar(m[4])) ErrorJSON(paste0("Invalid --dburl. Expected HOST[:PORT]/DB, got: '", dburl, "'."))
  host <- m[2]; port <- if (nzchar(m[3])) as.integer(m[3]) else 5432L; dbname <- m[4]
  tryCatch(DBI::dbConnect(RPostgres::Postgres(), host=host, port=port, dbname=dbname, user=pg_user, password=pg_pass),
           error = function(e) ErrorJSON(paste0("Failed to connect to PostgreSQL (", host, ":", port, "/", dbname, "): ", conditionMessage(e))))
}

fetch_db_data <- function(conn, geneset_db_id) {
  res <- DBI::dbGetQuery(conn, paste0("SELECT organism_id FROM gene_sets WHERE id=", geneset_db_id))
  if (nrow(res) == 0 || is.na(res$organism_id[1]) || res$organism_id[1] <= 0)
    ErrorJSON(paste0("No gene_set found in DB with id=", geneset_db_id, " or no organism associated."))
  organism_id <- res$organism_id[1]
  genes_df    <- DBI::dbGetQuery(conn, paste0("SELECT id, ensembl_id FROM genes WHERE organism_id=", organism_id))
  if (nrow(genes_df) == 0) ErrorJSON(paste0("No genes found in DB for organism_id=", organism_id, "."))
  ens_to_dbid <- setNames(genes_df$id,                      genes_df$ensembl_id)
  dbid_to_ens <- setNames(genes_df$ensembl_id, as.character(genes_df$id))
  gs_df       <- DBI::dbGetQuery(conn, paste0("SELECT identifier, name, content FROM gene_set_items WHERE gene_set_id=", geneset_db_id))
  if (nrow(gs_df) == 0) ErrorJSON(paste0("No gene_set_items found for gene_set_id=", geneset_db_id, "."))
  genesets <- lapply(seq_len(nrow(gs_df)), function(i) list(
    identifier = gs_df$identifier[i], name = gs_df$name[i],
    content    = as.integer(Filter(nzchar, strsplit(gs_df$content[i], ",")[[1]]))
  ))
  list(genesets=genesets, ens_to_dbid=ens_to_dbid, dbid_to_ens=dbid_to_ens, organism_id=organism_id)
}

## ── Read DEA results from LOOM ────────────────────────────────────────────────

h5_in    <- tryCatch(H5File$new(input_path, mode="r"), error = function(e) ErrorJSON(paste0("Could not open '", args$file, "': ", conditionMessage(e))))
dea_path <- sub("^/", "", args$input_meta)
if (!h5_in$exists(dea_path)) ErrorJSON(paste0("DEA matrix '", args$input_meta, "' not found in LOOM. Run dea_bulk.R first."))

# hdf5r reverses dims: HDF5 (n_genes × 7) → R matrix (7 × n_genes)
dea_mat  <- h5_in[[dea_path]][,]
dea_cols <- tryCatch(as.character(h5_in[[dea_path]]$attr_open("column_names")$read()),
                     error = function(e) c("ensembl_id","gene_name","log Fold-Change","p-value","FDR","Avg. Exp. Group 1","Avg. Exp. Group 2"))

# LOOM header dims (for JSON)
loom_dims <- if (h5_in$exists("matrix")) h5_in[["matrix"]]$dims else c(NA_integer_, NA_integer_)

# Optional filter mask (same hdf5r reversal: 1D row_attr, no transpose needed)
filter_mask <- NULL
if (!is.null(args$filter_meta)) {
  fmask_path <- sub("^/", "", args$filter_meta)
  if (!h5_in$exists(fmask_path)) ErrorJSON(paste0("Filter mask '", args$filter_meta, "' not found in LOOM."))
  filter_mask <- as.logical(h5_in[[fmask_path]][])
}

h5_in$close_all()

get_dea_col <- function(nm) {
  idx <- which(dea_cols == nm)
  if (length(idx) == 0) ErrorJSON(paste0("Column '", nm, "' not found in DEA matrix '", args$input_meta, "'. Columns: ", paste(dea_cols, collapse=", ")))
  as.character(dea_mat[idx, ])
}

ensembl_ids  <- get_dea_col("ensembl_id")
log_fc       <- parse_numvec(get_dea_col("log Fold-Change"))
pval_dea     <- parse_numvec(get_dea_col("p-value"))
fdr_dea      <- parse_numvec(get_dea_col("FDR"))
n_genes_total <- length(ensembl_ids)

## ── Connect to DB and fetch gene sets ────────────────────────────────────────

conn    <- connect_db(args$dburl)
db_data <- tryCatch(fetch_db_data(conn, args$geneset_db_id), finally = DBI::dbDisconnect(conn))
genesets    <- db_data$genesets
ens_to_dbid <- db_data$ens_to_dbid
dbid_to_ens <- db_data$dbid_to_ens

## ── Background definition ─────────────────────────────────────────────────────
# Background = genes that (a) map to the DB and (b) were testable.
# Testable = passed --filter_meta mask if provided, else had non-NA p-value in DEA
# (genes with NA p-value were filtered out during DEA and cannot be DE).

if (!is.null(filter_mask)) {
  if (length(filter_mask) != n_genes_total) ErrorJSON(paste0("Filter mask length (", length(filter_mask), ") != number of genes in DEA matrix (", n_genes_total, ")."))
  bg_mask <- filter_mask & !is.na(ensembl_ids) & (ensembl_ids %in% names(ens_to_dbid))
  warn_env$w <- c(warn_env$w, paste0("Background restricted to ", sum(filter_mask), " filter-passing genes (--filter_meta)."))
} else {
  # Default: genes that were actually tested in DEA (non-NA p-value)
  bg_mask <- !is.na(pval_dea) & !is.na(ensembl_ids) & (ensembl_ids %in% names(ens_to_dbid))
}

bg_ens   <- ensembl_ids[bg_mask]    # Ensembl IDs of background genes
bg_dbids <- as.integer(ens_to_dbid[bg_ens])
bg_set   <- unique(bg_dbids)
n_background <- length(bg_set)
if (n_background == 0) ErrorJSON("No background genes found. Check --filter_meta and that the correct organism's gene set is used.")

## ── Run enrichment ────────────────────────────────────────────────────────────

enrich_results           <- list()
n_foreground             <- NA_integer_
n_results_up             <- 0L
n_results_down           <- 0L
n_genesets_tested_unique <- 0L

# ── ORA methods (Fisher, clusterProfiler) ─────────────────────────────────────
if (args$method %in% ORA_METHODS) {

  # Separate base masks for each direction (background-mapped + valid LFC)
  fg_up_base   <- bg_mask & !is.na(log_fc) & log_fc > 0
  fg_down_base <- bg_mask & !is.na(log_fc) & log_fc < 0

  # Select foreground indices for one direction
  select_fg <- function(base_mask, direction) {
    if (!is.null(args$top_n)) {
      # Top N by p-value within this direction (re-rank independently of display sort)
      cand_idx <- which(base_mask)
      pv_order <- order(pval_dea[cand_idx], na.last = TRUE)
      cand_idx[pv_order][seq_len(min(args$top_n, length(cand_idx)))]
    } else {
      # Threshold mode: directional LFC cutoff (not symmetric abs())
      lfc_ok <- if (direction == "up") log_fc >= args$lfc_threshold else log_fc <= -args$lfc_threshold
      which(base_mask & !is.na(fdr_dea) & fdr_dea <= args$fdr_threshold & !is.na(lfc_ok) & lfc_ok)
    }
  }

  fg_up_idx   <- select_fg(fg_up_base,   "up")
  fg_down_idx <- select_fg(fg_down_base, "down")
  n_foreground <- length(fg_up_idx) + length(fg_down_idx)

  # Report foreground sizes
  if (!is.null(args$top_n)) {
    warn_env$w <- c(warn_env$w, paste0("Foreground: top ", args$top_n, " up-regulated (", length(fg_up_idx), " selected) + top ", args$top_n, " down-regulated (", length(fg_down_idx), " selected)."))
  } else {
    warn_env$w <- c(warn_env$w, paste0("Foreground: ", length(fg_up_idx), " up (FDR<=", args$fdr_threshold, ", LFC>=+", args$lfc_threshold, ") + ", length(fg_down_idx), " down (FDR<=", args$fdr_threshold, ", LFC<=-", args$lfc_threshold, ")."))
  }

  if (length(fg_up_idx) == 0 && length(fg_down_idx) == 0)
    ErrorJSON("No foreground genes in either direction. Try --top_n or relax --fdr_threshold / --lfc_threshold.")
  if (length(fg_up_idx)   == 0) warn_env$w <- c(warn_env$w, "No up-regulated foreground genes; skipping up enrichment.")
  if (length(fg_down_idx) == 0) warn_env$w <- c(warn_env$w, "No down-regulated foreground genes; skipping down enrichment.")

  ## Helper: run enrichment for one foreground set; sign_mult = +1 (up) or -1 (down)
  ## OR is negated for down-regulated genes so the sign encodes direction in the merged table.
  enrich_one_dir <- function(fg_idx, sign_mult) {
    if (length(fg_idx) == 0) return(list())
    fg_ens_d   <- ensembl_ids[fg_idx]
    fg_dbids_d <- as.integer(ens_to_dbid[fg_ens_d])
    fg_in_bg_d <- intersect(fg_dbids_d, bg_set)
    n_fg_d     <- length(fg_in_bg_d)
    results    <- list()

    if (args$method == "fisher") {
      for (gs in genesets) {
        gs_in_bg <- intersect(as.integer(gs$content), bg_set)
        gs_size  <- length(gs_in_bg)
        if (gs_size < args$min_geneset_size || gs_size > args$max_geneset_size) next
        a <- length(intersect(fg_in_bg_d, gs_in_bg))
        b <- gs_size - a;  c <- n_fg_d - a;  d <- n_background - gs_size - n_fg_d + a
        if (d < 0) next
        pval <- tryCatch(fisher.test(matrix(c(a,b,c,d), nrow=2), alternative="greater")$p.value, error = function(e) NA_real_)
        or   <- if (b == 0 || c == 0) Inf else (a * d) / (b * c)
        results[[length(results)+1]] <- list(name=gs$identifier, description=gs$name, pval=pval, effect_size=sign_mult * or, size=gs_size, overlap=a)
      }

    } else if (args$method == "clusterProfiler") {
      term2gene_list <- lapply(genesets, function(gs) {
        ens_in_gs <- as.character(dbid_to_ens[as.character(gs$content)])
        ens_in_gs <- ens_in_gs[!is.na(ens_in_gs) & ens_in_gs %in% bg_ens]
        if (length(ens_in_gs) == 0) return(NULL)
        data.frame(term=gs$identifier, gene=ens_in_gs, stringsAsFactors=FALSE)
      })
      term2gene <- do.call(rbind, Filter(Negate(is.null), term2gene_list))
      term2name <- data.frame(
        term = vapply(genesets, function(gs) gs$identifier, character(1)),
        name = vapply(genesets, function(gs) if (is.na(gs$name)||!nzchar(gs$name)) "" else gs$name, character(1)),
        stringsAsFactors = FALSE
      )
      if (is.null(term2gene) || nrow(term2gene) == 0) return(results)
      fg_ens_bg <- intersect(fg_ens_d, bg_ens)
      null_msg <- file(nullfile(), "w")
      sink(null_msg, type = "message")
      invisible(capture.output(
        cp_res <- capture_warnings(suppressMessages(clusterProfiler::enricher(
          gene=fg_ens_bg, universe=bg_ens, TERM2GENE=term2gene, TERM2NAME=term2name,
          minGSSize=args$min_geneset_size, maxGSSize=args$max_geneset_size,
          pAdjustMethod="BH", pvalueCutoff=1.0, qvalueCutoff=1.0
        )))
      ))
      sink(type = "message"); close(null_msg)
      if (!is.null(cp_res) && nrow(cp_res@result) > 0) {
        df      <- cp_res@result
        parse_n <- function(r, side) as.integer(strsplit(r, "/")[[1]][side])
        k_v  <- df$Count
        nfcp <- parse_n(df$GeneRatio[1], 2L)
        M_v  <- vapply(df$BgRatio, parse_n, integer(1), side=1L)
        Nbg  <- parse_n(df$BgRatio[1], 2L)
        a_v  <- k_v; b_v <- M_v-a_v; c_v <- nfcp-a_v; d_v <- Nbg-M_v-nfcp+a_v
        or_v <- ifelse(b_v==0|c_v==0, Inf, (a_v*d_v)/(b_v*c_v))
        results <- lapply(seq_len(nrow(df)), function(i) list(
          name=df$ID[i], description=df$Description[i], pval=df$pvalue[i],
          effect_size=sign_mult * or_v[i], size=M_v[i], overlap=k_v[i]
        ))
      }
    }
    results
  }

  results_up     <- enrich_one_dir(fg_up_idx,   +1)
  results_down   <- enrich_one_dir(fg_down_idx, -1)
  n_results_up   <- length(results_up)
  n_results_down <- length(results_down)
  # Gene sets tested is the unique count — up and down use the same background and
  # therefore pass or fail the size filter identically, so both counts are equal.
  n_genesets_tested_unique <- max(n_results_up, n_results_down)
  enrich_results <- c(results_up, results_down)

# ── GSEA methods (fgsea) ──────────────────────────────────────────────────────
} else if (args$method %in% GSEA_METHODS) {

  if (!is.null(args$top_n)) warn_env$w <- c(warn_env$w, "--top_n is ignored for fgsea: GSEA always uses the full ranked gene list.")

  # Ranked list: all background genes with valid log FC, sorted by log FC (no --top_n)
  # bg_mask already restricts to filter-passing + DB-mapped genes
  gsea_mask    <- bg_mask & !is.na(log_fc)
  ranked_stats <- setNames(log_fc[gsea_mask], ensembl_ids[gsea_mask])
  ranked_stats <- ranked_stats[!duplicated(names(ranked_stats))]  # deduplicate
  n_foreground <- length(ranked_stats)

  if (n_foreground == 0) ErrorJSON("No valid ranked genes for fgsea. Check DEA input and --filter_meta.")

  # Build pathway dict: identifier → [ensembl IDs present in ranked_stats]
  pathways   <- list(); path_names <- character(0)
  for (gs in genesets) {
    ens_in_gs <- as.character(dbid_to_ens[as.character(gs$content)])
    ens_in_gs <- ens_in_gs[!is.na(ens_in_gs) & ens_in_gs %in% names(ranked_stats)]
    if (length(ens_in_gs) >= args$min_geneset_size && length(ens_in_gs) <= args$max_geneset_size) {
      pathways[[gs$identifier]]   <- ens_in_gs
      path_names[[gs$identifier]] <- gs$name
    }
  }
  if (length(pathways) == 0) ErrorJSON("No gene sets passed size filters for fgsea. Try --min_geneset_size / --max_geneset_size.")

  fgsea_res <- capture_warnings(suppressMessages(fgsea::fgsea(
    pathways    = pathways,
    stats       = ranked_stats,
    minSize     = args$min_geneset_size,
    maxSize     = args$max_geneset_size,
    nPermSimple = 10000L
  )))

  enrich_results <- lapply(seq_len(nrow(fgsea_res)), function(i) list(
    name        = fgsea_res$pathway[i],
    description = path_names[[fgsea_res$pathway[i]]],
    pval        = fgsea_res$pval[i],
    effect_size = fgsea_res$NES[i],   # NES is already signed: + = up-enriched, - = down-enriched
    size        = fgsea_res$size[i],
    overlap     = length(fgsea_res$leadingEdge[[i]])
  ))
  n_genesets_tested_unique <- length(enrich_results)
}

## ── P-value adjustment ────────────────────────────────────────────────────────

n_total_entries <- length(enrich_results)
if (n_total_entries == 0) {
  warn_env$w <- c(warn_env$w, "No gene sets passed size filters. TSV and /attrs/ will be empty.")
  pvals <- adj_pvals <- numeric(0)
  n_sig_up <- n_sig_down <- 0L
} else {
  pvals     <- sapply(enrich_results, function(r) if (is.null(r$pval) || is.na(r$pval)) NA_real_ else r$pval)
  adj_pvals <- p.adjust(pvals, method = switch(args$padj_method, BH="BH", bonferroni="bonferroni", none="none"))
  for (i in seq_along(enrich_results)) enrich_results[[i]]$adj_pval <- adj_pvals[i]

  # Compute directional significant counts BEFORE sorting (positional knowledge still valid)
  sig_mask <- !is.na(adj_pvals) & adj_pvals <= 0.05
  if (args$method %in% ORA_METHODS) {
    # up-results occupy positions 1..n_results_up; down-results the remainder
    n_sig_up   <- sum(sig_mask[seq_len(n_results_up)],                           na.rm = TRUE)
    n_sig_down <- sum(sig_mask[n_results_up + seq_len(n_results_down)],           na.rm = TRUE)
  } else {
    # fgsea: split by NES sign (NES > 0 = enriched in upregulated genes)
    nes_vals   <- sapply(enrich_results, function(r) { v <- r$effect_size; if (is.null(v) || is.na(v)) NA_real_ else v })
    n_sig_up   <- sum(sig_mask & !is.na(nes_vals) & nes_vals > 0, na.rm = TRUE)
    n_sig_down <- sum(sig_mask & !is.na(nes_vals) & nes_vals < 0, na.rm = TRUE)
  }
}

## ── Two-block sort ────────────────────────────────────────────────────────────
# Block 1 (FDR <= 0.05): significant — descending |effect_size|, ties by ascending p-value
# Block 2 (FDR >  0.05): non-significant — ascending p-value

if (n_total_entries > 0) {
  is_sig    <- !is.na(adj_pvals) & adj_pvals <= 0.05
  eff_vals  <- sapply(enrich_results, function(r) { v <- r$effect_size; if (is.null(v) || is.na(v) || is.nan(v)) NA_real_ else as.numeric(v) })
  pv_safe   <- ifelse(is.na(pvals), Inf, pvals)
  block_key <- as.integer(!is_sig)
  primary   <- ifelse(is_sig, -abs(eff_vals), pv_safe)
  secondary <- ifelse(is_sig,  pv_safe,        0L)
  sort_idx  <- order(block_key, primary, secondary, na.last = TRUE)
  enrich_results <- enrich_results[sort_idx]
}

## ── Build output string matrix (identical for TSV and LOOM /attrs/) ──────────

n_results <- length(enrich_results)
if (n_results > 0) {
  attrs_matrix <- matrix(NA_character_, nrow = n_results, ncol = length(ENRICHMENT_COLUMNS))
  for (i in seq_len(n_results)) {
    r <- enrich_results[[i]]
    attrs_matrix[i, 1] <- if (is.null(r$name)        || is.na(r$name))        "NA" else r$name
    attrs_matrix[i, 2] <- if (is.null(r$description) || is.na(r$description)) "NA" else r$description
    attrs_matrix[i, 3] <- fmt_val(r$pval)
    attrs_matrix[i, 4] <- fmt_val(r$adj_pval)
    attrs_matrix[i, 5] <- fmt_val(r$effect_size)
    attrs_matrix[i, 6] <- as.character(r$size)
    attrs_matrix[i, 7] <- as.character(r$overlap)
  }
} else {
  attrs_matrix <- matrix(character(0), nrow=0, ncol=length(ENRICHMENT_COLUMNS))
}

## ── Write TSV ─────────────────────────────────────────────────────────────────

tsv_df <- as.data.frame(attrs_matrix, stringsAsFactors = FALSE)
names(tsv_df) <- ENRICHMENT_COLUMNS
write.table(tsv_df, file=output_tsv_path, sep="\t", row.names=FALSE, quote=FALSE)

## ── Write LOOM /attrs/ ────────────────────────────────────────────────────────

if (n_results > 0) {
  out_path_h5 <- sub("^/", "", args$output_meta)
  h5_loom <- tryCatch(H5File$new(input_path, mode="a"), error = function(e) ErrorJSON(paste0("Could not open LOOM in append mode: ", conditionMessage(e))))
  if (!h5_loom$exists("attrs")) h5_loom$create_group("attrs")
  if (h5_loom$exists(out_path_h5)) {
    warn_env$w <- c(warn_env$w, paste0("'", args$output_meta, "' already exists and will be overwritten."))
    parts      <- Filter(nzchar, strsplit(out_path_h5, "/")[[1]])
    parent_grp <- if (length(parts) > 1) h5_loom[[paste(parts[-length(parts)], collapse="/")]] else h5_loom
    parent_grp$link_delete(parts[length(parts)])
  }
  # Write (n_cols × n_results) in R → hdf5r → HDF5 (n_results × n_cols) ✓
  h5_loom[[out_path_h5]] <- t(attrs_matrix)
  h5attr(h5_loom[[out_path_h5]], "column_names") <- ENRICHMENT_COLUMNS
  h5_loom$close_all()
}

## ── JSON ─────────────────────────────────────────────────────────────────────

result <- list(
  loom_path              = input_path,
  tool                   = tool,
  tool_version           = tool_version,
  method                 = args$method,
  geneset_db_id          = args$geneset_db_id,
  nber_rows              = if (!is.na(loom_dims[2])) as.integer(loom_dims[2]) else n_genes_total,
  nber_cols              = if (!is.na(loom_dims[1])) as.integer(loom_dims[1]) else NULL,
  input_loom_path        = args$input_meta,
  output_loom_path       = args$output_meta,
  table_columns          = ENRICHMENT_COLUMNS,
  parameters             = list(
    method           = args$method,
    padj_method      = args$padj_method,
    filter_meta      = args$filter_meta,
    top_n            = if (args$method %in% ORA_METHODS) args$top_n else NULL,
    fdr_threshold    = if (args$method %in% ORA_METHODS && is.null(args$top_n)) args$fdr_threshold else NULL,
    lfc_threshold    = if (args$method %in% ORA_METHODS && is.null(args$top_n)) args$lfc_threshold else NULL,
    min_geneset_size = args$min_geneset_size,
    max_geneset_size = args$max_geneset_size
  ),
  n_genes_total               = n_genes_total,
  n_genes_background          = n_background,
  n_genes_foreground          = if (!is.na(n_foreground)) n_foreground else NULL,
  n_genesets_tested           = n_genesets_tested_unique,
  n_genesets_significant_up   = n_sig_up,
  n_genesets_significant_down = n_sig_down,
  tsv_path               = output_tsv_path
)
if (length(warn_env$w) > 0) result$warnings <- as.list(warn_env$w)
json_str <- toJSON(result, auto_unbox=TRUE, null="null", na="null", pretty=FALSE, digits=7)
if (!is.null(args$output_dir)) writeLines(json_str, con=output_json_path) else cat(json_str, "\n")