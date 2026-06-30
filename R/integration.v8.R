#!/usr/bin/env Rscript
# integration.v8.R — Integrate N LOOM files into a single LOOM (v8 pipeline).
#
# Fully self-contained (no source() of other pipeline scripts). R counterpart to
# integration.v8.py.
#
# Workflow:
#   1. Read N input LOOM files (each produced by parse.v8.py).
#   2. Restrict to common genes (matched on /row_attrs/Accession Ensembl IDs).
#   3. Concatenate cells across files.
#   4. Transfer/merge metadata:
#        - per-cell /col_attrs/* from the UNION of inputs are concatenated across cells;
#          absent values in any input are filled with typed missing values
#        - per-gene /row_attrs/* are taken from the first input carrying that key,
#          subset to common genes
#        - metadata that parse.v8.py recomputes is omitted to avoid stale fields
#   5. Run the integration method; for corrected methods store the embedding under /col_attrs/.
#   6. Write a new LOOM file and emit a minimal output.json.
#
# The output LOOM is meant to be re-parsed by parse.v8.py, which recomputes all
# heavy statistics and transfers the metadata we wrote. This script does NOT
# compute those stats.
#
# Required packages: hdf5r, optparse, jsonlite, Matrix, Seurat
#   + harmony (for --method harmony)
# Seurat v5 IntegrateLayers is used for --method cca / rpca.

## ── Phase 0: HELP ─────────────────────────────────────────────────────────────

HELP_TEXT <- "
Integration Script (single-cell) — Seurat backend

Reads N LOOM files (parsed via parse.v8.py), restricts to common Ensembl genes,
concatenates cells, transfers metadata, runs integration, and writes a new LOOM
intended to be re-parsed by parse.v8.py (which recomputes all heavy statistics).

Options:
  --input_looms       Comma-separated list of input LOOM file paths.        [required]
  --batch_paths       Comma-separated list of /col_attrs/<batch> paths,     [optional]
                        or 'null' per loom (then each loom becomes its own batch).
  --output_path       Output LOOM file path.                                [required]
  -o / --output_dir   Output folder for output.json. [optional, default: stdout]
  --method            Integration method: harmony | cca | rpca | uncorrected. [required]
                        harmony     : harmony batch correction in PCA space
                        cca         : Seurat v5 CCAIntegration (via IntegrateLayers)
                        rpca        : Seurat v5 RPCAIntegration (via IntegrateLayers)
                        uncorrected : naive concatenation (no batch correction)
  --n_pcs             Number of principal components.                       [default: 50]
                        Used by: harmony, cca, rpca. Ignored for 'uncorrected'.
  --n_top_genes       Number of highly variable genes.                      [default: 2000]
                        Used by: harmony, cca, rpca. Ignored for 'uncorrected'.
  --seed              Random seed.                                          [default: 42]
  --convergence_plot  Path for the Harmony convergence PNG (harmony only).  [optional]

  --help              Show this message and exit.

Mandatory parameters: --input_looms, --output_path, --method
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
LOOM_CHUNK_GENES <- 1024L
LOOM_CHUNK_CELLS <- 1024L
LOOM_GZIP_LEVEL  <- 4L
VALID_METHODS    <- c("harmony", "cca", "rpca", "uncorrected")

# Canonical paths this script writes itself; never auto-transferred from inputs.
CANONICAL_COL <- c("CellID", "_StableID", "_OrigFile", "_Batch")
CANONICAL_ROW <- c("Accession", "Gene", "Name", "Original_Gene", "_StableID")

# Metadata that parse.v8.py recomputes after integration. Do not propagate old
# values from the input LOOMs: they either become stale after concatenation/gene
# intersection, or parse.v8.py will write fresh equivalents from the matrix/DB.
# Keep this list conservative: user annotations, batches, doublet calls, clusters,
# embeddings, etc. are NOT included here and will be merged below.
PARSER_RECOMPUTED_COL <- c(
  "_Depth", "_Detected_Genes", "_Mitochondrial_Content",
  "_Ribosomal_Content", "_Protein_Coding_Content",
  # Common Scanpy/Seurat-style QC aliases of the vectors above.
  "total_counts", "log1p_total_counts", "n_counts", "nCount_RNA",
  "nCount_RNA_Log", "n_genes_by_counts", "log1p_n_genes_by_counts",
  "nFeature_RNA", "nFeature_RNA_Log", "mito_frac", "mt_frac",
  "percent.mt", "pct_counts_mt", "pct_counts_mito", "RP_frac", "RR_frac"
)
PARSER_RECOMPUTED_ROW <- c(
  "_Sum",
  "feature_biotype", "feature_type", "feature_reference",
  "feature_chromosome", "feature_length",
  # Legacy ASAP names for the same DB-derived feature fields.
  "_Biotypes", "_Chromosomes", "_SumExonLength"
)

is_parser_recomputed_col <- function(k) {
  k %in% c(CANONICAL_COL, PARSER_RECOMPUTED_COL) || grepl("^pct_counts_in_top_[0-9]+_genes$", k)
}
is_parser_recomputed_row <- function(k) {
  k %in% c(CANONICAL_ROW, PARSER_RECOMPUTED_ROW)
}

option_list <- list(
  make_option("--input_looms",      type = "character", default = NULL, help = "Comma-separated input LOOM paths. [required]"),
  make_option("--batch_paths",      type = "character", default = NULL, help = "Comma-separated /col_attrs/<batch> paths or 'null'. [optional]"),
  make_option("--output_path",      type = "character", default = NULL, help = "Output LOOM file path. [required]"),
  make_option(c("-o", "--output_dir"), type = "character", default = NULL, help = "Output folder for output.json. [optional]"),
  make_option("--method",           type = "character", default = NULL, help = "harmony | cca | rpca | uncorrected. [required]"),
  make_option("--n_pcs",            type = "integer",   default = 50L,  help = "Number of PCs. [default: 50]"),
  make_option("--n_top_genes",      type = "integer",   default = 2000L,help = "Number of HVGs. [default: 2000]"),
  make_option("--seed",             type = "integer",   default = 42L,  help = "Random seed. [default: 42]"),
  make_option("--convergence_plot", type = "character", default = NULL, help = "Harmony convergence PNG path. [optional]")
)

parser <- OptionParser(option_list = option_list, add_help_option = FALSE)
args   <- parse_args(parser)

if (is.null(args$input_looms)) ErrorJSON("Missing required argument --input_looms.")
if (is.null(args$output_path)) ErrorJSON("Missing required argument --output_path.")
if (is.null(args$method))      ErrorJSON("Missing required argument --method.")
if (!args$method %in% VALID_METHODS) ErrorJSON(paste0("Unknown --method '", args$method, "'. Valid: ", paste(VALID_METHODS, collapse = ", "), "."))

input_loom_list <- trimws(strsplit(args$input_looms, ",")[[1]])
input_loom_list <- input_loom_list[nzchar(input_loom_list)]
if (length(input_loom_list) == 0) ErrorJSON("--input_looms is empty.")
for (p in input_loom_list) if (!file.exists(p)) ErrorJSON(paste0("Input LOOM file not found: ", p))

if (!is.null(args$batch_paths)) {
  batch_list <- trimws(strsplit(args$batch_paths, ",")[[1]])
  if (length(batch_list) != length(input_loom_list))
    ErrorJSON(paste0("--batch_paths length (", length(batch_list), ") != --input_looms length (", length(input_loom_list), ")."))
} else {
  batch_list <- rep("null", length(input_loom_list))
}

out_dir          <- if (!is.null(args$output_dir)) normalizePath(args$output_dir, mustWork = FALSE) else NULL
if (!is.null(out_dir)) dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
output_json_path <- if (!is.null(out_dir)) file.path(out_dir, OUTPUT_JSON_NAME) else NULL

## ── Phase 2: heavy imports ────────────────────────────────────────────────────

warn_env    <- new.env(parent = emptyenv()); warn_env$w <- character(0)
add_warning <- function(msg) warn_env$w <- c(warn_env$w, msg)

# Record warnings from Seurat/harmony calls without redirecting stdout. The calls
# pass verbose=FALSE to stay silent; sink()/capture.output() are deliberately
# avoided as they can deadlock CreateSeuratObject and related functions.
capture_warnings <- function(expr) {
  withCallingHandlers(
    expr,
    warning = function(w) { warn_env$w <- c(warn_env$w, conditionMessage(w)); invokeRestart("muffleWarning") }
  )
}

suppressPackageStartupMessages({ library(hdf5r); library(Matrix); library(Seurat) })
seurat_version <- tryCatch(as.character(packageVersion("Seurat")), error = function(e) "unknown")
# Seurat v5 IntegrateLayers (cca/rpca) exports data objects to future workers; the
# default 500 MiB cap is easily exceeded on real datasets. Lift it generously.
options(future.globals.maxSize = 16 * 1024^3)   # 16 GiB
if (args$method == "harmony") {
  if (!requireNamespace("harmony", quietly = TRUE)) ErrorJSON("harmony not installed. Use: install.packages('harmony')")
  suppressPackageStartupMessages(library(harmony))
}

## ── Helpers ───────────────────────────────────────────────────────────────────

decode_strings <- function(x) {
  if (is.character(x)) return(x)
  if (is.raw(x))       return(rawToChar(x))
  vapply(x, function(v) if (is.raw(v)) rawToChar(v) else as.character(v), character(1))
}

read_str_dataset <- function(h5, path) {
  if (!h5$exists(path)) return(NULL)
  decode_strings(h5[[path]][])
}

# List 1D col/row_attrs keys with correct length (returns character vector of names)
list_attr_keys <- function(h5, group, expected_len) {
  if (!h5$exists(group)) return(character(0))
  keys <- names(h5[[group]])
  out  <- character(0)
  for (k in keys) {
    node <- h5[[paste0(group, "/", k)]]
    d <- tryCatch(node$dims, error = function(e) NULL)
    if (!is.null(d) && length(d) == 1 && d[1] == expected_len) out <- c(out, k)
  }
  out
}

## ── Read header/metadata from each input LOOM ────────────────────────────────

read_loom_header <- function(loom_path) {
  h5 <- tryCatch(H5File$new(loom_path, mode = "r"), error = function(e) ErrorJSON(paste0("Could not open '", loom_path, "': ", conditionMessage(e))))
  on.exit({ try(h5$close_all(), silent = TRUE) })
  if (!h5$exists("matrix")) ErrorJSON(paste0("'", loom_path, "': /matrix not found."))
  d         <- h5[["matrix"]]$dims     # hdf5r reverses: dims[1]=n_cells, dims[2]=n_genes
  n_cells   <- as.integer(d[1]); n_genes <- as.integer(d[2])
  accession <- read_str_dataset(h5, "row_attrs/Accession")
  if (is.null(accession)) ErrorJSON(paste0("'", loom_path, "': /row_attrs/Accession not found (run parse.v8.py first)."))
  cell_ids       <- read_str_dataset(h5, "col_attrs/CellID");        if (is.null(cell_ids))       cell_ids       <- paste0("Cell_", seq_len(n_cells))
  original_genes <- read_str_dataset(h5, "row_attrs/Original_Gene"); if (is.null(original_genes)) original_genes <- accession
  col_keys <- list_attr_keys(h5, "col_attrs", n_cells)
  row_keys <- list_attr_keys(h5, "row_attrs", n_genes)
  list(loom_path = loom_path, n_genes = n_genes, n_cells = n_cells,
       accession = accession, cell_ids = cell_ids, original_genes = original_genes,
       col_keys = col_keys, row_keys = row_keys)
}

read_attr_vec <- function(loom_path, path) {
  h5 <- H5File$new(loom_path, mode = "r"); on.exit({ try(h5$close_all(), silent = TRUE) })
  v <- h5[[path]][]
  if (is.raw(v) || is.character(v)) decode_strings(v) else v
}

# Merge a /col_attrs/<key> vector over all inputs, even when the key is absent
# from some files. Missing values are encoded using values that parse.v8.py's
# count_missing() recognizes:
#   - numeric metadata: NaN (keeps the vector numeric; integers promote to double)
#   - character/logical/mixed metadata: "nan" (HDF5 has no per-element NULL)
# We deliberately avoid 0 because zero can be a real biological/technical value.
merge_col_attr_with_missing <- function(parts, headers, key) {
  present <- parts[!vapply(parts, is.null, logical(1))]
  if (length(present) == 0) return(NULL)

  is_num <- vapply(present, function(x) is.numeric(x) || is.integer(x), logical(1))
  is_log <- vapply(present, is.logical, logical(1))
  is_chr <- vapply(present, is.character, logical(1))

  if (all(is_num) && !any(is_log) && !any(is_chr)) {
    out <- vector("list", length(parts))
    for (i in seq_along(parts)) {
      if (is.null(parts[[i]])) {
        out[[i]] <- rep(NaN, headers[[i]]$n_cells)
      } else {
        x <- as.numeric(parts[[i]])
        x[is.na(x)] <- NaN
        out[[i]] <- x
      }
    }
    return(do.call(c, out))
  }

  out <- vector("list", length(parts))
  for (i in seq_along(parts)) {
    if (is.null(parts[[i]])) {
      out[[i]] <- rep("nan", headers[[i]]$n_cells)
    } else {
      x <- as.character(parts[[i]])
      x[is.na(x)] <- "nan"
      out[[i]] <- x
    }
  }
  do.call(c, out)
}

read_matrix_subset <- function(loom_path, keep_idx) {
  h5 <- H5File$new(loom_path, mode = "r"); on.exit({ try(h5$close_all(), silent = TRUE) })
  ds <- h5[["matrix"]]
  nc <- as.integer(ds$dims[1])           # reversed dims: [1]=n_cells, [2]=n_genes_total
  # Read in contiguous cell-blocks, subsetting genes in memory. hdf5r scattered
  # column indexing (ds[, keep_idx]) issues a hyperslab read per index on gzipped
  # chunks and is pathologically slow; contiguous block reads avoid that.
  block_sz <- 2048L
  n_blk    <- ceiling(nc / block_sz)
  parts    <- vector("list", n_blk)
  for (b in seq_len(n_blk)) {
    r0 <- (b - 1L) * block_sz + 1L
    r1 <- min(b * block_sz, nc)
    slab <- ds[r0:r1, ]                   # (block_cells x n_genes_total), contiguous
    parts[[b]] <- Matrix::Matrix(slab[, keep_idx, drop = FALSE], sparse = TRUE)
  }
  do.call(rbind, parts)
}

headers <- lapply(input_loom_list, read_loom_header)

## ── Common-gene intersection ────────────────────────────────────────────────

common <- NULL
for (h in headers) {
  valid <- h$accession[!is.na(h$accession) & h$accession != "" & h$accession != "__unknown"]
  valid <- valid[!duplicated(valid)]
  common <- if (is.null(common)) valid else intersect(common, valid)
}
if (length(common) == 0) ErrorJSON("No common Ensembl gene IDs across input LOOM files.")
n_genes <- length(common)
n_cells <- sum(vapply(headers, function(h) h$n_cells, integer(1)))

## ── col_attrs keys to merge ─────────────────────────────────────────────────

# Use the UNION of input /col_attrs keys instead of the intersection. For inputs
# where a key is absent, merge_col_attr_with_missing() fills a typed missing value.
# Parser-recomputed/QC/canonical columns are intentionally omitted.
all_col_keys <- character(0)
for (h in headers) all_col_keys <- union(all_col_keys, h$col_keys)
transfer_col_keys <- all_col_keys[!vapply(all_col_keys, is_parser_recomputed_col, logical(1))]

## ── Read matrices + build merged structures ──────────────────────────────────

cell_offsets    <- integer(length(headers) + 1L); cell_offsets[1] <- 0L
orig_ident      <- character(n_cells)
cell_ids_all    <- character(n_cells)
batch_labels    <- character(n_cells)
orig_genes_kept <- NULL
keep_idx_first  <- NULL
blocks_cxg      <- vector("list", length(headers))

for (i in seq_along(headers)) {
  h   <- headers[[i]]
  bp  <- batch_list[i]
  idx_map  <- setNames(seq_along(h$accession), h$accession)
  keep_idx <- as.integer(idx_map[common])
  if (any(is.na(keep_idx))) ErrorJSON(paste0("Gene reordering mismatch in '", h$loom_path, "'."))
  if (is.null(keep_idx_first)) keep_idx_first <- keep_idx

  blocks_cxg[[i]] <- tryCatch(read_matrix_subset(h$loom_path, keep_idx),
                              error = function(e) ErrorJSON(paste0("Failed to read /matrix from '", h$loom_path, "': ", conditionMessage(e))))
  cell_offsets[i + 1L] <- cell_offsets[i] + h$n_cells
  if (is.null(orig_genes_kept)) orig_genes_kept <- h$original_genes[keep_idx]

  rng <- (cell_offsets[i] + 1L):cell_offsets[i + 1L]
  orig_ident[rng]   <- h$loom_path
  cell_ids_all[rng] <- paste0(h$cell_ids, "_ASAP_", i)

  if (!is.na(bp) && bp != "null" && nzchar(bp)) {
    bpath <- sub("^/", "", bp)
    h5 <- H5File$new(h$loom_path, mode = "r")
    if (!h5$exists(bpath)) { h5$close_all(); ErrorJSON(paste0("Batch path '", bp, "' not found in '", h$loom_path, "'.")) }
    b_vec <- decode_strings(h5[[bpath]][]); h5$close_all()
    if (length(b_vec) != h$n_cells) ErrorJSON(paste0("Batch '", bp, "' length ", length(b_vec), " != ", h$n_cells, " cells in '", h$loom_path, "'."))
    batch_labels[rng] <- paste0("ASAP_I_", b_vec)
  } else {
    batch_labels[rng] <- paste0("ASAP_C_", i)
  }
}

X_cxg <- do.call(rbind, blocks_cxg)
if (nrow(X_cxg) != n_cells || ncol(X_cxg) != n_genes)
  ErrorJSON(paste0("Concatenation shape mismatch: ", nrow(X_cxg), " x ", ncol(X_cxg), " vs ", n_cells, " x ", n_genes, "."))
rm(blocks_cxg); invisible(gc(verbose = FALSE))

counts_gxc <- methods::as(Matrix::t(X_cxg), "CsparseMatrix")   # genes x cells, dgCMatrix (Seurat expects CSC)
# Seurat rewrites underscores in feature/cell names to '-' and would warn on each.
# Pre-sanitize to avoid the warning flood. Cell IDs are already unique (suffixed
# with _ASAP_<i>); make.unique on genes guards against any duplicate accession.
safe_cells <- gsub("_", "-", cell_ids_all)
safe_genes <- make.unique(gsub("_", "-", as.character(common)))
rownames(counts_gxc) <- safe_genes
colnames(counts_gxc) <- safe_cells

## ── Run integration ──────────────────────────────────────────────────────────

set.seed(args$seed)
embedding      <- NULL
embedding_path <- NULL
pca_emb        <- NULL

build_seurat <- function(counts, batch) {
  # names.delim="|" (a char absent from all names) disables Seurat's name-based
  # identity parsing, which otherwise splits cell names on "_" and can hang.
  obj <- capture_warnings(CreateSeuratObject(counts = counts, min.cells = 0, min.features = 0,
                                             names.field = 1, names.delim = "|"))
  obj$orig.ident <- batch
  Idents(obj) <- "orig.ident"
  obj
}

invisible(if (args$method == "uncorrected") {
  embedding_path <- "/matrix"   # no low-dimensional embedding; advertise raw integrated matrix in JSON
  NULL   # no integration; matrix is written as-is below
} else if (args$method == "harmony") {
  seurat <- build_seurat(counts_gxc, batch_labels)
  seurat <- capture_warnings(NormalizeData(seurat, normalization.method = "LogNormalize", scale.factor = 1e4, verbose = FALSE))
  seurat <- capture_warnings(FindVariableFeatures(seurat, selection.method = "vst", nfeatures = args$n_top_genes, verbose = FALSE))
  seurat <- capture_warnings(ScaleData(seurat, features = rownames(seurat), verbose = FALSE))
  seurat <- capture_warnings(RunPCA(seurat, features = VariableFeatures(seurat), npcs = args$n_pcs, verbose = FALSE))
  if (!is.null(args$convergence_plot)) {
    grDevices::png(args$convergence_plot, width = 800, height = 800, type = "cairo")
    seurat <- capture_warnings(harmony::RunHarmony(object = seurat, group.by.vars = "orig.ident", plot_convergence = TRUE, early_stop = TRUE, reduction.use = "pca", dims.use = 1:args$n_pcs, verbose = FALSE))
    grDevices::dev.off()
  } else {
    seurat <- capture_warnings(harmony::RunHarmony(object = seurat, group.by.vars = "orig.ident", plot_convergence = FALSE, early_stop = TRUE, reduction.use = "pca", dims.use = 1:args$n_pcs, verbose = FALSE))
  }
  pca_emb        <- Seurat::Embeddings(seurat, reduction = "pca")
  embedding      <- Seurat::Embeddings(seurat, reduction = "harmony")
  embedding_path <- "/col_attrs/_harmony"
} else if (args$method %in% c("cca", "rpca")) {
  if (utils::packageVersion("Seurat") < "5.0.0")
    ErrorJSON(paste0("--method ", args$method, " requires Seurat >= 5.0.0 (found ", seurat_version, ")."))
  seurat <- build_seurat(counts_gxc, batch_labels)
  seurat[["RNA"]] <- split(seurat[["RNA"]], f = seurat$orig.ident)
  seurat <- capture_warnings(NormalizeData(seurat, normalization.method = "LogNormalize", scale.factor = 1e4, verbose = FALSE))
  seurat <- capture_warnings(FindVariableFeatures(seurat, selection.method = "vst", nfeatures = args$n_top_genes, verbose = FALSE))
  seurat <- capture_warnings(ScaleData(seurat, verbose = FALSE))
  seurat <- capture_warnings(RunPCA(seurat, npcs = args$n_pcs, verbose = FALSE))
  integ_method <- if (args$method == "cca") Seurat::CCAIntegration else Seurat::RPCAIntegration
  new_red <- paste0("integrated.", args$method)
  seurat <- capture_warnings(Seurat::IntegrateLayers(object = seurat, method = integ_method, orig.reduction = "pca", new.reduction = new_red, verbose = FALSE))
  seurat[["RNA"]] <- JoinLayers(seurat[["RNA"]])
  pca_emb        <- Seurat::Embeddings(seurat, reduction = "pca")
  embedding      <- Seurat::Embeddings(seurat, reduction = new_red)
  embedding_path <- paste0("/col_attrs/_", new_red)
})

## ── Reorder embeddings to canonical cell order ───────────────────────────────
# Seurat (esp. layer split for cca/rpca) may reorder cells. Embeddings rownames are
# the sanitized cell names; reorder rows to match our canonical safe_cells order so
# they align positionally with CellID / _OrigFile / _Batch in the output LOOM.
reorder_emb <- function(emb) {
  if (is.null(emb)) return(NULL)
  if (is.null(rownames(emb))) return(emb)
  idx <- match(safe_cells, rownames(emb))
  if (anyNA(idx)) return(emb)   # names don't match expectation; leave as-is
  emb[idx, , drop = FALSE]
}
pca_emb   <- reorder_emb(pca_emb)
embedding <- reorder_emb(embedding)

## ── Write output LOOM via hdf5r ──────────────────────────────────────────────

  if (file.exists(args$output_path)) invisible(file.remove(args$output_path))
h5_out <- H5File$new(args$output_path, mode = "w")
for (grp in c("attrs", "col_attrs", "row_attrs", "layers", "row_graphs", "col_graphs")) h5_out$create_group(grp)
h5_out[["attrs/LOOM_SPEC_VERSION"]] <- "3.0.0"

write_dataset <- function(h5, path, data) {
  path_h5 <- sub("^/", "", path)
  if (h5$exists(path_h5)) {
    parts <- strsplit(path_h5, "/")[[1]]
    parent <- if (length(parts) > 1) h5[[paste(parts[-length(parts)], collapse = "/")]] else h5
    parent$link_delete(parts[length(parts)])
  }
  parts <- strsplit(path_h5, "/")[[1]]
  if (length(parts) > 1) {
    grp_path <- paste(parts[-length(parts)], collapse = "/")
    if (!h5$exists(grp_path)) h5$create_group(grp_path)
  }
  if (is.character(data)) h5[[path_h5]] <- as.character(data)
  else if (is.matrix(data)) h5[[path_h5]] <- data
  else h5[[path_h5]] <- data
}

# Matrix: write dense (n_genes x n_cells) in cell blocks; hdf5r reverses dims on write
write_matrix <- function(h5, X_cxg, n_cells, n_genes) {
  chunk_g <- min(n_genes, LOOM_CHUNK_GENES); chunk_c <- min(n_cells, LOOM_CHUNK_CELLS)
  space <- H5S$new("simple", dims = c(n_cells, n_genes), maxdims = c(n_cells, n_genes))
  dcpl  <- H5P_DATASET_CREATE$new(); dcpl$set_chunk(c(chunk_c, chunk_g)); dcpl$set_deflate(LOOM_GZIP_LEVEL)
  ds <- h5$create_dataset(name = "matrix", dtype = h5types$H5T_IEEE_F32LE, space = space, dataset_create_pl = dcpl)
  block_sz <- LOOM_CHUNK_CELLS; n_blk <- ceiling(n_cells / block_sz)
  for (b in seq_len(n_blk)) {
    c0 <- (b - 1L) * block_sz + 1L; c1 <- min(b * block_sz, n_cells)
    ds[c0:c1, ] <- as.matrix(X_cxg[c0:c1, , drop = FALSE])
  }
}

# Minimal identifiers/provenance. _StableID is intentionally not written here:
# parse.v8.py recomputes it after integration.
write_dataset(h5_out, "/col_attrs/CellID",        cell_ids_all)
write_dataset(h5_out, "/col_attrs/_OrigFile",     orig_ident)
write_dataset(h5_out, "/col_attrs/_Batch",        batch_labels)
write_dataset(h5_out, "/row_attrs/Accession",     common)
write_dataset(h5_out, "/row_attrs/Original_Gene", orig_genes_kept)

# Transfer per-cell metadata from the union of inputs (concatenate across cells).
# If a key is missing in a loom, fill only that loom's cell segment with typed
# missing values (numeric -> NaN; string/logical/mixed -> "nan").
for (k in transfer_col_keys) {
  parts <- vector("list", length(headers)); ok <- TRUE
  for (i in seq_along(headers)) {
    if (!(k %in% headers[[i]]$col_keys)) {
      parts[i] <- list(NULL)
      next
    }
    col <- read_attr_vec(headers[[i]]$loom_path, paste0("col_attrs/", k))
    if (length(col) != headers[[i]]$n_cells) {
      add_warning(paste0("Skipping /col_attrs/", k, ": length mismatch in '", headers[[i]]$loom_path, "'."))
      ok <- FALSE
      break
    }
    parts[[i]] <- col
  }
  if (ok) {
    merged <- merge_col_attr_with_missing(parts, headers, k)
    if (!is.null(merged)) write_dataset(h5_out, paste0("/col_attrs/", k), merged)
  }
}

# Transfer per-gene metadata from the first input that carries each key, subset to
# common genes. Parser-recomputed gene annotations are omitted to avoid stale
# values and duplicate-skip warnings after parse.v8.py regenerates them.
all_row_keys <- character(0)
for (h in headers) all_row_keys <- union(all_row_keys, h$row_keys)
transfer_row_keys <- all_row_keys[!vapply(all_row_keys, is_parser_recomputed_row, logical(1))]
for (k in transfer_row_keys) {
  src_i <- which(vapply(headers, function(h) k %in% h$row_keys, logical(1)))[1]
  h <- headers[[src_i]]
  idx_map <- setNames(seq_along(h$accession), h$accession)
  keep_idx <- as.integer(idx_map[common])
  vals <- read_attr_vec(h$loom_path, paste0("row_attrs/", k))
  write_dataset(h5_out, paste0("/row_attrs/", k), vals[keep_idx])
}

# Embedding(s)
if (!is.null(pca_emb))   write_dataset(h5_out, "/col_attrs/_pca", pca_emb)
if (!is.null(embedding)) write_dataset(h5_out, embedding_path, embedding)

# Matrix
write_matrix(h5_out, X_cxg, n_cells, n_genes)
invisible(h5_out$close_all())

## ── Minimal JSON ──────────────────────────────────────────────────────────────

result <- list(
  detected_format    = "LOOM",
  input_path         = paste(input_loom_list, collapse = ","),
  output_path        = args$output_path,
  nber_rows          = as.integer(n_genes),
  nber_cols          = as.integer(n_cells),
  integration_method = args$method
)
if (!is.null(embedding_path)) result$output_embedding <- embedding_path
if (args$method %in% c("harmony", "cca", "rpca")) {
  result$n_pcs       <- as.integer(args$n_pcs)
  result$n_top_genes <- as.integer(args$n_top_genes)
}
if (length(warn_env$w) > 0) result$warnings <- as.list(warn_env$w)

json_str <- toJSON(result, auto_unbox = TRUE, null = "null", na = "null", pretty = FALSE, digits = 7)
if (!is.null(output_json_path)) writeLines(json_str, con = output_json_path) else cat(json_str, "\n")