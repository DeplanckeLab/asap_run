##################################################
## Project: ASAP
## Script purpose: Integration
## Date: 2025-03-14
## Updated: 2026-02-19
## Author: Vincent Gardeux (vincent.gardeux@epfl.ch)
##################################################

### Parameters handling
options(echo=TRUE)
args <- commandArgs(trailingOnly = TRUE)

## Libraries
suppressPackageStartupMessages(library(Seurat))
suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(harmony))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(plotly))
suppressPackageStartupMessages(source("hdf5_lib.R"))

## Functions
serialize <- function(widget) {
  htmlwidgets:::toJSON2(widget, pretty=TRUE, digits = 3)
}

# Arguments
input_loom_path_list <- args[1] # List of looms to integrate
input_batch_path_list <- args[2] # Existing batches to integrate for each Loom
input_n_pcs <- as.numeric(args[3]) # Number of PCs of PCA to use for computing Harmony embeddings

output_rds_path <- args[4]
output_convergence_plot <- args[5]
#output_mapping_orig <- args[6]

# Test 1: 2 droso datasets, no metadata
#input_loom_path_list <- "/data/gardeux/07cuwi_parsing_output.loom,/data/gardeux/hxta68_parsing_output.loom"
#input_batch_path_list <- "null,null"
#input_n_pcs <- 50 # Default
#output_convergence_plot <- "/data/gardeux/convergence.plot.png"
#output_rds_path <- "/data/gardeux/seurat.rds"
#output_mapping_orig <- "/data/gardeux/mapping.orig.txt"

# Test 2: 2 droso datasets, no metadata
#input_loom_path_list <- "/data/gardeux/output.1.loom,/data/gardeux/output.2.loom"
#input_batch_path_list <- "null,null"
#input_n_pcs <- 30
#output_convergence_plot <- "/data/gardeux/convergence.plot.png"
#output_rds_path <- "/data/gardeux/seurat.rds"
#output_mapping_orig <- "/data/gardeux/mapping.orig.txt"

# Parameters
set.seed(42)
data.warnings <- NULL
output_orig_mapping <- list()
time_idle <- 0

# Split input in array
input_loom_list <- strsplit(input_loom_path_list, ",")[[1]]
input_batch_list <- strsplit(input_batch_path_list, ",")[[1]]
# Error case: not same dimension
if(length(input_loom_list) != length(input_batch_list)) error.json(paste0("Loom list and batch list should be of same size!"))

# Load each Loom file and build corresponding Seurat object
data.seurat.list <- list()
common.genes <- c()
for(i in 1:length(input_loom_list)){
  loom_path <- input_loom_list[i]
  batch_path <- input_batch_list[i]
  
  # Error case: Loom file does not exist
  if(!file.exists(loom_path)) error.json(paste0("This file: '", loom_path, "', does not exist!"))
  
  # Open the existing Loom in read-only mode and recuperate the info (not optimized for OUT-OF-RAM computation)
  data.loom <- open_with_lock(loom_path, "r")
  data.matrix <- fetch_dataset(data.loom, "/matrix", transpose = T)
  data.genes <- fetch_dataset(data.loom, "/row_attrs/Accession", transpose = F)
  data.cols <- fetch_dataset(data.loom, "/col_attrs/CellID", transpose = F)
  data.batch <- fetch_dataset(data.loom, batch_path, transpose = T)
  close_all()
  
  # Keep rows where gene is not __unknown and not a duplicate
  keep <- data.genes != "__unknown" & !duplicated(data.genes)
  data.matrix <- data.matrix[keep, ]
  data.genes <- data.genes[keep]
  
  # Track common genes across datasets
  if(length(common.genes) == 0) {
    common.genes <- data.genes
  } else {
    common.genes <- intersect(common.genes, data.genes)
  }
  
  # Assign gene/cell names
  colnames(data.matrix) <- data.cols
  rownames(data.matrix) <- data.genes
  
  # Error case: Path in Loom file does not exist
  if(is.null(data.matrix)) error.json(paste0("This file: '", loom_path, "', does not contain any dataset at path '", "/matrix", "'!"))
  if(is.null(data.batch) && batch_path != "null") error.json(paste0("This file: '", loom_path, "', does not contain any dataset at path '", batch_path, "'!"))
  
  # Creating Seurat object
  data.seurat <- CreateSeuratObject(counts = as(as.matrix(data.matrix), "dgCMatrix"), min.cells = 0, min.features = 0) # Create our Seurat object using our data matrix (no filtering)
  if(is.null(data.batch)) { 
    data.seurat$orig.ident <- paste0("ASAP_C_", i) # Created 
  } else {
    data.seurat$orig.ident <- paste0("ASAP_I_", data.batch) # Imported. To avoid duplicate batch names from different datasets.
  }
  data.seurat <- RenameCells(data.seurat, new.names = paste0(colnames(data.seurat), "_", data.seurat$orig.ident))
  data.seurat.list[[i]] <- data.seurat
  
  # Update mapping orig / original loom file
  output_orig_mapping[[loom_path]] <- unique(data.seurat$orig.ident)
}

# Restrict all Seurat objects to common genes
data.seurat.list <- lapply(data.seurat.list, function(s) s[common.genes, ])

# Merge the seurat objects (if needed)
if(length(data.seurat.list) > 1){
  data.seurat <- merge(x = data.seurat.list[[1]], y = data.seurat.list[2:length(data.seurat.list)])
} else {
  data.seurat <- data.seurat.list[[1]]
}

# Saving RAM
rm(data.seurat.list); gc()

# Preprocessing dataset following Seurat default options
data.seurat <- NormalizeData(data.seurat, normalization.method = "LogNormalize", scale.factor = 10000, verbose = F) 
data.seurat <- FindVariableFeatures(data.seurat, selection.method = "vst", nfeatures = 2000, verbose = F)
data.seurat <- ScaleData(data.seurat, features = rownames(data.seurat), verbose = F)
data.seurat <- RunPCA(data.seurat, features = VariableFeatures(data.seurat), verbose = F, npcs = input_n_pcs)

# Run Harmony
png(output_convergence_plot, width = 800, height = 800, type = "cairo")
data.seurat <- RunHarmony(object = data.seurat, group.by.vars = "orig.ident", plot_convergence = TRUE, early_stop = T, reduction.use = "pca", dims.use = 1:input_n_pcs)
#data.seurat <- RunHarmony(object = data.seurat, group.by.vars = "orig.ident", plot_convergence = TRUE, max_iter = 100, early_stop = F, reduction.use = "pca", dims.use = 1:input_n_pcs)
dev.off()

# Save output
saveRDS(data.seurat, file = output_rds_path)

# Save orig/file mapping
cat(toJSON(output_orig_mapping))

