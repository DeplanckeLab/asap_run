##################################################
## Project: ASAP
## Script purpose: Integration
## Date: 2025-03-14
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
for(i in 1:length(input_loom_list)){
  loom_path <- input_loom_list[i]
  batch_path <- input_batch_list[i]
  
  # Error case: Loom file does not exist
  if(!file.exists(loom_path)) error.json(paste0("This file: '", loom_path, "', does not exist!"))
  
  # Open the existing Loom in read-only mode and recuperate the infos (not optimized for OUT-OF-RAM computation)
  data.loom <- open_with_lock(loom_path, "r")
  data.matrix <- fetch_dataset(data.loom, "/matrix", transpose = T)
  data.batch <- fetch_dataset(data.loom, batch_path, transpose = T)
  close_all()
  
  # Error case: Path in Loom file does not exist
  if(is.null(data.matrix)) error.json(paste0("This file: '", loom_path, "', does not contain any dataset at path '", "/matrix", "'!"))
  if(is.null(data.batch) && batch_path != "null") error.json(paste0("This file: '", loom_path, "', does not contain any dataset at path '", batch_path, "'!"))
  
  # Creating Seurat object
  data.seurat.list[[i]] <- CreateSeuratObject(counts = data.matrix, min.cells = 0, min.features = 0) # Create our Seurat object using our data matrix (no filtering)
  if(is.null(data.batch)) { 
    data.seurat.list[[i]]$orig.ident <- paste0("ASAP_C_", i) # Created 
  } else {
    data.seurat.list[[i]]$orig.ident <- paste0("ASAP_I_", data.batch) # Imported. To avoid duplicate batch names from different datasets.
  }
  data.seurat.list[[i]] <- RenameCells(data.seurat.list[[i]], new.names = paste0(colnames(data.seurat.list[[i]]), "_", data.seurat.list[[i]]$orig.ident))
  
  # Update mapping orig / original loom file
  output_orig_mapping[[loom_path]] <- unique(data.seurat.list[[i]]$orig.ident)
}

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
png(output_convergence_plot, width = 800, height = 800)
data.seurat <- RunHarmony(object = data.seurat, group.by.vars = "orig.ident", plot_convergence = TRUE, early_stop = T, reduction.use = "pca", dims.use = 1:input_n_pcs)
#data.seurat <- RunHarmony(object = data.seurat, group.by.vars = "orig.ident", plot_convergence = TRUE, max_iter = 100, early_stop = F, reduction.use = "pca", dims.use = 1:input_n_pcs)
dev.off()

# Save output
saveRDS(data.seurat, file = output_rds_path)

# Save orig/file mapping
cat(toJSON(output_orig_mapping))

# Creating Loom file and adding the Harmony embeddings in it
#data.loom <- create_with_lock(input_loom, "r+")
#add_array_dataset(handle = data.loom, dataset_path = output_dataset_path, storage.mode_param = "integer", dataset_object = as.numeric(data.seurat@meta.data$seurat_clusters))
#datasetSize <- get_dataset_size(data.loom, output_dataset_path)
#close_all()

# Generate default JSON file
#stats <- list()
#stats$time_idle = time_idle
#if(!is.null(data.warnings)) stats$warnings = data.warnings
#stats$metadata = list(list(name = output_dataset_path, on = "CELL", type = "NUMERIC", nber_rows = 1, nber_cols = ncol(data.seurat), dataset_size = datasetSize))
#stats$nber_clusters = length(unique(data.seurat@meta.data$seurat_clusters))
#if(exists('output_dir') & !is.null(output_dir) & !is.na(output_dir)){
#  cat(toJSON(stats, method="C", auto_unbox=T, digits = NA), file = paste0(output_dir, "output.json"))
#} else {
#  cat(toJSON(stats, method="C", auto_unbox=T, digits = NA))
#}
