##################################################
## Project: ASAP
## Script purpose: Normalization v3
## Date: 2023 October 12
## Author: Vincent Gardeux (vincent.gardeux@epfl.ch)
##################################################

# Parameters handling
options(echo=TRUE)
args <- commandArgs(trailingOnly = TRUE)

# Libraries
suppressPackageStartupMessages(library(Seurat))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(source("hdf5_lib.R"))

# Arguments
input_loom <- args[1]
input_dataset_path <- args[2]
output_dataset_path <- args[3]
output_dir <- args[4]

#input_loom <- "grrpvn_parsing_output.loom"
#input_dataset_path <- "/matrix"
#output_dataset_path <- "/layers/normalized"

# Parameters
set.seed(42)
data.warnings <- NULL
time_idle <- 0
if(exists('output_dir') & !is.null(output_dir) & !is.na(output_dir)){
  if(!endsWith(output_dir, "/")) output_dir <- paste0(output_dir, "/")
}

# Error case: Loom file does not exist
if(!file.exists(input_loom)) error.json(paste0("This file: '", input_loom, "', does not exist!"))
 
## Open the existing Loom in read-only mode and recuperate the infos (not optimized for OUT-OF-RAM computation)
data.loom <- open_with_lock(input_loom, "r")
data.matrix <- fetch_dataset(data.loom, input_dataset_path, transpose = T) # If run on dimension reduction (like PCA), then do not transpose the matrix
close_all()

# Error case: Path in Loom file does not exist
if(is.null(data.matrix)) error.json(paste0("This file: '", input_loom, "', does not contain any dataset at path '", input_dataset_path, "'!"))

## Create Seurat object
data.seurat <- CreateSeuratObject(counts = data.matrix, min.cells = 0, min.features = 0) # Create our Seurat object using our data matrix (no filtering)

## Save some RAM
rm(data.matrix)
rm(data.loom)

### Normalize the raw matrix
data.seurat <- NormalizeData(data.seurat, normalization.method = "LogNormalize", scale.factor = 10000, verbose = F)

# Extract normalized matrix in a Seurat-version-compatible way.
# Seurat v5 stores normalized values in Assay5 layers (not @data slot).
get_normalized_matrix <- function(seurat_obj, assay_name = "RNA") {
  if(is.null(seurat_obj@assays[[assay_name]])) {
    error.json(paste0("Assay '", assay_name, "' not found in Seurat object"))
  }
  assay_obj <- seurat_obj@assays[[assay_name]]

  if(inherits(assay_obj, "Assay5")) {
    if(!exists("LayerData", mode = "function")) {
      error.json("Seurat Assay5 detected but LayerData() is not available")
    }
    norm <- LayerData(object = seurat_obj, assay = assay_name, layer = "data")
    if(is.null(norm)) {
      error.json("Normalized data layer 'data' is missing in Assay5 object")
    }
    return(as.matrix(norm))
  }

  if("data" %in% slotNames(assay_obj)) {
    return(as.matrix(assay_obj@data))
  }

  error.json(paste0("Unsupported assay class for normalization output: ", class(assay_obj)[1]))
}

# Writing the dataset in the loom file
data.loom <- open_with_lock(input_loom, "r+")
normalized_matrix <- get_normalized_matrix(data.seurat, assay_name = "RNA")
add_matrix_dataset(handle = data.loom, dataset_path = output_dataset_path, dataset_object = t(normalized_matrix))
datasetSize <- get_dataset_size(data.loom, output_dataset_path)
close_all()

# Generate default JSON file
stats <- list()
stats$time_idle = time_idle
stats$nber_rows = nrow(data.seurat)
stats$nber_cols = ncol(data.seurat)
if(!is.null(data.warnings)) stats$warnings = data.warnings
stats$metadata = list(list(name = output_dataset_path, on = "EXPRESSION_MATRIX", type = "NUMERIC", nber_rows = nrow(data.seurat), nber_cols = ncol(data.seurat), dataset_size = datasetSize))
if(exists('output_dir') & !is.null(output_dir) & !is.na(output_dir)){
  cat(toJSON(stats, method="C", auto_unbox=T, digits = NA), file = paste0(output_dir, "output.json"))
} else {
  cat(toJSON(stats, method="C", auto_unbox=T, digits = NA))
}
