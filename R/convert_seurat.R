# Title: Creating ASAP/SCENIC-compatible Loom file from Seurat object
# Author: Vincent Gardeux
# Date: 2024/03/05
# Updated: 2024/11/07

# Libraries & Functions
suppressPackageStartupMessages(library(Seurat)) # For single-cell pipeline
suppressPackageStartupMessages(library(SeuratObject)) # For single-cell pipeline
suppressPackageStartupMessages(library(data.table)) # For writing DE gene file
suppressPackageStartupMessages(library(loomR)) # For building a file for ASAP
suppressPackageStartupMessages(library(Matrix)) # For building a file for ASAP
suppressPackageStartupMessages(library(crayon)) # Just for bolding the console output :D

cat(bold("Seurat"), "version", as.character(packageVersion("Seurat")), "\n")
cat(bold("SeuratObject"), "version", as.character(packageVersion("SeuratObject")), "\n")
cat(bold("data.table"), "version", as.character(packageVersion("data.table")), "\n")
cat(bold("loomR"), "version", as.character(packageVersion("loomR")), "\n")
cat(bold("Matrix"), "version", as.character(packageVersion("Matrix")), "\n")

# Parameters
#seurat_input <- "/home/gardeux/seurat_object_integrated.Rds"
#loom_output <- "/home/gardeux/test.loom"

# Parameters from command line
args <- commandArgs(trailingOnly = TRUE)

# Check if the correct number of arguments are provided
if (length(args) != 2) {
  stop("Usage: Rscript your_script.R <seurat_input> <loom_output>")
}

# Assign command line arguments to variables
seurat_input <- args[1]
loom_output <- args[2]


# Random seed
set.seed(42)

# I.1 Reading Seurat object
message("Loading Seurat object...")
data.seurat <- readRDS(seurat_input)

# Convert the version to a character string and split by "."
version_string <- as.character(data.seurat@version)
real_major_version <- ifelse(class(data.seurat@assays$RNA) == "Assay5", "v5", "v4") # v4 stands for v1-v4
message("Version of Seurat object: ", version_string)
message("Real version of Seurat object: ", real_major_version)

# Main /matrix (I remove colnames and rownames)
if (real_major_version == "v5") {
  cell_names <- rownames(attributes(data.seurat@assays$RNA)$cells@.Data)
  gene_names <- rownames(attributes(data.seurat@assays$RNA)$features@.Data)

  count_matrix <- Matrix(0, nrow = length(gene_names), ncol = length(cell_names), sparse = TRUE)
  rownames(count_matrix) <- gene_names
  colnames(count_matrix) <- cell_names

  layer.names <- names(attributes(data.seurat@assays$RNA)$layers)
  counts.layers <- grep("^counts\\.", layer.names, value = TRUE)
  for (layer in counts.layers) {
    count_matrix_tmp <- attributes(data.seurat@assays$RNA)$layers[[layer]]
    
    cell_names_tmp <- attributes(data.seurat@assays$RNA)$cells@.Data[,layer]
    cell_names_tmp <- names(cell_names_tmp)[cell_names_tmp]
    colnames(count_matrix_tmp) <- cell_names_tmp
    
    gene_names_tmp <- attributes(data.seurat@assays$RNA)$features@.Data[,layer]
    gene_names_tmp <- names(gene_names_tmp)[gene_names_tmp]
    rownames(count_matrix_tmp) <- gene_names_tmp
    
    gene_indices <- match(gene_names_tmp, gene_names)
    cell_indices <- match(cell_names_tmp, cell_names)
  
    count_matrix[gene_indices, cell_indices] <- count_matrix_tmp
    
    meta.matrix <- attributes(data.seurat@assays$RNA)$cells@.Data[,counts.layers]
    split.ident <- unlist(apply(meta.matrix, 1, function(row) {
      if (sum(row) == 1) {
        return(gsub(x = colnames(meta.matrix)[which(row)], pattern = "counts.", replacement = ""))
      } else {
        stop("Error: More than one TRUE value in a row.", which(row))
      }
    }))
    data.seurat[["split.ident"]] <- split.ident
  }
} else {
  count_matrix <- data.seurat@assays$RNA@counts
}

colnames(count_matrix) <- NULL
rownames(count_matrix) <- NULL

# Prepare Cell attributes
if (real_major_version == "v5") {
  attributes_cells <- list(CellID = rownames(attributes(data.seurat@assays$RNA)$cells@.Data))
} else {
  attributes_cells <- list(CellID = data.seurat@assays$RNA@counts@Dimnames[[2]])
}

for (m in names(data.seurat@meta.data)) {
  attributes_cells[[m]] <- data.seurat@meta.data[m]
}

# Prepare Gene names
if (real_major_version == "v5") {
  gene_names <- list(Gene = rownames(attributes(data.seurat@assays$RNA)$features@.Data))
} else {
  gene_names <- list(Gene = data.seurat@assays$RNA@counts@Dimnames[[1]])
}

# Create Loom file
data.loom <- create(
  filename = loom_output, 
  data = count_matrix, 
  gene.attrs = gene_names, 
  cell.attrs = attributes_cells, 
  overwrite = TRUE
)

# Adding NN/SNN graphs
for (graph in names(data.seurat@graphs)) {
  b_vec <- rep(0, length(data.seurat@graphs[[graph]]@i))
  val <- 0
  ind <- 0
  for (i in 2:length(data.seurat@graphs[[graph]]@p)) {
    b_vec[(ind + 1):data.seurat@graphs[[graph]]@p[i]] <- val
    val <- val + 1
    ind <- data.seurat@graphs[[graph]]@p[i]
  }
  data.loom$add.graph(name = graph, a = data.seurat@graphs[[graph]]@i, b = b_vec, w = data.seurat@graphs[[graph]]@x)
}

# Adding Layers
if (real_major_version == "v5") {
  cell_names <- rownames(attributes(data.seurat@assays$RNA)$cells@.Data)
  gene_names <- rownames(attributes(data.seurat@assays$RNA)$features@.Data)

  normalized_matrix <- Matrix(0, nrow = length(gene_names), ncol = length(cell_names), sparse = TRUE)
  rownames(normalized_matrix) <- gene_names
  colnames(normalized_matrix) <- cell_names

  layer.names <- names(attributes(data.seurat@assays$RNA)$layers)
  counts.layers <- grep("^counts\\.", layer.names, value = TRUE)
  for (layer in counts.layers) {
    normalized_matrix_tmp <- attributes(data.seurat@assays$RNA)$layers[[layer]]
    
    cell_names_tmp <- attributes(data.seurat@assays$RNA)$cells@.Data[,layer]
    cell_names_tmp <- names(cell_names_tmp)[cell_names_tmp]
    colnames(normalized_matrix_tmp) <- cell_names_tmp
    
    gene_names_tmp <- attributes(data.seurat@assays$RNA)$features@.Data[,layer]
    gene_names_tmp <- names(gene_names_tmp)[gene_names_tmp]
    rownames(normalized_matrix_tmp) <- gene_names_tmp
    
    gene_indices <- match(gene_names_tmp, gene_names)
    cell_indices <- match(cell_names_tmp, cell_names)
  
    normalized_matrix[gene_indices, cell_indices] <- normalized_matrix_tmp
  }
} else {
  normalized_matrix <- data.seurat@assays$RNA@data
}

if (!is.null(normalized_matrix)) {
  colnames(normalized_matrix) <- NULL
  rownames(normalized_matrix) <- NULL
  data.loom$add.layer(layers = list(normalized = normalized_matrix))
}

# Add Embeddings
for (embedding in names(data.seurat@reductions)) {
  data.loom$add.col.attribute(setNames(list(data.seurat@reductions[[embedding]]@cell.embeddings), embedding), overwrite = TRUE)
}

# Close Loom file
data.loom$close_all()