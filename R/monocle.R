options(echo = TRUE)
args <- commandArgs(trailingOnly = TRUE)

library(monocle)
library(reshape2)
library(igraph)
library(jsonlite)
library(scales)

plot_type <- 'scattergl'
branch_node_color <- '#000000'
selected_branch_node_color <- '#FFFFFF'
cell_fate_dash <- c('solid', 'dot')
plotly_default_colors <- c(
  '#1f77b4',  # muted blue
  '#ff7f0e',  # safety orange
  '#2ca02c',  # cooked asparagus green
  '#d62728',  # brick red
  '#9467bd',  # muted purple
  '#8c564b',  # chestnut brown
  '#e377c2',  # raspberry yogurt pink
  '#7f7f7f',  # middle gray
  '#bcbd22',  # curry yellow-green
  '#17becf'   # blue-teal
);

GM_state <- function(cds){
  print(pData(cds)$State)
  print(pData(cds)$timeseries)
  if (length(unique(pData(cds)$State)) > 1 && !is.null(pData(cds)$timeseries)) {
    mt <- min(pData(cds)$timeseries)
    tab <- table(pData(cds)$State, pData(cds)$timeseries)
    t0_counts <- table(pData(cds)$State, pData(cds)$timeseries)[,toString(min(pData(cds)$timeseries))]
    return(as.numeric(names(t0_counts)[which(t0_counts == max(t0_counts))]))
  }else {
    return (1)
  }
}

genes_branched_pseudotime <- function(genes_plot, branch_number, gene_name, index) {
  branches <- levels(genes_plot$data$Branch)

  gene_data <- genes_plot$data[genes_plot$data$gene_short_name == gene_name,]
  ordered_gene_data <- gene_data[with(gene_data, order(Pseudotime)),]
  unique_states <- levels(ordered_gene_data$State)

  gene_traces <- vector("list", length(branches) + length(unique_states))

  for(i in 1:length(branches)) {
    ordered_branch_data <- ordered_gene_data[ordered_gene_data$Branch==branches[i],]
    gene_traces[[i]] <- list(
      x = ordered_branch_data$Pseudotime,
      y = ordered_branch_data$full_model_expectation,
      xaxis = unbox('x'),
      yaxis = unbox(paste0('y', index)),
      type = unbox(plot_type),
      mode = unbox('lines'),
      name = unbox(branches[i]),
      line = list(dash = unbox(cell_fate_dash[i]), color = unbox('#000000')),
      legendgroup = unbox(branches[i]),
      showlegend = unbox(FALSE)
    )
  }

  for(i in 1:length(unique_states)) {
    trace_i <- length(branches) + i
    state_indices <- ordered_gene_data$State == unique_states[i]
    gene_traces[[trace_i]] <- list(
      x = ordered_gene_data$Pseudotime[state_indices],
      y = ordered_gene_data$expression[state_indices],
      xaxis = unbox('x'),
      yaxis = unbox(paste0('y', index)),
      type = unbox(plot_type),
      mode = unbox('markers'),
      name = unbox(paste('State', unique_states[i])),
      marker = list(color = unbox(plotly_default_colors[as.numeric(unique_states[i])])),
      # text = unbox(unique_states[i]),
      legendgroup = unbox(paste('State', unique_states[i])),
      showlegend = unbox(FALSE)
    )
  }

  writeLines(toJSON(gene_traces), paste(output_dir, paste0(paste('monocle', 'geneexpression', 'branch', branch_number, gene_name, sep='_'), '.json'), sep='/'))
}

gene_expression_analysis <- function(cds, branch_number, gene_names) {
  genes_plot <- plot_genes_branched_pseudotime(cds[gene_names,],
                                               branch_point = branch_number,
                                               color_by = "State",
                                               ncol = 1)
  for(i in 1:length(gene_names)) {
    genes_branched_pseudotime(genes_plot, branch_number, gene_names[[i]], i)
  }
}

branch_heatmap <- function(cds, branch_number, num_clusters) {
  BEAM_res <- BEAM(cds, branch_point = branch_number, cores = 1)
  BEAM_res <- BEAM_res[order(BEAM_res$qval),]
  BEAM_res <- BEAM_res[,c("gene_short_name", "pval", "qval")]
  genes_branched_heatmap <- plot_genes_branched_heatmap(cds[row.names(BEAM_res),],
                                                        branch_point = branch_number,
                                                        num_clusters = num_clusters,
                                                        cores = 1,
                                                        use_gene_short_name = TRUE,
                                                        show_rownames = TRUE,
                                                        return_heatmap = TRUE)
  
  annotation_row <- genes_branched_heatmap$annotation_row
  annotation_order <- genes_branched_heatmap$ph$tree_row$order
  heatmap_matrix <- genes_branched_heatmap$heatmap_matrix[annotation_order,]
  gene_names <- rownames(heatmap_matrix)
  annotation_reverse_order <- match(rownames(annotation_row), gene_names)
  merge_indicies <- genes_branched_heatmap$ph$tree_row$merge
  heatmap_genes_cluster <- annotation_row[annotation_order,]
  
  #Change indicies to plotting order
  merge_indicies[merge_indicies<0] <- -annotation_reverse_order[-merge_indicies[merge_indicies<0]]
  
  parent <- list()
  count <- list()
  
  for(i in 1:length(gene_names)) {
    count[[toString(-i)]] <- 1
  }
  
  for(i in 1:dim(merge_indicies)[1]) {
    parent[[toString(merge_indicies[i,1])]] <- i
    parent[[toString(merge_indicies[i,2])]] <- i
    count[[toString(i)]] <- (count[[toString(merge_indicies[i,1])]] + count[[toString(merge_indicies[i,2])]])
  }
  
  nodes <- list()
  metadata_nodes <- list()
  
  for(i in 1:length(gene_names)) {
    index <- toString(-i)
    nodes[[index]] = list(
      count = unbox(count[[index]]),
      distance = unbox(0),
      objects = gene_names[i],
      features = heatmap_matrix[i,],
      parent = unbox(parent[[index]])
    )
    metadata_nodes[[index]] <- heatmap_genes_cluster[i]
  }
  
  for(i in 1:dim(merge_indicies)[1]) {
    index <- toString(i)
    nodes[[index]] = list(
      count = unbox(count[[index]]),
      distance = unbox(genes_branched_heatmap$ph$tree_row$height[i]),
      left_child = unbox(merge_indicies[i,1]),
      right_child = unbox(merge_indicies[i,2])
    )
    if (!is.null(parent[[index]])) {
      nodes[[index]]$parent <- unbox(parent[[index]])
    }
  }
  
  heatmap <- list(
    data=list(nodes=nodes),
    column_metadata=list(
      features=list(genes_branched_heatmap$annotation_col[,1])
    ),
    metadata=list(nodes=metadata_nodes, feature_names=c('Cluster'))
  )
  
  writeLines(toJSON(heatmap), paste(output_dir, paste('monocle_heatmap_branch_',branch_number,'.json',sep=''), sep='/'))
  
  return(list(gene_names=gene_names))
}

trajectory_analysis <- function(expr_matrix, reduction_method, timeseries) {
  gene_annotation <- data.frame(rownames(expr_matrix), rownames(expr_matrix), as.numeric(rowSums(expr_matrix > 0)), row.names = 1)
  colnames(gene_annotation) <- c('gene_short_name', 'num_cells_expressed')
  
  if (!is.null(timeseries)) {
    print('TIME-SERIES ADDED')
    print(colnames(expr_matrix)[1:5])
    print(rownames(timeseries)[1:5])
    timeseries <- timeseries[match(colnames(expr_matrix), rownames(timeseries)), ]
    sample_sheet <- data.frame(colnames(expr_matrix), colnames(expr_matrix), timeseries, row.names = 1)
    colnames(sample_sheet) <- c('cell_name', 'timeseries')
  } else {
    print('NO TIME-SERIES')
    sample_sheet <- data.frame(colnames(expr_matrix), colnames(expr_matrix), row.names = 1)
    #sample_sheet <- data.frame(colnames(expr_matrix))
    colnames(sample_sheet) <- c('cell_name')
  }

  pd <- new("AnnotatedDataFrame", data = sample_sheet)
  fd <- new("AnnotatedDataFrame", data = gene_annotation)
  
  # first create a celldataset from the relative expression levels
  cds <- newCellDataSet(as.matrix(expr_matrix),
                        phenoData = pd,
                        featureData = fd,
                        lowerDetectionLimit = 0.5,
                        expressionFamily = negbinomial.size())
                        #expressionFamily = tobit())
 
  print("cds")
  cds <- estimateSizeFactors(cds)
  print("est size fact")
  cds <- estimateDispersions(cds)
  print("est disp")
  
  #REMOVE: ordering_genes should be an input
  #disp_table <- dispersionTable(cds)
  #ordering_genes <- subset(disp_table, mean_expression >= 0.5 & dispersion_empirical >= 1 * dispersion_fit)$gene_id
  
  #All genes are ordering genes (genes are filtered before the R script)
  ordering_genes <- rownames(cds)
  
  #STEP 1
  cds <- setOrderingFilter(cds, ordering_genes)
  plot_ordering_genes(cds)
  
  #STEP 2
  #Parralelizable
  cds <- reduceDimension(cds, max_components = 2, reduction_method = reduction_method)
  
  #STEP 3
  #Parallelizable
  cds <- orderCells(cds)
  
  plot_cell_trajectory(cds, color_by = "State", show_backbone = t)
  cds <- orderCells(cds, root_state = GM_state(cds))
  plot_cell_trajectory(cds, color_by = "Pseudotime", show_backbone = t) 
  
  #SAVE TO JSON FILES
  reduced_dim_coords <- reducedDimS(cds)
  
  if (reduction_method == 'ICA') {
    mst_reduced_dim_coords <- reducedDimS(cds)
  } else {
    mst_reduced_dim_coords <- reducedDimK(cds)
  }
  
  mst <- minSpanningTree(cds)
  vertices_names <- get.vertex.attribute(mst)$name
  edge_list = get.edgelist(mst)
  source_pts <- mst_reduced_dim_coords[,edge_list[,1]]
  target_pts <- mst_reduced_dim_coords[,edge_list[,2]]
  
  mst_branch_nodes <- cds@auxOrderingData[[cds@dim_reduce_type]]$branch_points
  mst_branch_nodes_coords = mst_reduced_dim_coords[,mst_branch_nodes, drop = FALSE]
  
  # trajectory = list(
  #   reduced_dim_coords = rbind(colnames(reduced_dim_coords), reduced_dim_coords),
  #   state = cds@phenoData$State,
  #   pseudotime = cds@phenoData$Pseudotime,
  #   vertices = rbind(vertices_names, mst_reduced_dim_coords[,vertices_names]),
  #   source_pts = source_pts,
  #   target_pts = target_pts,
  #   branch_nodes = rbind(mst_branch_nodes, mst_branch_nodes_coords)
  # )
  # writeLines(toJSON(trajectory), paste(output_dir, 'monocle_trajectory.json', sep='/'))
  
  state = cds@phenoData$State
  cell_names = colnames(reduced_dim_coords)
  unique_states = unique(state)
  unique_states = unique_states[order(unique_states)] 

  # state_traces <- vector("list", length(unique_states))
  state_traces <- rep(list(list(x=c(), y=c(), mode=unbox("markers"), type=unbox(plot_type), name=unbox("State"), text=c(), marker=list(size=unbox("")))), length(unique_states))
  for (i in 1:length(unique_states)) {
    state_traces[[i]]$name <- unbox(paste(state_traces[[i]]$name, unique_states[i]))
    state_subset <- state==unique_states[i]
    state_traces[[i]]$x <- reduced_dim_coords[1,state_subset]
    state_traces[[i]]$y <- reduced_dim_coords[2,state_subset]
    state_traces[[i]]$text <- colnames(reduced_dim_coords[,state_subset])
  }
  
  branch_trace <- list(list(
    x = mst_branch_nodes_coords[1,],
    y = mst_branch_nodes_coords[2,],
    mode = unbox("markers"),
    type = unbox(plot_type),
    name = unbox("Branch nodes"),
    # text = ifelse(length(mst_branch_nodes) == 0, vector(), paste("Branch nodes", paste0("[", mst_branch_nodes, "]"))),
    marker = list(size = unbox(""), color = c(selected_branch_node_color, rep(branch_node_color, max(length(mst_branch_nodes)-1, 1))), line = list(color = unbox(branch_node_color), width = unbox(2)))
  ))
  if (length(mst_branch_nodes) > 0) {
    branch_trace[[1]][["text"]] <- paste("Branch nodes", paste0("[", mst_branch_nodes, "]"))
  }

  #Make the branch lines in order to plot in one trace
  edges <- vector("list", length(vertices_names))
  source_names <- colnames(source_pts)
  target_names <- colnames(target_pts)
  source_target_names <- c(source_names, target_names)
  source_target_pts <- cbind(source_pts, target_pts)
  for(i in 1:dim(source_pts)[2]) {
    from_idx <- match(source_names[i], vertices_names)
    to_idx <- match(target_names[i], vertices_names)
    edges[[from_idx]] <- c(edges[[from_idx]], to_idx)
    edges[[to_idx]] <- c(edges[[to_idx]], from_idx)
  }

  start_idx <- which(sapply(edges, length) == 1)
  visited <- numeric(length(vertices_names))
  dfs_idx <- c()
  loop_idx <- c(start_idx, 1:length(vertices_names))
  tot <- 0
  tree_segs <- list()
  for(i in 1:length(loop_idx)) {
    if(visited[loop_idx[i]] == 0) {
      cur <- loop_idx[i]
      dfs_idx <- c(dfs_idx, cur)
      while(visited[cur] == 0) {
        visited[cur] <- i
        flag <- FALSE
        for(j in 1:length(edges[[cur]])) {
          if(visited[edges[[cur]][j]] == 0) {
            cur <- edges[[cur]][j];
            dfs_idx <- c(dfs_idx, cur)
            flag <- TRUE
            break
          }
        }
        if(!flag) {
          for(j in 1:length(edges[[cur]])) {
            if (visited[edges[[cur]][j]] != i) {
              cur <- edges[[cur]][j];
              dfs_idx <- c(dfs_idx, cur)
              break
            }
          }
        }
      }
    }
    if(length(dfs_idx) > 0) {
      tree_segs[[length(tree_segs)+1]] <- list(
        #x=c(source_pts[1, match(vertices_names[dfs_idx], source_names)], target_pts[1, match(vertices_names[dfs_idx], target_names)]),
        #y=c(source_pts[2, match(vertices_names[dfs_idx], source_names)], target_pts[2, match(vertices_names[dfs_idx], target_names)]),
        x=source_target_pts[1, match(vertices_names[dfs_idx], source_target_names)],
        y=source_target_pts[2, match(vertices_names[dfs_idx], source_target_names)],
        mode=unbox("lines"),
        type=unbox(plot_type),
        line=list(color=unbox("Black")),
        showlegend=unbox(FALSE),
        hoverinfo=unbox("skip")
      )
    }
    tot <- tot + length(dfs_idx)
    dfs_idx <- c()
  }

  tree_trace <- list(list(
    x=c(source_pts[1,], target_pts[1,]),
    y=c(source_pts[2,], target_pts[2,]),
    mode=unbox("lines"),
    type=unbox(plot_type),
    line=list(color=unbox("Black")),
    showlegend=unbox(FALSE),
    hoverinfo=unbox("skip")
  ))

  tree_traces <- vector("list", dim(source_pts)[2])
  for(i in 1:dim(source_pts)[2]) {
    tree_traces[[i]] <- list(
      x=c(source_pts[1,i], target_pts[1,i]),
      y=c(source_pts[2,i], target_pts[2,i]),
      mode=unbox("lines"),
      type=unbox(plot_type),
      line=list(color=unbox("Black")),
      showlegend=unbox(FALSE),
      hoverinfo=unbox("skip")
    )
  }

  trajectory_data_state <- c(state_traces, tree_segs, branch_trace)
  
  pseudotime_trace <- list(list(x=reduced_dim_coords[1,], y=reduced_dim_coords[2,], mode=unbox("markers"), type=unbox(plot_type), name=unbox("Cells"), text=colnames(reduced_dim_coords), marker=list(size=unbox(""), color=cds@phenoData$Pseudotime, colorbar=list(title="Colorbar"), colorscale=unbox("Bluered"))))
  trajectory_data_pseudotime <- c(pseudotime_trace, tree_traces, branch_trace)

  if (!is.null(timeseries)) {
    timeseries <- cds@phenoData$timeseries
    unique_timeseries <- unique(timeseries)
    unique_timeseries <- unique_timeseries[order(unique_timeseries)] 

    timeseries_traces <- rep(list(list(x=c(), y=c(), mode=unbox("markers"), type=unbox(plot_type), name=unbox("Time-series"), text=c(), marker=list(size=unbox("")))), length(unique_timeseries))
    for (i in 1:length(unique_timeseries)) {
      timeseries_traces[[i]]$name <- unbox(paste(timeseries_traces[[i]]$name, unique_timeseries[i]))
      timeseries_subset <- timeseries==unique_timeseries[i]
      timeseries_traces[[i]]$x <- reduced_dim_coords[1,timeseries_subset]
      timeseries_traces[[i]]$y <- reduced_dim_coords[2,timeseries_subset]
      timeseries_traces[[i]]$text <- cell_names[timeseries_subset]
      #timeseries_traces[[i]]$text <- colnames(reduced_dim_coords[,timeseries_subset])
    }
    trajectory_data_timeseries <- c(timeseries_traces, tree_segs, branch_trace)

    trajectory_data <- list('state'=trajectory_data_state, 'pseudotime'=trajectory_data_pseudotime, 'timeseries'=trajectory_data_timeseries)
  } else {
    trajectory_data <- list('state'=trajectory_data_state, 'pseudotime'=trajectory_data_pseudotime) 
  }
  writeLines(toJSON(trajectory_data), paste(output_dir, 'monocle_trajectory.json', sep='/'))
  return(list(cds=cds[ordering_genes,], branch_nodes_size=length(mst_branch_nodes)))
}

branch_analysis <- function(cds, branch_nodes_size, num_clusters = 4) {
  print(paste('Number of branches:', branch_nodes_size))
  if (branch_nodes_size <= 0) {
    return(list())
  }
  gene_names_per_branch <- vector("list", branch_nodes_size)
  for (i in 1:branch_nodes_size) {
    heatmap <- branch_heatmap(cds, i, num_clusters)
    gene_names_per_branch[[i]] <- heatmap$gene_names
  }
  return(list(gene_names_per_branch=gene_names_per_branch))
}

gene_analysis <- function(cds, branch_nodes_size, gene_names_per_branch) {
  if (branch_nodes_size <= 0) {
    return(list())
  }
  for (i in 1:branch_nodes_size) {
    gene_expression_analysis(cds, i, gene_names_per_branch[[i]])
  }
}

input_file <- args[1]
output_dir <- args[2]
reduction_method <- args[3]
timeseries_file <- args[4] 
expr_matrix <- read.table(input_file, header = TRUE, row.names = 1)
timeseries <- if (file.exists(timeseries_file)) read.table(timeseries_file, header = FALSE, row.names = 1) else NULL

cat("EXPR_MATRIX:", dim(expr_matrix), "\n")
trajectory <- trajectory_analysis(expr_matrix, reduction_method, timeseries)
branch <- branch_analysis(trajectory$cds, trajectory$branch_nodes_size)
gene_analysis(trajectory$cds, trajectory$branch_nodes_size, branch$gene_names_per_branch)
writeLines(toJSON(NULL), paste(output_dir, 'output.json', sep='/'))
