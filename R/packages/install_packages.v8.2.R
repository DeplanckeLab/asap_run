# Latest version made 2026-06-22

# Compile options
if (!nzchar(Sys.getenv("MAKEFLAGS"))) Sys.setenv(MAKEFLAGS = "-j2")

# Functions
install_and_verify <- function(pkgfile, package) {
  install.packages(pkgfile, repos = NULL, type = "source", verbose = FALSE, quiet = T)
  if (!requireNamespace(package, quietly = TRUE)) {
	message("Package ",package," failed installation")
	quit(status = 1)
  }
}

install.cran <- function(package, version, host="cran.r-project.org") {
  main_url <- paste0("https://", host, "/src/contrib/", package, "_", version, ".tar.gz")
  archive_url <- paste0("https://", host, "/src/contrib/Archive/", package, "/", package, "_", version, ".tar.gz")
 
  tmp_file <- tempfile(fileext = ".tar.gz")

  tryCatch({
      message("Installing ", package," v", version," from main CRAN repository...", appendLF = FALSE)
      suppressWarnings(download.file(main_url, tmp_file, mode = "wb", quiet = TRUE))
      install_and_verify(tmp_file, package)
      unlink(tmp_file)
    }, error = function(e) {
      message("\t[FAILED - RETRY] ", e$message)
      message("Trying to install ", package," v", version," from CRAN archive...", appendLF = FALSE)
      tryCatch({
          suppressWarnings(download.file(archive_url, tmp_file, mode = "wb", quiet = TRUE))
          install_and_verify(tmp_file, package)
          unlink(tmp_file)
        }, error = function(e) {
          message("\t[FAILED - EXIT] ", e$message)
		  message("Both installation attempts of ", package," v", version, " failed. Stopping installation.")
		  quit(status = 1)
        })
    })
	message("\t[SUCCESS]")
}

install.bioconductor <- function(package, version, bioconductor_version, source = "bioc") {
  main_url      <- paste0("https://www.bioconductor.org/packages/release/", source, "/src/contrib/", package, "_", version, ".tar.gz")
  archive_url 	<- paste0("https://www.bioconductor.org/packages/", bioconductor_version, "/", source, "/src/contrib/", package, "_", version, ".tar.gz")
  archive_url_2	<- paste0("https://www.bioconductor.org/packages/", bioconductor_version, "/", source, "/src/contrib/Archive/", package, "/", package, "_", version, ".tar.gz")
  
  tmp_file <- tempfile(fileext = ".tar.gz")
  tryCatch({
      message("Installing ", package, " v", version, " from Bioconductor archive...", appendLF = FALSE)
      suppressWarnings(download.file(archive_url, tmp_file, mode = "wb", quiet = TRUE))
      install_and_verify(tmp_file, package)
      unlink(tmp_file)
    }, error = function(e) {
      message("\t[FAILED - RETRY] ", e$message)
      message("Trying to install ", package, " v", version, " from main Bioconductor repository...", appendLF = FALSE)
      tryCatch({
          suppressWarnings(download.file(main_url, tmp_file, mode = "wb", quiet = TRUE))
          install_and_verify(tmp_file, package)
          unlink(tmp_file)
        }, error = function(e) {
          message("\t[FAILED - RETRY] ", e$message)
          message("Trying to install ", package, " v", version, " from Bioconductor archive 2...", appendLF = FALSE)
          tryCatch({
              suppressWarnings(download.file(archive_url_2, tmp_file, mode = "wb", quiet = TRUE))
              install_and_verify(tmp_file, package)
              unlink(tmp_file)
            }, error = function(e) {
              message("\t[FAILED - EXIT] ", e$message)
              message("All three installation attempts of ", package, " v", version, " failed. Stopping installation.")
              quit(status = 1)
            })
        })
    })
  message("\t[SUCCESS]")
}

install.github <- function(package, version, bioconductor_version, source = "bioc") { # Suppose it's already downloaded (for safety)
  tmp_file <- paste0("/tmp/", package, "_", version, ".tar.gz") # Copied to /tmp/ by Dockerfile COPY prior running this script

  tryCatch({
      message("Installing ", package, " v", version, " from local GitHub tarball...", appendLF = FALSE)
      install_and_verify(tmp_file, package)
    }, error = function(e) {
      message("\t[FAILED - EXIT] ", e$message)
      message("Installation of ", package, " v", version, " from local tarball failed. Stopping installation.")
      quit(status = 1)
    })
  message("\t[SUCCESS]")
}

# CRAN packages
install.cran("cli", version="3.6.6") #UPDATED
install.cran("R6", version="2.6.1") #OK
install.cran("desc", version="1.4.3") #OK
install.cran("otel", version="0.2.0") #OK
install.cran("ps", version="1.9.3") #UPDATED
install.cran("processx", version="3.9.0") #UPDATED
install.cran("callr", version="3.8.0") #UPDATED
install.cran("rlang", version="1.2.0") #UPDATED
install.cran("glue", version="1.8.1") #UPDATED
install.cran("lifecycle", version="1.0.5") #UPDATED
install.cran("rprojroot", version="2.1.1") #UPDATED
install.cran("fs", version="2.1.0") #UPDATED, now requires libuv1-dev
install.cran("utf8", version="1.2.6") #UPDATED
install.cran("generics", version="0.1.4") #UPDATED
install.cran("numDeriv", version="2016.8-1.1") #OK
install.cran("backports", version="1.5.1") #UPDATED
install.cran("abind", version="1.4-8") #OK
install.cran("tensorA", version="0.36.2.1") #OK
install.cran("vctrs", version="0.7.3") #UPDATED
install.cran("pillar", version="1.11.1") #UPDATED
install.cran("distributional", version="0.7.1") #UPDATED
install.cran("cpp11", version="0.5.5") #UPDATED
install.cran("farver", version="2.1.2") #OK
install.cran("RColorBrewer", version="1.1-3") #OK
install.cran("viridisLite", version="0.4.3") #UPDATED
install.cran("gtable", version="0.3.6") #OK
install.cran("checkmate", version="2.3.4") #UPDATED
install.cran("matrixStats", version="1.5.0") #OK
install.cran("magrittr", version="2.0.5") #UPDATED
install.cran("isoband", version="0.3.0") #UPDATED # Dependency of ggplot2
install.cran("S7", version="0.2.2") #OK # Dependency of ggplot2
install.cran("inline", version="0.3.21") #OK
install.cran("gridExtra", version="2.3") #OK
install.cran("QuickJSR", version="1.10.0") #UPDATED
install.cran("jsonlite", version="2.0.0") #UPDATED
install.cran("curl", version="7.1.0") #UPDATED
install.cran("sys", version="3.4.3") #OK
install.cran("askpass", version="1.2.1") #OK
install.cran("openssl", version="2.4.2") #UPDATED
install.cran("purrr", version="1.2.2") #UPDATED
install.cran("pkgconfig", version="2.0.3") #OK
install.cran("tibble", version="3.3.1") #UPDATED
install.cran("whisker", version="0.4.1") #OK
install.cran("withr", version="3.0.3") #OK
install.cran("xml2", version="1.5.2") #UPDATED
install.cran("yaml", version="2.3.12") #UPDATED
install.cran("posterior", version="1.7.0") #UPDATED
install.cran("loo", version="2.9.0") #UPDATED
install.cran("labeling", version="0.4.3") #UPDATED
install.cran("scales", version="1.4.0") #UPDATED
install.cran("ggplot2", version="4.0.3") #UPDATED
install.cran("BH", version="1.90.0-1") #UPDATED
install.cran("digest", version="0.6.39") #UPDATED
install.cran("fastmap", version="1.2.0") #OK
install.cran("cachem", version="1.1.0") #OK
install.cran("memoise", version="2.0.1") #OK
install.cran("base64enc", version="0.1-6") #UPDATED
install.cran("htmltools", version="0.5.9") #UPDATED
install.cran("lazyeval", version="0.2.3") #UPDATED
install.cran("fansi", version="1.0.7") #UPDATED
install.cran("fontawesome", version="0.5.3") #OK
install.cran("rappdirs", version="0.3.4") #UPDATED
install.cran("sass", version="0.4.10") #UPDATED
install.cran("mime", version="0.13") #UPDATED
install.cran("httr2", version="1.2.2") #UPDATED
install.cran("systemfonts", version="1.3.2") #UPDATED
install.cran("stringi", version="1.8.7") #UPDATED
install.cran("stringr", version="1.6.0") #UPDATED
install.cran("textshaping", version="1.0.5") #UPDATED
install.cran("ragg", version="1.5.2") #UPDATED
install.cran("jquerylib", version="0.1.4") #OK
install.cran("bslib", version="0.11.0") #UPDATED
install.cran("xfun", version="0.59") #UPDATED
install.cran("highr", version="0.12") #UPDATED
install.cran("evaluate", version="1.0.5") #UPDATED
install.cran("knitr", version="1.51") #UPDATED
install.cran("tinytex", version="0.60") #UPDATED
install.cran("rmarkdown", version="2.31") #UPDATED
install.cran("brio", version="1.1.5") #OK
install.cran("praise", version="1.0.0") #OK
install.cran("downlit", version="0.4.5") #UPDATED
install.cran("pkgbuild", version="1.4.8") #UPDATED
install.cran("pkgload", version="1.5.3") #UPDATED
install.cran("pkgdown", version="2.2.0") #UPDATED
install.cran("Rcpp", version="1.1.1-1.1") #UPDATED
install.cran("RcppArmadillo", version="15.4.0-1") #UPDATED
install.cran("RcppParallel", version="5.1.11-2") #UPDATED
install.cran("RcppEigen", version="0.3.4.0.2") #OK
install.cran("RcppML", version="0.3.7.1") #UPDATED
install.cran("RcppAnnoy", version="0.0.23") #UPDATED
install.cran("RcppTOML", version="0.2.3") #OK
install.cran("RcppHNSW", version="0.7.0") #UPDATED
install.cran("RcppNumerical", version="0.7-0") #OK
install.cran("RcppProgress", version="0.4.2") #OK
install.cran("StanHeaders", version="2.32.10") #OK
install.cran("rstan", version="2.32.7") #OK
install.cran("rstantools", version="2.6.0") #UPDATED
install.cran("bitops", version="1.0-9") #OK
install.cran("Rtsne", version="0.17") #OK
install.cran("MASS", version="7.3-65") #OK
install.cran("colorspace", version="2.1-2") #UPDATED
install.cran("munsell", version="0.5.1") #OK
install.cran("plyr", version="1.8.9") #OK
install.cran("reshape2", version="1.4.5") #UPDATED
install.cran("plogr", version="0.2.0") #OK # Now Archived
install.cran("pheatmap", version="1.0.13") #UPDATED
install.cran("viridis", version="0.6.5") #OK
install.cran("htmlwidgets", version="1.6.4") #OK
install.cran("png", version="0.1-9") #UPDATED
install.cran("modeltools", version="0.2-24") #UPDATED
install.cran("flexmix", version="2.3-20") #OK
install.cran("mvtnorm", version="1.4-1") #UPDATED
install.cran("DEoptimR", version="1.2-0") #UPDATED
install.cran("robustbase", version="0.99-7") #UPDATED
install.cran("mgcv", version="1.9-4") #UPDATED
install.cran("nlme", version="3.1-169") #UPDATED
install.cran("iterators", version="1.0.14") #OK
install.cran("foreach", version="1.5.2") #OK
install.cran("lattice", version="0.22-9") #UPDATED
install.cran("irlba", version="2.3.7") #UPDATED
install.cran("igraph", version="2.3.2") #UPDATED
install.cran("cluster", version="2.1.8.2") #UPDATED
install.cran("data.table", version="1.18.4") #UPDATED
install.cran("Formula", version="1.2-5") #OK
install.cran("rstudioapi", version="0.19.0") #UPDATED
install.cran("htmlTable", version="2.5.0") #UPDATED
install.cran("Hmisc", version="5.2-6") #UPDATED
install.cran("later", version="1.4.8") #UPDATED
install.cran("promises", version="1.5.0") #UPDATED
install.cran("httpuv", version="1.6.17") #UPDATED
install.cran("xtable", version="1.8-8") #UPDATED
install.cran("sourcetools", version="0.1.7-2") #UPDATED
install.cran("crayon", version="1.5.3") #OK
install.cran("commonmark", version="2.0.0") #UPDATED
install.cran("shiny", version="1.14.0") #UPDATED
install.cran("gtools", version="3.9.5") #OK
install.cran("doParallel", version="1.0.17") #OK
install.cran("caTools", version="1.18.3") #OK
install.cran("gplots", version="3.3.0") #UPDATED
install.cran("ROCR", version="1.0-12") #UPDATED
install.cran("blob", version="1.3.0") #UPDATED
install.cran("WriteXLS", version="6.8.0") #UPDATED
install.cran("brew", version="1.0-10") #OK
install.cran("survival", version="3.8-6") #UPDATED
install.cran("DBI", version="1.3.0") #UPDATED
install.cran("rngtools", version="1.5.2") #OK
install.cran("foreign", version="0.8-91") #UPDATED
install.cran("bit", version="4.6.0") #OK
install.cran("XML", version="3.99-0.23") #UPDATED
install.cran("nnet", version="7.3-20") #OK
install.cran("locfit", version="1.5-9.12") #OK
install.cran("bit64", version="4.8.2") #UPDATED
install.cran("RSQLite", version="3.53.2") #UPDATED
install.cran("doRNG", version="1.8.6.3") #UPDATED
install.cran("beeswarm", version="0.4.0") #OK
install.cran("proxy", version="0.4-29") #UPDATED
install.cran("e1071", version="1.7-17") #UPDATED
install.cran("statmod", version="1.5.2") #UPDATED
install.cran("pcaPP", version="2.0-5") #OK
install.cran("Matrix", version="1.7-5") #UPDATED
install.cran("rrcov", version="1.7-7") #UPDATED
install.cran("KernSmooth", version="2.23-26") #OK
install.cran("vipor", version="0.4.7") #OK
install.cran("codetools", version="0.2-20") #OK
install.cran("rpart", version="4.1.27") #UPDATED
install.cran("class", version="7.3-23") #OK
install.cran("Cairo", version="1.7-0") #UPDATED
install.cran("ggbeeswarm", version="0.7.3") #UPDATED
install.cran("httr", version="1.4.8") #UPDATED
install.cran("tidyselect", version="1.2.1") #OK
install.cran("dplyr", version="1.2.1") #UPDATED
install.cran("ggrepel", version="0.9.8") #UPDATED
install.cran("clipr", version="0.8.1") #UPDATED
install.cran("credentials", version="2.0.3") #UPDATED
install.cran("zip", version="3.0.0") #UPDATED
install.cran("gitcreds", version="0.1.2") #OK
install.cran("ini", version="0.3.1") #OK
install.cran("gh", version="1.6.0") #UPDATED
install.cran("gert", version="2.3.1") #UPDATED
install.cran("usethis", version="3.2.1") #UPDATED
install.cran("ellipsis", version="0.3.3") #UPDATED
install.cran("miniUI", version="0.1.2") #UPDATED
install.cran("profvis", version="0.4.0") #OK
install.cran("prettyunits", version="1.2.0") #OK
install.cran("sessioninfo", version="1.2.4") #UPDATED
install.cran("xopen", version="1.0.1") #OK
install.cran("rcmdcheck", version="1.4.0") #OK
install.cran("remotes", version="2.5.0") #OK
install.cran("roxygen2", version="8.0.0") #UPDATED
install.cran("rversions", version="3.0.0") #UPDATED
install.cran("urlchecker", version="1.0.1") #OK
install.cran("diffobj", version="0.3.6") #UPDATED
install.cran("waldo", version="0.6.2") #UPDATED
install.cran("testthat", version="3.3.2") #UPDATED
install.cran("pak", version="0.10.0") # Dependency of devtools
install.cran("devtools", version="2.5.2") #UPDATED
install.cran("sp", version="2.2-1") #UPDATED
install.cran("FNN", version="1.1.4.1") #OK
install.cran("RANN", version="2.6.2") #OK
install.cran("snow", version="0.4-4") #OK
install.cran("formatR", version="1.14") #OK
install.cran("futile.options", version="1.0.1") #OK
install.cran("lambda.r", version="1.2.4") #OK
install.cran("futile.logger", version="1.4.9") #UPDATED
install.cran("boot", version="1.3-32") #UPDATED
install.cran("spatial", version="7.3-18") #OK
install.cran("bdsmatrix", version="1.3-7") #OK
install.cran("bbmle", version="1.0.25.1") #OK
install.cran("densEstBayes", version="1.0-2.2") #OK
install.cran("reldist", version="1.7-2") #OK
install.cran("rsvd", version="1.0.5") #OK
install.cran("RhpcBLASctl", version="0.23-42") #OK
install.cran("cowplot", version="1.2.0") #UPDATED
install.cran("harmony", version="2.0.5") #OK # Archived
install.cran("ggrastr", version="1.0.2") #OK
install.cran("sitmo", version="2.0.2") #OK
install.cran("dqrng", version="0.4.1") #OK
install.cran("RSpectra", version="0.16-2") #OK
install.cran("uwot", version="0.2.4") #UPDATED
install.cran("tensor", version="1.5.1") #UPDATED
install.cran("BiocManager", version="1.30.27") #UPDATED
install.cran("crosstalk", version="1.2.2") #UPDATED
install.cran("globals", version="0.19.1") #UPDATED
install.cran("listenv", version="1.0.0") #UPDATED
install.cran("parallelly", version="1.47.0") #UPDATED
install.cran("future", version="1.70.0") #UPDATED
install.cran("future.apply", version="1.20.2") #UPDATED
install.cran("spatstat.utils", version="3.2-3") #UPDATED
install.cran("spatstat.data", version="3.1-9") #UPDATED
install.cran("spatstat.univar", version="3.2-0") #UPDATED
install.cran("spatstat.sparse", version="3.2-0") #UPDATED
install.cran("goftest", version="1.2-3") #OK
install.cran("deldir", version="2.0-4") #OK
install.cran("polyclip", version="1.10-7") #OK
install.cran("spatstat.geom", version="3.8-1") #UPDATED
install.cran("spatstat.random", version="3.5-0") #UPDATED
install.cran("spatstat.explore", version="3.8-1") #UPDATED
install.cran("hdf5r", version="1.3.12") #OK
install.cran("dotCall64", version="1.2") #OK
install.cran("spam", version="2.11-4") #UPDATED
install.cran("here", version="1.0.2") #UPDATED
install.cran("pbapply", version="1.7-4") #UPDATED
install.cran("fitdistrplus", version="1.2-6") #UPDATED
install.cran("zoo", version="1.8-15") #UPDATED
install.cran("lmtest", version="0.9-40") #OK
install.cran("progressr", version="0.19.0") #UPDATED
install.cran("ica", version="1.0-3") #OK
install.cran("sanon", version="1.6") #OK
install.cran("leidenbase", version="0.1.37") #UPDATED
install.cran("fastDummies", version="1.7.6") #UPDATED
install.cran("ggridges", version="0.5.7") #UPDATED
install.cran("patchwork", version="1.3.2") #UPDATED
install.cran("reticulate", version="1.46.0") #UPDATED # Fixed that to avoid reticulate warning when using rpy2
install.cran("tidyr", version="1.3.2") #UPDATED
install.cran("scattermore", version="1.2") #OK
install.cran("sctransform", version="0.4.3") #UPDATED
install.cran("plotly", version="4.12.0") #UPDATED
install.cran("leiden", version="0.4.3.1") #OK
install.cran("SeuratObject", version="5.4.0") #UPDATED
install.cran("Seurat", version="5.5.0") #UPDATED
install.cran("hms", version="1.1.4") #OK
install.cran("progress", version="1.2.3") #OK
install.cran("tzdb", version="0.5.0") #OK
install.cran("vroom", version="1.7.1") #UPDATED
install.cran("readr", version="2.2.0") #UPDATED
install.cran("hexbin", version="1.28.5") #OK
install.cran("optparse", version="1.8.2") #OK
install.cran("maps", version="3.4.3") #OK
install.cran("fields", version="17.3") #OK
install.cran("timechange", version="0.4.0") #OK
install.cran("lubridate", version="1.9.5") #OK
install.cran("RPostgres", version="1.4.10") #OK
install.cran("coda", version="0.19-4.1") #OK
install.cran("emdbook", version="1.3.14") #OK
install.cran("fastmatch", version="1.1-8") #OK # Dependency of fgsea
install.cran("aisdk", version="1.4.12") #OK # Dependency of clusterProfiler
install.cran("yulab.utils", version="0.2.4") #OK # Dependency of clusterProfiler
install.cran("enrichit", version="0.1.5") #OK # Dependency of clusterProfiler
install.cran("gson", version="0.1.0") #OK # Dependency of clusterProfiler
install.cran("gridGraphics", version="0.5-1") #OK # Dependency of aplot
install.cran("ggfun", version="0.2.0") #OK # Dependency of aplot
install.cran("ggplotify", version="0.1.3") #OK # Dependency of aplot
install.cran("aplot", version="0.2.9") #OK # Dependency of enrichplot
install.cran("ggnewscale", version="0.5.2") #OK # Dependency of enrichplot
install.cran("ggtangle", version="0.1.2") #OK # Dependency of enrichplot
install.cran("tweenr", version="2.0.3") #OK # Dependency of scatterpie
install.cran("ggforce", version="0.5.0") #OK # Dependency of scatterpie
install.cran("scatterpie", version="0.2.6") #OK # Dependency of enrichplot..
install.cran("R.methodsS3", version="1.8.2") #OK # Dependency of R.utils
install.cran("R.oo", version="1.27.1") #OK # Dependency of R.utils
install.cran("R.utils", version="2.13.0") #OK # Dependency of GOSemSim
install.cran("ape", version="5.8-1") #OK # Dependency of ggtree
install.cran("tidytree", version="0.4.7") #OK # Dependency of ggtree
install.cran("truncnorm", version="1.0-9") #OK # Dependency of ashr
install.cran("mixsqp", version="0.3-54") #OK # Dependency of ashr
install.cran("SQUAREM", version="2026.1") #OK # Dependency of ashr
install.cran("etrunct", version="0.1") #OK # Dependency of ashr
install.cran("invgamma", version="1.2") #OK # Dependency of ashr
install.cran("ashr", version="2.2-63") #OK # ashr is an optional DESeq2 shrinkage alternative — de.v8.bulk.R
install.cran("fontBitstreamVera", version="0.1.1") # Dependency of ggiraph
install.cran("fontLiberation", version="0.1.0") # Dependency of ggiraph
install.cran("fontquiver", version="0.2.1") # Dependency of ggiraph
install.cran("gdtools", version="0.5.1") # Dependency of ggiraph
install.cran("ggiraph", version="0.9.6") # Dependency of ggtree
install.cran("ggnewscale", version="0.5.2") # Dependency of enrichplot
install.cran("ggrepel", version="0.9.8") # Dependency of enrichplot
install.cran("ggtangle", version="0.1.2") # Dependency of enrichplot
install.cran("tidydr", version="0.0.6") # Dependency of enrichplot
install.cran("ggforce", version="0.5.0") # Dependency of scatterpie
install.cran("scatterpie", version="0.2.6") # Dependency of enrichplot

# Bioconductor packages
# Note: Upgraded to 3.22 for the version of R we use ('4.5'). 3.23 requires R version '4.6'
# Here latest 3.22 version from https://mghp.osn.xsede.org/bir190004-bucket01/index.html#archive.bioconductor.org/packages/3.22/bioc/src/contrib/
install.bioconductor("BiocVersion", version="3.22.0", "3.22") #UPDATED # Main Bioconductor version
install.bioconductor("limma", version="3.66.0", "3.22") #UPDATED
install.bioconductor("edgeR", version="4.8.2", "3.22") #UPDATED
install.bioconductor("BiocGenerics", version="0.56.0", "3.22") #UPDATED
install.bioconductor("Biobase", version="2.70.0", "3.22") #UPDATED
install.bioconductor("S4Vectors", version="0.48.1", "3.22") #UPDATED
install.bioconductor("BiocParallel", version="1.44.0", "3.22") #UPDATED
install.bioconductor("IRanges", version="2.44.0", "3.22") #UPDATED
install.bioconductor("zlibbioc", version="1.54.0", "3.21") #UPDATED # deprecated in 3.22
install.bioconductor("XVector", version="0.50.0", "3.22") #UPDATED
install.bioconductor("GenomeInfoDbData", version="1.2.15", "3.22", "data/annotation") #UPDATED
install.bioconductor("UCSC.utils", version="1.6.1", "3.22") #UPDATED
install.bioconductor("Seqinfo", version="1.0.0", "3.22") # Dependency of GenomicRanges
install.bioconductor("GenomeInfoDb", version="1.46.2", "3.22") #UPDATED
install.bioconductor("GenomicRanges", version="1.62.1", "3.22") #UPDATED # Dependency of SummarizedExperiment
install.bioconductor("MatrixGenerics", version="1.22.0", "3.22") #UPDATED
install.bioconductor("S4Arrays", version="1.10.1", "3.22") #UPDATED
install.bioconductor("SparseArray", version="1.10.10", "3.22") #UPDATED
install.bioconductor("DelayedArray", version="0.36.1", "3.22") #UPDATED
install.bioconductor("SummarizedExperiment", version="1.40.0", "3.22") #UPDATED
install.bioconductor("Biostrings", version="2.78.0", "3.22") #UPDATED
install.bioconductor("KEGGREST", version="1.50.0", "3.22") #UPDATED
install.bioconductor("AnnotationDbi", version="1.72.0", "3.22") #UPDATED
install.bioconductor("annotate", version="1.88.0", "3.22") #UPDATED
install.bioconductor("genefilter", version="1.92.0", "3.22") #UPDATED
install.bioconductor("DESeq2", version="1.50.2", "3.22") #UPDATED
install.bioconductor("sva", version="3.58.0", "3.22") #UPDATED
install.bioconductor("SingleCellExperiment", version="1.32.0", "3.22") #UPDATED
install.bioconductor("SC3", version="1.38.0", "3.22") #UPDATED
install.bioconductor("ScaledMatrix", version="1.18.0", "3.22") #UPDATED
install.bioconductor("assorthead", version="1.4.0", "3.22") #UPDATED
install.bioconductor("beachmat", version="2.26.0", "3.22") #UPDATED
install.bioconductor("BiocSingular", version="1.26.1", "3.22") #UPDATED
install.bioconductor("BiocIO", version="1.20.0", "3.22") #UPDATED
install.bioconductor("BiocNeighbors", version="2.4.0", "3.22") #UPDATED
install.bioconductor("scuttle", version="1.20.0", "3.22") #UPDATED
install.bioconductor("scater", version="1.38.1", "3.22") #UPDATED
install.bioconductor("bluster", version="1.20.0", "3.22") #UPDATED
install.bioconductor("metapod", version="1.18.0", "3.22") #UPDATED
install.bioconductor("scran", version="1.38.1", "3.22") #UPDATED
install.bioconductor("multtest", version="2.66.0", "3.22") #UPDATED
install.bioconductor("Rhdf5lib", version="1.32.0", "3.22") #UPDATED
install.bioconductor("rhdf5filters", version="1.22.0", "3.22") #UPDATED
install.bioconductor("rhdf5", version="2.54.1", "3.22") #UPDATED
install.bioconductor("h5mread", version="1.2.1", "3.22") # Dependency of HDF5Array
install.bioconductor("HDF5Array", version="1.38.0", "3.22") #UPDATED
install.bioconductor("LoomExperiment", version="1.28.0", "3.22") #UPDATED
install.bioconductor("sparseMatrixStats", version="1.22.0", "3.22") #UPDATED
install.bioconductor("DelayedMatrixStats", version="1.32.0", "3.22") #UPDATED
install.bioconductor("ResidualMatrix", version="1.20.0", "3.22") #UPDATED
install.bioconductor("batchelor", version="1.26.0", "3.22") #UPDATED
install.bioconductor("apeglm", version="1.32.0", "3.22")  #UPDATED # LFC shrinkage (DESeq2 --shrinkage apeglm) — de.v8.bulk.R
install.bioconductor("fgsea", version="1.36.2", "3.22") #UPDATED # GSEA — enrichment.v8.bulk.R
install.bioconductor("GO.db", version="3.22.0", "3.22", "data/annotation") #UPDATED # Dependency of clusterProfiler
install.bioconductor("GOSemSim", version="2.36.0", "3.22") #UPDATED # Dependency of clusterProfiler
install.bioconductor("qvalue", version="2.42.0", "3.22") #UPDATED # Dependency of clusterProfiler, DOSE
install.bioconductor("DOSE", version="4.4.0", "3.22") #UPDATED # Dependency of enrichplot
install.bioconductor("treeio", version="1.34.0", "3.22") #UPDATED # Dependency of ggtree
install.bioconductor("ggtree", version="4.0.5", "3.22") #UPDATED # Dependency of enrichplot
install.bioconductor("enrichplot", version="1.30.5", "3.22") #UPDATED # Dependency of clusterProfiler
install.bioconductor("clusterProfiler", version="4.18.4", "3.22") #UPDATED # ClusterProfiler for ORA methods — enrichment.v8.bulk.R

# Git packages (copied locally for ensuring long-term compatibility)
install.github("sceasy", version="0.0.7")
install.github("loomR", version="0.2.1.9000")
install.github("M3Drop", version="3.10.6")
install.github("BPCells", version="0.3.1")
install.github("DoubletFinder", version="2.0.6")
