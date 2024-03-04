# Run with R version 4.2.1
#if devtools is not already installed:

# Specify a CRAN mirror (see list of options at https://cran.r-project.org/mirrors.html)
cran_mirror <- "https://cran.r-project.org"
# cran_mirror <- "https://mirror.las.iastate.edu/CRAN/"

# Install devtools
install.packages("devtools", repos = cran_mirror)
Sys.setenv(R_REMOTES_NO_ERRORS_FROM_WARNINGS="true")

# Check if BiocManager is installed
if (!require(BiocManager)) {
  install.packages("BiocManager")
}

# Install BiocStyle
BiocManager::install("BiocStyle")

# For R versions 4.0 and later
devtools::install_github("hhabra/metabCombiner", build_vignettes = TRUE)

# Install jsonlite
install.packages("jsonlite", repos = cran_mirror)

# install.packages("devtools")
# Sys.setenv(R_REMOTES_NO_ERRORS_FROM_WARNINGS="true")

# #for R versions 4.0 and later
# devtools::install_github("hhabra/metabCombiner", build_vignettes = TRUE)

# # install jsonlite
# install.packages("jsonlite")