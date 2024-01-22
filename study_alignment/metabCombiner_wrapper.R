library(metabCombiner)


main <- function(dataset1_path, dataset2_path, output_path, yes_warning) {
  
  if (missing(yes_warning)) {
    yes_warning = TRUE
  }

  if (yes_warning) {
    options(warn = 1)
  } else {
    options(warn = -1)
  }

  # read in datasets from file; be sure that stringsAsFactors = FALSE
  dataset1 <- read.delim(dataset1_path, sep = "\t",
                        stringsAsFactors = FALSE)
  dataset2 <- read.delim(dataset2_path, sep = "\t",
                        stringsAsFactors = FALSE)

  # it would be better to read this from a parameters json file
  rtmin1 <- 0.95*min(dataset1$rtmed)
  rtmax1 <- 1.05*max(dataset1$rtmed)
  rtmin2 <- 0.95*min(dataset2$rtmed)
  rtmax2 <- 1.05*max(dataset2$rtmed)

  data1 <- metabData(dataset1, mz = "mzmed", rt = "rtmed", id = "feats",
                    adduct = NULL, extra = NULL,
                    rtmin = rtmin1, rtmax = rtmax1, misspc = 90, measure = "median",
                    zero = TRUE, duplicate = opts.duplicate())

  data2 <- metabData(dataset2, mz = "mzmed", rt = "rtmed", id = "feats",  adduct = NULL,
                      extra = NULL, rtmin = rtmin2, rtmax = rtmax2,
                    misspc = 90, measure = "median", zero = TRUE,
                    duplicate = opts.duplicate())    


  ######## Create metabCombiner Object and Group Paired Features by m/z #########

  data.combined <- metabCombiner(xdata = data1, ydata = data2, binGap = 0.005,
                                xid = "d1", yid = "d2")


  ########################### Compute RT Mapping ################################
  data.combined <- selectAnchors(data.combined, useID = FALSE, windx = 0.03,
                                windy = 0.03, tolmz = 0.003, tolQ = 0.3)

  anchors <- getAnchors(data.combined)   #to view the results of anchor selection

  set.seed(100)
  data.combined <- fit_gam(data.combined, useID = FALSE, k = seq(12, 20, 2),
                          iterFilter = 2, coef = 2, prop = 0.5, bs = "bs",
                          family = "scat", weights = 1, method = "REML",
                          optimizer = "newton")                                                

  ###################### Score Feature Pair Alignments ##########################
  data.combined <- calcScores(data.combined, A = 90, B = 15, C = 0.5,
                            fit = "gam", usePPM = FALSE, groups = NULL)

  ################### Reduce Feature Pair Alignment Report ####################
  #option 1: fully reduced table of 1-1 alignments
  data.combined <- labelRows(data.combined, maxRankX = 2, maxRankY = 2,
                            minScore = 0.5, delta = 0.1, method = "score",
                            resolveConflicts = TRUE, remove = TRUE)


  data.report <- combinedTable(data.combined)
  selected_data <- data.report[, c("idx", "idy", "score")]
  write.table(selected_data, file = output_path, sep = ",",
              na = "", row.names = FALSE)


}


# # Get the command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Assign the arguments to variables
dataset1_path <- args[1]
dataset2_path <- args[2]
output_path <- args[3]
yes_warning <- args[4]


main(dataset1_path, dataset2_path, output_path, yes_warning)
