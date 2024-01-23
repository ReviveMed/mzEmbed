library(metabCombiner)
library(jsonlite)

main <- function(dataset1_path, dataset2_path, output_path, params_json_path) {

  if (missing(params_json_path)) {
    params_json_path <- "params.json"
  }
  # check the params path exists
  if (!file.exists(params_json_path)) {
    print(paste("params.json file does not exist at path:", params_json_path))
    # use default values
    params <- list()

  } else {
    # read parameters from json file

    print(paste("params.json file exists at path:", params_json_path))
    params <- fromJSON(params_json_path)
  }

  # use parameters from json file if they exist, otherwise use default values
  yes_warning <- ifelse(is.null(params$yes_warning), TRUE, params$yes_warning)


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


  # metabData parameters
  rtmin1 <- ifelse(is.null(params$rtmin1), 
                  0.95*min(dataset1$rtmed), params$rtmin1)
  rtmax1 <- ifelse(is.null(params$rtmax1), 
                  1.05*max(dataset1$rtmed), params$rtmax1)
  rtmin2 <- ifelse(is.null(params$rtmin2),
                  0.95*min(dataset2$rtmed), params$rtmin2)
  rtmax2 <- ifelse(is.null(params$rtmax2), 
                  1.05*max(dataset2$rtmed), params$rtmax2)
  binGap <- ifelse(is.null(params$binGap), 0.005, params$binGap)
  misspc1 <- ifelse(is.null(params$misspc1), 90, params$misspc1)
  misspc2 <- ifelse(is.null(params$misspc2), 90, params$misspc2)

  # selectAnchors parameters
  windx <- ifelse(is.null(params$windx), 0.03, params$windx)
  windy <- ifelse(is.null(params$windy), 0.03, params$windy)
  tolmz <- ifelse(is.null(params$tolmz), 0.003, params$tolmz)
  tolQ <- ifelse(is.null(params$tolQ), 0.3, params$tolQ)
  tolrtq <- ifelse(is.null(params$tolrtq), 0.3, params$tolrtq)

  # print the parameters
  print(paste("rtmin1:", rtmin1))
  print(paste("rtmax1:", rtmax1))
  print(paste("rtmin2:", rtmin2))
  print(paste("rtmax2:", rtmax2))
  print(paste("binGap:", binGap))
  print(paste("misspc1:", misspc1))
  print(paste("misspc2:", misspc2))
  print(paste("windx:", windx))
  print(paste("windy:", windy))
  print(paste("tolmz:", tolmz))
  print(paste("tolQ:", tolQ))

  data1 <- metabData(dataset1, mz = "mzmed", rt = "rtmed", 
                    id = "feats", adduct = NULL, 
                    extra = NULL, rtmin = rtmin1, rtmax = rtmax1, 
                    misspc = misspc1, measure = "median",
                    zero = TRUE, duplicate = opts.duplicate())

  data2 <- metabData(dataset2, mz = "mzmed", rt = "rtmed", 
                    id = "feats",  adduct = NULL,
                    extra = NULL, rtmin = rtmin2, rtmax = rtmax2,
                    misspc = misspc2, measure = "median", zero = TRUE,
                    duplicate = opts.duplicate())    


  ######## Create metabCombiner Object and Group Paired Features by m/z #########

  data.combined <- metabCombiner(xdata = data1, ydata = data2, binGap = binGap,
                                xid = "d1", yid = "d2")


  ########################### Compute RT Mapping ################################
  # data.combined <- selectAnchors(data.combined, useID = FALSE, windx = 0.03,
  #                               windy = 0.03, tolmz = 0.003, tolQ = 0.3)
  data.combined <- selectAnchors(data.combined, useID = FALSE, windx = windx,
                                windy = windy, tolmz = tolmz, tolQ = tolQ, tolrtq = tolrtq)

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
json_path <- args[4]


main(dataset1_path, dataset2_path, output_path, json_path)
