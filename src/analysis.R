library(xcms)
library(MsExperiment)
library(BiocParallel)
library(yaml)
library(rhdf5)

h5createFile("chrom_data.h5")


# --- Configuration ---

# 1. Data Source
# Replace with the actual path to your mzML folder
FOLDER_PATH <- "/Users/weimin/mzml"

# 2. Parallel Config
N_CORES <- 4

# 3. Peak Detection (CentWave)
CW_PPM <- 5
CW_PEAKWIDTH <- c(20, 60)
CW_SNTHRESH <- 10
CW_NOISE <- 1000
CW_MZDIFF <- 0.005
CW_PREFILTER <- c(3, 100) # k, I
CW_FITGAUSS <- FALSE
CW_MZCENTERFUN <- "wMean" # "wMean", "mean", "apex", "wMeanApex3"
CW_INTEGRATE <- 1 # 1 or 2



# --- Processing Logic ---

# Check Data
if (!dir.exists(FOLDER_PATH) || FOLDER_PATH == "") {
    stop(paste("Error: Directory not found or not set:", FOLDER_PATH))
}

fls <- list.files(FOLDER_PATH, pattern = "\\.mzML$", full.names = TRUE, ignore.case = TRUE)

if (length(fls) == 0) {
    stop("No .mzML files found in the directory.")
}

message(paste("Loading", length(fls), "files..."))

# Parallel Backend
if (N_CORES > 1) {
    # Using SnowParam for consistency, though MulticoreParam is fine for scripts on Mac/Linux
    bp_param <- SnowParam(workers = N_CORES, progressbar = TRUE)
} else {
    bp_param <- SerialParam()
}
register(bp_param)

# 1. READ
# Assuming sample groups are not critical or defaults are fine if we don't have metadata
sample_names <- basename(fls)
df <- data.frame(
    sample_name = sample_names,
    sample_group = "Group1", # Default grouping
    stringsAsFactors = FALSE
)

mse_obj <- readMsExperiment(spectraFiles = fls, sampleData = df)

# 2. ALIGNMENT & PEAK PICKING (Global)
cw_param <- CentWaveParam(
    ppm = CW_PPM,
    peakwidth = CW_PEAKWIDTH,
    snthresh = CW_SNTHRESH,
    noise = CW_NOISE,
    mzdiff = CW_MZDIFF,
    prefilter = CW_PREFILTER,
    fitgauss = CW_FITGAUSS,
    mzCenterFun = CW_MZCENTERFUN,
    integrate = as.integer(CW_INTEGRATE),
    verboseColumns = TRUE
)

mse_obj <- findChromPeaks(
  object = mse_obj,
  param = cw_param,
  BPPARAM = bp_param,
  chunkSize = N_CORES
)


# Grouping (PeakDensityParam)
PDP_BW <- 30
PDP_MIN_FRACTION <- 0.5
PDP_MIN_SAMPLES <- 1
PDP_BIN_SIZE <- 0.25
PDP_MAX_FEATURES <- 50
PDP_PPM <- 0
pdp <- PeakDensityParam(
    sampleGroups = df$sample_group,
    bw = PDP_BW,
    minFraction = PDP_MIN_FRACTION,
    minSamples = PDP_MIN_SAMPLES,
    binSize = PDP_BIN_SIZE,
    maxFeatures = PDP_MAX_FEATURES,
    ppm = PDP_PPM
)
mse_obj <- groupChromPeaks(mse_obj, param = pdp)


# Alignment (PeakGroupsParam)
ALIGN_MIN_FRAC <- 0.8
ALIGN_SPAN <- 0.2
pgp <- PeakGroupsParam(
    minFraction = ALIGN_MIN_FRAC,
    span = ALIGN_SPAN
)
mse_obj <- adjustRtime(mse_obj, param = pgp)
saveRDS(mse_obj, file='AEGIS_emily.rds')

diffRt <- rtime(mse_obj) - rtime(mse_obj, adjusted = FALSE)

## By default, rtime and most other accessor methods return a numeric vector. To
## get the values grouped by sample we have to split this vector by file/sample
diffRt <- split(diffRt, fromFile(mse_obj))


#set up some colors for the squid plot
cols <- RColorBrewer::brewer.pal(8,name = "Dark2")
pall <- colorRampPalette(cols)
colors <-pall(length(mse_obj))

#plot the squid plot with and without the legend
plotAdjustedRtime(mse_obj,col = colors,peakGroupsPch = 4)

# 5. Analysis Region (ROI)
compound = yaml::read_yaml('/Users/weimin/10-Project/GDGT_peak_integration/shiny/cmpds.yaml')
target_cmpds = c('C46-GDGT','brGDGT_IIIa','brGDGT_IIIb', 'brGDGT_IIIc', 'brGDGT_IIa','brGDGT_IIb','brGDGT_IIc', 'brGDGT_Ia', 'brGDGT_Ib','brGDGT_Ic' )

mz_tol = 0.01
rt_tol = 4*60

for (sample_name in sample_names) {
  h5createGroup("chrom_data.h5", paste0("/", sample_name))
  for (cmpd in target_cmpds) {
    group_path <- paste0("/", sample_name, "/", cmpd)
    h5createGroup("chrom_data.h5", group_path)
  }
}


### extract the EICs, and save them to a hdf5 file for later use.
eics <- list()

mz_mat <- do.call(rbind, lapply(target_cmpds, function(key) {
  exp_compound <- compound[[key]]
  c(exp_compound$mz - mz_tol,
    exp_compound$mz + mz_tol)
}))
rt_mat <- do.call(rbind, lapply(target_cmpds, function(key) {
  exp_compound <- compound[[key]]
  c(exp_compound$rt * 60 - rt_tol,
    exp_compound$rt * 60 + rt_tol)
}))
chr_raw <- chromatogram(
  mse_obj,
  mz = mz_mat,
  rt = rt_mat,
  BPPARAM = MulticoreParam(workers = N_CORES)
)

for (cmpd_idx in seq_along(target_cmpds)) {
  
  cmpd_name <- target_cmpds[cmpd_idx]
  
  for (sample_name in colnames(chr_raw)) {
    
    chr <- chr_raw[cmpd_idx, sample_name]
    
    group_path <- paste0("/", sample_name, "/", cmpd_name)
    
    rt  <- rtime(chr)
    int <- intensity(chr)
    
    h5write(rt,  "chrom_data.h5", paste0(group_path, "/rt"))
    h5write(int, "chrom_data.h5", paste0(group_path, "/intensity"))
  }
}


CW_NOISE <- 100
CW_FITGAUSS <- TRUE
cw_param_roi <- CentWaveParam(
    ppm = CW_PPM,
    peakwidth = c(10,50),
    snthresh = 10,
    noise = CW_NOISE,
    mzdiff = -0.001,
    mzCenterFun = CW_MZCENTERFUN,
    integrate = 2,
    fitgauss = CW_FITGAUSS,
    verboseColumns = TRUE
)

for (cmpd_idx in seq_along(target_cmpds)) {
  
  cmpd_name <- target_cmpds[cmpd_idx]
  # Finding peaks in extracted chromatogram
  xchr <- findChromPeaks(chr_raw[cmpd_idx], param = cw_param_roi)
  
  # Plotting
  plot(xchr, col="black", peakType="point",
       peakBg="#ff000040", peakCol="red")
  
  cat("Click start and end point of the horizontal line...\n")
  pts <- locator(2)
  
  segments(pts$x[1], pts$y[1], pts$x[2], pts$y[1], col="blue", lwd=2)
  
  rt_window <- sort(pts$x)
  abline(v = rt_window, col="blue", lwd=2, lty=2)
  usr <- par("usr")
  rect(rt_window[1], usr[3], rt_window[2], usr[4],
       border = NA, col = "#0000ff20")
  
  pks <- chromPeaks(xchr)
  rownames(pks) <- sample_names[pks[,"column"]]
  pks_in <- pks[pks[, "rt"] >= rt_window[1] & pks[, "rt"] <= rt_window[2], , drop = FALSE]
  write.csv(pks_in, paste0(cmpd_name, ".csv"))
}
