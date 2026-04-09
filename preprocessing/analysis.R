#!/usr/bin/env Rscript
# ══════════════════════════════════════════════════════════════════════════════
#  analysis.R — XCMS Preprocessing for PeakIntegrate
#
#  Performs chromatographic peak detection and EIC extraction from mzML files.
#  Outputs:
#    • chrom_data.h5  — HDF5 file with per-sample, per-compound EIC data
#    • <compound>.csv — per-compound peak tables
#    • <rds_name>.rds — serialised XCMS result object
#
#  Usage:
#    Rscript analysis.R                           # use defaults below
#    Rscript analysis.R --mzml_dir /path/to/mzml  # override mzML folder
#
#  All tuneable parameters are in Section 1 below.
# ══════════════════════════════════════════════════════════════════════════════

library(xcms)
library(MsExperiment)
library(BiocParallel)
library(yaml)
library(rhdf5)

# ──────────────────────────────────────────────────────────────────────────────
#  1. CONFIGURATION — edit these or override via command-line arguments
# ──────────────────────────────────────────────────────────────────────────────

# Resolve paths relative to this script's location
SCRIPT_DIR <- tryCatch(
  dirname(sys.frame(1)$ofile), # sourced
  error = function(e) {
    dirname(commandArgs(trailingOnly = FALSE)[
      grep("--file=", commandArgs(trailingOnly = FALSE))
    ] |> sub("--file=", "", x = _))
  } # Rscript
)
if (is.null(SCRIPT_DIR) || length(SCRIPT_DIR) == 0 || SCRIPT_DIR == "") {
  SCRIPT_DIR <- "."
}
#PROJECT_ROOT <- normalizePath(file.path(SCRIPT_DIR, ".."), mustWork = FALSE)
PROJECT_ROOT <- "/Users/weimin/Project/AngkorWat/data/APCI/1st"

# --- Paths ---
MZML_DIR <- file.path(PROJECT_ROOT, "mzml") # Input mzML folder
CMPDS_YAML <- "/Users/weimin/Project/cmpds.yaml"
HDF5_OUT <- file.path(PROJECT_ROOT, "chrom_data.h5")
CSV_OUT_DIR <- file.path(PROJECT_ROOT, "tables") # per-compound CSVs
RDS_OUT <- file.path(PROJECT_ROOT, "Angkorwat.rds")

# --- Parallelism ---
N_CORES <- 4L

# --- CentWave: Global Peak Detection ---
CW_PPM <- 5
CW_PEAKWIDTH <- c(20, 60)
CW_SNTHRESH <- 10
CW_NOISE <- 1000
CW_MZDIFF <- 0.005
CW_PREFILTER <- c(3, 100) # k, I
CW_FITGAUSS <- FALSE
CW_MZCENTERFUN <- "wMean" # "wMean", "mean", "apex", "wMeanApex3"
CW_INTEGRATE <- 1L # 1 or 2

# --- CentWave: ROI Peak Detection (tighter parameters) ---
CW_ROI_NOISE <- 100
CW_ROI_PEAKWIDTH <- c(10, 50)
CW_ROI_SNTHRESH <- 10
CW_ROI_MZDIFF <- -0.001
CW_ROI_FITGAUSS <- TRUE
CW_ROI_INTEGRATE <- 2L

# --- Peak Grouping (PeakDensityParam) ---
PDP_BW <- 30
PDP_MIN_FRACTION <- 0.5
PDP_MIN_SAMPLES <- 1L
PDP_BIN_SIZE <- 0.25
PDP_MAX_FEATURES <- 50L
PDP_PPM <- 0

# --- RT Alignment (PeakGroupsParam) ---
ALIGN_MIN_FRAC <- 0.8
ALIGN_SPAN <- 0.2

# --- EIC Extraction Tolerances ---
MZ_TOL <- 0.005 # m/z window (± Da)
RT_TOL <- 8 * 60 # RT window (± seconds)


# ──────────────────────────────────────────────────────────────────────────────
#  2. LOAD COMPOUND DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────

message("Reading compound definitions from: ", CMPDS_YAML)
compound_defs <- yaml::read_yaml(CMPDS_YAML)

# Base compound names (exclude isomer variants like brGDGT_IIIa_1, _2)
all_cmpd_names <- names(compound_defs)
target_cmpds <- Filter(function(name) {
  !any(name != all_cmpd_names &
    startsWith(name, paste0(all_cmpd_names, "_")) &
    name != all_cmpd_names)
}, all_cmpd_names)

# Simpler approach: only keep compounds that don't have a parent in the list
target_cmpds <- c()
for (name in all_cmpd_names) {
  is_child <- FALSE
  for (other in all_cmpd_names) {
    if (name != other && startsWith(name, paste0(other, "_"))) {
      is_child <- TRUE
      break
    }
  }
  # Also skip compounds with NULL rt (can't extract EIC without RT)
  if (!is_child && !is.null(compound_defs[[name]]$rt)) {
    target_cmpds <- c(target_cmpds, name)
  }
}

message("Target compounds: ", paste(target_cmpds, collapse = ", "))


# ──────────────────────────────────────────────────────────────────────────────
#  3. LOAD mzML DATA
# ──────────────────────────────────────────────────────────────────────────────

if (!dir.exists(MZML_DIR) || MZML_DIR == "") {
  stop("Error: mzML directory not found: ", MZML_DIR)
}

fls <- list.files(MZML_DIR,
  pattern = "\\.mzML$",
  full.names = TRUE, ignore.case = TRUE
)

if (length(fls) == 0) {
  stop("No .mzML files found in: ", MZML_DIR)
}

message("Loading ", length(fls), " mzML files...")

# Parallel backend
if (N_CORES > 1) {
  bp_param <- SnowParam(workers = N_CORES, progressbar = TRUE)
} else {
  bp_param <- SerialParam()
}
register(bp_param)

sample_names <- basename(fls)
sample_df <- data.frame(
  sample_name = sample_names,
  sample_group = "Group1",
  stringsAsFactors = FALSE
)

mse_obj <- readMsExperiment(spectraFiles = fls, sampleData = sample_df)


# ──────────────────────────────────────────────────────────────────────────────
#  4. GLOBAL PEAK DETECTION (CentWave)
# ──────────────────────────────────────────────────────────────────────────────

message("Running global CentWave peak detection...")

cw_param <- CentWaveParam(
  ppm            = CW_PPM,
  peakwidth      = CW_PEAKWIDTH,
  snthresh       = CW_SNTHRESH,
  noise          = CW_NOISE,
  mzdiff         = CW_MZDIFF,
  prefilter      = CW_PREFILTER,
  fitgauss       = CW_FITGAUSS,
  mzCenterFun    = CW_MZCENTERFUN,
  integrate      = as.integer(CW_INTEGRATE),
  verboseColumns = TRUE
)

mse_obj <- findChromPeaks(
  object = mse_obj,
  param = cw_param,
  BPPARAM = bp_param,
  chunkSize = N_CORES
)


# ──────────────────────────────────────────────────────────────────────────────
#  5. PEAK GROUPING & RT ALIGNMENT
# ──────────────────────────────────────────────────────────────────────────────

message("Grouping peaks (PeakDensityParam)...")
pdp <- PeakDensityParam(
  sampleGroups = sample_df$sample_group,
  bw           = PDP_BW,
  minFraction  = PDP_MIN_FRACTION,
  minSamples   = PDP_MIN_SAMPLES,
  binSize      = PDP_BIN_SIZE,
  maxFeatures  = PDP_MAX_FEATURES,
  ppm          = PDP_PPM
)
mse_obj <- groupChromPeaks(mse_obj, param = pdp)

message("Aligning retention times (PeakGroupsParam)...")
pgp <- PeakGroupsParam(
  minFraction = ALIGN_MIN_FRAC,
  span        = ALIGN_SPAN
)
mse_obj <- adjustRtime(mse_obj, param = pgp)

message("Saving RDS to: ", RDS_OUT)
saveRDS(mse_obj, file = RDS_OUT)

# RT adjustment diagnostic plot
diffRt <- rtime(mse_obj) - rtime(mse_obj, adjusted = FALSE)
diffRt <- split(diffRt, fromFile(mse_obj))

cols <- RColorBrewer::brewer.pal(8, name = "Dark2")
pall <- colorRampPalette(cols)
colors <- pall(length(mse_obj))

plotAdjustedRtime(mse_obj, col = colors, peakGroupsPch = 4)


# ──────────────────────────────────────────────────────────────────────────────
#  6. EXTRACT EICs → HDF5
# ──────────────────────────────────────────────────────────────────────────────

message("Extracting EICs to HDF5: ", HDF5_OUT)

h5createFile(HDF5_OUT)

# Create HDF5 group structure
for (sname in sample_names) {
  h5createGroup(HDF5_OUT, paste0("/", sname))
  for (cmpd in target_cmpds) {
    h5createGroup(HDF5_OUT, paste0("/", sname, "/", cmpd))
  }
}

# Build m/z and RT extraction matrices from YAML definitions
mz_mat <- do.call(rbind, lapply(target_cmpds, function(key) {
  c(
    compound_defs[[key]]$mz - MZ_TOL,
    compound_defs[[key]]$mz + MZ_TOL
  )
}))

rt_mat <- do.call(rbind, lapply(target_cmpds, function(key) {
  c(
    compound_defs[[key]]$rt * 60 - RT_TOL,
    compound_defs[[key]]$rt * 60 + RT_TOL
  )
}))

chr_raw <- chromatogram(
  mse_obj,
  mz      = mz_mat,
  rt      = rt_mat,
  BPPARAM = MulticoreParam(workers = N_CORES)
)

# Write EIC data to HDF5
for (cmpd_idx in seq_along(target_cmpds)) {
  cmpd_name <- target_cmpds[cmpd_idx]

  for (sname in colnames(chr_raw)) {
    chr <- chr_raw[cmpd_idx, sname]
    group_path <- paste0("/", sname, "/", cmpd_name)

    h5write(rtime(chr), HDF5_OUT, paste0(group_path, "/rt"))
    h5write(intensity(chr), HDF5_OUT, paste0(group_path, "/intensity"))
  }
}

message("HDF5 export complete.")


# ──────────────────────────────────────────────────────────────────────────────
#  7. ROI PEAK PICKING → PER-COMPOUND CSVs
# ──────────────────────────────────────────────────────────────────────────────

message("ROI peak detection and interactive peak selection...")

# Ensure output directory exists
if (!dir.exists(CSV_OUT_DIR)) {
  dir.create(CSV_OUT_DIR, recursive = TRUE)
}

cw_param_roi <- CentWaveParam(
  ppm            = CW_PPM,
  peakwidth      = CW_ROI_PEAKWIDTH,
  snthresh       = CW_ROI_SNTHRESH,
  noise          = CW_ROI_NOISE,
  mzdiff         = CW_ROI_MZDIFF,
  mzCenterFun    = CW_MZCENTERFUN,
  integrate      = CW_ROI_INTEGRATE,
  fitgauss       = CW_ROI_FITGAUSS,
  verboseColumns = TRUE
)

for (cmpd_idx in seq_along(target_cmpds)) {
  cmpd_name <- target_cmpds[cmpd_idx]
  message("  Processing: ", cmpd_name)

  # Find peaks in the extracted chromatogram
  xchr <- findChromPeaks(chr_raw[cmpd_idx], param = cw_param_roi)

  # Interactive plot: user clicks to define retention time window
  plot(xchr,
    col = "black", peakType = "point",
    peakBg = "#ff000040", peakCol = "red"
  )

  cat("Click start and end point of the horizontal line...\n")
  pts <- locator(2)

  segments(pts$x[1], pts$y[1], pts$x[2], pts$y[1], col = "blue", lwd = 2)

  rt_window <- sort(pts$x)
  abline(v = rt_window, col = "blue", lwd = 2, lty = 2)
  usr <- par("usr")
  rect(rt_window[1], usr[3], rt_window[2], usr[4],
    border = NA, col = "#0000ff20"
  )

  # Filter peaks within the selected RT window and export
  pks <- chromPeaks(xchr)
  rownames(pks) <- sample_names[pks[, "column"]]
  pks_in <- pks[pks[, "rt"] >= rt_window[1] &
    pks[, "rt"] <= rt_window[2], , drop = FALSE]

  out_csv <- file.path(CSV_OUT_DIR, paste0(cmpd_name, ".csv"))
  write.csv(pks_in, out_csv)
  message("    Saved ", nrow(pks_in), " peaks → ", out_csv)
}

message("\n✓ Preprocessing complete!")
message("  HDF5:  ", HDF5_OUT)
message("  CSVs:  ", CSV_OUT_DIR)
message("  RDS:   ", RDS_OUT)
