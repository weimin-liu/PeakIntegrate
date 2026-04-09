library(shiny)
library(plotly)
library(yaml)
library(xcms)
library(MsExperiment)
library(BiocParallel)
library(rhdf5)
library(RColorBrewer)

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0) y else x
}

find_project_root <- function() {
  candidates <- c(
    normalizePath(".", mustWork = FALSE),
    normalizePath("..", mustWork = FALSE),
    normalizePath(file.path(".", "PeakIntegrate"), mustWork = FALSE)
  )

  for (candidate in unique(candidates)) {
    if (file.exists(file.path(candidate, "config", "cmpds.yaml"))) {
      return(candidate)
    }
  }

  normalizePath(".", mustWork = FALSE)
}

PROJECT_ROOT <- find_project_root()
CMPDS_YAML <- file.path(PROJECT_ROOT, "config", "cmpds.yaml")
DEFAULT_DATA_ROOT <- normalizePath(file.path(PROJECT_ROOT, ".."), mustWork = FALSE)

get_target_compounds <- function(compound_defs) {
  all_names <- names(compound_defs)
  targets <- character()

  for (name in all_names) {
    is_child <- FALSE

    for (other in all_names) {
      if (name != other && startsWith(name, paste0(other, "_"))) {
        is_child <- TRUE
        break
      }
    }

    if (!is_child && !is.null(compound_defs[[name]]$rt)) {
      targets <- c(targets, name)
    }
  }

  targets
}

build_roi_param <- function(input) {
  CentWaveParam(
    ppm = input$cw_ppm,
    peakwidth = input$roi_peakwidth,
    snthresh = input$roi_snthresh,
    noise = input$roi_noise,
    mzdiff = input$roi_mzdiff,
    mzCenterFun = input$cw_mzcenterfun,
    integrate = as.integer(input$roi_integrate),
    fitgauss = isTRUE(input$roi_fitgauss),
    verboseColumns = TRUE
  )
}

ui <- fluidPage(
  titlePanel("PeakIntegrate Preprocessing"),
  tags$p("Shiny frontend for the XCMS workflow in analysis.R."),
  tabsetPanel(
    id = "main_tabs",
    tabPanel(
      "1. Setup",
      fluidRow(
        column(
          4,
          wellPanel(
            textInput("mzml_dir", "mzML Directory", value = file.path(DEFAULT_DATA_ROOT, "mzml")),
            textInput("hdf5_out", "HDF5 Output Path", value = file.path(DEFAULT_DATA_ROOT, "chrom_data.h5")),
            textInput("csv_out_dir", "CSV Output Directory", value = file.path(DEFAULT_DATA_ROOT, "tables")),
            textInput("rds_out", "RDS Output Path", value = file.path(DEFAULT_DATA_ROOT, "xcms_result.rds")),
            textInput("cmpds_yaml", "Compound YAML", value = CMPDS_YAML),
            numericInput("n_cores", "Parallel Cores", value = 4, min = 1, max = 32),
            actionButton("btn_load", "Load mzML Files")
          )
        ),
        column(
          8,
          h4("Detected inputs"),
          verbatimTextOutput("data_summary")
        )
      )
    ),
    tabPanel(
      "2. XCMS",
      fluidRow(
        column(
          4,
          wellPanel(
            h4("Global CentWave"),
            numericInput("cw_ppm", "PPM", value = 5),
            sliderInput("cw_peakwidth", "Peak Width (s)", min = 5, max = 120, value = c(20, 60)),
            numericInput("cw_snthresh", "S/N Threshold", value = 10),
            numericInput("cw_noise", "Noise", value = 1000),
            numericInput("cw_mzdiff", "mzdiff", value = 0.005, step = 0.001),
            numericInput("cw_prefilter_k", "Prefilter k", value = 3, min = 1),
            numericInput("cw_prefilter_i", "Prefilter I", value = 100, min = 0),
            selectInput("cw_mzcenterfun", "mzCenterFun", choices = c("wMean", "mean", "apex", "wMeanApex3"), selected = "wMean"),
            selectInput("cw_integrate", "Integrate", choices = c("1", "2"), selected = "1"),
            checkboxInput("cw_fitgauss", "Fit Gaussian", value = FALSE),
            tags$hr(),
            h4("Grouping / Alignment"),
            numericInput("pdp_bw", "PeakDensity bw", value = 30),
            numericInput("pdp_min_fraction", "PeakDensity minFraction", value = 0.5, step = 0.1),
            numericInput("pdp_min_samples", "PeakDensity minSamples", value = 1, min = 1),
            numericInput("pdp_bin_size", "PeakDensity binSize", value = 0.25, step = 0.05),
            numericInput("pdp_max_features", "PeakDensity maxFeatures", value = 50, min = 1),
            numericInput("pdp_ppm", "PeakDensity ppm", value = 0),
            numericInput("align_min_fraction", "PeakGroups minFraction", value = 0.8, step = 0.1),
            numericInput("align_span", "PeakGroups span", value = 0.2, step = 0.05),
            actionButton("btn_process", "Run XCMS Pipeline")
          )
        ),
        column(
          8,
          h4("Processing Log"),
          verbatimTextOutput("processing_log"),
          h4("RT Alignment Plot"),
          plotOutput("rt_alignment_plot", height = "420px")
        )
      )
    ),
    tabPanel(
      "3. Extract EICs",
      fluidRow(
        column(
          4,
          wellPanel(
            numericInput("mz_tol", "m/z Tolerance (Da)", value = 0.01, step = 0.001),
            numericInput("rt_tol_min", "RT Tolerance (min)", value = 4, step = 0.5),
            actionButton("btn_extract", "Extract EICs")
          )
        ),
        column(
          8,
          h4("Extraction Log"),
          verbatimTextOutput("extract_log")
        )
      )
    ),
    tabPanel(
      "4. ROI Windows",
      fluidRow(
        column(
          4,
          wellPanel(
            selectInput("sel_compound", "Compound", choices = character(0)),
            numericInput("roi_noise", "ROI Noise", value = 100),
            sliderInput("roi_peakwidth", "ROI Peak Width (s)", min = 5, max = 100, value = c(10, 50)),
            numericInput("roi_snthresh", "ROI S/N Threshold", value = 10),
            numericInput("roi_mzdiff", "ROI mzdiff", value = -0.001, step = 0.001),
            selectInput("roi_integrate", "ROI Integrate", choices = c("1", "2"), selected = "2"),
            checkboxInput("roi_fitgauss", "ROI Fit Gaussian", value = TRUE),
            tags$hr(),
            numericInput("rt_left", "RT Left (s)", value = NA_real_),
            numericInput("rt_right", "RT Right (s)", value = NA_real_),
            actionButton("btn_save_window", "Save Window and Export CSV"),
            actionButton("btn_next_cmpd", "Next Compound")
          )
        ),
        column(
          8,
          h4("Chromatogram"),
          plotlyOutput("eic_plot", height = "520px"),
          tags$p("Drag horizontally on the plot to populate the RT window fields."),
          h4("Window Status"),
          verbatimTextOutput("window_status")
        )
      )
    ),
    tabPanel(
      "5. Summary",
      fluidRow(
        column(
          12,
          h4("Outputs"),
          verbatimTextOutput("final_summary")
        )
      )
    )
  )
)

server <- function(input, output, session) {
  rv <- reactiveValues(
    fls = NULL,
    sample_names = NULL,
    sample_df = NULL,
    compound_defs = NULL,
    target_cmpds = NULL,
    mse_obj = NULL,
    chr_raw = NULL,
    windows = list(),
    processing_log = character(),
    extract_log = character(),
    current_idx = 1
  )

  append_log <- function(message_text) {
    rv$processing_log <- c(
      rv$processing_log,
      paste(format(Sys.time(), "%H:%M:%S"), message_text)
    )
  }

  append_extract_log <- function(message_text) {
    rv$extract_log <- c(
      rv$extract_log,
      paste(format(Sys.time(), "%H:%M:%S"), message_text)
    )
  }

  current_compound <- reactive({
    req(rv$target_cmpds, length(rv$target_cmpds) > 0)
    rv$target_cmpds[[rv$current_idx]]
  })

  detect_roi_peaks <- reactive({
    req(rv$chr_raw, rv$target_cmpds, rv$current_idx)

    tryCatch(
      findChromPeaks(rv$chr_raw[rv$current_idx], param = build_roi_param(input)),
      error = function(e) NULL
    )
  })

  observeEvent(input$btn_load, {
    req(nzchar(input$mzml_dir), nzchar(input$cmpds_yaml))

    if (!dir.exists(input$mzml_dir)) {
      showNotification("mzML directory does not exist.", type = "error")
      return()
    }

    if (!file.exists(input$cmpds_yaml)) {
      showNotification("Compound YAML file does not exist.", type = "error")
      return()
    }

    fls <- list.files(input$mzml_dir, pattern = "\\.mzML$", full.names = TRUE, ignore.case = TRUE)
    if (length(fls) == 0) {
      showNotification("No .mzML files found in the selected directory.", type = "error")
      return()
    }

    compound_defs <- yaml::read_yaml(input$cmpds_yaml)
    target_cmpds <- get_target_compounds(compound_defs)

    rv$fls <- fls
    rv$sample_names <- basename(fls)
    rv$sample_df <- data.frame(
      sample_name = rv$sample_names,
      sample_group = "Group1",
      stringsAsFactors = FALSE
    )
    rv$compound_defs <- compound_defs
    rv$target_cmpds <- target_cmpds
    rv$current_idx <- 1
    rv$windows <- list()
    rv$mse_obj <- NULL
    rv$chr_raw <- NULL
    rv$processing_log <- character()
    rv$extract_log <- character()

    updateSelectInput(session, "sel_compound", choices = target_cmpds, selected = target_cmpds[[1]])
    showNotification(paste("Loaded", length(fls), "mzML files and", length(target_cmpds), "target compounds."), type = "message")
  })

  output$data_summary <- renderText({
    if (is.null(rv$fls)) {
      return("Load mzML files and the compound YAML to begin.")
    }

    paste(
      paste("Project root:", PROJECT_ROOT),
      paste("YAML:", input$cmpds_yaml),
      paste("mzML files:", length(rv$fls)),
      paste("Samples:", paste(rv$sample_names, collapse = ", ")),
      paste("Target compounds:", paste(rv$target_cmpds, collapse = ", ")),
      sep = "\n"
    )
  })

  observeEvent(input$btn_process, {
    req(rv$fls, rv$sample_df)

    rv$processing_log <- character()

    withProgress(message = "Running XCMS pipeline", value = 0, {
      n_cores <- max(1L, as.integer(input$n_cores))
      bp <- if (n_cores > 1L) {
        SnowParam(workers = n_cores, progressbar = FALSE)
      } else {
        SerialParam()
      }
      register(bp)

      incProgress(0.15, detail = "Reading mzML files")
      append_log("Reading mzML files.")
      mse_obj <- readMsExperiment(spectraFiles = rv$fls, sampleData = rv$sample_df)

      incProgress(0.25, detail = "Finding chromatographic peaks")
      append_log("Running global CentWave peak detection.")
      cw_param <- CentWaveParam(
        ppm = input$cw_ppm,
        peakwidth = input$cw_peakwidth,
        snthresh = input$cw_snthresh,
        noise = input$cw_noise,
        mzdiff = input$cw_mzdiff,
        prefilter = c(as.integer(input$cw_prefilter_k), as.numeric(input$cw_prefilter_i)),
        fitgauss = isTRUE(input$cw_fitgauss),
        mzCenterFun = input$cw_mzcenterfun,
        integrate = as.integer(input$cw_integrate),
        verboseColumns = TRUE
      )
      mse_obj <- findChromPeaks(mse_obj, param = cw_param, BPPARAM = bp, chunkSize = n_cores)

      incProgress(0.2, detail = "Grouping peaks")
      append_log("Grouping peaks with PeakDensityParam.")
      pdp <- PeakDensityParam(
        sampleGroups = rv$sample_df$sample_group,
        bw = input$pdp_bw,
        minFraction = input$pdp_min_fraction,
        minSamples = as.integer(input$pdp_min_samples),
        binSize = input$pdp_bin_size,
        maxFeatures = as.integer(input$pdp_max_features),
        ppm = input$pdp_ppm
      )
      mse_obj <- groupChromPeaks(mse_obj, param = pdp)

      incProgress(0.2, detail = "Aligning retention times")
      append_log("Adjusting retention times with PeakGroupsParam.")
      pgp <- PeakGroupsParam(
        minFraction = input$align_min_fraction,
        span = input$align_span
      )
      mse_obj <- adjustRtime(mse_obj, param = pgp)

      incProgress(0.1, detail = "Saving RDS")
      saveRDS(mse_obj, file = input$rds_out)
      append_log(paste("Saved XCMS object to", input$rds_out))

      rv$mse_obj <- mse_obj
      incProgress(0.1, detail = "Finished")
      append_log("XCMS pipeline complete.")
    })
  })

  output$processing_log <- renderText({
    if (length(rv$processing_log) == 0) {
      "Run the XCMS pipeline to populate this log."
    } else {
      paste(rv$processing_log, collapse = "\n")
    }
  })

  output$rt_alignment_plot <- renderPlot({
    req(rv$mse_obj)
    if (!hasAdjustedRtime(rv$mse_obj)) {
      return(invisible(NULL))
    }

    palette_values <- colorRampPalette(brewer.pal(8, "Dark2"))(length(rv$sample_names))
    plotAdjustedRtime(rv$mse_obj, col = palette_values, peakGroupsPch = 4)
  })

  observeEvent(input$btn_extract, {
    req(rv$mse_obj, rv$target_cmpds, rv$compound_defs)

    rv$extract_log <- character()

    withProgress(message = "Extracting EICs", value = 0, {
      hdf5_path <- input$hdf5_out
      mz_tol <- input$mz_tol
      rt_tol <- input$rt_tol_min * 60

      if (file.exists(hdf5_path)) {
        file.remove(hdf5_path)
      }

      incProgress(0.1, detail = "Creating HDF5 structure")
      h5createFile(hdf5_path)
      append_extract_log(paste("Created", hdf5_path))

      for (sample_name in rv$sample_names) {
        h5createGroup(hdf5_path, paste0("/", sample_name))
        for (compound_name in rv$target_cmpds) {
          h5createGroup(hdf5_path, paste0("/", sample_name, "/", compound_name))
        }
      }

      mz_mat <- do.call(rbind, lapply(rv$target_cmpds, function(key) {
        c(rv$compound_defs[[key]]$mz - mz_tol, rv$compound_defs[[key]]$mz + mz_tol)
      }))
      rt_mat <- do.call(rbind, lapply(rv$target_cmpds, function(key) {
        c(rv$compound_defs[[key]]$rt * 60 - rt_tol, rv$compound_defs[[key]]$rt * 60 + rt_tol)
      }))

      incProgress(0.45, detail = "Extracting chromatograms")
      n_cores <- max(1L, as.integer(input$n_cores))
      chr_raw <- chromatogram(
        rv$mse_obj,
        mz = mz_mat,
        rt = rt_mat,
        BPPARAM = if (n_cores > 1L) MulticoreParam(workers = n_cores) else SerialParam()
      )

      incProgress(0.35, detail = "Writing chromatograms to HDF5")
      for (cmpd_idx in seq_along(rv$target_cmpds)) {
        cmpd_name <- rv$target_cmpds[[cmpd_idx]]
        for (sample_name in rv$sample_names) {
          chr <- chr_raw[cmpd_idx, sample_name]
          h5write(rtime(chr), hdf5_path, paste0("/", sample_name, "/", cmpd_name, "/rt"))
          h5write(intensity(chr), hdf5_path, paste0("/", sample_name, "/", cmpd_name, "/intensity"))
        }
      }

      rv$chr_raw <- chr_raw
      append_extract_log(paste("Wrote", length(rv$target_cmpds), "compounds across", length(rv$sample_names), "samples."))
      incProgress(0.1, detail = "Finished")
    })
  })

  output$extract_log <- renderText({
    if (length(rv$extract_log) == 0) {
      "Run EIC extraction after the XCMS pipeline completes."
    } else {
      paste(rv$extract_log, collapse = "\n")
    }
  })

  observe({
    req(rv$target_cmpds, length(rv$target_cmpds) > 0)
    updateSelectInput(session, "sel_compound", choices = rv$target_cmpds, selected = current_compound())
  })

  observeEvent(input$sel_compound, {
    req(rv$target_cmpds, input$sel_compound)
    idx <- match(input$sel_compound, rv$target_cmpds)
    if (!is.na(idx)) {
      rv$current_idx <- idx
      saved_window <- rv$windows[[input$sel_compound]]
      if (!is.null(saved_window)) {
        updateNumericInput(session, "rt_left", value = saved_window[[1]])
        updateNumericInput(session, "rt_right", value = saved_window[[2]])
      }
    }
  }, ignoreInit = TRUE)

  observeEvent(input$btn_next_cmpd, {
    req(rv$target_cmpds)
    if (rv$current_idx < length(rv$target_cmpds)) {
      rv$current_idx <- rv$current_idx + 1
    }
  })

  output$eic_plot <- renderPlotly({
    req(rv$chr_raw, rv$sample_names)

    cmpd_name <- current_compound()
    cmpd_idx <- rv$current_idx
    xchr <- detect_roi_peaks()
    fig <- plot_ly(source = "rt_window")

    for (sample_name in rv$sample_names) {
      chr <- rv$chr_raw[cmpd_idx, sample_name]
      fig <- fig %>% add_lines(
        x = rtime(chr),
        y = intensity(chr),
        name = sample_name,
        hovertemplate = paste0(sample_name, "<br>RT: %{x:.1f}s<br>Intensity: %{y:.3g}<extra></extra>")
      )
    }

    if (!is.null(xchr)) {
      peaks <- chromPeaks(xchr)
      if (nrow(peaks) > 0) {
        fig <- fig %>% add_markers(
          x = peaks[, "rt"],
          y = peaks[, "maxo"],
          name = "Detected peaks",
          marker = list(color = "red", size = 8, symbol = "diamond"),
          hovertemplate = "Peak<br>RT: %{x:.1f}s<br>Intensity: %{y:.3g}<extra></extra>"
        )
      }
    }

    saved_window <- rv$windows[[cmpd_name]]
    shapes <- list()
    if (!is.null(saved_window)) {
      shapes <- list(list(
        type = "rect",
        x0 = saved_window[[1]],
        x1 = saved_window[[2]],
        y0 = 0,
        y1 = 1,
        yref = "paper",
        fillcolor = "rgba(30, 144, 255, 0.15)",
        line = list(color = "rgba(30, 144, 255, 0.7)")
      ))
    }

    fig %>%
      layout(
        title = paste("EIC:", cmpd_name),
        xaxis = list(title = "Retention time (s)"),
        yaxis = list(title = "Intensity"),
        dragmode = "select",
        selectdirection = "h",
        shapes = shapes
      ) %>%
      event_register("plotly_selected")
  })

  observeEvent(event_data("plotly_selected", source = "rt_window"), {
    selected <- event_data("plotly_selected", source = "rt_window")
    if (!is.null(selected) && nrow(selected) > 0) {
      rt_range <- range(selected$x, na.rm = TRUE)
      updateNumericInput(session, "rt_left", value = round(rt_range[[1]], 1))
      updateNumericInput(session, "rt_right", value = round(rt_range[[2]], 1))
    }
  })

  observeEvent(input$btn_save_window, {
    req(rv$chr_raw, rv$sample_names, rv$target_cmpds)

    rt_left <- suppressWarnings(as.numeric(input$rt_left))
    rt_right <- suppressWarnings(as.numeric(input$rt_right))
    if (is.na(rt_left) || is.na(rt_right) || rt_left >= rt_right) {
      showNotification("Please provide a valid RT window.", type = "error")
      return()
    }

    cmpd_name <- current_compound()
    cmpd_idx <- rv$current_idx
    rv$windows[[cmpd_name]] <- c(rt_left, rt_right)

    xchr <- detect_roi_peaks()
    if (is.null(xchr)) {
      showNotification("ROI peak detection returned no result for this compound.", type = "warning")
      return()
    }

    peaks <- chromPeaks(xchr)
    if (nrow(peaks) == 0) {
      peaks_in <- peaks
    } else {
      rownames(peaks) <- rv$sample_names[peaks[, "column"]]
      peaks_in <- peaks[peaks[, "rt"] >= rt_left & peaks[, "rt"] <= rt_right, , drop = FALSE]
    }

    if (!dir.exists(input$csv_out_dir)) {
      dir.create(input$csv_out_dir, recursive = TRUE)
    }

    out_csv <- file.path(input$csv_out_dir, paste0(cmpd_name, ".csv"))
    write.csv(peaks_in, out_csv)
    showNotification(paste("Saved", nrow(peaks_in), "peaks to", basename(out_csv)), type = "message")
  })

  output$window_status <- renderText({
    if (is.null(rv$target_cmpds)) {
      return("Load data first.")
    }

    lines <- c(sprintf("Progress: %d / %d compounds", length(rv$windows), length(rv$target_cmpds)), "")
    for (compound_name in rv$target_cmpds) {
      saved_window <- rv$windows[[compound_name]]
      if (is.null(saved_window)) {
        lines <- c(lines, paste("[pending]", compound_name))
      } else {
        lines <- c(lines, sprintf("[saved] %s: %.1f to %.1f s", compound_name, saved_window[[1]], saved_window[[2]]))
      }
    }
    paste(lines, collapse = "\n")
  })

  output$final_summary <- renderText({
    csv_count <- if (dir.exists(input$csv_out_dir)) {
      length(list.files(input$csv_out_dir, pattern = "\\.csv$", ignore.case = TRUE))
    } else {
      0L
    }

    paste(
      paste("HDF5:", input$hdf5_out, if (file.exists(input$hdf5_out)) "[exists]" else "[missing]"),
      paste("RDS:", input$rds_out, if (file.exists(input$rds_out)) "[exists]" else "[missing]"),
      paste("CSV directory:", input$csv_out_dir, sprintf("[%d csv files]", csv_count)),
      paste("RT windows saved:", length(rv$windows), "/", length(rv$target_cmpds %||% character())),
      sep = "\n"
    )
  })
}

shinyApp(ui = ui, server = server)
