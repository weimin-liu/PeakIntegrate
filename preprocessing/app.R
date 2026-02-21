# ══════════════════════════════════════════════════════════════════════════════
#  PeakIntegrate — Shiny Preprocessing App
#
#  Full XCMS preprocessing pipeline with interactive RT window selection.
#  Launch:  shiny::runApp("PeakIntegrate/preprocessing")
#
#  Requirements:
#    install.packages(c("shiny", "bslib", "plotly", "yaml", "RColorBrewer"))
#    BiocManager::install(c("xcms", "MsExperiment", "rhdf5"))
# ══════════════════════════════════════════════════════════════════════════════

library(shiny)
library(bslib)
library(plotly)
library(yaml)
library(xcms)
library(MsExperiment)
library(BiocParallel)
library(rhdf5)

# ── Paths ──
SCRIPT_DIR <- dirname(rstudioapi::getActiveDocumentContext()$path %||% ".")
PROJECT_ROOT <- normalizePath(file.path(SCRIPT_DIR, ".."), mustWork = FALSE)
CMPDS_YAML <- file.path(PROJECT_ROOT, "config", "cmpds.yaml")


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

ui <- page_navbar(
    title = "PeakIntegrate — Preprocessing",
    theme = bs_theme(
        version = 5,
        bootswatch = "darkly",
        primary = "#8b5cf6",
        "navbar-bg" = "#1a1a2e"
    ),
    fillable = TRUE,

    # ── Tab 1: Data & Config ──
    nav_panel(
        title = "1. Data & Config",
        icon = icon("folder-open"),
        layout_sidebar(
            sidebar = sidebar(
                title = "Configuration",
                width = 350,
                textInput("mzml_dir", "mzML Directory",
                    value = "/Users/weimin/mzml"
                ),
                textInput("hdf5_out", "HDF5 Output Path",
                    value = file.path(PROJECT_ROOT, "..", "chrom_data.h5")
                ),
                textInput("csv_out_dir", "CSV Output Directory",
                    value = file.path(PROJECT_ROOT, "..", "tables")
                ),
                textInput("rds_out", "RDS Output Path",
                    value = file.path(PROJECT_ROOT, "..", "AEGIS_emily.rds")
                ),
                hr(),
                numericInput("n_cores", "Parallel Cores", value = 4, min = 1, max = 16),
                hr(),
                h6("CentWave — Global"),
                numericInput("cw_ppm", "PPM", value = 5),
                sliderInput("cw_peakwidth", "Peak Width (s)",
                    min = 5, max = 120, value = c(20, 60)
                ),
                numericInput("cw_snthresh", "S/N Threshold", value = 10),
                numericInput("cw_noise", "Noise", value = 1000),
                numericInput("cw_mzdiff", "mzdiff", value = 0.005, step = 0.001),
                selectInput("cw_mzcenterfun", "mzCenterFun",
                    choices = c("wMean", "mean", "apex", "wMeanApex3"),
                    selected = "wMean"
                ),
                selectInput("cw_integrate", "Integrate Method",
                    choices = c("1" = 1, "2" = 2), selected = 1
                ),
                hr(),
                h6("Peak Grouping"),
                numericInput("pdp_bw", "Bandwidth", value = 30),
                numericInput("pdp_min_frac", "Min Fraction", value = 0.5, step = 0.1),
                numericInput("pdp_bin_size", "Bin Size", value = 0.25, step = 0.05),
                hr(),
                h6("RT Alignment"),
                numericInput("align_min_frac", "Min Fraction", value = 0.8, step = 0.1),
                numericInput("align_span", "Span", value = 0.2, step = 0.05),
                hr(),
                h6("EIC Extraction"),
                numericInput("mz_tol", "m/z Tolerance (Da)", value = 0.01, step = 0.005),
                numericInput("rt_tol_min", "RT Tolerance (min)", value = 4, step = 0.5)
            ),
            card(
                card_header("Data Summary"),
                verbatimTextOutput("data_summary"),
                actionButton("btn_load", "Load mzML Files",
                    class = "btn-primary btn-lg mt-3",
                    icon  = icon("upload")
                )
            )
        )
    ),

    # ── Tab 2: XCMS Processing ──
    nav_panel(
        title = "2. XCMS Processing",
        icon = icon("cogs"),
        layout_columns(
            col_widths = c(12),
            card(
                card_header("Processing Pipeline"),
                actionButton("btn_process", "Run Full XCMS Pipeline",
                    class = "btn-primary btn-lg",
                    icon  = icon("play")
                ),
                hr(),
                verbatimTextOutput("processing_log"),
                hr(),
                card_header("RT Alignment Diagnostic"),
                plotOutput("rt_alignment_plot", height = "400px")
            )
        )
    ),

    # ── Tab 3: EIC Extraction ──
    nav_panel(
        title = "3. EIC → HDF5",
        icon = icon("database"),
        card(
            card_header("Extract EICs and Save to HDF5"),
            actionButton("btn_extract", "Extract EICs",
                class = "btn-primary btn-lg",
                icon  = icon("download")
            ),
            hr(),
            verbatimTextOutput("extract_log")
        )
    ),

    # ── Tab 4: RT Window Selection ──
    nav_panel(
        title = "4. RT Windows",
        icon = icon("crop"),
        layout_sidebar(
            sidebar = sidebar(
                title = "Compound",
                width = 250,
                selectInput("sel_compound", "Select Compound",
                    choices = NULL
                ),
                hr(),
                h6("CentWave — ROI"),
                numericInput("roi_noise", "Noise", value = 100),
                sliderInput("roi_peakwidth", "Peak Width (s)",
                    min = 5, max = 100, value = c(10, 50)
                ),
                numericInput("roi_mzdiff", "mzdiff", value = -0.001, step = 0.001),
                hr(),
                h6("Selected Window"),
                numericInput("rt_left", "RT Left (s)", value = NA),
                numericInput("rt_right", "RT Right (s)", value = NA),
                actionButton("btn_save_window", "Save Window & Export CSV",
                    class = "btn-success",
                    icon  = icon("check")
                ),
                hr(),
                actionButton("btn_next_cmpd", "Next Compound →",
                    class = "btn-outline-primary",
                    icon  = icon("arrow-right")
                )
            ),
            card(
                card_header("Chromatogram — click/drag to set RT window"),
                plotlyOutput("eic_plot", height = "500px"),
                hr(),
                verbatimTextOutput("window_status")
            )
        )
    ),

    # ── Tab 5: Summary ──
    nav_panel(
        title = "5. Summary",
        icon = icon("check-circle"),
        card(
            card_header("Export Summary"),
            verbatimTextOutput("final_summary"),
            hr(),
            actionButton("btn_done", "Open Output Folder",
                class = "btn-success btn-lg",
                icon  = icon("folder-open")
            )
        )
    )
)


# ══════════════════════════════════════════════════════════════════════════════
#  SERVER
# ══════════════════════════════════════════════════════════════════════════════

server <- function(input, output, session) {
    # ── Reactive values ──
    rv <- reactiveValues(
        mse_obj = NULL,
        sample_names = NULL,
        sample_df = NULL,
        fls = NULL,
        compound_defs = NULL,
        target_cmpds = NULL,
        chr_raw = NULL,
        windows = list(), # compound → c(rt_left, rt_right)
        log = "",
        extract_log = "",
        current_idx = 1
    )

    # ── Helper: append to log ──
    add_log <- function(msg) {
        rv$log <- paste0(rv$log, "\n", format(Sys.time(), "%H:%M:%S"), " — ", msg)
    }

    # ════════════════════════════════════════════
    #  Tab 1: Load Data
    # ════════════════════════════════════════════

    observeEvent(input$btn_load, {
        req(input$mzml_dir)

        withProgress(message = "Loading mzML files...", {
            fls <- list.files(input$mzml_dir,
                pattern = "\\.mzML$",
                full.names = TRUE, ignore.case = TRUE
            )

            if (length(fls) == 0) {
                showNotification("No .mzML files found!", type = "error")
                return()
            }

            rv$fls <- fls
            rv$sample_names <- basename(fls)
            rv$sample_df <- data.frame(
                sample_name = rv$sample_names,
                sample_group = "Group1",
                stringsAsFactors = FALSE
            )

            # Load compound definitions
            if (file.exists(CMPDS_YAML)) {
                rv$compound_defs <- yaml::read_yaml(CMPDS_YAML)

                # Filter to base compounds (not isomer children)
                all_names <- names(rv$compound_defs)
                base <- c()
                for (name in all_names) {
                    is_child <- FALSE
                    for (other in all_names) {
                        if (name != other && startsWith(name, paste0(other, "_"))) {
                            is_child <- TRUE
                            break
                        }
                    }
                    if (!is_child && !is.null(rv$compound_defs[[name]]$rt)) {
                        base <- c(base, name)
                    }
                }
                rv$target_cmpds <- base
                updateSelectInput(session, "sel_compound", choices = base)
            }

            showNotification(paste("Loaded", length(fls), "files"), type = "message")
        })
    })

    output$data_summary <- renderText({
        if (is.null(rv$fls)) {
            "No data loaded yet. Set the mzML directory and click 'Load mzML Files'."
        } else {
            paste0(
                "mzML files:   ", length(rv$fls), "\n",
                "Samples:      ", paste(head(rv$sample_names, 5), collapse = ", "),
                if (length(rv$sample_names) > 5) " ..." else "", "\n",
                "Compounds:    ", length(rv$target_cmpds), " (",
                paste(head(rv$target_cmpds, 5), collapse = ", "),
                if (length(rv$target_cmpds) > 5) " ..." else "", ")\n",
                "YAML config:  ", CMPDS_YAML
            )
        }
    })

    # ════════════════════════════════════════════
    #  Tab 2: XCMS Processing
    # ════════════════════════════════════════════

    observeEvent(input$btn_process, {
        req(rv$fls)

        withProgress(message = "Running XCMS pipeline...", value = 0, {
            # Parallel backend
            n_cores <- input$n_cores
            if (n_cores > 1) {
                bp <- SnowParam(workers = n_cores, progressbar = FALSE)
            } else {
                bp <- SerialParam()
            }
            register(bp)

            # 1. Read
            incProgress(0.1, detail = "Reading mzML files...")
            add_log("Reading mzML files...")
            rv$mse_obj <- readMsExperiment(
                spectraFiles = rv$fls,
                sampleData   = rv$sample_df
            )

            # 2. Peak detection
            incProgress(0.2, detail = "CentWave peak detection...")
            add_log("Running CentWave peak detection...")
            cw_param <- CentWaveParam(
                ppm            = input$cw_ppm,
                peakwidth      = input$cw_peakwidth,
                snthresh       = input$cw_snthresh,
                noise          = input$cw_noise,
                mzdiff         = input$cw_mzdiff,
                prefilter      = c(3, 100),
                fitgauss       = FALSE,
                mzCenterFun    = input$cw_mzcenterfun,
                integrate      = as.integer(input$cw_integrate),
                verboseColumns = TRUE
            )
            rv$mse_obj <- findChromPeaks(rv$mse_obj,
                param = cw_param,
                BPPARAM = bp, chunkSize = n_cores
            )

            # 3. Grouping
            incProgress(0.2, detail = "Grouping peaks...")
            add_log("Grouping peaks (PeakDensityParam)...")
            pdp <- PeakDensityParam(
                sampleGroups = rv$sample_df$sample_group,
                bw           = input$pdp_bw,
                minFraction  = input$pdp_min_frac,
                minSamples   = 1,
                binSize      = input$pdp_bin_size,
                maxFeatures  = 50,
                ppm          = 0
            )
            rv$mse_obj <- groupChromPeaks(rv$mse_obj, param = pdp)

            # 4. RT alignment
            incProgress(0.2, detail = "Aligning retention times...")
            add_log("Aligning retention times...")
            pgp <- PeakGroupsParam(
                minFraction = input$align_min_frac,
                span        = input$align_span
            )
            rv$mse_obj <- adjustRtime(rv$mse_obj, param = pgp)

            # 5. Save RDS
            incProgress(0.1, detail = "Saving RDS...")
            saveRDS(rv$mse_obj, file = input$rds_out)
            add_log(paste("RDS saved to:", input$rds_out))

            incProgress(0.2, detail = "Done!")
            add_log("XCMS pipeline complete!")
            showNotification("XCMS processing complete!", type = "message")
        })
    })

    output$processing_log <- renderText({
        rv$log
    })

    output$rt_alignment_plot <- renderPlot({
        req(rv$mse_obj)
        if (!hasAdjustedRtime(rv$mse_obj)) {
            return(NULL)
        }

        cols <- RColorBrewer::brewer.pal(8, name = "Dark2")
        pall <- colorRampPalette(cols)
        colors <- pall(length(rv$sample_names))
        plotAdjustedRtime(rv$mse_obj, col = colors, peakGroupsPch = 4)
    })

    # ════════════════════════════════════════════
    #  Tab 3: EIC Extraction
    # ════════════════════════════════════════════

    observeEvent(input$btn_extract, {
        req(rv$mse_obj, rv$target_cmpds, rv$compound_defs)

        withProgress(message = "Extracting EICs...", value = 0, {
            hdf5_path <- input$hdf5_out
            mz_tol <- input$mz_tol
            rt_tol <- input$rt_tol_min * 60 # convert min → sec

            # Create HDF5
            if (file.exists(hdf5_path)) file.remove(hdf5_path)
            h5createFile(hdf5_path)
            rv$extract_log <- paste0("Created: ", hdf5_path)

            # Create groups
            for (sname in rv$sample_names) {
                h5createGroup(hdf5_path, paste0("/", sname))
                for (cmpd in rv$target_cmpds) {
                    h5createGroup(hdf5_path, paste0("/", sname, "/", cmpd))
                }
            }

            # Build extraction matrices
            mz_mat <- do.call(rbind, lapply(rv$target_cmpds, function(key) {
                c(
                    rv$compound_defs[[key]]$mz - mz_tol,
                    rv$compound_defs[[key]]$mz + mz_tol
                )
            }))
            rt_mat <- do.call(rbind, lapply(rv$target_cmpds, function(key) {
                c(
                    rv$compound_defs[[key]]$rt * 60 - rt_tol,
                    rv$compound_defs[[key]]$rt * 60 + rt_tol
                )
            }))

            incProgress(0.3, detail = "Extracting chromatograms...")

            n_cores <- input$n_cores
            rv$chr_raw <- chromatogram(
                rv$mse_obj,
                mz      = mz_mat,
                rt      = rt_mat,
                BPPARAM = MulticoreParam(workers = n_cores)
            )

            incProgress(0.4, detail = "Writing to HDF5...")

            # Write EIC data
            for (cmpd_idx in seq_along(rv$target_cmpds)) {
                cmpd_name <- rv$target_cmpds[cmpd_idx]
                for (sname in colnames(rv$chr_raw)) {
                    chr <- rv$chr_raw[cmpd_idx, sname]
                    gp <- paste0("/", sname, "/", cmpd_name)
                    h5write(rtime(chr), hdf5_path, paste0(gp, "/rt"))
                    h5write(intensity(chr), hdf5_path, paste0(gp, "/intensity"))
                }
            }

            incProgress(0.3, detail = "Done!")
            rv$extract_log <- paste0(
                rv$extract_log, "\n",
                length(rv$target_cmpds), " compounds × ",
                length(rv$sample_names), " samples written to HDF5.\n",
                "Ready for RT window selection."
            )
            showNotification("EIC extraction complete!", type = "message")
        })
    })

    output$extract_log <- renderText({
        rv$extract_log
    })

    # ════════════════════════════════════════════
    #  Tab 4: RT Window Selection
    # ════════════════════════════════════════════

    # Current compound reactive
    current_compound <- reactive({
        req(rv$target_cmpds)
        rv$target_cmpds[rv$current_idx]
    })

    # Update selector when index changes
    observe({
        req(rv$target_cmpds)
        updateSelectInput(session, "sel_compound",
            selected = rv$target_cmpds[rv$current_idx]
        )
    })

    # Sync index when user selects manually
    observeEvent(input$sel_compound, {
        req(rv$target_cmpds)
        idx <- which(rv$target_cmpds == input$sel_compound)
        if (length(idx) > 0) rv$current_idx <- idx[1]
    })

    # Next compound button
    observeEvent(input$btn_next_cmpd, {
        req(rv$target_cmpds)
        if (rv$current_idx < length(rv$target_cmpds)) {
            rv$current_idx <- rv$current_idx + 1
        } else {
            showNotification("Last compound reached!", type = "warning")
        }
    })

    # ROI peak detection + interactive plotly chart
    output$eic_plot <- renderPlotly({
        req(rv$chr_raw, rv$target_cmpds)

        cmpd_idx <- rv$current_idx
        cmpd_name <- rv$target_cmpds[cmpd_idx]

        # ROI peak detection
        cw_roi <- CentWaveParam(
            ppm            = input$cw_ppm,
            peakwidth      = input$roi_peakwidth,
            snthresh       = 10,
            noise          = input$roi_noise,
            mzdiff         = input$roi_mzdiff,
            mzCenterFun    = input$cw_mzcenterfun,
            integrate      = 2L,
            fitgauss       = TRUE,
            verboseColumns = TRUE
        )

        xchr <- tryCatch(
            findChromPeaks(rv$chr_raw[cmpd_idx], param = cw_roi),
            error = function(e) NULL
        )

        # Build plotly figure with all samples overlaid
        fig <- plot_ly()

        for (s_idx in seq_along(rv$sample_names)) {
            sname <- rv$sample_names[s_idx]
            chr <- rv$chr_raw[cmpd_idx, sname]
            rt_v <- rtime(chr)
            int_v <- intensity(chr)

            fig <- fig %>% add_trace(
                x = rt_v, y = int_v, type = "scatter", mode = "lines",
                name = sname,
                line = list(width = 1),
                hovertemplate = paste0(sname, "<br>RT: %{x:.1f}s<br>Int: %{y:.2e}<extra></extra>")
            )
        }

        # Overlay detected peaks
        if (!is.null(xchr)) {
            pks <- chromPeaks(xchr)
            if (nrow(pks) > 0) {
                fig <- fig %>% add_trace(
                    x = pks[, "rt"], y = pks[, "maxo"],
                    type = "scatter", mode = "markers",
                    name = "Detected Peaks",
                    marker = list(color = "red", size = 10, symbol = "diamond"),
                    hovertemplate = "Peak<br>RT: %{x:.1f}s<br>MaxInt: %{y:.2e}<extra></extra>"
                )
            }
        }

        # Show existing window if set
        if (!is.null(rv$windows[[cmpd_name]])) {
            w <- rv$windows[[cmpd_name]]
            fig <- fig %>%
                layout(shapes = list(
                    list(
                        type = "rect",
                        x0 = w[1], x1 = w[2], y0 = 0, y1 = 1,
                        yref = "paper",
                        fillcolor = "rgba(99, 102, 241, 0.15)",
                        line = list(color = "rgba(99, 102, 241, 0.5)", width = 2)
                    )
                ))
        }

        fig %>%
            layout(
                title = paste("EIC:", cmpd_name),
                xaxis = list(title = "RT (s)"),
                yaxis = list(title = "Intensity"),
                template = "plotly_dark",
                dragmode = "select",
                selectdirection = "h",
                showlegend = TRUE,
                legend = list(font = list(size = 9))
            ) %>%
            event_register("plotly_selected") %>%
            config(displayModeBar = TRUE)
    })

    # Capture box/lasso selection as RT window
    observeEvent(event_data("plotly_selected", source = "A"), {
        sel <- event_data("plotly_selected")
        if (!is.null(sel) && nrow(sel) > 0) {
            rt_range <- range(sel$x)
            updateNumericInput(session, "rt_left", value = round(rt_range[1], 1))
            updateNumericInput(session, "rt_right", value = round(rt_range[2], 1))
        }
    })

    # Save window & export CSV
    observeEvent(input$btn_save_window, {
        req(input$rt_left, input$rt_right, rv$chr_raw)

        cmpd_name <- current_compound()
        cmpd_idx <- rv$current_idx

        rt_left <- input$rt_left
        rt_right <- input$rt_right

        # Store window
        rv$windows[[cmpd_name]] <- c(rt_left, rt_right)

        # ROI peak detection
        cw_roi <- CentWaveParam(
            ppm            = input$cw_ppm,
            peakwidth      = input$roi_peakwidth,
            snthresh       = 10,
            noise          = input$roi_noise,
            mzdiff         = input$roi_mzdiff,
            mzCenterFun    = input$cw_mzcenterfun,
            integrate      = 2L,
            fitgauss       = TRUE,
            verboseColumns = TRUE
        )

        xchr <- tryCatch(
            findChromPeaks(rv$chr_raw[cmpd_idx], param = cw_roi),
            error = function(e) NULL
        )

        if (is.null(xchr)) {
            showNotification("No peaks detected!", type = "error")
            return()
        }

        pks <- chromPeaks(xchr)
        rownames(pks) <- rv$sample_names[pks[, "column"]]
        pks_in <- pks[pks[, "rt"] >= rt_left & pks[, "rt"] <= rt_right, , drop = FALSE]

        # Ensure output directory exists
        csv_dir <- input$csv_out_dir
        if (!dir.exists(csv_dir)) dir.create(csv_dir, recursive = TRUE)

        out_csv <- file.path(csv_dir, paste0(cmpd_name, ".csv"))
        write.csv(pks_in, out_csv)

        showNotification(
            paste0("Saved ", nrow(pks_in), " peaks → ", basename(out_csv)),
            type = "message"
        )
    })

    output$window_status <- renderText({
        n_done <- length(rv$windows)
        n_total <- length(rv$target_cmpds)

        status <- paste0("Progress: ", n_done, " / ", n_total, " compounds\n\n")
        for (cmpd in rv$target_cmpds) {
            w <- rv$windows[[cmpd]]
            if (!is.null(w)) {
                status <- paste0(
                    status, "✅ ", cmpd, ": ",
                    round(w[1], 1), " – ", round(w[2], 1), " s\n"
                )
            } else {
                status <- paste0(status, "⬜ ", cmpd, "\n")
            }
        }
        status
    })

    # ════════════════════════════════════════════
    #  Tab 5: Summary
    # ════════════════════════════════════════════

    output$final_summary <- renderText({
        n_windows <- length(rv$windows)
        n_total <- length(rv$target_cmpds)
        csv_dir <- input$csv_out_dir
        hdf5_path <- input$hdf5_out
        rds_path <- input$rds_out

        paste0(
            "═══ Preprocessing Summary ═══\n\n",
            "Samples:         ", length(rv$sample_names %||% character(0)), "\n",
            "Compounds:       ", n_total, "\n",
            "RT windows set:  ", n_windows, " / ", n_total, "\n\n",
            "─── Output Files ───\n",
            "HDF5:  ", hdf5_path, "  ",
            if (file.exists(hdf5_path %||% "")) "✅" else "❌", "\n",
            "RDS:   ", rds_path, "  ",
            if (file.exists(rds_path %||% "")) "✅" else "❌", "\n",
            "CSVs:  ", csv_dir, "  ",
            if (dir.exists(csv_dir %||% "")) paste0("(", length(list.files(csv_dir, "*.csv")), " files)") else "❌", "\n\n",
            "─── Next Step ───\n",
            "Launch the Python GUI:\n",
            "  streamlit run PeakIntegrate/app.py\n",
            "Or in Docker:\n",
            "  docker run -p 8501:8501 -v ./data:/data peakintegrate"
        )
    })

    observeEvent(input$btn_done, {
        csv_dir <- input$csv_out_dir
        if (dir.exists(csv_dir)) {
            browseURL(csv_dir)
        }
    })
}


# ══════════════════════════════════════════════════════════════════════════════
#  Launch
# ══════════════════════════════════════════════════════════════════════════════

shinyApp(ui = ui, server = server)
