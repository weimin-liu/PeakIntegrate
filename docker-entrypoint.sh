#!/usr/bin/env bash
set -euo pipefail

MODE="${APP_MODE:-streamlit}"

case "$MODE" in
  streamlit)
    exec streamlit run /app/PeakIntegrate/app.py \
      --server.port="${STREAMLIT_PORT:-8501}" \
      --server.address=0.0.0.0 \
      --server.headless=true \
      "$@"
    ;;
  shiny)
    exec R -q -e "shiny::runApp('/app/PeakIntegrate/preprocessing', host = '0.0.0.0', port = as.integer(Sys.getenv('SHINY_PORT', '3838')))"
    ;;
  analysis)
    exec Rscript /app/PeakIntegrate/preprocessing/analysis.R "$@"
    ;;
  *)
    exec "$MODE" "$@"
    ;;
esac
