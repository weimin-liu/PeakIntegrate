# syntax=docker/dockerfile:1.7

# ══════════════════════════════════════════════════════════════════════════════
#  PeakIntegrate — Combined R + Python image
#
#  Includes:
#    • `preprocessing/analysis.R` for the XCMS preprocessing pipeline
#    • `preprocessing/app.R` as a Shiny frontend for preprocessing
#    • `app.py` as the Streamlit frontend for the Python integration workflow
#
#  Runtime selection is handled with `APP_MODE`:
#    • `streamlit` (default) → Python web app on port 8501
#    • `shiny`               → R Shiny preprocessing app on port 3838
#    • `analysis`            → run `analysis.R`
# ══════════════════════════════════════════════════════════════════════════════

FROM bioconductor/bioconductor_docker:RELEASE_3_21

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV PEAKINTEGRATE_CONFIG=/app/PeakIntegrate/config/cmpds.yaml
ENV PEAKINTEGRATE_DATA_ROOT=/data

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --break-system-packages -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

RUN R -q -e "install.packages(c('shiny', 'plotly', 'yaml', 'RColorBrewer'), repos='https://cloud.r-project.org/')" \
    && R -q -e "BiocManager::install(c('xcms', 'MsExperiment', 'BiocParallel', 'rhdf5'), ask=FALSE, update=FALSE)"

WORKDIR /app
COPY . /app/PeakIntegrate/
COPY docker-entrypoint.sh /usr/local/bin/peakintegrate-entrypoint

RUN chmod +x /usr/local/bin/peakintegrate-entrypoint

ENV PYTHONPATH="/app:${PYTHONPATH}"

EXPOSE 8501 3838

ENTRYPOINT ["peakintegrate-entrypoint"]
CMD []
