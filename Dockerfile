# ══════════════════════════════════════════════════════════════════════════════
#  PeakIntegrate — Python-only Docker Image
#
#  The R preprocessing step (analysis.R) runs locally on your machine
#  (requires XCMS + interactive display). This container handles only
#  the Python pipeline: data loading → RT correction → clustering →
#  Gaussian integration, plus the Streamlit GUI.
#
#  Build:   docker build -t peakintegrate .
#  Run GUI: docker run -p 8501:8501 -v ./data:/data peakintegrate
# ══════════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim

# System libraries for HDF5
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    pkg-config \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy the package
WORKDIR /app
COPY . /app/PeakIntegrate/

# PeakIntegrate importable
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Streamlit config
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

# Default: launch the Streamlit GUI
CMD ["streamlit", "run", "PeakIntegrate/app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
