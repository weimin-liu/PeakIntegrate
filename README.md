# PeakIntegrate

Automated GDGT chromatographic peak integration pipeline for branched and isoprenoid GDGTs from HPLC-MS data.

## Features

- **Data ingestion** — Loads picked-peak CSVs and raw chromatograms from HDF5 into a structured `Experiment` object
- **Retention-time correction** — Polynomial RT alignment using calibration compounds with optional manual anchors
- **Peak clustering** — KMeans-based grouping of co-eluting isomers (e.g. brGDGT_IIIa → IIIa_0, IIIa_1, IIIa_2)
- **Gaussian deconvolution** — Single and double Gaussian fitting with automatic peak selection closest to expected RT
- **Interactive plotting** — Plotly-based EIC overlays and picked-peak scatter plots

## Installation

### Requirements

- Python ≥ 3.11
- numpy, scipy, pandas, scikit-learn, h5py, plotly

### Install

```bash
pip install numpy scipy pandas scikit-learn h5py plotly streamlit
```

Ensure the parent directory of `PeakIntegrate/` is on your `PYTHONPATH` (or work from that directory).

## Quick Start

```python
from PeakIntegrate.src import Experiment, load_experiment, integrate_experiment

# Step 1: Load data
exp = load_experiment(
    datafolder="path/to/tables",
    hdf5_path="path/to/chrom_data.h5",
)

# Step 2: RT correction
exp = exp.rt_shift()

# Step 3: Cluster isomers
exp = exp.point_cluster_batch({
    "brGDGT_IIIa": 3,
    "brGDGT_IIa": 2,
})

# Step 4: Save & integrate
import pickle
with open("experiment.pkl", "wb") as f:
    pickle.dump(exp, f, protocol=pickle.HIGHEST_PROTOCOL)

df = integrate_experiment(pkl_path="experiment.pkl", output_csv="results.csv")
```

## GUI (Streamlit)

Launch the interactive GUI:

```bash
pip install streamlit   # if not already installed
streamlit run PeakIntegrate/app.py
```

The GUI provides a guided 4-step workflow:
1. **RT Correction** — select calibrants, reference sample, manual anchors, before/after comparison
2. **Visualization** — interactive EIC overlays and picked-peak scatter plots
3. **Clustering** — configure and run KMeans isomer grouping
4. **Integration** — Gaussian fitting, results table, CSV download

## Project Structure

```
PeakIntegrate/
├── README.md                 ← This file
├── USAGE.md                  ← Detailed workflow guide
├── app.py                    ← Streamlit GUI
├── config/
│   └── cmpds.yaml            ← Compound definitions (m/z + expected RT)
├── preprocessing/
│   └── analysis.R            ← XCMS peak detection & EIC extraction
└── src/
    ├── __init__.py            ← Public API exports
    ├── config.py              ← YAML compound loader + name resolver
    ├── models.py              ← Data model (PickedPeak, EIC, Chromatogram, Experiment)
    ├── loader.py              ← Data loading from CSVs + HDF5
    └── integration.py         ← Gaussian deconvolution & peak integration
```

## Full Pipeline

```
┌─────────────────────────────────┐
│  1. R Preprocessing (Shiny)     │
│  shiny::runApp("preprocessing") │
│  mzML files → chrom_data.h5     │
│              → per-compound CSVs│
├─────────────────────────────────┤
│  2. Python Processing (GUI)     │
│  streamlit run app.py           │
│  → RT correction → clustering   │
│  → Gaussian integration         │
│  → results.csv                  │
└─────────────────────────────────┘

Config: config/cmpds.yaml (shared by R and Python)
```

### Launching the Shiny App

```r
# From R console
shiny::runApp("PeakIntegrate/preprocessing")
```

The Shiny app provides a 5-step GUI:
1. **Data & Config** — set paths, CentWave parameters
2. **XCMS Processing** — peak detection, grouping, RT alignment
3. **EIC → HDF5** — extract and save chromatograms
4. **RT Windows** — interactive plotly drag-to-select (replaces `locator()`)
5. **Summary** — review outputs and next steps

## API Summary

| Class / Function | Module | Purpose |
|---|---|---|
| `PickedPeak` | models | Single peak with RT, area, sigma |
| `EIC` | models | Extracted ion chromatogram + picked peaks |
| `Chromatogram` | models | Per-sample EIC collection with O(1) lookup |
| `Experiment` | models | Multi-sample container, RT correction, clustering |
| `load_experiment()` | loader | Build Experiment from CSVs + HDF5 |
| `integrate_experiment()` | integration | Gaussian deconvolution → CSV export |

## Docker

Share the Python pipeline + GUI as a lightweight container (~200 MB).  
R preprocessing runs locally (requires XCMS + interactive display).

```bash
# Build
cd PeakIntegrate
docker build -t peakintegrate .

# Launch GUI (http://localhost:8501)
# Mount the directory containing chrom_data.h5, CSVs, experiment.pkl
docker run -p 8501:8501 -v /path/to/data:/data peakintegrate

# Or use Docker Compose
docker compose up
```

> **Workflow:** Run `analysis.R` locally first (produces HDF5 + CSVs), then
> use Docker for everything else.

## License

Internal research tool
