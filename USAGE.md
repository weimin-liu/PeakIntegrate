# PeakIntegrate — Detailed Usage Guide

This document walks through the complete workflow from raw data to final integrated peak areas.

---

## 1. Data Preparation

You need two types of input:

### 1.1 Picked-peak CSVs

One CSV per compound (e.g. `brGDGT_IIIa.csv`) in a `tables/` directory. Required columns:

| Column | Description |
|---|---|
| *(first column)* | Full sample identifier (must contain e.g. `AEGIS-139`) |
| `rt` | Peak apex retention time (seconds) |
| `rtmin` | Left peak boundary |
| `rtmax` | Right peak boundary |
| `into` | Integrated area |
| `intb` | Baseline-corrected area |
| `sigma` | Gaussian sigma estimate |

### 1.2 HDF5 chromatogram file

An HDF5 file (`chrom_data.h5`) organised as:

```
/AEGIS-139-full-name/
    brGDGT_IIIa/
        rt          → float64 array
        intensity   → float64 array
    brGDGT_IIa/
        rt          → ...
        intensity   → ...
/AEGIS-140-full-name/
    ...
```

---

## 2. Loading Data

```python
from PeakIntegrate.src import load_experiment

exp = load_experiment(
    datafolder="/path/to/tables",      # directory with CSVs
    hdf5_path="/path/to/chrom_data.h5",
    sample_regex=r"AEGIS-(\d+)",       # extracts sample IDs from CSV
)

print(exp)  # Experiment(samples=50, rt_corrected=False)
```

---

## 3. Retention-Time Correction

RT drift between runs is corrected using a polynomial fit anchored to known calibration peaks.

### Basic correction

```python
exp_corrected = exp.rt_shift()
# Uses default calibrants: C46-GDGT, brGDGT_Ib, brGDGT_Ia
# Polynomial degree: 2 (quadratic)
# Reference sample: first sample in the dict
```

### Custom calibrants & reference

```python
exp_corrected = exp.rt_shift(
    calibs=["C46-GDGT", "brGDGT_Ib", "brGDGT_Ia"],
    more_calibs=["brGDGT_Ic"],  # append additional anchors
    degree=3,                    # cubic polynomial
    ref_sample_name="AEGIS-100", # explicit reference sample
)
```

### Manual anchors (for problematic samples)

If automated alignment fails for specific samples, inject manual corrections:

```python
exp_corrected = exp_corrected.rt_shift(
    manual_anchors={
        "AEGIS-158": [(2512, 2534)],  # (observed_rt, target_rt)
        "AEGIS-200": [(1800, 1815), (2600, 2620)],
    }
)
```

> **Note:** `rt_shift()` returns a **deep copy** — the original `exp` is unchanged. You can chain corrections.

### Parameters Reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `calibs` | list[str] | `['C46-GDGT', 'brGDGT_Ib', 'brGDGT_Ia']` | Calibration compound names |
| `more_calibs` | list[str] | `None` | Additional calibrants to append |
| `degree` | int | `2` | Polynomial degree |
| `ref_sample_name` | str | First sample | Reference sample name |
| `manual_anchors` | dict | `None` | `{sample: [(obs_rt, target_rt), ...]}` |

---

## 4. Peak Clustering

Some compounds (e.g. brGDGT_IIIa) have multiple isomers that need to be
separated into distinct groups via KMeans clustering on retention time.

```python
exp_corrected = exp_corrected.point_cluster_batch({
    "brGDGT_IIIa": 3,  # split into 3 clusters → IIIa_0, IIIa_1, IIIa_2
    "brGDGT_IIa": 2,   # split into 2 clusters → IIa_0, IIa_1
})
```

After clustering, peaks are renamed with a `_0`, `_1`, `_2` suffix ordered by ascending RT.

**Important:** Only one peak per sample per cluster is kept. Duplicates are discarded.

---

## 5. Visualization

### EIC overlay (per compound)

```python
exp_corrected.plot_eic("brGDGT_IIIa", corrected=True)
```

### Picked-peak scatter (all compounds)

```python
exp_corrected.plot_picked_peaks()
```

Both use Plotly — interactive zoom, hover with sample/RT/intensity info.

---

## 6. Saving the Experiment

```python
import pickle

with open("experiment.pkl", "wb") as f:
    pickle.dump(exp_corrected, f, protocol=pickle.HIGHEST_PROTOCOL)
```

---

## 7. Gaussian Peak Integration

The final step fits Gaussian models and exports areas:

### From Python

```python
from PeakIntegrate.src import integrate_experiment

df = integrate_experiment(
    pkl_path="experiment.pkl",
    output_csv="results.csv",
)
print(df.head())
```

### From CLI

```bash
python -m PeakIntegrate.src.integration --pkl experiment.pkl --out results.csv
```

### Integration Parameters

| Parameter | Default | Description |
|---|---|---|
| `pkl_path` | `../../experiment.pkl` | Path to saved experiment |
| `output_csv` | `results.csv` | Output CSV path (`None` to skip) |
| `target_cmpds` | All 13 GDGTs | Compounds to integrate |
| `min_points` | `11` | Minimum data points for fitting |
| `savgol_window` | `11` | Savitzky–Golay window length |
| `savgol_poly` | `3` | Savitzky–Golay polynomial order |
| `prominence_frac` | `0.05` | Peak prominence threshold (fraction of max) |

### How it works

1. **EIC extraction** — matches compound name to the correct EIC
2. **RT masking** — crops data to `[rtmin, rtmax]` window from the experiment
3. **Smoothing** — Savitzky–Golay filter removes noise
4. **Peak counting** — `scipy.signal.find_peaks` with prominence threshold
5. **Gaussian fit:**
   - **1 peak** → single Gaussian → area = A × σ × √(2π)
   - **2+ peaks** → double Gaussian → pick the peak closest to expected median RT
   - **0 peaks** → `NaN`

---

## 8. Troubleshooting

| Problem | Solution |
|---|---|
| `Missing calibration peak` error | Check that the calibration compound exists in the reference sample. Try different `calibs`. |
| Too many `NaN` in results | Lower `min_points`, check if RT windows are too narrow. |
| Clustering gives wrong isomer count | Use `point_cluster()` directly to manually inspect. Check `plot_eic()` to visualise peaks. |
| RT correction looks wrong | Plot before/after with `plot_eic(corrected=True/False)`. Add `manual_anchors` if needed. |
| Import errors | Ensure the parent directory of `PeakIntegrate/` is in your `PYTHONPATH`. |
