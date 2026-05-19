"""
loader.py — Load chromatographic data into an Experiment.

Reads picked-peak CSVs and raw EIC data from an HDF5 file,
constructs the full :class:`Experiment` object.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from PeakIntegrate.src.config import load_compounds
from PeakIntegrate.src.models import (
    PickedPeak,
    EIC,
    Chromatogram,
    Experiment,
)


def load_experiment(
    datafolder: str = "/Users/weimin/Downloads/KapK/tables",
    hdf5_path: str = "/Users/weimin/Downloads/KapK/chrom_data.h5",
) -> Experiment:
    """Load an :class:`Experiment` from CSV peak-pick tables and an HDF5 file.

    Parameters:
        datafolder:    Directory containing one CSV per compound with
                       columns ``rt, rtmin, rtmax, into, intb, sigma``.
        hdf5_path:     Path to the HDF5 file with raw chromatographic data
                       (groups: ``<sample>/<compound>/rt`` and ``intensity``).
        sample_regex:  Regex to extract a short sample name from the first
                       column of each CSV.

    Returns:
        A fully constructed :class:`Experiment`.
    """
    import h5py


    # Read all CSVs into a dict keyed by compound name
    data: dict[str, pd.DataFrame] = {}
    sample_names_set: set[str] = set()

    for filename in os.listdir(datafolder):
        if not filename.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(datafolder, filename))
        df["SampleName"] = df.iloc[:, 0]
        sample_names_set.update(df["SampleName"].unique())
        data[filename.split(".")[0]] = df

    sample_names = sorted(sample_names_set)

    # Build per-sample EIC lists
    samples: dict[str, list[EIC]] = {s: [] for s in sample_names}

    with h5py.File(hdf5_path, "r") as f:
        hdf_keys = list(f.keys())

        for compound_name, compound_df in data.items():
            for sample_name in sample_names:
                sub = compound_df[compound_df["SampleName"] == sample_name]

                picked_peaks = []
                if not sub.empty:
                    for _, row in sub.iterrows():
                        picked_peaks.append(PickedPeak(
                            name=compound_name,
                            rt=row["rt"],
                            rtmin=row["rtmin"],
                            rtmax=row["rtmax"],
                            into=row["into"],
                            intb=row["intb"],
                            sigma=row["sigma"],
                        ))

                real_sample_name = next(
                    (k for k in hdf_keys if sample_name in k), None
                )
                if real_sample_name is None:
                    continue

                grp = f[real_sample_name][compound_name]
                eic = EIC(
                    name=compound_name,
                    mz=None,
                    rt=grp["rt"][:],
                    intensity=grp["intensity"][:],
                    picked=picked_peaks,
                )
                samples[sample_name].append(eic)

    chromatograms = {name: Chromatogram(eics) for name, eics in samples.items()}
    return Experiment(chromatograms)


def _build_peak_from_window(
    compound_name: str,
    rt: np.ndarray,
    intensity: np.ndarray,
    target_rt_seconds: float | None,
    window_seconds: float,
    rtmin_seconds: float | None = None,
    rtmax_seconds: float | None = None,
) -> Optional[PickedPeak]:
    """Create a coarse initial peak from a fixed RT window around the YAML RT."""
    if rt.size == 0 or intensity.size == 0:
        return None

    if rtmin_seconds is not None and rtmax_seconds is not None:
        left = float(min(rtmin_seconds, rtmax_seconds))
        right = float(max(rtmin_seconds, rtmax_seconds))
    elif target_rt_seconds is not None:
        left = target_rt_seconds - window_seconds
        right = target_rt_seconds + window_seconds
    else:
        return None
    mask = (rt >= left) & (rt <= right)
    if not np.any(mask):
        return None

    rt_win = rt[mask]
    intensity_win = intensity[mask]
    if rt_win.size == 0 or intensity_win.size == 0:
        return None

    apex_idx = int(np.nanargmax(intensity_win))
    apex_rt = float(rt_win[apex_idx])
    area = float(np.trapz(intensity_win, rt_win))
    sigma = float(max(window_seconds / 3.0, 1.0))

    return PickedPeak(
        name=compound_name,
        rt=apex_rt,
        rtmin=float(rt_win[0]),
        rtmax=float(rt_win[-1]),
        into=area,
        intb=area,
        sigma=sigma,
    )


def load_experiment_from_eic(
    hdf5_path: str,
    yaml_path: str,
    window_seconds: float = 60.0,
) -> Experiment:
    """Load an experiment directly from EIC traces and YAML RT hints.

    For each sample/compound EIC in the HDF5 file, this loader keeps the full
    trace and creates one coarse initial picked peak using a symmetric RT window
    centred on the expected RT from ``cmpds.yaml``.
    """
    import h5py

    compounds = load_compounds(yaml_path)
    chromatograms: dict[str, Chromatogram] = {}

    with h5py.File(hdf5_path, "r") as f:
        for sample_name in sorted(f.keys()):
            eics: list[EIC] = []
            sample_group = f[sample_name]

            for compound_name in sorted(sample_group.keys()):
                grp = sample_group[compound_name]
                rt = np.asarray(grp["rt"][:], dtype=float)
                intensity = np.asarray(grp["intensity"][:], dtype=float)

                picked: list[PickedPeak] = []
                cmpd_def = compounds.get(compound_name)
                if cmpd_def is not None and (
                    cmpd_def.rt is not None
                    or (cmpd_def.rtmin is not None and cmpd_def.rtmax is not None)
                ):
                    peak = _build_peak_from_window(
                        compound_name=compound_name,
                        rt=rt,
                        intensity=intensity,
                        target_rt_seconds=float(cmpd_def.rt) * 60.0 if cmpd_def.rt is not None else None,
                        window_seconds=float(window_seconds),
                        rtmin_seconds=float(cmpd_def.rtmin) * 60.0 if cmpd_def.rtmin is not None else None,
                        rtmax_seconds=float(cmpd_def.rtmax) * 60.0 if cmpd_def.rtmax is not None else None,
                    )
                    if peak is not None:
                        picked.append(peak)

                eics.append(EIC(
                    name=compound_name,
                    mz=cmpd_def.mz if cmpd_def is not None else None,
                    rt=rt,
                    intensity=intensity,
                    picked=picked,
                ))

            chromatograms[sample_name] = Chromatogram(eics)

    return Experiment(chromatograms)


def load_experiment_from_eic_csv(
    eic_folder: str,
    yaml_path: str,
    window_seconds: float = 60.0,
    file_axis: str = "sample",
    rt_unit: str = "seconds",
) -> Experiment:
    """Load an experiment from wide EIC CSV files plus YAML RT hints.

    Expected format:
        - ``file_axis="compound"``:
          one CSV per compound, first column RT, remaining columns samples
        - ``file_axis="sample"``:
          one CSV per sample, first column RT, remaining columns compounds
    """
    compounds = load_compounds(yaml_path)
    csv_files = sorted(
        filename for filename in os.listdir(eic_folder)
        if filename.lower().endswith(".csv")
    )
    if not csv_files:
        raise ValueError(f"No CSV files found in: {eic_folder}")

    chromatograms: dict[str, Chromatogram] = {}
    if file_axis not in {"compound", "sample"}:
        raise ValueError("file_axis must be 'compound' or 'sample'")
    if rt_unit not in {"seconds", "minutes"}:
        raise ValueError("rt_unit must be 'seconds' or 'minutes'")

    for filename in csv_files:
        csv_path = os.path.join(eic_folder, filename)
        df = pd.read_csv(csv_path)
        if df.shape[1] < 2:
            continue

        rt = np.asarray(df.iloc[:, 0], dtype=float)
        if rt_unit == "minutes":
            rt = rt * 60.0
        if file_axis == "compound":
            compound_name = os.path.splitext(filename)[0]
            cmpd_def = compounds.get(compound_name)

            for sample_name in df.columns[1:]:
                intensity = np.asarray(df[sample_name], dtype=float)
                picked: list[PickedPeak] = []

            if cmpd_def is not None and (
                cmpd_def.rt is not None
                or (cmpd_def.rtmin is not None and cmpd_def.rtmax is not None)
            ):
                peak = _build_peak_from_window(
                    compound_name=compound_name,
                    rt=rt,
                    intensity=intensity,
                    target_rt_seconds=float(cmpd_def.rt) * 60.0 if cmpd_def.rt is not None else None,
                    window_seconds=float(window_seconds),
                    rtmin_seconds=float(cmpd_def.rtmin) * 60.0 if cmpd_def.rtmin is not None else None,
                    rtmax_seconds=float(cmpd_def.rtmax) * 60.0 if cmpd_def.rtmax is not None else None,
                )
                if peak is not None:
                    picked.append(peak)

                chrom = chromatograms.setdefault(sample_name, Chromatogram())
                chrom.add_eic(EIC(
                    name=compound_name,
                    mz=cmpd_def.mz if cmpd_def is not None else None,
                    rt=rt,
                    intensity=intensity,
                    picked=picked,
                ))
        else:
            sample_name = os.path.splitext(filename)[0]
            chrom = chromatograms.setdefault(sample_name, Chromatogram())

            for compound_name in df.columns[1:]:
                intensity = np.asarray(df[compound_name], dtype=float)
                picked: list[PickedPeak] = []
                cmpd_def = compounds.get(compound_name)

                if cmpd_def is not None and (
                    cmpd_def.rt is not None
                    or (cmpd_def.rtmin is not None and cmpd_def.rtmax is not None)
                ):
                    peak = _build_peak_from_window(
                        compound_name=compound_name,
                        rt=rt,
                        intensity=intensity,
                        target_rt_seconds=float(cmpd_def.rt) * 60.0 if cmpd_def.rt is not None else None,
                        window_seconds=float(window_seconds),
                        rtmin_seconds=float(cmpd_def.rtmin) * 60.0 if cmpd_def.rtmin is not None else None,
                        rtmax_seconds=float(cmpd_def.rtmax) * 60.0 if cmpd_def.rtmax is not None else None,
                    )
                    if peak is not None:
                        picked.append(peak)

                chrom.add_eic(EIC(
                    name=compound_name,
                    mz=cmpd_def.mz if cmpd_def is not None else None,
                    rt=rt,
                    intensity=intensity,
                    picked=picked,
                ))

    return Experiment(chromatograms)


# ════════════════════════════════════════════
#  CLI entry point
# ════════════════════════════════════════════

def main() -> None:
    """Run the default processing pipeline (load → RT shift → cluster → export)."""
    import pickle

    exp = load_experiment()
    exp = exp.rt_shift()
    exp = exp.point_cluster_batch({
        "brGDGT_IIIa": 3,
        "brGDGT_IIa": 2,
    })
    exp.plot_picked_peaks()
    print(exp.get_rt("brGDGT_IIIa_0"))

    pkl_path = os.path.join(os.path.dirname(__file__), "..", "..", "experiment.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(exp, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nExperiment saved to {pkl_path}")


if __name__ == "__main__":
    main()
