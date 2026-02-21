"""
loader.py — Load chromatographic data into an Experiment.

Reads picked-peak CSVs and raw EIC data from an HDF5 file,
constructs the full :class:`Experiment` object.
"""

from __future__ import annotations

import os
import re
from typing import Optional

import numpy as np
import pandas as pd

from PeakIntegrate.src.models import (
    PickedPeak,
    EIC,
    Chromatogram,
    Experiment,
)


def load_experiment(
    datafolder: str = "/Users/weimin/10-Project/GDGT_peak_integration/tables",
    hdf5_path: str = "/Users/weimin/chrom_data.h5",
    sample_regex: str = r"AEGIS-(\d+)",
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

    reg = re.compile(sample_regex)

    # Read all CSVs into a dict keyed by compound name
    data: dict[str, pd.DataFrame] = {}
    sample_names_set: set[str] = set()

    for filename in os.listdir(datafolder):
        if not filename.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(datafolder, filename))
        df["SampleName"] = df.iloc[:, 0].map(
            lambda x, _reg=reg: f"AEGIS-{_reg.findall(str(x))[0]}"
        )
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


# ════════════════════════════════════════════
#  CLI entry point
# ════════════════════════════════════════════

def main() -> None:
    """Run the default processing pipeline (load → RT shift → cluster → export)."""
    import pickle

    exp = load_experiment()
    exp = exp.rt_shift()
    exp = exp.rt_shift(
        manual_anchors={
            "AEGIS-158": [(2512, 2534)],
        }
    )
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
