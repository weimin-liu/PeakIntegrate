"""
integration.py — Gaussian peak deconvolution and integration.

Loads a pickled :class:`Experiment`, fits single or double Gaussian models
to each target compound's EIC, and exports integrated areas to CSV.

Usage::

    python -m PeakIntegrate.src.integration                   # defaults
    python -m PeakIntegrate.src.integration --pkl exp.pkl     # custom

Or programmatically::

    from PeakIntegrate.src.integration import integrate_experiment
    df = integrate_experiment("experiment.pkl")
"""

from __future__ import annotations

import os
import pickle

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter

from PeakIntegrate.src.models import (  # noqa: F401 — pickle needs these
    PickedPeak,
    EIC,
    Chromatogram,
    Experiment,
)


# ════════════════════════════════════════════
#  Mathematical Models
# ════════════════════════════════════════════

def gauss(x: np.ndarray, A: float, mu: float, sigma: float) -> np.ndarray:
    """Single Gaussian peak model."""
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def double_gauss(
    x: np.ndarray,
    A1: float, mu1: float, sigma1: float,
    A2: float, mu2: float, sigma2: float,
) -> np.ndarray:
    """Sum of two Gaussian peaks (for overlapping peak deconvolution)."""
    return gauss(x, A1, mu1, sigma1) + gauss(x, A2, mu2, sigma2)


# ════════════════════════════════════════════
#  Target Compounds (derived from cmpds.yaml)
# ════════════════════════════════════════════

def _default_target_compounds() -> list[str]:
    """Load target compounds from cmpds.yaml with default clustering."""
    try:
        from PeakIntegrate.src.config import load_compounds, resolve_target_compounds
        compounds = load_compounds()
        return resolve_target_compounds(compounds)
    except Exception:
        # Fallback if YAML is missing or unreadable
        return [
            "C46-GDGT",
            "brGDGT_IIIa_0", "brGDGT_IIIa_1", "brGDGT_IIIa_2",
            "brGDGT_IIIb", "brGDGT_IIIc",
            "brGDGT_IIa_0", "brGDGT_IIa_1",
            "brGDGT_IIb", "brGDGT_IIc",
            "brGDGT_Ia", "brGDGT_Ib", "brGDGT_Ic",
        ]

TARGET_COMPOUNDS: list[str] = _default_target_compounds()


# ════════════════════════════════════════════
#  Integration Pipeline
# ════════════════════════════════════════════

def integrate_experiment(
    pkl_path: str = "../../experiment.pkl",
    output_csv: str = "results.csv",
    target_cmpds: list[str] | None = None,
    min_points: int = 11,
    savgol_window: int = 11,
    savgol_poly: int = 3,
    prominence_frac: float = 0.05,
) -> "pd.DataFrame":
    """Integrate peaks for all target compounds in a saved Experiment.

    Parameters:
        pkl_path:        Path to the pickled :class:`Experiment`.
        output_csv:      Path for the output CSV (``None`` to skip).
        target_cmpds:    Compounds to process. Defaults to :data:`TARGET_COMPOUNDS`.
        min_points:      Minimum data points required for fitting.
        savgol_window:   Savitzky–Golay smoothing window length.
        savgol_poly:     Savitzky–Golay polynomial order.
        prominence_frac: Peak prominence threshold (fraction of max intensity).

    Returns:
        DataFrame with samples as rows, compounds as columns.
    """
    import pandas as pd

    if target_cmpds is None:
        target_cmpds = TARGET_COMPOUNDS

    with open(pkl_path, "rb") as f:
        exp: Experiment = pickle.load(f)

    all_results: dict[str, dict[str, float]] = {}

    for cmpd in target_cmpds:
        print(f"Processing {cmpd}...")

        try:
            rtmin, rtmax, rtmed = exp.get_rt(cmpd).values()
        except Exception as e:
            print(f"  Skipping {cmpd}: Could not get RT ({e})")
            continue

        cmpd_results: dict[str, float] = {}

        for sample_name, chrom_obj in exp.chromatograms.items():

            matching_eic = next(
                (eic for eic in chrom_obj.eics if cmpd.startswith(eic.name)),
                None,
            )
            if matching_eic is None:
                cmpd_results[sample_name] = np.nan
                continue

            rt = np.asarray(matching_eic.shifted_rt, dtype=float)
            intensity = np.asarray(matching_eic.intensity, dtype=float)

            mask = (
                (rt > rtmin) & (rt < rtmax)
                & np.isfinite(rt) & np.isfinite(intensity)
            )
            x = rt[mask]
            y = intensity[mask]
            y_s = savgol_filter(y, window_length=savgol_window, polyorder=savgol_poly)

            num_peaks = 0
            n_iter = 0
            while (num_peaks == 0) and (len(x) >= min_points):
                print(len(x))
                if not n_iter == 0:
                    x = x[1:]
                    y_s = y_s[1:]
                n_iter += 1
                if len(x) < min_points:
                    cmpd_results[sample_name] = np.nan
                    continue

                max_intensity = y_s.max()

                if max_intensity <= 0:
                    cmpd_results[sample_name] = 0.0
                    continue

                peaks_indices, _ = find_peaks(y_s, prominence=max_intensity * prominence_frac)
                num_peaks = len(peaks_indices)

            if sample_name == 'AEGIS-139' and cmpd == 'brGDGT_IIIa_1':
                print('yeah')
            if num_peaks == 0:
                x = rt[mask]
                y = intensity[mask]
                y_s = savgol_filter(y, window_length=savgol_window, polyorder=savgol_poly)
                peaks_indices = [np.argmin(abs(rtmed-x))]
                x = x[peaks_indices[0]:]
                y_s = y_s[peaks_indices[0]:]
                y = y[peaks_indices[0]:]
                num_peaks = 1

            area_main = np.nan

            # Single peak → single Gaussian
            if num_peaks == 1:
                apex_rt = x[peaks_indices[0]]
                apex_int = y_s[peaks_indices[0]]
                try:
                    popt, _ = curve_fit(
                        gauss, x, y_s,
                        p0=[apex_int, apex_rt, 5],
                        bounds=([0, apex_rt-2, 1], [np.inf, apex_rt+2, 20]), # add very rigid peak apex boundary, +/- 2s, as the rt correction is very good. #TODO: parameterize the boundary
                        maxfev=10000,
                    )
                    A, mu, sigma = popt
                    area_main = A * sigma * np.sqrt(2 * np.pi)
                except Exception:
                    pass

            # Two+ peaks → double Gaussian, pick closest to expected RT
            elif num_peaks >= 2:
                top2 = sorted(peaks_indices, key=lambda i: y_s[i], reverse=True)[:2]
                rt1, rt2 = x[top2[0]], x[top2[1]]
                int1, int2 = y_s[top2[0]], y_s[top2[1]]
                try:
                    popt, _ = curve_fit(
                        double_gauss, x, y_s,
                        p0=[int1, rt1, 5, int2, rt2, 5],
                        bounds=(
                            [0, rtmin, 1, 0, rtmin, 1],
                            [np.inf, rtmax, 30, np.inf, rtmax, 30],
                        ),
                        maxfev=10000,
                    )
                    A1, mu1, sigma1, A2, mu2, sigma2 = popt
                    if abs(mu1 - rtmed) < abs(mu2 - rtmed):
                        area_main = A1 * sigma1 * np.sqrt(2 * np.pi)
                    else:
                        area_main = A2 * sigma2 * np.sqrt(2 * np.pi)
                except Exception:
                    pass


            cmpd_results[sample_name] = area_main

        all_results[cmpd] = cmpd_results

    results_df = pd.DataFrame(all_results)
    results_df.index.name = "Sample"
    results_df = results_df.sort_index()

    if output_csv:
        results_df.to_csv(output_csv)
        print(f"\nResults saved to '{output_csv}'.")

    return results_df


# ════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════

def main() -> None:
    """CLI entry point for peak integration."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Integrate GDGT peaks from a pickled Experiment.",
    )
    parser.add_argument(
        "--pkl", default="/Users/weimin/10-Project/GDGT_peak_integration/experiment.pkl",
        help="Path to experiment.pkl (default: ../../experiment.pkl)",
    )
    parser.add_argument(
        "--out", default="results.csv",
        help="Output CSV path (default: results.csv)",
    )
    args = parser.parse_args()

    integrate_experiment(pkl_path=args.pkl, output_csv=args.out)


if __name__ == "__main__":
    main()
