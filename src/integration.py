"""
integration.py — Gaussian peak deconvolution and integration.

Loads a pickled :class:`Experiment`, fits single or double Gaussian models
to each target compound's EIC, and exports integrated areas to CSV.
Optionally produces a multi-page PDF with Gaussian overlays on the raw EICs.

Usage::

    python -m PeakIntegrate.src.integration                   # defaults
    python -m PeakIntegrate.src.integration --pkl exp.pkl     # custom
    python -m PeakIntegrate.src.integration --pdf results.pdf # with PDF

Or programmatically::

    from PeakIntegrate.src.integration import integrate_experiment
    df = integrate_experiment("experiment.pkl", output_pdf="results.pdf")
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Optional

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
#  Fit Result Container
# ════════════════════════════════════════════

@dataclass
class FitResult:
    """Stores everything needed to plot a single fit overlay."""
    compound: str
    sample: str
    x: np.ndarray                   # RT values in the fit window
    y_raw: np.ndarray               # raw intensity (before smoothing)
    y_smoothed: np.ndarray          # Savgol-smoothed intensity
    fit_type: str                   # "single" or "double"
    chosen_A: float = 0.0           # amplitude of the chosen Gaussian
    chosen_mu: float = 0.0          # centre of the chosen Gaussian
    chosen_sigma: float = 0.0       # sigma of the chosen Gaussian
    area: float = np.nan            # integrated area
    rtmed: float = 0.0              # expected RT


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
#  PDF Export
# ════════════════════════════════════════════

def _export_fit_pdf(fit_results: list[FitResult], output_pdf: str) -> None:
    """Write all fit overlays to a multi-page PDF.

    Each page shows one (compound, sample) pair with:
      - Grey line: raw EIC intensity
      - Black line: Savgol-smoothed EIC
      - Coloured filled curve: the chosen fitted Gaussian
      - Dashed vertical line: expected RT (rtmed)
      - Title with compound, sample, and integrated area
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    # Group by compound, then sort samples within each compound
    from collections import OrderedDict
    grouped: dict[str, list[FitResult]] = OrderedDict()
    for fr in fit_results:
        grouped.setdefault(fr.compound, []).append(fr)
    for v in grouped.values():
        v.sort(key=lambda fr: fr.sample)

    with PdfPages(output_pdf) as pdf:
        for cmpd, results in grouped.items():
            for fr in results:
                fig, ax = plt.subplots(figsize=(8, 4))

                # Raw EIC
                ax.plot(fr.x, fr.y_raw, color="0.70", linewidth=0.8,
                        label="raw EIC")
                # Smoothed EIC
                ax.plot(fr.x, fr.y_smoothed, color="black", linewidth=1.0,
                        label="smoothed EIC")

                # Fitted Gaussian overlay (only if fit succeeded)
                if np.isfinite(fr.area) and fr.chosen_A > 0:
                    x_fine = np.linspace(fr.x.min(), fr.x.max(), 500)
                    y_fit = gauss(x_fine, fr.chosen_A, fr.chosen_mu,
                                  fr.chosen_sigma)
                    ax.fill_between(x_fine, y_fit, alpha=0.35,
                                    color="tab:blue", label="fitted Gaussian")
                    ax.plot(x_fine, y_fit, color="tab:blue", linewidth=1.2)

                # Expected RT
                ax.axvline(fr.rtmed, color="tab:red", linestyle="--",
                           linewidth=0.8, label=f"expected RT ({fr.rtmed:.1f})")

                area_str = f"{fr.area:.2e}" if np.isfinite(fr.area) else "N/A"
                ax.set_title(f"{fr.compound}  —  {fr.sample}  "
                             f"(area = {area_str})", fontsize=10)
                ax.set_xlabel("RT (s)")
                ax.set_ylabel("Intensity")
                ax.legend(fontsize=7, loc="upper right")
                fig.tight_layout()

                pdf.savefig(fig)
                plt.close(fig)

    print(f"\nPDF saved to '{output_pdf}' ({sum(len(v) for v in grouped.values())} pages).")


# ════════════════════════════════════════════
#  Integration Pipeline
# ════════════════════════════════════════════

def integrate_experiment(
    pkl_path: str = "../../experiment.pkl",
    output_csv: str = "results.csv",
    output_pdf: str | None = None,
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
        output_pdf:      Path for the output PDF with Gaussian overlays
                         (``None`` to skip).
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
    fit_results: list[FitResult] = []

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
            chosen_A, chosen_mu, chosen_sigma = 0.0, 0.0, 0.0

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
                    chosen_A, chosen_mu, chosen_sigma = A, mu, sigma
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
                        chosen_A, chosen_mu, chosen_sigma = A1, mu1, sigma1
                    else:
                        area_main = A2 * sigma2 * np.sqrt(2 * np.pi)
                        chosen_A, chosen_mu, chosen_sigma = A2, mu2, sigma2
                except Exception:
                    pass


            cmpd_results[sample_name] = area_main

            # Store fit result for PDF plotting
            if output_pdf is not None:
                fit_results.append(FitResult(
                    compound=cmpd,
                    sample=sample_name,
                    x=x.copy(),
                    y_raw=y.copy() if 'y' in dir() else y_s.copy(),
                    y_smoothed=y_s.copy(),
                    fit_type="single" if num_peaks == 1 else "double",
                    chosen_A=chosen_A,
                    chosen_mu=chosen_mu,
                    chosen_sigma=chosen_sigma,
                    area=area_main,
                    rtmed=rtmed,
                ))

        all_results[cmpd] = cmpd_results

    results_df = pd.DataFrame(all_results)
    results_df.index.name = "Sample"
    results_df = results_df.sort_index()

    if output_csv:
        results_df.to_csv(output_csv)
        print(f"\nResults saved to '{output_csv}'.")

    if output_pdf and fit_results:
        _export_fit_pdf(fit_results, output_pdf)

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
    parser.add_argument(
        "--pdf", default=None,
        help="Output PDF path for Gaussian overlay plots (default: None, no PDF)",
    )
    args = parser.parse_args()

    integrate_experiment(pkl_path=args.pkl, output_csv=args.out, output_pdf=args.pdf)


if __name__ == "__main__":
    main()
