"""
integration.py — Peak deconvolution and integration.

Loads a pickled :class:`Experiment`, fits peak models (Gaussian,
exponentially modified Gaussian, or bi-Gaussian) to each target
compound's EIC, and exports integrated areas to CSV. Optionally
produces a multi-page PDF with model overlays on the raw EICs.

Usage::

    python -m PeakIntegrate.src.integration                          # defaults
    python -m PeakIntegrate.src.integration --pkl exp.pkl            # custom
    python -m PeakIntegrate.src.integration --pdf results.pdf        # with PDF
    python -m PeakIntegrate.src.integration --model emg              # EMG model

Or programmatically::

    from PeakIntegrate.src.integration import integrate_experiment
    df = integrate_experiment("experiment.pkl", output_pdf="results.pdf",
                              fit_model="bigauss")
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from scipy.special import erfc

from PeakIntegrate.src.models import (  # noqa: F401 — pickle needs these
    PickedPeak,
    EIC,
    Chromatogram,
    Experiment,
)

# Allowed model names
VALID_MODELS = ("gaussian", "emg", "bigauss")


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


def emg(x: np.ndarray, A: float, mu: float, sigma: float, tau: float) -> np.ndarray:
    """Exponentially Modified Gaussian (EMG).

    Parameters:
        A:     Amplitude scaling factor.
        mu:    Centre of the underlying Gaussian.
        sigma: Width of the underlying Gaussian.
        tau:   Exponential relaxation time (>0 = right tail, <0 = left tail).
    """
    tau = np.abs(tau) + 1e-10  # ensure positive & non-zero
    z = (x - mu) / sigma - sigma / tau
    prefactor = A * (sigma / tau) * np.sqrt(np.pi / 2)
    return prefactor * np.exp(0.5 * (sigma / tau) ** 2 - (x - mu) / tau) * erfc(-z / np.sqrt(2))


def double_emg(
    x: np.ndarray,
    A1: float, mu1: float, sigma1: float, tau1: float,
    A2: float, mu2: float, sigma2: float, tau2: float,
) -> np.ndarray:
    """Sum of two EMG peaks."""
    return emg(x, A1, mu1, sigma1, tau1) + emg(x, A2, mu2, sigma2, tau2)


def bigauss(x: np.ndarray, A: float, mu: float, sigma_l: float, sigma_r: float) -> np.ndarray:
    """Bi-Gaussian (split Gaussian) peak model.

    Uses ``sigma_l`` for ``x < mu`` and ``sigma_r`` for ``x >= mu``.
    """
    sigma = np.where(x < mu, sigma_l, sigma_r)
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def double_bigauss(
    x: np.ndarray,
    A1: float, mu1: float, sigma_l1: float, sigma_r1: float,
    A2: float, mu2: float, sigma_l2: float, sigma_r2: float,
) -> np.ndarray:
    """Sum of two bi-Gaussian peaks."""
    return bigauss(x, A1, mu1, sigma_l1, sigma_r1) + bigauss(x, A2, mu2, sigma_l2, sigma_r2)


# ── Area helpers ──

def _gauss_area(A: float, sigma: float) -> float:
    return A * sigma * np.sqrt(2 * np.pi)

def _emg_area(A: float, sigma: float, tau: float) -> float:
    """EMG area equals A * sigma * sqrt(2π) (same as Gaussian; tau only affects shape)."""
    return A * sigma * np.sqrt(2 * np.pi)

def _bigauss_area(A: float, sigma_l: float, sigma_r: float) -> float:
    return A * np.sqrt(2 * np.pi) * (sigma_l + sigma_r) / 2


def _eval_single_model(x: np.ndarray, popt: tuple, model: str) -> np.ndarray:
    """Evaluate the single-peak model for plotting."""
    if model == "gaussian":
        return gauss(x, *popt)
    elif model == "emg":
        return emg(x, *popt)
    elif model == "bigauss":
        return bigauss(x, *popt)
    raise ValueError(f"Unknown model: {model}")


def _single_area(popt: tuple, model: str) -> float:
    """Compute the area of a single-peak fit."""
    if model == "gaussian":
        A, mu, sigma = popt
        return _gauss_area(A, sigma)
    elif model == "emg":
        A, mu, sigma, tau = popt
        return _emg_area(A, sigma, tau)
    elif model == "bigauss":
        A, mu, sigma_l, sigma_r = popt
        return _bigauss_area(A, sigma_l, sigma_r)
    raise ValueError(f"Unknown model: {model}")


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
    baseline: np.ndarray            # estimated linear baseline
    fit_type: str                   # "single" or "double"
    model_name: str = "gaussian"    # "gaussian", "emg", or "bigauss"
    chosen_popt: tuple = ()         # optimised parameters for chosen peak
    area: float = np.nan            # integrated area
    rtmed: float = 0.0              # expected RT

    # Legacy convenience (used by gauss-only path; kept for compat)
    @property
    def chosen_A(self) -> float:
        return self.chosen_popt[0] if self.chosen_popt else 0.0

    @property
    def chosen_mu(self) -> float:
        return self.chosen_popt[1] if len(self.chosen_popt) > 1 else 0.0


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

MODEL_LABELS = {
    "gaussian": "Gaussian",
    "emg": "EMG",
    "bigauss": "Bi-Gaussian",
}

def _export_fit_pdf(fit_results: list[FitResult], output_pdf: str) -> None:
    """Write all fit overlays to a multi-page PDF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

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

                ax.plot(fr.x, fr.y_raw, color="0.70", linewidth=0.8,
                        label="raw EIC")
                ax.plot(fr.x, fr.y_smoothed, color="black", linewidth=1.0,
                        label="smoothed EIC")

                # Baseline
                if fr.baseline is not None and len(fr.baseline) > 0:
                    ax.plot(fr.x, fr.baseline, color="tab:orange",
                            linestyle=":", linewidth=0.9, label="baseline")

                # Fitted model overlay
                if np.isfinite(fr.area) and fr.chosen_A > 0 and len(fr.chosen_popt) > 0:
                    x_fine = np.linspace(fr.x.min(), fr.x.max(), 500)
                    y_model = _eval_single_model(x_fine, fr.chosen_popt, fr.model_name)
                    bl_fine = np.interp(x_fine, fr.x, fr.baseline)
                    y_fit = y_model + bl_fine
                    label = f"fitted {MODEL_LABELS.get(fr.model_name, fr.model_name)}"
                    ax.fill_between(x_fine, bl_fine, y_fit, alpha=0.35,
                                    color="tab:blue", label=label)
                    ax.plot(x_fine, y_fit, color="tab:blue", linewidth=1.2)

                ax.axvline(fr.rtmed, color="tab:red", linestyle="--",
                           linewidth=0.8, label=f"expected RT ({fr.rtmed:.1f})")

                area_str = f"{fr.area:.2e}" if np.isfinite(fr.area) else "N/A"
                model_tag = MODEL_LABELS.get(fr.model_name, fr.model_name)
                ax.set_title(f"{fr.compound}  —  {fr.sample}  "
                             f"(area = {area_str}, {model_tag})", fontsize=10)
                ax.set_xlabel("RT (s)")
                ax.set_ylabel("Intensity")
                ax.legend(fontsize=7, loc="upper right")
                fig.tight_layout()

                pdf.savefig(fig)
                plt.close(fig)

    print(f"\nPDF saved to '{output_pdf}' ({sum(len(v) for v in grouped.values())} pages).")


# ════════════════════════════════════════════
#  Baseline Estimation
# ════════════════════════════════════════════

def _estimate_baseline(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Estimate baseline as a line between the low-intensity edges.

    Uses the 25th percentile of the outermost ~15% of points on each
    side, which is more robust to peak tails than the median.
    """
    n = len(y)
    if n < 5:
        return np.zeros_like(y)

    n_edge = max(3, n // 7)  # ~15% from each edge
    bl_left = np.percentile(y[:n_edge], 25)
    bl_right = np.percentile(y[-n_edge:], 25)
    baseline = np.linspace(bl_left, bl_right, n)

    # Never let baseline exceed the signal
    baseline = np.minimum(baseline, y)
    baseline = np.clip(baseline, 0, None)

    return baseline


# ════════════════════════════════════════════
#  Fitting Helpers
# ════════════════════════════════════════════

def _estimate_sigma(x: np.ndarray, y: np.ndarray, apex_idx: int) -> float:
    """Estimate sigma from the half-width at half-maximum of the peak."""
    half_max = y[apex_idx] / 2.0
    if half_max <= 0:
        return 5.0

    # Search left for the half-max crossing
    left_idx = apex_idx
    for i in range(apex_idx - 1, -1, -1):
        if y[i] <= half_max:
            left_idx = i
            break
    else:
        left_idx = 0

    # Search right for the half-max crossing
    right_idx = apex_idx
    for i in range(apex_idx + 1, len(y)):
        if y[i] <= half_max:
            right_idx = i
            break
    else:
        right_idx = len(y) - 1

    fwhm = x[right_idx] - x[left_idx]
    sigma = max(fwhm / 2.355, 1.0)  # FWHM ≈ 2.355 * sigma
    return sigma


def _fit_single_peak(
    x: np.ndarray, y: np.ndarray,
    apex_rt: float, apex_int: float,
    model: str,
    sigma_est: float = 5.0,
) -> tuple[tuple, float]:
    """Fit a single peak and return (popt, area).

    Raises on failure so the caller can catch and fall back.
    """
    sig_lo = max(0.3, sigma_est * 0.3)
    sig_hi = max(sigma_est * 5, 30)
    mu_tol = max(5, sigma_est * 2)   # allow mu to shift more for broad peaks

    if model == "gaussian":
        popt, _ = curve_fit(
            gauss, x, y,
            p0=[apex_int, apex_rt, sigma_est],
            bounds=([0, apex_rt - mu_tol, sig_lo],
                    [np.inf, apex_rt + mu_tol, sig_hi]),
            maxfev=10000,
        )
        return tuple(popt), _single_area(tuple(popt), model)

    elif model == "emg":
        popt, _ = curve_fit(
            emg, x, y,
            p0=[apex_int, apex_rt, sigma_est, sigma_est * 0.5],
            bounds=([0, apex_rt - mu_tol, sig_lo, 0.1],
                    [np.inf, apex_rt + mu_tol, sig_hi, sig_hi * 2]),
            maxfev=10000,
        )
        return tuple(popt), _single_area(tuple(popt), model)

    elif model == "bigauss":
        popt, _ = curve_fit(
            bigauss, x, y,
            p0=[apex_int, apex_rt, sigma_est, sigma_est],
            bounds=([0, apex_rt - mu_tol, sig_lo, sig_lo],
                    [np.inf, apex_rt + mu_tol, sig_hi, sig_hi]),
            maxfev=10000,
        )
        return tuple(popt), _single_area(tuple(popt), model)

    raise ValueError(f"Unknown model: {model}")


def _fit_double_peak(
    x: np.ndarray, y: np.ndarray,
    rt1: float, int1: float,
    rt2: float, int2: float,
    rtmin: float, rtmax: float, rtmed: float,
    model: str,
    sigma_est1: float = 5.0,
    sigma_est2: float = 5.0,
) -> tuple[tuple, float]:
    """Fit a double peak and return (chosen_popt, area) for the peak closest to rtmed."""
    sig_lo = max(0.3, min(sigma_est1, sigma_est2) * 0.3)
    sig_hi = max(max(sigma_est1, sigma_est2) * 5, 30)

    if model == "gaussian":
        popt, _ = curve_fit(
            double_gauss, x, y,
            p0=[int1, rt1, sigma_est1, int2, rt2, sigma_est2],
            bounds=([0, rtmin, sig_lo, 0, rtmin, sig_lo],
                    [np.inf, rtmax, sig_hi, np.inf, rtmax, sig_hi]),
            maxfev=10000,
        )
        A1, mu1, sigma1, A2, mu2, sigma2 = popt
        if abs(mu1 - rtmed) < abs(mu2 - rtmed):
            return (A1, mu1, sigma1), _single_area((A1, mu1, sigma1), model)
        else:
            return (A2, mu2, sigma2), _single_area((A2, mu2, sigma2), model)

    elif model == "emg":
        popt, _ = curve_fit(
            double_emg, x, y,
            p0=[int1, rt1, sigma_est1, sigma_est1 * 0.5,
                int2, rt2, sigma_est2, sigma_est2 * 0.5],
            bounds=([0, rtmin, sig_lo, 0.1, 0, rtmin, sig_lo, 0.1],
                    [np.inf, rtmax, sig_hi, sig_hi * 2,
                     np.inf, rtmax, sig_hi, sig_hi * 2]),
            maxfev=10000,
        )
        A1, mu1, s1, t1, A2, mu2, s2, t2 = popt
        if abs(mu1 - rtmed) < abs(mu2 - rtmed):
            return (A1, mu1, s1, t1), _single_area((A1, mu1, s1, t1), model)
        else:
            return (A2, mu2, s2, t2), _single_area((A2, mu2, s2, t2), model)

    elif model == "bigauss":
        popt, _ = curve_fit(
            double_bigauss, x, y,
            p0=[int1, rt1, sigma_est1, sigma_est1,
                int2, rt2, sigma_est2, sigma_est2],
            bounds=([0, rtmin, sig_lo, sig_lo, 0, rtmin, sig_lo, sig_lo],
                    [np.inf, rtmax, sig_hi, sig_hi,
                     np.inf, rtmax, sig_hi, sig_hi]),
            maxfev=10000,
        )
        A1, mu1, sl1, sr1, A2, mu2, sl2, sr2 = popt
        if abs(mu1 - rtmed) < abs(mu2 - rtmed):
            return (A1, mu1, sl1, sr1), _single_area((A1, mu1, sl1, sr1), model)
        else:
            return (A2, mu2, sl2, sr2), _single_area((A2, mu2, sl2, sr2), model)

    raise ValueError(f"Unknown model: {model}")


# ════════════════════════════════════════════
#  Integration Pipeline
# ════════════════════════════════════════════

def integrate_experiment(
    pkl_path: str = "../../experiment.pkl",
    output_csv: str = "results.csv",
    output_pdf: str | None = None,
    target_cmpds: list[str] | None = None,
    fit_model: str = "gaussian",
    subtract_baseline: bool = True,
    min_points: int = 11,
    savgol_window: int = 11,
    savgol_poly: int = 3,
    prominence_frac: float = 0.05,
) -> "pd.DataFrame":
    """Integrate peaks for all target compounds in a saved Experiment.

    Parameters:
        pkl_path:        Path to the pickled :class:`Experiment`.
        output_csv:      Path for the output CSV (``None`` to skip).
        output_pdf:      Path for the output PDF with model overlays
                         (``None`` to skip).
        target_cmpds:    Compounds to process. Defaults to :data:`TARGET_COMPOUNDS`.
        fit_model:       Peak model: ``"gaussian"`` (default), ``"emg"``
                         (exponentially modified Gaussian), or ``"bigauss"``
                         (bi-Gaussian / split Gaussian).
        subtract_baseline: If ``True`` (default), subtract a linear baseline
                         estimated from the window edges before fitting.
        min_points:      Minimum data points required for fitting.
        savgol_window:   Savitzky–Golay smoothing window length.
        savgol_poly:     Savitzky–Golay polynomial order.
        prominence_frac: Peak prominence threshold (fraction of max intensity).

    Returns:
        DataFrame with samples as rows, compounds as columns.
    """
    import pandas as pd

    fit_model = fit_model.lower()
    if fit_model not in VALID_MODELS:
        raise ValueError(f"fit_model must be one of {VALID_MODELS}, got '{fit_model}'")

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
                cut = np.argmin(abs(rtmed - x))
                x = x[cut:]
                y_s = y_s[cut:]
                y = y[cut:]
                peaks_indices = [0]  # apex is now at index 0 after slicing
                num_peaks = 1

            area_main = np.nan
            chosen_popt: tuple = ()

            # ── Baseline subtraction (optional) ──
            if subtract_baseline:
                baseline = _estimate_baseline(x, y_s)
                y_bc = y_s - baseline
                y_bc = np.clip(y_bc, 0, None)
            else:
                baseline = np.zeros_like(y_s)
                y_bc = y_s

            # Single peak
            if num_peaks == 1:
                apex_rt = x[peaks_indices[0]]
                apex_int = y_bc[peaks_indices[0]]
                sig_est = _estimate_sigma(x, y_bc, peaks_indices[0])
                try:
                    chosen_popt, area_main = _fit_single_peak(
                        x, y_bc, apex_rt, apex_int, fit_model,
                        sigma_est=sig_est,
                    )
                except Exception:
                    pass

            # Two+ peaks → double model, pick closest to expected RT
            elif num_peaks >= 2:
                top2 = sorted(peaks_indices, key=lambda i: y_bc[i], reverse=True)[:2]
                rt1, rt2 = x[top2[0]], x[top2[1]]
                int1, int2 = y_bc[top2[0]], y_bc[top2[1]]
                sig1 = _estimate_sigma(x, y_bc, top2[0])
                sig2 = _estimate_sigma(x, y_bc, top2[1])
                try:
                    chosen_popt, area_main = _fit_double_peak(
                        x, y_bc, rt1, int1, rt2, int2,
                        rtmin, rtmax, rtmed, fit_model,
                        sigma_est1=sig1, sigma_est2=sig2,
                    )
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
                    baseline=baseline.copy(),
                    fit_type="single" if num_peaks == 1 else "double",
                    model_name=fit_model,
                    chosen_popt=chosen_popt,
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
        help="Output PDF path for model overlay plots (default: None, no PDF)",
    )
    parser.add_argument(
        "--model", default="gaussian",
        choices=VALID_MODELS,
        help="Peak model: gaussian (default), emg, or bigauss",
    )
    parser.add_argument(
        "--no-baseline", action="store_true",
        help="Disable baseline subtraction before fitting",
    )
    args = parser.parse_args()

    integrate_experiment(
        pkl_path=args.pkl,
        output_csv=args.out,
        output_pdf=args.pdf,
        fit_model=args.model,
        subtract_baseline=not args.no_baseline,
    )


if __name__ == "__main__":
    main()
