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
VALID_BASELINE_SCOPES = ("narrow", "global")


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


def _as_trace_array(values: object) -> np.ndarray:
    """Return a 1-D float trace array, tolerating missing/scalar inputs."""
    if values is None:
        return np.array([], dtype=float)
    arr = np.asarray(values, dtype=float)
    return np.atleast_1d(arr)


# ════════════════════════════════════════════
#  Fit Result Container
# ════════════════════════════════════════════

@dataclass
class FitResult:
    """Stores everything needed to plot a single fit overlay."""
    compound: str
    eic_name: str
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
    x_fit: np.ndarray = field(default_factory=lambda: np.array([]))  # sub-window used for fitting

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
    """Write one PDF page per sample with all integrated peaks overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    from collections import OrderedDict
    grouped: dict[str, list[FitResult]] = OrderedDict()
    for fr in fit_results:
        grouped.setdefault(fr.sample, []).append(fr)
    for v in grouped.values():
        v.sort(key=lambda fr: (fr.eic_name, fr.compound))

    with PdfPages(output_pdf) as pdf:
        for sample, results in grouped.items():
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue"])
            eic_groups: dict[str, list[FitResult]] = OrderedDict()
            for fr in results:
                eic_groups.setdefault(fr.eic_name, []).append(fr)

            expected_line_drawn = False
            for idx, (eic_name, eic_results) in enumerate(eic_groups.items()):
                color = colors[idx % len(colors)]
                base = eic_results[0]

                ax.plot(base.x, base.y_raw, color=color, linewidth=0.7,
                        alpha=0.18, label=f"{eic_name} raw")
                ax.plot(base.x, base.y_smoothed, color=color, linewidth=1.0,
                        alpha=0.45, label=f"{eic_name} smoothed")

                if base.baseline is not None and len(base.baseline) > 0:
                    ax.plot(base.x, base.baseline, color=color,
                            linestyle=":", linewidth=0.8, alpha=0.6,
                            label=f"{eic_name} baseline")

                if not expected_line_drawn:
                    ax.axvline(base.rtmed, color="tab:red", linestyle="--",
                               linewidth=0.8, label=f"expected RT ({base.rtmed:.1f})")
                    expected_line_drawn = True

                for fr in eic_results:
                    if np.isfinite(fr.area) and fr.chosen_A > 0 and len(fr.chosen_popt) > 0:
                        x_range = fr.x_fit if len(fr.x_fit) >= 2 else fr.x
                        x_fine = np.linspace(x_range.min(), x_range.max(), 500)
                        y_model = _eval_single_model(x_fine, fr.chosen_popt, fr.model_name)
                        bl_fine = np.interp(x_fine, fr.x, fr.baseline)
                        y_fit = y_model + bl_fine
                        model_tag = MODEL_LABELS.get(fr.model_name, fr.model_name)
                        area_str = f"{fr.area:.2e}"
                        label = f"{fr.compound} ({model_tag}, {area_str})"
                        ax.fill_between(x_fine, bl_fine, y_fit, alpha=0.20,
                                        color=color, label=label)
                        ax.plot(x_fine, y_fit, color=color, linewidth=1.3)
                        ax.axvline(fr.chosen_mu, color=color, linestyle="--", linewidth=0.8)

            ax.set_title(sample, fontsize=10)
            ax.set_xlabel("RT (s)")
            ax.set_ylabel("Intensity")
            ax.legend(fontsize=7, loc="upper right")
            fig.tight_layout()

            pdf.savefig(fig)
            plt.close(fig)

    print(f"\nPDF saved to '{output_pdf}' ({len(grouped)} pages).")


def build_sample_overlay_figures(fit_results: list[FitResult]) -> dict[str, object]:
    """Build one interactive Plotly figure per sample."""
    import plotly.graph_objects as go
    from collections import OrderedDict

    grouped: dict[str, list[FitResult]] = OrderedDict()
    for fr in fit_results:
        grouped.setdefault(fr.sample, []).append(fr)
    for v in grouped.values():
        v.sort(key=lambda fr: (fr.eic_name, fr.compound))

    figures: dict[str, object] = {}
    colors = [
        "#0f766e", "#c2410c", "#1d4ed8", "#b91c1c", "#6d28d9",
        "#15803d", "#be185d", "#7c2d12", "#0369a1", "#4d7c0f",
    ]

    for sample, results in grouped.items():
        fig = go.Figure()
        eic_groups: dict[str, list[FitResult]] = OrderedDict()
        for fr in results:
            eic_groups.setdefault(fr.eic_name, []).append(fr)

        for idx, (eic_name, eic_results) in enumerate(eic_groups.items()):
            color = colors[idx % len(colors)]
            base = eic_results[0]

            fig.add_trace(go.Scatter(
                x=base.x, y=base.y_raw,
                mode="lines",
                name=f"{eic_name} raw",
                line=dict(color=color, width=1),
                opacity=0.18,
                legendgroup=eic_name,
                hovertemplate=f"{eic_name} raw<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=base.x, y=base.y_smoothed,
                mode="lines",
                name=f"{eic_name} smoothed",
                line=dict(color=color, width=2),
                opacity=0.45,
                legendgroup=eic_name,
                hovertemplate=f"{eic_name} smoothed<extra></extra>",
            ))

            if base.baseline is not None and len(base.baseline) > 0:
                fig.add_trace(go.Scatter(
                    x=base.x, y=base.baseline,
                    mode="lines",
                    name=f"{eic_name} baseline",
                    line=dict(color=color, width=1, dash="dot"),
                    opacity=0.7,
                    legendgroup=eic_name,
                    hovertemplate=f"{eic_name} baseline<extra></extra>",
                ))

            for fr_idx, fr in enumerate(eic_results):
                if not (np.isfinite(fr.area) and fr.chosen_A > 0 and len(fr.chosen_popt) > 0):
                    continue
                fit_color = colors[(idx + fr_idx) % len(colors)]
                x_range = fr.x_fit if len(fr.x_fit) >= 2 else fr.x
                x_fine = np.linspace(x_range.min(), x_range.max(), 500)
                y_model = _eval_single_model(x_fine, fr.chosen_popt, fr.model_name)
                bl_fine = np.interp(x_fine, fr.x, fr.baseline)
                y_fit = y_model + bl_fine
                fill_x = np.concatenate([x_fine, x_fine[::-1]])
                fill_y = np.concatenate([y_fit, bl_fine[::-1]])
                fig.add_trace(go.Scatter(
                    x=fill_x, y=fill_y,
                    mode="lines",
                    fill="toself",
                    fillcolor=fit_color,
                    opacity=0.18,
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=eic_name,
                ))
                fig.add_trace(go.Scatter(
                    x=x_fine, y=y_fit,
                    mode="lines",
                    name=fr.compound,
                    line=dict(color=fit_color, width=2.5),
                    legendgroup=eic_name,
                    hovertemplate=f"{fr.compound}<extra></extra>",
                ))
                peak_y = float(np.nanmax(y_fit))
                fig.add_trace(go.Scatter(
                    x=[fr.chosen_mu, fr.chosen_mu],
                    y=[peak_y, peak_y],
                    mode="lines",
                    line=dict(color=fit_color, width=1, dash="dash"),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=eic_name,
                ))
                fig.add_annotation(
                    x=fr.chosen_mu,
                    y=0.98,
                    text=fr.compound,
                    showarrow=False,
                    textangle=-90,
                    font=dict(color=fit_color, size=12),
                    xanchor="center",
                    yanchor="top",
                    yref="paper",
                )

        fig.update_layout(
            title=sample,
            xaxis_title="RT (s)",
            yaxis_title="Intensity",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="v"),
            margin=dict(l=40, r=20, t=110, b=40),
        )
        figures[sample] = fig

    return figures


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


def _smooth_trace(y: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    """Smooth a trace when enough points are available, else return it unchanged."""
    if len(y) < window_length:
        return y.copy()
    return savgol_filter(y, window_length=window_length, polyorder=polyorder)


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
#  Peak Sub-window Helper
# ════════════════════════════════════════════

def _find_peak_subwindow(
    y: np.ndarray, apex_idx: int, all_peak_indices: list[int]
) -> tuple[int, int]:
    """Return (left_idx, right_idx) bounding the sub-window for one peak.

    Boundaries are placed at the valley between this peak and its nearest
    neighbours. If there is no neighbour on a side the boundary is the edge
    of the array.
    """
    sorted_peaks = sorted(all_peak_indices)
    pos = sorted_peaks.index(apex_idx)

    if pos > 0:
        prev_peak = sorted_peaks[pos - 1]
        left_i = prev_peak + int(np.argmin(y[prev_peak : apex_idx + 1]))
    else:
        left_i = 0

    if pos < len(sorted_peaks) - 1:
        next_peak = sorted_peaks[pos + 1]
        right_i = apex_idx + int(np.argmin(y[apex_idx : next_peak + 1]))
    else:
        right_i = len(y) - 1

    return left_i, right_i


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
    baseline_scope: str = "narrow",
    min_points: int = 11,
    savgol_window: int = 11,
    savgol_poly: int = 3,
    prominence_frac: float = 0.05,
    propagate_consensus_splits: bool = True,
    experiment: "Experiment | None" = None,
    return_fit_results: bool = False,
) -> "pd.DataFrame | tuple[pd.DataFrame, list[FitResult]]":
    """Integrate peaks for all target compounds in a saved Experiment.

    Parameters:
        pkl_path:        Path to the pickled :class:`Experiment`.
                         Ignored when ``experiment`` is provided.
        output_csv:      Path for the output CSV (``None`` to skip).
        output_pdf:      Path for the output PDF with model overlays
                         (``None`` to skip).
        target_cmpds:    Compounds to process. Defaults to :data:`TARGET_COMPOUNDS`.
        fit_model:       Peak model: ``"gaussian"`` (default), ``"emg"``
                         (exponentially modified Gaussian), or ``"bigauss"``
                         (bi-Gaussian / split Gaussian).
        subtract_baseline: If ``True`` (default), subtract a linear baseline
                         estimated from the window edges before fitting.
        baseline_scope:  Baseline estimation window. ``"narrow"`` uses the
                         compound RT window only; ``"global"`` uses the full
                         chromatogram trace and interpolates onto the fitting
                         window.
        min_points:      Minimum data points required for fitting.
        savgol_window:   Savitzky–Golay smoothing window length.
        savgol_poly:     Savitzky–Golay polynomial order.
        prominence_frac: Peak prominence threshold (fraction of max intensity).
        propagate_consensus_splits:
                         If ``True``, multi-peak detections in some samples can
                         propagate to other samples of the same compound via the
                         compound-wide consensus logic.
        experiment:      An already-loaded :class:`Experiment` object. When
                         supplied, ``pkl_path`` is not used. Pass this from
                         in-process callers (e.g. the Streamlit app) to avoid
                         pickle class-identity mismatches.
        return_fit_results:
                         If ``True``, also return the per-fit metadata used for
                         plotting interactive overlays.

    Returns:
        DataFrame with samples as rows, compounds as columns.
    """
    import pandas as pd

    fit_model = fit_model.lower()
    if fit_model not in VALID_MODELS:
        raise ValueError(f"fit_model must be one of {VALID_MODELS}, got '{fit_model}'")
    baseline_scope = baseline_scope.lower()
    if baseline_scope not in VALID_BASELINE_SCOPES:
        raise ValueError(
            f"baseline_scope must be one of {VALID_BASELINE_SCOPES}, got '{baseline_scope}'"
        )

    if target_cmpds is None:
        target_cmpds = TARGET_COMPOUNDS

    if experiment is not None:
        exp: Experiment = experiment
    else:
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

        # ══════════════════════════════════════════════════════════════════════
        #  Pass 1 — independent peak detection per sample
        # ══════════════════════════════════════════════════════════════════════
        # Stores raw signal arrays keyed by sample; fitting happens in Pass 2
        # after the compound-wide consensus peak count is established.
        per_sample_peaks: dict[str, list] = {}

        for sample_name, chrom_obj in exp.chromatograms.items():

            matching_eic = next(
                (eic for eic in chrom_obj.eics if cmpd.startswith(eic.name)),
                None,
            )
            if matching_eic is None:
                per_sample_peaks[sample_name] = []   # sentinel: no EIC
                continue

            rt = _as_trace_array(
                matching_eic.shifted_rt
                if matching_eic.shifted_rt is not None
                else matching_eic.rt
            )
            intensity = _as_trace_array(matching_eic.intensity)

            if rt.size == 0 or intensity.size == 0 or rt.size != intensity.size:
                per_sample_peaks[sample_name] = []   # invalid trace payload
                continue

            finite_mask = np.isfinite(rt) & np.isfinite(intensity)
            rt_full = rt[finite_mask]
            intensity_full = intensity[finite_mask]
            if len(rt_full) == 0 or len(intensity_full) == 0:
                per_sample_peaks[sample_name] = []   # no finite signal
                continue

            if len(rt_full) > 1:
                order = np.argsort(rt_full)
                rt_full = rt_full[order]
                intensity_full = intensity_full[order]

            mask = (rt_full > rtmin) & (rt_full < rtmax)
            x = rt_full[mask]
            y = intensity_full[mask]
            if len(x) == 0 or len(y) == 0:
                per_sample_peaks[sample_name] = []   # no signal in RT window
                continue

            trace_smoothed = _smooth_trace(intensity_full, savgol_window, savgol_poly)

            if len(y) < savgol_window:
                y_s = y.copy()
                peaks_indices = np.array([int(np.argmax(y))])
                num_peaks = 1
            else:
                y_s = _smooth_trace(y, savgol_window, savgol_poly)

                num_peaks = 0
                n_iter = 0
                while (num_peaks == 0) and (len(x) >= min_points):
                    if n_iter > 0:
                        x = x[1:]
                        y = y[1:]
                        y_s = y_s[1:]
                    n_iter += 1
                    if len(x) < min_points:
                        break
                    max_intensity = y_s.max()
                    if max_intensity <= 0:
                        break
                    peaks_indices, _ = find_peaks(y_s, prominence=max_intensity * prominence_frac)
                    num_peaks = len(peaks_indices)

                if num_peaks == 0:
                    # Fallback: treat the point nearest rtmed as the apex
                    x = rt_full[mask]
                    y = intensity_full[mask]
                    y_s = _smooth_trace(y, savgol_window, savgol_poly)
                    cut = int(np.argmin(np.abs(rtmed - x)))
                    x = x[cut:]; y = y[cut:]; y_s = y_s[cut:]
                    peaks_indices = np.array([0])
                    num_peaks = 1

            # ── Baseline subtraction ──
            if subtract_baseline:
                if baseline_scope == "global":
                    baseline_full = _estimate_baseline(rt_full, trace_smoothed)
                    baseline = np.interp(x, rt_full, baseline_full)
                    baseline = np.minimum(baseline, y_s)
                    baseline = np.clip(baseline, 0, None)
                else:
                    baseline = _estimate_baseline(x, y_s)
                y_bc = np.clip(y_s - baseline, 0, None)
            else:
                baseline = np.zeros_like(y_s)
                y_bc = y_s

            # Store everything needed for Pass 2
            # atleast_1d guards against 0-d peaks_indices from edge cases
            per_sample_peaks[sample_name] = [
                num_peaks,
                np.atleast_1d(peaks_indices).astype(int),
                x, y, y_s, y_bc, baseline,
            ]

        # ══════════════════════════════════════════════════════════════════════
        #  Consensus — decide compound-wide peak count from all samples
        # ══════════════════════════════════════════════════════════════════════

        detected_counts = [
            data[0] for data in per_sample_peaks.values() if data
        ]

        consensus_n = 1
        if detected_counts:
            if propagate_consensus_splits:
                # If ≥15 % of samples detect N > 1 peaks, treat ALL samples as having
                # N peaks. This propagates the split from clearly-resolved samples to
                # those where the valley between peaks is too shallow to trigger
                # find_peaks independently.
                max_n = max(detected_counts)
                frac_max = sum(1 for c in detected_counts if c == max_n) / len(detected_counts)
                if max_n >= 2 and frac_max >= 0.15:
                    consensus_n = max_n
                else:
                    from collections import Counter
                    consensus_n = Counter(detected_counts).most_common(1)[0][0]
            else:
                consensus_n = 1

        if consensus_n == 1:
            consensus_labels = [""]
        else:
            consensus_labels = [f"_{i + 1}" for i in range(consensus_n)]

        # Collect apex RTs from samples that match the consensus count so we
        # can estimate the median valley position used to force-split others.
        ref_apex_rts: list[list[float]] = [[] for _ in range(consensus_n)]
        for data in per_sample_peaks.values():
            if not data:
                continue
            n_det, pk_idx, x_s = data[0], data[1], data[2]
            if n_det != consensus_n:
                continue
            sorted_pi = sorted(int(i) for i in pk_idx)
            for j, pi in enumerate(sorted_pi[:consensus_n]):
                ref_apex_rts[j].append(float(x_s[pi]))

        median_apex_rts: list[float | None] = [
            float(np.median(rts)) if rts else None for rts in ref_apex_rts
        ]
        # One valley RT between each consecutive pair of expected apices
        median_valley_rts: list[float | None] = [
            (median_apex_rts[j] + median_apex_rts[j + 1]) / 2  # type: ignore[operator]
            if (median_apex_rts[j] is not None and median_apex_rts[j + 1] is not None)
            else None
            for j in range(len(median_apex_rts) - 1)
        ]

        print(
            f"  consensus_n={consensus_n}  "
            f"median_apices={[f'{r:.1f}' if r is not None else 'N/A' for r in median_apex_rts]}"
        )

        # ══════════════════════════════════════════════════════════════════════
        #  Pass 2 — fit every sample using the consensus layout
        # ══════════════════════════════════════════════════════════════════════

        final_per_sample: dict[str, list[tuple]] = {}

        for sample_name, data in per_sample_peaks.items():

            # No EIC found for this sample
            if not data:
                final_per_sample[sample_name] = [("", np.nan, (), np.array([]))]
                continue

            num_peaks, peaks_indices, x, y, y_s, y_bc, baseline = data
            peaks_indices = np.atleast_1d(peaks_indices).astype(int)

            # ── Single-peak compound ──────────────────────────────────────────
            if consensus_n == 1:
                pi = peaks_indices[0]
                sig_est = _estimate_sigma(x, y_bc, pi)
                popt: tuple = ()
                area = np.nan
                try:
                    popt, area = _fit_single_peak(
                        x, y_bc, x[pi], y_bc[pi], fit_model, sigma_est=sig_est,
                    )
                except Exception:
                    pass
                final_per_sample[sample_name] = [("", area, popt, x.copy())]
                if output_pdf is not None or return_fit_results:
                    fit_results.append(FitResult(
                        compound=cmpd, eic_name=cmpd, sample=sample_name,
                        x=x.copy(), y_raw=y.copy(), y_smoothed=y_s.copy(),
                        baseline=baseline.copy(), fit_type="single",
                        model_name=fit_model, chosen_popt=popt,
                        area=area, rtmed=rtmed, x_fit=x.copy(),
                    ))

            # ── Multi-peak compound ───────────────────────────────────────────
            else:
                # Determine apex indices for this sample under the consensus layout.

                if num_peaks >= consensus_n:
                    # Greedy match: assign each reference apex its nearest detected peak.
                    remaining: list[int] = sorted(int(i) for i in peaks_indices)
                    assigned: list[int] = []
                    for ref_rt in [r for r in median_apex_rts if r is not None]:
                        if not remaining:
                            break
                        best = int(min(remaining, key=lambda i: abs(x[i] - ref_rt)))
                        assigned.append(best)
                        remaining.remove(best)
                    sorted_pkidxs: list[int] = sorted(assigned)

                else:
                    # Fewer peaks detected — use median valley RTs to force-split
                    # the window into consensus_n regions, then find the local apex
                    # (argmax of y_bc) within each region.
                    split_pts = [0]
                    for vrt in median_valley_rts:
                        vidx = int(np.argmin(np.abs(x - vrt))) if vrt is not None else len(x) // 2
                        split_pts.append(vidx)
                    split_pts.append(len(x))

                    sorted_pkidxs = []
                    for k in range(len(split_pts) - 1):
                        seg = y_bc[split_pts[k] : split_pts[k + 1]]
                        if len(seg) > 0:
                            sorted_pkidxs.append(split_pts[k] + int(np.argmax(seg)))
                        elif median_apex_rts[k] is not None:
                            sorted_pkidxs.append(int(np.argmin(np.abs(x - median_apex_rts[k]))))

                # Fit each sub-peak in its valley-bounded sub-window
                sample_entries: list[tuple] = []
                for pk_idx, label in zip(sorted_pkidxs, consensus_labels):
                    left_i, right_i = _find_peak_subwindow(
                        y_bc, int(pk_idx), [int(i) for i in sorted_pkidxs]
                    )
                    x_sub = x[left_i : right_i + 1]
                    y_sub = y_bc[left_i : right_i + 1]
                    local_idx = int(pk_idx) - left_i
                    sig_est = _estimate_sigma(x_sub, y_sub, local_idx)

                    popt = ()
                    area = np.nan
                    try:
                        popt, area = _fit_single_peak(
                            x_sub, y_sub, x[pk_idx], y_bc[pk_idx],
                            fit_model, sigma_est=sig_est,
                        )
                    except Exception:
                        pass

                    sample_entries.append((label, area, popt, x_sub.copy()))

                    if output_pdf is not None or return_fit_results:
                        fit_results.append(FitResult(
                            compound=cmpd + label,
                            eic_name=cmpd,
                            sample=sample_name,
                            x=x.copy(),
                            y_raw=y.copy(),
                            y_smoothed=y_s.copy(),
                            baseline=baseline.copy(),
                            fit_type="multi",
                            model_name=fit_model,
                            chosen_popt=popt,
                            area=area,
                            rtmed=rtmed,
                            x_fit=x_sub.copy(),
                        ))

                final_per_sample[sample_name] = sample_entries

        # ── Aggregate into all_results ────────────────────────────────────────
        all_labels_used: set[str] = set()
        for entries in final_per_sample.values():
            for label, *_ in entries:
                all_labels_used.add(label)

        for label in all_labels_used:
            key = cmpd + label
            all_results.setdefault(key, {})
            for sample_name, entries in final_per_sample.items():
                match = [a for (lbl, a, *_) in entries if lbl == label]
                all_results[key][sample_name] = match[0] if match else np.nan

    results_df = pd.DataFrame(all_results)
    results_df.index.name = "Sample"
    results_df = results_df.sort_index()

    if output_csv:
        results_df.to_csv(output_csv)
        print(f"\nResults saved to '{output_csv}'.")

    if output_pdf and fit_results:
        _export_fit_pdf(fit_results, output_pdf)

    if return_fit_results:
        return results_df, fit_results
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
    parser.add_argument(
        "--baseline-scope", default="narrow",
        choices=VALID_BASELINE_SCOPES,
        help="Baseline window: narrow uses the compound RT window; global uses the full trace",
    )
    args = parser.parse_args()

    integrate_experiment(
        pkl_path=args.pkl,
        output_csv=args.out,
        output_pdf=args.pdf,
        fit_model=args.model,
        subtract_baseline=not args.no_baseline,
        baseline_scope=args.baseline_scope,
    )


if __name__ == "__main__":
    main()
