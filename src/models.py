"""
models.py — Core data model for chromatographic peak integration.

Classes:
    PickedPeak   — A single picked peak with RT and area metadata.
    EIC          — Extracted ion chromatogram with associated picked peaks.
    Chromatogram — Collection of EICs for one sample, with dict-indexed lookup.
    Experiment   — Collection of Chromatograms across samples.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ════════════════════════════════════════════
#  Data Classes
# ════════════════════════════════════════════

@dataclass(slots=True)
class PickedPeak:
    """A single picked chromatographic peak.

    Attributes:
        name:   Compound name (e.g. ``'brGDGT_IIa'``).
        rt:     Retention time of peak apex (seconds).
        rtmin:  Left boundary of the peak (seconds).
        rtmax:  Right boundary of the peak (seconds).
        into:   Integrated peak area.
        intb:   Baseline-corrected peak area.
        sigma:  Gaussian sigma estimate of peak width.
    """
    name: str
    rt: float
    rtmin: float
    rtmax: float
    into: float
    intb: float
    sigma: float


@dataclass(slots=True)
class EIC:
    """Extracted Ion Chromatogram for a single compound.

    Attributes:
        name:        Compound name (used as lookup key).
        mz:          Target m/z value (may be ``None`` if not applicable).
        rt:          Raw retention-time axis (numpy array, seconds).
        intensity:   Intensity axis (numpy array).
        picked:      List of :class:`PickedPeak` objects found in this EIC.
        shifted_rt:  RT axis after retention-time correction (``None`` until
                     :meth:`Chromatogram.shift_rt` is called).
    """
    name: str
    mz: Optional[float]
    rt: np.ndarray
    intensity: np.ndarray
    picked: list[PickedPeak]
    shifted_rt: Optional[np.ndarray] = None


# ════════════════════════════════════════════
#  Chromatogram
# ════════════════════════════════════════════

class Chromatogram:
    """Collection of EICs for a single sample.

    Provides O(1) compound-name → EIC lookup via an internal index.

    Parameters:
        eics: Optional list of :class:`EIC` objects to initialise with.
    """

    def __init__(self, eics: Optional[list[EIC]] = None):
        self.eics: list[EIC] = eics if eics is not None else []
        self._eic_index: dict[str, EIC] = {eic.name: eic for eic in self.eics}

    # ---- EIC access ----

    def get_eic(self, name: str) -> list[EIC]:
        """Return EICs whose name is a prefix of *name* (O(1) for exact match).

        Falls back to a linear scan when no exact match is found so that
        compound variants (e.g. ``brGDGT_IIa_0``) still match the parent
        EIC ``brGDGT_IIa``.
        """
        eic = self._eic_index.get(name)
        if eic is not None:
            return [eic]
        return [eic for eic in self.eics if name.startswith(eic.name)]

    def add_eic(self, eic: EIC) -> None:
        """Append an EIC and update the internal index."""
        self.eics.append(eic)
        self._eic_index[eic.name] = eic

    # ---- Peak access ----

    def get_peaks(self, name: str) -> list[PickedPeak]:
        """Return all picked peaks whose name matches *name*."""
        peaks: list[PickedPeak] = []
        for eic in self.eics:
            peaks.extend(p for p in eic.picked if p.name == name)
        return peaks

    def get_rt(self) -> list[float]:
        """Return RT values of all picked peaks across all EICs."""
        return [p.rt for eic in self.eics for p in eic.picked]

    def get_into(self) -> list[float]:
        """Return integrated intensities of all picked peaks."""
        return [p.into for eic in self.eics for p in eic.picked]

    def get_min_rt_by_cmpd(self, cmpd: str) -> float:
        """Return minimum ``rtmin`` of peaks matching *cmpd*."""
        vals = [p.rtmin for eic in self.eics for p in eic.picked if p.name == cmpd]
        return min(vals) if vals else np.nan

    def get_max_rt_by_cmpd(self, cmpd: str) -> float:
        """Return maximum ``rtmax`` of peaks matching *cmpd*."""
        vals = [p.rtmax for eic in self.eics for p in eic.picked if p.name == cmpd]
        return max(vals) if vals else np.nan

    def pop(self, cmpd_name: str) -> list[PickedPeak]:
        """Remove and return all peaks with the given compound name."""
        removed: list[PickedPeak] = []
        for eic in self.eics:
            keep = []
            for p in eic.picked:
                if p.name == cmpd_name:
                    removed.append(p)
                else:
                    keep.append(p)
            eic.picked = keep
        return removed

    # ---- RT correction ----

    def shift_rt(self, poly: np.poly1d) -> None:
        """Apply a polynomial RT correction to every EIC and its peaks."""
        for eic in self.eics:
            if eic.shifted_rt is None:
                eic.shifted_rt = eic.rt.copy()

            shift_vals = poly(eic.shifted_rt)
            eic.shifted_rt = eic.shifted_rt + shift_vals

            for peak in eic.picked:
                shift = poly(peak.rt)
                peak.rt += shift
                peak.rtmin += shift
                peak.rtmax += shift

    # ---- Diagnostics ----

    def summary(self) -> dict:
        """Return a brief summary dict with EIC and peak counts."""
        return {
            "n_eics": len(self.eics),
            "n_peaks": sum(len(eic.picked) for eic in self.eics),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return f"Chromatogram(EICs={s['n_eics']}, Peaks={s['n_peaks']})"


# ════════════════════════════════════════════
#  Experiment
# ════════════════════════════════════════════

class Experiment:
    """Top-level container: maps sample names to Chromatogram objects.

    Parameters:
        chromatograms: ``dict[str, Chromatogram]`` keyed by sample name.
    """

    def __init__(self, chromatograms: dict[str, Chromatogram]):
        self.chromatograms = chromatograms
        self.rt_corrected: bool = False
        self.rt_model: Optional[np.poly1d] = None

    # ---- Dict-like helpers ----

    def keys(self):
        return self.chromatograms.keys()

    def items(self):
        return self.chromatograms.items()

    def values(self):
        return self.chromatograms.values()

    def get_sample_names(self) -> list[str]:
        """Return a list of all sample names."""
        return list(self.chromatograms.keys())

    get_sample_name = get_sample_names  # backward compat

    def __getitem__(self, key: str) -> Chromatogram:
        return self.chromatograms[key]

    # ---- RT query ----

    def get_rt(self, compound: str) -> dict[str, float]:
        """Aggregate RT statistics for *compound* across all samples.

        Returns:
            dict with keys ``rtmin``, ``rtmax``, ``rtmed``.
        """
        rt_vals: list[float] = []
        rtmin_vals: list[float] = []
        rtmax_vals: list[float] = []

        for chrom in self.chromatograms.values():
            eics = chrom.get_eic(compound)
            if not eics:
                continue
            eic = eics[0]
            for peak in eic.picked:
                if peak.name == compound:
                    rt_vals.append(peak.rt)
                    rtmin_vals.append(peak.rtmin)
                    rtmax_vals.append(peak.rtmax)

        if not rt_vals:
            return {"rtmin": np.nan, "rtmax": np.nan, "rtmed": np.nan}

        return {
            "rtmin": float(np.nanmin(rtmin_vals)),
            "rtmax": float(np.nanmax(rtmax_vals)),
            "rtmed": float(np.nanmedian(rt_vals)),
        }

    # ---- Plotting ----

    def plot_picked_peaks(self) -> None:
        """Interactive scatter plot of all picked peaks (RT vs. Area)."""
        import plotly.graph_objects as go

        groups: dict[str, dict] = {}

        for sample_name, chrom in self.chromatograms.items():
            for eic in chrom.eics:
                for peak in eic.picked:
                    groups.setdefault(peak.name, {"rt": [], "into": [], "sample": []})
                    groups[peak.name]["rt"].append(peak.rt)
                    groups[peak.name]["into"].append(peak.into)
                    groups[peak.name]["sample"].append(sample_name)

        fig = go.Figure()
        for name, g in groups.items():
            fig.add_trace(go.Scatter(
                x=g["rt"],
                y=g["into"],
                mode="markers",
                name=name,
                customdata=g["sample"],
                hovertemplate="rt=%{x:.2f}<br>into=%{y:.2e}<br>sample=%{customdata}<extra></extra>",
            ))

        fig.update_layout(
            xaxis_title="RT (s)",
            yaxis_title="Integrated area",
            template="simple_white",
        )
        fig.show()

    def plot_eic(self, compound_name: str, corrected: bool = True) -> None:
        """Overlay EIC traces for *compound_name* across all samples."""
        import plotly.graph_objects as go

        fig = go.Figure()

        for sample_name, chrom in self.chromatograms.items():
            eics = chrom.get_eic(compound_name)
            if not eics:
                continue
            for eic in eics:
                rt_axis = (
                    eic.shifted_rt
                    if corrected and eic.shifted_rt is not None
                    else eic.rt
                )
                fig.add_trace(go.Scatter(
                    x=rt_axis,
                    y=eic.intensity,
                    mode="lines",
                    name=sample_name,
                    hovertemplate=(
                        f"sample={sample_name}"
                        "<br>rt=%{x:.2f}"
                        "<br>intensity=%{y:.2e}"
                        "<extra></extra>"
                    ),
                ))

        fig.update_layout(
            title=f"EIC: {compound_name}",
            xaxis_title="RT (s)",
            yaxis_title="Intensity",
            template="simple_white",
        )
        fig.show()

    # ---- RT Correction ----

    def rt_shift(
        self,
        calibs: Optional[list[str]] = None,
        more_calibs: Optional[list[str]] = None,
        degree: int = 2,
        ref_sample_name: Optional[str] = None,
        manual_anchors: Optional[dict[str, list[tuple[float, float]]]] = None,
    ) -> "Experiment":
        """Create a deep copy of this experiment with polynomial RT correction.

        Parameters:
            calibs:          Calibration compounds. Defaults to
                             ``['C46-GDGT', 'brGDGT_Ib', 'brGDGT_Ia']``.
            more_calibs:     Additional calibration compounds to append.
            degree:          Polynomial degree for the correction model.
            ref_sample_name: Reference sample. Defaults to the first sample.
            manual_anchors:  ``{sample: [(observed_rt, target_rt), ...]}``

        Returns:
            A new :class:`Experiment` with corrected RTs.
        """
        calibs = ["C46-GDGT", "brGDGT_Ib", "brGDGT_Ia"] if calibs is None else list(calibs)
        if more_calibs:
            calibs.extend(more_calibs)

        shifted_exp = copy.deepcopy(self)

        ref_sample = (
            list(shifted_exp.chromatograms.keys())[0]
            if ref_sample_name is None
            else ref_sample_name
        )

        ref_rts: list[float] = []
        for calib in calibs:
            peaks = shifted_exp.chromatograms[ref_sample].get_peaks(calib)
            if not peaks:
                raise ValueError(f"Missing calibration peak '{calib}' in reference sample '{ref_sample}'")
            ref_rts.append(peaks[0].rt)

        poly = None
        for sample_name, chrom in shifted_exp.chromatograms.items():
            obs_rts: list[float] = []
            rt_diffs: list[float] = []

            for idx, calib in enumerate(calibs):
                peaks = chrom.get_peaks(calib)
                if not peaks:
                    continue
                obs_rt = peaks[0].rt
                obs_rts.append(obs_rt)
                rt_diffs.append(ref_rts[idx] - obs_rt)

            if manual_anchors and sample_name in manual_anchors:
                for observed_rt, target_rt in manual_anchors[sample_name]:
                    obs_rts.append(observed_rt)
                    rt_diffs.append(target_rt - observed_rt)
                    print(f"  Manual anchor → {sample_name}: {observed_rt} → {target_rt}")

            if len(obs_rts) < degree + 1:
                continue

            coef = np.polyfit(obs_rts, rt_diffs, degree)
            poly = np.poly1d(coef)
            chrom.shift_rt(poly)

        shifted_exp.rt_corrected = True
        shifted_exp.rt_model = poly
        return shifted_exp

    # ---- Peak Clustering ----

    @staticmethod
    def find_optimal_clusters(
        rt_vals: np.ndarray,
        max_k: int = 6,
    ) -> int:
        """Determine the optimal number of clusters using silhouette score.

        Parameters:
            rt_vals: 1-D array of retention times (will be reshaped if needed).
            max_k:   Maximum number of clusters to try.

        Returns:
            Optimal ``k`` (between 2 and ``max_k``).
        """
        X = rt_vals.reshape(-1, 1) if rt_vals.ndim == 1 else rt_vals
        n_samples = len(X)

        if n_samples < 3:
            return 1

        # Upper bound: can't have more clusters than samples
        max_k = min(max_k, n_samples - 1)
        if max_k < 2:
            return 1

        best_k = 1
        best_score = 0.45  # minimum silhouette threshold to justify k≥2

        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=0, n_init="auto")
            labels = km.fit_predict(X)

            # Silhouette score needs at least 2 distinct labels
            if len(set(labels)) < 2:
                continue

            score = silhouette_score(X, labels)
            print(f"    k={k}: silhouette={score:.3f}")
            if score > best_score:
                best_score = score
                best_k = k

        return best_k

    def point_cluster_batch(
        self,
        compounds: dict[str, int | str] | list[str],
        clusters: Optional[list[int | str]] = None,
    ) -> "Experiment":
        """Run KMeans peak-clustering for multiple compounds.

        Parameters:
            compounds: ``{compound: n_clusters}`` dict or a list of names.
                       Use ``0`` or ``"auto"`` as the cluster count to
                       automatically determine the best number of clusters.
            clusters:  Required if *compounds* is a list.

        Returns:
            ``self`` (modified in place).
        """
        if isinstance(compounds, dict):
            items = list(compounds.items())
        else:
            if clusters is None:
                raise ValueError("clusters must be provided with list input")
            if len(compounds) != len(clusters):
                raise ValueError("compounds and clusters length mismatch")
            items = list(zip(compounds, clusters))

        for cmpd_name, n_cluster in items:
            # Normalise "auto" → 0
            if isinstance(n_cluster, str) and n_cluster.lower() == "auto":
                n_cluster = 0
            n_cluster = int(n_cluster)

            label = "auto" if n_cluster == 0 else str(n_cluster)
            print(f"\nClustering {cmpd_name} → {label} clusters")
            self.point_cluster(cmpd_name, n_cluster=n_cluster)

        print("\nBatch clustering complete")
        return self

    def point_cluster(
        self, cmpd_name: str, n_cluster: int = 3, max_k: int = 6,
    ) -> "Experiment":
        """KMeans clustering of picked peaks by RT.

        Peaks are extracted, clustered, renamed with a suffix
        (e.g. ``brGDGT_IIIa_0``), and written back. At most one peak
        per sample per cluster is retained.

        Parameters:
            cmpd_name: Compound name to cluster.
            n_cluster: Number of clusters. Use ``0`` for automatic
                       selection via silhouette score.
            max_k:     Maximum k to try when ``n_cluster=0``.
        """
        extracted: list[tuple[str, PickedPeak]] = []

        for sample_name, chrom in self.chromatograms.items():
            eics = chrom.get_eic(cmpd_name)
            if not eics:
                continue
            eic = eics[0]

            remaining = []
            for peak in eic.picked:
                if peak.name == cmpd_name:
                    extracted.append((sample_name, peak))
                else:
                    remaining.append(peak)
            eic.picked = remaining

        if not extracted:
            print("  No peaks found for clustering")
            return self

        rt_vals = np.array([peak.rt for _, peak in extracted]).reshape(-1, 1)

        # Auto-detect optimal cluster count
        if n_cluster <= 0:
            n_cluster = self.find_optimal_clusters(rt_vals, max_k=max_k)
            print(f"  Auto-detected optimal k = {n_cluster}")

        kmeans = KMeans(n_clusters=n_cluster, random_state=0, n_init="auto")
        kmeans.fit(rt_vals)

        centers = kmeans.cluster_centers_.flatten()
        order = np.argsort(centers)
        sorted_centers = centers[order]

        occupancy: dict[str, set[int]] = {s: set() for s in self.chromatograms}
        cluster_groups: dict[int, list[tuple[str, PickedPeak]]] = {i: [] for i in range(n_cluster)}

        for sample_name, peak in extracted:
            dists = np.abs(sorted_centers - peak.rt)
            cluster_id = int(np.argmin(dists))

            if cluster_id in occupancy[sample_name]:
                continue

            occupancy[sample_name].add(cluster_id)
            new_peak = copy.deepcopy(peak)
            new_peak.name = f"{cmpd_name}_{cluster_id}"
            cluster_groups[cluster_id].append((sample_name, new_peak))

        for _cluster_id, items_list in cluster_groups.items():
            for sample_name, peak in items_list:
                chrom = self.chromatograms[sample_name]
                eic = chrom.get_eic(cmpd_name)[0]
                eic.picked.append(peak)

        print(f"  Clustering complete → {cmpd_name} ({n_cluster} clusters)")
        return self

    # ---- Diagnostics ----

    def summary(self) -> dict:
        """Return a dict summarising the experiment."""
        return {
            "n_samples": len(self.chromatograms),
            "rt_corrected": self.rt_corrected,
        }

    def __repr__(self) -> str:
        s = self.summary()
        return f"Experiment(samples={s['n_samples']}, rt_corrected={s['rt_corrected']})"
