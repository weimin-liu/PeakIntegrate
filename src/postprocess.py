from dataclasses import dataclass
import os
import re
import pandas as pd
import copy
import numpy as np
from sklearn.cluster import KMeans
import h5py

@dataclass
class PickedPeak:
    name: str
    rt: float
    rtmin: float
    rtmax: float
    into: float
    intb: float
    sigma: float

@dataclass
class EIC:
    name: str
    mz: float
    rt: np.ndarray
    intensity: np.ndarray
    picked: list[PickedPeak]
    shifted_rt: np.ndarray | None = None

class Chromatogram:
    def __init__(self, eics=None):
        self.eics = eics if eics is not None else []

    # -----------------------------
    # EIC-level operations
    # -----------------------------

    def get_eic(self, name):
        """Return all EICs matching compound name."""
        return [eic for eic in self.eics if name.startswith(eic.name)]

    def add_eic(self, eic: EIC):
        self.eics.append(eic)

    # -----------------------------
    # Peak-level operations
    # -----------------------------

    def get_peaks(self, name):
        """Return all picked peaks with given compound name."""
        peaks = []
        for eic in self.eics:
            peaks.extend([p for p in eic.picked if p.name == name])
        return peaks

    def get_rt(self):
        """Return RT of all picked peaks."""
        return [p.shifted_rt for eic in self.eics for p in eic.picked]

    def get_into(self):
        """Return integrated intensities."""
        return [p.into for eic in self.eics for p in eic.picked]

    def get_min_rt_by_cmpd(self, cmpd):
        vals = [p.rtmin for eic in self.eics for p in eic.picked if p.name == cmpd]
        return min(vals) if vals else np.nan

    def get_max_rt_by_cmpd(self, cmpd):
        vals = [p.rtmax for eic in self.eics for p in eic.picked if p.name == cmpd]
        return max(vals) if vals else np.nan

    def pop(self, cmpd_name):
        """Remove peaks with given name, keep EICs."""
        removed = []

        for eic in self.eics:
            keep = []
            for p in eic.picked:
                if p.name == cmpd_name:
                    removed.append(p)
                else:
                    keep.append(p)
            eic.picked = keep

        return removed

    # -----------------------------
    # RT correction (CRITICAL)
    # -----------------------------

    def shift_rt(self, poly):
        for eic in self.eics:

            # ---- Shift chromatographic axis ----
            if eic.shifted_rt is None:
                eic.shifted_rt = eic.rt.copy()

            shift_vals = poly(eic.shifted_rt)
            eic.shifted_rt = eic.shifted_rt + shift_vals

            # ---- Shift picked peaks ----
            for peak in eic.picked:
                shift = poly(peak.rt)
                peak.rt += shift
                peak.rtmin += shift
                peak.rtmax += shift

    # -----------------------------
    # Diagnostics / utilities
    # -----------------------------

    def summary(self):
        return {
            "n_eics": len(self.eics),
            "n_peaks": sum(len(eic.picked) for eic in self.eics)
        }

    def __repr__(self):
        s = self.summary()
        return f"Chromatogram(EICs={s['n_eics']}, Peaks={s['n_peaks']})"

class Experiment:

    def __init__(self, chromatograms):
        self.chromatograms = chromatograms  # dict[sample_name → Chromatogram]

        self.rt_corrected = False
        self.rt_model = None   # store polynomial / correction model

    # -----------------------------
    # Dict-like helpers (safe)
    # -----------------------------

    def keys(self):
        return self.chromatograms.keys()

    def items(self):
        return self.chromatograms.items()

    def values(self):
        return self.chromatograms.values()

    def get_sample_name(self):
        return list(self.chromatograms.keys())

    def __getitem__(self, key):
        return self.chromatograms[key]

    def get_rt(self, compound):

        rt_vals = []
        rtmin_vals = []
        rtmax_vals = []

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

        if len(rt_vals) == 0:
            return {
                "rtmin": np.nan,
                "rtmax": np.nan,
                "rtmed": np.nan
            }

        return {
            "rtmin": float(np.nanmin(rtmin_vals)),
            "rtmax": float(np.nanmax(rtmax_vals)),
            "rtmed": float(np.nanmedian(rt_vals))
        }




    def plot_picked_peaks(self):

        import plotly.graph_objects as go

        groups = {}

        for sample_name, chrom in self.chromatograms.items():

            for eic in chrom.eics:
                for peak in eic.picked:

                    groups.setdefault(peak.name, {
                        "rt": [],
                        "into": [],
                        "sample": []
                    })

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
                hovertemplate="rt=%{x:.2f}<br>into=%{y:.2e}<br>sample=%{customdata}<extra></extra>"
            ))

        fig.update_layout(
            xaxis_title="rt",
            yaxis_title="into",
            template="simple_white"
        )

        fig.show()

    # -----------------------------
    # RT Correction
    # -----------------------------

    def rt_shift(self,
                 calibs=None,
                 more_calibs=None,
                 degree=2,
                 ref_sample_name=None,
                 manual_anchors=None):

        calibs = ['C46-GDGT', 'brGDGT_Ib', 'brGDGT_Ia'] if calibs is None else list(calibs)

        if more_calibs:
            calibs.extend(more_calibs)

        shifted_exp = copy.deepcopy(self)

        ref_sample = (
            list(shifted_exp.chromatograms.keys())[0]
            if ref_sample_name is None
            else ref_sample_name
        )

        # ---- Reference RTs ----
        ref_rts = []

        for calib in calibs:
            peaks = shifted_exp.chromatograms[ref_sample].get_peaks(calib)

            if not peaks:
                raise ValueError(f"Missing calibration peak {calib} in reference sample")

            ref_rts.append(peaks[0].rt)

        # ---- Align samples ----
        for sample_name, chrom in shifted_exp.chromatograms.items():

            obs_rts = []
            rt_diffs = []

            for idx, calib in enumerate(calibs):

                peaks = chrom.get_peaks(calib)
                if not peaks:
                    continue

                obs_rt = peaks[0].rt

                obs_rts.append(obs_rt)
                rt_diffs.append(ref_rts[idx] - obs_rt)

            # ✅ Inject manual anchors HERE
            if manual_anchors and sample_name in manual_anchors:

                for observed_rt, target_rt in manual_anchors[sample_name]:
                    obs_rts.append(observed_rt)
                    rt_diffs.append(target_rt - observed_rt)

                    print(f"Manual anchor added → {sample_name}: "
                          f"{observed_rt} → {target_rt}")

            if len(obs_rts) < degree + 1:
                continue

            coef = np.polyfit(obs_rts, rt_diffs, degree)
            poly = np.poly1d(coef)

            chrom.shift_rt(poly)

        shifted_exp.rt_corrected = True
        shifted_exp.rt_model = poly

        return shifted_exp

    def point_cluster_batch(self, compounds, clusters=None):

        # ---- Normalize input ----
        if isinstance(compounds, dict):

            items = compounds.items()

        else:

            if clusters is None:
                raise ValueError("clusters must be provided with list input")

            if len(compounds) != len(clusters):
                raise ValueError("compounds and clusters length mismatch")

            items = zip(compounds, clusters)

        # ---- Sequential clustering ----
        for cmpd_name, n_cluster in items:

            print(f"\nClustering {cmpd_name} → {n_cluster} clusters")

            self.point_cluster(cmpd_name, n_cluster=n_cluster)

        print("\nBatch clustering complete")

        return self


    def point_cluster(self, cmpd_name, n_cluster=3):

        extracted = []

        # ---- Extract peaks from EICs ----
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

            eic.picked = remaining   # remove target peaks

        if len(extracted) == 0:
            print("No peaks found for clustering")
            return self

        # ---- KMeans on RT ----
        rt_vals = np.array([peak.rt for _, peak in extracted]).reshape(-1, 1)

        kmeans = KMeans(n_clusters=n_cluster, random_state=0)
        kmeans.fit(rt_vals)

        centers = kmeans.cluster_centers_.flatten()

        order = np.argsort(centers)
        sorted_centers = centers[order]

        # ---- Enforce one peak per sample per cluster ----
        occupancy = {sample: set() for sample in self.chromatograms}
        clusters = {i: [] for i in range(n_cluster)}

        for sample_name, peak in extracted:

            dists = abs(sorted_centers - peak.rt)
            cluster_id = np.argmin(dists)

            if cluster_id in occupancy[sample_name]:
                continue   # discard extra peak

            occupancy[sample_name].add(cluster_id)

            new_peak = copy.deepcopy(peak)
            new_peak.name = f"{cmpd_name}_{cluster_id}"

            clusters[cluster_id].append((sample_name, new_peak))

        # ---- Write back to EIC ----
        for cluster_id, items in clusters.items():

            for sample_name, peak in items:

                chrom = self.chromatograms[sample_name]
                eic = chrom.get_eic(cmpd_name)[0]

                eic.picked.append(peak)

        print(f"Clustering complete → {cmpd_name}")

        return self

    # -----------------------------
    # Diagnostics
    # -----------------------------

    def summary(self):
        return {
            "n_samples": len(self.chromatograms),
            "rt_corrected": self.rt_corrected
        }

    def plot_eic(self, compound_name, corrected=True):

        import plotly.graph_objects as go

        fig = go.Figure()

        for sample_name, chrom in self.chromatograms.items():

            eics = chrom.get_eic(compound_name)

            if not eics:
                continue

            for eic in eics:
                rt_axis = (
                    eic.shifted_rt
                    if corrected and getattr(eic, "shifted_rt", None) is not None
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
                    )
                ))

        fig.update_layout(
            title=f"EIC: {compound_name}",
            xaxis_title="rt",
            yaxis_title="intensity",
            template="simple_white"
        )

        fig.show()


def main():
    datafolder = '/Users/weimin/10-Project/GDGT_peak_integration/tables'
    reg = r'AEGIS-(\d+)'
    # read all the csv files to dictionary
    data = {}
    SampleNames = []
    for filename in os.listdir(datafolder):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(datafolder, filename))
            df['SampleName'] = df.iloc[:,0].map(lambda x: f'AEGIS-{re.findall(reg, x)[0]}')
            SampleNames.extend(df['SampleName'].unique())
            data[filename.split('.')[0]] = df
    SampleNames = list(set(SampleNames))

    samples = {}

    for CompoundName, CompoundData in data.items():
        for SampleName in SampleNames:
            if SampleName not in samples.keys():
                samples[SampleName] = []
            SubSample = CompoundData[CompoundData["SampleName"] == SampleName]
            picked_peaks = []
            if not SubSample.empty:
                for index, row in SubSample.iterrows():
                    peak = PickedPeak(
                        name=CompoundName,
                        rt=row['rt'],
                        rtmin=row['rtmin'],
                        rtmax=row['rtmax'],
                        into=row['into'],
                        intb=row['intb'],
                        sigma=row['sigma']
                    )

                    picked_peaks.append(peak)

            with h5py.File("/Users/weimin/chrom_data.h5", "r") as f:
                real_sample_name = [k for k in f.keys() if SampleName in k][0]
                rt = f[real_sample_name][CompoundName]["rt"][:]
                intensity = f[real_sample_name][CompoundName]["intensity"][:]
                eic = EIC(
                    name=CompoundName,
                    mz=None,
                    rt=rt,
                    intensity=intensity,
                    picked=picked_peaks
                )
                samples[SampleName].append(eic)

    for sample, peaks in samples.items():
        samples[sample] = Chromatogram(peaks)
    return Experiment(samples)

if __name__ == '__main__':
    exp = main()
    exp = exp.rt_shift()
    exp = exp.rt_shift(
        manual_anchors={
        "AEGIS-158": [(2512, 2534)]
    })

    exp = exp.point_cluster_batch(
        {'brGDGT_IIIa':3,
         'brGDGT_IIa':2,
         }
    )
    exp.plot_picked_peaks()

    exp.get_rt('brGDGT_IIIa_0')

    import pickle

    with open("../../experiment.pkl", "wb") as f:
        pickle.dump(exp, f, protocol=pickle.HIGHEST_PROTOCOL)

