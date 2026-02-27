"""
PeakIntegrate — Streamlit GUI

Launch:
    streamlit run PeakIntegrate/app.py
"""

import sys
import os
import pickle
import copy

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── Ensure the project root is importable ──
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from PeakIntegrate.src.models import (
    PickedPeak,
    EIC,
    Chromatogram,
    Experiment,
)
from PeakIntegrate.src.loader import load_experiment
from PeakIntegrate.src.integration import (
    integrate_experiment,
    gauss,
    double_gauss,
    TARGET_COMPOUNDS,
    VALID_MODELS,
)

# ════════════════════════════════════════════
#  Page Configuration
# ════════════════════════════════════════════

st.set_page_config(
    page_title="PeakIntegrate",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    .stApp {
        background: #ffffff;
    }
    [data-testid="stSidebar"] {
        background: #f8f9fb;
        border-right: 1px solid #e2e4e9;
    }
    .step-header {
        background: linear-gradient(135deg, #eef2ff, #f0ebff);
        border: 1px solid #c7d2fe;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.2rem;
    }
    .step-header h2 {
        margin: 0;
        color: #4338ca;
    }
    .step-header p {
        margin: 0.3rem 0 0 0;
        color: #64748b;
        font-size: 0.9rem;
    }
    .metric-card {
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-card h3 {
        margin: 0;
        color: #4338ca;
        font-size: 2rem;
    }
    .metric-card p {
        margin: 0.2rem 0 0 0;
        color: #64748b;
        font-size: 0.85rem;
    }
    div[data-testid="stTabs"] button {
        font-weight: 600;
    }
    .success-banner {
        background: linear-gradient(135deg, #ecfdf5, #f0fdf4);
        border: 1px solid #86efac;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        color: #166534;
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════
#  Session State Helpers
# ════════════════════════════════════════════

def _init_state():
    """Initialise session state defaults."""
    defaults = {
        "exp": None,
        "exp_corrected": None,
        "exp_clustered": None,
        "results_df": None,
        "manual_anchors": {},
        "cluster_config": {"brGDGT_IIIa": 3, "brGDGT_IIa": 2},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


def _active_exp():
    """Return the most processed version of the experiment."""
    for key in ("exp_clustered", "exp_corrected", "exp"):
        if st.session_state[key] is not None:
            return st.session_state[key]
    return None


# ════════════════════════════════════════════
#  Sidebar — Data Loading
# ════════════════════════════════════════════

with st.sidebar:
    st.markdown("# 🔬 PeakIntegrate")
    st.caption("GDGT Chromatographic Peak Integration")
    st.divider()

    st.markdown("### 📂 Data Source")

    datafolder = st.text_input(
        "CSV tables folder",
        value="/Users/weimin/10-Project/GDGT_peak_integration/tables",
        help="Directory with one CSV per compound",
    )
    hdf5_path = st.text_input(
        "HDF5 chromatogram file",
        value="/Users/weimin/chrom_data.h5",
        help="Path to the raw EIC data file",
    )

    if st.button("🚀 Load Data", use_container_width=True, type="primary"):
        with st.spinner("Loading experiment..."):
            try:
                exp = load_experiment(
                    datafolder=datafolder,
                    hdf5_path=hdf5_path,
                )
                st.session_state["exp"] = exp
                st.session_state["exp_corrected"] = None
                st.session_state["exp_clustered"] = None
                st.session_state["results_df"] = None
                st.success(f"Loaded {len(exp.chromatograms)} samples")
            except Exception as e:
                st.error(f"Failed to load: {e}")

    st.divider()

    # ── Load from pickle ──
    st.markdown("### 📦 Or Load Pickle")
    pkl_path = st.text_input(
        "Experiment pickle path",
        value="/Users/weimin/10-Project/GDGT_peak_integration/experiment.pkl",
    )
    if st.button("📥 Load Pickle", use_container_width=True):
        try:
            with open(pkl_path, "rb") as f:
                exp = pickle.load(f)
            st.session_state["exp"] = exp
            st.session_state["exp_corrected"] = None
            st.session_state["exp_clustered"] = None
            st.session_state["results_df"] = None
            st.success(f"Loaded {len(exp.chromatograms)} samples from pickle")
        except Exception as e:
            st.error(f"Failed: {e}")

    st.divider()

    # ── Status ──
    exp = _active_exp()
    if exp is not None:
        st.markdown("### 📊 Status")
        st.markdown(f"**Samples:** {len(exp.chromatograms)}")
        st.markdown(f"**RT corrected:** {'✅' if exp.rt_corrected else '❌'}")
        n_peaks = sum(
            len(eic.picked)
            for chrom in exp.chromatograms.values()
            for eic in chrom.eics
        )
        st.markdown(f"**Total peaks:** {n_peaks:,}")
    else:
        st.info("No data loaded yet")


# ════════════════════════════════════════════
#  Main Area — Tabs
# ════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "🔧 RT Correction",
    "📊 Visualization",
    "🔬 Clustering",
    "📈 Integration",
])


# ════════════════════════════════════════════
#  Tab 1 — RT Correction
# ════════════════════════════════════════════

with tab1:
    st.markdown("""
    <div class="step-header">
        <h2>🔧 Step 1 — Retention Time Correction</h2>
        <p>Align retention times across samples using polynomial fitting on calibration peaks.</p>
    </div>
    """, unsafe_allow_html=True)

    exp = st.session_state["exp"]
    if exp is None:
        st.warning("⬅️ Load data first using the sidebar.")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Calibrants")

            default_calibs = ["C46-GDGT", "brGDGT_Ib", "brGDGT_Ia"]

            # Gather all compound names from EICs
            all_compounds = sorted({
                eic.name
                for chrom in exp.chromatograms.values()
                for eic in chrom.eics
            })

            calibs = st.multiselect(
                "Calibration compounds",
                options=all_compounds,
                default=[c for c in default_calibs if c in all_compounds],
            )

            degree = st.slider("Polynomial degree", 1, 4, 2)

            sample_names = list(exp.chromatograms.keys())
            ref_sample = st.selectbox(
                "Reference sample",
                options=sample_names,
                index=0,
            )

        with col2:
            st.markdown("#### Manual Anchors")
            st.caption("For samples that need extra correction points.")

            anchor_sample = st.selectbox(
                "Sample", options=sample_names, key="anchor_sample"
            )
            acol1, acol2 = st.columns(2)
            with acol1:
                obs_rt = st.number_input("Observed RT", value=0.0, step=1.0)
            with acol2:
                target_rt = st.number_input("Target RT", value=0.0, step=1.0)

            if st.button("➕ Add Anchor"):
                anchors = st.session_state["manual_anchors"]
                anchors.setdefault(anchor_sample, [])
                anchors[anchor_sample].append((obs_rt, target_rt))
                st.success(f"Added: {anchor_sample} {obs_rt} → {target_rt}")

            # Show current anchors
            if st.session_state["manual_anchors"]:
                st.markdown("**Current anchors:**")
                for s, pairs in st.session_state["manual_anchors"].items():
                    for obs, tgt in pairs:
                        st.markdown(f"- `{s}`: {obs:.1f} → {tgt:.1f}")
                if st.button("🗑️ Clear All Anchors"):
                    st.session_state["manual_anchors"] = {}
                    st.rerun()

        st.divider()

        if st.button("▶️  Run RT Correction", use_container_width=True, type="primary"):
            with st.spinner("Correcting retention times..."):
                try:
                    exp_c = exp.rt_shift(
                        calibs=calibs if calibs else None,
                        degree=degree,
                        ref_sample_name=ref_sample,
                    )
                    # Apply manual anchors if any
                    if st.session_state["manual_anchors"]:
                        exp_c = exp_c.rt_shift(
                            manual_anchors=st.session_state["manual_anchors"],
                        )
                    st.session_state["exp_corrected"] = exp_c
                    st.session_state["exp_clustered"] = None
                    st.session_state["results_df"] = None
                    st.markdown("""
                    <div class="success-banner">
                        ✅ <strong>RT correction complete!</strong> Switch to the Visualization tab to inspect results.
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"RT correction failed: {e}")

        # ── Before/After comparison ──
        exp_c = st.session_state.get("exp_corrected")
        if exp_c is not None and exp is not None:
            st.markdown("#### Before / After Comparison")
            cmpd_compare = st.selectbox(
                "Compound to compare",
                options=all_compounds,
                key="rt_compare_cmpd",
            )

            fig_before = go.Figure()
            fig_after = go.Figure()

            for sname, chrom in exp.chromatograms.items():
                eics = chrom.get_eic(cmpd_compare)
                if not eics:
                    continue
                eic = eics[0]
                rt_axis = eic.shifted_rt if eic.shifted_rt is not None else eic.rt
                fig_before.add_trace(go.Scatter(
                    x=rt_axis, y=eic.intensity, mode="lines",
                    name=sname, showlegend=False,
                    line=dict(width=1),
                ))

            for sname, chrom in exp_c.chromatograms.items():
                eics = chrom.get_eic(cmpd_compare)
                if not eics:
                    continue
                eic = eics[0]
                rt_axis = eic.shifted_rt if eic.shifted_rt is not None else eic.rt
                fig_after.add_trace(go.Scatter(
                    x=rt_axis, y=eic.intensity, mode="lines",
                    name=sname, showlegend=False,
                    line=dict(width=1),
                ))

            fig_before.update_layout(
                title="Before correction",
                xaxis_title="RT (s)", yaxis_title="Intensity",
                template="simple_white", height=350,
                margin=dict(l=50, r=20, t=40, b=40),
            )
            fig_after.update_layout(
                title="After correction",
                xaxis_title="RT (s)", yaxis_title="Intensity",
                template="simple_white", height=350,
                margin=dict(l=50, r=20, t=40, b=40),
            )

            bcol, acol = st.columns(2)
            with bcol:
                st.plotly_chart(fig_before, use_container_width=True)
            with acol:
                st.plotly_chart(fig_after, use_container_width=True)


# ════════════════════════════════════════════
#  Tab 2 — Visualization
# ════════════════════════════════════════════

with tab2:
    st.markdown("""
    <div class="step-header">
        <h2>📊 Step 2 — Visualization</h2>
        <p>Explore EIC traces and picked peaks interactively.</p>
    </div>
    """, unsafe_allow_html=True)

    exp_vis = _active_exp()
    if exp_vis is None:
        st.warning("⬅️ Load data first using the sidebar.")
    else:
        all_cmpds_vis = sorted({
            eic.name
            for chrom in exp_vis.chromatograms.values()
            for eic in chrom.eics
        })

        # Also include clustered compound names from peaks
        all_peak_names = sorted({
            p.name
            for chrom in exp_vis.chromatograms.values()
            for eic in chrom.eics
            for p in eic.picked
        })
        all_selectable = sorted(set(all_cmpds_vis) | set(all_peak_names))

        vcol1, vcol2 = st.columns([3, 1])
        with vcol2:
            use_corrected = st.toggle("Use corrected RT", value=True)
        with vcol1:
            selected_cmpd = st.selectbox(
                "Select compound",
                options=all_selectable,
                key="vis_compound",
            )

        # ── EIC overlay ──
        st.markdown("#### EIC Overlay")
        fig_eic = go.Figure()

        for sname, chrom in exp_vis.chromatograms.items():
            eics = chrom.get_eic(selected_cmpd)
            if not eics:
                continue
            for eic in eics:
                rt_axis = (
                    eic.shifted_rt
                    if use_corrected and eic.shifted_rt is not None
                    else eic.rt
                )
                fig_eic.add_trace(go.Scatter(
                    x=rt_axis, y=eic.intensity, mode="lines",
                    name=sname,
                    hovertemplate=(
                        f"sample={sname}<br>"
                        "rt=%{x:.2f}<br>"
                        "intensity=%{y:.2e}<extra></extra>"
                    ),
                    line=dict(width=1.2),
                ))

        fig_eic.update_layout(
            title=f"EIC: {selected_cmpd}",
            xaxis_title="RT (s)", yaxis_title="Intensity",
            template="simple_white", height=450,
            legend=dict(font=dict(size=10)),
            margin=dict(l=50, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_eic, use_container_width=True)

        # ── Picked-peak scatter ──
        st.markdown("#### Picked Peaks (All Compounds)")

        groups: dict[str, dict] = {}
        for sname, chrom in exp_vis.chromatograms.items():
            for eic in chrom.eics:
                for peak in eic.picked:
                    groups.setdefault(peak.name, {"rt": [], "into": [], "sample": []})
                    groups[peak.name]["rt"].append(peak.rt)
                    groups[peak.name]["into"].append(peak.into)
                    groups[peak.name]["sample"].append(sname)

        fig_peaks = go.Figure()
        for name, g in groups.items():
            fig_peaks.add_trace(go.Scatter(
                x=g["rt"], y=g["into"], mode="markers",
                name=name, customdata=g["sample"],
                hovertemplate="rt=%{x:.2f}<br>into=%{y:.2e}<br>sample=%{customdata}<extra></extra>",
                marker=dict(size=6, opacity=0.7),
            ))

        fig_peaks.update_layout(
            xaxis_title="RT (s)", yaxis_title="Integrated Area",
            template="simple_white", height=400,
            margin=dict(l=50, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_peaks, use_container_width=True)


# ════════════════════════════════════════════
#  Tab 3 — Peak Clustering
# ════════════════════════════════════════════

with tab3:
    st.markdown("""
    <div class="step-header">
        <h2>🔬 Step 3 — Peak Clustering</h2>
        <p>Group co-eluting isomers via KMeans clustering on retention time.</p>
    </div>
    """, unsafe_allow_html=True)

    exp_pre_cluster = st.session_state.get("exp_corrected") or st.session_state.get("exp")
    if exp_pre_cluster is None:
        st.warning("⬅️ Load data first. RT correction is recommended before clustering.")
    else:
        st.markdown("#### Clustering Configuration")
        st.caption("Add compounds to cluster and specify the number of isomer groups.")

        # Dynamic cluster config editor
        config = st.session_state["cluster_config"]

        # Show existing config
        to_remove = []
        for cmpd_name, n_clust in list(config.items()):
            ccol1, ccol2, ccol3, ccol4 = st.columns([3, 1, 1, 1])
            with ccol1:
                st.text(cmpd_name)
            with ccol2:
                is_auto = st.toggle(
                    "Auto", value=(n_clust == 0),
                    key=f"auto_{cmpd_name}",
                )
            with ccol3:
                if is_auto:
                    config[cmpd_name] = 0
                    st.text("auto")
                else:
                    new_n = st.number_input(
                        "Clusters",
                        min_value=1, max_value=10,
                        value=n_clust if n_clust >= 1 else 3,
                        key=f"clust_{cmpd_name}", label_visibility="collapsed",
                    )
                    config[cmpd_name] = new_n
            with ccol4:
                if st.button("🗑️", key=f"rm_{cmpd_name}"):
                    to_remove.append(cmpd_name)

        for r in to_remove:
            del config[r]
            st.rerun()

        # Add new compound
        with st.expander("➕ Add compound to cluster"):
            all_eic_names = sorted({
                eic.name
                for chrom in exp_pre_cluster.chromatograms.values()
                for eic in chrom.eics
            })
            new_cmpd = st.selectbox("Compound", options=all_eic_names, key="new_cluster_cmpd")
            new_auto = st.toggle("Auto-detect clusters", value=False, key="new_auto")
            if new_auto:
                new_n_clusters = 0
            else:
                new_n_clusters = st.number_input("Number of clusters", min_value=1, max_value=10, value=3, key="new_n_clust")
            if st.button("Add"):
                config[new_cmpd] = new_n_clusters
                st.rerun()

        st.divider()

        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            if st.button("▶️  Run Clustering", use_container_width=True, type="primary"):
                if not config:
                    st.warning("Add at least one compound to cluster.")
                else:
                    with st.spinner("Clustering peaks..."):
                        try:
                            exp_clust = copy.deepcopy(exp_pre_cluster)
                            exp_clust = exp_clust.point_cluster_batch(dict(config))
                            st.session_state["exp_clustered"] = exp_clust
                            st.session_state["results_df"] = None

                            st.markdown("""
                            <div class="success-banner">
                                ✅ <strong>Clustering complete!</strong> Check the Visualization tab to see the results.
                            </div>
                            """, unsafe_allow_html=True)

                        except Exception as e:
                            st.error(f"Clustering failed: {e}")

        with btn_col2:
            if st.button("🤖 Auto-Cluster All Compounds", use_container_width=True):
                with st.spinner("Auto-detecting clusters for all compounds..."):
                    try:
                        # Collect all unique EIC compound names
                        all_eic_cmpds = sorted({
                            eic.name
                            for chrom in exp_pre_cluster.chromatograms.values()
                            for eic in chrom.eics
                            if any(p.name == eic.name for p in eic.picked)  # only if it has picked peaks
                        })
                        auto_config = {cmpd: 0 for cmpd in all_eic_cmpds}

                        exp_clust = copy.deepcopy(exp_pre_cluster)
                        exp_clust = exp_clust.point_cluster_batch(auto_config)
                        st.session_state["exp_clustered"] = exp_clust
                        st.session_state["results_df"] = None

                        st.markdown("""
                        <div class="success-banner">
                            ✅ <strong>Auto-clustering complete!</strong> Check the Visualization tab to see the results.
                        </div>
                        """, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Auto-clustering failed: {e}")

        # Show cluster summary and before/after plots if clustering has been done
        exp_clust_display = st.session_state.get("exp_clustered")
        if exp_clust_display is not None:
            clustered_names = sorted({
                p.name
                for chrom in exp_clust_display.chromatograms.values()
                for eic in chrom.eics
                for p in eic.picked
            })
            st.markdown("**Resulting compound groups:**")
            for n in clustered_names:
                count = sum(
                    1
                    for chrom in exp_clust_display.chromatograms.values()
                    for eic in chrom.eics
                    for p in eic.picked
                    if p.name == n
                )
                st.markdown(f"- `{n}`: {count} peaks")

            # ── Before / After peak distribution ──
            st.divider()
            st.markdown("#### Peak Distribution — Before vs After Clustering")

            # Build "before" data from the pre-cluster experiment
            before_data: dict[str, dict] = {}
            for sname, chrom in exp_pre_cluster.chromatograms.items():
                for eic in chrom.eics:
                    for p in eic.picked:
                        before_data.setdefault(p.name, {"rt": [], "into": [], "sample": []})
                        before_data[p.name]["rt"].append(p.rt)
                        before_data[p.name]["into"].append(p.into)
                        before_data[p.name]["sample"].append(sname)

            # Build "after" data from the clustered experiment
            after_data: dict[str, dict] = {}
            for sname, chrom in exp_clust_display.chromatograms.items():
                for eic in chrom.eics:
                    for p in eic.picked:
                        after_data.setdefault(p.name, {"rt": [], "into": [], "sample": []})
                        after_data[p.name]["rt"].append(p.rt)
                        after_data[p.name]["into"].append(p.into)
                        after_data[p.name]["sample"].append(sname)

            fig_before = go.Figure()
            for name, g in sorted(before_data.items()):
                fig_before.add_trace(go.Scatter(
                    x=g["rt"], y=g["into"], mode="markers",
                    name=name, customdata=g["sample"],
                    hovertemplate="rt=%{x:.2f}<br>into=%{y:.2e}<br>sample=%{customdata}<extra></extra>",
                    marker=dict(size=5, opacity=0.7),
                ))
            fig_before.update_layout(
                title="Before Clustering",
                xaxis_title="RT (s)", yaxis_title="Integrated Area",
                template="simple_white", height=400,
                margin=dict(l=50, r=20, t=40, b=40),
                legend=dict(font=dict(size=9)),
            )

            fig_after = go.Figure()
            for name, g in sorted(after_data.items()):
                fig_after.add_trace(go.Scatter(
                    x=g["rt"], y=g["into"], mode="markers",
                    name=name, customdata=g["sample"],
                    hovertemplate="rt=%{x:.2f}<br>into=%{y:.2e}<br>sample=%{customdata}<extra></extra>",
                    marker=dict(size=5, opacity=0.7),
                ))
            fig_after.update_layout(
                title="After Clustering",
                xaxis_title="RT (s)", yaxis_title="Integrated Area",
                template="simple_white", height=400,
                margin=dict(l=50, r=20, t=40, b=40),
                legend=dict(font=dict(size=9)),
            )

            bcol, acol = st.columns(2)
            with bcol:
                st.plotly_chart(fig_before, use_container_width=True)
            with acol:
                st.plotly_chart(fig_after, use_container_width=True)


# ════════════════════════════════════════════
#  Tab 4 — Integration & Export
# ════════════════════════════════════════════

with tab4:
    st.markdown("""
    <div class="step-header">
        <h2>📈 Step 4 — Integration & Export</h2>
        <p>Fit peak models (Gaussian, EMG, or Bi-Gaussian), compute areas, and export results.</p>
    </div>
    """, unsafe_allow_html=True)

    exp_int = _active_exp()
    if exp_int is None:
        st.warning("⬅️ Load data first using the sidebar.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Target Compounds")
            available_peak_names = sorted({
                p.name
                for chrom in exp_int.chromatograms.values()
                for eic in chrom.eics
                for p in eic.picked
            })
            target_cmpds = st.multiselect(
                "Compounds to integrate",
                options=available_peak_names,
                default=[c for c in TARGET_COMPOUNDS if c in available_peak_names],
            )

        with col2:
            st.markdown("#### Parameters")
            model_labels = {"gaussian": "Gaussian", "emg": "EMG (Exp. Modified Gaussian)", "bigauss": "Bi-Gaussian"}
            selected_model = st.selectbox(
                "Peak model",
                options=list(VALID_MODELS),
                format_func=lambda m: model_labels.get(m, m),
                index=0,
            )
            min_points = st.number_input("Min data points", value=11, min_value=5, max_value=50)
            savgol_window = st.number_input("Savgol window", value=11, min_value=5, max_value=51, step=2)
            savgol_poly = st.number_input("Savgol poly order", value=3, min_value=1, max_value=7)
            prominence_frac = st.slider("Prominence threshold", 0.01, 0.20, 0.05, 0.01)
            subtract_bl = st.toggle("📉 Subtract baseline", value=True)
            generate_pdf = st.toggle("📄 Generate peak overlay PDF", value=False)

        st.divider()

        if st.button("▶️  Run Integration", use_container_width=True, type="primary"):
            if not target_cmpds:
                st.warning("Select at least one compound.")
            else:
                with st.spinner(f"Fitting {model_labels[selected_model]} models and integrating peaks..."):
                    try:
                        # Save experiment to a temp pickle so integrate_experiment can load it
                        import tempfile
                        tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
                        pickle.dump(exp_int, tmp, protocol=pickle.HIGHEST_PROTOCOL)
                        tmp.close()

                        # Prepare PDF output path if requested
                        pdf_tmp = None
                        if generate_pdf:
                            pdf_tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
                            pdf_tmp.close()

                        df = integrate_experiment(
                            pkl_path=tmp.name,
                            output_csv=None,
                            output_pdf=pdf_tmp.name if pdf_tmp else None,
                            target_cmpds=target_cmpds,
                            fit_model=selected_model,
                            subtract_baseline=subtract_bl,
                            min_points=min_points,
                            savgol_window=savgol_window,
                            savgol_poly=savgol_poly,
                            prominence_frac=prominence_frac,
                        )
                        os.unlink(tmp.name)

                        # Read PDF bytes into session state for download
                        if pdf_tmp and os.path.exists(pdf_tmp.name):
                            with open(pdf_tmp.name, "rb") as pf:
                                st.session_state["results_pdf"] = pf.read()
                            os.unlink(pdf_tmp.name)
                        else:
                            st.session_state["results_pdf"] = None

                        st.session_state["results_df"] = df

                        st.markdown("""
                        <div class="success-banner">
                            ✅ <strong>Integration complete!</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Integration failed: {e}")

        # ── Results display ──
        results_df = st.session_state.get("results_df")
        if results_df is not None:
            st.markdown("#### Results Preview")

            # Metrics row
            mcols = st.columns(4)
            with mcols[0]:
                st.markdown(f"""
                <div class="metric-card"><h3>{len(results_df)}</h3><p>Samples</p></div>
                """, unsafe_allow_html=True)
            with mcols[1]:
                st.markdown(f"""
                <div class="metric-card"><h3>{len(results_df.columns)}</h3><p>Compounds</p></div>
                """, unsafe_allow_html=True)
            with mcols[2]:
                valid_pct = (results_df.notna().sum().sum() / results_df.size * 100)
                st.markdown(f"""
                <div class="metric-card"><h3>{valid_pct:.0f}%</h3><p>Valid fits</p></div>
                """, unsafe_allow_html=True)
            with mcols[3]:
                nan_count = results_df.isna().sum().sum()
                st.markdown(f"""
                <div class="metric-card"><h3>{nan_count}</h3><p>NaN values</p></div>
                """, unsafe_allow_html=True)

            st.dataframe(
                results_df.style.format("{:.2e}", na_rep="NaN"),
                use_container_width=True,
                height=400,
            )

            # ── Export buttons ──
            st.markdown("#### Export")
            ecol1, ecol2, ecol3, ecol4 = st.columns(4)

            with ecol1:
                csv_data = results_df.to_csv()
                st.download_button(
                    "📥 Download CSV",
                    data=csv_data,
                    file_name="peak_integration_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with ecol2:
                pdf_bytes = st.session_state.get("results_pdf")
                if pdf_bytes:
                    st.download_button(
                        "📄 Download PDF",
                        data=pdf_bytes,
                        file_name="gaussian_overlays.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                else:
                    st.button(
                        "📄 No PDF generated",
                        disabled=True,
                        use_container_width=True,
                        help="Enable 'Generate Gaussian overlay PDF' and re-run integration.",
                    )

            with ecol3:
                save_path = st.text_input(
                    "Save CSV to path",
                    value="results.csv",
                    key="csv_save_path",
                )
                if st.button("💾 Save CSV to Disk", use_container_width=True):
                    try:
                        results_df.to_csv(save_path)
                        st.success(f"Saved to {save_path}")
                    except Exception as e:
                        st.error(f"Failed: {e}")

            with ecol4:
                pkl_save_path = st.text_input(
                    "Save experiment to",
                    value="experiment.pkl",
                    key="pkl_save_path",
                )
                if st.button("💾 Save Experiment (.pkl)", use_container_width=True):
                    try:
                        with open(pkl_save_path, "wb") as f:
                            pickle.dump(exp_int, f, protocol=pickle.HIGHEST_PROTOCOL)
                        st.success(f"Experiment saved to {pkl_save_path}")
                    except Exception as e:
                        st.error(f"Failed: {e}")
