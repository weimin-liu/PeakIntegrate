"""
PeakIntegrate — Streamlit GUI

Launch:
    streamlit run PeakIntegrate/app.py
"""

import sys
import os
import pickle
import copy
import io
import inspect
import zipfile

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yaml

# ── Ensure the project root is importable ──
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_zoom_component = components.declare_component(
    "plotly_zoom_capture",
    path=os.path.join(os.path.dirname(__file__), "components", "plotly_zoom_capture"),
)

from PeakIntegrate.src.models import (
    PickedPeak,
    EIC,
    Chromatogram,
    Experiment,
)
from PeakIntegrate.src.loader import (
    load_experiment,
    load_experiment_from_eic_csv,
)
from PeakIntegrate.src.config import load_compounds
from PeakIntegrate.src.integration import (
    build_sample_overlay_figures,
    integrate_experiment,
    gauss,
    double_gauss,
    TARGET_COMPOUNDS,
    VALID_BASELINE_SCOPES,
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
        "results_figures": {},
        "result_column_aliases": {},
        "manual_anchors": {},
        "cluster_config": {"brGDGT_IIIa": 3, "brGDGT_IIa": 2},
        "source_mode": "legacy",
        "eic_yaml_path": None,
        "eic_default_half_window": 60.0,
        "latest_zoom_windows": {},
        "peak_input_versions": {},
        "peak_blueprint": {},
        "local_yaml_path": "",
        "last_zoom_commit": {},
        "pending_peak_pick_entry": {},
        "custom_index_presets": {},
        "active_index_presets": [],
        "session_path": "",
        "session_path_context": "",
        "active_index_presets_initialized": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


def _rt_shift_supports_method(exp) -> bool:
    """Return True when the experiment exposes the new RT-shift API."""
    if exp is None or not hasattr(exp, "rt_shift"):
        return False
    try:
        return "method" in inspect.signature(exp.rt_shift).parameters
    except (TypeError, ValueError):
        return False


def _upgrade_experiment_object(exp):
    """Rebuild stale experiment objects into the current in-app classes."""
    if exp is None:
        return None
    if isinstance(exp, Experiment) and _rt_shift_supports_method(exp):
        return exp

    chromatograms: dict[str, Chromatogram] = {}
    for sample_name, chrom in getattr(exp, "chromatograms", {}).items():
        eics: list[EIC] = []
        for eic in getattr(chrom, "eics", []):
            picked = [
                PickedPeak(
                    name=peak.name,
                    rt=float(peak.rt),
                    rtmin=float(peak.rtmin),
                    rtmax=float(peak.rtmax),
                    into=float(peak.into),
                    intb=float(peak.intb),
                    sigma=float(peak.sigma),
                )
                for peak in getattr(eic, "picked", [])
            ]
            shifted_rt = getattr(eic, "shifted_rt", None)
            eics.append(EIC(
                name=eic.name,
                mz=getattr(eic, "mz", None),
                rt=np.array(eic.rt, copy=True),
                intensity=np.array(eic.intensity, copy=True),
                picked=picked,
                shifted_rt=None if shifted_rt is None else np.array(shifted_rt, copy=True),
            ))
        chromatograms[sample_name] = Chromatogram(eics)

    upgraded = Experiment(chromatograms)
    upgraded.rt_corrected = bool(getattr(exp, "rt_corrected", False))
    upgraded.rt_model = getattr(exp, "rt_model", None)
    return upgraded


def _upgrade_session_experiments() -> None:
    """Normalize any stale session-state experiments to the current class layout."""
    for key in ("exp", "exp_corrected", "exp_clustered"):
        current = st.session_state.get(key)
        upgraded = _upgrade_experiment_object(current)
        if upgraded is not current:
            st.session_state[key] = upgraded


_upgrade_session_experiments()


def _active_exp():
    """Return the most processed version of the experiment."""
    for key in ("exp_clustered", "exp_corrected", "exp"):
        if st.session_state[key] is not None:
            return st.session_state[key]
    return None


def _display_result_name(column_name: str) -> str:
    """Return the preferred display name for a result column."""
    aliases = st.session_state.get("result_column_aliases", {})
    alias = aliases.get(column_name)
    if isinstance(alias, str) and alias.strip():
        return f"{alias.strip()} ({column_name})"
    return column_name


def _result_alias_frame(columns: list[str]) -> pd.DataFrame:
    """Build the editable result-name mapping table."""
    aliases = st.session_state.get("result_column_aliases", {})
    return pd.DataFrame({
        "original_name": columns,
        "display_name": [aliases.get(col, "") for col in columns],
    })


SESSION_BUNDLE_KEYS = [
    "exp",
    "exp_corrected",
    "exp_clustered",
    "results_df",
    "results_figures",
    "result_column_aliases",
    "manual_anchors",
    "cluster_config",
    "source_mode",
    "eic_yaml_path",
    "eic_default_half_window",
    "latest_zoom_windows",
    "peak_input_versions",
    "peak_blueprint",
    "local_yaml_path",
    "last_zoom_commit",
    "pending_peak_pick_entry",
    "custom_index_presets",
    "active_index_presets",
    "active_index_presets_initialized",
    "session_path",
    "session_path_context",
]


def _build_session_bundle() -> dict[str, object]:
    """Serialize the current app session into a portable bundle."""
    return {
        "bundle_version": 1,
        "app": "PeakIntegrate",
        "state": {
            key: copy.deepcopy(st.session_state.get(key))
            for key in SESSION_BUNDLE_KEYS
        },
    }


def _load_session_bundle(bundle: dict[str, object]) -> None:
    """Restore app session state from a previously saved bundle."""
    state = bundle.get("state", {})
    if not isinstance(state, dict):
        raise ValueError("Invalid session bundle: missing state dictionary")

    for key in SESSION_BUNDLE_KEYS:
        if key in state:
            st.session_state[key] = state[key]


def _default_session_path(
    *,
    source_mode: str,
    eic_yaml_path: str | None = None,
    eic_folder: str | None = None,
    datafolder: str | None = None,
    hdf5_path: str | None = None,
) -> str:
    """Suggest a session file path tied to the current input project."""
    if source_mode == "eic_csv":
        if eic_yaml_path:
            base, _ = os.path.splitext(eic_yaml_path)
            return f"{base}.session.pkl"
        if eic_folder:
            folder = os.path.abspath(eic_folder)
            return os.path.join(folder, "peakintegrate.session.pkl")
    else:
        if hdf5_path:
            base, _ = os.path.splitext(hdf5_path)
            return f"{base}.session.pkl"
        if datafolder:
            folder = os.path.abspath(datafolder)
            return os.path.join(folder, "peakintegrate.session.pkl")
    return os.path.abspath("peakintegrate.session.pkl")


INDEX_PRESETS = {
    "MBT'5ME": {
        "latex": r"\mathrm{MBT'_{5ME}} = \frac{Ia + Ib + Ic}{Ia + Ib + Ic + IIa + IIb + IIc + IIIa}",
        "variables": ["Ia", "Ib", "Ic", "IIa", "IIb", "IIc", "IIIa"],
        "numerator_terms": [("Ia", 1.0), ("Ib", 1.0), ("Ic", 1.0)],
        "denominator_terms": [
            ("Ia", 1.0), ("Ib", 1.0), ("Ic", 1.0),
            ("IIa", 1.0), ("IIb", 1.0), ("IIc", 1.0), ("IIIa", 1.0),
        ],
        "defaults": {
            "Ia": ["brGDGT_Ia", "Ia"],
            "Ib": ["brGDGT_Ib", "Ib"],
            "Ic": ["brGDGT_Ic", "Ic"],
            "IIa": ["brGDGT_IIa", "IIa"],
            "IIb": ["brGDGT_IIb", "IIb"],
            "IIc": ["brGDGT_IIc", "IIc"],
            "IIIa": ["brGDGT_IIIa", "IIIa"],
        },
    },
    "IR6ME": {
        "latex": r"\mathrm{IR_{6ME}} = \frac{IIa' + IIb' + IIc' + IIIa' + IIIb' + IIIc'}{IIa + IIb + IIc + IIIa + IIIb + IIIc + IIa' + IIb' + IIc' + IIIa' + IIIb' + IIIc'}",
        "variables": ["IIa", "IIb", "IIc", "IIIa", "IIIb", "IIIc", "IIa'", "IIb'", "IIc'", "IIIa'", "IIIb'", "IIIc'"],
        "numerator_terms": [
            ("IIa'", 1.0), ("IIb'", 1.0), ("IIc'", 1.0),
            ("IIIa'", 1.0), ("IIIb'", 1.0), ("IIIc'", 1.0),
        ],
        "denominator_terms": [
            ("IIa", 1.0), ("IIb", 1.0), ("IIc", 1.0),
            ("IIIa", 1.0), ("IIIb", 1.0), ("IIIc", 1.0),
            ("IIa'", 1.0), ("IIb'", 1.0), ("IIc'", 1.0),
            ("IIIa'", 1.0), ("IIIb'", 1.0), ("IIIc'", 1.0),
        ],
        "defaults": {
            "IIa": ["brGDGT_IIa", "IIa"],
            "IIb": ["brGDGT_IIb", "IIb"],
            "IIc": ["brGDGT_IIc", "IIc"],
            "IIIa": ["brGDGT_IIIa", "IIIa"],
            "IIIb": ["brGDGT_IIIb", "IIIb"],
            "IIIc": ["brGDGT_IIIc", "IIIc"],
            "IIa'": ["brGDGT_IIa_", "IIa'"],
            "IIb'": ["brGDGT_IIb_", "IIb'"],
            "IIc'": ["brGDGT_IIc_", "IIc'"],
            "IIIa'": ["brGDGT_IIIa_", "IIIa'"],
            "IIIb'": ["brGDGT_IIIb_", "IIIb'"],
            "IIIc'": ["brGDGT_IIIc_", "IIIc'"],
        },
    },
    "ACE": {
        "latex": r"\mathrm{ACE} = \frac{archaeol}{archaeol + 10 \times GDGT0} \times 100",
        "variables": ["archaeol", "GDGT0"],
        "numerator_terms": [("archaeol", 100.0)],
        "denominator_terms": [("archaeol", 1.0), ("GDGT0", 10.0)],
        "defaults": {
            "archaeol": ["Archaeol", "archaeol"],
            "GDGT0": ["GDGT0", "GDGT-0"],
        },
    },
    "ACE'": {
        "latex": r"\mathrm{ACE'} = \frac{archaeol}{archaeol + 10 \times brGDGTs} \times 100",
        "variables": ["archaeol", "brGDGTs"],
        "numerator_terms": [("archaeol", 100.0)],
        "denominator_terms": [("archaeol", 1.0), ("brGDGTs", 10.0)],
        "defaults": {
            "archaeol": ["Archaeol", "archaeol"],
            "brGDGTs": ["brGDGT_Ia", "brGDGT_Ib", "brGDGT_Ic", "brGDGT_IIa", "brGDGT_IIb", "brGDGT_IIc", "brGDGT_IIIa", "brGDGT_IIIb", "brGDGT_IIIc"],
        },
        "group_variables": {
            "brGDGTs": [
                "brGDGT_Ia", "brGDGT_Ib", "brGDGT_Ic",
                "brGDGT_IIa", "brGDGT_IIb", "brGDGT_IIc",
                "brGDGT_IIIa", "brGDGT_IIIb", "brGDGT_IIIc",
                "brGDGT_IIa_", "brGDGT_IIb_", "brGDGT_IIc_",
                "brGDGT_IIIa_", "brGDGT_IIIb_", "brGDGT_IIIc_",
                "brGDGT_IIa__", "brGDGT_IIIa__",
            ],
        },
    },
    "IR7ME": {
        "latex": r"\mathrm{IR_{7ME}} = \frac{IIIa'' + IIa''}{IIIa + IIIa' + IIIa'' + IIa + IIa' + IIa''}",
        "variables": ["IIIa", "IIIa'", "IIIa''", "IIa", "IIa'", "IIa''"],
        "numerator_terms": [("IIIa''", 1.0), ("IIa''", 1.0)],
        "denominator_terms": [
            ("IIIa", 1.0), ("IIIa'", 1.0), ("IIIa''", 1.0),
            ("IIa", 1.0), ("IIa'", 1.0), ("IIa''", 1.0),
        ],
        "defaults": {
            "IIIa": ["brGDGT_IIIa", "IIIa"],
            "IIIa'": ["brGDGT_IIIa_", "IIIa'"],
            "IIIa''": ["brGDGT_IIIa__", "IIIa''", "IIIa'''"],
            "IIa": ["brGDGT_IIa", "IIa"],
            "IIa'": ["brGDGT_IIa_", "IIa'"],
            "IIa''": ["brGDGT_IIa__", "IIa''", "IIa'''"],
        },
    },
    "IR6+7ME": {
        "latex": r"\mathrm{IR_{6+7ME}} = \frac{IR_{6ME} + IR_{7ME}}{2}",
        "variables": [],
        "preset_numerator_terms": [("IR6ME", 1.0), ("IR7ME", 1.0)],
        "preset_denominator_terms": [("__const__", 2.0)],
    },
    "IR'6+7ME": {
        "latex": r"\mathrm{IR'_{6+7ME}} = \frac{0.5 \times (IIa' + IIb' + IIc' + IIIa' + IIIb' + IIIc') + IIIa'' + IIa''}{IIa + IIb + IIc + IIIa + IIIb + IIIc + IIa' + IIb' + IIc' + IIIa' + IIIb' + IIIc' + IIIa'' + IIa''}",
        "variables": ["IIa", "IIb", "IIc", "IIIa", "IIIb", "IIIc", "IIa'", "IIb'", "IIc'", "IIIa'", "IIIb'", "IIIc'", "IIIa''", "IIa''"],
        "numerator_terms": [
            ("IIa'", 0.5), ("IIb'", 0.5), ("IIc'", 0.5),
            ("IIIa'", 0.5), ("IIIb'", 0.5), ("IIIc'", 0.5),
            ("IIIa''", 1.0), ("IIa''", 1.0),
        ],
        "denominator_terms": [
            ("IIa", 1.0), ("IIb", 1.0), ("IIc", 1.0),
            ("IIIa", 1.0), ("IIIb", 1.0), ("IIIc", 1.0),
            ("IIa'", 1.0), ("IIb'", 1.0), ("IIc'", 1.0),
            ("IIIa'", 1.0), ("IIIb'", 1.0), ("IIIc'", 1.0),
            ("IIIa''", 1.0), ("IIa''", 1.0),
        ],
        "defaults": {
            "IIa": ["brGDGT_IIa", "IIa"],
            "IIb": ["brGDGT_IIb", "IIb"],
            "IIc": ["brGDGT_IIc", "IIc"],
            "IIIa": ["brGDGT_IIIa", "IIIa"],
            "IIIb": ["brGDGT_IIIb", "IIIb"],
            "IIIc": ["brGDGT_IIIc", "IIIc"],
            "IIa'": ["brGDGT_IIa_", "IIa'"],
            "IIb'": ["brGDGT_IIb_", "IIb'"],
            "IIc'": ["brGDGT_IIc_", "IIc'"],
            "IIIa'": ["brGDGT_IIIa_", "IIIa'"],
            "IIIb'": ["brGDGT_IIIb_", "IIIb'"],
            "IIIc'": ["brGDGT_IIIc_", "IIIc'"],
            "IIIa''": ["brGDGT_IIIa__", "IIIa''", "IIIa'''"],
            "IIa''": ["brGDGT_IIa__", "IIa''", "IIa'''"],
        },
    },
    "TEX86": {
        "latex": r"\mathrm{TEX_{86}} = \frac{GDGT\!-\!2 + GDGT\!-\!3 + cren'}{GDGT\!-\!1 + GDGT\!-\!2 + GDGT\!-\!3 + cren'}",
        "variables": ["GDGT1", "GDGT2", "GDGT3", "cren'"],
        "numerator_terms": [("GDGT2", 1.0), ("GDGT3", 1.0), ("cren'", 1.0)],
        "denominator_terms": [("GDGT1", 1.0), ("GDGT2", 1.0), ("GDGT3", 1.0), ("cren'", 1.0)],
        "defaults": {
            "GDGT1": ["GDGT1", "GDGT-1"],
            "GDGT2": ["GDGT2", "GDGT-2"],
            "GDGT3": ["GDGT3", "GDGT-3"],
            "cren'": ["cren_", "cren'", "cren_prime", "Crenarchaeol regioisomer"],
        },
    },
}


def _find_default_result_column(columns: list[str], candidates: list[str]) -> str | None:
    """Return the first result column matching any candidate name."""
    normalized_columns = {
        _normalize_name(col): col
        for col in columns
    }
    for candidate in candidates:
        match = normalized_columns.get(_normalize_name(candidate))
        if match is not None:
            return match
    return None


def _compute_ratio_index(
    df: pd.DataFrame,
    numerator_columns: list[str],
    denominator_columns: list[str],
) -> pd.Series:
    """Compute a row-wise ratio index from integration-result columns."""
    numerator = df[numerator_columns].sum(axis=1, min_count=1)
    denominator = df[denominator_columns].sum(axis=1, min_count=1)
    return numerator.div(denominator.where(denominator != 0, np.nan))


def _compute_weighted_sum(
    df: pd.DataFrame,
    mapped_columns: dict[str, str | list[str]],
    terms: list[tuple[str, float]],
) -> pd.Series:
    """Compute a weighted row-wise sum from mapped variables."""
    result = pd.Series(0.0, index=df.index, dtype=float)
    has_any = pd.Series(False, index=df.index, dtype=bool)

    for variable, coefficient in terms:
        if variable == "__const__":
            result = result + float(coefficient)
            has_any[:] = True
            continue
        selection = mapped_columns.get(variable)
        if selection is None:
            continue
        columns = selection if isinstance(selection, list) else [selection]
        if not columns:
            continue
        series = df[columns].sum(axis=1, min_count=1)
        has_any = has_any | series.notna()
        result = result + float(coefficient) * series.fillna(0.0)

    result = result.where(has_any, np.nan)
    return result


def _build_peak_from_bounds(
    compound_name: str,
    rt_axis: np.ndarray,
    intensity_axis: np.ndarray,
    rtmin: float,
    rtmax: float,
) -> PickedPeak | None:
    """Create a picked peak from explicit RT bounds."""
    left, right = sorted((float(rtmin), float(rtmax)))
    mask = (rt_axis >= left) & (rt_axis <= right)
    if not np.any(mask):
        return None

    rt_win = np.asarray(rt_axis[mask], dtype=float)
    intensity_win = np.asarray(intensity_axis[mask], dtype=float)
    if rt_win.size == 0 or intensity_win.size == 0:
        return None

    apex_idx = int(np.nanargmax(intensity_win))
    apex_rt = float(rt_win[apex_idx])
    area = float(np.trapz(intensity_win, rt_win))
    sigma = float(max((right - left) / 6.0, 1.0))

    return PickedPeak(
        name=compound_name,
        rt=apex_rt,
        rtmin=float(rt_win[0]),
        rtmax=float(rt_win[-1]),
        into=area,
        intb=area,
        sigma=sigma,
    )


def _eic_names(exp: Experiment) -> list[str]:
    """Return sorted unique EIC names present in the experiment."""
    return sorted({
        eic.name
        for chrom in exp.chromatograms.values()
        for eic in chrom.eics
    })


def _normalize_name(value: str) -> str:
    """Normalize a compound/EIC name for tolerant matching."""
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def _resolve_source_eic(
    compound_name: str,
    eic_names: list[str],
    explicit_source_eic: str | None = None,
) -> str | None:
    """Map a compound name to the source EIC that should carry its peaks."""
    normalized_lookup = {
        _normalize_name(eic_name): eic_name
        for eic_name in eic_names
    }

    if explicit_source_eic:
        explicit_match = normalized_lookup.get(_normalize_name(explicit_source_eic))
        if explicit_match is not None:
            return explicit_match

    if compound_name in eic_names:
        return compound_name

    normalized_compound = _normalize_name(compound_name)
    exact_match = normalized_lookup.get(normalized_compound)
    if exact_match is not None:
        return exact_match

    matches = [
        eic_name for eic_name in eic_names
        if compound_name.startswith(f"{eic_name}_")
    ]
    if not matches:
        matches = [
            eic_name
            for eic_name in eic_names
            if normalized_compound.startswith(_normalize_name(eic_name))
        ]
    if not matches:
        return None
    return max(matches, key=len)


def _initial_peak_blueprint(
    exp: Experiment,
    yaml_path: str,
    half_window_seconds: float,
) -> dict[str, list[dict[str, float | str | None]]]:
    """Build an editable peak blueprint from the starting YAML."""
    compounds = load_compounds(yaml_path)
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw_compounds = yaml.safe_load(f) or {}
    eic_names = _eic_names(exp)
    blueprint: dict[str, list[dict[str, float | str | None]]] = {name: [] for name in eic_names}

    for compound_name, cmpd in compounds.items():
        if cmpd.rt is None:
            continue
        raw_entry = raw_compounds.get(compound_name, {}) if isinstance(raw_compounds, dict) else {}
        explicit_source_eic = None
        if isinstance(raw_entry, dict):
            explicit_source_eic = raw_entry.get("source_eic") or raw_entry.get("eic")
        source_eic = _resolve_source_eic(compound_name, eic_names, explicit_source_eic=explicit_source_eic)
        if source_eic is None:
            continue
        if cmpd.rtmin is not None and cmpd.rtmax is not None:
            rtmin = float(cmpd.rtmin) * 60.0
            rtmax = float(cmpd.rtmax) * 60.0
        else:
            center = float(cmpd.rt) * 60.0
            rtmin = center - half_window_seconds
            rtmax = center + half_window_seconds
        blueprint.setdefault(source_eic, []).append({
            "name": compound_name,
            "rtmin": rtmin,
            "rtmax": rtmax,
            "mz": float(cmpd.mz),
        })

    for source_eic in blueprint:
        blueprint[source_eic].sort(key=lambda entry: (float(entry["rtmin"]), str(entry["name"])))
    return blueprint


def _default_window_for_compound(
    compound_name: str,
    yaml_path: str | None,
    half_window_seconds: float,
) -> tuple[float, float] | None:
    """Return a YAML-derived default RT window for one compound."""
    if not yaml_path:
        return None
    try:
        compounds = load_compounds(yaml_path)
    except Exception:
        return None
    cmpd = compounds.get(compound_name)
    if cmpd is None:
        return None
    if cmpd.rtmin is not None and cmpd.rtmax is not None:
        return (float(cmpd.rtmin) * 60.0, float(cmpd.rtmax) * 60.0)
    if cmpd.rt is None:
        return None
    center = float(cmpd.rt) * 60.0
    return (center - half_window_seconds, center + half_window_seconds)


def _upsert_blueprint_entry(
    blueprint: dict[str, list[dict[str, float | str | None]]],
    source_eic: str,
    original_name: str | None,
    compound_name: str,
    rtmin: float,
    rtmax: float,
    mz: float | None,
) -> None:
    """Insert or update one compound window under a source EIC."""
    entries = blueprint.setdefault(source_eic, [])
    new_entry = {
        "name": compound_name,
        "rtmin": float(min(rtmin, rtmax)),
        "rtmax": float(max(rtmin, rtmax)),
        "mz": None if mz is None else float(mz),
    }
    replaced = False
    for idx, entry in enumerate(entries):
        if original_name and str(entry["name"]) == original_name:
            entries[idx] = new_entry
            replaced = True
            break
    if not replaced:
        entries.append(new_entry)
    entries.sort(key=lambda entry: (float(entry["rtmin"]), str(entry["name"])))


def _delete_blueprint_entry(
    blueprint: dict[str, list[dict[str, float | str | None]]],
    source_eic: str,
    compound_name: str,
) -> None:
    """Delete one compound window from a source EIC."""
    entries = blueprint.get(source_eic, [])
    blueprint[source_eic] = [
        entry for entry in entries
        if str(entry["name"]) != compound_name
    ]


def _apply_peak_blueprint(
    exp: Experiment,
    peak_blueprint: dict[str, list[dict[str, float | str | None]]],
    use_corrected_rt: bool = False,
) -> Experiment:
    """Return a fresh experiment with peaks rebuilt from the editable blueprint."""
    updated = copy.deepcopy(exp)
    for chrom in updated.chromatograms.values():
        for eic in chrom.eics:
            entries = peak_blueprint.get(eic.name, [])
            peaks: list[PickedPeak] = []
            rt_axis = (
                eic.shifted_rt
                if use_corrected_rt and eic.shifted_rt is not None
                else eic.rt
            )
            for entry in entries:
                peak = _build_peak_from_bounds(
                    compound_name=str(entry["name"]),
                    rt_axis=rt_axis,
                    intensity_axis=eic.intensity,
                    rtmin=float(entry["rtmin"]),
                    rtmax=float(entry["rtmax"]),
                )
                if peak is not None:
                    peaks.append(peak)
            eic.picked = peaks
    return updated


def _blueprint_to_yaml_dict(
    peak_blueprint: dict[str, list[dict[str, float | str | None]]],
    exp: Experiment | None = None,
) -> dict[str, dict[str, float | None]]:
    """Convert the editable blueprint to YAML-ready compound definitions."""
    out: dict[str, dict[str, float | None]] = {}
    for _source_eic, entries in peak_blueprint.items():
        for entry in entries:
            rtmin_s = float(entry["rtmin"])
            rtmax_s = float(entry["rtmax"])
            rt_center_s = (rtmin_s + rtmax_s) / 2.0
            if exp is not None:
                rt_stats = _compound_rt_stats(exp, str(entry["name"]))
                if (
                    np.isfinite(rt_stats["rtmin"])
                    and np.isfinite(rt_stats["rtmax"])
                    and np.isfinite(rt_stats["rtmed"])
                ):
                    rtmin_s = float(rt_stats["rtmin"])
                    rtmax_s = float(rt_stats["rtmax"])
                    rt_center_s = float(rt_stats["rtmed"])

            rtmin_min = rtmin_s / 60.0
            rtmax_min = rtmax_s / 60.0
            rt_center_min = rt_center_s / 60.0
            item = {
                "mz": None if entry.get("mz") is None else float(entry["mz"]),
                "rt": {
                    "min": rtmin_min,
                    "max": rtmax_min,
                    "center": rt_center_min,
                },
            }
            if str(entry["name"]) != str(_source_eic):
                item["source_eic"] = str(_source_eic)
            out[str(entry["name"])] = item
    return out


def _source_eic_for_compound(
    compound_name: str,
    peak_blueprint: dict[str, list[dict[str, float | str | None]]],
) -> str:
    """Resolve which source EIC should be plotted for a picked compound."""
    for source_eic, entries in peak_blueprint.items():
        if any(str(entry["name"]) == compound_name for entry in entries):
            return source_eic
    return compound_name


def _compound_rt_stats(exp: Experiment, compound_name: str) -> dict[str, float]:
    """Compute RT stats directly from picked peaks in the current experiment."""
    rt_vals: list[float] = []
    rtmin_vals: list[float] = []
    rtmax_vals: list[float] = []

    for chrom in exp.chromatograms.values():
        for eic in chrom.eics:
            for peak in eic.picked:
                if peak.name != compound_name:
                    continue
                if np.isfinite(peak.rt):
                    rt_vals.append(float(peak.rt))
                if np.isfinite(peak.rtmin):
                    rtmin_vals.append(float(peak.rtmin))
                if np.isfinite(peak.rtmax):
                    rtmax_vals.append(float(peak.rtmax))

    if not rt_vals or not rtmin_vals or not rtmax_vals:
        return {"rtmin": np.nan, "rtmed": np.nan, "rtmax": np.nan}

    return {
        "rtmin": float(np.min(rtmin_vals)),
        "rtmed": float(np.median(rt_vals)),
        "rtmax": float(np.max(rtmax_vals)),
    }


def _peak_picking_target_exp() -> tuple[Experiment | None, bool, str]:
    """Return the experiment currently used for peak picking."""
    exp_corrected = st.session_state.get("exp_corrected")
    if exp_corrected is not None:
        return exp_corrected, True, "corrected"
    exp_raw = st.session_state.get("exp")
    if exp_raw is not None:
        return exp_raw, False, "raw"
    return None, False, "raw"


def _default_local_yaml_path(yaml_path: str | None) -> str:
    """Suggest a local editable YAML path next to the input YAML."""
    if not yaml_path:
        return "cmpds.local.yaml"
    base, ext = os.path.splitext(yaml_path)
    ext = ext if ext else ".yaml"
    return f"{base}.local{ext}"


def _plotly_zoom_capture(fig: go.Figure, key: str, height: int = 430) -> object:
    """Render a Plotly figure in a custom component that can commit the current zoom range."""
    post_script = """
const gd = document.getElementById('{plot_id}');
if (gd) {
  gd.on('plotly_relayout', function(eventData) {
    const x0 = eventData['xaxis.range[0]'];
    const x1 = eventData['xaxis.range[1]'];
    if (x0 !== undefined && x1 !== undefined && window.setPendingZoomRange) {
      window.setPendingZoomRange([Number(x0), Number(x1)]);
    }
    if (eventData['xaxis.autorange'] && window.setPendingZoomRange) {
      window.setPendingZoomRange(null);
    }
  });
}
"""
    plot_html = pio.to_html(
        fig,
        include_plotlyjs=True,
        full_html=False,
        config={"scrollZoom": True, "displaylogo": False},
        post_script=post_script,
    )
    return _zoom_component(plot_html=plot_html, height=height, key=key, default=None)


# ════════════════════════════════════════════
#  Sidebar — Data Loading
# ════════════════════════════════════════════

with st.sidebar:
    st.markdown("# 🔬 PeakIntegrate")
    st.caption("GDGT Chromatographic Peak Integration")
    st.divider()

    st.markdown("### 📂 Data Source")
    input_mode = st.radio(
        "Input mode",
        options=("CSV + HDF5", "EIC CSV + YAML"),
        help="Keep the original peak-table workflow or build initial peaks directly from wide EIC CSV files.",
    )

    if input_mode == "CSV + HDF5":
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

        if st.button("🚀 Load Data", width="stretch", type="primary"):
            with st.spinner("Loading experiment..."):
                try:
                    exp = load_experiment(
                        datafolder=datafolder,
                        hdf5_path=hdf5_path,
                    )
                    st.session_state["exp"] = exp
                    st.session_state["source_mode"] = "legacy"
                    st.session_state["eic_yaml_path"] = None
                    st.session_state["exp_corrected"] = None
                    st.session_state["exp_clustered"] = None
                    st.session_state["results_df"] = None
                    st.session_state["results_figures"] = {}
                    st.session_state["result_column_aliases"] = {}
                    st.session_state["peak_blueprint"] = {}
                    st.session_state["local_yaml_path"] = ""
                    st.success(f"Loaded {len(exp.chromatograms)} samples")
                except Exception as e:
                    st.error(f"Failed to load: {e}")
    else:
        eic_folder = st.text_input(
            "EIC CSV folder",
            value="/Users/weimin/10-Project/GDGT_peak_integration/eic",
            help="Wide EIC CSV folder. Choose below whether each file is a sample or a compound.",
        )
        eic_file_axis = st.radio(
            "EIC file layout",
            options=("Each file is a sample", "Each file is a compound"),
            index=0,
            help=(
                "Sample layout: filename is sample name, first column RT, remaining columns compounds. "
                "Compound layout: filename is compound name, first column RT, remaining columns samples."
            ),
        )
        eic_rt_unit = st.radio(
            "RT unit in EIC CSV",
            options=("Seconds", "Minutes"),
            index=0,
            help="The first RT column in the EIC CSV files. PeakIntegrate converts everything to seconds internally.",
        )
        cmpds_yaml_path = st.text_input(
            "Compound YAML",
            value=os.path.join(_project_root, "PeakIntegrate", "config", "cmpds.yaml"),
            help="cmpds.yaml with expected RT values used to create initial peak ranges",
        )
        peak_window_seconds = st.number_input(
            "Initial peak half-window (s)",
            min_value=5.0,
            max_value=600.0,
            value=60.0,
            step=5.0,
            help="For each compound, use YAML RT +/- this window to create one initial picked peak per sample.",
        )

        if st.button("🚀 Load EIC Data", width="stretch", type="primary"):
            with st.spinner("Loading EIC traces and generating initial peaks..."):
                try:
                    exp = load_experiment_from_eic_csv(
                        eic_folder=eic_folder,
                        yaml_path=cmpds_yaml_path,
                        window_seconds=float(peak_window_seconds),
                        file_axis="sample" if eic_file_axis == "Each file is a sample" else "compound",
                        rt_unit="seconds" if eic_rt_unit == "Seconds" else "minutes",
                    )
                    st.session_state["exp"] = exp
                    st.session_state["source_mode"] = "eic_csv"
                    st.session_state["eic_yaml_path"] = cmpds_yaml_path
                    st.session_state["eic_default_half_window"] = float(peak_window_seconds)
                    st.session_state["peak_blueprint"] = _initial_peak_blueprint(
                        exp,
                        cmpds_yaml_path,
                        float(peak_window_seconds),
                    )
                    st.session_state["local_yaml_path"] = _default_local_yaml_path(cmpds_yaml_path)
                    st.session_state["exp_corrected"] = None
                    st.session_state["exp_clustered"] = None
                    st.session_state["results_df"] = None
                    st.session_state["results_figures"] = {}
                    st.session_state["result_column_aliases"] = {}
                    st.success(
                        f"Loaded {len(exp.chromatograms)} samples from EICs and created initial peak ranges"
                    )
                except Exception as e:
                    st.error(f"Failed to load: {e}")

    st.divider()

    # ── Save / load full session ──
    st.markdown("### 💾 Project Session")
    suggested_session_path = _default_session_path(
        source_mode="eic_csv" if input_mode == "EIC CSV + YAML" else "legacy",
        eic_yaml_path=cmpds_yaml_path if input_mode == "EIC CSV + YAML" else st.session_state.get("eic_yaml_path"),
        eic_folder=eic_folder if input_mode == "EIC CSV + YAML" else None,
        datafolder=datafolder if input_mode == "CSV + HDF5" else None,
        hdf5_path=hdf5_path if input_mode == "CSV + HDF5" else None,
    )
    session_context = "||".join([
        input_mode,
        str(cmpds_yaml_path if input_mode == "EIC CSV + YAML" else ""),
        str(eic_folder if input_mode == "EIC CSV + YAML" else ""),
        str(datafolder if input_mode == "CSV + HDF5" else ""),
        str(hdf5_path if input_mode == "CSV + HDF5" else ""),
    ])
    if (
        not st.session_state.get("session_path")
        or st.session_state.get("session_path_context") != session_context
    ):
        st.session_state["session_path"] = suggested_session_path
        st.session_state["session_path_context"] = session_context

    session_path = st.text_input(
        "Session file path",
        key="session_path",
        help="Save or restore the full PeakIntegrate working session.",
    )
    sess_col1, sess_col2 = st.columns(2)
    with sess_col1:
        if st.button("💾 Save Session", width="stretch"):
            try:
                bundle = _build_session_bundle()
                with open(session_path, "wb") as f:
                    pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
                st.success(f"Saved project session to {session_path}")
            except Exception as e:
                st.error(f"Failed to save session: {e}")
    with sess_col2:
        if st.button("📥 Load Session", width="stretch"):
            try:
                with open(session_path, "rb") as f:
                    bundle = pickle.load(f)
                _load_session_bundle(bundle)
                _upgrade_session_experiments()
                st.success("Project session restored.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load session: {e}")

    st.divider()

    # ── Load from pickle ──
    st.markdown("### 📦 Or Load Pickle")
    pkl_path = st.text_input(
        "Experiment pickle path",
        value="/Users/weimin/10-Project/GDGT_peak_integration/experiment.pkl",
    )
    if st.button("📥 Load Pickle", width="stretch"):
        try:
            with open(pkl_path, "rb") as f:
                exp = pickle.load(f)
            st.session_state["exp"] = _upgrade_experiment_object(exp)
            st.session_state["source_mode"] = "legacy"
            st.session_state["eic_yaml_path"] = None
            st.session_state["peak_blueprint"] = {}
            st.session_state["local_yaml_path"] = ""
            st.session_state["exp_corrected"] = None
            st.session_state["exp_clustered"] = None
            st.session_state["results_df"] = None
            st.session_state["results_figures"] = {}
            st.session_state["result_column_aliases"] = {}
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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Peak Picking",
    "🔧 RT Correction",
    "📊 Visualization",
    "🔬 Clustering",
    "📈 Integration",
    "🧮 Indices",
])


# ════════════════════════════════════════════
#  Tab 1 — Peak Picking
# ════════════════════════════════════════════

with tab1:
    st.markdown("""
    <div class="step-header">
        <h2>🎯 Step 1 — Peak Picking</h2>
        <p>Adjust retention-time windows for EIC imports and rebuild the initial picked peaks.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.get("source_mode") != "eic_csv":
        st.info("Peak-picking controls are available for the `EIC CSV + YAML` input mode.")
    elif st.session_state["exp"] is None:
        st.warning("⬅️ Load data first using the sidebar.")
    else:
        exp_base, use_corrected_rt, picking_mode = _peak_picking_target_exp()
        if exp_base is None:
            st.warning("⬅️ Load data first using the sidebar.")
            st.stop()
        peak_blueprint = st.session_state["peak_blueprint"]
        all_eics = _eic_names(exp_base)

        if use_corrected_rt:
            st.info("Peak Picking is currently using corrected RT coordinates from the last RT correction run.")
        else:
            st.info("Peak Picking is currently using raw RT coordinates.")

        if not all_eics:
            st.warning("No compounds found in the loaded EIC files.")
        else:
            selected_eic = st.selectbox(
                "Source EIC",
                options=all_eics,
                key="peak_pick_source_eic",
            )
            entries = peak_blueprint.get(selected_eic, [])
            entry_names = [str(entry["name"]) for entry in entries]
            entry_options = ["+ New compound"] + entry_names
            pending_entry = st.session_state["pending_peak_pick_entry"].pop(selected_eic, None)
            entry_index = 0
            if pending_entry in entry_options:
                entry_index = entry_options.index(pending_entry)
            elif selected_eic in entry_options:
                entry_index = entry_options.index(selected_eic)
            elif entry_names:
                entry_index = 1
            selected_entry_name = st.selectbox(
                "Picked compound",
                options=entry_options,
                index=entry_index,
                key=f"peak_pick_entry_{selected_eic}",
            )

            input_versions = st.session_state["peak_input_versions"]
            editor_key = f"{selected_eic}::{selected_entry_name}"
            input_version = int(input_versions.get(editor_key, 0))
            selected_entry = next(
                (entry for entry in entries if str(entry["name"]) == selected_entry_name),
                None,
            )
            current_name = (
                str(selected_entry["name"])
                if selected_entry is not None
                else f"{selected_eic}_new"
            )
            current_bounds = None
            current_mz = None
            if selected_entry is not None:
                current_bounds = (
                    float(selected_entry["rtmin"]),
                    float(selected_entry["rtmax"]),
                )
                current_mz = selected_entry.get("mz")
            if current_bounds is None:
                inherited_entry = next(
                    (entry for entry in entries if str(entry["name"]) == selected_eic),
                    entries[0] if entries else None,
                )
                if inherited_entry is not None:
                    current_bounds = (
                        float(inherited_entry["rtmin"]),
                        float(inherited_entry["rtmax"]),
                    )
                    current_mz = inherited_entry.get("mz")
                else:
                    current_bounds = _default_window_for_compound(
                        selected_eic,
                        st.session_state.get("eic_yaml_path"),
                        float(st.session_state.get("eic_default_half_window", 60.0)),
                    )
            if current_bounds is None:
                eic_example = next(
                    eic
                    for chrom in exp_base.chromatograms.values()
                    for eic in chrom.eics
                    if eic.name == selected_eic
                )
                current_bounds = (float(np.nanmin(eic_example.rt)), float(np.nanmax(eic_example.rt)))
                current_mz = eic_example.mz
            if current_mz is None:
                current_mz = next(
                    (
                        eic.mz
                        for chrom in exp_base.chromatograms.values()
                        for eic in chrom.eics
                        if eic.name == selected_eic and eic.mz is not None
                    ),
                    None,
                )

            col1, col2 = st.columns([2, 1])
            with col1:
                current_bounds_min = (
                    float(current_bounds[0]) / 60.0,
                    float(current_bounds[1]) / 60.0,
                )
                compound_name = st.text_input(
                    "Output compound name",
                    value=current_name,
                    key=f"peak_pick_name_{editor_key}_{input_version}",
                )
                rtmin = st.number_input(
                    "RT min (min)",
                    value=float(current_bounds_min[0]),
                    step=0.1,
                    key=f"peak_pick_rtmin_{editor_key}_{input_version}",
                )
                rtmax = st.number_input(
                    "RT max (min)",
                    value=float(current_bounds_min[1]),
                    step=0.1,
                    key=f"peak_pick_rtmax_{editor_key}_{input_version}",
                )
            with col2:
                matching_peaks = sum(
                    1
                    for chrom in exp_base.chromatograms.values()
                    for eic in chrom.eics
                    for peak in eic.picked
                    if peak.name in entry_names
                )
                st.metric("Picked peaks on EIC", matching_peaks)
                st.caption("One source EIC can hold multiple named windows.")

            if entries:
                st.markdown("**Current windows on this EIC**")
                st.dataframe(
                    pd.DataFrame([
                        {
                            "compound": str(entry["name"]),
                            "rtmin_s": float(entry["rtmin"]),
                            "rtmax_s": float(entry["rtmax"]),
                        }
                        for entry in entries
                    ]),
                    use_container_width=True,
                    hide_index=True,
                )

            fig_pick = go.Figure()
            for sname, chrom in exp_base.chromatograms.items():
                eics = chrom.get_eic(selected_eic)
                if not eics:
                    continue
                eic = eics[0]
                rt_axis = (
                    eic.shifted_rt
                    if use_corrected_rt and eic.shifted_rt is not None
                    else eic.rt
                )
                fig_pick.add_trace(go.Scatter(
                    x=rt_axis / 60.0,
                    y=eic.intensity,
                    mode="lines",
                    name=sname,
                    line=dict(width=1.2),
                ))

            for entry in entries:
                color = "royalblue" if str(entry["name"]) == compound_name.strip() else "seagreen"
                fig_pick.add_vrect(
                    x0=float(entry["rtmin"]) / 60.0,
                    x1=float(entry["rtmax"]) / 60.0,
                    fillcolor=color,
                    opacity=0.10,
                    line_width=0,
                )
                fig_pick.add_vline(x=float(entry["rtmin"]) / 60.0, line_dash="dot", line_color=color)
                fig_pick.add_vline(x=float(entry["rtmax"]) / 60.0, line_dash="dot", line_color=color)
            if selected_entry is None:
                fig_pick.add_vrect(
                    x0=min(rtmin, rtmax),
                    x1=max(rtmin, rtmax),
                    fillcolor="royalblue",
                    opacity=0.16,
                    line_width=0,
                )
                fig_pick.add_vline(x=min(rtmin, rtmax), line_dash="dash", line_color="royalblue")
                fig_pick.add_vline(x=max(rtmin, rtmax), line_dash="dash", line_color="royalblue")
            fig_pick.update_layout(
                title=f"Peak picking preview: {selected_eic}",
                xaxis_title="RT (min)",
                yaxis_title="Intensity",
                template="simple_white",
                height=430,
                dragmode="zoom",
                showlegend=False,
                margin=dict(l=50, r=20, t=40, b=40),
            )
            st.caption("Zoom the x-axis to the peak window you want in minutes, then click `Use current zoom` below the plot.")
            zoom_range = _plotly_zoom_capture(
                fig_pick,
                key=f"peak_pick_plot_{selected_eic}_{selected_entry_name}_{input_version}",
                height=430,
            )
            if (
                isinstance(zoom_range, (list, tuple))
                and len(zoom_range) == 2
            ):
                try:
                    zoom_rtmin = float(zoom_range[0]) * 60.0
                    zoom_rtmax = float(zoom_range[1]) * 60.0
                    normalized = (float(min(zoom_rtmin, zoom_rtmax)), float(max(zoom_rtmin, zoom_rtmax)))
                    target_name = compound_name.strip()
                    if not target_name:
                        st.warning("Enter an output compound name first.")
                    else:
                        st.session_state["latest_zoom_windows"][selected_eic] = normalized
                        zoom_commit_key = f"{selected_eic}::{target_name}"
                        last_zoom_commit = st.session_state["last_zoom_commit"].get(zoom_commit_key)
                        if (
                            isinstance(last_zoom_commit, (list, tuple))
                            and len(last_zoom_commit) == 2
                            and all(abs(a - b) <= 1e-9 for a, b in zip(normalized, last_zoom_commit))
                        ):
                            pass
                        else:
                            current_window = current_bounds
                            current_window = (
                                float(min(current_window[0], current_window[1])),
                                float(max(current_window[0], current_window[1])),
                            )
                            if (
                                any(abs(a - b) > 1e-9 for a, b in zip(normalized, current_window))
                                or selected_entry is None
                                or target_name != current_name
                            ):
                                _upsert_blueprint_entry(
                                    peak_blueprint,
                                    source_eic=selected_eic,
                                    original_name=None if selected_entry is None else current_name,
                                    compound_name=target_name,
                                    rtmin=normalized[0],
                                    rtmax=normalized[1],
                                    mz=current_mz,
                                )
                                st.session_state["last_zoom_commit"][zoom_commit_key] = normalized
                                st.session_state["pending_peak_pick_entry"][selected_eic] = target_name
                                input_versions[editor_key] = input_version + 1
                                updated_exp = _apply_peak_blueprint(
                                    exp_base,
                                    peak_blueprint,
                                    use_corrected_rt=use_corrected_rt,
                                )
                                if use_corrected_rt:
                                    st.session_state["exp_corrected"] = updated_exp
                                else:
                                    st.session_state["exp"] = updated_exp
                                    st.session_state["exp_corrected"] = None
                                st.session_state["exp_clustered"] = None
                                st.session_state["results_df"] = None
                                st.session_state["results_figures"] = {}
                                st.session_state["result_column_aliases"] = {}
                                st.success(
                                    f"Updated {target_name} window to {normalized[0] / 60.0:.2f}–{normalized[1] / 60.0:.2f} min."
                                )
                                st.rerun()
                except (TypeError, ValueError):
                    pass

            if st.session_state["latest_zoom_windows"].get(selected_eic) is not None:
                latest_zoom = st.session_state["latest_zoom_windows"][selected_eic]
                st.caption(
                    f"Last captured zoom on {selected_eic}: {latest_zoom[0] / 60.0:.2f} to {latest_zoom[1] / 60.0:.2f} min"
                )

            action_col1, action_col2 = st.columns(2)
            with action_col1:
                if st.button("Save / Update Compound", width="stretch", type="primary"):
                    target_name = compound_name.strip()
                    if not target_name:
                        st.warning("Enter an output compound name.")
                    else:
                        _upsert_blueprint_entry(
                            peak_blueprint,
                            source_eic=selected_eic,
                            original_name=None if selected_entry is None else current_name,
                            compound_name=target_name,
                            rtmin=rtmin * 60.0,
                            rtmax=rtmax * 60.0,
                            mz=current_mz,
                        )
                        st.session_state["pending_peak_pick_entry"][selected_eic] = target_name
                        input_versions[editor_key] = input_version + 1
                        updated_exp = _apply_peak_blueprint(
                            exp_base,
                            peak_blueprint,
                            use_corrected_rt=use_corrected_rt,
                        )
                        if use_corrected_rt:
                            st.session_state["exp_corrected"] = updated_exp
                        else:
                            st.session_state["exp"] = updated_exp
                            st.session_state["exp_corrected"] = None
                        st.session_state["exp_clustered"] = None
                        st.session_state["results_df"] = None
                        st.session_state["results_figures"] = {}
                        st.session_state["result_column_aliases"] = {}
                        st.success(f"Saved {target_name}.")
                        st.rerun()
            with action_col2:
                if st.button("Delete Compound", width="stretch", disabled=(selected_entry is None)):
                    _delete_blueprint_entry(
                        peak_blueprint,
                        source_eic=selected_eic,
                        compound_name=current_name,
                    )
                    input_versions[editor_key] = input_version + 1
                    updated_exp = _apply_peak_blueprint(
                        exp_base,
                        peak_blueprint,
                        use_corrected_rt=use_corrected_rt,
                    )
                    if use_corrected_rt:
                        st.session_state["exp_corrected"] = updated_exp
                    else:
                        st.session_state["exp"] = updated_exp
                        st.session_state["exp_corrected"] = None
                    st.session_state["exp_clustered"] = None
                    st.session_state["results_df"] = None
                    st.session_state["results_figures"] = {}
                    st.session_state["result_column_aliases"] = {}
                    st.success(f"Deleted {current_name}.")
                    st.rerun()

            st.divider()
            if st.button("Reset All Windows from YAML", width="stretch"):
                yaml_path = st.session_state.get("eic_yaml_path")
                if not yaml_path:
                    st.error("No YAML path stored for this EIC session.")
                else:
                    st.session_state["peak_blueprint"] = _initial_peak_blueprint(
                        exp_base,
                        yaml_path,
                        half_window_seconds=float(st.session_state.get("eic_default_half_window", 60.0)),
                    )
                    input_versions[editor_key] = input_version + 1
                    updated_exp = _apply_peak_blueprint(
                        exp_base,
                        st.session_state["peak_blueprint"],
                        use_corrected_rt=use_corrected_rt,
                    )
                    if use_corrected_rt:
                        st.session_state["exp_corrected"] = updated_exp
                    else:
                        st.session_state["exp"] = updated_exp
                        st.session_state["exp_corrected"] = None
                    st.session_state["exp_clustered"] = None
                    st.session_state["results_df"] = None
                    st.session_state["results_figures"] = {}
                    st.session_state["result_column_aliases"] = {}
                    st.success("Reset peak windows from YAML defaults.")
                    st.rerun()

            st.divider()
            local_yaml_exp = _active_exp()
            local_yaml = _blueprint_to_yaml_dict(peak_blueprint, exp=local_yaml_exp)
            local_yaml_text = yaml.safe_dump(local_yaml, sort_keys=False, allow_unicode=False)
            local_yaml_path = st.text_input(
                "Local YAML path",
                value=st.session_state.get("local_yaml_path") or _default_local_yaml_path(st.session_state.get("eic_yaml_path")),
                key="local_yaml_path_input",
            )
            st.session_state["local_yaml_path"] = local_yaml_path
            st.download_button(
                "Download Local YAML",
                data=local_yaml_text,
                file_name=os.path.basename(local_yaml_path) or "cmpds.local.yaml",
                mime="text/yaml",
                width="stretch",
            )
            if st.button("Save Local YAML", width="stretch"):
                try:
                    with open(local_yaml_path, "w", encoding="utf-8") as f:
                        f.write(local_yaml_text)
                    st.success(f"Saved local YAML to {local_yaml_path}")
                except Exception as e:
                    st.error(f"Failed to save YAML: {e}")
            with st.expander("Preview Local YAML"):
                st.code(local_yaml_text, language="yaml")


# ════════════════════════════════════════════
#  Tab 2 — RT Correction
# ════════════════════════════════════════════

with tab2:
    st.markdown("""
    <div class="step-header">
        <h2>🔧 Step 2 — Retention Time Correction</h2>
        <p>Align retention times across samples using the peaks chosen in the previous step.</p>
    </div>
    """, unsafe_allow_html=True)

    exp = st.session_state["exp"]
    if exp is None:
        st.warning("⬅️ Load data first using the sidebar.")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Alignment Model")

            all_compounds = sorted({
                peak.name
                for chrom in exp.chromatograms.values()
                for eic in chrom.eics
                for peak in eic.picked
            })

            method = st.radio(
                "Correction method",
                options=["loess", "polynomial"],
                format_func=lambda x: "LOESS (all shared picked peaks)" if x == "loess" else "Polynomial (selected calibrants)",
                horizontal=True,
            )

            default_calibs = ["C46-GDGT", "brGDGT_Ib", "brGDGT_Ia"]
            if method == "loess":
                st.caption("LOESS uses all compounds with exactly one picked peak in both the sample and the reference. Optionally restrict it to a subset below.")
                calibs = st.multiselect(
                    "Optional compound subset",
                    options=all_compounds,
                    default=[],
                )
                loess_frac = st.slider("LOESS span", 0.2, 1.0, 0.4, 0.05)
                degree = 2
            else:
                calibs = st.multiselect(
                    "Calibration compounds",
                    options=all_compounds,
                    default=[c for c in default_calibs if c in all_compounds],
                )
                degree = st.slider("Polynomial degree", 1, 4, 2)
                loess_frac = 0.4

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

            if st.session_state["manual_anchors"]:
                st.markdown("**Current anchors:**")
                for s, pairs in st.session_state["manual_anchors"].items():
                    for obs, tgt in pairs:
                        st.markdown(f"- `{s}`: {obs:.1f} → {tgt:.1f}")
                if st.button("🗑️ Clear All Anchors"):
                    st.session_state["manual_anchors"] = {}
                    st.rerun()

        st.divider()

        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("▶️  Run RT Correction", width="stretch", type="primary"):
                with st.spinner("Correcting retention times..."):
                    try:
                        exp_c = exp.rt_shift(
                            calibs=calibs if calibs else None,
                            degree=degree,
                            ref_sample_name=ref_sample,
                            manual_anchors=st.session_state["manual_anchors"] or None,
                            method=method,
                            loess_frac=loess_frac,
                        )
                        st.session_state["exp_corrected"] = exp_c
                        st.session_state["exp_clustered"] = None
                        st.session_state["results_df"] = None
                        st.session_state["results_figures"] = {}
                        st.session_state["result_column_aliases"] = {}
                        st.markdown("""
                        <div class="success-banner">
                            ✅ <strong>RT correction complete!</strong> Switch to the Visualization tab to inspect results.
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"RT correction failed: {e}")

        with action_col2:
            if st.button(
                "↩️ Undo RT Correction",
                width="stretch",
                disabled=(st.session_state.get("exp_corrected") is None),
            ):
                st.session_state["exp_corrected"] = None
                st.session_state["exp_clustered"] = None
                st.session_state["results_df"] = None
                st.session_state["results_figures"] = {}
                st.session_state["result_column_aliases"] = {}
                st.success("RT correction cleared. You are back to the uncorrected experiment.")
                st.rerun()

        exp_c = st.session_state.get("exp_corrected")
        if exp_c is not None and exp is not None:
            st.markdown("#### Before / After Comparison")
            cmpd_compare = st.selectbox(
                "Compound to compare",
                options=all_compounds,
                key="rt_compare_cmpd",
            )
            compare_source_eic = _source_eic_for_compound(
                cmpd_compare,
                st.session_state.get("peak_blueprint", {}),
            )

            fig_before = go.Figure()
            fig_after = go.Figure()

            for sname, chrom in exp.chromatograms.items():
                eics = chrom.get_eic(compare_source_eic)
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
                eics = chrom.get_eic(compare_source_eic)
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

            rt_diagnostics = getattr(exp_c, "rt_diagnostics", {}) or {}
            if rt_diagnostics:
                st.markdown("#### RT Correction Curve")
                fig_curve = go.Figure()
                for sample_name in sorted(rt_diagnostics.keys()):
                    diag = rt_diagnostics[sample_name]
                    fig_curve.add_trace(go.Scatter(
                        x=diag["grid_corrected_rts"],
                        y=diag["grid_shift"],
                        mode="lines",
                        name=sample_name,
                        line=dict(width=1.5),
                        opacity=0.8,
                        hovertemplate=(
                            f"sample={sample_name}"
                            "<br>rt_adj=%{x:.2f}"
                            "<br>rt_adj - rt_raw=%{y:.2f}"
                            "<extra></extra>"
                        ),
                    ))
                    fig_curve.add_trace(go.Scatter(
                        x=diag["anchor_corrected_rts"],
                        y=diag["anchor_shift"],
                        mode="markers",
                        name=f"{sample_name} anchors",
                        marker=dict(size=4, opacity=0.45),
                        text=diag.get("anchor_labels"),
                        showlegend=False,
                        hovertemplate=(
                            f"sample={sample_name}"
                            "<br>anchor=%{text}"
                            "<br>rt_adj=%{x:.2f}"
                            "<br>rt_adj - rt_raw=%{y:.2f}"
                            "<extra></extra>"
                        ),
                    ))
                fig_curve.update_layout(
                    title="RT warp curves across all samples",
                    xaxis_title="rt_adj",
                    yaxis_title="rt_adj - rt_raw",
                    template="simple_white",
                    height=420,
                    margin=dict(l=50, r=20, t=40, b=40),
                    legend_title="Sample",
                )
                st.plotly_chart(fig_curve, use_container_width=True)


# ════════════════════════════════════════════
#  Tab 3 — Visualization
# ════════════════════════════════════════════

with tab3:
    st.markdown("""
    <div class="step-header">
        <h2>📊 Step 3 — Visualization</h2>
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
#  Tab 4 — Peak Clustering
# ════════════════════════════════════════════

with tab4:
    st.markdown("""
    <div class="step-header">
        <h2>🔬 Step 4 — Peak Clustering</h2>
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
            if st.button("▶️  Run Clustering", width="stretch", type="primary"):
                if not config:
                    st.warning("Add at least one compound to cluster.")
                else:
                    with st.spinner("Clustering peaks..."):
                        try:
                            exp_clust = copy.deepcopy(exp_pre_cluster)
                            exp_clust = exp_clust.point_cluster_batch(dict(config))
                            st.session_state["exp_clustered"] = exp_clust
                            st.session_state["results_df"] = None
                            st.session_state["results_figures"] = {}
                            st.session_state["result_column_aliases"] = {}

                            st.markdown("""
                            <div class="success-banner">
                                ✅ <strong>Clustering complete!</strong> Check the Visualization tab to see the results.
                            </div>
                            """, unsafe_allow_html=True)

                        except Exception as e:
                            st.error(f"Clustering failed: {e}")

        with btn_col2:
            if st.button("🤖 Auto-Cluster All Compounds", width="stretch"):
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
                        st.session_state["results_figures"] = {}
                        st.session_state["result_column_aliases"] = {}

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
#  Tab 5 — Integration & Export
# ════════════════════════════════════════════

with tab5:
    st.markdown("""
    <div class="step-header">
        <h2>📈 Step 5 — Integration & Export</h2>
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

            if st.session_state.get("source_mode") == "eic_csv":
                blueprint_targets = [
                    str(entry["name"])
                    for entries in st.session_state.get("peak_blueprint", {}).values()
                    for entry in entries
                ]
                default_selection = [
                    c for c in blueprint_targets if c in available_peak_names
                ]
            else:
                default_selection = [
                    c for c in TARGET_COMPOUNDS if c in available_peak_names
                ]

            if "int_cmpd_multiselect" not in st.session_state:
                st.session_state["int_cmpd_multiselect"] = default_selection
            else:
                st.session_state["int_cmpd_multiselect"] = [
                    c for c in st.session_state["int_cmpd_multiselect"]
                    if c in available_peak_names
                ]

            sel_col, clr_col = st.columns([1, 1])
            with sel_col:
                if st.button("✅ Select All", width="stretch"):
                    st.session_state["int_cmpd_multiselect"] = available_peak_names
            with clr_col:
                if st.button("❌ Clear", width="stretch"):
                    st.session_state["int_cmpd_multiselect"] = []

            target_cmpds = st.multiselect(
                "Compounds to integrate",
                options=available_peak_names,
                key="int_cmpd_multiselect",
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
            propagate_splits = st.toggle(
                "↔️ Propagate split peaks across samples",
                value=True,
                help="If enabled, compounds detected as multi-peak in some samples can be forced to the same split layout in other samples.",
            )
            baseline_scope = st.selectbox(
                "Baseline estimation window",
                options=list(VALID_BASELINE_SCOPES),
                format_func=lambda s: "Narrow RT window" if s == "narrow" else "Global RT window",
                index=0,
                disabled=not subtract_bl,
                help="Use only the compound RT window or estimate the baseline from the full chromatogram trace.",
            )

        st.divider()

        if target_cmpds:
            diagnostics: list[dict[str, object]] = []
            peak_blueprint = st.session_state.get("peak_blueprint", {})
            for cmpd in target_cmpds:
                rt_stats = _compound_rt_stats(exp_int, cmpd)
                source_eic = (
                    _source_eic_for_compound(cmpd, peak_blueprint)
                    if st.session_state.get("source_mode") == "eic_csv"
                    else cmpd
                )
                samples_with_peak = sum(
                    1
                    for chrom in exp_int.chromatograms.values()
                    if any(
                        peak.name == cmpd
                        for eic in chrom.eics
                        for peak in eic.picked
                    )
                )
                diagnostics.append({
                    "compound": cmpd,
                    "source_eic": source_eic,
                    "rtmin_s": rt_stats["rtmin"],
                    "rtmed_s": rt_stats["rtmed"],
                    "rtmax_s": rt_stats["rtmax"],
                    "samples_with_peak": samples_with_peak,
                })

            with st.expander("Integration diagnostics", expanded=False):
                st.dataframe(pd.DataFrame(diagnostics), use_container_width=True, hide_index=True)

        if st.button("▶️  Run Integration", width="stretch", type="primary"):
            if not target_cmpds:
                st.warning("Select at least one compound.")
            else:
                with st.spinner(f"Fitting {model_labels[selected_model]} models and integrating peaks..."):
                    try:
                        df, fit_results = integrate_experiment(
                            experiment=exp_int,          # pass object directly — no pickle round-trip
                            output_csv=None,
                            output_pdf=None,
                            target_cmpds=target_cmpds,
                            fit_model=selected_model,
                            subtract_baseline=subtract_bl,
                            baseline_scope=baseline_scope,
                            min_points=min_points,
                            savgol_window=savgol_window,
                            savgol_poly=savgol_poly,
                            prominence_frac=prominence_frac,
                            propagate_consensus_splits=propagate_splits,
                            return_fit_results=True,
                        )

                        st.session_state["results_df"] = df
                        st.session_state["results_figures"] = build_sample_overlay_figures(fit_results)

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
            display_df = results_df.rename(columns=_display_result_name)

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

            st.markdown("#### Column Labels")
            alias_df = _result_alias_frame(list(results_df.columns))
            edited_alias_df = st.data_editor(
                alias_df,
                width="stretch",
                height=min(360, 36 + 35 * len(alias_df)),
                hide_index=True,
                disabled=["original_name"],
                column_config={
                    "original_name": st.column_config.TextColumn("Original name"),
                    "display_name": st.column_config.TextColumn("Display name"),
                },
                key="result_alias_editor",
            )
            st.session_state["result_column_aliases"] = {
                row["original_name"]: row["display_name"].strip()
                for row in edited_alias_df.to_dict("records")
                if isinstance(row["display_name"], str) and row["display_name"].strip()
            }
            display_df = results_df.rename(columns=_display_result_name)

            st.dataframe(
                display_df.style.format("{:.2e}", na_rep="NaN"),
                width="stretch",
                height=400,
            )

            results_figures = st.session_state.get("results_figures", {})
            if results_figures:
                st.markdown("#### Interactive Sample View")
                sample_names = list(results_figures.keys())
                selected_sample = st.selectbox(
                    "Sample to inspect",
                    options=sample_names,
                    key="integration_sample_view",
                )
                sample_fig = results_figures[selected_sample]
                st.plotly_chart(sample_fig, use_container_width=True)

            # ── Export buttons ──
            st.markdown("#### Export")
            ecol1, ecol2, ecol3, ecol4 = st.columns(4)

            with ecol1:
                csv_data = display_df.to_csv()
                st.download_button(
                    "📥 Download CSV",
                    data=csv_data,
                    file_name="peak_integration_results.csv",
                    mime="text/csv",
                    width="stretch",
                )

            with ecol2:
                if results_figures:
                    selected_sample = st.session_state.get("integration_sample_view")
                    sample_fig = results_figures.get(selected_sample) if selected_sample else None
                    html_data = (
                        sample_fig.to_html(full_html=True, include_plotlyjs="cdn")
                        if sample_fig is not None else None
                    )
                    st.download_button(
                        "🌐 Download Sample HTML",
                        data=html_data,
                        file_name=f"{selected_sample}_integration.html",
                        mime="text/html",
                        width="stretch",
                    )
                else:
                    st.button(
                        "🌐 No HTML figure",
                        disabled=True,
                        width="stretch",
                        help="Run integration to generate interactive sample figures.",
                    )

            with ecol3:
                if results_figures:
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                        for sample_name, fig in results_figures.items():
                            safe_name = "".join(
                                ch if ch.isalnum() or ch in ("-", "_") else "_"
                                for ch in sample_name
                            )
                            zf.writestr(
                                f"{safe_name}_integration.html",
                                fig.to_html(full_html=True, include_plotlyjs="cdn"),
                            )
                    zip_buffer.seek(0)
                    st.download_button(
                        "🗂️ Download All HTMLs",
                        data=zip_buffer.getvalue(),
                        file_name="all_sample_integrations.zip",
                        mime="application/zip",
                        width="stretch",
                    )
                else:
                    st.button(
                        "🗂️ No HTML figures",
                        disabled=True,
                        width="stretch",
                        help="Run integration to generate interactive sample figures.",
                    )

            with ecol4:
                pkl_save_path = st.text_input(
                    "Save experiment to",
                    value="experiment.pkl",
                    key="pkl_save_path",
                )
                if st.button("💾 Save Experiment (.pkl)", width="stretch"):
                    try:
                        with open(pkl_save_path, "wb") as f:
                            pickle.dump(exp_int, f, protocol=pickle.HIGHEST_PROTOCOL)
                        st.success(f"Experiment saved to {pkl_save_path}")
                    except Exception as e:
                        st.error(f"Failed: {e}")


# ════════════════════════════════════════════
#  Tab 6 — Indices
# ════════════════════════════════════════════

with tab6:
    st.markdown("""
    <div class="step-header">
        <h2>🧮 Step 6 — Indices</h2>
        <p>Calculate derived indices from the integration table by mapping each formula term to a result column.</p>
    </div>
    """, unsafe_allow_html=True)

    results_df = st.session_state.get("results_df")
    if results_df is None:
        st.warning("Run integration first to generate the peak-area table used for index calculations.")
    else:
        result_columns = list(results_df.columns)
        computed_indices: dict[str, pd.Series] = {}
        custom_presets = st.session_state.get("custom_index_presets", {})
        all_presets = {**INDEX_PRESETS, **custom_presets}
        all_preset_names = list(all_presets.keys())
        active_preset_names = st.session_state.get("active_index_presets", [])
        if not st.session_state.get("active_index_presets_initialized", False):
            active_preset_names = list(INDEX_PRESETS.keys())
            st.session_state["active_index_presets_initialized"] = True
        active_preset_names = [name for name in active_preset_names if name in all_preset_names]
        st.session_state["active_index_presets"] = active_preset_names

        st.markdown("#### Preset Library")
        st.multiselect(
            "Preset indices to calculate",
            options=all_preset_names,
            key="active_index_presets",
        )

        st.markdown("#### Preset Indices")
        for preset_name in st.session_state.get("active_index_presets", []):
            preset = all_presets[preset_name]
            with st.container(border=True):
                header_col, action_col = st.columns([4, 1])
                with header_col:
                    st.markdown(f"##### {preset_name}")
                with action_col:
                    if preset.get("user_defined"):
                        if st.button("Delete", key=f"delete_index_preset_{preset_name}", width="stretch"):
                            custom_presets.pop(preset_name, None)
                            st.session_state["custom_index_presets"] = custom_presets
                            st.session_state["active_index_presets"] = [
                                name for name in st.session_state.get("active_index_presets", [])
                                if name != preset_name
                            ]
                            st.rerun()
                st.latex(preset["latex"])

                preset_numerator_terms = preset.get("preset_numerator_terms")
                preset_denominator_terms = preset.get("preset_denominator_terms")
                if preset_numerator_terms is not None and preset_denominator_terms is not None:
                    numerator = pd.Series(0.0, index=results_df.index, dtype=float)
                    denominator = pd.Series(0.0, index=results_df.index, dtype=float)
                    for dep_name, coefficient in preset_numerator_terms:
                        if dep_name not in computed_indices:
                            numerator[:] = np.nan
                            break
                        numerator = numerator + float(coefficient) * computed_indices[dep_name]
                    for dep_name, coefficient in preset_denominator_terms:
                        if dep_name == "__const__":
                            denominator = denominator + float(coefficient)
                            continue
                        if dep_name not in computed_indices:
                            denominator[:] = np.nan
                            break
                        denominator = denominator + float(coefficient) * computed_indices[dep_name]
                    computed_indices[preset_name] = numerator.div(denominator.where(denominator != 0, np.nan))
                    st.caption("This preset is calculated from previously defined preset indices.")
                    continue

                fixed_numerator_columns = preset.get("fixed_numerator_columns")
                fixed_denominator_columns = preset.get("fixed_denominator_columns")
                if fixed_numerator_columns is not None and fixed_denominator_columns is not None:
                    computed_indices[preset_name] = _compute_ratio_index(
                        results_df,
                        numerator_columns=fixed_numerator_columns,
                        denominator_columns=fixed_denominator_columns,
                    )
                    st.caption("This saved custom preset uses the stored numerator/denominator column selection.")
                    continue

                mapped_columns: dict[str, str | list[str]] = {}
                preset_variables = preset.get("variables", [])
                variable_columns = st.columns(min(4, max(1, len(preset_variables) or 1)))
                for idx, variable in enumerate(preset_variables):
                    defaults = preset.get("defaults", {}).get(variable, [variable])
                    group_defaults = [
                        col for col in preset.get("group_variables", {}).get(variable, [])
                        if col in result_columns
                    ]
                    default_col = _find_default_result_column(result_columns, defaults)
                    default_selection = (
                        group_defaults
                        if group_defaults
                        else ([default_col] if default_col in result_columns else [])
                    )
                    with variable_columns[idx % len(variable_columns)]:
                        mapped_columns[variable] = st.multiselect(
                            f"{variable}",
                            options=result_columns,
                            default=default_selection,
                            format_func=_display_result_name,
                            key=f"index_preset_{preset_name}_{variable}",
                            help="You can select one or multiple columns; multiple selections are summed.",
                        )

                numerator = _compute_weighted_sum(
                    results_df,
                    mapped_columns=mapped_columns,
                    terms=preset["numerator_terms"],
                )
                denominator = _compute_weighted_sum(
                    results_df,
                    mapped_columns=mapped_columns,
                    terms=preset["denominator_terms"],
                )
                computed_indices[preset_name] = numerator.div(denominator.where(denominator != 0, np.nan))

        st.divider()
        st.markdown("#### Custom Ratio")
        custom_col1, custom_col2 = st.columns([1, 2])
        with custom_col1:
            custom_index_name = st.text_input(
                "Index name",
                value="Custom index",
                key="custom_index_name",
            ).strip() or "Custom index"
        with custom_col2:
            st.caption("Build any ratio as sum(numerator columns) / sum(denominator columns).")

        numerator_columns = st.multiselect(
            "Numerator columns",
            options=result_columns,
            format_func=_display_result_name,
            key="custom_index_numerator",
        )
        denominator_columns = st.multiselect(
            "Denominator columns",
            options=result_columns,
            format_func=_display_result_name,
            key="custom_index_denominator",
        )

        if numerator_columns and denominator_columns:
            computed_indices[custom_index_name] = _compute_ratio_index(
                results_df,
                numerator_columns=numerator_columns,
                denominator_columns=denominator_columns,
            )
            if st.button("Save Custom Ratio as Preset", type="primary"):
                if custom_index_name in INDEX_PRESETS:
                    st.error("Choose a different name. This one is already used by a built-in preset.")
                else:
                    st.session_state["custom_index_presets"][custom_index_name] = {
                        "latex": custom_index_name,
                        "fixed_numerator_columns": list(numerator_columns),
                        "fixed_denominator_columns": list(denominator_columns),
                        "user_defined": True,
                    }
                    active_now = st.session_state.get("active_index_presets", [])
                    if custom_index_name not in active_now:
                        st.session_state["active_index_presets"] = active_now + [custom_index_name]
                    st.success(f"Saved preset: {custom_index_name}")
                    st.rerun()

        st.divider()
        if computed_indices:
            index_df = pd.DataFrame(computed_indices, index=results_df.index)
            index_df.index.name = "Sample"

            st.markdown("#### Index Results")
            st.dataframe(
                index_df.style.format("{:.4f}", na_rep="NaN"),
                width="stretch",
                height=420,
            )

            st.download_button(
                "📥 Download Index CSV",
                data=index_df.to_csv(),
                file_name="peak_integrate_indices.csv",
                mime="text/csv",
                width="stretch",
            )
        else:
            st.info("Select the columns for a preset formula or define a custom ratio to calculate indices.")
