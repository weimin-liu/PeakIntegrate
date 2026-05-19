"""
config.py — Compound configuration loader.

Reads ``cmpds.yaml`` and resolves compound names for integration,
handling the suffix convention (e.g. ``brGDGT_IIIa`` → ``brGDGT_IIIa_0``,
``brGDGT_IIIa_1``, ``brGDGT_IIIa_2`` after clustering).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import yaml


# ════════════════════════════════════════════
#  Data Types
# ════════════════════════════════════════════

@dataclass
class CompoundDef:
    """Definition of a single compound from cmpds.yaml.

    Attributes:
        name:        Compound name as it appears in the YAML key.
        mz:          Target m/z value.
        rt:          Expected retention time center in minutes (may be ``None``).
        rtmin:       Optional left RT boundary in minutes.
        rtmax:       Optional right RT boundary in minutes.
        source_eic:  Optional source EIC name for app-side mapping.
    """
    name: str
    mz: float
    rt: Optional[float]
    rtmin: Optional[float] = None
    rtmax: Optional[float] = None
    source_eic: Optional[str] = None


def _parse_rt_value(rt_value: object) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Parse ``rt`` as either a scalar center or a min/max window in minutes."""
    if rt_value is None:
        return None, None, None

    if isinstance(rt_value, (int, float)):
        return float(rt_value), None, None

    if isinstance(rt_value, dict):
        rtmin_raw = rt_value.get("min")
        rtmax_raw = rt_value.get("max")
        center_raw = rt_value.get("center")

        rtmin = float(rtmin_raw) if rtmin_raw is not None else None
        rtmax = float(rtmax_raw) if rtmax_raw is not None else None
        center = float(center_raw) if center_raw is not None else None

        if center is None and rtmin is not None and rtmax is not None:
            center = (rtmin + rtmax) / 2.0

        return center, rtmin, rtmax

    raise TypeError("rt must be a number, a {min,max[,center]} mapping, or null")


# ════════════════════════════════════════════
#  Loading
# ════════════════════════════════════════════

_DEFAULT_YAML = os.path.join(
    os.path.dirname(__file__), "..", "config", "cmpds.yaml"
)


def load_compounds(yaml_path: str = _DEFAULT_YAML) -> dict[str, CompoundDef]:
    """Load compound definitions from a YAML file.

    Parameters:
        yaml_path: Path to the YAML file.  Defaults to
                   ``PeakIntegrate/config/cmpds.yaml``.

    Returns:
        ``dict[name, CompoundDef]`` keyed by compound name.
    """
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    compounds: dict[str, CompoundDef] = {}
    for name, props in raw.items():
        rt, rtmin, rtmax = _parse_rt_value(props.get("rt"))
        compounds[name] = CompoundDef(
            name=name,
            mz=float(props["mz"]),
            rt=rt,
            rtmin=rtmin,
            rtmax=rtmax,
            source_eic=props.get("source_eic") or props.get("eic"),
        )
    return compounds


def resolve_target_compounds(
    compounds: dict[str, CompoundDef],
    cluster_config: Optional[dict[str, int]] = None,
) -> list[str]:
    """Build a list of target compound names for integration.

    For compounds that were clustered, generates suffixed names
    (e.g. ``brGDGT_IIIa`` with 3 clusters → ``brGDGT_IIIa_0``,
    ``brGDGT_IIIa_1``, ``brGDGT_IIIa_2``). Non-clustered compounds
    keep their original YAML name.

    Compound variants already in the YAML with a suffix matching a
    parent compound (e.g. ``brGDGT_IIIa_1`` when ``brGDGT_IIIa`` is
    a cluster parent) are excluded to avoid duplicates.

    Parameters:
        compounds:      Output of :func:`load_compounds`.
        cluster_config: ``{base_compound: n_clusters}`` dict.  Defaults
                        to ``{"brGDGT_IIIa": 3, "brGDGT_IIa": 2}``.

    Returns:
        List of compound names ready for integration.
    """
    if cluster_config is None:
        cluster_config = {"brGDGT_IIIa": 3, "brGDGT_IIa": 2}

    # Names that are children of a clustered parent
    clustered_parents = set(cluster_config.keys())

    targets: list[str] = []
    seen: set[str] = set()

    for name in compounds:
        # Check if this is a child of a clustered parent
        is_child = any(
            name != parent and name.startswith(parent + "_")
            for parent in clustered_parents
        )
        if is_child:
            continue  # skip — will be generated from the parent

        if name in clustered_parents:
            # Generate suffixed names
            n = cluster_config[name]
            for i in range(n):
                suffixed = f"{name}_{i}"
                if suffixed not in seen:
                    targets.append(suffixed)
                    seen.add(suffixed)
        else:
            if name not in seen:
                targets.append(name)
                seen.add(name)

    return targets


def get_base_compounds(yaml_path: str = _DEFAULT_YAML) -> list[str]:
    """Return just the base compound names (no isomer variants).

    Useful for the R preprocessing step which only needs the parent
    compound names.
    """
    compounds = load_compounds(yaml_path)

    # Find which names are children of other names
    all_names = list(compounds.keys())
    base_names: list[str] = []

    for name in all_names:
        is_child = any(
            name != other and name.startswith(other + "_")
            for other in all_names
        )
        if not is_child:
            base_names.append(name)

    return base_names
