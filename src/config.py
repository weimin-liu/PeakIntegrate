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
        name:  Compound name as it appears in the YAML key.
        mz:    Target m/z value.
        rt:    Expected retention time in minutes (may be ``None``).
    """
    name: str
    mz: float
    rt: Optional[float]


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
        compounds[name] = CompoundDef(
            name=name,
            mz=float(props["mz"]),
            rt=float(props["rt"]) if props.get("rt") is not None else None,
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
