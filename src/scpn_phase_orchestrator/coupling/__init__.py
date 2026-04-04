# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coupling subsystem

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "CouplingBuilder",
    "CouplingPrior",
    "CouplingState",
    "EIBalance",
    "GeometryConstraint",
    "HodgeResult",
    "KnmTemplate",
    "KnmTemplateSet",
    "LagModel",
    "NonNegativeConstraint",
    "SCPN_CALIBRATION_ANCHORS",
    "SCPN_LAYER_NAMES",
    "SCPN_LAYER_TIMESCALES",
    "SymmetryConstraint",
    "UniversalPrior",
    "adjust_ei_ratio",
    "compute_ei_balance",
    "compute_eligibility",
    "critical_coupling",
    "fiedler_partition",
    "fiedler_value",
    "fiedler_vector",
    "graph_laplacian",
    "hodge_decomposition",
    "load_hcp_connectome",
    "load_neurolib_hcp",
    "project_knm",
    "spectral_gap",
    "sync_convergence_rate",
    "te_adapt_coupling",
    "three_factor_update",
    "validate_knm",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "load_hcp_connectome": (".connectome", "load_hcp_connectome"),
    "load_neurolib_hcp": (".connectome", "load_neurolib_hcp"),
    "EIBalance": (".ei_balance", "EIBalance"),
    "adjust_ei_ratio": (".ei_balance", "adjust_ei_ratio"),
    "compute_ei_balance": (".ei_balance", "compute_ei_balance"),
    "GeometryConstraint": (".geometry_constraints", "GeometryConstraint"),
    "NonNegativeConstraint": (".geometry_constraints", "NonNegativeConstraint"),
    "SymmetryConstraint": (".geometry_constraints", "SymmetryConstraint"),
    "project_knm": (".geometry_constraints", "project_knm"),
    "validate_knm": (".geometry_constraints", "validate_knm"),
    "HodgeResult": (".hodge", "HodgeResult"),
    "hodge_decomposition": (".hodge", "hodge_decomposition"),
    "CouplingBuilder": (".knm", "CouplingBuilder"),
    "CouplingState": (".knm", "CouplingState"),
    "SCPN_CALIBRATION_ANCHORS": (".knm", "SCPN_CALIBRATION_ANCHORS"),
    "SCPN_LAYER_NAMES": (".knm", "SCPN_LAYER_NAMES"),
    "SCPN_LAYER_TIMESCALES": (".knm", "SCPN_LAYER_TIMESCALES"),
    "LagModel": (".lags", "LagModel"),
    "compute_eligibility": (".plasticity", "compute_eligibility"),
    "three_factor_update": (".plasticity", "three_factor_update"),
    "CouplingPrior": (".prior", "CouplingPrior"),
    "UniversalPrior": (".prior", "UniversalPrior"),
    "critical_coupling": (".spectral", "critical_coupling"),
    "fiedler_partition": (".spectral", "fiedler_partition"),
    "fiedler_value": (".spectral", "fiedler_value"),
    "fiedler_vector": (".spectral", "fiedler_vector"),
    "graph_laplacian": (".spectral", "graph_laplacian"),
    "spectral_gap": (".spectral", "spectral_gap"),
    "sync_convergence_rate": (".spectral", "sync_convergence_rate"),
    "te_adapt_coupling": (".te_adaptive", "te_adapt_coupling"),
    "KnmTemplate": (".templates", "KnmTemplate"),
    "KnmTemplateSet": (".templates", "KnmTemplateSet"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr_name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return __all__
