# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coupling subsystem

from __future__ import annotations

from scpn_phase_orchestrator.coupling.connectome import (
    load_hcp_connectome,
    load_neurolib_hcp,
)
from scpn_phase_orchestrator.coupling.ei_balance import (
    EIBalance,
    adjust_ei_ratio,
    compute_ei_balance,
)
from scpn_phase_orchestrator.coupling.geometry_constraints import (
    GeometryConstraint,
    NonNegativeConstraint,
    SymmetryConstraint,
    project_knm,
    validate_knm,
)
from scpn_phase_orchestrator.coupling.hodge import HodgeResult, hodge_decomposition
from scpn_phase_orchestrator.coupling.knm import (
    SCPN_CALIBRATION_ANCHORS,
    SCPN_LAYER_NAMES,
    SCPN_LAYER_TIMESCALES,
    CouplingBuilder,
    CouplingState,
)
from scpn_phase_orchestrator.coupling.lags import LagModel
from scpn_phase_orchestrator.coupling.plasticity import (
    compute_eligibility,
    three_factor_update,
)
from scpn_phase_orchestrator.coupling.prior import CouplingPrior, UniversalPrior
from scpn_phase_orchestrator.coupling.spectral import (
    critical_coupling,
    fiedler_partition,
    fiedler_value,
    fiedler_vector,
    graph_laplacian,
    spectral_gap,
    sync_convergence_rate,
)
from scpn_phase_orchestrator.coupling.te_adaptive import te_adapt_coupling
from scpn_phase_orchestrator.coupling.templates import KnmTemplate, KnmTemplateSet

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
