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
from scpn_phase_orchestrator.coupling.knm import (
    SCPN_CALIBRATION_ANCHORS,
    SCPN_LAYER_NAMES,
    SCPN_LAYER_TIMESCALES,
    CouplingBuilder,
    CouplingState,
)
from scpn_phase_orchestrator.coupling.plasticity import (
    compute_eligibility,
    three_factor_update,
)

__all__ = [
    "CouplingBuilder",
    "CouplingState",
    "SCPN_CALIBRATION_ANCHORS",
    "SCPN_LAYER_NAMES",
    "SCPN_LAYER_TIMESCALES",
    "compute_eligibility",
    "load_hcp_connectome",
    "load_neurolib_hcp",
    "three_factor_update",
]
