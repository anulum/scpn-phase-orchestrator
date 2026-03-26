# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — UPDE subsystem

from __future__ import annotations

from scpn_phase_orchestrator.upde.basin_stability import (
    BasinStabilityResult,
    basin_stability,
    multi_basin_stability,
)
from scpn_phase_orchestrator.upde.bifurcation import (
    BifurcationDiagram,
    BifurcationPoint,
    find_critical_coupling,
    trace_sync_transition,
)
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.envelope import EnvelopeState
from scpn_phase_orchestrator.upde.metrics import UPDEState
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

__all__ = [
    "BasinStabilityResult",
    "BifurcationDiagram",
    "BifurcationPoint",
    "EnvelopeState",
    "basin_stability",
    "StuartLandauEngine",
    "UPDEEngine",
    "UPDEState",
    "find_critical_coupling",
    "multi_basin_stability",
    "trace_sync_transition",
]
