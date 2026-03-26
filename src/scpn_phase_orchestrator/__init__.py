# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Public API

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("scpn-phase-orchestrator")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

from scpn_phase_orchestrator.audit.logger import AuditLogger
from scpn_phase_orchestrator.binding.types import BindingSpec
from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
from scpn_phase_orchestrator.exceptions import SPOError
from scpn_phase_orchestrator.monitor.boundaries import BoundaryObserver
from scpn_phase_orchestrator.monitor.lyapunov import lyapunov_spectrum
from scpn_phase_orchestrator.oscillators.base import PhaseExtractor, PhaseState
from scpn_phase_orchestrator.supervisor import ControlAction
from scpn_phase_orchestrator.supervisor.policy import SupervisorPolicy
from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
from scpn_phase_orchestrator.upde.bifurcation import (
    BifurcationDiagram,
    find_critical_coupling,
    trace_sync_transition,
)
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

__all__ = [
    "AuditLogger",
    "BifurcationDiagram",
    "BindingSpec",
    "BoundaryObserver",
    "ControlAction",
    "CouplingBuilder",
    "PhaseExtractor",
    "PhaseState",
    "RegimeManager",
    "SPOError",
    "StuartLandauEngine",
    "SupervisorPolicy",
    "UPDEEngine",
    "find_critical_coupling",
    "lyapunov_spectrum",
    "trace_sync_transition",
]
