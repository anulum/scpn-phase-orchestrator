# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Public API typing facade

from typing import Any

from scpn_phase_orchestrator.api import Orchestrator as Orchestrator
from scpn_phase_orchestrator.api import OrchestratorState as OrchestratorState
from scpn_phase_orchestrator.artifacts.qpu_data import (
    QPUDataArtifact as QPUDataArtifact,
)
from scpn_phase_orchestrator.artifacts.qpu_data import (
    compile_domain_to_qpu_artifact as compile_domain_to_qpu_artifact,
)
from scpn_phase_orchestrator.artifacts.qpu_data import (
    emit_qpu_data_artifact as emit_qpu_data_artifact,
)
from scpn_phase_orchestrator.artifacts.qpu_data import (
    validate_qpu_data_artifact as validate_qpu_data_artifact,
)
from scpn_phase_orchestrator.binding.types import BindingSpec as BindingSpec
from scpn_phase_orchestrator.coupling.knm import CouplingBuilder as CouplingBuilder
from scpn_phase_orchestrator.exceptions import SPOError as SPOError
from scpn_phase_orchestrator.monitor.boundaries import (
    BoundaryObserver as BoundaryObserver,
)
from scpn_phase_orchestrator.monitor.lyapunov import (
    lyapunov_spectrum as lyapunov_spectrum,
)
from scpn_phase_orchestrator.oscillators.base import PhaseExtractor as PhaseExtractor
from scpn_phase_orchestrator.oscillators.base import PhaseState as PhaseState
from scpn_phase_orchestrator.reactor_semantics import (
    ObservableDescriptor as ObservableDescriptor,
)
from scpn_phase_orchestrator.reactor_semantics import PhaseRelation as PhaseRelation
from scpn_phase_orchestrator.reactor_semantics import (
    PhaseSemanticRecord as PhaseSemanticRecord,
)
from scpn_phase_orchestrator.reactor_semantics import ReactorContext as ReactorContext
from scpn_phase_orchestrator.reactor_semantics import RegimeEstimate as RegimeEstimate
from scpn_phase_orchestrator.runtime.audit_logger import AuditLogger as AuditLogger
from scpn_phase_orchestrator.supervisor import ControlAction as ControlAction
from scpn_phase_orchestrator.supervisor.policy import (
    SupervisorPolicy as SupervisorPolicy,
)
from scpn_phase_orchestrator.supervisor.regimes import RegimeManager as RegimeManager
from scpn_phase_orchestrator.upde.bifurcation import (
    BifurcationDiagram as BifurcationDiagram,
)
from scpn_phase_orchestrator.upde.bifurcation import (
    find_critical_coupling as find_critical_coupling,
)
from scpn_phase_orchestrator.upde.bifurcation import (
    trace_sync_transition as trace_sync_transition,
)
from scpn_phase_orchestrator.upde.engine import UPDEEngine as UPDEEngine
from scpn_phase_orchestrator.upde.sheaf_engine import (
    SheafUPDEEngine as SheafUPDEEngine,
)
from scpn_phase_orchestrator.upde.sparse_engine import (
    SparseUPDEEngine as SparseUPDEEngine,
)
from scpn_phase_orchestrator.upde.stuart_landau import (
    StuartLandauEngine as StuartLandauEngine,
)

__version__: str
__all__: list[str]

def __getattr__(name: str) -> Any: ...
def __dir__() -> list[str]: ...
