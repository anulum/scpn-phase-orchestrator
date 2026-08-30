# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Public API

"""Stable public import surface for SCPN Phase Orchestrator.

The package root intentionally exports the compatibility-reviewed API listed
in ``docs/specs/public_api_manifest.txt``. Subpackages expose the wider
research and operator surfaces, but additions to ``__all__`` are release
managed because downstream code imports these names directly.
"""

from __future__ import annotations

__version__ = "1.2.0"

import os as _os
from importlib import import_module as _import_module
from typing import Any as _Any

# juliacall 0.9.34's init() references an undefined ``Base`` in its
# multithreaded-warning branch when ``PYTHON_JULIACALL_HANDLE_SIGNALS`` is unset
# and the host process is multithreaded (for example under coverage's thread
# tracer), raising NameError and aborting the optional Julia backend probe.
# The upstream-recommended value skips that branch; the guard keeps any
# operator-provided override intact. Must run before the first submodule import
# that may load juliacall.
if "PYTHON_JULIACALL_HANDLE_SIGNALS" not in _os.environ:
    _os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AuditLogger": ("scpn_phase_orchestrator.runtime.audit_logger", "AuditLogger"),
    "BifurcationDiagram": (
        "scpn_phase_orchestrator.upde.bifurcation",
        "BifurcationDiagram",
    ),
    "BindingSpec": ("scpn_phase_orchestrator.binding.types", "BindingSpec"),
    "BoundaryObserver": (
        "scpn_phase_orchestrator.monitor.boundaries",
        "BoundaryObserver",
    ),
    "ControlAction": ("scpn_phase_orchestrator.supervisor", "ControlAction"),
    "CouplingBuilder": ("scpn_phase_orchestrator.coupling.knm", "CouplingBuilder"),
    "ObservableDescriptor": (
        "scpn_phase_orchestrator.reactor_semantics",
        "ObservableDescriptor",
    ),
    "Orchestrator": ("scpn_phase_orchestrator.api", "Orchestrator"),
    "OrchestratorState": ("scpn_phase_orchestrator.api", "OrchestratorState"),
    "PhaseExtractor": ("scpn_phase_orchestrator.oscillators.base", "PhaseExtractor"),
    "PhaseRelation": (
        "scpn_phase_orchestrator.reactor_semantics",
        "PhaseRelation",
    ),
    "PhaseSemanticRecord": (
        "scpn_phase_orchestrator.reactor_semantics",
        "PhaseSemanticRecord",
    ),
    "PhaseState": ("scpn_phase_orchestrator.oscillators.base", "PhaseState"),
    "QPUDataArtifact": (
        "scpn_phase_orchestrator.artifacts.qpu_data",
        "QPUDataArtifact",
    ),
    "ReactorContext": (
        "scpn_phase_orchestrator.reactor_semantics",
        "ReactorContext",
    ),
    "RegimeEstimate": (
        "scpn_phase_orchestrator.reactor_semantics",
        "RegimeEstimate",
    ),
    "RegimeManager": ("scpn_phase_orchestrator.supervisor.regimes", "RegimeManager"),
    "SPOError": ("scpn_phase_orchestrator.exceptions", "SPOError"),
    "SheafUPDEEngine": (
        "scpn_phase_orchestrator.upde.sheaf_engine",
        "SheafUPDEEngine",
    ),
    "SparseUPDEEngine": (
        "scpn_phase_orchestrator.upde.sparse_engine",
        "SparseUPDEEngine",
    ),
    "StuartLandauEngine": (
        "scpn_phase_orchestrator.upde.stuart_landau",
        "StuartLandauEngine",
    ),
    "SupervisorPolicy": (
        "scpn_phase_orchestrator.supervisor.policy",
        "SupervisorPolicy",
    ),
    "UPDEEngine": ("scpn_phase_orchestrator.upde.engine", "UPDEEngine"),
    "compile_domain_to_qpu_artifact": (
        "scpn_phase_orchestrator.artifacts.qpu_data",
        "compile_domain_to_qpu_artifact",
    ),
    "emit_qpu_data_artifact": (
        "scpn_phase_orchestrator.artifacts.qpu_data",
        "emit_qpu_data_artifact",
    ),
    "find_critical_coupling": (
        "scpn_phase_orchestrator.upde.bifurcation",
        "find_critical_coupling",
    ),
    "lyapunov_spectrum": (
        "scpn_phase_orchestrator.monitor.lyapunov",
        "lyapunov_spectrum",
    ),
    "trace_sync_transition": (
        "scpn_phase_orchestrator.upde.bifurcation",
        "trace_sync_transition",
    ),
    "validate_qpu_data_artifact": (
        "scpn_phase_orchestrator.artifacts.qpu_data",
        "validate_qpu_data_artifact",
    ),
}

def __getattr__(name: str) -> _Any:
    """Resolve one compatibility export only when it is explicitly requested."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(_import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "AuditLogger",
    "BifurcationDiagram",
    "BindingSpec",
    "BoundaryObserver",
    "ControlAction",
    "CouplingBuilder",
    "PhaseExtractor",
    "PhaseState",
    "Orchestrator",
    "OrchestratorState",
    "ObservableDescriptor",
    "PhaseRelation",
    "PhaseSemanticRecord",
    "QPUDataArtifact",
    "ReactorContext",
    "RegimeEstimate",
    "RegimeManager",
    "SPOError",
    "SparseUPDEEngine",
    "SheafUPDEEngine",
    "StuartLandauEngine",
    "SupervisorPolicy",
    "UPDEEngine",
    "compile_domain_to_qpu_artifact",
    "emit_qpu_data_artifact",
    "find_critical_coupling",
    "lyapunov_spectrum",
    "trace_sync_transition",
    "validate_qpu_data_artifact",
]


def __dir__() -> list[str]:
    """List compatibility exports without resolving their owning modules."""
    return sorted(set(globals()) | set(__all__))
