# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Runtime STL monitor

"""Signal Temporal Logic monitor, synthesis, and runtime actuation gating.

rtamt is an optional dependency: ``pip install rtamt``. The implementation is
split into responsibility modules (monitor, automaton synthesis, controller
synthesis, action projection, runtime actuation gate, and closed-loop plan)
behind a stable re-export surface; ``HAS_RTAMT`` reports rtamt availability.
"""

from __future__ import annotations

from .actuation_gate import (
    STLRuntimeActuationGate,
    validate_stl_runtime_actuation_gate,
)
from .automaton import (
    STLAutomatonState,
    STLAutomatonTransition,
    STLMonitoringAutomaton,
    synthesise_stl_monitoring_automaton,
    synthesize_stl_monitoring_automaton,
)
from .closed_loop import (
    STLClosedLoopSynthesisPlan,
    synthesise_stl_closed_loop_plan,
    synthesize_stl_closed_loop_plan,
)
from .controller import (
    STLControllerCandidate,
    STLControllerSynthesis,
    synthesise_stl_controller_candidates,
    synthesize_stl_controller_candidates,
)
from .monitor import (
    HAS_RTAMT,
    STLMonitor,
    STLTraceResult,
)
from .monitor import (
    _predicate_robustness as _predicate_robustness,
)
from .projection import (
    STLActionProjectionTemplate,
    STLProjectedActionPlan,
    project_stl_controller_candidates,
)

__all__ = [
    "HAS_RTAMT",
    "STLActionProjectionTemplate",
    "STLAutomatonState",
    "STLAutomatonTransition",
    "STLControllerCandidate",
    "STLControllerSynthesis",
    "STLClosedLoopSynthesisPlan",
    "STLMonitor",
    "STLMonitoringAutomaton",
    "STLProjectedActionPlan",
    "STLRuntimeActuationGate",
    "STLTraceResult",
    "project_stl_controller_candidates",
    "synthesise_stl_closed_loop_plan",
    "synthesise_stl_monitoring_automaton",
    "synthesise_stl_controller_candidates",
    "synthesize_stl_closed_loop_plan",
    "synthesize_stl_monitoring_automaton",
    "synthesize_stl_controller_candidates",
    "validate_stl_runtime_actuation_gate",
]
