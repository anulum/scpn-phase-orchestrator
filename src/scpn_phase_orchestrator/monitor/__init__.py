# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Monitor subsystem

from __future__ import annotations

from scpn_phase_orchestrator.monitor.boundaries import BoundaryObserver
from scpn_phase_orchestrator.monitor.chimera import ChimeraState, detect_chimera
from scpn_phase_orchestrator.monitor.coherence import CoherenceMonitor
from scpn_phase_orchestrator.monitor.dimension import (
    CorrelationDimensionResult,
    correlation_dimension,
    correlation_integral,
    kaplan_yorke_dimension,
)
from scpn_phase_orchestrator.monitor.embedding import (
    EmbeddingResult,
    auto_embed,
    delay_embed,
    optimal_delay,
    optimal_dimension,
)
from scpn_phase_orchestrator.monitor.evs import EVSMonitor, EVSResult
from scpn_phase_orchestrator.monitor.itpc import compute_itpc, itpc_persistence
from scpn_phase_orchestrator.monitor.lyapunov import (
    LyapunovGuard,
    LyapunovState,
    lyapunov_spectrum,
)
from scpn_phase_orchestrator.monitor.poincare import (
    PoincareResult,
    phase_poincare,
    poincare_section,
    return_times,
)
from scpn_phase_orchestrator.monitor.psychedelic import (
    entropy_from_phases,
    reduce_coupling,
    simulate_psychedelic_trajectory,
)
from scpn_phase_orchestrator.monitor.recurrence import (
    RQAResult,
    cross_recurrence_matrix,
    cross_rqa,
    recurrence_matrix,
    rqa,
)
from scpn_phase_orchestrator.monitor.session_start import (
    SessionCoherenceReport,
    check_session_start,
)
from scpn_phase_orchestrator.monitor.sleep_staging import (
    classify_sleep_stage,
    ultradian_phase,
)

__all__ = [
    "BoundaryObserver",
    "ChimeraState",
    "CoherenceMonitor",
    "CorrelationDimensionResult",
    "EVSMonitor",
    "correlation_dimension",
    "correlation_integral",
    "EVSResult",
    "EmbeddingResult",
    "LyapunovGuard",
    "PoincareResult",
    "LyapunovState",
    "RQAResult",
    "SessionCoherenceReport",
    "auto_embed",
    "check_session_start",
    "classify_sleep_stage",
    "compute_itpc",
    "cross_recurrence_matrix",
    "cross_rqa",
    "delay_embed",
    "detect_chimera",
    "entropy_from_phases",
    "itpc_persistence",
    "kaplan_yorke_dimension",
    "lyapunov_spectrum",
    "optimal_delay",
    "phase_poincare",
    "poincare_section",
    "optimal_dimension",
    "recurrence_matrix",
    "reduce_coupling",
    "return_times",
    "rqa",
    "simulate_psychedelic_trajectory",
    "ultradian_phase",
]
