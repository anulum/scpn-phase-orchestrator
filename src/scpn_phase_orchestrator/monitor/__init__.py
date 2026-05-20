# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Monitor subsystem

"""Lazy public registry for safety, coherence, and diagnostics monitors.

The monitor package exposes boundary checks, coherence partitioning,
entrainment verification, information measures, recurrence and embedding
diagnostics, runtime verification, sleep staging, and related phase-analysis
helpers. Exports are intentionally lazy so optional backend or toolchain
dependencies do not make package import expensive; numeric validation and
fail-closed policies remain in the owning monitor modules.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "BoundaryObserver",
    "ChimeraState",
    "CoherenceMonitor",
    "CorrelationDimensionResult",
    "EVSMonitor",
    "EVSResult",
    "EmbeddingResult",
    "HAS_RTAMT",
    "IntegratedInformationBenchmarkCase",
    "IntegratedInformationBenchmarkReport",
    "IntegratedInformationResult",
    "LyapunovGuard",
    "LyapunovState",
    "PoincareResult",
    "RQAResult",
    "STLActionProjectionTemplate",
    "STLAutomatonState",
    "STLAutomatonTransition",
    "STLControllerCandidate",
    "STLControllerSynthesis",
    "STLMonitor",
    "STLMonitoringAutomaton",
    "STLProjectedActionPlan",
    "STLTraceResult",
    "SessionCoherenceReport",
    "auto_embed",
    "check_session_start",
    "classify_sleep_stage",
    "compute_itpc",
    "compute_npe",
    "correlation_dimension",
    "correlation_integral",
    "cross_recurrence_matrix",
    "cross_rqa",
    "delay_embed",
    "detect_chimera",
    "entropy_from_phases",
    "entropy_production_rate",
    "itpc_persistence",
    "benchmark_integrated_information_approximations",
    "build_cyber_industrial_integrated_information_replays",
    "build_infrastructure_integrated_information_replays",
    "build_physiology_integrated_information_replays",
    "integrated_information",
    "kaplan_yorke_dimension",
    "lyapunov_spectrum",
    "optimal_delay",
    "optimal_dimension",
    "phase_distance_matrix",
    "phase_poincare",
    "phase_transfer_entropy",
    "project_stl_controller_candidates",
    "poincare_section",
    "recurrence_matrix",
    "reduce_coupling",
    "redundancy",
    "return_times",
    "rqa",
    "simulate_psychedelic_trajectory",
    "synergy",
    "synthesise_stl_monitoring_automaton",
    "synthesise_stl_controller_candidates",
    "synthesize_stl_monitoring_automaton",
    "synthesize_stl_controller_candidates",
    "transfer_entropy_matrix",
    "ultradian_phase",
    "winding_numbers",
    "winding_vector",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "BoundaryObserver": (".boundaries", "BoundaryObserver"),
    "ChimeraState": (".chimera", "ChimeraState"),
    "detect_chimera": (".chimera", "detect_chimera"),
    "CoherenceMonitor": (".coherence", "CoherenceMonitor"),
    "CorrelationDimensionResult": (".dimension", "CorrelationDimensionResult"),
    "correlation_dimension": (".dimension", "correlation_dimension"),
    "correlation_integral": (".dimension", "correlation_integral"),
    "kaplan_yorke_dimension": (".dimension", "kaplan_yorke_dimension"),
    "EmbeddingResult": (".embedding", "EmbeddingResult"),
    "auto_embed": (".embedding", "auto_embed"),
    "delay_embed": (".embedding", "delay_embed"),
    "optimal_delay": (".embedding", "optimal_delay"),
    "optimal_dimension": (".embedding", "optimal_dimension"),
    "entropy_production_rate": (".entropy_prod", "entropy_production_rate"),
    "EVSMonitor": (".evs", "EVSMonitor"),
    "EVSResult": (".evs", "EVSResult"),
    "compute_itpc": (".itpc", "compute_itpc"),
    "itpc_persistence": (".itpc", "itpc_persistence"),
    "IntegratedInformationResult": (
        ".information_integration",
        "IntegratedInformationResult",
    ),
    "IntegratedInformationBenchmarkCase": (
        ".information_integration",
        "IntegratedInformationBenchmarkCase",
    ),
    "IntegratedInformationBenchmarkReport": (
        ".information_integration",
        "IntegratedInformationBenchmarkReport",
    ),
    "benchmark_integrated_information_approximations": (
        ".information_integration",
        "benchmark_integrated_information_approximations",
    ),
    "build_cyber_industrial_integrated_information_replays": (
        ".information_replay_cyber_industrial",
        "build_cyber_industrial_integrated_information_replays",
    ),
    "build_infrastructure_integrated_information_replays": (
        ".information_replay_infrastructure",
        "build_infrastructure_integrated_information_replays",
    ),
    "build_physiology_integrated_information_replays": (
        ".information_replay_physiology",
        "build_physiology_integrated_information_replays",
    ),
    "integrated_information": (
        ".information_integration",
        "integrated_information",
    ),
    "LyapunovGuard": (".lyapunov", "LyapunovGuard"),
    "LyapunovState": (".lyapunov", "LyapunovState"),
    "lyapunov_spectrum": (".lyapunov", "lyapunov_spectrum"),
    "compute_npe": (".npe", "compute_npe"),
    "phase_distance_matrix": (".npe", "phase_distance_matrix"),
    "redundancy": (".pid", "redundancy"),
    "synergy": (".pid", "synergy"),
    "PoincareResult": (".poincare", "PoincareResult"),
    "phase_poincare": (".poincare", "phase_poincare"),
    "poincare_section": (".poincare", "poincare_section"),
    "return_times": (".poincare", "return_times"),
    "entropy_from_phases": (".psychedelic", "entropy_from_phases"),
    "reduce_coupling": (".psychedelic", "reduce_coupling"),
    "simulate_psychedelic_trajectory": (
        ".psychedelic",
        "simulate_psychedelic_trajectory",
    ),
    "RQAResult": (".recurrence", "RQAResult"),
    "cross_recurrence_matrix": (".recurrence", "cross_recurrence_matrix"),
    "cross_rqa": (".recurrence", "cross_rqa"),
    "recurrence_matrix": (".recurrence", "recurrence_matrix"),
    "rqa": (".recurrence", "rqa"),
    "SessionCoherenceReport": (".session_start", "SessionCoherenceReport"),
    "check_session_start": (".session_start", "check_session_start"),
    "classify_sleep_stage": (".sleep_staging", "classify_sleep_stage"),
    "ultradian_phase": (".sleep_staging", "ultradian_phase"),
    "HAS_RTAMT": (".stl", "HAS_RTAMT"),
    "STLActionProjectionTemplate": (".stl", "STLActionProjectionTemplate"),
    "STLAutomatonState": (".stl", "STLAutomatonState"),
    "STLAutomatonTransition": (".stl", "STLAutomatonTransition"),
    "STLControllerCandidate": (".stl", "STLControllerCandidate"),
    "STLControllerSynthesis": (".stl", "STLControllerSynthesis"),
    "STLMonitor": (".stl", "STLMonitor"),
    "STLMonitoringAutomaton": (".stl", "STLMonitoringAutomaton"),
    "STLProjectedActionPlan": (".stl", "STLProjectedActionPlan"),
    "STLTraceResult": (".stl", "STLTraceResult"),
    "project_stl_controller_candidates": (
        ".stl",
        "project_stl_controller_candidates",
    ),
    "synthesise_stl_monitoring_automaton": (
        ".stl",
        "synthesise_stl_monitoring_automaton",
    ),
    "synthesise_stl_controller_candidates": (
        ".stl",
        "synthesise_stl_controller_candidates",
    ),
    "synthesize_stl_monitoring_automaton": (
        ".stl",
        "synthesize_stl_monitoring_automaton",
    ),
    "synthesize_stl_controller_candidates": (
        ".stl",
        "synthesize_stl_controller_candidates",
    ),
    "phase_transfer_entropy": (".transfer_entropy", "phase_transfer_entropy"),
    "transfer_entropy_matrix": (".transfer_entropy", "transfer_entropy_matrix"),
    "winding_numbers": (".winding", "winding_numbers"),
    "winding_vector": (".winding", "winding_vector"),
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
