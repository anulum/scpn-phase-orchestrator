# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — UPDE subsystem

"""Lazy public facade for UPDE engines, diagnostics, and numeric helpers.

The UPDE package exposes phase-dynamics engines, reductions, prediction
models, stochastic injectors, stability utilities, and diagnostic dataclasses
without importing every optional backend at package import time. Concrete
modules own numeric validation, integration methods, Rust/JAX fallback behavior,
and mutation boundaries; this facade only resolves documented public symbols on
demand.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "BasinStabilityResult",
    "BayesianBackendStatus",
    "BayesianUPDEConfig",
    "BayesianUPDEResult",
    "BifurcationDiagram",
    "BifurcationPoint",
    "DelayBuffer",
    "DelayedEngine",
    "DopplerEngine",
    "EnvelopeState",
    "GaussianArrayDistribution",
    "GaussianUPDEPosteriorFit",
    "HAS_JAX",
    "Hyperedge",
    "HypergraphEngine",
    "InertialKuramotoEngine",
    "IntegrationConfig",
    "JaxUPDEEngine",
    "LayerState",
    "LockSignature",
    "MovingFrameState",
    "MovingFrameUPDEEngine",
    "NoiseProfile",
    "OAState",
    "PHA_C_HANDOFF_CLAIM_BOUNDARY",
    "PHA_C_HANDOFF_EVIDENCE_KIND",
    "PHA_C_HANDOFF_MARGIN_REPLAY_TOLERANCE",
    "PHA_C_ACCEPTANCE_CLAIM_BOUNDARY",
    "PHA_C_ACCEPTANCE_EVIDENCE_KIND",
    "PHA_C_TIMELINE_CLAIM_BOUNDARY",
    "PHA_C_TIMELINE_EVIDENCE_KIND",
    "PHA_C_FORMAL_CERTIFICATE_PREDICATE",
    "PHA_C_FORMAL_CERTIFICATE_THEOREM",
    "PHA_C_FORMAL_DEFAULT_SCALE_M",
    "PHA_C_FORMAL_DEFAULT_SCALE_RAD",
    "PHA_C_FORMAL_LEAN_MODULE",
    "PHA_C_FORMAL_OBLIGATION_CLAIM_BOUNDARY",
    "PHA_C_FORMAL_OBLIGATION_EVIDENCE_KIND",
    "PHA_C_FORMAL_OBLIGATION_SCHEMA",
    "PHACAcceptanceRecord",
    "PHACHandoffRecord",
    "PHACKinematicProofObligation",
    "PHACTimelineRecord",
    "OttAntonsenReduction",
    "PredictionModel",
    "PredictionState",
    "SheafUPDEEngine",
    "SimplicialEngine",
    "SparseUPDEEngine",
    "SplittingEngine",
    "StochasticInjector",
    "StuartLandauEngine",
    "SwarmalatorEngine",
    "TorusEngine",
    "UPDEEngine",
    "UPDEState",
    "VariationalPredictor",
    "VariationalState",
    "audit_bayesian_backend_status",
    "basin_stability",
    "bayesian_upde_run",
    "check_stability",
    "compute_layer_coherence",
    "compute_order_parameter",
    "compute_plv",
    "cost_R",
    "detect_regimes",
    "doppler_run",
    "doppler_term",
    "extract_phase",
    "find_critical_coupling",
    "find_optimal_noise",
    "fit_gaussian_upde_posterior",
    "gradient_knm_fd",
    "gradient_knm_jax",
    "market_order_parameter",
    "market_plv",
    "modulation_index",
    "moving_frame_run",
    "multi_basin_stability",
    "pac_gate",
    "pac_matrix",
    "pha_c_handoff_record_to_dict",
    "pha_c_acceptance_record_to_dict",
    "pha_c_kinematic_proof_obligation_to_dict",
    "pha_c_event_timeline_to_dict",
    "verify_pha_c_handoff_record",
    "verify_pha_c_acceptance_record",
    "verify_pha_c_kinematic_proof_obligation",
    "verify_pha_c_event_timeline",
    "sync_warning",
    "trace_sync_transition",
    "upde_run_omega_schedule",
    "build_pha_c_handoff_record",
    "build_pha_c_acceptance_record",
    "build_pha_c_kinematic_proof_obligation",
    "build_pha_c_event_timeline",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "cost_R": (".adjoint", "cost_R"),
    "gradient_knm_fd": (".adjoint", "gradient_knm_fd"),
    "gradient_knm_jax": (".adjoint", "gradient_knm_jax"),
    "BasinStabilityResult": (".basin_stability", "BasinStabilityResult"),
    "basin_stability": (".basin_stability", "basin_stability"),
    "multi_basin_stability": (".basin_stability", "multi_basin_stability"),
    "BayesianUPDEConfig": (".bayesian", "BayesianUPDEConfig"),
    "BayesianBackendStatus": (".bayesian", "BayesianBackendStatus"),
    "BayesianUPDEResult": (".bayesian", "BayesianUPDEResult"),
    "GaussianArrayDistribution": (".bayesian", "GaussianArrayDistribution"),
    "GaussianUPDEPosteriorFit": (".bayesian", "GaussianUPDEPosteriorFit"),
    "audit_bayesian_backend_status": (
        ".bayesian",
        "audit_bayesian_backend_status",
    ),
    "bayesian_upde_run": (".bayesian", "bayesian_upde_run"),
    "fit_gaussian_upde_posterior": (".bayesian", "fit_gaussian_upde_posterior"),
    "BifurcationDiagram": (".bifurcation", "BifurcationDiagram"),
    "BifurcationPoint": (".bifurcation", "BifurcationPoint"),
    "find_critical_coupling": (".bifurcation", "find_critical_coupling"),
    "trace_sync_transition": (".bifurcation", "trace_sync_transition"),
    "DelayBuffer": (".delay", "DelayBuffer"),
    "DelayedEngine": (".delay", "DelayedEngine"),
    "DopplerEngine": (".doppler", "DopplerEngine"),
    "doppler_run": (".doppler", "doppler_run"),
    "doppler_term": (".doppler", "doppler_term"),
    "MovingFrameState": (".moving_frame", "MovingFrameState"),
    "MovingFrameUPDEEngine": (".moving_frame", "MovingFrameUPDEEngine"),
    "moving_frame_run": (".moving_frame", "moving_frame_run"),
    "PHA_C_HANDOFF_CLAIM_BOUNDARY": (
        ".pha_c_handoff",
        "PHA_C_HANDOFF_CLAIM_BOUNDARY",
    ),
    "PHA_C_HANDOFF_EVIDENCE_KIND": (
        ".pha_c_handoff",
        "PHA_C_HANDOFF_EVIDENCE_KIND",
    ),
    "PHA_C_HANDOFF_MARGIN_REPLAY_TOLERANCE": (
        ".pha_c_handoff",
        "PHA_C_HANDOFF_MARGIN_REPLAY_TOLERANCE",
    ),
    "PHACHandoffRecord": (".pha_c_handoff", "PHACHandoffRecord"),
    "build_pha_c_handoff_record": (
        ".pha_c_handoff",
        "build_pha_c_handoff_record",
    ),
    "pha_c_handoff_record_to_dict": (
        ".pha_c_handoff",
        "pha_c_handoff_record_to_dict",
    ),
    "verify_pha_c_handoff_record": (
        ".pha_c_handoff",
        "verify_pha_c_handoff_record",
    ),
    "PHA_C_ACCEPTANCE_CLAIM_BOUNDARY": (
        ".pha_c_acceptance",
        "PHA_C_ACCEPTANCE_CLAIM_BOUNDARY",
    ),
    "PHA_C_ACCEPTANCE_EVIDENCE_KIND": (
        ".pha_c_acceptance",
        "PHA_C_ACCEPTANCE_EVIDENCE_KIND",
    ),
    "PHACAcceptanceRecord": (".pha_c_acceptance", "PHACAcceptanceRecord"),
    "build_pha_c_acceptance_record": (
        ".pha_c_acceptance",
        "build_pha_c_acceptance_record",
    ),
    "pha_c_acceptance_record_to_dict": (
        ".pha_c_acceptance",
        "pha_c_acceptance_record_to_dict",
    ),
    "verify_pha_c_acceptance_record": (
        ".pha_c_acceptance",
        "verify_pha_c_acceptance_record",
    ),
    "PHA_C_FORMAL_CERTIFICATE_PREDICATE": (
        ".pha_c_formal_obligation",
        "PHA_C_FORMAL_CERTIFICATE_PREDICATE",
    ),
    "PHA_C_FORMAL_CERTIFICATE_THEOREM": (
        ".pha_c_formal_obligation",
        "PHA_C_FORMAL_CERTIFICATE_THEOREM",
    ),
    "PHA_C_FORMAL_DEFAULT_SCALE_M": (
        ".pha_c_formal_obligation",
        "PHA_C_FORMAL_DEFAULT_SCALE_M",
    ),
    "PHA_C_FORMAL_DEFAULT_SCALE_RAD": (
        ".pha_c_formal_obligation",
        "PHA_C_FORMAL_DEFAULT_SCALE_RAD",
    ),
    "PHA_C_FORMAL_LEAN_MODULE": (
        ".pha_c_formal_obligation",
        "PHA_C_FORMAL_LEAN_MODULE",
    ),
    "PHA_C_FORMAL_OBLIGATION_CLAIM_BOUNDARY": (
        ".pha_c_formal_obligation",
        "PHA_C_FORMAL_OBLIGATION_CLAIM_BOUNDARY",
    ),
    "PHA_C_FORMAL_OBLIGATION_EVIDENCE_KIND": (
        ".pha_c_formal_obligation",
        "PHA_C_FORMAL_OBLIGATION_EVIDENCE_KIND",
    ),
    "PHA_C_FORMAL_OBLIGATION_SCHEMA": (
        ".pha_c_formal_obligation",
        "PHA_C_FORMAL_OBLIGATION_SCHEMA",
    ),
    "PHACKinematicProofObligation": (
        ".pha_c_formal_obligation",
        "PHACKinematicProofObligation",
    ),
    "build_pha_c_kinematic_proof_obligation": (
        ".pha_c_formal_obligation",
        "build_pha_c_kinematic_proof_obligation",
    ),
    "pha_c_kinematic_proof_obligation_to_dict": (
        ".pha_c_formal_obligation",
        "pha_c_kinematic_proof_obligation_to_dict",
    ),
    "verify_pha_c_kinematic_proof_obligation": (
        ".pha_c_formal_obligation",
        "verify_pha_c_kinematic_proof_obligation",
    ),
    "PHA_C_TIMELINE_CLAIM_BOUNDARY": (
        ".pha_c_timeline",
        "PHA_C_TIMELINE_CLAIM_BOUNDARY",
    ),
    "PHA_C_TIMELINE_EVIDENCE_KIND": (
        ".pha_c_timeline",
        "PHA_C_TIMELINE_EVIDENCE_KIND",
    ),
    "PHACTimelineRecord": (".pha_c_timeline", "PHACTimelineRecord"),
    "build_pha_c_event_timeline": (
        ".pha_c_timeline",
        "build_pha_c_event_timeline",
    ),
    "pha_c_event_timeline_to_dict": (
        ".pha_c_timeline",
        "pha_c_event_timeline_to_dict",
    ),
    "verify_pha_c_event_timeline": (
        ".pha_c_timeline",
        "verify_pha_c_event_timeline",
    ),
    "UPDEEngine": (".engine", "UPDEEngine"),
    "upde_run_omega_schedule": (".engine", "upde_run_omega_schedule"),
    "EnvelopeState": (".envelope", "EnvelopeState"),
    "TorusEngine": (".geometric", "TorusEngine"),
    "Hyperedge": (".hypergraph", "Hyperedge"),
    "HypergraphEngine": (".hypergraph", "HypergraphEngine"),
    "InertialKuramotoEngine": (".inertial", "InertialKuramotoEngine"),
    "HAS_JAX": (".jax_engine", "HAS_JAX"),
    "JaxUPDEEngine": (".jax_engine", "JaxUPDEEngine"),
    "detect_regimes": (".market", "detect_regimes"),
    "extract_phase": (".market", "extract_phase"),
    "market_order_parameter": (".market", "market_order_parameter"),
    "market_plv": (".market", "market_plv"),
    "sync_warning": (".market", "sync_warning"),
    "LayerState": (".metrics", "LayerState"),
    "LockSignature": (".metrics", "LockSignature"),
    "UPDEState": (".metrics", "UPDEState"),
    "IntegrationConfig": (".numerics", "IntegrationConfig"),
    "check_stability": (".numerics", "check_stability"),
    "compute_layer_coherence": (".order_params", "compute_layer_coherence"),
    "compute_order_parameter": (".order_params", "compute_order_parameter"),
    "compute_plv": (".order_params", "compute_plv"),
    "modulation_index": (".pac", "modulation_index"),
    "pac_gate": (".pac", "pac_gate"),
    "pac_matrix": (".pac", "pac_matrix"),
    "PredictionModel": (".prediction", "PredictionModel"),
    "PredictionState": (".prediction", "PredictionState"),
    "VariationalPredictor": (".prediction", "VariationalPredictor"),
    "VariationalState": (".prediction", "VariationalState"),
    "OAState": (".reduction", "OAState"),
    "OttAntonsenReduction": (".reduction", "OttAntonsenReduction"),
    "SheafUPDEEngine": (".sheaf_engine", "SheafUPDEEngine"),
    "SimplicialEngine": (".simplicial", "SimplicialEngine"),
    "SparseUPDEEngine": (".sparse_engine", "SparseUPDEEngine"),
    "SplittingEngine": (".splitting", "SplittingEngine"),
    "NoiseProfile": (".stochastic", "NoiseProfile"),
    "StochasticInjector": (".stochastic", "StochasticInjector"),
    "find_optimal_noise": (".stochastic", "find_optimal_noise"),
    "StuartLandauEngine": (".stuart_landau", "StuartLandauEngine"),
    "SwarmalatorEngine": (".swarmalator", "SwarmalatorEngine"),
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
