# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — UPDE subsystem

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "BasinStabilityResult",
    "BifurcationDiagram",
    "BifurcationPoint",
    "DelayBuffer",
    "DelayedEngine",
    "EnvelopeState",
    "HAS_JAX",
    "Hyperedge",
    "HypergraphEngine",
    "InertialKuramotoEngine",
    "IntegrationConfig",
    "JaxUPDEEngine",
    "LayerState",
    "LockSignature",
    "NoiseProfile",
    "OAState",
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
    "basin_stability",
    "check_stability",
    "compute_layer_coherence",
    "compute_order_parameter",
    "compute_plv",
    "cost_R",
    "detect_regimes",
    "extract_phase",
    "find_critical_coupling",
    "find_optimal_noise",
    "gradient_knm_fd",
    "gradient_knm_jax",
    "market_order_parameter",
    "market_plv",
    "modulation_index",
    "multi_basin_stability",
    "pac_gate",
    "pac_matrix",
    "sync_warning",
    "trace_sync_transition",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "cost_R": (".adjoint", "cost_R"),
    "gradient_knm_fd": (".adjoint", "gradient_knm_fd"),
    "gradient_knm_jax": (".adjoint", "gradient_knm_jax"),
    "BasinStabilityResult": (".basin_stability", "BasinStabilityResult"),
    "basin_stability": (".basin_stability", "basin_stability"),
    "multi_basin_stability": (".basin_stability", "multi_basin_stability"),
    "BifurcationDiagram": (".bifurcation", "BifurcationDiagram"),
    "BifurcationPoint": (".bifurcation", "BifurcationPoint"),
    "find_critical_coupling": (".bifurcation", "find_critical_coupling"),
    "trace_sync_transition": (".bifurcation", "trace_sync_transition"),
    "DelayBuffer": (".delay", "DelayBuffer"),
    "DelayedEngine": (".delay", "DelayedEngine"),
    "UPDEEngine": (".engine", "UPDEEngine"),
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
