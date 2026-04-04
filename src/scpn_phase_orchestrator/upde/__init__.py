# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — UPDE subsystem

from __future__ import annotations

from scpn_phase_orchestrator.upde.adjoint import (
    cost_R,
    gradient_knm_fd,
    gradient_knm_jax,
)
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
from scpn_phase_orchestrator.upde.delay import DelayBuffer, DelayedEngine
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.envelope import EnvelopeState
from scpn_phase_orchestrator.upde.geometric import TorusEngine
from scpn_phase_orchestrator.upde.hypergraph import Hyperedge, HypergraphEngine
from scpn_phase_orchestrator.upde.inertial import InertialKuramotoEngine
from scpn_phase_orchestrator.upde.jax_engine import HAS_JAX, JaxUPDEEngine
from scpn_phase_orchestrator.upde.market import (
    detect_regimes,
    extract_phase,
    market_order_parameter,
    market_plv,
    sync_warning,
)
from scpn_phase_orchestrator.upde.metrics import LayerState, LockSignature, UPDEState
from scpn_phase_orchestrator.upde.numerics import IntegrationConfig, check_stability
from scpn_phase_orchestrator.upde.order_params import (
    compute_layer_coherence,
    compute_order_parameter,
    compute_plv,
)
from scpn_phase_orchestrator.upde.pac import modulation_index, pac_gate, pac_matrix
from scpn_phase_orchestrator.upde.prediction import (
    PredictionModel,
    PredictionState,
    VariationalPredictor,
    VariationalState,
)
from scpn_phase_orchestrator.upde.reduction import OAState, OttAntonsenReduction
from scpn_phase_orchestrator.upde.sheaf_engine import SheafUPDEEngine
from scpn_phase_orchestrator.upde.simplicial import SimplicialEngine
from scpn_phase_orchestrator.upde.sparse_engine import SparseUPDEEngine
from scpn_phase_orchestrator.upde.splitting import SplittingEngine
from scpn_phase_orchestrator.upde.stochastic import (
    NoiseProfile,
    StochasticInjector,
    find_optimal_noise,
)
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine
from scpn_phase_orchestrator.upde.swarmalator import SwarmalatorEngine

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
