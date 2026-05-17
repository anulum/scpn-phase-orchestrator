# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Differentiable neural network module

"""GPU-first differentiable phase dynamics for neural network integration.

Requires: ``pip install scpn-phase-orchestrator[nn]`` (installs jax + equinox).
Production training code should call ``require_accelerator()`` during start-up
to fail fast when JAX is present but only a CPU backend is visible.

Runtime API:
    HAS_JAX, JaxRuntimeInfo, jax_runtime_info, require_jax,
    require_accelerator, default_device

Functional API (jax only):
    kuramoto_step, kuramoto_rk4_step, kuramoto_forward,
    simplicial_step, simplicial_rk4_step, simplicial_forward,
    stuart_landau_step, stuart_landau_rk4_step, stuart_landau_forward,
    order_parameter, plv

Layer API (jax + equinox):
    KuramotoLayer — phase-only, learnable K and omegas
    StuartLandauLayer — phase + amplitude, learnable K, K_r, omegas, mu

All imports are lazy: ``import scpn_phase_orchestrator.nn`` succeeds without
JAX installed.  Symbols are resolved on first attribute access.
"""

from __future__ import annotations

from .runtime import (
    HAS_JAX,
    JaxRuntimeInfo,
    default_device,
    jax_runtime_info,
    require_accelerator,
    require_jax,
)

__all__ = [
    "HAS_JAX",
    "JaxRuntimeInfo",
    "jax_runtime_info",
    "require_jax",
    "require_accelerator",
    "default_device",
    "KuramotoLayer",
    "SimplicialKuramotoLayer",
    "StuartLandauLayer",
    "UDEKuramotoLayer",
    "CouplingResidual",
    "kuramoto_step",
    "kuramoto_rk4_step",
    "kuramoto_forward",
    "kuramoto_step_masked",
    "kuramoto_rk4_step_masked",
    "kuramoto_forward_masked",
    "simplicial_step",
    "simplicial_rk4_step",
    "simplicial_forward",
    "stuart_landau_step",
    "stuart_landau_rk4_step",
    "stuart_landau_forward",
    "order_parameter",
    "plv",
    "coupling_laplacian",
    "saf_order_parameter",
    "saf_loss",
    "analytical_inverse",
    "hybrid_inverse",
    "infer_coupling",
    "inverse_loss",
    "coupling_correlation",
    "oim_step",
    "oim_forward",
    "oim_solve",
    "extract_coloring",
    "extract_coloring_soft",
    "coloring_violations",
    "coloring_energy",
    "bold_from_neural",
    "bold_signal",
    "reservoir_drive",
    "reservoir_features",
    "reservoir_predict",
    "ridge_readout",
    "sync_loss",
    "trajectory_loss",
    "coupling_sparsity_loss",
    "train_step",
    "train",
    "generate_kuramoto_data",
    "generate_chimera_data",
    "winfree_step",
    "winfree_rk4_step",
    "winfree_forward",
    "theta_neuron_step",
    "theta_neuron_rk4_step",
    "theta_neuron_forward",
    "ThetaNeuronLayer",
    "laplacian_spectrum",
    "algebraic_connectivity",
    "eigenratio",
    "sync_threshold",
    "local_order_parameter",
    "chimera_index",
    "detect_chimera",
    "DifferentiableSupervisorConfig",
    "DifferentiableSupervisorPolicy",
    "SupervisorAction",
    "SupervisorActionProjection",
    "KuramotoSupervisorScenario",
    "SupervisorLossAux",
    "SupervisorPPOBatch",
    "SupervisorPPOCorpusRollout",
    "SupervisorPPOAux",
    "SupervisorPPOCheckpoint",
    "SupervisorPPOTrainResult",
    "SupervisorReplayProposal",
    "SupervisorScenarioCorpus",
    "masked_order_parameter",
    "apply_supervisor_action",
    "closed_loop_supervisor_loss",
    "supervisor_train_step",
    "pack_supervisor_action",
    "unpack_supervisor_action",
    "sample_supervisor_action",
    "supervisor_action_bound_penalty",
    "supervisor_action_log_prob",
    "project_supervisor_action_for_audit",
    "build_supervisor_replay_proposal",
    "ppo_supervisor_loss",
    "ppo_supervisor_train_step",
    "ppo_supervisor_train_epochs",
    "ppo_supervisor_train_with_checkpoint",
    "collect_supervisor_corpus_rollouts",
    "collect_supervisor_rollouts",
    "build_supervisor_scenario_corpus",
    "save_supervisor_ppo_checkpoint",
    "load_supervisor_ppo_checkpoint",
    "control_actions_from_supervisor",
]

_FUNCTIONAL = {
    "kuramoto_step",
    "kuramoto_rk4_step",
    "kuramoto_forward",
    "kuramoto_step_masked",
    "kuramoto_rk4_step_masked",
    "kuramoto_forward_masked",
    "simplicial_step",
    "simplicial_rk4_step",
    "simplicial_forward",
    "stuart_landau_step",
    "stuart_landau_rk4_step",
    "stuart_landau_forward",
    "order_parameter",
    "plv",
    "coupling_laplacian",
    "saf_order_parameter",
    "saf_loss",
    "winfree_step",
    "winfree_rk4_step",
    "winfree_forward",
}
_BOLD = {"bold_from_neural", "bold_signal"}
_INVERSE = {
    "analytical_inverse",
    "hybrid_inverse",
    "infer_coupling",
    "inverse_loss",
    "coupling_correlation",
}
_OIM = {
    "oim_step",
    "oim_forward",
    "oim_solve",
    "extract_coloring",
    "extract_coloring_soft",
    "coloring_violations",
    "coloring_energy",
}
_RESERVOIR = {
    "reservoir_drive",
    "reservoir_features",
    "reservoir_predict",
    "ridge_readout",
}
_LAYERS = {"KuramotoLayer", "SimplicialKuramotoLayer", "StuartLandauLayer"}
_UDE = {"UDEKuramotoLayer", "CouplingResidual"}
_TRAINING = {
    "sync_loss",
    "trajectory_loss",
    "coupling_sparsity_loss",
    "train_step",
    "train",
    "generate_kuramoto_data",
    "generate_chimera_data",
}
_THETA = {
    "theta_neuron_step",
    "theta_neuron_rk4_step",
    "theta_neuron_forward",
    "ThetaNeuronLayer",
}
_SPECTRAL = {
    "laplacian_spectrum",
    "algebraic_connectivity",
    "eigenratio",
    "sync_threshold",
}
_CHIMERA = {
    "local_order_parameter",
    "chimera_index",
    "detect_chimera",
}
_SUPERVISOR = {
    "DifferentiableSupervisorConfig",
    "DifferentiableSupervisorPolicy",
    "SupervisorAction",
    "SupervisorActionProjection",
    "KuramotoSupervisorScenario",
    "SupervisorLossAux",
    "SupervisorPPOBatch",
    "SupervisorPPOCorpusRollout",
    "SupervisorPPOAux",
    "SupervisorPPOCheckpoint",
    "SupervisorPPOTrainResult",
    "SupervisorReplayProposal",
    "SupervisorScenarioCorpus",
    "masked_order_parameter",
    "apply_supervisor_action",
    "closed_loop_supervisor_loss",
    "supervisor_train_step",
    "pack_supervisor_action",
    "unpack_supervisor_action",
    "sample_supervisor_action",
    "supervisor_action_bound_penalty",
    "supervisor_action_log_prob",
    "project_supervisor_action_for_audit",
    "build_supervisor_replay_proposal",
    "ppo_supervisor_loss",
    "ppo_supervisor_train_step",
    "ppo_supervisor_train_epochs",
    "ppo_supervisor_train_with_checkpoint",
    "collect_supervisor_corpus_rollouts",
    "collect_supervisor_rollouts",
    "build_supervisor_scenario_corpus",
    "save_supervisor_ppo_checkpoint",
    "load_supervisor_ppo_checkpoint",
    "control_actions_from_supervisor",
}


def __getattr__(name: str) -> object:  # noqa: ANN204
    try:
        return _resolve(name)
    except ImportError as exc:
        raise AttributeError(
            f"nn.{name} requires JAX: pip install scpn-phase-orchestrator[nn]"
        ) from exc


def _resolve(name: str) -> object:
    if name in _FUNCTIONAL:
        from . import functional

        return getattr(functional, name)
    if name in _BOLD:
        from . import bold

        return getattr(bold, name)
    if name in _INVERSE:
        from . import inverse

        return getattr(inverse, name)
    if name in _OIM:
        from . import oim

        return getattr(oim, name)
    if name in _RESERVOIR:
        from . import reservoir

        return getattr(reservoir, name)
    if name in _LAYERS:
        if name == "KuramotoLayer":
            from .kuramoto_layer import KuramotoLayer

            return KuramotoLayer
        if name == "SimplicialKuramotoLayer":
            from .simplicial_layer import SimplicialKuramotoLayer

            return SimplicialKuramotoLayer
        from .stuart_landau_layer import StuartLandauLayer

        return StuartLandauLayer
    if name in _UDE:
        from .ude import CouplingResidual, UDEKuramotoLayer

        return UDEKuramotoLayer if name == "UDEKuramotoLayer" else CouplingResidual
    if name in _TRAINING:
        from . import training

        return getattr(training, name)
    if name in _THETA:
        from . import theta_neuron

        return getattr(theta_neuron, name)
    if name in _SPECTRAL:
        from . import spectral

        return getattr(spectral, name)
    if name in _CHIMERA:
        from . import chimera

        return getattr(chimera, name)
    if name in _SUPERVISOR:
        from . import supervisor

        return getattr(supervisor, name)
    msg = f"module 'scpn_phase_orchestrator.nn' has no attribute {name!r}"
    raise AttributeError(msg)
