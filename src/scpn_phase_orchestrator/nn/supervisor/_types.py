# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — supervisor typed records

"""Typed configuration, action, rollout, and comparison records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import jax

from ._shared import _json_object, _supervisor_action_to_record

if TYPE_CHECKING:
    from .policy import DifferentiableSupervisorPolicy


@dataclass(frozen=True)
class DifferentiableSupervisorConfig:
    """Static configuration for ``DifferentiableSupervisorPolicy``.

    Parameters
    ----------
    n_oscillators : object
        Number of oscillators in the controlled Kuramoto system.
    hidden_width : object
        Width of each MLP hidden layer.
    hidden_depth : object
        Number of hidden layers in the MLP.
    n_layer_controls : object
        Number of mask-scoped ``K`` controls. The default maps to good and bad
        partitions in ``KuramotoSupervisorScenario``.
    max_global_delta_K : object
        Absolute bound for global coupling increments.
    max_global_delta_zeta : object
        Absolute bound for global damping/drive command.
    max_layer_delta_K : object
        Absolute bound for partition-local coupling deltas.
    control_energy_weight : object
        Quadratic penalty on control action magnitude.
    bad_sync_weight : object
        Penalty for synchronising the bad partition.
    smoothness_weight : object
        Quadratic penalty on action changes over rollout.
    """

    n_oscillators: int
    hidden_width: int = 32
    hidden_depth: int = 2
    n_layer_controls: int = 2
    max_global_delta_K: float = 0.05
    max_global_delta_zeta: float = 0.1
    max_layer_delta_K: float = 0.03
    control_energy_weight: float = 1.0e-2
    bad_sync_weight: float = 0.25
    smoothness_weight: float = 1.0e-3


class SupervisorAction(NamedTuple):
    """Continuous differentiable control emitted by the neural supervisor."""

    delta_K_global: jax.Array
    delta_zeta_global: jax.Array
    delta_K_layers: jax.Array
    value_estimate: jax.Array


class SupervisorActionProjection(NamedTuple):
    """Non-actuating safety projection record for a neural supervisor proposal."""

    action: SupervisorAction
    ttl_s: float
    audit_record: dict[str, Any]


class SupervisorReplayProposal(NamedTuple):
    """Replay-only neural supervisor proposal record for audit review."""

    action: SupervisorAction
    projection: SupervisorActionProjection
    scenario_summary: dict[str, Any]
    scenario_metadata: dict[str, Any]
    metrics: dict[str, float]
    actuation_permitted: bool = False

    def to_audit_record(self) -> dict[str, Any]:
        """Return a JSON-serialisable replay proposal record.

        Returns
        -------
        dict[str, Any]
            Return a JSON-serialisable replay proposal record.
        """
        return {
            "proposal_type": "differentiable_supervisor_replay_proposal",
            "actuation_permitted": self.actuation_permitted,
            "scenario_summary": dict(self.scenario_summary),
            "scenario_metadata": dict(self.scenario_metadata),
            "metrics": dict(self.metrics),
            "action": _supervisor_action_to_record(self.action),
            "projection": dict(self.projection.audit_record),
        }


class SupervisorBaselineReport(NamedTuple):
    """Aggregate audit report for supervisor baseline comparison records."""

    comparisons: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    report_label: str
    actuation_permitted: bool = False

    def to_audit_record(self) -> dict[str, Any]:
        """Return a JSON-serialisable baseline aggregation report.

        Returns
        -------
        dict[str, Any]
            Return a JSON-serialisable baseline aggregation report.
        """
        return _json_object(
            {
                "proposal_type": "differentiable_supervisor_baseline_report",
                "actuation_permitted": self.actuation_permitted,
                "report_label": self.report_label,
                "summary": dict(self.summary),
                "comparisons": [dict(record) for record in self.comparisons],
            },
            "supervisor_baseline_report",
        )


class SupervisorExperimentManifest(NamedTuple):
    """Reproducibility manifest for supervisor baseline experiment artefacts."""

    baseline_report: dict[str, Any]
    command: str
    git_sha: str
    dependency_lock: dict[str, Any]
    device_info: dict[str, Any]
    seed_list: tuple[int, ...]
    artifacts: dict[str, Any]
    actuation_permitted: bool = False

    def to_audit_record(self) -> dict[str, Any]:
        """Return a JSON-serialisable reproducibility manifest.

        Returns
        -------
        dict[str, Any]
            Return a JSON-serialisable reproducibility manifest.
        """
        return _json_object(
            {
                "proposal_type": "differentiable_supervisor_experiment_manifest",
                "actuation_permitted": self.actuation_permitted,
                "command": self.command,
                "git_sha": self.git_sha,
                "dependency_lock": dict(self.dependency_lock),
                "device_info": dict(self.device_info),
                "seed_list": list(self.seed_list),
                "artifacts": dict(self.artifacts),
                "baseline_report": dict(self.baseline_report),
            },
            "supervisor_experiment_manifest",
        )


class SupervisorCorpusReplayProposals(NamedTuple):
    """Replay-only proposal set generated from a supervisor scenario corpus."""

    proposals: tuple[SupervisorReplayProposal, ...]
    actuation_permitted: bool = False

    def to_audit_record(self) -> dict[str, Any]:
        """Return a JSON-serialisable corpus proposal record.

        Returns
        -------
        dict[str, Any]
            Return a JSON-serialisable corpus proposal record.
        """
        return {
            "proposal_type": "differentiable_supervisor_corpus_replay",
            "actuation_permitted": self.actuation_permitted,
            "scenario_count": len(self.proposals),
            "proposals": [proposal.to_audit_record() for proposal in self.proposals],
        }


class SupervisorReplayComparison(NamedTuple):
    """Audit-only comparison between neural supervisor and replay policy search."""

    supervisor: dict[str, Any]
    replay_policy_search: dict[str, Any]
    comparison_label: str
    metrics: dict[str, float]
    actuation_permitted: bool = False

    def to_audit_record(self) -> dict[str, Any]:
        """Return a JSON-serialisable non-actuating comparison record.

        Returns
        -------
        dict[str, Any]
            Return a JSON-serialisable non-actuating comparison record.
        """
        return _json_object(
            {
                "proposal_type": "differentiable_supervisor_replay_comparison",
                "actuation_permitted": self.actuation_permitted,
                "comparison_label": self.comparison_label,
                "supervisor": dict(self.supervisor),
                "replay_policy_search": dict(self.replay_policy_search),
                "metrics": dict(self.metrics),
            },
            "supervisor_replay_comparison",
        )


class SupervisorLearnerProposalComparison(NamedTuple):
    """Audit-only comparison against learner-shaped autotune proposal records."""

    supervisor: dict[str, Any]
    learner_proposals: tuple[dict[str, Any], ...]
    comparison_label: str
    metrics: dict[str, float]
    actuation_permitted: bool = False

    def to_audit_record(self) -> dict[str, Any]:
        """Return a JSON-serialisable learner-proposal comparison record.

        Returns
        -------
        dict[str, Any]
            Return a JSON-serialisable learner-proposal comparison record.
        """
        return _json_object(
            {
                "proposal_type": "differentiable_supervisor_learner_comparison",
                "actuation_permitted": self.actuation_permitted,
                "comparison_label": self.comparison_label,
                "supervisor": dict(self.supervisor),
                "learner_proposals": [
                    dict(proposal) for proposal in self.learner_proposals
                ],
                "metrics": dict(self.metrics),
            },
            "supervisor_learner_proposal_comparison",
        )


class SupervisorStaticBaselineComparison(NamedTuple):
    """Audit-only comparison against a static zero-action supervisor baseline."""

    baseline: dict[str, Any]
    supervisor: dict[str, Any]
    scenario_summary: dict[str, Any]
    comparison_label: str
    metrics: dict[str, float]
    actuation_permitted: bool = False

    def to_audit_record(self) -> dict[str, Any]:
        """Return a JSON-serialisable static-baseline comparison record.

        Returns
        -------
        dict[str, Any]
            Return a JSON-serialisable static-baseline comparison record.
        """
        return _json_object(
            {
                "proposal_type": "differentiable_supervisor_static_baseline",
                "actuation_permitted": self.actuation_permitted,
                "comparison_label": self.comparison_label,
                "scenario_summary": dict(self.scenario_summary),
                "baseline": dict(self.baseline),
                "supervisor": dict(self.supervisor),
                "metrics": dict(self.metrics),
            },
            "supervisor_static_baseline_comparison",
        )


class SupervisorRandomBaselineComparison(NamedTuple):
    """Audit-only comparison against a seeded bounded-random action baseline."""

    baseline: dict[str, Any]
    supervisor: dict[str, Any]
    scenario_summary: dict[str, Any]
    comparison_label: str
    metrics: dict[str, float]
    actuation_permitted: bool = False

    def to_audit_record(self) -> dict[str, Any]:
        """Return a JSON-serialisable random-baseline comparison record.

        Returns
        -------
        dict[str, Any]
            Return a JSON-serialisable random-baseline comparison record.
        """
        return _json_object(
            {
                "proposal_type": "differentiable_supervisor_random_baseline",
                "actuation_permitted": self.actuation_permitted,
                "comparison_label": self.comparison_label,
                "scenario_summary": dict(self.scenario_summary),
                "baseline": dict(self.baseline),
                "supervisor": dict(self.supervisor),
                "metrics": dict(self.metrics),
            },
            "supervisor_random_baseline_comparison",
        )


class SupervisorHandTunedBaselineComparison(NamedTuple):
    """Audit-only comparison against the rule-based ``SupervisorPolicy``."""

    baseline: dict[str, Any]
    supervisor: dict[str, Any]
    scenario_summary: dict[str, Any]
    comparison_label: str
    metrics: dict[str, float]
    actuation_permitted: bool = False

    def to_audit_record(self) -> dict[str, Any]:
        """Return a JSON-serialisable hand-tuned-baseline comparison record.

        Returns
        -------
        dict[str, Any]
            Return a JSON-serialisable hand-tuned-baseline comparison record.
        """
        return _json_object(
            {
                "proposal_type": "differentiable_supervisor_hand_tuned_baseline",
                "actuation_permitted": self.actuation_permitted,
                "comparison_label": self.comparison_label,
                "scenario_summary": dict(self.scenario_summary),
                "baseline": dict(self.baseline),
                "supervisor": dict(self.supervisor),
                "metrics": dict(self.metrics),
            },
            "supervisor_hand_tuned_baseline_comparison",
        )


class KuramotoSupervisorScenario(NamedTuple):
    """Closed-loop differentiable Kuramoto control problem.

    ``good_mask`` and ``bad_mask`` are non-negative oscillator membership
    weights. Binary masks are typical, but soft memberships are supported for
    differentiable curriculum construction.
    """

    phases: jax.Array
    omegas: jax.Array
    base_K: jax.Array
    good_mask: jax.Array
    bad_mask: jax.Array
    dt: float
    inner_steps: int
    horizon: int


class SupervisorScenarioCorpus(NamedTuple):
    """Validated replay scenario corpus for supervisor training."""

    scenarios: tuple[KuramotoSupervisorScenario, ...]
    metadata: tuple[dict[str, Any], ...]


class SupervisorLossAux(NamedTuple):
    """Diagnostics returned by ``closed_loop_supervisor_loss``."""

    final_R_good: jax.Array
    final_R_bad: jax.Array
    control_energy: jax.Array
    smoothness: jax.Array


class SupervisorPPOBatch(NamedTuple):
    """On-policy PPO batch for the differentiable supervisor.

    Arrays carry a leading batch dimension. ``actions`` must contain bounded
    continuous actions produced by ``pack_supervisor_action``.
    """

    phases: jax.Array
    omegas: jax.Array
    base_K: jax.Array
    good_mask: jax.Array
    bad_mask: jax.Array
    actions: jax.Array
    old_log_probs: jax.Array
    advantages: jax.Array
    returns: jax.Array
    values: jax.Array
    dt: float
    inner_steps: int
    horizon: int


class SupervisorPPORollout(NamedTuple):
    """Replay-only rollout outputs for PPO-style supervisor training."""

    batch: SupervisorPPOBatch
    episode_returns: jax.Array
    episode_return_mean: jax.Array
    episode_return_std: jax.Array


class SupervisorPPOCorpusRollout(NamedTuple):
    """Corpus-wide replay rollout with per-episode scenario provenance."""

    batch: SupervisorPPOBatch
    episode_returns: jax.Array
    episode_return_mean: jax.Array
    episode_return_std: jax.Array
    scenario_indices: jax.Array
    metadata: tuple[dict[str, Any], ...]


class SupervisorPPOAux(NamedTuple):
    """Diagnostics returned by ``ppo_supervisor_loss``."""

    policy_loss: jax.Array
    value_loss: jax.Array
    entropy: jax.Array
    approx_kl: jax.Array
    clip_fraction: jax.Array


class SupervisorPPOCheckpoint(NamedTuple):
    """Loaded PPO checkpoint state for deterministic supervisor training resume."""

    policy: DifferentiableSupervisorPolicy
    opt_state: Any
    key: jax.Array
    loss_history: jax.Array
    n_updates: int
    metadata: dict[str, Any]


class SupervisorPPOTrainResult(NamedTuple):
    """PPO training result with checkpoint-resume bookkeeping."""

    policy: DifferentiableSupervisorPolicy
    opt_state: Any
    key: jax.Array
    loss_history: jax.Array
    n_updates: int
    checkpoint_path: Path | None


class _SupervisorPPOCheckpointPayload(NamedTuple):
    policy: DifferentiableSupervisorPolicy
    opt_state: Any
    key: jax.Array
    loss_history: jax.Array
