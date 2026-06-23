# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — supervisor comparison metrics

"""Replay, learner, and baseline comparison metrics for supervisor policies."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
from scpn_phase_orchestrator.supervisor.policy import SupervisorPolicy
from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

from ..functional import kuramoto_forward, order_parameter
from ._shared import (
    _action_bounds,
    _audit_record_from_object,
    _control_energy,
    _is_finite_number,
    _json_object,
    _supervisor_action_to_record,
    masked_order_parameter,
)
from ._types import (
    SupervisorAction,
    SupervisorHandTunedBaselineComparison,
    SupervisorLearnerProposalComparison,
    SupervisorRandomBaselineComparison,
    SupervisorReplayComparison,
    SupervisorStaticBaselineComparison,
)
from .policy import (
    apply_supervisor_action,
    supervisor_action_bound_penalty,
    unpack_supervisor_action,
)

if TYPE_CHECKING:
    from ._types import (
        DifferentiableSupervisorConfig,
        KuramotoSupervisorScenario,
        SupervisorCorpusReplayProposals,
        SupervisorReplayProposal,
    )
    from .policy import DifferentiableSupervisorPolicy


def compare_supervisor_replay_proposal(
    supervisor_proposal: SupervisorReplayProposal | SupervisorCorpusReplayProposals,
    replay_policy_search: Any,
    *,
    comparison_label: str = "replay_policy_search",
) -> SupervisorReplayComparison:
    """Compare supervisor replay proposals with a replay policy-search result.

    The result is deliberately audit-only: it records both proposal surfaces and
    scalar comparison metrics, but it never authorises live actuation.

    Parameters
    ----------
    supervisor_proposal : SupervisorReplayProposal | SupervisorCorpusReplayProposals
        The supervisor replay proposal(s).
    replay_policy_search : Any
        The replay policy-search result.
    comparison_label : str
        Label identifying the comparison.

    Returns
    -------
    SupervisorReplayComparison
        The supervisor-vs-policy-search comparison.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    if not comparison_label:
        raise ValueError("comparison_label must not be empty")
    supervisor_record = _json_object(
        supervisor_proposal.to_audit_record(),
        "supervisor proposal audit record",
    )
    replay_record = _audit_record_from_object(
        replay_policy_search,
        "replay_policy_search",
    )
    metrics = _supervisor_replay_comparison_metrics(
        supervisor_record,
        replay_record,
    )
    return SupervisorReplayComparison(
        supervisor=supervisor_record,
        replay_policy_search=replay_record,
        comparison_label=comparison_label,
        metrics=metrics,
        actuation_permitted=False,
    )


def compare_supervisor_learner_proposals(
    supervisor_proposal: SupervisorReplayProposal | SupervisorCorpusReplayProposals,
    learner_proposals: Iterable[Any],
    *,
    comparison_label: str = "learner_proposal_generators",
) -> SupervisorLearnerProposalComparison:
    """Compare supervisor replay output with replay-only autotune learner proposals.

    Parameters
    ----------
    supervisor_proposal : SupervisorReplayProposal | SupervisorCorpusReplayProposals
        The supervisor replay proposal(s).
    learner_proposals : Iterable[Any]
        The replay-only learner proposals.
    comparison_label : str
        Label identifying the comparison.

    Returns
    -------
    SupervisorLearnerProposalComparison
        The supervisor-vs-learner comparison.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    if not comparison_label:
        raise ValueError("comparison_label must not be empty")
    supervisor_record = _json_object(
        supervisor_proposal.to_audit_record(),
        "supervisor proposal audit record",
    )
    learner_records = tuple(
        _learner_proposal_record_from_object(proposal, f"learner proposal {index}")
        for index, proposal in enumerate(learner_proposals)
    )
    if not learner_records:
        raise ValueError("learner comparison requires at least one learner proposal")
    return SupervisorLearnerProposalComparison(
        supervisor=supervisor_record,
        learner_proposals=learner_records,
        comparison_label=comparison_label,
        metrics=_learner_proposal_comparison_metrics(learner_records),
        actuation_permitted=False,
    )


def compare_supervisor_static_baseline(
    policy: DifferentiableSupervisorPolicy,
    scenario: KuramotoSupervisorScenario,
    *,
    comparison_label: str = "static_zero_action",
) -> SupervisorStaticBaselineComparison:
    """Compare one deterministic supervisor proposal against zero-action control.

    This is a benchmark/audit primitive only. It runs both candidates on the
    same scenario and records scalar metrics without returning actuation
    objects or enabling any live adapter handoff.

    Parameters
    ----------
    policy : DifferentiableSupervisorPolicy
        The differentiable supervisor policy.
    scenario : KuramotoSupervisorScenario
        The Kuramoto supervisor scenario.
    comparison_label : str
        Label identifying the comparison.

    Returns
    -------
    SupervisorStaticBaselineComparison
        The supervisor-vs-zero-action comparison.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    if not comparison_label:
        raise ValueError("comparison_label must not be empty")
    zero_action = SupervisorAction(
        delta_K_global=jnp.array(0.0),
        delta_zeta_global=jnp.array(0.0),
        delta_K_layers=jnp.zeros(policy.config.n_layer_controls),
        value_estimate=jnp.array(0.0),
    )
    baseline_record = _rollout_static_supervisor_action(
        "static_zero_action",
        zero_action,
        scenario,
        policy.config,
    )
    supervisor_record = _rollout_static_supervisor_action(
        "differentiable_supervisor",
        policy(scenario),
        scenario,
        policy.config,
    )
    baseline_metrics = _mapping_value(baseline_record, "metrics")
    supervisor_metrics = _mapping_value(supervisor_record, "metrics")
    metrics = _prefixed_float_metrics("baseline", baseline_metrics)
    metrics.update(_prefixed_float_metrics("supervisor", supervisor_metrics))
    metrics["delta_reward"] = metrics["supervisor_reward"] - metrics["baseline_reward"]
    return SupervisorStaticBaselineComparison(
        baseline=baseline_record,
        supervisor=supervisor_record,
        scenario_summary=_supervisor_scenario_summary(scenario),
        comparison_label=comparison_label,
        metrics=metrics,
        actuation_permitted=False,
    )


def compare_supervisor_random_baseline(
    policy: DifferentiableSupervisorPolicy,
    scenario: KuramotoSupervisorScenario,
    *,
    key: jax.Array,
    comparison_label: str = "bounded_random_action",
) -> SupervisorRandomBaselineComparison:
    """Compare one deterministic supervisor proposal against bounded randomness.

    Parameters
    ----------
    policy : DifferentiableSupervisorPolicy
        The differentiable supervisor policy.
    scenario : KuramotoSupervisorScenario
        The Kuramoto supervisor scenario.
    key : jax.Array
        JAX PRNG key.
    comparison_label : str
        Label identifying the comparison.

    Returns
    -------
    SupervisorRandomBaselineComparison
        The supervisor-vs-random comparison.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    if not comparison_label:
        raise ValueError("comparison_label must not be empty")
    random_action = _bounded_random_supervisor_action(policy.config, key)
    baseline_record = _rollout_static_supervisor_action(
        "bounded_random_action",
        random_action,
        scenario,
        policy.config,
    )
    baseline_record["seed"] = _jax_key_record(key)
    supervisor_record = _rollout_static_supervisor_action(
        "differentiable_supervisor",
        policy(scenario),
        scenario,
        policy.config,
    )
    baseline_metrics = _mapping_value(baseline_record, "metrics")
    supervisor_metrics = _mapping_value(supervisor_record, "metrics")
    metrics = _prefixed_float_metrics("baseline", baseline_metrics)
    metrics.update(_prefixed_float_metrics("supervisor", supervisor_metrics))
    metrics["delta_reward"] = metrics["supervisor_reward"] - metrics["baseline_reward"]
    return SupervisorRandomBaselineComparison(
        baseline=baseline_record,
        supervisor=supervisor_record,
        scenario_summary=_supervisor_scenario_summary(scenario),
        comparison_label=comparison_label,
        metrics=metrics,
        actuation_permitted=False,
    )


def compare_supervisor_hand_tuned_baseline(
    policy: DifferentiableSupervisorPolicy,
    scenario: KuramotoSupervisorScenario,
    *,
    boundary_state: BoundaryState | None = None,
    comparison_label: str = "hand_tuned_supervisor_policy",
) -> SupervisorHandTunedBaselineComparison:
    """Compare a neural supervisor proposal against rule-based ``SupervisorPolicy``.

    Parameters
    ----------
    policy : DifferentiableSupervisorPolicy
        The differentiable supervisor policy.
    scenario : KuramotoSupervisorScenario
        The Kuramoto supervisor scenario.
    boundary_state : BoundaryState | None
        The boundary-observer state, or ``None``.
    comparison_label : str
        Label identifying the comparison.

    Returns
    -------
    SupervisorHandTunedBaselineComparison
        The supervisor-vs-rule-based comparison.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    if not comparison_label:
        raise ValueError("comparison_label must not be empty")
    active_boundary = boundary_state or BoundaryState()
    upde_state = _upde_state_from_supervisor_scenario(scenario)
    legacy_policy = SupervisorPolicy(RegimeManager(cooldown_steps=0))
    policy_actions = legacy_policy.decide(upde_state, active_boundary)
    baseline_action = _supervisor_action_from_control_actions(
        policy_actions, policy.config
    )
    baseline_record = _rollout_static_supervisor_action(
        "hand_tuned_supervisor_policy",
        baseline_action,
        scenario,
        policy.config,
    )
    baseline_record["policy_actions"] = [
        _control_action_record(action) for action in policy_actions
    ]
    supervisor_record = _rollout_static_supervisor_action(
        "differentiable_supervisor",
        policy(scenario),
        scenario,
        policy.config,
    )
    baseline_metrics = _mapping_value(baseline_record, "metrics")
    supervisor_metrics = _mapping_value(supervisor_record, "metrics")
    metrics = _prefixed_float_metrics("baseline", baseline_metrics)
    metrics.update(_prefixed_float_metrics("supervisor", supervisor_metrics))
    metrics["delta_reward"] = metrics["supervisor_reward"] - metrics["baseline_reward"]
    return SupervisorHandTunedBaselineComparison(
        baseline=baseline_record,
        supervisor=supervisor_record,
        scenario_summary=_supervisor_scenario_summary(scenario),
        comparison_label=comparison_label,
        metrics=metrics,
        actuation_permitted=False,
    )


def _learner_proposal_record_from_object(value: Any, field: str) -> dict[str, Any]:
    """Return a validated learner-proposal record from an object."""
    record = _audit_record_from_object(value, field)
    if record.get("actuation_permitted") is not False:
        raise ValueError(f"{field} must be non-actuating")
    learner_kind = record.get("learner_kind")
    if not isinstance(learner_kind, str) or not learner_kind.endswith("_replay"):
        raise ValueError(f"{field} must be a replay learner proposal")
    if not isinstance(record.get("policy_search"), dict):
        raise ValueError(f"{field} must include policy_search")
    return record


def _learner_proposal_comparison_metrics(
    records: tuple[dict[str, Any], ...],
) -> dict[str, float]:
    """Return comparison metrics between learner and baseline proposals."""
    accepted_count = 0.0
    selected_rewards = []
    for record in records:
        policy_search = _mapping_value(record, "policy_search")
        proposal = policy_search.get("proposal")
        if isinstance(proposal, dict):
            if proposal.get("accepted") is True:
                accepted_count += 1.0
            selected = proposal.get("selected")
            if isinstance(selected, dict):
                reward = selected.get("reward")
                if _is_finite_number(reward):
                    selected_rewards.append(float(reward))
    metrics = {
        "learner_proposal_count": float(len(records)),
        "accepted_learner_count": accepted_count,
        "rejected_learner_count": float(len(records)) - accepted_count,
    }
    if selected_rewards:
        metrics["best_learner_selected_reward"] = max(selected_rewards)
        metrics["mean_learner_selected_reward"] = sum(selected_rewards) / len(
            selected_rewards
        )
    return metrics


def _supervisor_replay_comparison_metrics(
    supervisor_record: Mapping[str, Any],
    replay_record: Mapping[str, Any],
) -> dict[str, float]:
    """Return replay comparison metrics for the supervisor."""
    metrics: dict[str, float] = {}
    supervisor_type = supervisor_record.get("proposal_type")
    if supervisor_type == "differentiable_supervisor_replay_proposal":
        supervisor_metrics = _mapping_value(supervisor_record, "metrics")
        _put_finite_metric(
            metrics,
            "supervisor_value_estimate",
            supervisor_metrics.get("value_estimate"),
        )
        _put_finite_metric(
            metrics,
            "supervisor_current_R_good",
            supervisor_metrics.get("current_R_good"),
        )
        _put_finite_metric(
            metrics,
            "supervisor_current_R_bad",
            supervisor_metrics.get("current_R_bad"),
        )
        projection = _mapping_value(supervisor_record, "projection")
        metrics["supervisor_rejected_count"] = (
            1.0 if bool(projection.get("rejected")) else 0.0
        )
    elif supervisor_type == "differentiable_supervisor_corpus_replay":
        proposals = supervisor_record.get("proposals")
        if not isinstance(proposals, list):
            raise ValueError("supervisor corpus audit record proposals must be a list")
        metrics["supervisor_scenario_count"] = float(len(proposals))
        values = []
        rejected_count = 0.0
        for index, proposal in enumerate(proposals):
            if not isinstance(proposal, dict):
                raise ValueError(
                    f"supervisor corpus proposal {index} must be a JSON object"
                )
            proposal_metrics = _mapping_value(proposal, "metrics")
            value = proposal_metrics.get("value_estimate")
            if _is_finite_number(value):
                values.append(float(value))
            projection = _mapping_value(proposal, "projection")
            if bool(projection.get("rejected")):
                rejected_count += 1.0
        metrics["supervisor_rejected_count"] = rejected_count
        if values:
            metrics["supervisor_mean_value_estimate"] = sum(values) / len(values)
    else:
        raise ValueError("unsupported supervisor proposal audit record")

    replay_proposal = replay_record.get("proposal")
    if isinstance(replay_proposal, dict):
        selected = replay_proposal.get("selected")
        if isinstance(selected, dict):
            _put_finite_metric(
                metrics,
                "replay_selected_reward",
                selected.get("reward"),
            )
    rounds = replay_record.get("rounds")
    if isinstance(rounds, list):
        metrics["replay_round_count"] = float(len(rounds))
        metrics["replay_candidate_count"] = float(
            sum(_replay_round_candidate_count(round_record) for round_record in rounds)
        )
    config = replay_record.get("config")
    if isinstance(config, dict):
        _put_finite_metric(
            metrics,
            "replay_config_iterations",
            config.get("iterations"),
        )
    return metrics


def _replay_round_candidate_count(round_record: object) -> int:
    """Return the candidate count for a replay round."""
    if not isinstance(round_record, dict):
        raise ValueError("adaptive replay round must be a JSON object")
    candidates = round_record.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("adaptive replay round candidates must be a list")
    return len(candidates)


def _rollout_static_supervisor_action(
    name: str,
    action: SupervisorAction,
    scenario: KuramotoSupervisorScenario,
    config: DifferentiableSupervisorConfig,
) -> dict[str, Any]:
    """Return the static (deterministic) supervisor action for a rollout."""
    started = time.perf_counter()
    controlled_K = apply_supervisor_action(scenario.base_K, action, scenario)
    final_phases, _ = kuramoto_forward(
        scenario.phases,
        scenario.omegas,
        controlled_K,
        scenario.dt,
        scenario.inner_steps * scenario.horizon,
    )
    elapsed = time.perf_counter() - started
    final_R_good = float(masked_order_parameter(final_phases, scenario.good_mask))
    final_R_bad = float(masked_order_parameter(final_phases, scenario.bad_mask))
    control_energy = float(_control_energy(action))
    reward = (
        final_R_good
        - config.bad_sync_weight * final_R_bad
        - config.control_energy_weight * control_energy
    )
    bound_penalty = float(supervisor_action_bound_penalty(action, config))
    return _json_object(
        {
            "name": name,
            "action": _supervisor_action_to_record(action),
            "metrics": {
                "final_R_good": final_R_good,
                "final_R_bad": final_R_bad,
                "reward": reward,
                "control_energy": control_energy,
                "smoothness": 0.0,
                "safety_violations": 1.0 if bound_penalty > 0.0 else 0.0,
                "wall_time_s": elapsed,
            },
        },
        f"{name} baseline record",
    )


def _bounded_random_supervisor_action(
    config: DifferentiableSupervisorConfig,
    key: jax.Array,
) -> SupervisorAction:
    """Return a bounded random supervisor action for a rollout."""
    values = jax.random.uniform(
        key,
        shape=(2 + config.n_layer_controls,),
        minval=-1.0,
        maxval=1.0,
    ) * _action_bounds(config)
    return unpack_supervisor_action(
        values,
        value_estimate=jnp.array(0.0),
        config=config,
    )


def _jax_key_record(key: jax.Array) -> list[int]:
    """Return the audit record for a JAX PRNG key."""
    key_array = jnp.asarray(key)
    if key_array.ndim != 1:
        raise ValueError("key must be a one-dimensional JAX PRNG key")
    return [int(value) for value in key_array.tolist()]


def _upde_state_from_supervisor_scenario(
    scenario: KuramotoSupervisorScenario,
) -> UPDEState:
    """Build the UPDE simulation state from a supervisor scenario."""
    good_R = float(masked_order_parameter(scenario.phases, scenario.good_mask))
    bad_R = float(masked_order_parameter(scenario.phases, scenario.bad_mask))
    global_R = float(order_parameter(scenario.phases))
    return UPDEState(
        layers=[
            LayerState(R=good_R, psi=0.0),
            LayerState(R=bad_R, psi=0.0),
        ],
        cross_layer_alignment=np.asarray(
            [[1.0, global_R], [global_R, 1.0]], dtype=np.float64
        ),
        stability_proxy=global_R,
        regime_id="supervisor_baseline_comparison",
    )


def _supervisor_action_from_control_actions(
    actions: Iterable[ControlAction],
    config: DifferentiableSupervisorConfig,
) -> SupervisorAction:
    """Return the supervisor action reconstructed from control actions."""
    delta_K_global = 0.0
    delta_zeta_global = 0.0
    delta_K_layers = [0.0] * config.n_layer_controls
    for action in actions:
        if action.knob == "K" and action.scope == "global":
            delta_K_global += float(action.value)
        elif action.knob == "zeta" and action.scope == "global":
            delta_zeta_global += float(action.value)
        elif action.knob == "K" and action.scope.startswith("layer_"):
            layer_index = _layer_scope_index(action.scope)
            if layer_index < config.n_layer_controls:
                delta_K_layers[layer_index] += float(action.value)
    values = jnp.asarray([delta_K_global, delta_zeta_global, *delta_K_layers])
    bounded = jnp.clip(values, -_action_bounds(config), _action_bounds(config))
    return unpack_supervisor_action(
        bounded,
        value_estimate=jnp.array(0.0),
        config=config,
    )


def _layer_scope_index(scope: str) -> int:
    """Return the layer index for a control-action scope."""
    try:
        return int(scope.removeprefix("layer_"))
    except ValueError as exc:
        raise ValueError(f"invalid layer scope: {scope}") from exc


def _control_action_record(action: ControlAction) -> dict[str, Any]:
    """Return the JSON-safe record for a control action."""
    return {
        "knob": action.knob,
        "scope": action.scope,
        "value": float(action.value),
        "ttl_s": float(action.ttl_s),
        "justification": action.justification,
    }


def _supervisor_scenario_summary(
    scenario: KuramotoSupervisorScenario,
) -> dict[str, Any]:
    """Return a summary of a supervisor scenario."""
    return {
        "n_oscillators": int(scenario.phases.shape[0]),
        "dt": float(scenario.dt),
        "inner_steps": int(scenario.inner_steps),
        "horizon": int(scenario.horizon),
    }


def _prefixed_float_metrics(
    prefix: str,
    metrics: Mapping[str, Any],
) -> dict[str, float]:
    """Return the float metrics with a key prefix applied."""
    prefixed: dict[str, float] = {}
    for key, value in metrics.items():
        if not _is_finite_number(value):
            raise ValueError(f"{prefix} metric {key} must be finite")
        prefixed[f"{prefix}_{key}"] = float(value)
    return prefixed


def _mapping_value(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return a named value from a mapping, else raise."""
    child = value.get(key)
    if not isinstance(child, dict):
        raise ValueError(f"{key} must be a JSON object")
    return child


def _put_finite_metric(
    metrics: dict[str, float],
    key: str,
    value: object,
) -> None:
    """Store a finite metric into the target mapping, else raise."""
    if _is_finite_number(value):
        metrics[key] = float(value)
