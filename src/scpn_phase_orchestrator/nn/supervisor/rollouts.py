# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — supervisor rollout collection

"""Scenario rollout collection and corpus construction for PPO training."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import isfinite
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from ..functional import kuramoto_forward
from ._shared import (
    _bounded_unit_scalar,
    _json_object,
    _policy_mean_and_value,
    _positive_float,
    _positive_int,
    masked_order_parameter,
)
from ._types import (
    KuramotoSupervisorScenario,
    SupervisorPPOBatch,
    SupervisorPPOCorpusRollout,
    SupervisorPPORollout,
    SupervisorScenarioCorpus,
)
from .policy import (
    apply_supervisor_action,
    pack_supervisor_action,
    sample_supervisor_action,
)

if TYPE_CHECKING:
    from .policy import DifferentiableSupervisorPolicy


def collect_supervisor_rollouts(
    policy: DifferentiableSupervisorPolicy,
    scenario: KuramotoSupervisorScenario,
    *,
    key: jax.Array,
    n_episodes: int,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    trajectory_jitter: float = 0.0,
) -> SupervisorPPORollout:
    """Collect deterministic, replay-only PPO rollouts from a starting scenario.

    Parameters
    ----------
    policy : DifferentiableSupervisorPolicy
        The differentiable supervisor policy.
    scenario : KuramotoSupervisorScenario
        The Kuramoto supervisor scenario.
    key : jax.Array
        JAX PRNG key.
    n_episodes : int
        Number of rollout episodes.
    gamma : float
        Flow-dependent elimination rate.
    gae_lambda : float
        Generalised-advantage-estimation lambda.
    trajectory_jitter : float
        Trajectory jitter magnitude.

    Returns
    -------
    SupervisorPPORollout
        The collected PPO rollouts.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    n_episodes = _positive_int(n_episodes, "n_episodes")
    horizon = _positive_int(scenario.horizon, "scenario.horizon")
    inner_steps = _positive_int(scenario.inner_steps, "scenario.inner_steps")
    gamma = _bounded_unit_scalar(gamma, "gamma")
    gae_lambda = _bounded_unit_scalar(gae_lambda, "gae_lambda")
    if isinstance(trajectory_jitter, bool) or not isinstance(
        trajectory_jitter, int | float
    ):
        raise ValueError("trajectory_jitter must be a finite float")
    trajectory_jitter = float(trajectory_jitter)
    if not isfinite(trajectory_jitter) or trajectory_jitter < 0.0:
        raise ValueError("trajectory_jitter must be non-negative")

    rollout_keys = jax.random.split(key, n_episodes + 1)[1:]
    all_phases: list[jax.Array] = []
    all_omegas: list[jax.Array] = []
    all_base_k: list[jax.Array] = []
    all_good_masks: list[jax.Array] = []
    all_bad_masks: list[jax.Array] = []
    all_actions: list[jax.Array] = []
    all_old_log_probs: list[jax.Array] = []
    all_advantages: list[jax.Array] = []
    all_returns: list[jax.Array] = []
    all_values: list[jax.Array] = []
    episode_returns: list[jax.Array] = []

    for episode_key in rollout_keys:
        state = scenario
        if trajectory_jitter > 0.0:
            jitter_key, step_key = jax.random.split(episode_key)
            state = state._replace(
                phases=state.phases
                + trajectory_jitter
                * jax.random.normal(jitter_key, shape=state.phases.shape),
            )
            step_key = jax.random.fold_in(step_key, 1)
        else:
            step_key = episode_key

        step_rewards: list[jax.Array] = []
        step_values: list[jax.Array] = []
        step_actions: list[jax.Array] = []
        step_log_probs: list[jax.Array] = []
        step_phases: list[jax.Array] = []

        for _ in range(horizon):
            current_key, step_key = jax.random.split(step_key)
            action, action_log_prob = sample_supervisor_action(
                policy, state, key=current_key
            )
            controlled_K = apply_supervisor_action(scenario.base_K, action, state)
            next_phases, _ = kuramoto_forward(
                state.phases,
                state.omegas,
                controlled_K,
                state.dt,
                inner_steps,
            )
            reward = masked_order_parameter(next_phases, state.good_mask) - (
                policy.config.bad_sync_weight
                * masked_order_parameter(next_phases, state.bad_mask)
            )

            step_phases.append(state.phases)
            step_rewards.append(reward)
            step_values.append(action.value_estimate)
            step_actions.append(pack_supervisor_action(action))
            step_log_probs.append(action_log_prob)
            state = state._replace(phases=next_phases)

        terminal_value = _policy_mean_and_value(policy, state)[1]
        trajectory_values = jnp.stack(step_values + [terminal_value])
        rewards = jnp.stack(step_rewards)
        deltas = rewards + gamma * trajectory_values[1:] - trajectory_values[:-1]

        episode_advantages = [jnp.array(0.0)] * horizon
        running_advantage = jnp.array(0.0)
        for index in reversed(range(horizon)):
            running_advantage = deltas[index] + gamma * gae_lambda * running_advantage
            episode_advantages[index] = running_advantage
        advantages = jnp.stack(episode_advantages)
        returns = advantages + trajectory_values[:-1]

        episode_returns.append(jnp.sum(rewards))
        for index in range(horizon):
            all_phases.append(step_phases[index])
            all_omegas.append(scenario.omegas)
            all_base_k.append(scenario.base_K)
            all_good_masks.append(scenario.good_mask)
            all_bad_masks.append(scenario.bad_mask)
            all_actions.append(step_actions[index])
            all_old_log_probs.append(step_log_probs[index])
            all_advantages.append(advantages[index])
            all_returns.append(returns[index])
            all_values.append(step_values[index])

    episode_returns_array = jnp.stack(episode_returns)
    batch = SupervisorPPOBatch(
        phases=jnp.stack(all_phases),
        omegas=jnp.stack(all_omegas),
        base_K=jnp.stack(all_base_k),
        good_mask=jnp.stack(all_good_masks),
        bad_mask=jnp.stack(all_bad_masks),
        actions=jnp.stack(all_actions),
        old_log_probs=jnp.stack(all_old_log_probs),
        advantages=jnp.stack(all_advantages),
        returns=jnp.stack(all_returns),
        values=jnp.stack(all_values),
        dt=scenario.dt,
        inner_steps=inner_steps,
        horizon=horizon,
    )
    return SupervisorPPORollout(
        batch=batch,
        episode_returns=episode_returns_array,
        episode_return_mean=jnp.mean(episode_returns_array),
        episode_return_std=jnp.std(episode_returns_array),
    )


def collect_supervisor_corpus_rollouts(
    policy: DifferentiableSupervisorPolicy,
    corpus: SupervisorScenarioCorpus,
    *,
    key: jax.Array,
    n_episodes_per_scenario: int,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    trajectory_jitter: float = 0.0,
) -> SupervisorPPOCorpusRollout:
    """Collect replay-only PPO rollouts across a validated scenario corpus.

    The returned batch is a flat concatenation suitable for PPO epochs. All
    corpus scenarios must share ``dt``, ``inner_steps``, ``horizon``, and
    oscillator tensor shapes because ``SupervisorPPOBatch`` stores timing
    fields once for the full batch.

    Parameters
    ----------
    policy : DifferentiableSupervisorPolicy
        The differentiable supervisor policy.
    corpus : SupervisorScenarioCorpus
        The validated supervisor scenario corpus.
    key : jax.Array
        JAX PRNG key.
    n_episodes_per_scenario : int
        Number of episodes per corpus scenario.
    gamma : float
        Flow-dependent elimination rate.
    gae_lambda : float
        Generalised-advantage-estimation lambda.
    trajectory_jitter : float
        Trajectory jitter magnitude.

    Returns
    -------
    SupervisorPPOCorpusRollout
        The collected corpus PPO rollouts.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    n_episodes_per_scenario = _positive_int(
        n_episodes_per_scenario,
        "n_episodes_per_scenario",
    )
    if not corpus.scenarios:
        raise ValueError("scenario corpus requires at least one scenario")
    if len(corpus.metadata) != len(corpus.scenarios):
        raise ValueError("scenario corpus metadata must match scenario count")
    _validate_supervisor_corpus_rollout_compatibility(corpus.scenarios)

    scenario_keys = jax.random.split(key, len(corpus.scenarios))
    scenario_rollouts: list[SupervisorPPORollout] = []
    scenario_indices: list[jax.Array] = []
    for scenario_index, (scenario, scenario_key) in enumerate(
        zip(corpus.scenarios, scenario_keys, strict=True)
    ):
        rollout = collect_supervisor_rollouts(
            policy,
            scenario,
            key=scenario_key,
            n_episodes=n_episodes_per_scenario,
            gamma=gamma,
            gae_lambda=gae_lambda,
            trajectory_jitter=trajectory_jitter,
        )
        scenario_rollouts.append(rollout)
        scenario_indices.append(
            jnp.full(
                (n_episodes_per_scenario,),
                scenario_index,
                dtype=jnp.int32,
            )
        )

    batch = _concatenate_supervisor_ppo_batches(
        tuple(rollout.batch for rollout in scenario_rollouts)
    )
    episode_returns = jnp.concatenate(
        [rollout.episode_returns for rollout in scenario_rollouts],
        axis=0,
    )
    return SupervisorPPOCorpusRollout(
        batch=batch,
        episode_returns=episode_returns,
        episode_return_mean=jnp.mean(episode_returns),
        episode_return_std=jnp.std(episode_returns),
        scenario_indices=jnp.concatenate(scenario_indices, axis=0),
        metadata=tuple(dict(item) for item in corpus.metadata),
    )


def build_supervisor_scenario_corpus(
    records: Iterable[Mapping[str, Any]],
    *,
    dtype: Any = jnp.float32,
) -> SupervisorScenarioCorpus:
    """Convert replay/audit records into validated supervisor scenarios.

    Parameters
    ----------
    records : Iterable[Mapping[str, Any]]
        Replay/audit records to convert.
    dtype : Any
        Target array dtype.

    Returns
    -------
    SupervisorScenarioCorpus
        The validated supervisor scenario corpus.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    scenarios: list[KuramotoSupervisorScenario] = []
    metadata: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"record {index} must be a mapping")
        scenario, record_metadata = _supervisor_scenario_from_record(
            record,
            index=index,
            dtype=dtype,
        )
        scenarios.append(scenario)
        metadata.append(record_metadata)
    if not scenarios:
        raise ValueError("scenario corpus requires at least one record")
    return SupervisorScenarioCorpus(
        scenarios=tuple(scenarios),
        metadata=tuple(metadata),
    )


def _validate_supervisor_corpus_rollout_compatibility(
    scenarios: tuple[KuramotoSupervisorScenario, ...],
) -> None:
    reference = scenarios[0]
    reference_shapes = (
        reference.phases.shape,
        reference.omegas.shape,
        reference.base_K.shape,
        reference.good_mask.shape,
        reference.bad_mask.shape,
    )
    for index, scenario in enumerate(scenarios[1:], start=1):
        if (
            scenario.dt != reference.dt
            or scenario.inner_steps != reference.inner_steps
            or scenario.horizon != reference.horizon
        ):
            raise ValueError(
                "all corpus scenarios must share the same dt, inner_steps, and horizon"
            )
        scenario_shapes = (
            scenario.phases.shape,
            scenario.omegas.shape,
            scenario.base_K.shape,
            scenario.good_mask.shape,
            scenario.bad_mask.shape,
        )
        if scenario_shapes != reference_shapes:
            raise ValueError(
                "all corpus scenarios must share oscillator tensor shapes "
                f"for batch concatenation; scenario {index} differs"
            )


def _concatenate_supervisor_ppo_batches(
    batches: tuple[SupervisorPPOBatch, ...],
) -> SupervisorPPOBatch:
    reference = batches[0]
    for index, batch in enumerate(batches[1:], start=1):
        if (
            batch.dt != reference.dt
            or batch.inner_steps != reference.inner_steps
            or batch.horizon != reference.horizon
        ):
            raise ValueError(
                "all PPO batches must share the same dt, inner_steps, and horizon; "
                f"batch {index} differs"
            )
    return SupervisorPPOBatch(
        phases=jnp.concatenate([batch.phases for batch in batches], axis=0),
        omegas=jnp.concatenate([batch.omegas for batch in batches], axis=0),
        base_K=jnp.concatenate([batch.base_K for batch in batches], axis=0),
        good_mask=jnp.concatenate([batch.good_mask for batch in batches], axis=0),
        bad_mask=jnp.concatenate([batch.bad_mask for batch in batches], axis=0),
        actions=jnp.concatenate([batch.actions for batch in batches], axis=0),
        old_log_probs=jnp.concatenate(
            [batch.old_log_probs for batch in batches],
            axis=0,
        ),
        advantages=jnp.concatenate([batch.advantages for batch in batches], axis=0),
        returns=jnp.concatenate([batch.returns for batch in batches], axis=0),
        values=jnp.concatenate([batch.values for batch in batches], axis=0),
        dt=reference.dt,
        inner_steps=reference.inner_steps,
        horizon=reference.horizon,
    )


def _supervisor_scenario_from_record(
    record: Mapping[str, Any],
    *,
    index: int,
    dtype: Any,
) -> tuple[KuramotoSupervisorScenario, dict[str, Any]]:
    phases = _record_vector(record, "phases", index=index, dtype=dtype)
    omegas = _record_vector(record, "omegas", index=index, dtype=dtype)
    good_mask = _record_vector(record, "good_mask", index=index, dtype=dtype)
    bad_mask = _record_vector(record, "bad_mask", index=index, dtype=dtype)
    n_oscillators = int(phases.shape[0])
    for field_name, value in (
        ("omegas", omegas),
        ("good_mask", good_mask),
        ("bad_mask", bad_mask),
    ):
        if value.shape != (n_oscillators,):
            raise ValueError(
                f"record {index} {field_name} must have shape "
                f"({n_oscillators},), got {value.shape}"
            )
    if float(jnp.sum(good_mask)) <= 0.0:
        raise ValueError(f"record {index} good_mask must contain positive mass")
    if float(jnp.sum(bad_mask)) <= 0.0:
        raise ValueError(f"record {index} bad_mask must contain positive mass")

    base_k = _record_matrix(record, "base_K", index=index, dtype=dtype)
    if base_k.shape != (n_oscillators, n_oscillators):
        raise ValueError(
            f"record {index} base_K must have shape "
            f"({n_oscillators}, {n_oscillators}), got {base_k.shape}"
        )

    scenario = KuramotoSupervisorScenario(
        phases=phases,
        omegas=omegas,
        base_K=base_k,
        good_mask=good_mask,
        bad_mask=bad_mask,
        dt=_record_positive_float(record, "dt", index=index),
        inner_steps=_record_positive_int(record, "inner_steps", index=index),
        horizon=_record_positive_int(record, "horizon", index=index),
    )
    return scenario, _json_object(record.get("metadata"), "metadata")


def _record_vector(
    record: Mapping[str, Any],
    field: str,
    *,
    index: int,
    dtype: Any,
) -> jax.Array:
    value = _record_array(record, field, index=index, dtype=dtype)
    if value.ndim != 1 or value.shape[0] <= 0:
        raise ValueError(f"record {index} {field} must be a non-empty vector")
    if field.endswith("_mask") and bool(jnp.any(value < 0.0)):
        raise ValueError(f"record {index} {field} must be non-negative")
    return value


def _record_matrix(
    record: Mapping[str, Any],
    field: str,
    *,
    index: int,
    dtype: Any,
) -> jax.Array:
    value = _record_array(record, field, index=index, dtype=dtype)
    if value.ndim != 2:
        raise ValueError(f"record {index} {field} must be a matrix")
    return value


def _record_array(
    record: Mapping[str, Any],
    field: str,
    *,
    index: int,
    dtype: Any,
) -> jax.Array:
    if field not in record:
        raise ValueError(f"record {index} missing required field {field}")
    value = jnp.asarray(record[field], dtype=dtype)
    if not bool(jnp.all(jnp.isfinite(value))):
        raise ValueError(f"record {index} {field} must contain only finite values")
    return value


def _record_positive_float(
    record: Mapping[str, Any],
    field: str,
    *,
    index: int,
) -> float:
    if field not in record:
        raise ValueError(f"record {index} missing required field {field}")
    try:
        return _positive_float(record[field], f"record {index} {field}")
    except ValueError as exc:
        raise ValueError(f"record {index} {field} must be finite and positive") from exc


def _record_positive_int(
    record: Mapping[str, Any],
    field: str,
    *,
    index: int,
) -> int:
    if field not in record:
        raise ValueError(f"record {index} missing required field {field}")
    try:
        return _positive_int(record[field], f"record {index} {field}")
    except ValueError as exc:
        raise ValueError(f"record {index} {field} must be a positive integer") from exc
