# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — supervisor shared helpers

"""Shared numerical and record helpers for the supervisor package."""

from __future__ import annotations

import json
from collections.abc import Mapping
from math import isfinite
from typing import TYPE_CHECKING, Any, TypeGuard

import jax
import jax.numpy as jnp

from ..functional import order_parameter

if TYPE_CHECKING:
    from ._types import (
        DifferentiableSupervisorConfig,
        KuramotoSupervisorScenario,
        SupervisorAction,
    )
    from .policy import DifferentiableSupervisorPolicy


def masked_order_parameter(phases: jax.Array, weights: jax.Array) -> jax.Array:
    """Weighted Kuramoto order parameter for a partition of oscillators.

    Parameters
    ----------
    phases : jax.Array
        Oscillator phases in radians, shape ``(N,)``.
    weights : jax.Array
        Per-oscillator partition weights.

    Returns
    -------
    jax.Array
        The weighted Kuramoto order parameter.
    """
    safe_weights = jnp.clip(weights, min=0.0)
    total = jnp.maximum(jnp.sum(safe_weights), 1.0e-12)
    z = jnp.sum(safe_weights * jnp.exp(1j * phases)) / total
    return jnp.abs(z)


def _supervisor_features(scenario: KuramotoSupervisorScenario) -> jax.Array:
    R_global = order_parameter(scenario.phases)
    R_good = masked_order_parameter(scenario.phases, scenario.good_mask)
    R_bad = masked_order_parameter(scenario.phases, scenario.bad_mask)
    mean_omega = jnp.mean(scenario.omegas)
    std_omega = jnp.std(scenario.omegas)
    mean_K = jnp.mean(scenario.base_K)
    abs_K = jnp.mean(jnp.abs(scenario.base_K))
    phase_spread = 1.0 - R_global
    return jnp.array(
        [R_global, R_good, R_bad, phase_spread, mean_omega, std_omega, mean_K, abs_K]
    )


def _policy_mean_and_value(
    policy: DifferentiableSupervisorPolicy,
    scenario: KuramotoSupervisorScenario,
) -> tuple[jax.Array, jax.Array]:
    raw = policy.network(_supervisor_features(scenario))
    return raw[:-1], raw[-1]


def _action_bounds(config: DifferentiableSupervisorConfig) -> jax.Array:
    return jnp.concatenate(
        [
            jnp.array([config.max_global_delta_K, config.max_global_delta_zeta]),
            jnp.full((config.n_layer_controls,), config.max_layer_delta_K),
        ]
    )


def _supervisor_projection_control_records(
    *,
    proposed_values: jax.Array,
    projected_values: jax.Array,
    bounds: jax.Array,
) -> list[dict[str, object]]:
    controls: list[dict[str, object]] = []
    names = [("K", "global"), ("zeta", "global")]
    names.extend(("K", f"layer_{idx}") for idx in range(projected_values.shape[0] - 2))
    for index, (knob, scope) in enumerate(names):
        proposed = float(proposed_values[index])
        projected = float(projected_values[index])
        bound = float(bounds[index])
        controls.append(
            {
                "knob": knob,
                "scope": scope,
                "proposed": proposed,
                "projected": projected,
                "lower_bound": -bound,
                "upper_bound": bound,
                "clipped": projected != proposed,
            }
        )
    return controls


def _supervisor_action_to_record(action: SupervisorAction) -> dict[str, object]:
    return {
        "delta_K_global": float(action.delta_K_global),
        "delta_zeta_global": float(action.delta_zeta_global),
        "delta_K_layers": [float(value) for value in action.delta_K_layers],
        "value_estimate": float(action.value_estimate),
    }


def _squashed_gaussian_log_prob(
    mean: jax.Array,
    log_std: jax.Array,
    pre_squash: jax.Array,
) -> jax.Array:
    std = jnp.exp(log_std)
    normalised = (pre_squash - mean) / std
    gaussian = -0.5 * (normalised**2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
    squash_correction = jnp.log(1.0 - jnp.tanh(pre_squash) ** 2 + 1.0e-6)
    return jnp.sum(gaussian - squash_correction)


def _control_energy(action: SupervisorAction) -> jax.Array:
    return (
        action.delta_K_global**2
        + action.delta_zeta_global**2
        + jnp.mean(action.delta_K_layers**2)
    )


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _bounded_unit_scalar(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be in [0, 1]")
    value_float = float(value)
    if not isfinite(value_float) or value_float < 0.0 or value_float > 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return value_float


def _positive_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a finite positive scalar")
    value_float = float(value)
    if not isfinite(value_float) or value_float <= 0.0:
        raise ValueError(f"{field} must be a finite positive scalar")
    return value_float


def _non_negative_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a finite non-negative scalar")
    value_float = float(value)
    if not isfinite(value_float) or value_float < 0.0:
        raise ValueError(f"{field} must be a finite non-negative scalar")
    return value_float


def _audit_record_from_object(value: Any, field: str) -> dict[str, Any]:
    to_audit_record = getattr(value, "to_audit_record", None)
    if not callable(to_audit_record):
        raise TypeError(f"{field} must provide to_audit_record()")
    return _json_safe_object(to_audit_record(), f"{field} audit record")


def _is_finite_number(value: object) -> TypeGuard[float]:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and isfinite(value)
    )


def _json_safe_object(value: object, field: str) -> dict[str, Any]:
    return _json_object(_json_safe_value(value), field)


def _json_safe_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(child) for key, child in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe_value(child) for child in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if isfinite(value):
            return value
        return {"nonfinite_float": str(value)}
    return value


def _json_object(value: object, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    try:
        json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be JSON serialisable") from exc
    return dict(value)
