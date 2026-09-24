# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — supervisor action audit projection

"""Audit-side safety projection of supervisor actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from ._shared import (
    _action_bounds,
    _bounded_unit_scalar,
    _non_negative_float,
    _positive_float,
    _supervisor_projection_control_records,
)
from ._types import SupervisorActionProjection
from .policy import pack_supervisor_action, unpack_supervisor_action

if TYPE_CHECKING:
    from ._types import DifferentiableSupervisorConfig, SupervisorAction


def project_supervisor_action_for_audit(
    action: SupervisorAction,
    config: DifferentiableSupervisorConfig,
    *,
    previous_action: SupervisorAction | None = None,
    ttl_s: float = 5.0,
    max_ttl_s: float = 5.0,
    rate_limit_fraction: float = 1.0,
    include_layer_actions: bool = True,
    regime_churn_score: float | None = None,
    max_regime_churn: float | None = None,
) -> SupervisorActionProjection:
    """Project a neural proposal into replay-safe bounds with audit metadata.

    This is intentionally non-actuating. It creates the explicit audit envelope
    that callers can inspect before converting a proposal into ``ControlAction``
    objects for any live adapter path.

    Parameters
    ----------
    action : SupervisorAction
        The supervisor control action.
    config : DifferentiableSupervisorConfig
        The supervisor configuration.
    previous_action : SupervisorAction | None
        The previous supervisor action, or ``None``.
    ttl_s : float
        Action time-to-live in seconds.
    max_ttl_s : float
        Maximum action time-to-live in seconds.
    rate_limit_fraction : float
        Maximum fractional change per step.
    include_layer_actions : bool
        Whether to include per-layer actions.
    regime_churn_score : float | None
        The regime-churn score, or ``None``.
    max_regime_churn : float | None
        Maximum allowed regime churn, or ``None``.

    Returns
    -------
    SupervisorActionProjection
        The replay-safe action projection with audit metadata. A proposal or
        previous action with a NaN or infinite component is rejected
        (``non_finite_proposal`` / ``non_finite_previous_action``) and projected
        to the zero action: ``jnp.clip`` passes NaN through, so clipping alone
        would hand on a NaN control marked as projected.

    Raises
    ------
    ValueError
        If a scalar constraint is out of range, or the action (or previous
        action) does not have the component count the config's bounds define.
    """
    ttl_s = _positive_float(ttl_s, "ttl_s")
    max_ttl_s = _positive_float(max_ttl_s, "max_ttl_s")
    rate_limit_fraction = _bounded_unit_scalar(
        rate_limit_fraction,
        "rate_limit_fraction",
    )
    if regime_churn_score is not None:
        regime_churn_score = _non_negative_float(
            regime_churn_score,
            "regime_churn_score",
        )
    if max_regime_churn is not None:
        max_regime_churn = _positive_float(max_regime_churn, "max_regime_churn")
    projected_ttl = min(ttl_s, max_ttl_s)
    bounds = _action_bounds(config)
    proposed_values = pack_supervisor_action(action)
    _require_matching_shape(proposed_values, bounds, "action")
    rejection_reasons: list[str] = []
    if not bool(jnp.all(jnp.isfinite(proposed_values))):
        rejection_reasons.append("non_finite_proposal")
    bounded_values = jnp.clip(proposed_values, -bounds, bounds)
    rate_limited_values = bounded_values

    if previous_action is not None:
        previous_values = pack_supervisor_action(previous_action)
        _require_matching_shape(previous_values, bounds, "previous_action")
        if not bool(jnp.all(jnp.isfinite(previous_values))):
            rejection_reasons.append("non_finite_previous_action")
        max_delta = bounds * rate_limit_fraction
        lower = previous_values - max_delta
        upper = previous_values + max_delta
        rate_limited_values = jnp.clip(bounded_values, lower, upper)

    if not include_layer_actions:
        rate_limited_values = rate_limited_values.at[2:].set(0.0)

    if (
        regime_churn_score is not None
        and max_regime_churn is not None
        and regime_churn_score > max_regime_churn
    ):
        rejection_reasons.append("regime_churn")
    if rejection_reasons:
        rate_limited_values = jnp.zeros_like(rate_limited_values)

    projected_action = unpack_supervisor_action(
        rate_limited_values,
        value_estimate=action.value_estimate,
        config=config,
    )
    controls = _supervisor_projection_control_records(
        proposed_values=proposed_values,
        projected_values=rate_limited_values,
        bounds=bounds,
    )
    clipped = projected_ttl != ttl_s or any(
        bool(control["clipped"]) for control in controls
    )
    audit_record = {
        "proposal_type": "differentiable_supervisor_action_projection",
        "non_actuating": True,
        "rejected": bool(rejection_reasons),
        "rejection_reasons": rejection_reasons,
        "clipped": clipped,
        "ttl_s": projected_ttl,
        "requested_ttl_s": ttl_s,
        "constraints": {
            "max_ttl_s": max_ttl_s,
            "rate_limit_fraction": rate_limit_fraction,
            "include_layer_actions": include_layer_actions,
            "previous_action": previous_action is not None,
            "regime_churn_score": regime_churn_score,
            "max_regime_churn": max_regime_churn,
        },
        "controls": controls,
    }
    return SupervisorActionProjection(
        action=projected_action,
        ttl_s=projected_ttl,
        audit_record=audit_record,
    )


def _require_matching_shape(values: jax.Array, bounds: jax.Array, field: str) -> None:
    """Raise ``ValueError`` unless ``values`` has one component per bound."""
    if values.shape != bounds.shape:
        raise ValueError(
            f"{field} has {values.shape[0]} components but the config bounds "
            f"{bounds.shape[0]} (2 global + n_layer_controls)"
        )
