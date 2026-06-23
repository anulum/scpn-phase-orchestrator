# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — neural supervisor to autotune candidate bridge

"""Convert a neural supervisor's recommendation into an autotune candidate.

The differentiable supervisor learns a policy that emits a
:class:`~scpn_phase_orchestrator.nn.supervisor._types.SupervisorAction` — a global
coupling delta, a global damping delta, and per-layer coupling deltas. The
auditable candidate track scores, attributes, certifies, and bundles a
:class:`~scpn_phase_orchestrator.autotune.reward.KnobPolicyCandidate`. This module
is the bridge between the two, so a learned policy's recommendation feeds the same
review-only evidence pipeline as an offline replay search.

The mapping is by control meaning, not by position: the supervisor's global
coupling delta moves the candidate's ``K`` knob, its damping delta moves ``zeta``,
and its per-layer coupling deltas become the candidate's ``channel_weights``. The
knobs the supervisor does not control — ``alpha``, ``Psi`` and
``cross_channel_gains`` — are carried through unchanged from a caller-supplied
base candidate. The deltas are applied relative to that base, which defaults to
the zero candidate. The bridge runs the policy deterministically (its mean action)
and actuates nothing.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.autotune.reward import KnobPolicyCandidate
from scpn_phase_orchestrator.nn.supervisor._types import (
    KuramotoSupervisorScenario,
    SupervisorAction,
)
from scpn_phase_orchestrator.nn.supervisor.policy import (
    DifferentiableSupervisorPolicy,
)

__all__ = [
    "supervisor_action_to_candidate",
    "supervisor_policy_to_candidate",
]


def _add_global(
    base_value: float | NDArray[np.float64], delta: float
) -> float | NDArray[np.float64]:
    """Add a global scalar delta to a scalar or per-oscillator base knob."""
    if isinstance(base_value, (int, float)) and not isinstance(base_value, bool):
        return float(base_value) + delta
    return np.asarray(base_value, dtype=float) + delta


def supervisor_action_to_candidate(
    action: SupervisorAction,
    *,
    base: KnobPolicyCandidate | None = None,
) -> KnobPolicyCandidate:
    """Map a supervisor action onto a candidate, relative to a base candidate.

    Parameters
    ----------
    action : SupervisorAction
        The supervisor control action: a global coupling delta, a global damping
        delta, and per-layer coupling deltas.
    base : KnobPolicyCandidate | None
        The candidate the deltas are applied to. The ``alpha``, ``Psi`` and
        ``cross_channel_gains`` knobs are carried through from it unchanged.
        Defaults to the zero candidate.

    Returns
    -------
    KnobPolicyCandidate
        A candidate with ``K`` and ``zeta`` advanced by the global deltas and
        ``channel_weights`` advanced by the per-layer coupling deltas.
    """
    reference = base if base is not None else KnobPolicyCandidate()
    delta_coupling = float(action.delta_K_global)
    delta_damping = float(action.delta_zeta_global)
    layer_deltas = np.asarray(action.delta_K_layers, dtype=float).ravel()
    base_weights = reference.channel_weights
    channel_weights = tuple(
        (base_weights[index] if index < len(base_weights) else 0.0)
        + float(layer_deltas[index])
        for index in range(len(layer_deltas))
    )
    return KnobPolicyCandidate(
        K=_add_global(reference.K, delta_coupling),
        alpha=reference.alpha,
        zeta=_add_global(reference.zeta, delta_damping),
        Psi=reference.Psi,
        channel_weights=channel_weights,
        cross_channel_gains=reference.cross_channel_gains,
    )


def supervisor_policy_to_candidate(
    policy: DifferentiableSupervisorPolicy,
    scenario: KuramotoSupervisorScenario,
    *,
    base: KnobPolicyCandidate | None = None,
) -> KnobPolicyCandidate:
    """Run a supervisor policy deterministically and map its action to a candidate.

    Parameters
    ----------
    policy : DifferentiableSupervisorPolicy
        The learned supervisor policy. It is evaluated for its deterministic mean
        action; no stochastic sample is drawn.
    scenario : KuramotoSupervisorScenario
        The scenario the policy is evaluated on.
    base : KnobPolicyCandidate | None
        The base candidate the deltas are applied to. Defaults to the zero
        candidate.

    Returns
    -------
    KnobPolicyCandidate
        The candidate equivalent of the policy's recommendation.
    """
    return supervisor_action_to_candidate(policy(scenario), base=base)
