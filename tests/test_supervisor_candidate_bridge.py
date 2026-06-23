# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — neural-supervisor-to-candidate bridge tests

"""Tests for mapping a neural supervisor recommendation onto an autotune candidate.

The mapping is checked against the default zero base, a populated base (delta
application and carried-through knobs), an array-valued base coupling knob, and
the deterministic policy runner over a constructed supervisor policy.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("equinox")

from scpn_phase_orchestrator.autotune.reward import KnobPolicyCandidate
from scpn_phase_orchestrator.nn.supervisor._types import (
    DifferentiableSupervisorConfig,
    KuramotoSupervisorScenario,
    SupervisorAction,
)
from scpn_phase_orchestrator.nn.supervisor.candidate_bridge import (
    supervisor_action_to_candidate,
    supervisor_policy_to_candidate,
)
from scpn_phase_orchestrator.nn.supervisor.policy import (
    DifferentiableSupervisorPolicy,
)


def _action(
    delta_k: float = 0.3,
    delta_zeta: float = -0.1,
    layers: tuple[float, ...] = (0.05, 0.02),
) -> SupervisorAction:
    return SupervisorAction(
        delta_K_global=jnp.array(delta_k),
        delta_zeta_global=jnp.array(delta_zeta),
        delta_K_layers=jnp.array(layers),
        value_estimate=jnp.array(0.0),
    )


def _scenario() -> KuramotoSupervisorScenario:
    return KuramotoSupervisorScenario(
        phases=jnp.array([0.0, 0.1, 2.7, 3.1]),
        omegas=jnp.array([0.04, 0.03, -0.03, -0.04]),
        base_K=jnp.full((4, 4), 0.03) - jnp.eye(4) * 0.03,
        good_mask=jnp.array([1.0, 1.0, 0.0, 0.0]),
        bad_mask=jnp.array([0.0, 0.0, 1.0, 1.0]),
        dt=0.02,
        inner_steps=4,
        horizon=3,
    )


def test_action_to_candidate_with_zero_base() -> None:
    candidate = supervisor_action_to_candidate(_action())
    assert pytest.approx(0.3) == candidate.K
    assert candidate.zeta == pytest.approx(-0.1)
    assert candidate.channel_weights[0] == pytest.approx(0.05)
    assert candidate.channel_weights[1] == pytest.approx(0.02)
    assert candidate.alpha == 0.0
    assert candidate.Psi == 0.0
    assert candidate.cross_channel_gains == ()


def test_action_to_candidate_applies_deltas_to_base() -> None:
    base = KnobPolicyCandidate(
        K=1.0,
        zeta=0.5,
        alpha=0.7,
        Psi=0.2,
        channel_weights=(0.1,),
        cross_channel_gains=(0.3,),
    )
    candidate = supervisor_action_to_candidate(_action(), base=base)
    assert pytest.approx(1.3) == candidate.K
    assert candidate.zeta == pytest.approx(0.4)
    # Carried through unchanged.
    assert candidate.alpha == 0.7
    assert candidate.Psi == 0.2
    assert candidate.cross_channel_gains == (0.3,)
    # base weight[0] + layer[0]; index 1 beyond base -> 0 + layer[1].
    assert candidate.channel_weights[0] == pytest.approx(0.15)
    assert candidate.channel_weights[1] == pytest.approx(0.02)


def test_action_to_candidate_with_array_coupling_base() -> None:
    base = KnobPolicyCandidate(K=np.array([1.0, 2.0]))
    candidate = supervisor_action_to_candidate(_action(delta_k=0.5), base=base)
    assert np.allclose(np.asarray(candidate.K, dtype=float), [1.5, 2.5])


def test_policy_to_candidate_runs_deterministically() -> None:
    config = DifferentiableSupervisorConfig(n_oscillators=4, n_layer_controls=2)
    policy = DifferentiableSupervisorPolicy(config, key=jax.random.PRNGKey(1))
    scenario = _scenario()
    candidate = supervisor_policy_to_candidate(policy, scenario)
    assert isinstance(candidate, KnobPolicyCandidate)
    assert len(candidate.channel_weights) == 2
    assert isinstance(candidate.K, float)
    assert isinstance(candidate.zeta, float)
    # The policy is deterministic: the same scenario yields the same candidate.
    again = supervisor_policy_to_candidate(policy, scenario)
    assert pytest.approx(again.K) == candidate.K
    assert candidate.channel_weights == pytest.approx(again.channel_weights)
