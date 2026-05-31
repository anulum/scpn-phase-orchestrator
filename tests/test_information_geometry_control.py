# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Information geometry control tests

"""Tests for JAX-compatible information-geometry control proposals."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.supervisor.information_geometry import (
    InformationGeometryControlProposal,
    propose_information_geometry_control,
)


def test_fisher_rao_and_wasserstein_distances_depend_on_distribution_ordering() -> None:
    short_shift = propose_information_geometry_control(
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        max_step=1.0,
    )
    long_shift = propose_information_geometry_control(
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        max_step=1.0,
    )

    assert short_shift.wasserstein_distance == pytest.approx(1.0)
    assert long_shift.wasserstein_distance == pytest.approx(2.0)
    assert long_shift.wasserstein_distance > short_shift.wasserstein_distance
    assert short_shift.fisher_rao_distance == pytest.approx(math.pi)
    assert long_shift.fisher_rao_distance == pytest.approx(math.pi)


def test_proposal_hash_is_deterministic_for_identical_inputs() -> None:
    proposal_a = propose_information_geometry_control(
        [0.2, 0.8, 0.0],
        [0.5, 0.3, 0.2],
        max_step=0.4,
    )
    proposal_b = propose_information_geometry_control(
        (0.2, 0.8, 0.0),
        (0.5, 0.3, 0.2),
        max_step=0.4,
    )

    assert proposal_a.proposal_hash == proposal_b.proposal_hash


def test_audit_record_is_json_safe_without_nan_or_inf() -> None:
    proposal = propose_information_geometry_control(
        [0.2, 0.5, 0.3],
        [0.4, 0.2, 0.4],
        coupling_gradient=[-0.1, 0.0, 0.1],
        max_step=0.05,
        knob="alpha",
    )
    payload = proposal.to_audit_record()

    json.dumps(payload, allow_nan=False)
    assert not any(_contains_non_finite(value) for value in payload.values())
    assert payload["backend"] == "numpy_jax_compatible_information_geometry"
    assert (
        payload["claim_boundary"] == "information_geometry_control_not_live_actuation"
    )


def _contains_non_finite(value: object) -> bool:
    if isinstance(value, float):
        return not (math.isfinite(value) and isinstance(value, float))
    if isinstance(value, (int, float, str, bool)) or value is None:
        return False
    if isinstance(value, list):
        return any(_contains_non_finite(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_non_finite(item) for item in value.values())
    return False


def test_current_and_target_distributions_are_not_mutated() -> None:
    current = np.array([0.15, 0.75, 0.10], dtype=np.float64)
    target = np.array([0.20, 0.70, 0.10], dtype=np.float64)
    current_copy = current.copy()
    target_copy = target.copy()

    _ = propose_information_geometry_control(current, target, max_step=0.25)

    assert np.array_equal(current, current_copy)
    assert np.array_equal(target, target_copy)


def test_natural_gradient_is_clipped_to_max_step_norm() -> None:
    proposal = propose_information_geometry_control(
        [0.5, 0.5],
        [0.1, 0.9],
        coupling_gradient=[10.0, 10.0],
        max_step=0.05,
    )

    assert proposal.natural_gradient_norm == pytest.approx(0.05)
    assert np.linalg.norm(proposal.state.tangent_vector) <= 0.05 + 1e-12


def test_invalid_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        propose_information_geometry_control(
            [0.5, 0.4, -0.1],
            [0.2, 0.8, 0.0],
            max_step=0.1,
        )

    with pytest.raises(ValueError, match="must match"):
        propose_information_geometry_control(
            [0.5, 0.4],
            [0.2, 0.8, 0.0],
            max_step=0.1,
        )

    with pytest.raises(ValueError, match="coupling_gradient"):
        propose_information_geometry_control(
            [0.5, 0.5],
            [0.5, 0.5],
            coupling_gradient=[0.1, 0.2, 0.3],
            max_step=0.1,
        )

    with pytest.raises(ValueError, match="max_step"):
        propose_information_geometry_control(
            [0.5, 0.5],
            [0.2, 0.8],
            max_step=0.0,
        )


@pytest.mark.parametrize(
    ("current_distribution", "target_distribution", "coupling_gradient", "message"),
    [
        ([True, 1.0], [0.5, 0.5], None, "current_distribution"),
        ([np.bool_(True), 1.0], [0.5, 0.5], None, "current_distribution"),
        ([0.5, 0.5], [False, 1.0], None, "target_distribution"),
        ([0.5, 0.5], [0.5, 0.5], [True, 0.0], "coupling_gradient"),
    ],
)
def test_boolean_alias_inputs_fail_closed(
    current_distribution: list[object],
    target_distribution: list[object],
    coupling_gradient: list[object] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        propose_information_geometry_control(
            current_distribution,
            target_distribution,
            coupling_gradient=coupling_gradient,
            max_step=0.1,
        )


def test_boolean_alias_max_step_fails_closed() -> None:
    with pytest.raises(ValueError, match="max_step"):
        propose_information_geometry_control(
            [0.5, 0.5],
            [0.2, 0.8],
            max_step=np.bool_(True),
        )


def test_control_actions_are_review_only_and_required_boundary_fields() -> None:
    proposal = propose_information_geometry_control(
        [0.6, 0.4],
        [0.2, 0.8],
        max_step=0.15,
        knob="zeta",
        scope="layer_0",
        coupling_gradient=[0.1, -0.2],
    )

    assert isinstance(proposal, InformationGeometryControlProposal)
    assert proposal.claim_boundary == "information_geometry_control_not_live_actuation"
    assert proposal.non_actuating is True
    assert proposal.execution_disabled is True
    assert proposal.proposal_hash
    assert all(
        isinstance(action, ControlAction) for action in proposal.action_proposals
    )
    assert len(proposal.action_proposals) == 1
    assert proposal.action_proposals[0].knob == "zeta"
    assert proposal.action_proposals[0].scope == "layer_0"
    assert abs(proposal.action_proposals[0].value) <= 0.15 + 1e-12


def test_metrics_are_finite_in_state_and_proposal() -> None:
    proposal = propose_information_geometry_control(
        [0.15, 0.45, 0.40],
        [0.55, 0.25, 0.20],
        coupling_gradient=[0.01, -0.02, 0.01],
        max_step=0.2,
    )
    state = proposal.state

    assert np.isfinite(proposal.fisher_rao_distance)
    assert np.isfinite(proposal.wasserstein_distance)
    assert np.isfinite(proposal.natural_gradient_norm)
    assert np.isfinite(proposal.curvature_proxy)
    assert np.isfinite(state.geodesic_length)
    assert np.isfinite(state.curvature_proxy)
    assert np.isfinite(state.metric_tensor).all()
    assert np.isfinite(state.tangent_vector).all()


def test_explicit_jax_backend_matches_numpy_information_geometry_contract() -> None:
    numpy_proposal = propose_information_geometry_control(
        [0.16, 0.27, 0.18, 0.39],
        [0.21, 0.23, 0.27, 0.29],
        coupling_gradient=[0.05, -0.02, 0.04, -0.01],
        max_step=0.2,
        backend="numpy",
    )
    jax_proposal = propose_information_geometry_control(
        [0.16, 0.27, 0.18, 0.39],
        [0.21, 0.23, 0.27, 0.29],
        coupling_gradient=[0.05, -0.02, 0.04, -0.01],
        max_step=0.2,
        backend="jax",
    )

    assert jax_proposal.backend == "jax_native_information_geometry"
    assert jax_proposal.claim_boundary == numpy_proposal.claim_boundary
    assert jax_proposal.non_actuating is True
    assert jax_proposal.execution_disabled is True
    assert jax_proposal.fisher_rao_distance == pytest.approx(
        numpy_proposal.fisher_rao_distance
    )
    assert jax_proposal.wasserstein_distance == pytest.approx(
        numpy_proposal.wasserstein_distance
    )
    assert jax_proposal.natural_gradient_norm == pytest.approx(
        numpy_proposal.natural_gradient_norm
    )
    assert jax_proposal.curvature_proxy == pytest.approx(numpy_proposal.curvature_proxy)
    np.testing.assert_allclose(
        jax_proposal.state.metric_tensor,
        numpy_proposal.state.metric_tensor,
    )
    np.testing.assert_allclose(
        jax_proposal.state.tangent_vector,
        numpy_proposal.state.tangent_vector,
    )


def test_unknown_information_geometry_backend_fails_closed() -> None:
    with pytest.raises(ValueError, match="backend"):
        propose_information_geometry_control(
            [0.5, 0.5],
            [0.25, 0.75],
            max_step=0.1,
            backend="gpu_magic",
        )
