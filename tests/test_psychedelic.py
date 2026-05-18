# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Psychedelic simulation protocol tests

from __future__ import annotations

from typing import get_type_hints

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.psychedelic import (
    entropy_from_phases,
    reduce_coupling,
    simulate_psychedelic_trajectory,
)
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from tests.typing_contracts import assert_precise_ndarray_hint


def test_public_array_contracts_are_parameterised():
    hints = (
        get_type_hints(reduce_coupling)["knm"],
        get_type_hints(reduce_coupling)["return"],
        get_type_hints(entropy_from_phases)["phases"],
        get_type_hints(simulate_psychedelic_trajectory)["phases"],
        get_type_hints(simulate_psychedelic_trajectory)["omegas"],
        get_type_hints(simulate_psychedelic_trajectory)["knm"],
        get_type_hints(simulate_psychedelic_trajectory)["alpha"],
    )

    for hint in hints:
        assert_precise_ndarray_hint(hint)
        assert "float64" in str(hint)


def test_reduce_coupling_zero_factor_keeps_original():
    knm = np.ones((5, 5))
    result = reduce_coupling(knm, 0.0)
    np.testing.assert_array_equal(result, knm)


def test_reduce_coupling_full_reduction_gives_zero():
    knm = np.ones((5, 5)) * 3.0
    result = reduce_coupling(knm, 1.0)
    np.testing.assert_allclose(result, 0.0, atol=1e-15)


def test_reduce_coupling_half():
    knm = np.eye(4) * 2.0
    result = reduce_coupling(knm, 0.5)
    np.testing.assert_allclose(result, np.eye(4), atol=1e-15)


@pytest.mark.parametrize(
    ("knm", "match"),
    [
        (np.ones(4), "knm must be a finite 2-D matrix"),
        (np.array([[0.0, np.nan], [1.0, 0.0]]), "knm"),
    ],
)
def test_reduce_coupling_rejects_invalid_coupling_matrix(knm, match):
    with pytest.raises(ValueError, match=match):
        reduce_coupling(knm, 0.5)


@pytest.mark.parametrize("reduction_factor", [-0.1, 1.1, np.nan, True])
def test_reduce_coupling_rejects_invalid_reduction_factor(reduction_factor):
    with pytest.raises((TypeError, ValueError), match="reduction_factor"):
        reduce_coupling(np.eye(3), reduction_factor)


def test_entropy_uniform_phases_high():
    """Uniformly distributed phases should have near-maximal entropy."""
    phases = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    ent = entropy_from_phases(phases)
    max_entropy = np.log(36)  # 36 bins
    assert ent > 0.9 * max_entropy


def test_entropy_concentrated_phases_low():
    """All phases at same value → only 1 bin occupied → entropy = 0."""
    phases = np.full(100, 1.0)
    ent = entropy_from_phases(phases)
    assert ent == 0.0


def test_entropy_empty_phases():
    assert entropy_from_phases(np.array([])) == 0.0


@pytest.mark.parametrize("phases", [np.array([[0.0, 1.0]]), np.array([0.0, np.inf])])
def test_entropy_rejects_non_vector_or_non_finite_phases(phases):
    with pytest.raises(ValueError, match="phases"):
        entropy_from_phases(phases)


@pytest.mark.parametrize("n_bins", [0, 1, False, 18.5])
def test_entropy_rejects_invalid_bin_counts(n_bins):
    with pytest.raises((TypeError, ValueError), match="n_bins"):
        entropy_from_phases(np.linspace(0.0, 1.0, 8), n_bins=n_bins)


def test_simulate_trajectory_returns_correct_length():
    n = 10
    engine = UPDEEngine(n, dt=0.01)
    rng = np.random.default_rng(7)
    phases = rng.uniform(0, 2 * np.pi, n)
    omegas = rng.normal(1.0, 0.1, n)
    knm = np.ones((n, n)) * 0.5
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))
    schedule = [0.0, 0.3, 0.6, 0.9]

    results = simulate_psychedelic_trajectory(
        engine,
        phases,
        omegas,
        knm,
        alpha,
        schedule,
        n_steps_per_level=50,
    )
    assert len(results) == 4
    for rec in results:
        assert "R" in rec
        assert "entropy" in rec
        assert "chimera_index" in rec
        assert rec["phases"].shape == (n,)


def test_trajectory_entropy_increases_with_coupling_reduction():
    """Entropic brain hypothesis: lower coupling → higher entropy (on average)."""
    n = 30
    engine = UPDEEngine(n, dt=0.01)
    # Start synchronised
    phases = np.zeros(n)
    omegas = np.ones(n)
    knm = np.ones((n, n)) * 2.0
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))
    schedule = [0.0, 0.5, 0.9]

    results = simulate_psychedelic_trajectory(
        engine,
        phases,
        omegas,
        knm,
        alpha,
        schedule,
        n_steps_per_level=200,
    )
    # With strong coupling reduction, entropy should not decrease overall
    assert results[-1]["entropy"] >= results[0]["entropy"] - 0.5


def test_simulate_trajectory_rejects_mismatched_runtime_shapes():
    n = 4
    engine = UPDEEngine(n, dt=0.01)
    phases = np.linspace(0.0, 1.0, n)
    omegas = np.ones(n - 1)
    knm = np.eye(n)
    alpha = np.zeros((n, n))
    with pytest.raises(ValueError, match="omegas"):
        simulate_psychedelic_trajectory(
            engine,
            phases,
            omegas,
            knm,
            alpha,
            [0.0],
            n_steps_per_level=1,
        )


def test_simulate_trajectory_rejects_invalid_schedule_and_step_count():
    n = 4
    engine = UPDEEngine(n, dt=0.01)
    phases = np.linspace(0.0, 1.0, n)
    omegas = np.ones(n)
    knm = np.eye(n)
    alpha = np.zeros((n, n))
    with pytest.raises(ValueError, match="reduction_schedule"):
        simulate_psychedelic_trajectory(
            engine,
            phases,
            omegas,
            knm,
            alpha,
            [0.0, 1.2],
            n_steps_per_level=1,
        )
    with pytest.raises((TypeError, ValueError), match="n_steps_per_level"):
        simulate_psychedelic_trajectory(
            engine,
            phases,
            omegas,
            knm,
            alpha,
            [0.0],
            n_steps_per_level=True,
        )


class TestPsychedelicPipelineWiring:
    """Pipeline: UPDEEngine → psychedelic simulation → R + entropy."""

    def test_psychedelic_sim_produces_valid_metrics(self):
        """simulate_psychedelic_trajectory uses engine internally and
        produces R∈[0,1] and entropy≥0 at each coupling reduction level."""
        n = 8
        engine = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = np.ones((n, n)) * 0.5
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        results = simulate_psychedelic_trajectory(
            engine,
            phases,
            omegas,
            knm,
            alpha,
            reduction_schedule=[0.0, 0.5],
            n_steps_per_level=30,
        )
        for rec in results:
            assert 0.0 <= rec["R"] <= 1.0
            assert rec["entropy"] >= 0.0
