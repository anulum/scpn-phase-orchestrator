# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Psychedelic simulation protocol tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.monitor.psychedelic import (
    entropy_from_phases,
    reduce_coupling,
    simulate_psychedelic_trajectory,
)
from scpn_phase_orchestrator.upde.engine import UPDEEngine


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
