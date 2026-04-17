# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PhaseSINDy tests

"""Coverage for the Phase-SINDy symbolic coupling discoverer.

The pre-existing single happy-path case exercised recovery on a clean
two-oscillator signal. This suite adds the edge and error paths Gemini
S6 flagged as missing.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.autotune.sindy import PhaseSINDy


def _simulate(
    phases_init: np.ndarray,
    omega: np.ndarray,
    K: np.ndarray,
    dt: float,
    steps: int,
) -> np.ndarray:
    n = len(omega)
    phases = np.zeros((steps, n))
    phases[0] = phases_init
    for t in range(1, steps):
        d = omega.copy()
        for i in range(n):
            for j in range(n):
                if i != j:
                    d[i] += K[i, j] * np.sin(phases[t - 1, j] - phases[t - 1, i])
        phases[t] = (phases[t - 1] + dt * d) % (2 * np.pi)
    return phases


def test_sindy_recovery():
    """Baseline: two oscillators, bidirectional coupling, clean recovery."""
    n = 2
    dt = 0.05
    t_max = 20.0
    steps = int(t_max / dt)

    omega = np.array([1.5, 2.0])
    K = np.array([[0.0, 0.5], [0.5, 0.0]])

    phases = np.zeros((steps, n))
    phases[0] = [0.0, 1.0]
    for t in range(1, steps):
        d0 = omega[0] + K[0, 1] * np.sin(phases[t - 1, 1] - phases[t - 1, 0])
        d1 = omega[1] + K[1, 0] * np.sin(phases[t - 1, 0] - phases[t - 1, 1])
        phases[t] = (phases[t - 1] + dt * np.array([d0, d1])) % (2 * np.pi)

    sindy = PhaseSINDy(threshold=0.1)
    coeffs = sindy.fit(phases, dt)

    assert abs(coeffs[0][0] - 1.5) < 0.01
    assert abs(coeffs[0][1] - 0.5) < 0.01
    assert abs(coeffs[1][0] - 2.0) < 0.01
    assert abs(coeffs[1][1] - 0.5) < 0.01

    equations = sindy.get_equations()
    assert "1.5000 * 1" in equations[0]
    assert "0.5000 * sin(theta_1 - theta_0)" in equations[0]


def test_sindy_zero_coupling():
    """Uncoupled oscillators: SINDy must return K ≈ 0 under the threshold."""
    dt = 0.05
    steps = 400
    omega = np.array([1.0, 1.3])
    K = np.zeros((2, 2))
    phases = _simulate(np.array([0.0, 0.5]), omega, K, dt, steps)

    sindy = PhaseSINDy(threshold=0.05)
    coeffs = sindy.fit(phases, dt)

    # Frequency term close to ground truth, coupling term sparsified to ~0.
    assert abs(coeffs[0][0] - 1.0) < 0.02
    assert abs(coeffs[1][0] - 1.3) < 0.02
    # Coupling slot should either be missing or near-zero.
    if coeffs[0].size > 1:
        assert abs(coeffs[0][1]) < 0.05
    if coeffs[1].size > 1:
        assert abs(coeffs[1][1]) < 0.05


def test_sindy_large_n_stability():
    """N=5 network, moderate coupling, no blow-up."""
    n = 5
    dt = 0.02
    steps = 500
    rng = np.random.default_rng(12)
    omega = rng.uniform(0.8, 1.5, n)
    K = 0.3 * (np.ones((n, n)) - np.eye(n))  # uniform, no self
    phases = _simulate(rng.uniform(0, 2 * np.pi, n), omega, K, dt, steps)

    sindy = PhaseSINDy(threshold=0.05)
    coeffs = sindy.fit(phases, dt)

    assert len(coeffs) == n
    for i in range(n):
        assert np.all(np.isfinite(coeffs[i]))
        # Natural frequency term roughly recovers
        assert abs(coeffs[i][0] - omega[i]) < 0.1


def test_sindy_rejects_zero_max_iter():
    """Constructor guards max_iter — regression for U1b validation."""
    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        PhaseSINDy(max_iter=0)


def test_sindy_rejects_negative_threshold():
    """Negative threshold would invert the sparsification logic."""
    with pytest.raises(ValueError, match="threshold must be non-negative"):
        PhaseSINDy(threshold=-0.1)


def test_sindy_threshold_sparsifies_weak_terms():
    """High threshold wipes small coupling coefficients below it."""
    dt = 0.05
    steps = 400
    omega = np.array([1.0, 1.2])
    K = np.array([[0.0, 0.04], [0.04, 0.0]])  # below threshold
    phases = _simulate(np.array([0.0, 0.1]), omega, K, dt, steps)

    sindy_high = PhaseSINDy(threshold=0.2)
    coeffs_high = sindy_high.fit(phases, dt)
    sindy_low = PhaseSINDy(threshold=0.005)
    coeffs_low = sindy_low.fit(phases, dt)

    # High threshold (0.2) is above the true coupling (0.04), so the
    # STLSQ sparsifier must zero the coupling slot exactly.
    assert coeffs_high[0].size > 1, "fitted coefficients must include coupling term"
    assert coeffs_high[0][1] == 0.0, (
        f"coupling term should be sparsified under threshold 0.2, "
        f"got {coeffs_high[0][1]:.4f}"
    )
    # Low threshold (0.005) leaves the coupling intact; within regression
    # noise it recovers the ground-truth 0.04 to 0.005 precision.
    assert coeffs_low[0].size > 1
    assert abs(coeffs_low[0][1] - 0.04) < 0.005, (
        f"low-threshold fit should recover K_01 ≈ 0.04, got {coeffs_low[0][1]:.4f}"
    )


def test_sindy_equations_dump_format():
    """get_equations() returns one string per node with the 'theta_' marker."""
    n = 3
    dt = 0.05
    steps = 200
    rng = np.random.default_rng(3)
    omega = rng.uniform(0.8, 1.2, n)
    K = 0.2 * (np.ones((n, n)) - np.eye(n))
    phases = _simulate(rng.uniform(0, 2 * np.pi, n), omega, K, dt, steps)

    sindy = PhaseSINDy(threshold=0.05)
    sindy.fit(phases, dt)
    eqs = sindy.get_equations()

    assert len(eqs) == n
    for node_i, eq in enumerate(eqs):
        assert isinstance(eq, str)
        assert f"d(theta_{node_i})/dt" in eq


def test_sindy_empty_phases_does_not_crash():
    """Empty trajectory: fit must not crash and must return finite output.

    The specific coefficient dimensionality is an implementation detail —
    the contract covered here is "no NaN, no exception, one entry per
    oscillator".
    """
    sindy = PhaseSINDy()
    coeffs = sindy.fit(np.zeros((0, 2)), 0.01)
    assert isinstance(coeffs, list)
    assert len(coeffs) == 2
    for c in coeffs:
        assert np.all(np.isfinite(c))


# Pipeline wiring: PhaseSINDy feeds the auto-tune pipeline (see
# autotune/pipeline.py). These tests pin the contract every consumer
# relies on: zero-coupling must sparsify, the threshold must suppress
# weak terms, and invalid constructor arguments must fail loudly.
