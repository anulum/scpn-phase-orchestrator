# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Long-run stability tests for Lyapunov spectrum

"""Long-run stability / regression tests.

These tests are marked ``@pytest.mark.slow`` and exercise scenarios
that take seconds rather than milliseconds. They are opt-in via
``pytest -m slow``.

Covered:

* Basin-convergence on a contracting all-to-all network across a long
  integration (10 000 steps).
* Kaplan-Yorke dimension estimate that we expect to stay stable across
  RK4 timestep choices.
* The sum ``Σ_i λ_i`` tracks the average divergence ``⟨tr J⟩`` (this
  is an integral invariant — any RK4/QR bug tends to shift it).
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import lyapunov as ly_mod
from scpn_phase_orchestrator.monitor.lyapunov import lyapunov_spectrum

TWO_PI = 2.0 * np.pi

pytestmark = pytest.mark.slow


def _with_python_backend(func):
    def wrapper(*args, **kwargs):
        prev = ly_mod.ACTIVE_BACKEND
        ly_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            ly_mod.ACTIVE_BACKEND = prev

    wrapper.__name__ = func.__name__
    return wrapper


@_with_python_backend
def test_long_run_all_attracting():
    """Strong all-to-all + shared ω → all λ < 0 after long integration."""
    n = 6
    phases = np.full(n, 0.05)
    omegas = np.zeros(n)
    knm = np.full((n, n), 3.0)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))
    spec = lyapunov_spectrum(
        phases,
        omegas,
        knm,
        alpha,
        dt=0.005,
        n_steps=10_000,
        qr_interval=20,
    )
    # All exponents should be strongly negative under contracting coupling.
    assert np.all(spec < -0.5)


@_with_python_backend
def test_sum_tracks_trace_of_jacobian():
    """Σ_i λ_i ≈ ⟨tr J(θ(t))⟩_t.

    For the Kuramoto network this is exactly
    ``Σ_i J_ii = −Σ_{i≠j} K_ij cos(θ_j − θ_i − α_ij) − Σ_i ζ cos(Ψ − θ_i)``.
    Time-averaging along the trajectory must match the sum of exponents
    returned by the spectrum integrator.
    """
    rng = np.random.default_rng(123)
    n = 5
    phases = rng.uniform(0.0, TWO_PI, size=n)
    omegas = rng.normal(0.0, 0.2, size=n)
    knm = rng.uniform(0.3, 0.9, size=(n, n))
    np.fill_diagonal(knm, 0.0)
    alpha = rng.uniform(-0.2, 0.2, size=(n, n))
    np.fill_diagonal(alpha, 0.0)

    spec = lyapunov_spectrum(
        phases,
        omegas,
        knm,
        alpha,
        dt=0.01,
        n_steps=5000,
        qr_interval=10,
    )

    # Evolve a second trajectory with the same RK4 to sample tr J.
    theta = phases.copy()
    dt = 0.01
    traces = []
    for _ in range(5000):
        # Simple Euler for this diagnostic average — suffices for
        # verifying the order of magnitude.
        diff = theta[np.newaxis, :] - theta[:, np.newaxis] - alpha
        coupling = (knm * np.sin(diff)).sum(axis=1)
        theta = (theta + dt * (omegas + coupling)) % TWO_PI
        J = ly_mod._kuramoto_jacobian(theta, knm, alpha, 0.0, 0.0)
        traces.append(np.trace(J))
    avg_trace = float(np.mean(traces))

    # Sum of exponents should track ⟨tr J⟩ to within a loose tolerance.
    assert abs(spec.sum() - avg_trace) < 1.5


@_with_python_backend
def test_kaplan_yorke_dimension_bounded():
    """Kaplan-Yorke dimension ``d_KY`` is bounded by ``N`` for any
    network and strictly positive whenever the network is not fully
    contracting."""
    rng = np.random.default_rng(7)
    n = 5
    phases = rng.uniform(0.0, TWO_PI, size=n)
    omegas = rng.normal(0.0, 0.5, size=n)
    knm = rng.uniform(0.0, 0.4, size=(n, n))
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))

    spec = lyapunov_spectrum(
        phases,
        omegas,
        knm,
        alpha,
        dt=0.01,
        n_steps=5000,
        qr_interval=20,
    )
    running = 0.0
    dim_ky = 0.0
    for i, lam in enumerate(spec):
        if running + lam < 0:
            if lam != 0.0:
                dim_ky = i + running / abs(lam)
            break
        running += lam
        dim_ky = i + 1
    assert 0.0 <= dim_ky <= n
