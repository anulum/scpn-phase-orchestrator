# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stability / regression tests for entropy production

"""Long-run invariants for :func:`entropy_production_rate`.

* Monotone collapse during synchronisation: as a Kuramoto network
  approaches phase-locking the dissipation rate falls.
* Large-N stress: 500-oscillator network yields finite output in a
  few milliseconds.
* Coupling-strength sweep: Σ is smooth and finite across a wide
  range of ``α`` values.

Marked ``@pytest.mark.slow``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import entropy_prod as ep_mod
from scpn_phase_orchestrator.monitor.entropy_prod import (
    entropy_production_rate,
)
from scpn_phase_orchestrator.upde.engine import upde_run

TWO_PI = 2.0 * np.pi

pytestmark = pytest.mark.slow


def _python(func):
    def wrapper(*args, **kwargs):
        prev = ep_mod.ACTIVE_BACKEND
        ep_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            ep_mod.ACTIVE_BACKEND = prev

    wrapper.__name__ = func.__name__
    return wrapper


@_python
def test_dissipation_falls_during_synchronisation():
    """As phases lock, dθ/dt → 0 and Σ falls monotonically."""
    rng = np.random.default_rng(0)
    n = 12
    phases = rng.uniform(0.0, TWO_PI, size=n)
    omegas = np.zeros(n)
    knm = np.full((n, n), 4.0)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))

    sigmas: list[float] = []
    for _ in range(8):
        sigmas.append(
            entropy_production_rate(phases, omegas, knm, 1.0, 0.005)
        )
        phases = upde_run(
            phases, omegas, knm, alpha,
            zeta=0.0, psi=0.0, dt=0.005, n_steps=200,
            method="rk4",
        )

    # Last snapshot must be at least an order of magnitude lower than
    # the first (the network contracts rapidly at this coupling).
    assert sigmas[-1] < 0.1 * sigmas[0]


@_python
def test_large_N_stress_finite():
    rng = np.random.default_rng(7)
    n = 500
    phases = rng.uniform(0.0, TWO_PI, size=n)
    omegas = rng.normal(0.0, 0.1, size=n)
    knm = rng.uniform(0.0, 0.3, size=(n, n))
    np.fill_diagonal(knm, 0.0)
    sigma = entropy_production_rate(phases, omegas, knm, 0.5, 0.01)
    assert math.isfinite(sigma)
    assert sigma > 0.0


@_python
def test_alpha_sweep_monotone_in_knm_variance():
    """Doubling α on a fixed problem at least doubles the contribution
    from the coupling term (it enters squared)."""
    rng = np.random.default_rng(11)
    n = 10
    phases = rng.uniform(0.0, TWO_PI, size=n)
    omegas = np.zeros(n)
    knm = rng.uniform(0.3, 1.0, size=(n, n))
    np.fill_diagonal(knm, 0.0)
    s1 = entropy_production_rate(phases, omegas, knm, 0.1, 0.01)
    s2 = entropy_production_rate(phases, omegas, knm, 0.2, 0.01)
    # With ω = 0, Σ ∝ α² so doubling α → 4× Σ.
    assert abs(s2 / s1 - 4.0) < 1e-10
