# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Long-run stability for upde_run

"""Long-run stability / regression invariants for ``upde_run``.

* Kuramoto order parameter ``R`` reaches ≥ 0.95 on a strongly coupled
  network after several thousand integration steps.
* A synchronised initial condition with zero natural frequencies
  remains at ``R = 1`` indefinitely (no numerical drift).
* RK45 never blows up: its output remains bounded across 10 000
  steps on a randomly coupled network.

Marked ``@pytest.mark.slow`` and opt-in via ``pytest -m slow``.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde import engine as eng_mod
from scpn_phase_orchestrator.upde.engine import upde_run

TWO_PI = 2.0 * np.pi

pytestmark = pytest.mark.slow


def _python(func):
    def wrapper(*args, **kwargs):
        prev = eng_mod.ACTIVE_BACKEND
        eng_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            eng_mod.ACTIVE_BACKEND = prev

    wrapper.__name__ = func.__name__
    return wrapper


@_python
def test_strong_coupling_reaches_R_above_0p95():
    """After long integration on strong all-to-all coupling,
    R = |<exp(iθ)>| should exceed 0.95."""
    n = 8
    rng = np.random.default_rng(0)
    phases = rng.uniform(0.0, TWO_PI, size=n)
    omegas = np.zeros(n)
    knm = np.full((n, n), 4.0)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))
    out = upde_run(
        phases, omegas, knm, alpha,
        zeta=0.0, psi=0.0, dt=0.005, n_steps=3000,
        method="rk4",
    )
    R = float(np.abs(np.mean(np.exp(1j * out))))
    assert R > 0.95


@_python
def test_synchronised_stays_synchronised():
    """Perfect sync + ω = 0 must stay at R = 1 (no numerical drift)."""
    n = 5
    phases = np.full(n, 1.234)
    omegas = np.zeros(n)
    knm = np.full((n, n), 1.0)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))
    out = upde_run(
        phases, omegas, knm, alpha,
        zeta=0.0, psi=0.0, dt=0.01, n_steps=5000,
        method="rk4",
    )
    R = float(np.abs(np.mean(np.exp(1j * out))))
    assert R > 1.0 - 1e-9


@_python
def test_rk45_does_not_blow_up():
    """RK45 on a random Kuramoto problem must stay finite across
    10 000 steps."""
    rng = np.random.default_rng(2027)
    n = 6
    phases = rng.uniform(0.0, TWO_PI, size=n)
    omegas = rng.normal(0.0, 0.5, size=n)
    knm = rng.uniform(0.0, 2.0, size=(n, n))
    np.fill_diagonal(knm, 0.0)
    alpha = rng.uniform(-0.3, 0.3, size=(n, n))
    np.fill_diagonal(alpha, 0.0)
    out = upde_run(
        phases, omegas, knm, alpha,
        zeta=0.2, psi=0.3, dt=0.01, n_steps=10_000,
        method="rk45",
    )
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)
    assert np.all(out < TWO_PI + 1e-12)
