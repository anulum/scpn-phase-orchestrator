# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stability tests for chimera detection

"""Long-run invariants for chimera detection.

* Global shift invariance: ``R_i(θ + φ) == R_i(θ)`` (local order
  only depends on phase *differences*).
* Staged chimera: half the oscillators locked at ``θ = 0`` on a
  ring + half uniformly random → coherent and incoherent classes
  both non-empty after many seeds.
* N=500 stress run finishes with finite output.

Marked ``@pytest.mark.slow``.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import chimera as ch_mod
from scpn_phase_orchestrator.monitor.chimera import (
    detect_chimera,
    local_order_parameter,
)

TWO_PI = 2.0 * np.pi

pytestmark = pytest.mark.slow


def _python(func):
    def wrapper(*args, **kwargs):
        prev = ch_mod.ACTIVE_BACKEND
        ch_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            ch_mod.ACTIVE_BACKEND = prev

    wrapper.__name__ = func.__name__
    return wrapper


@_python
def test_global_shift_invariance():
    rng = np.random.default_rng(0)
    n = 40
    phases = rng.uniform(0, TWO_PI, n)
    knm = rng.uniform(0.0, 1.0, (n, n))
    knm = (knm > 0.3).astype(np.float64) * knm
    np.fill_diagonal(knm, 0.0)

    shifted = (phases + 1.234) % TWO_PI
    np.testing.assert_allclose(
        local_order_parameter(phases, knm),
        local_order_parameter(shifted, knm),
        atol=1e-10,
    )


@_python
def test_staged_chimera_on_local_coupling_ring():
    """Half-locked + half-random on a **narrow-kernel ring** produces
    a true Kuramoto-Battogtokh chimera: the locked half has high
    ``R_i``; the random half has low ``R_i``. All-to-all would smear
    both into the boundary band, so the test uses a radius-2 ring
    where each oscillator only sees its 4 nearest neighbours.
    """
    successes = 0
    for seed in range(10):
        rng = np.random.default_rng(seed)
        n = 40
        # Ring coupling with radius 2 — each i connected to i±1, i±2.
        knm = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for off in (-2, -1, 1, 2):
                knm[i, (i + off) % n] = 1.0
        phases = np.concatenate(
            [np.zeros(n // 2), rng.uniform(0, TWO_PI, n // 2)]
        )
        state = detect_chimera(phases, knm)
        if state.coherent_indices and state.incoherent_indices:
            successes += 1
    assert successes >= 7


@_python
def test_large_N_stress_finite():
    rng = np.random.default_rng(7)
    n = 500
    phases = rng.uniform(0, TWO_PI, n)
    knm = rng.uniform(0.0, 1.0, (n, n))
    knm = (knm > 0.2).astype(np.float64) * knm
    np.fill_diagonal(knm, 0.0)
    r = local_order_parameter(phases, knm)
    assert r.shape == (n,)
    assert np.all(np.isfinite(r))
    assert np.all(r <= 1.0 + 1e-12)
