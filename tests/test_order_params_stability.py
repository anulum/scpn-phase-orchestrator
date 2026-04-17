# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stability tests for order parameters

"""Stability / physical-invariant tests for ``upde/order_params.py``.

Covers the numerical properties that matter when the kernels run in
a long simulation loop:

* ``R`` stays in ``[0, 1]`` under repeated noisy inputs (no drift
  past the unit interval bound).
* PLV is stable under phase noise: small Gaussian noise added to a
  fully-synchronised pair only lowers PLV smoothly (no sign flips).
* Layer coherence of a synchronised subset stays high across seeds.
* Large-N sanity: ``R`` of a uniform distribution decays as
  ``~1/√N`` (finite-size law) within a generous envelope.

Marked ``pytest.mark.slow`` so the fast CI lane skips them.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.upde.order_params import (
    compute_layer_coherence,
    compute_order_parameter,
    compute_plv,
)

TWO_PI = 2.0 * np.pi

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------
# R in [0, 1] under long repeated calls
# ---------------------------------------------------------------------


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(
    max_examples=3,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_r_stays_in_unit_interval_long_run(seed: int) -> None:
    """Over 1000 repeated evaluations on changing phases, R must never
    leave [0, 1] (guards against numerical drift or overflow)."""
    rng = np.random.default_rng(seed)
    n = 64
    phases = rng.uniform(0.0, TWO_PI, size=n)
    for _ in range(1000):
        phases += rng.normal(0.0, 0.01, size=n)  # random walk
        phases %= TWO_PI
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0 + 1e-12


# ---------------------------------------------------------------------
# PLV decreases smoothly under noise
# ---------------------------------------------------------------------


def test_plv_monotonic_noise_decrease() -> None:
    """Adding more noise to one of two identical series produces
    monotonically smaller PLV (up to sampling error)."""
    rng = np.random.default_rng(42)
    n = 500
    ref = rng.uniform(0.0, TWO_PI, size=n)
    last = 1.0
    for sigma in (0.0, 0.05, 0.1, 0.2, 0.5, 1.0):
        noisy = ref + rng.normal(0.0, sigma, size=n)
        val = compute_plv(ref, noisy)
        # allow 5% uplift within the stochastic budget
        assert val <= last + 0.05
        last = val


# ---------------------------------------------------------------------
# Layer coherence of synchronised subset
# ---------------------------------------------------------------------


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(
    max_examples=5,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_synchronised_layer_high_r(seed: int) -> None:
    """A synchronised 8-element subset inside a random N=32 population
    has layer coherence > 0.95 (the sync group's own R)."""
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, TWO_PI, size=32)
    # Replace indices 0..7 with a tightly-clustered group.
    phases[0:8] = 1.23 + rng.normal(0.0, 0.05, size=8)
    r = compute_layer_coherence(phases, np.arange(8, dtype=np.int64))
    assert r > 0.95


# ---------------------------------------------------------------------
# Finite-N law for uniform phases
# ---------------------------------------------------------------------


@pytest.mark.parametrize("n", [100, 400, 1600])
def test_uniform_r_follows_sqrt_n_law(n: int) -> None:
    """Finite-sample R for uniformly random phases scales as ~1/√N
    (within a generous envelope: factor 5 covers finite-N tail)."""
    rng = np.random.default_rng(2026)
    trials = 20
    rs = []
    for _ in range(trials):
        phases = rng.uniform(0.0, TWO_PI, size=n)
        r, _ = compute_order_parameter(phases)
        rs.append(r)
    r_mean = float(np.mean(rs))
    expected_ceiling = 5.0 / np.sqrt(n)
    assert r_mean < expected_ceiling, (
        f"R̄ = {r_mean:.3f} exceeds finite-N ceiling "
        f"{expected_ceiling:.3f} at N={n}"
    )
