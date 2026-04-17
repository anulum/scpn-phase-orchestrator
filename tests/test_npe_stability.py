# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stability tests for NPE

"""Stability / physical invariants for ``monitor/npe.py``. Marked
``pytest.mark.slow``."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.monitor.npe import (
    compute_npe,
    phase_distance_matrix,
)

TWO_PI = 2.0 * np.pi

pytestmark = pytest.mark.slow


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(
    max_examples=5,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_npe_bounded(seed: int) -> None:
    """NPE stays in ``[0, 1]`` under 100 random shuffles."""
    rng = np.random.default_rng(seed)
    for _ in range(100):
        n = int(rng.integers(4, 64))
        phases = rng.uniform(0.0, TWO_PI, size=n)
        v = compute_npe(phases)
        assert 0.0 <= v <= 1.0 + 1e-12


def test_npe_synchronised_low() -> None:
    """Fully synchronised phases give NPE near 0 (one dominant cluster)."""
    phases = np.full(32, 1.23)
    assert compute_npe(phases) == pytest.approx(0.0, abs=1e-12)


def test_npe_uniform_high() -> None:
    """Uniform circle gives NPE close to 1 (maximally spread lifetimes)."""
    phases = np.linspace(0.0, TWO_PI, 64, endpoint=False)
    assert compute_npe(phases) > 0.95


def test_pdm_bounded_pi() -> None:
    """Distance matrix entries are all in ``[0, π]``."""
    rng = np.random.default_rng(11)
    phases = rng.uniform(-10.0, 10.0, size=30)  # unwrapped
    pdm = phase_distance_matrix(phases)
    assert pdm.min() >= 0.0
    assert pdm.max() <= np.pi + 1e-12


def test_pdm_symmetric() -> None:
    rng = np.random.default_rng(7)
    phases = rng.uniform(0.0, TWO_PI, size=25)
    pdm = phase_distance_matrix(phases)
    np.testing.assert_allclose(pdm, pdm.T, atol=1e-12)


def test_pdm_zero_diagonal() -> None:
    rng = np.random.default_rng(1)
    phases = rng.uniform(0.0, TWO_PI, size=12)
    pdm = phase_distance_matrix(phases)
    np.testing.assert_allclose(np.diag(pdm), np.zeros(12), atol=1e-12)


@pytest.mark.parametrize("n", [32, 128])
def test_npe_drops_with_clustering(n: int) -> None:
    """Two tight clusters give lower NPE than full uniform."""
    rng = np.random.default_rng(3 + n)
    uniform = rng.uniform(0.0, TWO_PI, size=n)
    mid = n // 2
    clustered = np.concatenate(
        [
            1.0 + rng.normal(0.0, 0.05, mid),
            4.0 + rng.normal(0.0, 0.05, n - mid),
        ]
    )
    npe_uniform = compute_npe(uniform)
    npe_clustered = compute_npe(clustered)
    assert npe_clustered < npe_uniform
