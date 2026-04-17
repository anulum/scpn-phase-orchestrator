# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stability tests for transfer entropy

"""Physical invariants for ``monitor/transfer_entropy.py``. Marked
``pytest.mark.slow``.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.transfer_entropy import (
    phase_transfer_entropy,
    transfer_entropy_matrix,
)

TWO_PI = 2.0 * np.pi

pytestmark = pytest.mark.slow


def test_te_non_negative_across_seeds() -> None:
    """TE is non-negative by construction; verify across many seeds."""
    for seed in range(20):
        rng = np.random.default_rng(seed)
        src = rng.uniform(0.0, TWO_PI, size=200)
        tgt = rng.uniform(0.0, TWO_PI, size=200)
        assert phase_transfer_entropy(src, tgt, 16) >= 0.0


def test_te_upper_bound_by_log_n_bins() -> None:
    """TE cannot exceed ``log(n_bins)`` — the target's conditional entropy
    is bounded by the bin count."""
    rng = np.random.default_rng(0)
    n_bins = 16
    src = rng.uniform(0.0, TWO_PI, size=1000)
    tgt = src.copy()  # perfect driver → maximal TE
    assert phase_transfer_entropy(src, tgt, n_bins) <= np.log(n_bins) + 1e-9


def test_te_independent_signals_near_zero() -> None:
    """Large-N independent signals give TE near the finite-sample floor."""
    rng = np.random.default_rng(7)
    n = 5000
    src = rng.uniform(0.0, TWO_PI, size=n)
    tgt = rng.uniform(0.0, TWO_PI, size=n)
    te = phase_transfer_entropy(src, tgt, 8)
    assert te < 0.1


def test_te_matrix_diagonal_zero() -> None:
    rng = np.random.default_rng(3)
    series = rng.uniform(0.0, TWO_PI, size=(5, 200))
    m = transfer_entropy_matrix(series, 16)
    np.testing.assert_allclose(np.diag(m), np.zeros(5), atol=1e-12)


def test_te_matrix_asymmetric_for_directed_coupling() -> None:
    """When channel 0 drives channel 1 (lagged copy), TE(0→1) exceeds
    TE(1→0)."""
    rng = np.random.default_rng(11)
    n = 1000
    ch0 = np.cumsum(0.01 + 0.01 * rng.standard_normal(n))
    ch1 = np.roll(ch0, 1) + 0.05 * rng.standard_normal(n)
    series = np.stack([ch0 % TWO_PI, ch1 % TWO_PI])
    m = transfer_entropy_matrix(series, 16)
    assert m[0, 1] > m[1, 0]


def test_te_short_series_returns_zero() -> None:
    """Length < 3 cannot form a 1-step estimator; returns 0."""
    assert phase_transfer_entropy(
        np.array([0.0, 1.0]), np.array([0.0, 1.0]), 16
    ) == 0.0
