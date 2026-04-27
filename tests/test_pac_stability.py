# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stability tests for PAC

"""Physical / numerical stability tests for ``upde/pac.py``.

Covers the invariants that matter when the kernel runs on long
signals or in tight loops:

* MI stays in ``[0, 1]`` on adversarial inputs.
* Tort 2010 signature: strong phase-locked modulation gives MI
  growing monotonically with modulation depth.
* Uncorrelated (phase, amplitude) pairs give MI near the small
  finite-sample floor.
* Long-run sample does not drift outside the bound.

Marked ``pytest.mark.slow`` so the fast CI lane skips.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.upde.pac import modulation_index, pac_matrix

TWO_PI = 2.0 * np.pi

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------
# Bound invariants
# ---------------------------------------------------------------------


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(
    max_examples=5,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_mi_always_in_unit_interval(seed: int) -> None:
    """MI stays in ``[0, 1]`` under adversarial inputs over 200 calls."""
    rng = np.random.default_rng(seed)
    for _ in range(200):
        n = rng.integers(50, 500)
        theta = rng.uniform(0.0, TWO_PI, size=int(n))
        amp = rng.standard_normal(int(n)) * rng.uniform(0.1, 10.0)
        mi = modulation_index(theta, amp, 18)
        assert 0.0 <= mi <= 1.0 + 1e-12


# ---------------------------------------------------------------------
# Signature: modulation depth correlates with MI
# ---------------------------------------------------------------------


def test_mi_monotonic_in_modulation_depth() -> None:
    """Stronger phase-locked amplitude modulation produces larger MI
    (ordered by modulation depth across 5 levels)."""
    rng = np.random.default_rng(0)
    n = 5000
    theta = rng.uniform(0.0, TWO_PI, size=n)
    last = -1.0
    for depth in (0.0, 0.1, 0.3, 0.6, 0.9):
        amp = 1.0 + depth * np.cos(theta) + 0.05 * rng.standard_normal(n)
        mi = modulation_index(theta, amp, 18)
        # Allow tiny noise backslide (Tort MI is stochastic for small n).
        assert mi >= last - 0.005, (
            f"MI non-monotonic: depth={depth}, mi={mi:.4f} vs last={last:.4f}"
        )
        last = mi


# ---------------------------------------------------------------------
# Uncorrelated signal noise floor
# ---------------------------------------------------------------------


@pytest.mark.parametrize("n", [500, 2000, 8000])
def test_uncorrelated_mi_near_floor(n: int) -> None:
    """Independent phase / amplitude gives MI near the noise floor.

    Tort 2010 needs a non-negative amplitude envelope (KL is
    undefined on negative "distributions"). The test uses
    ``|randn|`` to mimic a Hilbert-envelope magnitude; the resulting
    MI should scale as ``log(n_bins) / N`` and stay below 0.1 for
    N ≥ 500.
    """
    rng = np.random.default_rng(42 + n)
    theta = rng.uniform(0.0, TWO_PI, size=n)
    amp = np.abs(rng.standard_normal(n))
    mi = modulation_index(theta, amp, 18)
    assert mi < 0.1


# ---------------------------------------------------------------------
# Matrix-level invariants
# ---------------------------------------------------------------------


def test_pac_matrix_shape_and_bounds() -> None:
    rng = np.random.default_rng(9)
    t, n = 200, 6
    phases = rng.uniform(0.0, TWO_PI, size=(t, n))
    amps = 1.0 + 0.3 * np.cos(phases) + 0.1 * rng.standard_normal((t, n))
    mat = pac_matrix(phases, amps, 18)
    assert mat.shape == (n, n)
    assert mat.min() >= 0.0
    assert mat.max() <= 1.0 + 1e-12


def test_pac_matrix_diagonal_stronger_than_off_for_locked_signal() -> None:
    """When channel ``i`` phase drives channel ``i`` amplitude, the
    diagonal dominates the off-diagonal entries."""
    rng = np.random.default_rng(11)
    t, n = 1000, 4
    phases = rng.uniform(0.0, TWO_PI, size=(t, n))
    # Amplitude locked only to its own phase.
    amps = 1.0 + 0.6 * np.cos(phases)
    mat = pac_matrix(phases, amps, 18)
    # Diagonal mean must exceed off-diagonal mean. Absolute gap is
    # small in Tort MI units (~0.03 for this setup) but the relative
    # ratio is large — check both.
    diag = np.diagonal(mat).mean()
    off = (mat.sum() - np.trace(mat)) / (n * (n - 1))
    assert diag > off
    assert diag > 5.0 * off  # order-of-magnitude signature
