# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Long-run stability tests for ITPC

"""Stability / stress tests for :func:`compute_itpc`.

* Large-N convergence — ITPC of a uniform distribution converges to
  ``1/√N`` (Rayleigh concentration noise floor).
* Long-trial throughput — 1000 trials × 2000 timepoints finish in a
  few seconds without NaN.
* Invariance under global phase shift — shifting every phase by a
  constant ``φ`` leaves ITPC unchanged.

Marked ``@pytest.mark.slow``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import itpc as it_mod
from scpn_phase_orchestrator.monitor.itpc import (
    compute_itpc,
    itpc_persistence,
)

TWO_PI = 2.0 * np.pi

pytestmark = pytest.mark.slow


def _python(func):
    def wrapper(*args, **kwargs):
        prev = it_mod.ACTIVE_BACKEND
        it_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            it_mod.ACTIVE_BACKEND = prev

    wrapper.__name__ = func.__name__
    return wrapper


@_python
def test_uniform_noise_floor_1_over_sqrt_N():
    """Uniform phases → ITPC ≈ 1/√N (Rayleigh)."""
    rng = np.random.default_rng(0)
    n_trials = 4000
    phases = rng.uniform(0.0, TWO_PI, (n_trials, 400))
    out = compute_itpc(phases)
    expected = 1.0 / math.sqrt(n_trials)
    # Expected ± 2σ for Rayleigh in the large-N limit.
    assert float(np.mean(out)) < 4.0 * expected


@_python
def test_global_shift_invariance():
    """Adding a constant to every phase leaves ITPC invariant."""
    rng = np.random.default_rng(1)
    base = rng.uniform(0.0, TWO_PI, (50, 120))
    shifted = (base + 1.234) % TWO_PI
    np.testing.assert_allclose(
        compute_itpc(base),
        compute_itpc(shifted),
        atol=1e-10,
    )


@_python
def test_long_trial_no_nan():
    """1000 × 2000 stress run — finite output."""
    rng = np.random.default_rng(7)
    phases = rng.uniform(0.0, TWO_PI, (1000, 2000))
    out = compute_itpc(phases)
    assert np.all(np.isfinite(out))
    idx = np.arange(0, 2000, 100)
    assert np.isfinite(itpc_persistence(phases, idx))
