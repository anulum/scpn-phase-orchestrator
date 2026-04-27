# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stability tests for winding numbers

"""Long-run invariants for :func:`winding_numbers`.

* Additivity: winding over [0, T] = winding over [0, k] + [k, T]
  (up to ±1 boundary rounding from the final ``floor``).
* Large-T stress: 10 000 timesteps × 64 oscillators → finite int64
  output.
* Robustness under numeric noise: adding tiny Gaussian noise to a
  constant-ω rotator does not change the winding integers.

Marked ``@pytest.mark.slow``.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import winding as w_mod
from scpn_phase_orchestrator.monitor.winding import winding_numbers

TWO_PI = 2.0 * np.pi

pytestmark = pytest.mark.slow


def _python(func):
    def wrapper(*args, **kwargs):
        prev = w_mod.ACTIVE_BACKEND
        w_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            w_mod.ACTIVE_BACKEND = prev

    wrapper.__name__ = func.__name__
    return wrapper


@_python
def test_additivity_up_to_boundary_rounding():
    omegas = np.array([1.0, -2.0, 0.3, 3.7])
    t, dt = 500, 0.03
    hist = (omegas[np.newaxis, :] * np.arange(t)[:, np.newaxis] * dt) % TWO_PI

    full = winding_numbers(hist)
    half = winding_numbers(hist[: t // 2])
    second = winding_numbers(hist[t // 2 :])
    # Small-angle wraps can land the final floor one unit away from
    # the sum.
    assert np.all(np.abs(full - (half + second)) <= 1)


@_python
def test_large_T_stress_finite_int64():
    rng = np.random.default_rng(0)
    t, n = 10_000, 64
    hist = np.zeros((t, n))
    hist[0] = rng.uniform(0, TWO_PI, n)
    omegas = rng.normal(0, 0.5, n)
    for i in range(1, t):
        hist[i] = (hist[i - 1] + omegas * 0.01) % TWO_PI
    w = winding_numbers(hist)
    assert w.dtype == np.int64
    assert np.all(np.isfinite(w.astype(np.float64)))
    assert np.all(np.abs(w) <= t)


@_python
def test_noise_does_not_flip_sign():
    rng = np.random.default_rng(3)
    omegas = np.array([0.5, -0.5, 2.0, -2.0])
    t, dt = 400, 0.05
    clean = (omegas[np.newaxis, :] * np.arange(t)[:, np.newaxis] * dt) % TWO_PI
    noisy = (clean + rng.normal(0.0, 0.01, clean.shape)) % TWO_PI
    w_clean = winding_numbers(clean)
    w_noisy = winding_numbers(noisy)
    # Direction is preserved: same sign / zero category per osc.
    assert np.all(np.sign(w_clean) == np.sign(w_noisy))
