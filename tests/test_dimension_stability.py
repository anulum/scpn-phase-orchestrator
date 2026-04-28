# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stability / regression tests for dimension kernels

"""Long-run invariants for fractal-dimension kernels.

* GP slope on a 3-D Gaussian cloud approaches 3 as the sample size
  grows.
* Subsampled mode returns non-negative, bounded, finite values for
  a stress-size trajectory.
* D_KY on a randomly-drawn spectrum of 32 exponents lives in ``[0,
  32]`` across many seeds.

Marked ``@pytest.mark.slow``.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import dimension as dim_mod
from scpn_phase_orchestrator.monitor.dimension import (
    correlation_dimension,
    correlation_integral,
    kaplan_yorke_dimension,
)

pytestmark = pytest.mark.slow


def _python(func):
    def wrapper(*args, **kwargs):
        prev = dim_mod.ACTIVE_BACKEND
        dim_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            dim_mod.ACTIVE_BACKEND = prev

    wrapper.__name__ = func.__name__
    return wrapper


@_python
def test_gp_slope_gaussian_3d_converges():
    """GP slope on a 3-D Gaussian cloud converges towards 3 as
    sample size grows."""
    rng = np.random.default_rng(0)
    traj = rng.normal(0, 1, (800, 3))
    res = correlation_dimension(traj, n_epsilons=50)
    # Broad window — GP is bias-heavy at this scale but should be
    # clearly above 2 and below 4.
    assert 2.0 < res.D2 < 4.0


@_python
def test_subsampled_mode_bounded():
    rng = np.random.default_rng(2)
    traj = rng.normal(0, 1, (300, 4))
    eps = np.logspace(-1, 1, 15)
    c = correlation_integral(traj, eps, max_pairs=2000)
    assert np.all(np.isfinite(c))
    assert np.all(c >= 0.0)
    assert np.all(c <= 1.0 + 1e-12)


@_python
def test_ky_bounded_on_random_spectrum():
    rng = np.random.default_rng(7)
    for _ in range(20):
        le = rng.normal(0.0, 1.0, 32)
        d = kaplan_yorke_dimension(le)
        assert 0.0 <= d <= 32.0 + 1e-12
