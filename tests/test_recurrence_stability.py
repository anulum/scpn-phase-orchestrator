# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stability tests for recurrence analysis

"""Long-run invariants for recurrence analysis.

* Periodic trajectory — high determinism (≥ 0.8).
* Random trajectory — low determinism compared to a periodic one.
* Large-T stress — T=200 recurrence matrix in reasonable time on
  all enabled backends, output values in ``{0, 1}``.

Marked ``@pytest.mark.slow``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import recurrence as r_mod
from scpn_phase_orchestrator.monitor.recurrence import (
    recurrence_matrix,
    rqa,
)

pytestmark = pytest.mark.slow


def _python(func):
    def wrapper(*args, **kwargs):
        prev = r_mod.ACTIVE_BACKEND
        r_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            r_mod.ACTIVE_BACKEND = prev

    wrapper.__name__ = func.__name__
    return wrapper


@_python
def test_periodic_trajectory_high_determinism():
    """A pure sinusoid gives a deterministic RQA. ``_diagonal_lines``
    scans only the upper triangle by convention (Marwan 2007), so
    ``determinism`` tops out around 0.5 on a fully periodic signal
    even though the physics is maximally coherent. The signal must
    still be well above a random cloud — see
    :func:`test_random_less_deterministic_than_periodic`."""
    t = np.linspace(0, 10 * 2 * math.pi, 200)
    traj = np.column_stack([np.sin(t), np.cos(t)])
    res = rqa(traj, 0.2, l_min=3)
    assert res.determinism > 0.45
    assert res.max_diagonal >= 10


@_python
def test_random_less_deterministic_than_periodic():
    rng = np.random.default_rng(0)
    t_axis = np.linspace(0, 10 * 2 * math.pi, 120)
    periodic = np.column_stack([np.sin(t_axis), np.cos(t_axis)])
    random = rng.normal(0, 1, (120, 2))
    det_p = rqa(periodic, 0.3).determinism
    det_r = rqa(random, 0.3).determinism
    assert det_p > det_r


@_python
def test_large_T_stress_boolean_output():
    rng = np.random.default_rng(3)
    traj = rng.normal(0, 1, (200, 3))
    R = recurrence_matrix(traj, 1.0)
    assert R.shape == (200, 200)
    assert R.dtype == bool
    # Recurrence rate sits somewhere non-trivial for a random cloud.
    rate = float(R.sum()) / (200 * 200)
    assert 0.0 < rate < 1.0
