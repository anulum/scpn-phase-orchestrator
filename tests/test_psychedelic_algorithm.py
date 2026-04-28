# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Algorithmic tests for circular phase entropy

"""Algorithmic properties of :func:`entropy_from_phases`.

Covered: non-negativity, upper bound ``log(n_bins)`` for the uniform
distribution, zero entropy for constant phases, translation
invariance (entropy unchanged by a uniform phase shift), empty-input
safety, Hypothesis property coverage.
"""

from __future__ import annotations

import functools
import math

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.monitor import psychedelic as py_mod
from scpn_phase_orchestrator.monitor.psychedelic import (
    entropy_from_phases,
)

TWO_PI = 2.0 * math.pi


def _python(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        prev = py_mod.ACTIVE_BACKEND
        py_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            py_mod.ACTIVE_BACKEND = prev

    return wrapper


class TestEntropy:
    @_python
    def test_non_negative(self):
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, TWO_PI, 500)
        assert entropy_from_phases(phases, 36) >= 0.0

    @_python
    def test_constant_phases_zero_entropy(self):
        assert entropy_from_phases(np.full(100, 0.3), 36) == 0.0

    @_python
    def test_uniform_distribution_upper_bound(self):
        """A uniform sample of ~N phases spread across all bins
        approaches ``log(n_bins)``. The bound is an upper limit; we
        check the measured value stays below it + ε."""
        rng = np.random.default_rng(1)
        phases = rng.uniform(0, TWO_PI, 10_000)
        h = entropy_from_phases(phases, 36)
        assert h <= math.log(36) + 1e-9
        assert h > math.log(36) - 0.1  # within 0.1 of theoretical cap

    @_python
    def test_translation_invariance(self):
        """Shifting every phase by a constant leaves entropy
        unchanged — bin populations rotate but counts per bin stay
        fixed (modulo boundary rounding)."""
        rng = np.random.default_rng(3)
        phases = rng.uniform(0, TWO_PI, 1000)
        h1 = entropy_from_phases(phases, 36)
        h2 = entropy_from_phases(phases + 1.234, 36)
        # The modulo shift can reshuffle boundary cases so a tiny
        # histogram difference is permissible.
        assert abs(h1 - h2) < 0.05

    @_python
    def test_empty_input(self):
        assert entropy_from_phases(np.array([]), 36) == 0.0

    @_python
    def test_n_bins_parameter(self):
        rng = np.random.default_rng(5)
        phases = rng.uniform(0, TWO_PI, 5000)
        h36 = entropy_from_phases(phases, 36)
        h128 = entropy_from_phases(phases, 128)
        # More bins → higher entropy ceiling, so H should rise.
        assert h128 > h36


class TestHypothesis:
    @_python
    @given(
        n=st.integers(min_value=5, max_value=1000),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_bounded_by_log_n_bins(self, n: int, seed: int):
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        h = entropy_from_phases(phases, 36)
        assert 0.0 <= h <= math.log(36) + 1e-9


class TestDispatcherSurface:
    def test_available_non_empty(self):
        assert py_mod.AVAILABLE_BACKENDS
        assert "python" in py_mod.AVAILABLE_BACKENDS

    def test_active_is_first(self):
        assert py_mod.AVAILABLE_BACKENDS[0] == py_mod.ACTIVE_BACKEND
